import itertools
import json
import math
import os
import random
from collections import defaultdict
from collections.abc import Sized
from itertools import chain, repeat
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    List,
    Optional,
    Tuple,
)

import spacy
import torch
from accelerate import Accelerator
from confit import Cli
from confit.utils.random import set_seed
from rich_logger import RichTablePrinter
from torch.utils.data import DataLoader
from tqdm import tqdm

from eds_pseudo.adapter import PseudoReader
from eds_pseudo.scorer import PseudoScorer
from edsnlp.core.pipeline import Pipeline
from edsnlp.core.registries import registry
from edsnlp.optimization import LinearSchedule, ScheduledOptimizer
from edsnlp.pipes.trainable.embeddings.transformer.transformer import Transformer
from edsnlp.utils.collections import batchify
from edsnlp.utils.typing import AsList

app = Cli(pretty_exceptions_show_locals=False)

BASE_DIR = Path.cwd()

LOGGER_FIELDS = {
    "step": {},
    "(.*loss)": {
        "goal": "lower_is_better",
        "format": "{:.2e}",
        "goal_wait": 2,
        "name": r"\1",
    },
    "(p|r|f|redact|full)": {
        "goal": "higher_is_better",
        "format": "{:.2%}",
        "goal_wait": 2,
        "name": r"\1",
    },
    "lr": {"format": "{:.2e}"},
    "speed/(.*)": {"format": "{:.2f}", r"name": r"\1"},
    "labels": {"format": "{:.2f}"},
}


class BatchSizeArg:
    """
    Batch size argument validator / caster for confit/pydantic

    Examples
    --------
    ```python
    def fn(batch_size: BatchSizeArg):
        return batch_size


    print(fn("10 samples"))
    # Out: (10, "samples")

    print(fn("10 words"))
    # Out: (10, "words")

    print(fn(10))
    # Out: (10, "samples")
    ```
    """

    @classmethod
    def validate(cls, value, config=None):
        value = str(value)
        parts = value.split()
        num = int(parts[0])
        unit = parts[1] if len(parts) == 2 else "samples"
        if len(parts) == 2 and str(num) == parts[0] and unit in ("words", "samples"):
            return num, unit
        raise Exception(f"Invalid batch size: {value}, must be <int> samples|words")

    @classmethod
    def __get_validators__(cls):
        yield cls.validate


if TYPE_CHECKING:
    BatchSizeArg = Tuple[int, str]  # noqa: F811


class LengthSortedBatchSampler:
    """
    Batch sampler that sorts the dataset by length and then batches
    sequences of similar length together. This is useful for transformer
    models that can then be padded more efficiently.

    Parameters
    ----------
    dataset: Iterable
        The dataset to sample from (can be a generator or a fixed size collection)
    batch_size: int
        The batch size
    batch_unit: str
        The unit of the batch size, either "words" or "samples"
    noise: int
        The amount of noise to add to the sequence length before sorting
        (uniformly sampled in [-noise, noise])
    drop_last: bool
        Whether to drop the last batch if it is smaller than the batch size
    buffer_size: Optional[int]
        The size of the buffer to use to shuffle the batches. If None, the buffer
        will be approximately the size of the dataset.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        batch_unit: str,
        noise=1,
        drop_last=True,
        buffer_size: Optional[int] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_unit = batch_unit
        self.noise = noise
        self.drop_last = drop_last
        self.buffer_size = buffer_size

    def __iter__(self):
        # Shuffle the dataset
        if self.batch_unit == "words":

            def sample_len(idx, noise=True):
                count = sum(
                    len(x)
                    for x in next(
                        v
                        for k, v in self.dataset[idx].items()
                        if k.endswith("word_lengths")
                    )
                )
                if not noise:
                    return count
                return count + random.randint(-self.noise, self.noise)

        else:
            sample_len = lambda idx, noise=True: 1  # noqa: E731

        def make_batches():
            total = 0
            batch = []
            for seq_size, idx in sorted_sequences:
                if total and total + seq_size > self.batch_size:
                    yield batch
                    total = 0
                    batch = []
                total += seq_size
                batch.append(idx)

        # Shuffle the batches in buffer that contain approximately
        # the full dataset to add more randomness
        if isinstance(self.dataset, Sized):
            total_count = sum(sample_len(i, False) for i in range(len(self.dataset)))

        assert (
            isinstance(self.dataset, Sized) or self.buffer_size is not None
        ), "Dataset must have a length or buffer_size must be specified"
        buffer_size = self.buffer_size or math.ceil(total_count / self.batch_size)

        # Sort sequences by length +- some noise
        sorted_sequences = chain.from_iterable(
            sorted((sample_len(i), i) for i in range(len(self.dataset)))
            for _ in repeat(None)
        )

        # Batch sorted sequences
        batches = make_batches()
        buffers = batchify(batches, buffer_size)
        for buffer in buffers:
            random.shuffle(buffer)
            yield from buffer


class SubBatchCollater:
    """
    Collater that splits batches into sub-batches of a maximum size

    Parameters
    ----------
    nlp: Pipeline
        The pipeline object
    embedding: Transformer
        The transformer embedding pipe
    grad_accumulation_max_tokens: int
        The maximum number of tokens (word pieces) to accumulate in a single batch
    """

    def __init__(self, nlp, embedding, grad_accumulation_max_tokens):
        self.nlp = nlp
        self.embedding: Transformer = embedding
        self.grad_accumulation_max_tokens = grad_accumulation_max_tokens

    def __call__(self, seq):
        total = 0
        mini_batches = [[]]
        for sample_features in seq:
            num_tokens = sum(
                math.ceil(len(p) / self.embedding.stride) * self.embedding.window
                for key in sample_features
                if key.endswith("/input_ids")
                for p in sample_features[key]
            )
            if total + num_tokens > self.grad_accumulation_max_tokens:
                print(
                    f"Mini batch size was becoming too large: {total} > "
                    f"{self.grad_accumulation_max_tokens} so it was split"
                )
                total = 0
                mini_batches.append([])
            total += num_tokens
            mini_batches[-1].append(sample_features)
        return [self.nlp.collate(b) for b in mini_batches if len(b)]


@app.command(name="train", registry=registry)
def train(
    *,
    nlp: Pipeline,
    train_data: AsList[PseudoReader],
    val_data: PseudoReader,
    seed: int = 42,
    data_seed: int = 42,
    max_steps: int = 1000,
    batch_size: BatchSizeArg = 2000,
    embedding_lr: float = 5e-5,
    task_lr: float = 3e-4,
    validation_interval: int = 10,
    grad_max_norm: float = 5.0,
    grad_accumulation_max_tokens: int = 96 * 128,
    scorer: PseudoScorer,
    output_dir: Optional[Path] = None,
    cpu: bool = False,
):
    """
    Train a model on a dataset.

    Parameters
    ----------
    nlp: Pipeline
        The edsnlp pipeline object
    train_data: AsList[PseudoReader]
        The training data, can be a PseudoReader or a list of PseudoReaders
    val_data: PseudoReader
        The validation data
    seed: int
        The seed to use for random number generators when initializing the model
    data_seed: int
        The seed to use for random number generators when shuffling the data
    max_steps: int
        The maximum number of training steps
    batch_size: BatchSizeArg
        The batch size to use for training, support the following units:

        - <int> samples: the number of samples per batch
        - <int> words: the number of words per batch
    embedding_lr: float
        The learning rate for the transformer embedding
    task_lr: float
        The learning rate for the task (NER) head
    validation_interval: int
        The number of steps between each validation
    grad_max_norm: float
        The maximum gradient norm to use for gradient clipping
    grad_accumulation_max_tokens: int
        The maximum number of tokens to accumulate in a single batch
    scorer: PseudoScorer
        The scorer object to use for validation
    output_dir: Optional[Path]
        The output directory to save the model and training metrics.
    cpu: bool
        Whether to force the training to run on CPU (useful for M1 chips for which
        all the ops of transformers are not yet supported)

    Returns
    -------
    Pipeline
        The model (trained in place). The artifacts are saved in `artifacts/model-last`
        and `artifacts/train_metrics.json`.
    """
    trf_pipe = next(
        module
        for name, pipe in nlp.torch_components()
        for module_name, module in pipe.named_component_modules()
        if isinstance(module, Transformer)
    )
    if nlp.has_pipe("dates-normalizer"):
        nlp.select_pipes(disable=["dates-normalizer"])

    output_dir = Path(output_dir or BASE_DIR / "artifacts")

    set_seed(seed)
    with set_seed(data_seed):
        train_docs: List[spacy.tokens.Doc] = list(
            chain.from_iterable(td(nlp) for td in train_data)
        )
        val_docs: List[spacy.tokens.Doc] = list(val_data(nlp))

    model_path = output_dir / "model-last"
    train_metrics_path = output_dir / "train_metrics.json"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize pipeline with training documents
    nlp.post_init(train_docs)

    # Preprocessing training data
    print("Preprocessing data")

    preprocessed = list(
        nlp.preprocess_many(train_docs, supervision=True).set_processing(
            backend="multiprocessing", show_progress=True
        )
    )
    print(f"TRAINING DATASET SIZE: {len(preprocessed)}")
    dataloader = DataLoader(
        preprocessed,
        batch_sampler=LengthSortedBatchSampler(
            preprocessed,
            batch_size=batch_size[0],
            batch_unit=batch_size[1],
        ),
        collate_fn=SubBatchCollater(
            nlp,
            trf_pipe,
            grad_accumulation_max_tokens=grad_accumulation_max_tokens,
        ),
    )

    trained_pipes = nlp.torch_components()
    print("Training", ", ".join([name for name, c in trained_pipes]))

    trf_params = set(trf_pipe.parameters())
    params = set(nlp.parameters())
    optimizer = ScheduledOptimizer(
        torch.optim.AdamW(
            [
                {
                    "params": list(params - trf_params),
                    "lr": task_lr,
                    "schedules": [
                        LinearSchedule(
                            total_steps=max_steps,
                            warmup_rate=0.1,
                            start_value=task_lr,
                            path="lr",
                        )
                    ],
                },
                {
                    "params": list(trf_params),
                    "lr": embedding_lr,
                    "schedules": [
                        LinearSchedule(
                            total_steps=max_steps,
                            warmup_rate=0.1,
                            start_value=0,
                            path="lr",
                        )
                    ],
                },
            ]
        )
    )
    grad_params = {p for group in optimizer.param_groups for p in group["params"]}
    print(
        "Optimizing:"
        + "".join(
            f"\n - {len(group['params'])} params "
            f"({sum(p.numel() for p in group['params'])} total)"
            for group in optimizer.param_groups
        )
    )
    print(f"Not optimizing {len(params - grad_params)} params")

    accelerator = Accelerator(cpu=cpu)
    trained_pipes = dict(nlp.torch_components())
    print("Device:", accelerator.device)
    [dataloader, optimizer, *accelerated_pipes] = accelerator.prepare(
        dataloader,
        optimizer,
        *trained_pipes.values(),
    )
    trained_pipes = list(zip(trained_pipes.keys(), accelerated_pipes))
    del accelerated_pipes

    cumulated_data = defaultdict(lambda: 0.0, count=0)

    iterator = itertools.chain.from_iterable(itertools.repeat(dataloader))
    all_metrics = []
    nlp.train(True)
    set_seed(seed)

    with RichTablePrinter(LOGGER_FIELDS, auto_refresh=False) as logger:
        # Training loop
        with tqdm(
            range(max_steps + 1),
            "Training model",
            leave=True,
            mininterval=5.0,
        ) as bar:
            for step in bar:
                if (step % validation_interval) == 0:
                    scores = scorer(nlp, val_docs)
                    cumulated_data = defaultdict(lambda: 0.0, count=0)
                    all_metrics.append(
                        {
                            "step": step,
                            "lr": optimizer.param_groups[0]["lr"],
                            **cumulated_data,
                            **scores,
                        }
                    )
                    logger.log_metrics(all_metrics[-1])
                    train_metrics_path.write_text(json.dumps(all_metrics, indent=2))

                    nlp.to_disk(model_path)

                if step == max_steps:
                    break

                mini_batches = next(iterator)

                optimizer.zero_grad()
                for mini_batch in mini_batches:
                    loss = torch.zeros((), device=accelerator.device)
                    with nlp.cache():
                        for name, pipe in trained_pipes:
                            output = pipe.module_forward(mini_batch[name])
                            if "loss" in output:
                                loss += output["loss"]
                            for key, value in output.items():
                                if key.endswith("loss"):
                                    cumulated_data[key] += float(value)
                    accelerator.backward(loss)

                torch.nn.utils.clip_grad_norm_(
                    (p for g in optimizer.param_groups for p in g["params"]),
                    grad_max_norm,
                )
                optimizer.step()

        return nlp


if __name__ == "__main__":
    app()
