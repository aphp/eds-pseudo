import itertools
import json
import math
import os
import random
from collections import defaultdict
from itertools import chain, repeat
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Tuple

import spacy
import torch
from accelerate import Accelerator
from confit import Cli
from confit.utils.random import set_seed
from edsnlp.core.pipeline import Pipeline
from edsnlp.core.registry import registry
from edsnlp.optimization import LinearSchedule, ScheduledOptimizer
from edsnlp.pipelines.trainable.embeddings.transformer.transformer import Transformer
from edsnlp.utils.collections import batchify
from rich_logger import RichTablePrinter
from spacy.tokens import Doc
from torch.utils.data import DataLoader
from tqdm import tqdm

import eds_pseudonymisation.adapter  # noqa: F401
from eds_pseudonymisation.scorer import PseudoScorer

app = Cli(pretty_exceptions_show_locals=False)

BASE_DIR = Path(__file__).parent.parent
LOGGER_FIELDS = {
    "step": {},
    "(.*_)?loss": {
        "goal": "lower_is_better",
        "format": "{:.2e}",
        "goal_wait": 2,
    },
    "(p|r|f|redact|full)": {
        "goal": "higher_is_better",
        "format": "{:.2%}",
        "goal_wait": 2,
        "name": r"\1",
    },
    "lr": {"format": "{:.2e}"},
    "speed/wps": {"format": "{:.2f}", "name": "wps"},
    "labels": {"format": "{:.2f}"},
}


class BatchSizeArg:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    @classmethod
    def validate(cls, value, config=None):
        value = str(value)
        parts = value.split()
        num = int(parts[0])
        if str(num) == parts[0]:
            if len(parts) == 1:
                return num, "samples"
            if parts[1] in ("words", "samples"):
                return num, parts[1]
        raise Exception(f"Invalid batch size: {value}, must be <int> samples|words")

    @classmethod
    def __get_validators__(cls):
        yield cls.validate


if TYPE_CHECKING:
    BatchSizeArg = Tuple[int, str]  # noqa: F811


class LengthSortedBatchSampler:
    def __init__(
        self, dataset, batch_size: int, batch_unit: str, noise=1, drop_last=True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_unit = batch_unit
        self.noise = noise
        self.drop_last = drop_last

    def __iter__(self):
        # Shuffle the dataset
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

        def make_batches():
            current_count = 0
            current_batch = []
            for idx in sorted_sequences:
                if self.batch_unit == "words":
                    seq_size = sample_len(idx, noise=False)
                    if current_count + seq_size > self.batch_size:
                        yield current_batch
                        current_batch = []
                        current_count = 0
                    current_count += seq_size
                    current_batch.append(idx)
                else:
                    if len(current_batch) == self.batch_size:
                        yield current_batch
                        current_batch = []
                    current_batch.append(idx)
            if len(current_batch):
                yield current_batch

        # Sort sequences by length +- some noise
        sorted_sequences = chain.from_iterable(
            sorted(range(len(self.dataset)), key=sample_len) for _ in repeat(None)
        )

        # Batch sorted sequences
        batches = make_batches()

        # Shuffle the batches in buffer that contain approximately
        # the full dataset to add more randomness
        if self.batch_unit == "words":
            total_count = sum(
                sample_len(idx, noise=False) for idx in range(len(self.dataset))
            )
        else:
            total_count = len(self.dataset)
        buffers = batchify(batches, math.ceil(total_count / self.batch_size))
        for buffer in buffers:
            random.shuffle(buffer)
            yield from buffer


class SubBatchCollater:
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
        return [self.nlp.collate(b) for b in mini_batches]


def flatten_dict(root: Dict[str, Any], depth=-1) -> Dict[str, Any]:
    res = {}

    def rec(d, path, current_depth):
        for k, v in d.items():
            if isinstance(v, dict) and current_depth != depth:
                rec(v, path + "/" + k if path is not None else k, current_depth + 1)
            else:
                res[path + "/" + k if path is not None else k] = v

    rec(root, None, 0)
    return res


@app.command(name="train", registry=registry)
def train(
    *,
    nlp: Pipeline,
    train_data: Callable[[Pipeline], Iterable[Doc]],
    val_data: Callable[[Pipeline], Iterable[Doc]],
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
    cpu: bool = False,
):
    trf_pipe = next(
        module
        for name, pipe in nlp.torch_components()
        for module_name, module in pipe.named_component_modules()
        if isinstance(module, Transformer)
    )

    set_seed(seed)
    with RichTablePrinter(LOGGER_FIELDS, auto_refresh=False) as logger:
        with set_seed(data_seed):
            train_docs: List[spacy.tokens.Doc] = list(train_data(nlp))
            val_docs: List[spacy.tokens.Doc] = list(val_data(nlp))

        model_path = BASE_DIR / "artifacts/model-last"
        train_metrics_path = BASE_DIR / "artifacts/train_metrics.jsonl"
        os.makedirs(BASE_DIR / "artifacts", exist_ok=True)

        # Initialize pipeline with training documents
        nlp.post_init(train_docs)

        # Preprocessing training data
        print("Preprocessing data")

        preprocessed = list(nlp.preprocess_many(train_docs, supervision=True))
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
        # total_wp = 0
        # total_seq = 0
        # total_sample = 0
        # for sample in preprocessed:
        #     input_ids = sample['ner/embedding/embedding/input_ids']
        #     total_wp += sum(len(d) for d in input_ids[0])
        #     total_seq += len(input_ids[0])
        #     total_sample += len(input_ids)
        # print("AVERAGE WINDOW SIZE", total_wp / total_seq)
        # print("AVERAGE SAMPLE WP", total_wp / total_sample)
        # print("AVERAGE SAMPLE WINDOWS", total_seq / total_sample)

        # Training loop
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
        trained_pipes = [pipe for name, pipe in nlp.torch_components()]
        print("Device:", accelerator.device)
        [dataloader, optimizer, *trained_pipes] = accelerator.prepare(
            dataloader,
            optimizer,
            *trained_pipes,
        )

        cumulated_data = defaultdict(lambda: 0.0, count=0)

        iterator = itertools.chain.from_iterable(itertools.repeat(dataloader))
        all_metrics = []
        nlp.train(True)
        set_seed(seed)
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
                # trf_batch = batch["ner"]["embedding"]["embedding"]
                # n_words = trf_batch["mask"].sum().item()
                # n_padded = torch.numel(trf_batch["mask"])
                # n_words_bert = trf_batch["input_ids"].mask.sum().item()
                # n_padded_bert = torch.numel(trf_batch["input_ids"])
                # bar.set_postfix(
                #     n_words=n_words,
                #     ratio=n_words / n_padded,
                #     n_wp=n_words_bert,
                #     bert_ratio=n_words_bert / n_padded_bert,
                # )

                for mini_batch in mini_batches:
                    optimizer.zero_grad()
                    loss = torch.zeros((), device=accelerator.device)
                    with nlp.cache():
                        for pipe in trained_pipes:
                            output = pipe.module_forward(mini_batch[pipe.name])
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


if __name__ == "__main__":
    app()
