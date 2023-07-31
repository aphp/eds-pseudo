import itertools
import json
import math
import random
from collections import defaultdict
from itertools import chain, repeat
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import spacy
import torch
from accelerate import Accelerator
from confit import Cli
from edsnlp.core.pipeline import Pipeline
from edsnlp.core.registry import registry
from edsnlp.optimization import LinearSchedule, ScheduledOptimizer
from edsnlp.utils.collections import batchify
from edsnlp.utils.filter import filter_spans
from edsnlp.utils.random import set_seed
from rich_logger import RichTablePrinter
from spacy.tokens import Doc, Span
from torch.utils.data import DataLoader
from tqdm import tqdm

app = Cli(pretty_exceptions_show_locals=False)

BASE_DIR = Path(__file__).parent.parent


class LengthSortedBatchSampler:
    def __init__(self, dataset, batch_size, noise=1, drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.noise = noise
        self.drop_last = drop_last

    def __iter__(self):
        # Shuffle the dataset
        def sample_len(idx):
            wt = next(
                v for k, v in self.dataset[idx].items() if k.endswith("word_tokens")
            )
            return len(wt) + random.randint(-self.noise, self.noise)

        # Sort sequences by length +- some noise
        sequences = chain.from_iterable(
            sorted(range(len(self.dataset)), key=sample_len) for _ in repeat(None)
        )

        # Batch sorted sequences
        batches = batchify(sequences, self.batch_size)

        # Shuffle the batches in buffer that contain approximately
        # the full dataset to add more randomness
        buffers = batchify(batches, math.ceil(len(self.dataset) / self.batch_size))
        for buffer in buffers:
            random.shuffle(buffer)
            yield from buffer


@registry.misc.register("deft_span_getter")
def make_span_getter():
    def span_getter(doclike: Union[Doc, Span]) -> List[Span]:
        """
        Get the spans of a span group that are contained inside a doclike object.
        Parameters
        ----------
        doclike : Union[Doc, Span]
            Doclike object to act as a mask.
        group : str
            Group name from which to get the spans.
        Returns
        -------
        List[Span]
            List of spans.
        """
        if isinstance(doclike, Doc):
            return doclike.ents
            # return [
            #     ent
            #     for group in doclike.doc.spans
            #     for ent in doclike.spans.get(group, ())
            # ]
        else:
            return doclike.ents
            # return [
            #     span
            #     for group in doclike.doc.spans
            #     for span in doclike.doc.spans.get(group, ())
            #     if span.start >= doclike.start and span.end <= doclike.end
            # ]

    return span_getter


@registry.misc.register("brat_dataset")
def pseudo_dataset(path, limit: Optional[int] = None, span_getter=make_span_getter()):
    def load(nlp) -> List[Doc]:
        # Load the jsonl data from path
        raw_data = []
        with open(path, "r") as f:
            for line in f:
                raw_data.append(json.loads(line))

        assert len(raw_data) > 0, "No data found in {}".format(path)
        if limit is not None:
            raw_data = raw_data[:limit]

        # Initialize the docs (tokenize them)
        normalizer = nlp.get_pipe("normalizer")
        sentencizer = nlp.get_pipe("sentencizer")

        docs: List[Doc] = []
        for raw in raw_data:
            doc = nlp.make_doc(raw["note_text"])
            doc._.note_id = raw["note_id"]
            doc = normalizer(doc)
            doc = sentencizer(doc)
            docs.append(doc)

        # Annotate entities from the raw data
        for doc, raw in zip(docs, raw_data):
            ents = []
            span_groups = defaultdict(list)
            for ent in raw["entities"]:
                span = doc.char_span(
                    ent["start"],
                    ent["end"],
                    label=ent["label"],
                    alignment_mode="expand",
                )
                ents.append(span)
                span_groups[ent["label"]].append(span)
            doc.ents = filter_spans(ents)
            doc.spans.update(span_groups)

        new_docs = []
        for doc in docs:
            for sent in doc.sents:
                if len(span_getter(sent)):
                    new_doc = sent.as_doc(copy_user_data=True)
                    for group in doc.spans:
                        new_doc.spans[group] = [
                            Span(
                                new_doc,
                                span.start - sent.start,
                                span.end - sent.start,
                                span.label_,
                            )
                            for span in doc.spans.get(group, ())
                            if span.start >= sent.start and span.end <= sent.end
                        ]
                    new_docs.append(new_doc)
        return new_docs

    return load


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
    nlp: Pipeline,
    train_data: Callable[[Pipeline], Iterable[Doc]],
    val_data: Callable[[Pipeline], Iterable[Doc]],
    seed: int = 42,
    data_seed: int = 42,
    max_steps: int = 1000,
    batch_size: int = 4,
    lr: float = 8e-5,
    validation_interval: int = 10,
):
    set_seed(seed)
    with RichTablePrinter(
        {
            "step": {},
            "(.*_)?loss": {
                "goal": "lower_is_better",
                "format": "{:.2e}",
                "goal_wait": 2,
            },
            "ner/(ents_.*)": {
                "goal": "higher_is_better",
                "format": "{:.2%}",
                "goal_wait": 1,
                "name": r"\1",
            },
            "lr": {"format": "{:.2e}"},
            "speed": {"format": "{:.2f}"},
            "labels": {"format": "{:.2f}"},
        },
        auto_refresh=False,
    ) as logger:
        with set_seed(data_seed):
            train_docs: List[spacy.tokens.Doc] = list(train_data(nlp))
            if isinstance(val_data, float):
                offset = int(len(train_docs) * (1 - val_data))
                random.shuffle(train_docs)
                train_docs, val_docs = train_docs[:offset], train_docs[offset:]
            else:
                val_docs: List[spacy.tokens.Doc] = list(val_data(nlp))

        model_path = BASE_DIR / "artifacts/model-last"
        train_metrics_path = BASE_DIR / "artifacts/train_metrics.jsonl"

        # Initialize pipeline with training documents
        nlp.post_init(train_docs)

        # Preprocessing training data
        print("Preprocessing data")

        preprocessed = list(nlp.preprocess_many(train_docs, supervision=True))
        dataloader = DataLoader(
            preprocessed,
            batch_sampler=LengthSortedBatchSampler(preprocessed, batch_size),
            collate_fn=nlp.collate,
        )

        trf_params = set(nlp.get_pipe("ner").embedding.embedding.parameters())

        # Training loop
        trained_pipes = nlp.torch_components()
        print("Training", ", ".join([name for name, c in trained_pipes]))
        optimizer = ScheduledOptimizer(
            torch.optim.AdamW(
                [
                    {
                        "params": list(set(nlp.parameters()) - trf_params),
                        "lr": lr,
                        "schedules": [
                            LinearSchedule(
                                total_steps=max_steps,
                                start_value=lr,
                                path="lr",
                            )
                        ],
                    },
                    {
                        "params": list(trf_params),
                        "lr": lr,
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
        optimized_parameters = {
            p for group in optimizer.param_groups for p in group["params"]
        }
        print(
            "Optimizing:"
            + "".join(
                f"\n - {len(group['params'])} params "
                f"({sum(p.numel() for p in group['params'])} total)"
                for group in optimizer.param_groups
            )
        )
        print(
            f"Not optimizing {len(set(nlp.parameters()) - optimized_parameters)} params"
        )

        accelerator = Accelerator(cpu=True)
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
                    count = cumulated_data.pop("count")
                    scores = flatten_dict(nlp.score(val_docs))
                    print(scores)
                    metrics = {
                        "step": step,
                        "lr": optimizer.param_groups[0]["lr"],
                        # "emb_lr": optimizer.param_groups[1]["lr"],
                        **cumulated_data,
                        **scores,
                        "labels": cumulated_data["labels"] / max(count, 1),
                    }
                    cumulated_data = defaultdict(lambda: 0.0, count=0)
                    all_metrics.append(metrics)
                    logger.log_metrics(metrics)
                    train_metrics_path.write_text(json.dumps(all_metrics, indent=2))

                    nlp.to_disk(model_path)

                if step == max_steps:
                    break

                batch = next(iterator)
                n_words = batch["ner"]["embedding"]["mask"].sum().item()
                n_padded = torch.numel(batch["ner"]["embedding"]["mask"])
                n_words_bert = batch["ner"]["embedding"]["attention_mask"].sum().item()
                n_padded_bert = torch.numel(batch["ner"]["embedding"]["attention_mask"])
                bar.set_postfix(
                    n_words=n_words,
                    ratio=n_words / n_padded,
                    n_wp=n_words_bert,
                    bert_ratio=n_words_bert / n_padded_bert,
                )

                optimizer.zero_grad()
                with nlp.cache():
                    loss = torch.zeros((), device=accelerator.device)
                    for pipe in trained_pipes:
                        output = pipe.module_forward(batch[pipe.name])
                        if "loss" in output:
                            loss += output["loss"]
                        for key, value in output.items():
                            if key.endswith("loss"):
                                cumulated_data[key] += float(value)
                accelerator.backward(loss)
                optimizer.step()


if __name__ == "__main__":
    app()
