import itertools
import json
import math
import os
import random
import time
from collections import defaultdict
from itertools import chain, repeat
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import spacy
import torch
from accelerate import Accelerator
from confit import Cli, validate_arguments
from confit.utils.random import set_seed
from edsnlp.core.pipeline import Pipeline
from edsnlp.core.registry import registry
from edsnlp.optimization import LinearSchedule, ScheduledOptimizer
from edsnlp.scorers import Scorer
from edsnlp.utils.collections import batchify
from edsnlp.utils.filter import filter_spans
from rich_logger import RichTablePrinter
from spacy.tokens import Doc
from torch.utils.data import DataLoader
from tqdm import tqdm

app = Cli(pretty_exceptions_show_locals=False)

BASE_DIR = Path(__file__).parent.parent
LOGGER_FIELDS = {
    "step": {},
    "(.*_)?loss": {
        "goal": "lower_is_better",
        "format": "{:.2e}",
        "goal_wait": 2,
    },
    "token_ner_hybrid/ents_(.*)": {
        "goal": "higher_is_better",
        "format": "{:.2%}",
        "goal_wait": 1,
        "name": r"tok_hyb_\1",
    },
    "lr": {"format": "{:.2e}"},
    "speed/wps": {"format": "{:.2f}", "name": "wps"},
    "labels": {"format": "{:.2f}"},
}


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


@registry.misc.register("pseudo-dataset")
def pseudo_dataset(
    path,
    limit: Optional[int] = None,
    max_length: int = 0,
):
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

        def subset_doc(doc, start, end):
            # TODO: review user_data copy strategy
            new_doc = doc[start:end].as_doc(copy_user_data=True)
            new_doc.user_data.update(doc.user_data)

            for name, group in doc.spans.items():
                new_doc.spans[name] = [
                    spacy.tokens.Span(
                        new_doc,
                        max(0, span.start - start),
                        min(end, span.end) - start,
                        span.label,
                    )
                    for span in group
                    if span.end > start and span.start < end
                ]

            return new_doc

        def split_doc(doc):
            if max_length == 0 or len(doc) < max_length:
                yield doc
            else:
                start = 0
                end = 0
                for sent in (
                    doc.sents if doc.has_annotation("SENT_START") else (doc[:],)
                ):
                    if len(sent) == 0:
                        continue
                    # If the sentence adds too many tokens
                    if sent.end - start > max_length:
                        # But the current buffer too large
                        while sent.end - start > max_length:
                            yield subset_doc(doc, start, start + max_length)
                            start = start + max_length
                        yield subset_doc(doc, start, sent.end)
                        start = sent.end

                    # Otherwise, extend the current buffer
                    end = sent.end

                yield subset_doc(doc, start, end)

        docs: List[Doc] = []
        for raw in raw_data:
            doc = nlp.make_doc(raw["note_text"])
            doc._.note_id = raw["note_id"]
            doc._.note_datetime = raw.get("note_datetime")
            doc._.note_class_source_value = raw.get("note_class_source_value")
            doc._.context = raw.get("context", {})
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
                # ents.append(span)
                span_groups["pseudo-rb"].append(span)
                span_groups["pseudo-ml"].append(span)
                span_groups["pseudo-hybrid"].append(span)
            doc.ents = filter_spans(ents)
            doc.spans.update(span_groups)

        new_docs = []
        for doc in docs:
            for new_doc in split_doc(doc):
                if len(new_doc.text.strip()):
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


@validate_arguments
class PseudoScorer:
    def __init__(self, **scorers: Scorer):
        self.scorers = scorers

    def __call__(self, nlp, docs):
        clean_docs: List[spacy.tokens.Doc] = [d.copy() for d in docs]
        for d in clean_docs:
            d.ents = []
            d.spans.clear()
        t0 = time.time()
        preds = list(nlp.pipe(clean_docs))
        duration = time.time() - t0
        scores = {
            scorer_name: scorer(docs, preds)
            for scorer_name, scorer in self.scorers.items()
        }
        scores["speed"] = dict(
            wps=sum(len(d) for d in docs) / duration,
            dps=len(docs) / duration,
        )
        return scores


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
    scorer: PseudoScorer = PseudoScorer(),
):
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
                    scores = scorer(nlp, val_docs)
                    metrics = flatten_dict(
                        {
                            "step": step,
                            "lr": optimizer.param_groups[0]["lr"],
                            **cumulated_data,
                            **scores,
                            "labels": cumulated_data["labels"] / max(count, 1),
                        }
                    )
                    cumulated_data = defaultdict(lambda: 0.0, count=0)
                    all_metrics.append(metrics)
                    logger.log_metrics(metrics)
                    train_metrics_path.write_text(json.dumps(all_metrics, indent=2))

                    nlp.to_disk(model_path)

                if step == max_steps:
                    break

                batch = next(iterator)
                trf_batch = batch["ner"]["embedding"]["embedding"]
                n_words = trf_batch["mask"].sum().item()
                n_padded = torch.numel(trf_batch["mask"])
                n_words_bert = trf_batch["attention_mask"].sum().item()
                n_padded_bert = torch.numel(trf_batch["attention_mask"])
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
