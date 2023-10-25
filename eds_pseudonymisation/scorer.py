import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List

import pandas as pd
import spacy
from confit import validate_arguments
from edsnlp import registry
from edsnlp.pipelines.base import SpanGetter, SpanGetterArg, get_spans
from edsnlp.scorers import make_examples
from edsnlp.scorers.ner import ner_exact_scorer, ner_token_scorer
from spacy.training import Example
from tqdm import tqdm

import eds_pseudonymisation.adapter  # noqa: F401


def redact_scorer(
    examples: Iterable[Example],
    span_getter: SpanGetter,
    micro_key: str = "micro",
) -> Dict[str, Any]:
    """
    Scores the extracted entities that may be overlapping or nested
    by looking in `doc.ents`, and `doc.spans`, and comparing the predicted
    and gold entities at the TOKEN level, without taking labels into account when
    matching two entities.

    Parameters
    ----------
    examples: Iterable[Example]
    span_getter: SpanGetter

    Returns
    -------
    Dict[str, Any]
    """
    # label -> pred, gold
    gold_labels = defaultdict(lambda: set())
    gold_labels[micro_key] = set()
    pred_labels = set()
    total_examples = 0
    for eg_idx, eg in enumerate(examples):
        for span in (
            span_getter(eg.predicted)
            if callable(span_getter)
            else get_spans(eg.predicted, span_getter)
        ):
            for i in range(span.start, span.end):
                pred_labels.add((eg_idx, i))

        for span in (
            span_getter(eg.reference)
            if callable(span_getter)
            else get_spans(eg.reference, span_getter)
        ):
            for i in range(span.start, span.end):
                gold_labels[span.label_].add((eg_idx, i))
                gold_labels[micro_key].add((eg_idx, i))

        total_examples += 1

    def prf(pred, gold):
        tp = len(set(pred) & set(gold))
        ng = len(gold)
        incomplete_docs = len(set(eg_idx for eg_idx, _ in set(gold) - set(pred)))
        return {
            "r": 1 if tp == ng else (tp / ng),
            "full": 1 - (incomplete_docs / total_examples),
            "tp": tp,  # num predicted
            "support": ng,  # num gold
        }

    return {name: prf(pred_labels, gold) for name, gold in gold_labels.items()}


@registry.scorers.register("eds_pseudo.ner_redact_scorer")
def create_redact_scorer(
    span_getter: SpanGetterArg,
):
    return lambda *args: redact_scorer(make_examples(*args), span_getter)


def table_prf(table):
    support = table["support"].sum()
    positives = table["positives"].sum(skipna=False)
    tp = table["tp"].sum()
    full_count = (table["tp"] == table["support"]).sum()
    count = len(table)
    res = {
        "p": float(support == tp if positives == 0 else tp / positives),
        "r": float(tp / support if support > 0 else 1.0),
        "f": float((tp * 2 / (support + positives)) if support > 0 else support == tp),
        "full": float(full_count / count),
    }
    return pd.Series(res)


@validate_arguments
class PseudoScorer:
    def __init__(
        self,
        ml_spans: str,
        rb_spans: str,
        hybrid_spans: str,
        main_mode="ml",
        # note_id: extract from test/edspdf/0123 -> edspdf
        note_id_regex: str = r"(?:.*/)?(?P<split>[^/]+)/(?:[^/]+)",
    ):
        self.ml_spans = ml_spans
        self.rb_spans = rb_spans
        self.hybrid_spans = hybrid_spans
        self.main_mode = main_mode
        self.note_id_regex = note_id_regex

    def __call__(self, nlp, docs, per_doc=False):
        clean_docs: List[spacy.tokens.Doc] = [d.copy() for d in docs]
        for d in clean_docs:
            d.ents = []
            d.spans.clear()
        t0 = time.time()
        preds = list(nlp.pipe(tqdm(clean_docs)))
        duration = time.time() - t0
        speeds = dict(
            wps=sum(len(d) for d in docs) / duration,
            dps=len(docs) / duration,
        )
        modes = {
            "ml": {self.ml_spans: True},
            "rb": {self.rb_spans: True},
            "hybrid": {self.hybrid_spans: True},
        }

        examples = make_examples(docs, preds)
        token_scores = ner_token_scorer(examples, modes[self.main_mode])["micro"]
        redact_scores = redact_scorer(examples, modes[self.main_mode])["micro"]

        scores = {
            **speeds,
            "p": token_scores["p"],
            "r": token_scores["r"],
            "f": token_scores["f"],
            "redact": redact_scores["r"],
            "full": redact_scores["full"],
        }

        if not per_doc:
            return scores

        per_doc_records = []
        scoring_fns = {
            "exact": ner_exact_scorer,
            "token": ner_token_scorer,
            "redact": redact_scorer,
        }
        for doc, pred in zip(docs, preds):
            for scoring_fn in ["exact", "token", "redact"]:
                for mode, span_getter in modes.items():
                    doc_scores = scoring_fns[scoring_fn](
                        [spacy.training.Example(pred, doc)],
                        span_getter=span_getter,
                    )
                    for label, values in doc_scores.items():
                        record = {
                            "matching": scoring_fn,
                            "label": label,
                            "mode": mode,
                            "note_class_source_value": doc._.note_class_source_value
                            or "unknown",
                            "note_id": doc._.note_id,
                        }
                        record.update(
                            {
                                key: value
                                for key, value in values.items()
                                if key in ("support", "positives", "tp")
                            }
                        )
                        per_doc_records.append(record)

        return scores, per_doc_records


class PRFEntry(dict):
    def __init__(self, tp=0, support=0, positives=0):
        self.tp = tp
        self.positives = positives
        self.support = support

    def items(self):
        prf = self.compute()
        return prf.items()

    def compute(self):
        res = {}
        if self.positives is not None:
            if self.positives == 0:
                res["f"] = res["p"] = float(self.positives == self.tp)
            else:
                res["f"] = 2 * self.tp / (self.positives + self.support)
                res["p"] = self.tp / self.positives
        if self.support > 0:
            res["r"] = self.tp / self.support
        else:
            res["r"] = float(self.tp == self.support)
        return res

    def __add__(self, other):
        return PRFEntry(
            tp=self.tp + other.tp,
            support=self.support + other.support,
            positives=self.positives + other.positives,
        )

    def __repr__(self):
        prf = self.compute()
        if "p" in prf:
            return "p: {p:.2f}, r: {r:.2f}, f: {f:.2f}".format(**prf)
        else:
            return "r: {r:.2f}".format(**prf)
