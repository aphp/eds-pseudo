import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Union

import spacy
from confit import validate_arguments
from spacy.training import Example

import eds_pseudo.adapter  # noqa: F401
from edsnlp.pipes.base import SpanGetter, get_spans
from edsnlp.scorers import make_examples
from edsnlp.scorers.ner import ner_exact_scorer, ner_token_scorer


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


@validate_arguments
class PseudoScorer:
    def __init__(
        self,
        ml_spans: str,
        rb_spans: str,
        hybrid_spans: str,
        main_mode="hybrid",
        # note_id: extract from test/edspdf/0123 -> edspdf
        note_id_regex: str = r"(?:.*/)?(?P<split>[^/]+)/(?:[^/]+)",
        labels: Union[bool, List[str]] = True,
    ):
        self.ml_spans = ml_spans
        self.rb_spans = rb_spans
        self.hybrid_spans = hybrid_spans
        self.main_mode = main_mode
        self.note_id_regex = note_id_regex
        self.labels = labels

    def __call__(self, nlp, docs, per_doc=False):
        clean_docs: List[spacy.tokens.Doc] = [d.copy() for d in docs]
        for d in clean_docs:
            d.ents = []
            d.spans.clear()
        t0 = time.time()
        preds = list(nlp.pipe(clean_docs).set_processing(show_progress=True))
        duration = time.time() - t0
        speeds = dict(
            wps=sum(len(d) for d in docs) / duration,
            dps=len(docs) / duration,
        )
        modes = {
            "ml": {self.ml_spans: self.labels},
            "rb": {self.rb_spans: self.labels},
            "hybrid": {self.hybrid_spans: self.labels},
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
