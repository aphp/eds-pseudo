import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List

import spacy
from confit import validate_arguments
from edsnlp import registry
from edsnlp.pipelines.base import SpanGetter, SpanGetterArg, get_spans
from edsnlp.scorers import Scorer, make_examples
from spacy.training import Example
from tqdm import tqdm

import eds_pseudonymisation.adapter  # noqa: F401


def pseudo_ner_redact_scorer(
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
            "redact": 1 if tp == ng else (tp / ng),
            "redact_full": 1 - (incomplete_docs / total_examples),
        }

    results = {name: prf(pred_labels, gold) for name, gold in gold_labels.items()}
    return {
        "redact": results[micro_key]["redact"],
        "redact_full": results[micro_key]["redact_full"],
        "redact_per_type": results,
    }


@registry.scorers.register("eds_pseudo.ner_redact_scorer")
def create_pseudo_ner_redact_scorer(
    span_getter: SpanGetterArg,
):
    return lambda *args: pseudo_ner_redact_scorer(make_examples(*args), span_getter)


@validate_arguments
class PseudoScorer:
    def __init__(self, **scorers: Scorer):
        self.scorers = scorers

    def __call__(self, nlp, docs, per_doc=False):
        clean_docs: List[spacy.tokens.Doc] = [d.copy() for d in docs]
        for d in clean_docs:
            d.ents = []
            d.spans.clear()
        t0 = time.time()
        preds = list(nlp.pipe(tqdm(clean_docs)))
        duration = time.time() - t0
        scores = {
            scorer_name: scorer(docs, preds)
            for scorer_name, scorer in self.scorers.items()
        }
        scores["speed"] = dict(
            wps=sum(len(d) for d in docs) / duration,
            dps=len(docs) / duration,
        )
        if per_doc:
            return scores, [
                {
                    **{
                        scorer_name: scorer([doc], [pred])
                        for scorer_name, scorer in self.scorers.items()
                    },
                    "meta": {
                        "note_id": doc._.note_id,
                        "note_class_source_value": doc._.note_class_source_value,
                    },
                }
                for doc, pred in zip(docs, preds)
            ]
        return scores
