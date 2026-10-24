from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from edsnlp import registry
from edsnlp.metrics import make_examples
from edsnlp.metrics.ner import NerExactMetric, NerOverlapMetric, NerTokenMetric
from edsnlp.pipes.base import SpanGetterArg, get_spans


def entity_set(entities: Sequence[Mapping[str, Any]], with_label: bool = True) -> set:
    """
    Set of `(start, end, label)` tuples of a list of entities, or of `(start, end)` with `with_label=False`, for
    the document metrics below
    """
    if with_label:
        return {(entity["start"], entity["end"], entity["label"]) for entity in entities}
    return {(entity["start"], entity["end"]) for entity in entities}


def exact(gold: Sequence[Mapping[str, Any]], pred: Sequence[Mapping[str, Any]]) -> bool:
    """
    Whether every gold entity of a document is predicted with the same boundaries and label
    """
    return entity_set(gold) <= entity_set(pred)


def span(gold: Sequence[Mapping[str, Any]], pred: Sequence[Mapping[str, Any]]) -> bool:
    """
    Whether every gold entity of a document is predicted with the same boundaries, whatever the label
    """
    return entity_set(gold, with_label=False) <= entity_set(pred, with_label=False)


def covered(gold: Sequence[Mapping[str, Any]], pred: Sequence[Mapping[str, Any]]) -> bool:
    """
    Whether the predictions cover every token of every gold entity whatever their label, which is what matters for
    redaction
    """
    predicted = {idx for entity in pred for idx in range(entity["start"], entity["end"])}
    return all(set(range(entity["start"], entity["end"])) <= predicted for entity in gold)


def partial(gold: Sequence[Mapping[str, Any]], pred: Sequence[Mapping[str, Any]]) -> bool:
    """
    Whether every gold entity overlaps at least one prediction
    """
    return all(
        any(entity["start"] < prediction["end"] and prediction["start"] < entity["end"] for prediction in pred)
        for entity in gold
    )


# Document metrics of `PopulationMetric`, where a document scores 1 if it passes the metric and 0 otherwise
DOCUMENT_METRICS = {
    "exact": exact,
    "span": span,
    "covered": covered,
    "partial": partial,
}


def covered_tokens(doc, span_getter: SpanGetterArg) -> set:
    """
    Indices of the tokens covered by the spans of `span_getter` whatever their label, which `redact_metric` and
    `PrivacyMetric` compare between gold and predicted docs (both docs have the same tokens)
    """
    return {idx for entity in get_spans(doc, span_getter) for idx in range(entity.start, entity.end)}


def redact_metric(examples, span_getter: SpanGetterArg) -> Dict[str, Any]:
    """
    Share of the gold entity tokens that predictions cover whatever their label (`token_r`), and share of documents
    whose gold tokens are all covered (`full`), for all the labels together (`micro`) and per gold label, e.g. the
    `NOM` scores count the gold `NOM` tokens covered by a prediction of any label, and a document without gold `NOM`
    counts as fully redacted for `NOM`
    """
    docs = []
    for example in make_examples(examples):
        gold = defaultdict(set)
        for entity in get_spans(example.reference, span_getter):
            gold[entity.label_].update(range(entity.start, entity.end))
        gold["micro"] = set().union(*gold.values())
        docs.append((covered_tokens(example.predicted, span_getter), gold))
    labels = sorted({label for _, gold in docs for label in gold} - {"micro"})
    result = {}
    for label in ["micro", *labels]:
        token_tp = sum(len(pred & gold[label]) for pred, gold in docs)
        token_support = sum(len(gold[label]) for _, gold in docs)
        result[label] = {
            "token_r": token_tp / token_support if token_support else 1.0,
            "full": sum(gold[label] <= pred for pred, gold in docs) / len(docs) if docs else 1.0,
            "token_tp": token_tp,
            "token_support": token_support,
            "documents": len(docs),
        }
    return result


@registry.metrics.register("eds_pseudo.redact")
class RedactMetric:
    """
    The `eds_pseudo.redact` metric scores redaction with `redact_metric`, so a gold name predicted as a city still
    counts as redacted
    """

    def __init__(self, span_getter: SpanGetterArg):
        self.span_getter = span_getter

    def __call__(self, *examples):
        return redact_metric(examples, span_getter=self.span_getter)


@registry.metrics.register("eds_pseudo.privacy")
class PrivacyMetric:
    """
    The `eds_pseudo.privacy` metric adds to `eds_pseudo.redact` the same scores per group of documents (question,
    encounter or person). A group is `full` only when all the gold tokens of all its documents are covered,
    e.g. a person whose name is missed in 1 of their 10 notes isn't fully protected.
    """

    def __init__(
        self,
        span_getter: SpanGetterArg,
        group_fields: Optional[List[str]] = None,
    ):
        self.span_getter = span_getter
        self.group_fields = group_fields or ["question_id", "encounter_id", "person_id"]

    def __call__(self, *examples) -> Dict[str, Any]:
        examples = make_examples(*examples)
        result = redact_metric(examples, span_getter=self.span_getter)
        for field in self.group_fields:
            groups = defaultdict(lambda: {"gold": 0, "covered": 0, "complete": True, "documents": 0})
            for idx, example in enumerate(examples):
                value = getattr(example.reference._, field, None)
                key = value if value is not None else f"document:{idx}"
                pred_tokens = covered_tokens(example.predicted, self.span_getter)
                gold_tokens = covered_tokens(example.reference, self.span_getter)
                groups[key]["gold"] += len(gold_tokens)
                groups[key]["covered"] += len(pred_tokens & gold_tokens)
                groups[key]["complete"] &= gold_tokens <= pred_tokens
                groups[key]["documents"] += 1
            total_gold = sum(group["gold"] for group in groups.values())
            result[field.removesuffix("_id")] = {
                "token_r": sum(group["covered"] for group in groups.values()) / total_gold if total_gold else 1.0,
                "full": sum(group["complete"] for group in groups.values()) / len(groups) if groups else 1.0,
                "groups": len(groups),
                "documents": sum(group["documents"] for group in groups.values()),
            }
        return result


def doc_entities(doc, span_getter: SpanGetterArg) -> List[Dict[str, Any]]:
    """
    Entities of a doc as `{start, end, label}` dicts of token indices, for the document metrics of `PopulationMetric`
    """
    return [
        {"start": entity.start, "end": entity.end, "label": entity.label_}
        for entity in get_spans(doc, span_getter)
    ]


@registry.metrics.register("eds_pseudo.population")
class PopulationMetric:
    """
    The `eds_pseudo.population` metric scores short field evaluation sets, where a question comes before each
    answer. It gives the document metrics of the answers overall and by source, exact entity P/R/F1 by source and
    label, and, with a `population_path` table of the questions of each encounter and person, the share of
    encounters and persons whose answers all pass each document metric, with its 95% bootstrap interval under `ci95`.

    These shares multiply, over the answers of an encounter, the share of passing test answers of each question, and a
    question often has a few test answers for thousands of answers in the population, so one test answer can move the
    share by several points. The interval shows how much, by resampling `bootstrap` times the test answers of each
    question.
    """

    def __init__(
        self,
        span_getter: SpanGetterArg,
        population_path: Optional[Path] = None,
        response_field: str = "original_text",
        source_field: str = "source",
        item_field: str = "question",
        population_group_fields: Sequence[str] = ("encounter_id", "person_id"),
        bootstrap: int = 1000,
    ):
        self.span_getter = span_getter
        self.population_path = population_path
        self.response_field = response_field
        self.source_field = source_field
        self.item_field = item_field
        self.population_group_fields = population_group_fields
        self.bootstrap = bootstrap

    def __call__(self, *examples) -> Dict[str, Any]:
        examples = make_examples(*examples)
        # Since the question comes before the answer in the text, we drop the predictions that end before the answer
        rows = []
        entity_counts = defaultdict(Counter)
        for idx, example in enumerate(examples):
            gold = doc_entities(example.reference, self.span_getter)
            pred = doc_entities(example.predicted, self.span_getter)
            response = getattr(example.reference._, self.response_field, None)
            response_char = example.reference.text.find(str(response)) if response is not None else 0
            response_char = response_char if response_char >= 0 else len(example.reference.text)
            # Index of the first token of the answer, since the entities are token spans
            response_start = next(
                (token.i for token in example.reference if token.idx >= response_char), len(example.reference)
            )
            pred = [entity for entity in pred if entity["end"] > response_start]
            source = getattr(example.reference._, self.source_field, None) or "unknown"
            row = {
                "note_id": getattr(example.reference._, "note_id", None) or idx,
                "source": source,
                "item": getattr(example.reference._, self.item_field, None),
                **{name: int(metric(gold, pred)) for name, metric in DOCUMENT_METRICS.items()},
            }
            rows.append(row)
            gold_counts = Counter((entity["start"], entity["end"], entity["label"]) for entity in gold)
            pred_counts = Counter((entity["start"], entity["end"], entity["label"]) for entity in pred)
            for entity, count in gold_counts.items():
                label = entity[2]
                entity_counts[(source, label)]["tp"] += min(count, pred_counts[entity])
                entity_counts[(source, label)]["fn"] += max(0, count - pred_counts[entity])
            for entity, count in pred_counts.items():
                entity_counts[(source, entity[2])]["fp"] += max(0, count - gold_counts[entity])

        # Document scores overall and by source, and exact entity P/R/F1 by source and label
        report = {
            "documents": len(rows),
            "document": {
                name: sum(row[name] for row in rows) / len(rows) if rows else 1.0 for name in DOCUMENT_METRICS
            },
            "source": {
                source: {
                    name: sum(row[name] for row in rows if row["source"] == source)
                    / sum(row["source"] == source for row in rows)
                    for name in DOCUMENT_METRICS
                }
                for source in sorted({row["source"] for row in rows})
            },
            "entity_f1": {},
        }
        for (source, label), counts in sorted(entity_counts.items()):
            precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0.0
            recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0.0
            report["entity_f1"].setdefault(source, {})[label] = {
                **counts,
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            }

        # An encounter or a person passes a metric only when all its answers pass it, so we average these passes
        # over the encounters and over the persons
        if self.population_path is not None:
            import pandas as pd

            population = pd.read_parquet(self.population_path)
            answers = pd.DataFrame(rows)
            scores = answers.groupby(["source", "item"], as_index=False)[list(DOCUMENT_METRICS)].mean()
            scores["answers"] = answers.groupby(["source", "item"]).size().to_numpy()
            population = population.rename(columns={self.source_field: "source", self.item_field: "item"})
            joined = population.merge(scores, on=["source", "item"], how="inner")
            report["population"] = {
                "items_in_evaluation": len(scores),
                "items_in_population": len(population[["source", "item"]].drop_duplicates()),
                "matched_items": len(joined[["source", "item"]].drop_duplicates()),
                "excluded_evaluation_items": len(
                    scores.merge(
                        population[["source", "item"]].drop_duplicates(),
                        on=["source", "item"],
                        how="left",
                        indicator=True,
                    ).query("_merge == 'left_only'")
                ),
                "source": {},
                "ci95": {},
            }
            rng = np.random.default_rng(0)
            for source, group in joined.groupby("source"):
                items = scores[scores.source == source].set_index("item").loc[sorted(group["item"].unique())]
                sizes = items["answers"].to_numpy()
                # (groups, items) number of answers of each encounter or person to each question, so that the share
                # of a group is exp(matrix @ log(rates)), and a rate of 0 gives exp(-inf) = 0 when the group answered
                matrices = {
                    field: group.groupby([field, "item"]).size().unstack(fill_value=0)[items.index].to_numpy()
                    for field in self.population_group_fields
                }
                report["population"]["source"][source], report["population"]["ci95"][source] = {}, {}
                for name in DOCUMENT_METRICS:
                    rates = items[name].to_numpy()
                    # (bootstrap, items) shares of passing test answers when we resample the answers of each question
                    resampled = rng.binomial(sizes, rates, size=(self.bootstrap, len(rates))) / sizes
                    report["population"]["source"][source][name], report["population"]["ci95"][source][name] = {}, {}
                    for field, matrix in matrices.items():
                        shares = [
                            np.exp(np.log(np.maximum(r, 1e-300)) @ matrix.T).mean(-1) for r in (rates, resampled)
                        ]
                        report["population"]["source"][source][name][field.removesuffix("_id")] = float(shares[0])
                        report["population"]["ci95"][source][name][field.removesuffix("_id")] = [
                            float(bound) for bound in np.percentile(shares[1], [2.5, 97.5])
                        ]
        return report


@registry.metrics.register("eds_pseudo.final")
class FinalMetric:
    """
    The `eds_pseudo.final` metric, which `eds-pseudo evaluate` runs on the whole pipeline: the exact, token and
    overlap NER metrics of EDS-NLP and `eds_pseudo.redact`, plus `eds_pseudo.privacy` when `group_fields` is set
    and `eds_pseudo.population` when `population_path` is set
    """

    def __init__(
        self,
        span_getter: SpanGetterArg,
        group_fields: Optional[List[str]] = None,
        population_path: Optional[Path] = None,
        response_field: str = "original_text",
        source_field: str = "source",
        item_field: str = "question",
        population_group_fields: Sequence[str] = ("encounter_id", "person_id"),
    ):
        self.exact = NerExactMetric(span_getter)
        self.token = NerTokenMetric(span_getter)
        self.overlap = NerOverlapMetric(span_getter)
        self.redact = RedactMetric(span_getter)
        self.privacy = PrivacyMetric(span_getter, group_fields) if group_fields else None
        self.population = (
            PopulationMetric(
                span_getter,
                population_path,
                response_field,
                source_field,
                item_field,
                population_group_fields,
            )
            if population_path
            else None
        )

    def __call__(self, *examples) -> Dict[str, Any]:
        result = {
            "exact": self.exact(*examples),
            "token": self.token(*examples),
            "overlap": self.overlap(*examples),
            "redact": self.redact(*examples),
        }
        if self.privacy is not None:
            result["privacy"] = self.privacy(*examples)
        if self.population is not None:
            result["population"] = self.population(*examples)
        return result
