# ruff: noqa: E501

import logging
import os
import re
from collections import Counter
from functools import partial, reduce
from itertools import product
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, TypeVar, Union

import altair as alt
import dvc.api
import numpy as np
import pandas as pd
import scipy.stats
import spacy
import spacy.tokens
import spacy.training
from dvc.repo import Repo
from dvc.repo.experiments.show import show as experiments_show
from edsnlp.utils.filter import filter_spans
from pandas.api.types import is_numeric_dtype
from spacy.tokens import Doc
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


if not Doc.has_extension("context"):
    Doc.set_extension("context", default=dict())
if not Doc.has_extension("note_id"):
    Doc.set_extension("note_id", default=None)
if not Doc.has_extension("split"):
    Doc.set_extension("split", default=None)

It = TypeVar("It", bound=Iterable)
T = TypeVar("T")


def flatten_dict(root: Dict[str, Any], sep=None) -> Dict[str, Any]:
    res = {}

    def rec(d, path):
        for k, v in d.items():
            if isinstance(v, dict):
                rec(
                    v,
                    ((*path, *k) if isinstance(k, tuple) else (*path, k))
                    if sep is None
                    else path + sep + k,
                )
            else:
                res[
                    ((*path, *k) if isinstance(k, tuple) else (*path, k))
                    if sep is None
                    else path + sep + k
                ] = v

    for k, v in root.items():
        if isinstance(v, dict):
            rec(v, k if sep is not None else (k,))
        else:
            res[k] = v
    return res


def only_changed(df, include=()):
    new_columns = []
    for column in df:
        if column in include or len(set(map(str, df[column]))) > 1:
            new_columns.append(column)
    return df[new_columns]


def compute_prf(preds, golds):
    tp = set(preds) & set(golds)
    return {
        "precision": len(tp) / len(preds) if len(preds) > 0 else float(len(tp) == 0),
        "recall": len(tp) / len(golds) if len(golds) > 0 else float(len(tp) == 0),
        "f1": len(tp) * 2 / (len(preds) + len(golds))
        if len(preds) + len(golds) > 0
        else float(len(tp) == 0),
        "tp": len(tp),
        "gold_count": len(golds),
        "pred_count": len(preds),
    }


def get_token_ner_prf(examples, labels_mapping):
    items = {
        label: {"pred": set(), "gold": set()}
        for label in ("ALL", *labels_mapping.values())
    }

    for i, eg in enumerate(examples):
        for ent in eg.predicted.ents:
            if ent.label_ in labels_mapping:
                label = labels_mapping[ent.label_]
                items[label]["pred"].update(
                    (i, t)
                    for t in range(ent.start, ent.end)
                    if not eg.reference[t].is_space
                )
                items["ALL"]["pred"].update(
                    (i, t)
                    for t in range(ent.start, ent.end)
                    if not eg.reference[t].is_space
                )
    for i, eg in enumerate(examples):
        for ent in eg.reference.ents:
            if ent.label_ in labels_mapping:
                label = labels_mapping[ent.label_]
                items[label]["gold"].update(
                    (i, t)
                    for t in range(ent.start, ent.end)
                    if not eg.reference[t].is_space
                )
                items["ALL"]["gold"].update(
                    (i, t)
                    for t in range(ent.start, ent.end)
                    if not eg.reference[t].is_space
                )
    metrics = {
        label: {
            **compute_prf(label_items["pred"], label_items["gold"]),
            "redact": compute_prf(items["ALL"]["pred"], label_items["gold"])["tp"],
        }
        for label, label_items in items.items()
    }

    return metrics


def get_exact_ner_prf(examples, labels_mapping):
    items = {
        label: {"pred": set(), "gold": set()}
        for label in ("ALL", *labels_mapping.values())
    }

    for i, eg in enumerate(examples):
        for ent in eg.predicted.ents:
            if ent.label_ in labels_mapping:
                label = labels_mapping[ent.label_]
                items[label]["pred"].add((i, ent.start, ent.end))
                items["ALL"]["pred"].add((i, ent.start, ent.end))
    for i, eg in enumerate(examples):
        for ent in eg.reference.ents:
            if ent.label_ in labels_mapping:
                label = labels_mapping[ent.label_]
                items[label]["gold"].add((i, ent.start, ent.end))
                items["ALL"]["gold"].add((i, ent.start, ent.end))
    metrics = {
        label: {
            **compute_prf(label_items["pred"], label_items["gold"]),
            "redact": compute_prf(items["ALL"]["pred"], label_items["gold"])["tp"],
        }
        for label, label_items in items.items()
    }

    return metrics


def get_scoring_function(name):
    if name == "token":
        return get_token_ner_prf
    if name == "exact":
        return get_exact_ner_prf
    raise Exception()


def fix_entities(doc):
    new_ents = []
    doc = doc.copy()
    for ent in doc.ents:
        # if labels_mapping is not None and ent.label_ not in labels_mapping:
        #    continue
        # if labels_mapping is not None:
        #    label = labels_mapping.get(ent.label_, ent.label_)
        # else:
        #    label = ent.label_
        m = re.match(r"^\s*(.*?)\s*$", ent.text, flags=re.DOTALL)
        new_begin = m.start(1)
        new_end = m.end(1)
        new_ents.append(
            doc.char_span(
                ent[0].idx + new_begin,
                ent[0].idx + new_end,
                label=ent.label_,
                alignment_mode="expand",
            )
        )

    doc.ents = new_ents
    return doc


class ValueList:
    def __init__(self, values):
        self.values = []
        for val in values:
            if isinstance(val, ValueList):
                self.values.extend(val.values)
            else:
                self.values.append(val)

    def __iter__(self):
        return iter(self.values)

    def __add__(self, other):
        return ValueList([self, other])

    def __sub__(self, other):
        return scipy.stats.ttest_ind(self.values, other.values, equal_var=False)[1]

    def __float__(self):
        return self.mean()

    def mean(self):
        return np.mean(self.values)

    def std(self):
        return np.std(self.values)

    def __repr__(self):
        try:
            if len(self.values) > 1:
                return f"{self.mean():.1f} ± {self.std():.1f}"
            else:
                return f"{self.mean():.1f}" + " " * len(f" ± {self.std():.1f}")
        except TypeError:
            return repr(self.values)


def score_examples(
    preds: List[spacy.tokens.Doc],
    golds: List[spacy.tokens.Doc],
    labels_mapping: Optional[Dict[str, str]] = None,
    alignment: str = "token",
    return_scores_per_doc: bool = True,
):
    """
    Score predictions of a given experiment.

    Parameters
    ----------
    preds: List[spacy.tokens.Doc]
        Predictions of the experiment
    golds: List[spacy.tokens.Doc]
        Gold annotated documents of the experiment
    labels_mapping: Optional[Dict[str, str]]
        Mapping between annotated labels and displayed labels
    alignment: str
        Alignment method to use for scoring. Can be "token" or "exact"
    return_scores_per_doc: bool
        Whether to return the scores per document

    Returns
    -------
    Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]]
        Returns a first dataframe with the scores per label (row), containing the
        following columns:

        - precision: precision score for the label
        - recall: recall score for the label
        - f1: f1 score for the label
        - tp: number of true positives for the label
        - gold_count: number of gold entities for the label
        - pred_count: number of predicted entities for the label

        If return_scores_per_doc is True, returns a second dataframe with the scores
        per document, containing the following columns:

        - label: label of the scored entities (ALL means all entities)
        - note_id: note id of the document
        - split: split of the document
        - precision: precision score for the document only
        - recall: recall score for the document only
        - f1: f1 score for the document only
        - tp: number of true positives in the doc
        - gold_count: number of gold entities in the doc
        - pred_count: number of predicted entities in the doc
        We most likely will not need precision, recall and f1 since we will recompute
        them for lists of documents later using tp, gold_count and pred_count across
    """
    rb_examples = []

    for p, g in zip(preds, golds):
        assert p.text == g.text
        eg = spacy.training.Example(p, g)
        rb_examples.append(eg)

    score_fn = get_scoring_function(alignment)

    global_scores = score_fn(rb_examples, labels_mapping)
    global_scores = pd.DataFrame(global_scores)
    if labels_mapping:
        cols = list(dict.fromkeys(labels_mapping.values())) + ["ALL"]
        cols = [c for c in cols if c in global_scores]
        global_scores = global_scores.T.loc[cols]
    else:
        global_scores = global_scores.T
    if return_scores_per_doc:
        scores_per_doc = []
        for i, eg in enumerate(rb_examples):
            scores = score_fn([eg], labels_mapping)
            for label, vals in scores.items():
                if label != "ALL":
                    scores_per_doc.append(
                        {
                            **vals,
                            "label": label,
                            "note_id": eg.reference._.note_id,
                            "split": eg.reference._.split,
                        }
                    )
        return global_scores, scores_per_doc
    return global_scores


def get_corpus_stats(gold_filepath: str, labels_mapping: Dict[str, str]):
    """
    Get corpus statistics

    Parameters
    ----------
    gold_filepath: str
        Path to the gold annotated documents
    labels_mapping: Dict[str, str]
        Mapping between annotated labels and displayed labels

    Returns
    -------
    pd.DataFrame
        Corpus statistics per splits as columns and labels as rows (+ ENTS and DOCS)
        for the total number of entities and documents in the corpus
    """
    nlp = spacy.blank("eds")
    docs = list(spacy.tokens.DocBin().from_disk(gold_filepath).get_docs(nlp.vocab))

    splits = list(dict.fromkeys(d._.note_id.rsplit("/", 1)[0] for d in docs))

    per_split = {}
    for split in splits:
        stats = Counter()
        total_ents = 0
        total_docs = 0
        for doc in docs:
            if doc._.note_id.startswith(split):
                for ent in doc.ents:
                    stats.update((labels_mapping.get(ent.label_, ent.label_),))
                    total_ents += 1
                total_docs += 1
        res = {k: v for k, v in stats.items()}
        res["ENTS"] = total_ents
        res["DOCS"] = total_docs
        per_split[split] = res
    return pd.DataFrame(per_split)


def merge_docs(docs_a: Iterable[Doc], docs_b: Iterable[Doc]) -> Iterable[Doc]:
    """
    Merge two sets of documents. The two sets of documents must have the same
    number of documents and the same order.
    Entities of the 2 documents are merged and duplicates are removed, with priority to:

    - the largest span
    - the leftmost span if same size
    - the first entity if same size and same start

    Parameters
    ----------
    docs_a: List[spacy.tokens.Doc]
        List of documents
    docs_b: List[spacy.tokens.Doc]
        List of documents

    Returns
    -------
    Iterable[Doc]
        Merged documents
    """
    for a, b in zip(docs_a, docs_b):
        assert a.text == b.text, (a._.note_id, b._.note_id)
        doc = a.copy()
        doc.ents = filter_spans((*a.ents, *b.ents))
        yield doc


def postprocess_experiment_column(col):
    """
    Renames and post-processes columns of the experiment results dataframe

    Parameters
    ----------
    col: pd.Series

    Returns
    -------
    pd.Series
    """
    if col.name in ["f1", "recall", "precision", "full", "redact", "redact_full"]:
        col = col * 100
    if col.name.endswith("seed"):
        col = col.astype(str)
    if col.name.endswith("limit"):
        col = col.replace(0, 3373)
    if col.name == "params/paths/bert":
        col = col.apply(lambda x: x.strip("/").split("/")[-1], "")
        col = col.replace("training-from-scratch-2021-08-13", "scratch-pseudo")
        col = col.replace("checkpoint-250000", "finetuned-raw")
    return col


def process_doc_results(xp: pd.DataFrame) -> pd.DataFrame:
    """
    Process the results of an experiment

    Parameters
    ----------
    xp: pd.DataFrame
        Results of an experiment as returned by

    Returns
    -------

    """

    def pd_prf(group):
        # fmt: off
        agg = group[["tp", "gold_count", "pred_count", "redact"]].sum()  # noqa: E501
        agg["f1"] = (agg["tp"] * 2 / (agg["gold_count"] + agg["pred_count"])) if agg["gold_count"] + agg["pred_count"] > 0 else 1.  # noqa: E501
        agg["precision"] = (agg["tp"] / agg["pred_count"]) if agg["pred_count"] > 0 else 1.  # noqa: E501
        agg["recall"] = (agg["tp"] / agg["gold_count"]) if agg["gold_count"] > 0 else 1  # noqa: E501
        agg["full"] = group.groupby("note_id", as_index=False).sum().eval("tp >= gold_count").mean()  # noqa: E501
        agg["redact"] = (agg["redact"] / agg["gold_count"]) if agg["gold_count"] > 0 else 1  # noqa: E501
        agg["redact_full"] = group.groupby("note_id", as_index=False).sum().eval("redact >= gold_count").mean()  # noqa: E501
        # fmt: on
        return agg

    return pd.concat(
        [
            xp.groupby(["label", "note_class_source_value"], as_index=False).apply(
                pd_prf
            ),
            xp.assign(label="ALL")
            .groupby(["label", "note_class_source_value"], as_index=False)
            .apply(pd_prf),
            xp.assign(note_class_source_value="ALL")
            .groupby(["label", "note_class_source_value"], as_index=False)
            .apply(pd_prf),
            xp.assign(label="ALL", note_class_source_value="ALL")
            .groupby(["label", "note_class_source_value"], as_index=False)
            .apply(pd_prf),
        ]
    )


def cast_categories(x):
    return x.astype(
        {
            c: "category" if not is_numeric_dtype(dtype) else dtype
            for c, dtype in x.dtypes.items()
        }
    )


def score_experiments(
    experiments: pd.DataFrame,
    xp_output_filepath: Union[str, Path],
    rb_only_filepath: Union[str, Path],
    rb_to_merge_filepath: Union[str, Path],
    gold_filepath: Union[str, Path],
    metadata_filepath: Union[str, Path],
    labels_mapping: Dict[str, str],
    split: str = "test/edspdf",
) -> pd.DataFrame:
    """
    Score experiments

    Parameters
    ----------
    experiments: pd.DataFrame
        Experiment as returned by `load_experiments`
    xp_output_filepath: Union[str, Path]
        Path to the output spacy.tokens.DocBin file containing the ML predictions
        This file is expected to change between DVC experiments
    rb_only_filepath: Union[str, Path]
        Path to the output spacy.tokens.DocBin file containing the RB predictions
    rb_to_merge_filepath: Union[str, Path]
        Path to the output spacy.tokens.DocBin file containing the RB predictions
        optimized for merging with the ML predictions (only the most precise rules)
    gold_filepath: Union[str, Path]
        Path to the gold spacy.tokens.DocBin file
    labels_mapping: Dict[str, str]
        Mapping from the labels in the gold file to the labels in the predictions
    metadata_filepath: Union[str, Path]
        Path to the metadata file
    split: str
        The split to use for the evaluation

    Returns
    -------
    pd.DataFrame
    """
    nlp = spacy.blank("eds")
    full = list(spacy.tokens.DocBin().from_disk(gold_filepath).get_docs(nlp.vocab))
    for doc in full:
        doc._.split = doc._.note_id.rsplit("/", 1)[0]
        # doc._.note_id = int(doc._.note_id)
    split_df = pd.DataFrame(
        [
            {
                "note_id": d._.note_id,
                "omop_note_id": int(d._.note_id.rsplit("/", 1)[1]),
                # "split": d._.note_id.rsplit("/", 1)[0]
            }
            for d in full
        ]
    )
    metadata = (
        pd.read_json(metadata_filepath, lines=True)[
            [
                "note_id",
                "note_class_source_value",
                "cdm_source",
                "visit_occurrence_id",
                "person_id",
                "note_datetime",
            ]
        ]
        .rename({"note_id": "omop_note_id"}, axis=1)
        .merge(split_df)
    )

    test_ids = set(d._.note_id for d in full if split in d._.split)
    rb_only_preds = sorted(
        [
            fix_entities(d)
            for d in spacy.tokens.DocBin()
            .from_disk(rb_only_filepath)
            .get_docs(nlp.vocab)
            if d._.note_id in test_ids
        ],
        key=lambda x: x._.note_id,
    )
    rb_to_merge_preds = sorted(
        [
            fix_entities(d)
            for d in spacy.tokens.DocBin()
            .from_disk(rb_to_merge_filepath)
            .get_docs(nlp.vocab)
            if d._.note_id in test_ids
        ],
        key=lambda x: x._.note_id,
    )

    # Loading and filtering predictions
    all_mode_results = []

    gold = sorted(
        [d for d in full if d._.note_id in test_ids], key=lambda x: x._.note_id
    )
    assert len(gold) > 0

    for alignment in ["token", "exact"]:
        per_doc_results = score_examples(
            rb_only_preds,
            gold,
            alignment=alignment,
            labels_mapping=labels_mapping,
        )[1]
        per_doc_results = process_doc_results(
            pd.DataFrame.from_records(per_doc_results).merge(metadata)
        ).assign(alignment=alignment, mode="rb", rev="rb")
        all_mode_results.append(per_doc_results)

    pool = Pool(8)
    [i.close() for i in list(tqdm._instances)]

    def update(*a):
        bar.update()

    bar = tqdm(total=len(experiments["rev"].drop_duplicates()), unit="experiment")
    futures = []

    for rev in experiments["rev"].drop_duplicates():
        fn = partial(
            score_experiment,
            xp_output_filepath=xp_output_filepath,
            nlp=nlp,
            test_ids=test_ids,
            rb_preds=rb_to_merge_preds,
            gold=gold,
            labels_mapping=labels_mapping,
            metadata=metadata,
        )
        futures.append(pool.apply_async(fn, args=(rev,), callback=update))

    def default(x, default):
        return default if x is None else x

    all_mode_results.extend(
        [df for future in futures for df in default(future.get(), ())]
    )

    all_mode_results = pd.concat(all_mode_results)
    return all_mode_results


def score_experiment(
    rev: str,
    xp_output_filepath: Union[str, Path],
    nlp: spacy.language.Language,
    test_ids: Set[str],
    rb_preds: List[spacy.tokens.Doc],
    gold: List[spacy.tokens.Doc],
    labels_mapping: Dict[str, str],
    metadata: pd.DataFrame,
):
    """
    Score a single experiment

    Parameters
    ----------
    rev: str
        The DVC revision to use
    xp_output_filepath: Union[str, Path]
        Path to the output spacy.tokens.DocBin file containing the ML predictions
    nlp: spacy.language.Language
        A spacy language model used to load the predictions from the DocBin
    test_ids: Set[str]
        The ids subset to use for the evaluation
    rb_preds: List[spacy.tokens.Doc]
        The rule-based predictions to merge with the ML predictions
    gold: List[spacy.tokens.Doc]
        The gold standard documents
    labels_mapping: Dict[str, str]
        Mapping from annotated labels to displayed labels
    metadata: pd.DataFrame
        The metadata of the documents

    Returns
    -------
    List[pd.DataFrame]
        A list of dataframes containing the results for each alignment and mode. Each
        dataframe containing the scores per document
    """
    try:
        results = []

        if not isinstance(rev, str) or rev == "workspace":
            db = spacy.tokens.DocBin().from_bytes(Path(xp_output_filepath).read_bytes())
        else:
            db = spacy.tokens.DocBin().from_bytes(
                dvc.api.read(path=xp_output_filepath, rev=rev, mode="rb")
            )
        xp_preds = sorted(
            [
                fix_entities(d)
                for d in db.get_docs(nlp.vocab)
                if d._.note_id in test_ids
            ],
            key=lambda x: x._.note_id,
        )

        # Computing merged outputs
        merged = [fix_entities(d) for d in list(merge_docs(xp_preds, rb_preds))]

        # Scoring outputs
        for alignment in ["exact", "token"]:
            for mode, df in [("ml", xp_preds), ("merged", merged)]:
                per_doc_results = pd.DataFrame.from_records(
                    score_examples(
                        preds=df,
                        golds=gold,
                        alignment=alignment,
                        labels_mapping=labels_mapping,
                    )[1]
                )
                per_doc_results = process_doc_results(
                    per_doc_results.merge(metadata, on="note_id")
                )
                per_doc_results = per_doc_results.assign(
                    alignment=alignment, mode=mode, rev=rev
                )
                results.append(per_doc_results)

        return results
    except Exception:
        logging.error(f"Could not score experiment: {rev}")
        # import traceback
        # traceback.print_exc()
        return None


def load_experiments(
    repo_path: Union[Path, str],
    params_mapping: Dict[str, Any] = {},
) -> pd.DataFrame:
    """
    Load experiments from a DVC repository

    Parameters
    ----------
    repo_path: Union[Path, str]
        Path to the DVC repository
    params_mapping: Dict[str, Any]
        Mapping between DVC parameters and displayed parameters

    Returns
    -------
    pd.DataFrame
        A dataframe containing the experiments and the following columns:

        - rev: The revision of the experiment
        - params.*: The parameters of the experiment
        - commit: The commit of the experiment
    """
    logging.info("Loading experiments")

    with Repo.open(repo_path) as repo:
        data = experiments_show(repo=repo, all_commits=True)

    results = []
    for commit, commit_data in data.items():
        for k, v in commit_data.items():
            v_data = v["data"] if "data" in v else v["baseline"]["data"]
            metrics = next(iter(v_data["metrics"].values()), {"data": {}})["data"]
            # This is only to ensure that the experiment has run
            if len(metrics) == 0:
                continue
            params = next(iter(v_data["params"].values()))["data"]
            results.append(
                flatten_dict(
                    {
                        "rev": v_data.get("name", "workspace"),
                        "params": params,
                        # "metrics": metrics,
                        "commit": commit,
                    },
                    "/",
                )
            )
    experiments = (
        only_changed(
            pd.DataFrame(
                results,
                columns=list(dict.fromkeys(col for r in results for col in r.keys())),
            )
            .apply(postprocess_experiment_column)
            .rename(params_mapping, axis=1),
            include=list(params_mapping.values()),
        )
        if len(results) > 1
        else pd.DataFrame(results, columns=results[-1].keys())
    )
    experiments["rev"] = experiments["rev"].fillna("workspace")
    experiments = cast_categories(experiments)
    logging.info("Found {} experiments".format(len(experiments)))
    return experiments


def evaluate_rules(
    gold_filepath: Union[Path, str] = "corpus/full.spacy",
    split: str = "dev",
):
    """
    Evaluate the rules on the gold corpus

    Parameters
    ----------
    gold_filepath: Union[Path, str]
        Path to the gold corpus
    split: str
        Split to evaluate on (dev or test)

    Returns
    -------
    pd.DataFrame
    """

    def score_pipeline(
        components,
        labels,
        name,
        split="dev",
        labels_mapping=None,
    ):
        nlp = spacy.blank("eds")
        nlp.add_pipe("eds.remove-lowercase", name="remove-lowercase")
        nlp.add_pipe("eds.accents", name="accents")
        for component in components:
            if isinstance(component, str):
                component_name, config = component, None
                nlp.add_pipe(component_name)
            else:
                component_name, config = component
                nlp.add_pipe(component_name, config=config)
        gold = list(spacy.tokens.DocBin().from_disk(gold_filepath).get_docs(nlp.vocab))
        gold = [g for g in gold if split in g._.note_id]
        gold = sorted(gold, key=lambda x: x._.note_id)
        pred = []
        for d in gold:
            d = d.copy()
            d.ents = []
            pred.append(d)
        pred = list(nlp.pipe(pred))
        return (
            score_examples(pred, gold, labels_mapping, alignment="token")[0]
            .loc[labels]
            .assign(name=name)
        )

    df = pd.concat(
        [
            score_pipeline(
                [("pseudonymisation-rules", {"pattern_keys": ["TEL"]})],
                ["PHONE"],
                name="static",
            ),
            score_pipeline(
                [("pseudonymisation-rules", {"pattern_keys": ["SECU"]})],
                ["NSS"],
                name="static",
            ),
            score_pipeline(
                [("pseudonymisation-rules", {"pattern_keys": ["MAIL"]})],
                ["EMAIL"],
                name="static",
            ),
            score_pipeline(
                [("pseudonymisation-rules", {"pattern_keys": ["NDA"]})],
                ["VISIT ID"],
                name="static",
            ),
            score_pipeline(
                [("pseudonymisation-rules", {"pattern_keys": ["PERSON"]})],
                ["FIRSTNAME", "LASTNAME"],
                name="static",
            ),
            score_pipeline(
                [
                    "pseudonymisation-dates",
                    ("pseudonymisation-rules", {"pattern_keys": []}),
                ],
                ["DATE"],
                name="static",
            ),
            score_pipeline(
                ["pseudonymisation-addresses"],
                ["CITY", "ADDRESS", "ZIP"],
                name="static",
            ),
            score_pipeline(["structured-data-matcher"], slice(None), name="dynamic"),
        ],
        axis=0,
    )

    # Put the data in a nice format
    return (
        df.reset_index()
        .rename({"index": "label"}, axis=1)
        .rename(columns={"precision": "Precision", "recall": "Recall", "f1": "F1"})
        .set_index(["name", "label"])  # [["precision", "recall", "f1"]]
        .unstack("name")
        .swaplevel(0, 1, 1)
        .sort_index(1)
        .reindex(
            columns=[
                (p, m)
                for p in ("static", "dynamic")
                for m in ("Precision", "Recall", "F1")
            ]
        )
        .loc[
            [
                "ADDRESS",
                "BIRTHDATE",
                "CITY",
                "DATE",
                "EMAIL",
                "FIRSTNAME",
                "LASTNAME",
                "NSS",
                "PATIENT ID",
                "PHONE",
                "VISIT ID",
                "ZIP",
            ]
        ]
    ).applymap(lambda x: "—" if pd.isna(x) else "{:.1f}".format(x * 100))


def get_annotators_docs(path: Union[Path, str]):
    """
    Load the annotations from the JSONL annotations file
    before the agreement computation

    Parameters
    ----------
    path: Union[Path, str]
        Path to the JSONL annotations file

    Returns
    -------
    pd.DataFrame
        Documents with the following columns:

        - note_id: The note id
        - split: The split of the note
        - subsplit: The subsplit of the note
        - annotator: The annotator of the note
        - doc: The spacy document
    """
    raw = pd.read_json(path, lines=True)

    if not spacy.tokens.Doc.has_extension("note_id"):
        spacy.tokens.Doc.set_extension("note_id", default=None)
    if not spacy.tokens.Doc.has_extension("split"):
        spacy.tokens.Doc.set_extension("split", default=None)
    if not spacy.tokens.Doc.has_extension("subsplit"):
        spacy.tokens.Doc.set_extension("subsplit", default=None)
    if not spacy.tokens.Doc.has_extension("annotator"):
        spacy.tokens.Doc.set_extension("annotator", default=None)

    nlp = spacy.blank("eds")
    docs = []
    for entry in tqdm(raw.itertuples(index=False)):
        doc = nlp(entry.data["text"])
        doc._.note_id = entry.data["note_id"]
        doc._.split = entry.data["meta_split"]
        doc._.subsplit = entry.data["split"]
        doc._.annotator = entry.annotator[0]
        ents = []
        for group in entry.annotations:
            for res in group["result"]:
                ent = doc.char_span(
                    res["value"]["start"],
                    res["value"]["end"],
                    res["value"]["labels"][0],
                    alignment_mode="expand",
                )
                ents.append(ent)
        doc.ents = filter_spans(ents)
        docs.append(doc)

    docs_df = pd.DataFrame(
        [
            {
                "annotator": doc._.annotator,
                "note_id": doc._.note_id,
                "subsplit": doc._.subsplit,
                "doc": fix_entities(doc),
            }
            for doc in docs
        ]
    )

    return docs_df


def plot_limit_ablation(results, experiments):
    max_limit = experiments["limit"].max()
    min_limit = experiments["limit"].min()

    plot_data = select_displayed_data(
        results.merge(
            pd.concat(
                [
                    experiments,
                    pd.DataFrame(
                        [
                            {
                                "rev": "rb",
                                "limit": min_limit,
                                "bert": "rule-based",
                            },
                            {
                                "rev": "rb",
                                "limit": max_limit,
                                "bert": "rule-based",
                            },
                        ]
                    ),
                ]
            )
        ),
        index=["limit", "alignment", "mode", "rev", "split", "label", "bert"],
        columns=["f1", "recall", "precision", "full", "redact", "redact_full"],
    ).rename(
        {
            "mode": "Model",
        },
        axis=1,
    )
    plot_data["Model"] = plot_data["Model"].apply(
        lambda x: {
            "merged": "Hybrid",
            "ml": "ML",
            "rb": "Rule-based",
        }[x]
    )

    selections = []
    selections, plot_data = add_altair_selectors(
        {
            "metric": "f1",
            # "note_class_source_value": "ALL",
            "label": "ALL",
            "alignment": "token",
            "split": "test/edspdf",
        },
        plot_data,
    )

    base_chart = alt.Chart(plot_data).encode(
        x=alt.X(
            "limit:Q",
            scale=alt.Scale(zero=False, domain=(min_limit, max_limit)),
        ),
        color="Model:N",
    )

    line = base_chart.mark_line().encode(
        y=alt.Y("mean(value):Q", scale=alt.Scale(zero=False)),
    )

    band = base_chart.mark_errorband(extent="ci").encode(
        y=alt.Y("value", scale=alt.Scale(zero=False)),
    )

    alt.data_transformers.disable_max_rows()
    chart = reduce(
        lambda x, s: x.transform_filter(s),
        selections,
        (line + band).add_selection(*selections),
    ).properties(height=400, width=400, title="Train size ablation")
    return chart


def select_displayed_data(x, index, columns=()):
    """

    Parameters
    ----------
    x
    index
    columns

    Returns
    -------

    """
    y = x.drop(
        columns=[c for c in x.columns if c in columns] + ["rev", "params/system/seed"]
    )
    y = y.drop_duplicates()
    y = y.groupby([c for c in index if c in y]).apply(
        lambda x: x.apply(lambda v: len(set(v)))
    )
    uncontrolled_columns = list(y.columns[(y > 1).any(0)])
    if len(uncontrolled_columns) > 0:
        raise Exception(
            "Data has an some uncontrolled columns: {}".format(uncontrolled_columns)
        )
    return x.melt(
        id_vars=index, value_vars=columns, var_name="metric", value_name="value"
    )


def plot_doc_type_ablation(results, experiments, return_data=False):
    logging.info("Plotting document type ablation experiments")

    ablation_experiments = experiments.copy()
    ablation_experiments = ablation_experiments[
        ~ablation_experiments.filter_expr.isna()
    ]
    ablation_experiments.filter_expr = ablation_experiments.filter_expr.apply(
        lambda x: x[34:-1]
    )
    ablations = list(ablation_experiments.filter_expr.drop_duplicates())

    plot_data = select_displayed_data(
        results.merge(
            pd.concat(
                [
                    ablation_experiments,
                    experiments.query(
                        'bert == "finetuned-raw" and '
                        "filter_expr.isna() and "
                        "limit == limit.max()"
                    ),
                ]
            ).reset_index(drop=True),
            on="rev",
        )
        .query("label == 'ALL'")
        .query(
            f"note_class_source_value == filter_expr or ("
            f"filter_expr.isna() and note_class_source_value.isin({ablations})"
            f")"
        )
        .rename({"filter_expr": "included"}, axis=1)
        .replace(
            to_replace={"included": r"^(?!yes).*$"},
            value={"included": "no"},
            regex=True,
        )
        .rename({"note_class_source_value": "Document type"}, axis=1)
        .astype(object)
        .fillna("yes"),
        # .eval("model = `bert`.str.cat(mode, ' & ')")
        index=["split", "alignment", "mode", "Document type", "included"],
        columns=["f1", "recall", "precision", "full", "redact", "redact_full"],
    )

    if return_data:
        return plot_data

    selections, plot_data = add_altair_selectors(
        {
            "mode": "merged",
            "metric": "f1",
            "alignment": "token",
            "split": "test/edspdf",
        },
        plot_data,
    )

    base_chart = alt.Chart(plot_data).encode(
        x=alt.X("included:N"),
    )

    line = base_chart.mark_bar().encode(
        y=alt.Y("mean(value):Q", scale=alt.Scale(zero=True), stack=None),
        color="included:N",
    )

    band = base_chart.mark_errorbar(extent="ci").encode(
        y=alt.Y("value", scale=alt.Scale(zero=False)),
        strokeWidth=alt.value(2),
    )

    alt.data_transformers.disable_max_rows()

    chart = (
        reduce(
            lambda x, s: x.transform_filter(s),
            selections,
            (line + band).add_selection(*selections),
        )
        .properties(
            height=400,
            width=200,
        )
        .facet(column="Document type", title="Document type ablation")
    )
    return chart


def plot_bert_ablation(results, experiments):
    logging.info("Plotting BERT ablation experiments")

    experiments["limit"].max()

    plot_data = select_displayed_data(
        results.merge(
            pd.concat(
                [
                    experiments,
                    pd.DataFrame(
                        [
                            {"rev": "rb", "bert": bert}
                            for bert in experiments.bert.drop_duplicates()
                        ]
                    ),
                ]
            )
        ),
        # .eval("model = `bert`.str.cat(mode, ' & ')")
        index=[
            "split",
            "bert",
            "alignment",
            "mode",
            "rev",
            "note_class_source_value",
            "label",
        ],
        columns=["f1", "recall", "precision", "full", "redact", "redact_full"],
    )

    selections = []
    for name, init in {
        "metric": "f1",
        "note_class_source_value": "ALL",
        "label": "ALL",
        "alignment": "token",
        "split": "test/edspdf",
    }.items():
        options = list(plot_data[name].unique())
        if len(options) == 1:
            plot_data = plot_data.drop(columns=[name])
            continue
        selection_box = alt.binding_select(options=options, name=name + " : ")
        selections.append(
            alt.selection_single(
                fields=[name],
                bind=selection_box,
                init={name: plot_data[name].iloc[0] if init not in options else init},
            )
        )

    base_chart = alt.Chart(plot_data).encode(x="mode:N")

    line = base_chart.mark_bar().encode(
        y=alt.Y("mean(value):Q", scale=alt.Scale(zero=False), stack=None),
        color="mode:N",
    )

    band = base_chart.mark_errorbar(extent="ci").encode(
        y=alt.Y("value", scale=alt.Scale(zero=False)),
        strokeWidth=alt.value(2),
    )

    alt.data_transformers.disable_max_rows()

    chart = (
        reduce(
            lambda x, s: x.transform_filter(s),
            selections,
            (line + band).add_selection(*selections),
        )
        .properties(
            height=400,
            width=200,
        )
        .facet(column="bert", title="BERT model ablation")
    )
    return chart


def add_altair_selectors(defaults, plot_data):
    selections = []
    for name, init in defaults.items():
        options = list(plot_data[name].unique())
        if len(options) == 1:
            plot_data = plot_data.drop(columns=[name])
            continue
        selection_box = alt.binding_select(options=options, name=name + " : ")
        selections.append(
            alt.selection_single(
                fields=[name],
                bind=selection_box,
                init={name: plot_data[name].iloc[0] if init not in options else init},
            )
        )
    return selections, plot_data


def plot_labels(results, experiments):
    logging.info("Plotting results by labels")

    max_limit = experiments["limit"].max()
    min_limit = experiments["limit"].min()

    plot_data = select_displayed_data(
        results.merge(
            pd.concat(
                [
                    experiments,
                    pd.DataFrame(
                        [
                            point
                            for bert in experiments.bert.drop_duplicates()
                            for point in (
                                {"rev": "rb", "limit": min_limit, "bert": bert},
                                {"rev": "rb", "limit": max_limit, "bert": bert},
                            )
                        ]
                    ),
                ]
            )
        ),
        index=[
            "split",
            "bert",
            "alignment",
            "mode",
            "rev",
            "note_class_source_value",
            "label",
        ],
        columns=["f1", "recall", "precision", "full", "redact", "redact_full"],
    )

    selections, plot_data = add_altair_selectors(
        {
            "metric": "f1",
            "note_class_source_value": "ALL",
            "alignment": "token",
            "bert": "finetuned-raw",
            "split": "test/edspdf",
        },
        plot_data,
    )

    base_chart = alt.Chart(plot_data).encode(
        x="mode:N",
    )

    line = base_chart.mark_bar().encode(
        y=alt.Y("mean(value):Q", scale=alt.Scale(zero=False), stack=None),
        color="mode:N",
    )

    band = base_chart.mark_errorbar(extent="ci").encode(
        y=alt.Y("value", scale=alt.Scale(zero=False)),
        strokeWidth=alt.value(2),
    )

    alt.data_transformers.disable_max_rows()

    chart = (
        reduce(
            lambda x, s: x.transform_filter(s),
            selections,
            (line + band).add_selection(*selections),
        )
        .properties(
            height=400,
            width=200,
        )
        .facet("label", columns=4, title="Label")
    )
    return chart


def plot_iaa_pairs(docs_df: pd.DataFrame, labels_mapping: Dict[str, str]):
    """
    Plot Inter-Annotator Agreement (IAA) between pairs of annotators

    Parameters
    ----------
    docs_df: pd.DataFrame
        Annotator documents as returned by the `get_annotator_docs` function
    labels_mapping: Dict[str, str]
        Mapping between annotated labels and displayed labels

    Returns
    -------
    altair.Chart
    """
    logging.info("Plotting IAA pairs")

    scores = []
    annotators = docs_df.annotator.drop_duplicates()
    for ann1, ann2 in tqdm(product(annotators, annotators)):
        if ann1 != ann2:
            cross_annotations = pd.merge(
                docs_df, docs_df, on=["note_id", "subsplit"]
            ).query(f"annotator_x == '{ann1}' and annotator_y == '{ann2}'")
            for alignment in ["exact", "token"]:
                metrics = score_examples(
                    preds=cross_annotations["doc_x"].tolist(),
                    golds=cross_annotations["doc_y"].tolist(),
                    labels_mapping=labels_mapping,
                    alignment=alignment,
                    return_scores_per_doc=True,
                )[0]
                for label, label_metrics in metrics.iterrows():
                    scores.append(
                        {
                            "ann1": ann1,
                            "ann2": ann2,
                            "value": label_metrics["f1"],
                            "tp": int(label_metrics["tp"]),
                            "tp_str": "TP: {}".format(int(label_metrics["tp"])),
                            "label": label,
                            "alignment": alignment,
                        }
                    )

    plot_data = pd.DataFrame(scores)

    selections = []
    for name, init in {
        "label": "ALL",
        "alignment": "token",
    }.items():
        selection_box = alt.binding_select(
            options=list(plot_data[name].unique()), name=name + " : "
        )
        selections.append(
            alt.selection_single(fields=[name], bind=selection_box, init={name: init})
        )

    base_chart = alt.Chart(plot_data).encode(
        x=alt.X("ann1:N", title="Annotator"),
        y=alt.Y("ann2:N", title="Annotator"),
    )

    rect = base_chart.mark_bar().encode(
        color=alt.Color(
            "value:Q",
            title="F1",
            scale=alt.Scale(scheme="goldred", domain=(0.9, 1.0)),
            legend=alt.Legend(format="%"),
        )
    )

    text1 = base_chart.mark_text(color="black").encode(
        text=alt.Text("value:Q", format="%")
    )
    text2 = base_chart.mark_text(color="black", dy=10).encode(text=alt.Text("tp_str:N"))

    alt.data_transformers.disable_max_rows()

    chart = reduce(
        lambda x, s: x.transform_filter(s),
        selections,
        (rect + text1 + text2).add_selection(*selections),
    )
    return chart.properties(
        height=400, width=400, title="Inter annotator agreement (IAA)"
    )


def plot_micro_iaa(docs_df: pd.DataFrame, labels_mapping: Dict[str, str]):
    """
    Plot Inter-Annotator Agreement (IAA) between all annotators, split by label
    and metric

    Parameters
    ----------
    docs_df: pd.DataFrame
        Annotator documents as returned by the `get_annotator_docs` function
    labels_mapping: Dict[str, str]
        Mapping between annotated labels and displayed labels

    Returns
    -------
    alt.Chart
    """
    logging.info("Plotting micro-averaged IAA")

    scores = []
    docs_df.annotator.drop_duplicates()
    cross_annotations = pd.merge(docs_df, docs_df, on=["note_id", "subsplit"]).query(
        "annotator_x != annotator_y"
    )
    for alignment in ["exact", "token"]:
        metrics = score_examples(
            preds=cross_annotations["doc_x"].tolist(),
            golds=cross_annotations["doc_y"].tolist(),
            labels_mapping=labels_mapping,
            alignment=alignment,
            return_scores_per_doc=True,
        )[0]
        for label, label_metrics in metrics.iterrows():
            for score in ["f1", "tp"]:
                scores.append(
                    {
                        "value": label_metrics[score],
                        "label": label,
                        "alignment": alignment,
                        "metric": score,
                    }
                )

    plot_data = pd.DataFrame(scores)

    selections = []
    for name, init in {
        "alignment": "token",
        "metric": "f1",
    }.items():
        selection_box = alt.binding_select(
            options=list(plot_data[name].unique()), name=name + " : "
        )
        selections.append(
            alt.selection_single(fields=[name], bind=selection_box, init={name: init})
        )

    base_chart = alt.Chart(plot_data).encode(
        x=alt.X("label:N", title="Label"),
        y=alt.Y("value:Q", title="Metric", scale=alt.Scale(zero=False)),
    )

    bar = base_chart.mark_bar().encode()

    alt.data_transformers.disable_max_rows()

    chart = reduce(
        lambda x, s: x.transform_filter(s),
        selections,
        bar.add_selection(*selections),
    )
    return chart.properties(height=400, width=400, title="Inter-annotator agreement")


def make_ml_vs_rb_table(
    results: pd.DataFrame,
    experiments: pd.DataFrame,
):
    logging.info("Building comparison table of ML vs rule-based")

    table = results
    table = table.merge(
        pd.concat(
            [
                experiments.query("limit == limit.max() and filter_expr.isna()"),
                pd.DataFrame([{"rev": "rb", "bert": "rb", "params/system/seed": 0}]),
            ]
        ).reset_index(drop=True),
        on="rev",
    )
    table = table.query("note_class_source_value == 'ALL'")
    table = table.query("alignment == 'token'")
    table = table.query("bert == 'finetuned-raw' or bert == 'rb'")
    table = table.eval("precision = precision * 100")
    table = table.eval("recall = recall * 100")
    table = table.eval("f1 = f1 * 100")
    table = table.eval("full = full * 100")
    table = table.eval("redact = redact * 100")
    table = table.eval("redact_full = redact_full * 100")
    table = table.astype(object).fillna(" ")
    table["Label"] = table["label"]
    table["Model"] = table["mode"].apply(
        lambda x: {
            "merged": "Hybrid",
            "ml": "ML",
            "rb": "RB",
        }[x]
    )
    table = table.groupby(["Label", "Model"])[
        ["precision", "recall", "f1", "redact", "redact_full"]
    ]
    table = table.agg(ValueList)
    table = table.rename(
        {
            "redact_full": "Full redact",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "redact": "Redact",
        },
        axis=1,
    )
    table = table.unstack("Model").reindex(
        columns=[
            (p, m)
            for p in ("Precision", "Recall", "F1", "Redact", "Full redact")
            for m in ("RB", "ML", "Hybrid")
        ]
    )
    table = table.loc[
        [
            "ADDRESS",
            "BIRTHDATE",
            "CITY",
            "DATE",
            "EMAIL",
            "FIRSTNAME",
            "LASTNAME",
            "NSS",
            "PATIENT ID",
            "PHONE",
            "VISIT ID",
            "ZIP",
            "ALL",
        ]
    ]
    table = table.applymap(lambda x: "{:.1f}".format(float(x)))
    table = table.reset_index()
    table.columns = pd.MultiIndex.from_tuples(table.columns)
    styled = (
        table.style.set_table_styles(
            {
                ("Precision", "RB"): [
                    {"props": "border-left-style: solid", "selector": ""}
                ],
                ("Recall", "RB"): [
                    {"props": "border-left-style: solid", "selector": ""}
                ],
                ("F1", "RB"): [{"props": "border-left-style: solid", "selector": ""}],
                ("Redact", "RB"): [
                    {"props": "border-left-style: solid", "selector": ""}
                ],
                ("Full redact", "RB"): [
                    {"props": "border-left-style: solid", "selector": ""}
                ],
            }
        )
        .hide_index()
        .set_table_attributes('class="dataframe"')
    )
    return styled


def make_pdf_comparison_table(
    results: pd.DataFrame,
    experiments: pd.DataFrame,
):
    logging.info("Building comparison table of PDF extraction methods")

    table = results.merge(
        pd.concat(
            [
                experiments.query("limit == limit.max() and filter_expr.isna()"),
            ]
        ).reset_index(drop=True),
        on="rev",
    )
    table = table.query("note_class_source_value == 'ALL'")
    table = table.query("alignment == 'token'")
    table = table.query("mode == 'ml'")
    table = table.query("label == 'ALL'")
    table = table.query("split == 'test/pdfbox' or split == 'test/edspdf'")
    table = table.query("bert == 'finetuned-raw'")
    table = table.eval("precision = precision * 100")
    table = table.eval("recall = recall * 100")
    table = table.eval("f1 = f1 * 100")
    table = table.eval("full = full * 100")
    table = table.eval("redact = redact * 100")
    table = table.eval("redact_full = redact_full * 100")
    table = table.astype(object).fillna(" ")
    table["PDF extractor"] = table["split"].apply(
        lambda x: {
            "test/edspdf": "edspdf",
            "test/pdfbox": "pdfbox",
        }[x]
    )
    table = table.groupby(["PDF extractor"])[
        ["precision", "recall", "f1", "redact", "redact_full"]
    ]
    table = table.agg(ValueList)
    table = table.rename(
        {
            "redact_full": "Full redact",
            "precision": "P",
            "recall": "R",
            "f1": "F1",
            "redact": "Redact",
        },
        axis=1,
    )
    table.loc[["edspdf", "pdfbox"], :]

    table = table.reset_index()
    styled = table.style.hide_index().set_table_attributes('class="dataframe"')
    return styled


def make_bert_ablation_table(results: pd.DataFrame, experiments: pd.DataFrame):
    logging.info("Building comparison table between pretrained embeddings")
    table = results.merge(
        pd.concat(
            [
                experiments.query(
                    f"limit == {experiments.limit.max()} and filter_expr.isna()"
                ),
            ]
        ).reset_index(drop=True),
        on="rev",
    )
    table = table.query("note_class_source_value == 'ALL'")
    table = table.query("alignment == 'exact'")
    table = table.query("mode == 'ml'")
    table = table.query("label == 'ALL'")
    table = table.query(
        "split == 'test/pdfbox' or split == 'test/edspdf' or split == 'test'"
    )
    table = table.eval("precision = precision * 100")
    table = table.eval("recall = recall * 100")
    table = table.eval("f1 = f1 * 100")
    table = table.eval("full = full * 100")
    table = table.eval("redact = redact * 100")
    table = table.eval("redact_full = redact_full * 100")
    table = table.astype(object).fillna(" ")
    # index=["alignment", "bert", "label", "split"],
    # columns=["precision", "recall", "f1", "full", "redact", "redact_full", "rev"],
    table["Transformer"] = table["bert"].apply(
        lambda x: {
            "camembert-base": "camembert base",
            "finetuned-raw": "finetuned raw",
            "scratch-pseudo": "scratch pseudo",
        }[x]
    )
    table["PDF extractor"] = table["split"].apply(
        lambda x: {
            "test/edspdf": "edspdf",
            "test/pdfbox": "pdfbox",
            "test": "all",
        }[x]
    )
    table = table.groupby(["PDF extractor", "Transformer"])[
        ["precision", "recall", "f1", "redact", "redact_full"]
    ]
    table = table.agg(ValueList)
    table = table.rename(
        {
            "redact_full": "Full redact",
            "precision": "P",
            "recall": "R",
            "f1": "F1",
            "redact": "Redact",
        },
        axis=1,
    )
    table = table.loc["all"].loc[
        ("finetuned raw", "camembert base", "scratch pseudo"), :
    ]
    table = table.reset_index()
    styled = table.style.hide_index().set_table_attributes('class="dataframe"')
    return styled


def make_corpus_stats_table(*split_paths, LABEL_MAPPING={}):
    logging.info("Building corpus statistics table")
    table = (
        pd.concat(
            [
                get_corpus_stats(path, labels_mapping=LABEL_MAPPING)
                for path in split_paths
            ],
            axis=1,
        )
        .fillna(0)
        .astype(int)
    )
    table = table.copy()
    table.columns = pd.MultiIndex.from_tuples(
        [tuple(col.split("/")) for col in table.columns]
    )
    styled = table.style.set_table_styles(
        {
            ("train", "edspdf"): [
                {"props": "border-left-style: solid", "selector": ""}
            ],
            ("dev", "edspdf"): [{"props": "border-left-style: solid", "selector": ""}],
            ("test", "edspdf"): [{"props": "border-left-style: solid", "selector": ""}],
        }
    ).set_table_attributes('class="dataframe"')
    return styled


def main(
    repo_path="/export/home/pwajsburt/eds-pseudonymisation",
    metadata_filepath="data/metadata.jsonl",
    xp_output_filepath="corpus/output.spacy",
    gold_filepath="corpus/full.spacy",
    train_filepath="corpus/train.spacy",
    rb_only_filepath="corpus/output-rb-full.spacy",
    rb_to_merge_filepath="corpus/output-rb-best.spacy",
):
    os.chdir(repo_path)

    LABEL_MAPPING = {
        "DATE": "DATE",
        "NOM": "LASTNAME",
        "PRENOM": "FIRSTNAME",
        "MAIL": "EMAIL",
        "NDA": "VISIT ID",
        "TEL": "PHONE",
        "DATE_NAISSANCE": "BIRTHDATE",
        "VILLE": "CITY",
        "ZIP": "ZIP",
        "ADRESSE": "ADDRESS",
        "IPP": "PATIENT ID",
        "SECU": "NSS",
    }

    experiments = load_experiments(
        repo_path,
        {
            "params/corpora/train/limit": "limit",
            "params/paths/bert": "bert",
            "params/corpora/train/filter_expr": "filter_expr",
        },
    ).query("rev != 'workspace'")

    try:
        results = pd.read_parquet("results.pq")
        logging.info("Re-using already computed results table")
    except Exception:
        logging.info("Computing results table, this can take a while...")
        results = pd.concat(
            [
                score_experiments(
                    experiments=experiments,
                    xp_output_filepath=xp_output_filepath,
                    rb_only_filepath=rb_only_filepath,
                    rb_to_merge_filepath=rb_to_merge_filepath,
                    metadata_filepath=metadata_filepath,
                    gold_filepath=gold_filepath,
                    labels_mapping=LABEL_MAPPING,
                ).assign(split=split)
                for split in ("test/edspdf", "test/pdfbox", "test")
            ]
        ).drop(columns=["tp", "pred_count", "gold_count"])
        results.to_parquet("results.pq")

    corpus_stats_table = make_corpus_stats_table(
        train_filepath, gold_filepath, LABEL_MAPPING=LABEL_MAPPING
    )
    Path("docs/assets/figures/corpus_stats_table.html").write_text(
        corpus_stats_table.to_html()
    )

    bert_chart = plot_bert_ablation(
        results.query("note_class_source_value == 'ALL'"),
        experiments.query("filter_expr.isna() and limit == limit.max()"),
    )
    bert_chart.save("docs/assets/figures/bert_ablation.json")

    limit_chart = plot_limit_ablation(
        results.query("note_class_source_value == 'ALL' and label == 'ALL'"),
        experiments.query("filter_expr.isna()"),
    )
    limit_chart.save("docs/assets/figures/limit_ablation.json")

    label_chart = plot_labels(
        results.query("note_class_source_value == 'ALL'"),
        experiments.query(
            f"`limit` == {experiments.limit.max()} and "
            f"bert == 'finetuned-raw' and filter_expr.isna()"
        ),
    )
    label_chart.save("docs/assets/figures/label_scores.json")

    doc_type_chart = plot_doc_type_ablation(
        results.query("label == 'ALL'"),
        experiments.query("limit == limit.max()"),
    )
    doc_type_chart.save("docs/assets/figures/doc_type_ablation.json")

    pdf_comparison_table = make_pdf_comparison_table(results, experiments)
    Path("docs/assets/figures/pdf_comparison_table.html").write_text(
        pdf_comparison_table.to_html()
    )

    ml_vs_rb_table = make_ml_vs_rb_table(results, experiments)
    Path("docs/assets/figures/ml_vs_rb_table.html").write_text(ml_vs_rb_table.to_html())

    bert_ablation_table = make_bert_ablation_table(results, experiments)
    Path("docs/assets/figures/bert_ablation_table.html").write_text(
        bert_ablation_table.to_html()
    )


if __name__ == "__main__":
    main()
