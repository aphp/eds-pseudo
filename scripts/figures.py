import os
import re
from collections import Counter, defaultdict
from functools import reduce, partial
from multiprocessing import Pool
from typing import Any, Dict, Iterable, List, Set, Tuple, TypeVar

import altair as alt
import dvc.api
import numpy as np
import pandas as pd
import spacy
import spacy.tokens
import spacy.training
from dvc.repo import Repo
from dvc.repo.experiments.show import show as experiments_show
from edsnlp.utils.filter import filter_spans
from tqdm import tqdm
from itertools import product
from pathlib import Path
from pandas.api.types import is_numeric_dtype

# Evaluation utils

from spacy.tokens import Doc

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
        "f1": len(tp) * 2 / (len(preds) + len(golds)) if len(preds) + len(golds) > 0 else float(len(tp) == 0),
        "tp": len(tp),
        "gold_count": len(golds),
        "pred_count": len(preds),
    }


def get_token_ner_prf(examples, labels_mapping):
    items = {label: {"pred": set(), "gold": set()} for label in ("ALL", *labels_mapping.values())}

    for i, eg in enumerate(examples):
        for ent in eg.predicted.ents:
            if ent.label_ in labels_mapping:
                label = labels_mapping[ent.label_]
                items[label]["pred"].update((i, t) for t in range(ent.start, ent.end) if not eg.reference[t].is_space)
                items["ALL"]["pred"].update((i, t) for t in range(ent.start, ent.end) if not eg.reference[t].is_space)
    for i, eg in enumerate(examples):
        for ent in eg.reference.ents:
            if ent.label_ in labels_mapping:
                label = labels_mapping[ent.label_]
                items[label]["gold"].update((i, t) for t in range(ent.start, ent.end) if not eg.reference[t].is_space)
                items["ALL"]["gold"].update((i, t) for t in range(ent.start, ent.end) if not eg.reference[t].is_space)
    metrics = {
        label: {
            **compute_prf(label_items["pred"], label_items["gold"]),
            "redact": compute_prf(items["ALL"]["pred"], label_items["gold"])["tp"],
        }
        for label, label_items in items.items()
    }

    return metrics


def get_exact_ner_prf(examples, labels_mapping):
    items = {label: {"pred": set(), "gold": set()} for label in ("ALL", *labels_mapping.values())}

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


def get_any_ner_prf(examples, labels_mapping):
    gold_count = {**{label: 0 for label in labels_mapping.values()}, "ALL": 0}
    pred_count = {**{label: 0 for label in labels_mapping.values()}, "ALL": 0}
    tp = {**{label: 0 for label in labels_mapping.values()}, "ALL": 0}
    redact = {**{label: 0 for label in labels_mapping.values()}, "ALL": 0}
    for i, eg in enumerate(examples):
        #for ent in eg.predicted.ents:
        #    if ent.label_ in labels_mapping:
        #        
        #        label = labels_mapping[ent.label_]
        #        items[label]["pred"].add((i, ent.start, ent.end))
        #        items["ALL"]["pred"].add((i, ent.start, ent.end))
        
        for ent in eg.predicted.ents:
            if ent.label_ in labels_mapping:
                label = labels_mapping[ent.label_]
                pred_count[label] += 1
                pred_count["ALL"] += 1
        
        for ent in eg.reference.ents:
            if ent.label_ in labels_mapping:
                label = labels_mapping[ent.label_]
                gold_count[label] += 1
                gold_count["ALL"] += 1
                if any(labels_mapping.get(eg.predicted[i].ent_type_, '') != "" for i in range(ent.start, ent.end)):
                    redact[label] += 1
                    redact["ALL"] += 1
                if any(eg.predicted[i].ent_type_ == ent.label_ for i in range(ent.start, ent.end)):
                    tp[label] += 1
                    tp["ALL"] += 1
    metrics = {
        label: {
            "redact": redact[label],
            "tp": tp[label],
            "gold_count": gold_count[label],
            "pred_count": pred_count[label],
            "recall": (tp[label] / gold_count[label]) if gold_count[label] > 0 else 1.
        }
        for label in labels_mapping.values()
    }

    return metrics


def fix_entities(doc):
    new_ents = []
    doc = doc.copy()
    for ent in doc.ents:
        #if labels_mapping is not None and ent.label_ not in labels_mapping:
        #    continue
        #if labels_mapping is not None:
        #    label = labels_mapping.get(ent.label_, ent.label_)
        #else:
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


def get_scoring_function(name):
    if name == "token":
        return get_token_ner_prf
    if name == "exact":
        return get_exact_ner_prf
    if name == "any":
        return get_any_ner_prf
    raise Exception()

    
def make_examples(rb_examples):
    if isinstance(rb_examples, tuple):
        preds, golds = rb_examples
        # preds = sorted([p.copy() for p in preds], key=lambda x: x._.note_id)
        # golds = sorted([g.copy() for g in golds], key=lambda x: x._.note_id)
        rb_examples = []

        for p, g in zip(preds, golds):
            #p = fix_entities(p)
            #g = fix_entities(g)
            assert p.text == g.text
            eg = spacy.training.Example(p, g)
            rb_examples.append(eg)
    else:
        fixed_rb_examples = []
        for eg in rb_examples:
            p = eg.predicted
            g = eg.reference
            assert eg.predicted.text == eg.reference.text
            eg = spacy.training.Example(p, g)
            fixed_rb_examples.append(eg)

        rb_examples = fixed_rb_examples
        
    return rb_examples

def score_examples(
      rb_examples, labels_mapping, alignment="token", return_scores_per_doc=True
):
    rb_examples = make_examples(rb_examples)

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


def get_corpus_stats(gold_filepath, labels_mapping):
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


def format_snippet(ent, before_n=10, after_n=10, ent_n=30):
    return "{}[{}]{}".format(
        ent.doc.text[ent.start_char - before_n: ent.start_char].rjust(before_n),
        (
              ent.doc.text[ent.start_char: ent.end_char]
              + "({})".format(ent.label_[:5]).rjust(ent_n - (ent.end_char - ent.start_char))
        ),
        ent.doc.text[ent.end_char: ent.end_char + after_n].ljust(after_n),
    ).replace("\n", "⏎")

def show_example(eg):
    if isinstance(eg, spacy.tokens.Doc):
        eg = spacy.training.Example(eg, eg)

    print(eg.predicted._.context)
    preds: Set[Tuple[int, int, str]] = set()
    for ent in eg.predicted.ents:
        preds.add((ent[0].idx, ent[-1].idx + len(ent[-1]), ent.label_))
    golds: Set[Tuple[int, int, str]] = set()
    for ent in eg.reference.ents:
        golds.add((ent[0].idx, ent[-1].idx + len(ent[-1]), ent.label_))

    matches: List[Tuple[Tuple[int, int, str], Tuple[int, int, str], int]] = []
    for p in preds:
        overlap, closest_g = max(
            ((max(0, min(g[1], p[1]) - max(g[0], p[0])), g) for g in golds),
            default=(0, None),
        )
        if overlap > 0:
            golds.remove(closest_g)
            matches.append((p, closest_g, overlap))
        else:
            matches.append((p, None, 0))

    matches.extend((None, g, 0) for g in golds)
    matches = sorted(
        matches, key=lambda pair: pair[0] if pair[0] is not None else pair[1]
    )

    before_n = 10
    after_n = 10
    ent_n = 30
    print(
        " " * before_n
        + "PRED".center(ent_n + 2)
        + " " * after_n
        + "    |    "
        + " " * before_n
        + "GOLD".center(ent_n + 2)
        + " " * after_n
    )
    for p, g, overlap in matches:
        s = ""
        correct = p is not None and g is not None and (p[0], p[1], p[2]) == (g[0], g[1], g[2])
        if p is not None:
            s += "{}[{}]{}".format(
                eg.reference.doc.text[p[0] - before_n: p[0]].rjust(before_n),
                (
                      eg.reference.doc.text[p[0]: p[1]]
                      + "({})".format(p[2][:5]).rjust(ent_n - (p[1] - p[0]))
                ),
                eg.reference.doc.text[p[1]: p[1] + after_n].ljust(after_n),
            ).replace("\n", "⏎")
        else:
            s += " " * (before_n + after_n + ent_n + 2)
        if g is not None:
            s += "    {}    {}[{}]{}".format(
                "|" if correct else "x",
                eg.reference.doc.text[g[0] - before_n: g[0]].rjust(before_n),
                (
                      eg.reference.doc.text[g[0]: g[1]]
                      + "({})".format(g[2][:5]).rjust(ent_n - (g[1] - g[0]))
                ),
                eg.reference.doc.text[g[1]: g[1] + after_n].ljust(after_n),
            ).replace("\n", "⏎")
        if correct:
            print("\x1b[6;30;42m" + s + "\x1b[0m" + str(overlap))
        else:
            # print(p, g)
            print(s + str(overlap))


def merge_docs(docs_a, docs_b):
    for a, b in zip(docs_a, docs_b):
        assert a.text == b.text, (a._.note_id, b._.note_id)
        doc = a.copy()
        doc.ents = filter_spans((*a.ents, *b.ents))
        yield doc


import scipy.stats

class ValueListDiff:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        
    def is_equal_p(self):
        return scipy.stats.ranksums(self.a, self.b)[1]
        
        
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


def postprocess_experiment_column(col):
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


def process_doc_results(xp):
    def pd_prf(group):
        agg = group[["tp", "gold_count", "pred_count", "redact"]].sum()
        agg["f1"] = (agg["tp"] * 2 / (agg["gold_count"] + agg["pred_count"])) if agg["gold_count"] + agg["pred_count"] > 0 else 1.
        agg["precision"] = (agg["tp"] / agg["pred_count"]) if agg["pred_count"] > 0 else 1.
        agg["recall"] = (agg["tp"] / agg["gold_count"]) if agg["gold_count"] > 0 else 1
        agg["full"] = group.groupby("note_id", as_index=False).sum().eval("tp >= gold_count").mean()
        # ((group["tp"].astype(float) / group["gold_count"]) >= 1.0).mean()
        agg["redact"] = (agg["redact"] / agg["gold_count"]) if agg["gold_count"] > 0 else 1
        agg["redact_full"] = group.groupby("note_id", as_index=False).sum().eval("redact >= gold_count").mean()
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
    return x.astype({
        c: 'category' if not is_numeric_dtype(dtype) else dtype
        for c, dtype in x.dtypes.items()
    })

def score_experiments(
      experiments,
      xp_output_filepath,
      metadata_filepath,
      rb_only_filepath,
      rb_to_merge_filepath,
      gold_filepath,
      labels_mapping,
      split="test/edspdf",
):
    nlp = spacy.blank("eds")
    full = list(spacy.tokens.DocBin().from_disk(gold_filepath).get_docs(nlp.vocab))
    for doc in full:
        doc._.split = doc._.note_id.rsplit("/", 1)[0]
        # doc._.note_id = int(doc._.note_id)
    split_df = pd.DataFrame(
        [{"note_id": d._.note_id,
          "omop_note_id": int(d._.note_id.rsplit("/", 1)[1]),
          # "split": d._.note_id.rsplit("/", 1)[0]
          } for d in full]
    )
    metadata = pd.read_json(metadata_filepath, lines=True)[
        [
            "note_id",
            "note_class_source_value",
            "cdm_source",
            "visit_occurrence_id",
            "person_id",
            "note_datetime",
        ]
    ].rename({"note_id": "omop_note_id"}, axis=1).merge(split_df)

    test_ids = set(d._.note_id for d in full if split in d._.split)
    rb_only_preds = sorted([
        fix_entities(d)
        for d in spacy.tokens.DocBin().from_disk(rb_only_filepath).get_docs(nlp.vocab)
        if d._.note_id in test_ids
    ], key=lambda x: x._.note_id)
    rb_to_merge_preds = sorted([
        fix_entities(d)
        for d in spacy.tokens.DocBin().from_disk(rb_to_merge_filepath).get_docs(nlp.vocab)
        if d._.note_id in test_ids
    ], key=lambda x: x._.note_id)

    # Loading and filtering predictions
    all_mode_results = []

    gold = sorted([d for d in full if d._.note_id in test_ids], key=lambda x: x._.note_id)
    assert len(gold) > 0

    for alignment in ["token", "exact", "any"]:
        per_doc_results = score_examples((rb_only_preds, gold), alignment=alignment, labels_mapping=labels_mapping)[1]
        per_doc_results = process_doc_results(
            pd.DataFrame.from_records(per_doc_results)
            .merge(metadata)
        ).assign(alignment=alignment, mode="rb", rev="rb")
        all_mode_results.append(per_doc_results)

    pool = Pool(8)
    [i.close() for i in list(tqdm._instances)]

    def update(*a):
        bar.update()

    bar = tqdm(total=len(experiments["rev"].drop_duplicates()), unit="experiment")
    futures = []

    for rev in experiments["rev"].drop_duplicates():
        fn = partial(score_experiment,
                     xp_output_filepath=xp_output_filepath,
                     nlp=nlp,
                     test_ids=test_ids,
                     rb_preds=rb_to_merge_preds,
                     gold=gold,
                     labels_mapping=labels_mapping,
                     metadata=metadata)
        futures.append(pool.apply_async(fn, args=(rev,), callback=update))
        
    def default(x, default):
        return default if x is None else x

    all_mode_results.extend([df for future in futures for df in default(future.get(), ())])
    
    all_mode_results = pd.concat(all_mode_results)
    return all_mode_results

def score_experiment(rev, xp_output_filepath, nlp, test_ids, rb_preds, gold, labels_mapping, metadata):
    try:
        results = []

        if not isinstance(rev, str) or rev == "workspace":
            db = spacy.tokens.DocBin().from_bytes(
                Path(xp_output_filepath).read_bytes()
            )
        else:
            db = spacy.tokens.DocBin().from_bytes(
                dvc.api.read(path=xp_output_filepath, rev=rev, mode="rb")
            )
        xp_preds = sorted([fix_entities(d) for d in db.get_docs(nlp.vocab) if d._.note_id in test_ids], key=lambda x: x._.note_id)

        # Computing merged outputs
        merged = [fix_entities(d) for d in list(merge_docs(xp_preds, rb_preds))]

        # Scoring outputs
        for alignment in ["exact", "token", "any"]:
            for mode, df in [("ml", xp_preds), ("merged", merged)]:
                per_doc_results = pd.DataFrame.from_records(score_examples((df, gold), alignment=alignment, labels_mapping=labels_mapping)[1])
                per_doc_results = process_doc_results(per_doc_results.merge(metadata, on="note_id"))
                per_doc_results = per_doc_results.assign(alignment=alignment, mode=mode, rev=rev)
                results.append(per_doc_results)

        return results
    except:
        print(f"Could not score experiment: {rev}")
        return None

def load_experiments(repo_path, params_mapping={
    "params/corpora/train/limit": "limit",
    "params/paths/bert": "bert",
}):
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
                        #"metrics": metrics,
                        "commit": commit,
                    },
                    "/",
                )
            )
    experiments = only_changed(
        pd.DataFrame(results, columns=list(dict.fromkeys(col for r in results for col in r.keys())))
        # .query("commit != 'workspace'")
        .apply(postprocess_experiment_column)
        .rename(params_mapping, axis=1),
        include=list(params_mapping.values()),
    ) if len(results) > 1 else pd.DataFrame(results, columns=results[-1].keys())
    experiments["rev"] = experiments["rev"].fillna("workspace")
    experiments = cast_categories(experiments)
    return experiments


def get_annotators_docs(path):
    raw = pd.read_json(path, lines=True)

    if not spacy.tokens.Doc.has_extension("note_id"):
        spacy.tokens.Doc.set_extension("note_id", default=None)
    if not spacy.tokens.Doc.has_extension("split"):
        spacy.tokens.Doc.set_extension("split", default=None)
    if not spacy.tokens.Doc.has_extension("subsplit"):
        spacy.tokens.Doc.set_extension("subsplit", default=None)
    if not spacy.tokens.Doc.has_extension("annotator"):
        spacy.tokens.Doc.set_extension("annotator", default=None)

    nlp = spacy.blank('eds')
    docs = []
    for entry in tqdm(raw.itertuples(index=False)):
        doc = nlp(entry.data['text'])
        doc._.note_id = entry.data["note_id"]
        doc._.split = entry.data["meta_split"]
        doc._.subsplit = entry.data["split"]
        doc._.annotator = entry.annotator[0]
        ents = []
        for group in entry.annotations:
            for res in group["result"]:
                ent = doc.char_span(res["value"]["start"], res["value"]["end"], res["value"]["labels"][0], alignment_mode='expand')
                ents.append(ent)
        doc.ents = filter_spans(ents)
        docs.append(doc)

    docs_df = pd.DataFrame([{
        "annotator": doc._.annotator,
        "note_id": doc._.note_id,
        "subsplit": doc._.subsplit,
        "doc": fix_entities(doc),
    } for doc in docs])

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
        index=["limit", "alignment", "mode", "rev", "split", "label"],
        columns=["f1", "recall", "precision", "full", "redact", "redact_full"],
    ).rename({
       "mode": "Model",
    }, axis=1)
    plot_data["Model"] = plot_data["Model"].apply(lambda x: {
        "merged": "Hybrid",
        "ml": "ML",
        "rb": "Rule-based",
    }[x])

    selections = []
    for name, init in {
        "metric": "f1",
        # "note_class_source_value": "ALL",
        "label": "ALL",
        "alignment": "exact",
        "split": "test/edspdf",
    }.items():
        selection_box = alt.binding_select(
            options=list(plot_data[name].unique()), name=name + " : "
        )
        selections.append(
            alt.selection_single(fields=[name], bind=selection_box, init={name: init})
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
    ).properties(
        height=400,
        width=400,
        title=f"Train size ablation"
    )
    return chart

def select_displayed_data(x, index, columns=()):
    y = x.drop(columns=[c for c in x.columns if c in columns] + ['rev', 'params/system/seed'])
    y = y.drop_duplicates()
    y = y.groupby([c for c in index if c in y]).apply(lambda x: x.apply(lambda v: len(set(v))))
    uncontrolled_columns = list(y.columns[(y > 1).any(0)])
    if len(uncontrolled_columns) > 0:
        raise Exception("Data has an some uncontrolled columns: {}".format(uncontrolled_columns))
    return x.melt(id_vars=index, value_vars=columns, var_name="metric", value_name="value")

def plot_doc_ablation(results, experiments, return_data=False):
    ablation_experiments = experiments.copy()
    ablation_experiments = ablation_experiments[~ablation_experiments.filter_expr.isna()]
    ablation_experiments.filter_expr = ablation_experiments.filter_expr.apply(lambda x: x[34:-1])
    ablations = list(ablation_experiments.filter_expr.drop_duplicates())

    yes_no = {False: 'no', True: 'yes'}
    plot_data = select_displayed_data(
        results.merge(pd.concat([
            ablation_experiments,
            experiments.query(f'bert == "finetuned-raw" and filter_expr.isna() and limit == {experiments.limit.max()}')
        ]).reset_index(drop=True), on="rev")
        .query(f"label == 'ALL'")
        .query(f"note_class_source_value == filter_expr or (filter_expr.isna() and note_class_source_value.isin({ablations}))")
        .rename({"filter_expr": "included"}, axis=1)
        .replace(to_replace={"included": r'^(?!yes).*$'}, value={"included": 'no'}, regex=True)
        .rename({"note_class_source_value": "Document type"}, axis=1)
        .astype(object).fillna("yes"),
        #.eval("model = `bert`.str.cat(mode, ' & ')")
        index=["split", "alignment", "mode", "Document type", "filter_expr"],
        columns=["f1", "recall", "precision", "full", "redact", "redact_full"]
    )
    
    if return_data:
        return plot_data

    selections = []
    for name, init in {
        "mode": "ml",
        "metric": "f1",
        "alignment": "exact",
        "split": "test/edspdf",
    }.items():
        options = list(plot_data[name].unique())
        selection_box = alt.binding_select(
            options=options, name=name + " : "
        )
        selections.append(
            alt.selection_single(fields=[name], bind=selection_box, init={name: plot_data[name].iloc[0] if init not in options else init})
        )

    base_chart = alt.Chart(plot_data).encode(
        x=alt.X("included:N"),
    )

    line = base_chart.mark_bar().encode(
        y=alt.Y("mean(value):Q", scale=alt.Scale(zero=False), stack=None),
        color="included:N",
    )

    band = base_chart.mark_errorbar(extent="ci").encode(
        y=alt.Y("value", scale=alt.Scale(zero=False)),
        strokeWidth=alt.value(2),
    )

    alt.data_transformers.disable_max_rows()

    chart = reduce(
        lambda x, s: x.transform_filter(s),
        selections,
        (line + band).add_selection(*selections),
    ).facet(
        column="Document type",
        title=f"Document type ablation"
    ).properties(
        height=400,
        width=200,
    )
    return chart

def plot_bert_ablation(results, experiments):
    max_limit = experiments["limit"].max()

    plot_data = select_displayed_data(
        results.merge(
            pd.concat([
                experiments,
                pd.DataFrame([
                    {"rev": "rb", "bert": bert}
                    for bert in experiments.bert.drop_duplicates()])
            ])
        ),
        #.eval("model = `bert`.str.cat(mode, ' & ')")
        index=["split", "bert", "alignment", "mode", "rev", "note_class_source_value", "label"],
        columns=["f1", "recall", "precision", "full", "redact", "redact_full"]
    )

    selections = []
    for name, init in {
        "metric": "f1",
        "note_class_source_value": "ALL",
        "label": "ALL",
        "alignment": "exact",
        "split": "test/edspdf",
    }.items():
        options = list(plot_data[name].unique())
        selection_box = alt.binding_select(
            options=options, name=name + " : "
        )
        selections.append(
            alt.selection_single(fields=[name], bind=selection_box, init={name: plot_data[name].iloc[0] if init not in options else init})
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

    chart = reduce(
        lambda x, s: x.transform_filter(s),
        selections,
        (line + band).add_selection(*selections),
    ).properties(
        height=400,
        width=200,
    ).facet(
        column="bert",
        title=f"BERT model ablation"
    )
    return chart


def plot_labels(results, experiments):
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
        index=["split", "bert", "alignment", "mode", "rev", "note_class_source_value", "label"],
        columns=["f1", "recall", "precision", "full", "redact", "redact_full"]
    )

    selections = []
    for name, init in {
        "metric": "f1",
        "note_class_source_value": "ALL",
        "alignment": "exact",
        # "bert": "finetuned-raw",
        "split": "test/edspdf",
    }.items():
        options = list(plot_data[name].unique())
        selection_box = alt.binding_select(
            options=options, name=name + " : "
        )
        selections.append(
            alt.selection_single(fields=[name], bind=selection_box, init={name: plot_data[name].iloc[0] if init not in options else init})
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

    chart = reduce(
        lambda x, s: x.transform_filter(s),
        selections,
        (line + band).add_selection(*selections),
    ).properties(
        height=400,
        width=200,
    ).facet(
        "label",
        columns=4,
        title=f"Label"
    )
    return chart


def main(
      repo_path="/export/home/pwajsburt/eds-pseudonymisation",
      metadata_filepath="data/metadata.jsonl",
      xp_output_filepath="corpus/output.spacy",
      full_filepath="corpus/full.spacy",
      rb_output_filepath="corpus/rb_preds.spacy",
):
    os.chdir(repo_path)

    experiments = load_experiments(repo_path)

    results = score_experiments(
        experiments=experiments,
        xp_output_filepath=xp_output_filepath,
        rb_output_filepath=rb_output_filepath,
        metadata_filepath=metadata_filepath,
        gold_filepath=full_filepath,
        labels_mapping={
            "DATE": "DATE",
            "NOM": "NOM",
            "PRENOM": "PRENOM",
            "MAIL": "MAIL",
            "NDA": "NDA",
            "TEL": "TEL",
            "DATE_NAISSANCE": "DATE_NAISSANCE",
            "VILLE": "VILLE",
            "ZIP": "ZIP",
            "ADRESSE": "ADRESSE",
            "IPP": "IPP",
            "SECU": "SECU",
        },
    )

    bert_chart = plot_bert_ablation(results, experiments)
    bert_chart.save("docs/assets/figures/bert-ablation.json")

    limit_chart = plot_limit_ablation(results, experiments)
    limit_chart.save("docs/assets/figures/limit-ablation.json")


def plot_iaa_pairs(docs_df, labels_mapping):
    scores = []
    annotators = docs_df.annotator.drop_duplicates()
    for ann1, ann2 in tqdm(product(annotators, annotators)):
        if ann1 != ann2:
            cross_annotations = pd.merge(docs_df, docs_df, on=["note_id", "subsplit"]).query(f"annotator_x == '{ann1}' and annotator_y == '{ann2}'")
            for alignment in ["exact", "token"]:
                metrics = score_examples(
                    (cross_annotations['doc_x'].tolist(),
                     cross_annotations['doc_y'].tolist()),
                    labels_mapping=labels_mapping,
                    alignment=alignment,
                    return_scores_per_doc=True,
                )[0]
                for label, label_metrics in metrics.iterrows():
                    scores.append({
                        "ann1": ann1,
                        "ann2": ann2,
                        "value": label_metrics["f1"],
                        "tp": int(label_metrics["tp"]),
                        "tp_str": "TP: {}".format(int(label_metrics["tp"])),
                        "label": label,
                        "alignment": alignment,
                    })

    plot_data = pd.DataFrame(scores)

    selections = []
    for name, init in {
        "label": "ALL",
        "alignment": "exact",
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

    text1 = base_chart.mark_text(color="black").encode(text=alt.Text("value:Q", format="%"))
    text2 = base_chart.mark_text(color="black", dy=10).encode(text=alt.Text("tp_str:N"))

    alt.data_transformers.disable_max_rows()

    chart = reduce(
        lambda x, s: x.transform_filter(s),
        selections,
        (rect + text1 + text2).add_selection(*selections),
    )
    return chart.properties(
        height=400, width=400, title=f"Accord inter-annotateur"
    )


def plot_micro_iaa(docs_df, labels_mapping):
    scores = []
    annotators = docs_df.annotator.drop_duplicates()
    cross_annotations = pd.merge(docs_df, docs_df, on=["note_id", "subsplit"]).query(f"annotator_x != annotator_y")
    for alignment in ["exact", "token"]:
        metrics = score_examples(
            (cross_annotations['doc_x'].tolist(),
             cross_annotations['doc_y'].tolist()),
            labels_mapping=labels_mapping,
            alignment=alignment,
            return_scores_per_doc=True,
        )[0]
        for label, label_metrics in metrics.iterrows():
            for score in ["f1", "tp"]:
                scores.append({
                    "value": label_metrics[score],
                    "label": label,
                    "alignment": alignment,
                    "metric": score,
                })

    plot_data = pd.DataFrame(scores)

    selections = []
    for name, init in {
        "alignment": "exact",
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
    return chart.properties(
        height=400, width=400, title=f"Inter-annotator agreement"
    )


# __all__ = [
#    "main",
#    "plot_limit_ablation",
#    "plot_bert_ablation",
#    "load_experiments",
#    "score_experiments",
#    "get_corpus_stats",
#    "score_examples",
#    "get_annotators_docs",
#    "plot_iaa_pairs",
#    "plot_micro_iaa",
#    'ValueList',
# ]

if __name__ == "__main__":
    main()
