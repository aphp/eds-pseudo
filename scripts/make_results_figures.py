"""
Build the figures and tables of the results page (`docs/results.md`) from the `<evaluation>-metrics.json` files that
`eds-pseudo evaluate` wrote in the run folders of the experiments listed in `docs/reproducibility.md`
"""

import json
import math
import subprocess
import warnings
from pathlib import Path
from typing import Any, Dict, List

import altair as alt
import pandas as pd
from confit import Cli

app = Cli(pretty_exceptions_show_locals=False)

# Labels of the metric files and their names in the article
LABELS = {
    "ADRESSE": "ADDRESS",
    "DATE_NAISSANCE": "BIRTHDATE",
    "VILLE": "CITY",
    "DATE": "DATE",
    "MAIL": "EMAIL",
    "PRENOM": "FIRSTNAME",
    "NOM": "LASTNAME",
    "SECU": "NSS",
    "IPP": "PATIENT ID",
    "TEL": "PHONE",
    "NDA": "VISIT ID",
    "ZIP": "ZIP",
}
# Language models compared by the language model section, keyed by experiment
TRANSFORMERS = {
    "aphp-camembert-aphp": "CamemBERT AP-HP",
    "aphp-camembert": "CamemBERT base",
    "aphp-doctobert": "DoctoBERT",
}
# Share of the training notes of each experiment of the training size section
TRAINING_NOTES = {"aphp-notes-10": 10, "aphp-notes-25": 25, "aphp-notes-50": 50, "aphp-camembert-aphp": 100}
NOTE_TYPES = ["lt-ordo", "cr-cons", "cr-image", "cr-urge"]
EXPERIMENTS = {*TRANSFORMERS, *TRAINING_NOTES, "rules", "public-doctobert"} | {f"aphp-without-{t}" for t in NOTE_TYPES}
METRICS = ["Precision", "Recall", "F1", "Redact", "Full redact"]
# Evaluations shown by the charts, all the test notes and the test notes extracted by both pdfbox and edspdf
TEST_EVALUATIONS = ["original-test", "test-edspdf", "test-pdfbox"]
# Evaluations of the short field table, the representative short field test set and its answers unseen in training
SHORT_EVALUATIONS = {"real-short-test": "Short field test", "real-short-answer-unseen": "Unseen answers"}
# Share of the encounters and persons whose short field answers all get redacted, per form source, which the
# `eds_pseudo.population` metric only gives for the short field test set
POPULATION = {
    (source, group): f"{group.capitalize()}s, {name}"
    for source, name in (("urgence", "emergency"), ("maternite", "maternity"))
    for group in ("encounter", "person")
}


def read_commits(run_dirs: List[Path], branch: str) -> pd.DataFrame:
    """
    Check that every run folder ran the config and the code of an eds-pseudo commit of `branch`, so that the results
    page can name the commits its figures come from.

    The training job records the state of the eds-pseudo checkout in `<run>/source-state/source-state.json`. A run
    fails the check when that checkout had uncommitted changes, when its commit is not in `branch` (the branch the
    docs are deployed from), or when the `config.yml` copied in the run folder differs from the experiment config of
    that commit. A run whose commit is older than the last change to `eds_pseudo/` only raises a warning.

    Returns
    -------
    pd.DataFrame
        One row per experiment and commit with its seeds, written to the runs table of the results page
    """
    repo = Path(__file__).resolve().parents[1]
    rows, errors = [], []
    for run_dir in run_dirs:
        experiment, _, seed = run_dir.name.rpartition("-")
        state = json.loads((run_dir / "source-state/source-state.json").read_text())["eds-pseudo"]
        commit = state["commit"]
        pushed = subprocess.run(["git", "-C", repo, "merge-base", "--is-ancestor", commit, branch]).returncode == 0
        same_config = (run_dir / "config.yml").read_bytes() == subprocess.run(
            ["git", "-C", repo, "show", f"{commit}:experiments/{experiment}.yml"], capture_output=True
        ).stdout
        if state["dirty"] or not pushed or not same_config:
            errors.append(
                f"{run_dir.name} at {commit[:7]}: uncommitted changes {state['dirty']}, in {branch} {pushed}, "
                f"same config {same_config}"
            )
        elif subprocess.run(["git", "-C", repo, "diff", "--quiet", commit, "HEAD", "--", "eds_pseudo"]).returncode:
            warnings.warn(f"eds_pseudo/ changed after {commit[:7]}, the commit of {run_dir.name}")
        rows.append({"Experiment": experiment, "eds-pseudo commit": commit[:7], "Seeds": int(seed)})
    if errors:
        raise ValueError(f"These runs did not run a committed experiment config of {branch}:\n" + "\n".join(errors))
    return (
        pd.DataFrame(rows)
        .groupby(["Experiment", "eds-pseudo commit"])
        .Seeds.agg(lambda seeds: ", ".join(map(str, sorted(seeds))))
        .reset_index()
    )


def read_scores(run_dirs: List[Path]) -> pd.DataFrame:
    """
    Read the metric files of the run folders, named `<experiment>-<seed>`

    Returns
    -------
    pd.DataFrame
        One row per experiment, seed, evaluation, model, matching (`exact` or `token` NER matching, the `overlap`
        scores of EDS-NLP 0.24 change between runs of the same code) and label, with the `METRICS` in percent. The
        `Model` is `Hybrid` for the `final` key (whole pipeline), `ML` for `final-ml` (NER model alone), `RB` for the
        rules run and `Public` for the hybrid public model. The `ALL` label sums the counts of every label but
        `HOPITAL`, like the model card, and is the only label with the redaction scores, `Redact` (gold tokens redacted
        whatever their predicted label) and `Full redact` (documents with every gold token redacted)
    """
    rows = []
    for metrics_path in [path for run_dir in run_dirs for path in sorted(run_dir.glob("*-metrics.json"))]:
        experiment, _, seed = metrics_path.parent.name.rpartition("-")
        for key, scores in json.loads(metrics_path.read_text()).items():
            # Since the rules run has no ML model and we only show the hybrid pipeline of the public model, we skip
            # their `final-ml` scores
            if key not in ("final", "final-ml") or (experiment in ("rules", "public-doctobert") and key != "final"):
                continue
            row = {
                "experiment": experiment,
                "seed": int(seed),
                "evaluation": metrics_path.name.removesuffix("-metrics.json"),
                "Model": {"rules": "RB", "public-doctobert": "Public"}.get(experiment)
                or {"final": "Hybrid", "final-ml": "ML"}[key],
            }
            for matching in ("exact", "token"):
                counts = {LABELS[label]: scores[matching][label] for label in LABELS if label in scores[matching]}
                counts["ALL"] = {
                    name: sum(value[name] for value in counts.values()) for name in ("tp", "positives", "support")
                }
                for label, value in counts.items():
                    tp, positives, support = value["tp"], value["positives"], value["support"]
                    rows.append(
                        {
                            **row,
                            "matching": matching,
                            "label": label,
                            "Precision": 100 * tp / positives if positives else math.nan,
                            "Recall": 100 * tp / support if support else math.nan,
                            "F1": 200 * tp / (positives + support) if positives + support else math.nan,
                            "Redact": 100 * scores["redact"]["micro"]["token_r"] if label == "ALL" else math.nan,
                            "Full redact": 100 * scores["redact"]["micro"]["full"] if label == "ALL" else math.nan,
                            # Population shares and the bounds of their 95% interval, e.g. `Encounters, maternity low`
                            **{
                                name: 100 * value if label == "ALL" else math.nan
                                for (source, group), column in POPULATION.items()
                                if "population" in scores
                                for name, value in zip(
                                    (column, f"{column} low", f"{column} high"),
                                    (
                                        scores["population"]["population"]["source"][source]["covered"][group],
                                        *scores["population"]["population"]["ci95"][source]["covered"][group],
                                    ),
                                )
                            },
                        }
                    )
    return pd.DataFrame(rows)


def mean_std(values: pd.Series) -> str:
    """
    Cell of the HTML tables, the mean over seeds and its standard deviation when there are several seeds
    """
    values = values.dropna()
    if values.empty:
        return ""
    return f"{values.mean():.1f} ± {values.std(ddof=0):.1f}" if len(values) > 1 else f"{values.mean():.1f}"


def interactive_chart(chart: alt.Chart, data: pd.DataFrame, selectors: Dict[str, str]) -> tuple:
    """
    Add one drop-down per `selectors` column (with its default value) that chooses the rows of `data` shown by
    `chart`, and return the filtered chart with the drop-down params, which the caller adds to its top level chart
    """
    params = []
    for column, default in selectors.items():
        options = sorted(data[column].dropna().unique().tolist())
        params.append(alt.param(name=column, value=default, bind=alt.binding_select(options=options, name=column)))
        chart = chart.transform_filter(alt.datum[column] == params[-1])
    return chart, params


def bar_chart(
    data: pd.DataFrame, x: str, facet: str, selectors: Dict[str, str], title: str, columns: Any = alt.Undefined
):
    """
    Bars of the mean `value` of each `x`, faceted by `facet`, with the 95% bootstrap interval over seeds. The label,
    language model and note type charts of the results page use it.
    """
    base, params = interactive_chart(alt.Chart(data).encode(x=alt.X(f"{x}:N", title=None)), data, selectors)
    bars = base.mark_bar().encode(y=alt.Y("mean(value):Q", title=None, scale=alt.Scale(zero=False)), color=f"{x}:N")
    errors = base.mark_errorbar(extent="ci").encode(y="value:Q", strokeWidth=alt.value(2))
    return (
        alt.layer(bars, errors)
        .properties(width=150, height=300)
        .facet(alt.Facet(f"{facet}:N", title=None), columns=columns, title=title)
        .add_params(*params)
    )


@app.command(name="make")
def make(
    runs_dir: Path = Path("artifacts"),
    output_dir: Path = Path(__file__).resolve().parents[1] / "docs/assets/figures",
    branch: str = "origin/main",
) -> None:
    """
    Write the figures (Vega-Lite JSON) and tables (HTML) of the results page to `output_dir`, from the run folders
    of `runs_dir`, which must all have run an eds-pseudo commit of `branch` (see `read_commits`).

    1. Check the eds-pseudo commit of every run of the results page and read their scores.
    2. General results: per label scores of the rules, of the ML model and of the hybrid pipeline of
       `aphp-camembert-aphp` (seeds 42, 13 and 123), and of the hybrid public model `public-doctobert`.
    3. Language model: the hybrid pipeline trained from each language model of `TRANSFORMERS`.
    4. PDF extraction: the 340 test notes extracted by both pdfbox and edspdf.
    5. Number of training examples: 10, 25, 50 and 100 percent of the training notes.
    6. Missing note types: training without the notes of one type, scored on the test notes of that type.
    7. Short fields: the models of the general results on the short field test set and on its unseen answers.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    alt.data_transformers.disable_max_rows()

    # 1. Check the runs, read the scores (one row per seed for the tables), then one row per metric for the charts
    run_dirs = [
        run_dir
        for run_dir in sorted(runs_dir.iterdir())
        if run_dir.name.rpartition("-")[0] in EXPERIMENTS and run_dir.name.rpartition("-")[2].isdigit()
    ]
    commits = read_commits(run_dirs, branch)
    (output_dir / "runs_table.html").write_text(commits.to_html(classes="dataframe", border=0, index=False))
    scores = read_scores(run_dirs)
    keys = ["experiment", "seed", "evaluation", "Model", "matching", "label"]
    long = scores.melt(id_vars=keys, value_vars=METRICS, var_name="metric").dropna(subset="value")
    is_all = long.label == "ALL"
    is_test = long.evaluation.isin(TEST_EVALUATIONS)

    # 2. General results, the table uses token matching on all the test notes like the article
    general = scores[scores.experiment.isin(["aphp-camembert-aphp", "rules", "public-doctobert"])]
    table = (
        general[(general.evaluation == "original-test") & (general.matching == "token")]
        .groupby(["label", "Model"])[METRICS]
        .agg(mean_std)
        .unstack("Model")
        .reindex(columns=pd.MultiIndex.from_product([METRICS, ["RB", "ML", "Hybrid", "Public"]]))
        .reindex([*sorted(set(LABELS.values()) & set(general.label)), "ALL"])
        .fillna("")
        .rename_axis("Label")
    )
    (output_dir / "ml_vs_rb_table.html").write_text(table.to_html(classes="dataframe", border=0))
    bar_chart(
        long[is_test & long.experiment.isin(general.experiment.unique()) & long.metric.isin(METRICS[:3])],
        x="Model",
        facet="label",
        selectors={"metric": "F1", "matching": "token", "evaluation": "original-test"},
        title="Scores per label",
        columns=4,
    ).save(output_dir / "label_scores.json")

    # 3. Language model, the table compares the hybrid pipelines while the chart also shows the ML models
    is_hybrid_test = (scores.evaluation == "original-test") & (scores.Model == "Hybrid") & (scores.label == "ALL")
    table = (
        scores[scores.experiment.isin(list(TRANSFORMERS)) & is_hybrid_test & (scores.matching == "exact")]
        .assign(Transformer=lambda x: x.experiment.map(TRANSFORMERS))
        .groupby("Transformer")[METRICS]
        .agg(mean_std)
        .reindex(list(TRANSFORMERS.values()))
    )
    (output_dir / "bert_ablation_table.html").write_text(table.to_html(classes="dataframe", border=0))
    bar_chart(
        long[is_test & long.experiment.isin(list(TRANSFORMERS))].assign(
            Transformer=lambda x: x.experiment.map(TRANSFORMERS)
        ),
        x="Model",
        facet="Transformer",
        selectors={"metric": "F1", "matching": "exact", "evaluation": "original-test", "label": "ALL"},
        title="Language model",
    ).save(output_dir / "bert_ablation.json")

    # 4. PDF extraction, the ML model of aphp-camembert-aphp on the test notes extracted by both tools
    is_pdf_test = scores.evaluation.isin(["test-edspdf", "test-pdfbox"]) & (scores.label == "ALL")
    table = (
        scores[(scores.experiment == "aphp-camembert-aphp") & (scores.Model == "ML") & (scores.matching == "token")]
        .loc[is_pdf_test]
        .assign(**{"PDF extractor": lambda x: x.evaluation.str.removeprefix("test-")})
        .groupby("PDF extractor")[METRICS]
        .agg(mean_std)
    )
    (output_dir / "pdf_comparison_table.html").write_text(table.to_html(classes="dataframe", border=0))

    # 5. Number of training examples, with the rules as a flat line since they don't train
    rules = long[(long.experiment == "rules") & is_all & is_test]
    limit_data = pd.concat(
        [
            long[long.experiment.isin(list(TRAINING_NOTES)) & is_all & is_test].assign(
                notes=lambda x: x.experiment.map(TRAINING_NOTES)
            ),
            rules.assign(notes=10),
            rules.assign(notes=100),
        ]
    )
    base = alt.Chart(limit_data).encode(
        x=alt.X("notes:Q", title="Training notes (%)", scale=alt.Scale(domain=(10, 100))), color="Model:N"
    )
    base, params = interactive_chart(
        base, limit_data, {"metric": "F1", "matching": "token", "evaluation": "original-test"}
    )
    alt.layer(
        base.mark_line(point=True).encode(y=alt.Y("mean(value):Q", title=None, scale=alt.Scale(zero=False))),
        base.mark_errorband(extent="ci").encode(y="value:Q"),
    ).properties(width=400, height=400, title="Number of training examples").add_params(*params).save(
        output_dir / "limit_ablation.json"
    )

    # 6. Missing note types, each type scored by the model trained with and without its notes
    note_type_data = pd.concat(
        [
            long[
                is_all
                & (long.evaluation == f"test-{note_type}")
                & long.experiment.isin(["aphp-camembert-aphp", f"aphp-without-{note_type}"])
            ].assign(
                **{"Note type": note_type.upper()},
                included=lambda x: x.experiment.eq("aphp-camembert-aphp").map({True: "yes", False: "no"}),
            )
            for note_type in NOTE_TYPES
        ]
    )
    bar_chart(
        note_type_data,
        x="included",
        facet="Note type",
        selectors={"metric": "F1", "matching": "token", "Model": "Hybrid"},
        title="Note types missing from training",
    ).save(output_dir / "doc_type_ablation.json")

    # 7. Short fields, one column group per evaluation, and only the short field test set has population scores
    short = scores[
        scores.experiment.isin(["aphp-camembert-aphp", *SINGLE_MODELS])
        & scores.evaluation.isin(list(SHORT_EVALUATIONS))
        & (scores.matching == "token")
        & (scores.label == "ALL")
    ]
    tables = {}
    for evaluation, title in SHORT_EVALUATIONS.items():
        rows = short[short.evaluation == evaluation].groupby("Model")
        tables[title] = rows[["F1", "Full redact"]].agg(mean_std)
        if evaluation == "real-short-test":
            # A population share gets the mean over seeds of its value and of the bounds of its 95% interval, since
            # this interval is much wider than the spread over seeds
            means = rows[[f"{c}{s}" for c in POPULATION.values() for s in ("", " low", " high")]].mean()
            for column in POPULATION.values():
                tables[title][column] = [
                    f"{value:.1f} [{low:.1f}, {high:.1f}]"
                    for value, low, high in means[[column, f"{column} low", f"{column} high"]].to_numpy()
                ]
    table = pd.concat(tables, axis=1).reindex(["RB", "ML", "Hybrid", "Public", "Public 0.4"])
    (output_dir / "short_field_table.html").write_text(table.to_html(classes="dataframe", border=0))


if __name__ == "__main__":
    app()
