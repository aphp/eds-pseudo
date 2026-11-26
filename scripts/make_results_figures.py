"""
Build the figures and tables of the results page (`docs/results.md`) from the `<evaluation>-metrics.json` files that
`eds-pseudo evaluate` wrote in the run folders of the experiments listed in `docs/reproducibility.md`
"""

import itertools
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
# Words of the AP-HP training documents of the training size section, keyed by their number of documents, i.e. the
# subsets of `scripts/make_ablation_data.py` and all 3,373 documents, with words split on whitespace
TRAINING_DOCUMENTS = {10: 2318, 30: 7229, 100: 31973, 300: 102530, 1000: 320482, 3373: 1033612}
# Experiments of the training size section, trained on these documents alone or with the synthetic notes of the public
# templates, with their number of documents and training data
TRAINING_SIZES = {
    f"aphp-docs-{n}{suffix}": (n, data)
    for n in TRAINING_DOCUMENTS
    for suffix, data in (("", "AP-HP documents only"), ("-synthetic", "AP-HP documents and synthetic notes"))
}
# Note types of the missing note type section, with their names on the results page
NOTE_TYPES = {
    "lt-ordo": "Prescriptions",
    "cr-cons": "Consultation reports",
    "cr-image": "Imaging reports",
    "cr-urge": "Emergency reports",
}
# Experiments that the tables show as one model, their whole pipeline (`final` key), keyed by experiment
SINGLE_MODELS = {"rules": "RB", "public-doctobert": "Public v2", "public-0.4": "Public 0.4", "presidio": "Presidio"}
MODELS = ["RB", "ML", "Hybrid", "Public v2", "Public 0.4", "Presidio"]
EXPERIMENTS = {*TRANSFORMERS, *TRAINING_SIZES, *SINGLE_MODELS} | {f"aphp-without-{t}" for t in NOTE_TYPES}
METRICS = ["F1", "Redact", "Full redact", "Precision", "Recall"]
# Evaluations of the chart menus with their names on the results page: all the test documents, the test notes
# extracted by both edspdf and pdfbox, and the form answer test set
TEST_EVALUATIONS = {
    "original-test": "All test documents",
    "test-edspdf": "edspdf text",
    "test-pdfbox": "pdfbox text",
    "real-short-test": "Form answers",
}
# NER matchings of the chart menus, an identifier counts once per token or once per entity (same bounds and label)
MATCHINGS = {"token": "token", "exact": "entity"}
# Evaluations of the form answer table, the representative form answer test set and its answers unseen in training
SHORT_EVALUATIONS = {"real-short-test": "Form answer test", "real-short-answer-unseen": "Unseen answers"}
# Share of the encounters and persons whose form answers all get redacted, per form source, which the
# `eds_pseudo.population` metric only gives for the form answer test set, with their headers in the population table
POPULATION = {
    (source, group): (name, f"{group.capitalize()}s")
    for source, name in (("urgence", "Emergency"), ("maternite", "Maternity"))
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
        rules run, `Public v2` for the public model, `Public 0.4` for model 0.4 of Hugging Face and `Presidio`, which
        only has redaction scores since it predicts other labels. Since the
        experiment configs score every label but `HOPITAL`, the `ALL` label sums the counts of the other labels. The
        redaction scores are `Redact` (share of the gold tokens covered by a prediction whatever its label) and `Full
        redact` (share of the documents whose gold tokens are all covered by predictions), per label and for `ALL`,
        and only `ALL` has the `POPULATION` columns on the form answer test set
    """
    rows = []
    for metrics_path in [path for run_dir in run_dirs for path in sorted(run_dir.glob("*-metrics.json"))]:
        experiment, _, seed = metrics_path.parent.name.rpartition("-")
        for key, scores in json.loads(metrics_path.read_text()).items():
            # Since the rules run has no ML model and we only show the whole pipeline of the public models, we skip
            # their `final-ml` scores
            if key not in ("final", "final-ml") or (experiment in SINGLE_MODELS and key != "final"):
                continue
            row = {
                "experiment": experiment,
                "seed": int(seed),
                "evaluation": metrics_path.name.removesuffix("-metrics.json"),
                "Model": SINGLE_MODELS.get(experiment) or {"final": "Hybrid", "final-ml": "ML"}[key],
            }
            # Since Presidio predicts its own labels, it only gets redaction scores on our labels, and the NER
            # scores of the Presidio section on its labels
            ner = experiment != "presidio"
            for matching in ("exact", "token"):
                counts = {LABELS[label]: scores[matching][label] for label in LABELS if label in scores[matching]}
                counts["ALL"] = {
                    name: sum(value[name] for value in counts.values()) for name in ("tp", "positives", "support")
                }
                # Redaction scores per label, a label without gold tokens in this evaluation has none
                redact = {"ALL" if key == "micro" else LABELS[key]: value for key, value in scores["redact"].items()}
                for label, value in counts.items():
                    tp, positives, support = value["tp"], value["positives"], value["support"]
                    rows.append(
                        {
                            **row,
                            "matching": matching,
                            "label": label,
                            "Precision": 100 * tp / positives if ner and positives else math.nan,
                            "Recall": 100 * tp / support if ner and support else math.nan,
                            "F1": 200 * tp / (positives + support) if ner and positives + support else math.nan,
                            "Redact": 100 * redact[label]["token_r"] if label in redact else math.nan,
                            "Full redact": 100 * redact[label]["full"] if label in redact else math.nan,
                            # Population shares and the bounds of their 95% interval, e.g. `Maternity, Encounters low`
                            **{
                                name: 100 * value if label == "ALL" else math.nan
                                for (source, group), header in POPULATION.items()
                                if "population" in scores
                                for name, value in zip(
                                    (", ".join(header), f"{', '.join(header)} low", f"{', '.join(header)} high"),
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


def write_table(table: pd.DataFrame, path: Path) -> None:
    """
    Write `table` as an HTML table of the results page, with its index as first columns and one `<colgroup>` per
    top level column (e.g. all the models of one metric), which `docs/assets/stylesheets/extra.css` separates by a
    vertical line
    """
    table = table.reset_index()
    names = [column[0] if isinstance(column, tuple) else column for column in table.columns]
    groups = [len(list(group)) for _, group in itertools.groupby(names)]
    # Since the index is written as plain columns, every header cell heads a column, and the merged cells of the top
    # level head a group of columns, so we give them a `scope` for screen readers
    html = (
        table.to_html(border=0, index=False, na_rep="", float_format="{:.1f}".format)
        .replace("<th>", '<th scope="col">')
        .replace("<th colspan", '<th scope="colgroup" colspan')
    )
    colgroups = "".join(f'<colgroup span="{span}"></colgroup>' for span in groups)
    path.write_text(html.replace(">", ">" + colgroups, 1))


def interactive_chart(chart: alt.Chart, data: pd.DataFrame, selectors: Dict[str, str]) -> tuple:
    """
    Add one drop-down per `selectors` column (with its default value) that chooses the rows of `data` shown by
    `chart`, and return the filtered chart with the drop-down params, which the caller adds to its top level chart
    """
    params = []
    for column, default in selectors.items():
        options = sorted(data[column].dropna().unique().tolist())
        bind = alt.binding_select(options=options, name=f"{column.lower()} : ")
        params.append(alt.param(name=column.replace(" ", "_"), value=default, bind=bind))
        chart = chart.transform_filter(alt.datum[column] == params[-1])
    return chart, params


def bar_chart(
    data: pd.DataFrame, x: str, facet: str, selectors: Dict[str, str], title: str, columns: Any = alt.Undefined
):
    """
    Bars of the mean `value` of each `x`, faceted by `facet`, with the 95% bootstrap interval over seeds. The label,
    language model and note type charts of the results page use it.

    Since the scores of most models are within a few points of each other, the bars start at 90, or a bit below the
    lowest score shown when it is under 90 (e.g. the recall of the rules on CITY) but never below 0, so that their
    differences stay visible without making a small gap look large. An invisible rule at that value sets the bottom
    of the y scale, and each facet gets its own scale, so that the low scores of the rules on one label don't flatten
    the other labels.
    """
    order = [value for value in [*MODELS, "with", "without"] if value in set(data[x])]
    base, params = interactive_chart(alt.Chart(data).encode(x=alt.X(f"{x}:N", axis=None, sort=order)), data, selectors)
    bars = base.mark_bar().encode(
        y=alt.Y("mean(value):Q", title=None, scale=alt.Scale(zero=False)),
        color=alt.Color(f"{x}:N", sort=order, legend=alt.Legend(title=None)),
    )
    errors = base.mark_errorbar(extent="ci").encode(y="value:Q", strokeWidth=alt.value(2))
    bottom = (
        base.transform_calculate(bottom="max(0, min(90, floor(datum.value) - 1))")
        .mark_rule(opacity=0)
        .encode(y="min(bottom):Q")
    )
    return (
        alt.layer(bars, errors, bottom)
        .properties(width=150, height=300)
        .facet(alt.Facet(f"{facet}:N", title=None), columns=columns, title=title)
        .resolve_scale(y="independent")
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
       `aphp-camembert-aphp` (seeds 42, 13 and 123), and of the public model `public-doctobert` next to model 0.4 of
       Hugging Face `public-0.4`, and the redaction scores of Presidio `presidio`.
    3. Language model: the hybrid pipeline trained from each language model of `TRANSFORMERS`.
    4. PDF extraction: the 340 test notes extracted by both pdfbox and edspdf.
    5. Training size: 10 to 3,373 AP-HP training documents, alone or with the synthetic notes, 3 seeds for 10 and 30
       documents and 1 above.
    6. Missing note types: training without the notes of one type, scored on the test notes of that type.
    7. Form answers: the models of the general results on the form answer test set and on its unseen answers.
    8. Presidio: the models of the general results and Presidio on the labels of Presidio.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    alt.data_transformers.disable_max_rows()

    # 1. Check the runs, read the scores (one row per seed for the tables), then one row per metric for the charts,
    # whose menus show the names of the results page for the evaluations and matchings
    run_dirs = [
        run_dir
        for run_dir in sorted(runs_dir.iterdir())
        if run_dir.name.rpartition("-")[0] in EXPERIMENTS and run_dir.name.rpartition("-")[2].isdigit()
    ]
    write_table(read_commits(run_dirs, branch).set_index("Experiment"), output_dir / "runs_table.html")
    scores = read_scores(run_dirs)
    keys = ["experiment", "seed", "evaluation", "Model", "matching", "label"]
    long = (
        scores[scores.evaluation.isin(list(TEST_EVALUATIONS))]
        .melt(id_vars=keys, value_vars=METRICS, var_name="metric")
        .dropna(subset="value")
        .assign(evaluation=lambda x: x.evaluation.map(TEST_EVALUATIONS), matching=lambda x: x.matching.map(MATCHINGS))
    )
    is_all = long.label == "ALL"
    menus = {"metric": "F1", "matching": "token", "evaluation": TEST_EVALUATIONS["original-test"]}

    # 2. General results, the table gives the mean over seeds with token matching on all the test documents, since
    # the spread over seeds stays within a few tenths of a point, and we drop the empty columns (the NER scores of
    # Presidio on our labels)
    general = scores[scores.experiment.isin(["aphp-camembert-aphp", *SINGLE_MODELS])]
    table = (
        general[(general.evaluation == "original-test") & (general.matching == "token")]
        .groupby(["label", "Model"])[METRICS]
        .mean()
        .unstack("Model")
        .reindex(columns=pd.MultiIndex.from_product([METRICS, MODELS]))
        .dropna(axis="columns", how="all")
        .reindex([*sorted(set(LABELS.values()) & set(general.label)), "ALL"])
        .rename_axis("Label")
    )
    write_table(table, output_dir / "ml_vs_rb_table.html")
    bar_chart(
        long[long.experiment.isin(general.experiment.unique())],
        x="Model",
        facet="label",
        selectors=menus,
        title="Scores per label",
        columns=4,
    ).save(output_dir / "label_scores.json")

    # 3. Language model, the table compares the hybrid pipelines while the chart also shows the ML models
    is_hybrid_test = (scores.evaluation == "original-test") & (scores.Model == "Hybrid") & (scores.label == "ALL")
    table = (
        scores[scores.experiment.isin(list(TRANSFORMERS)) & is_hybrid_test & (scores.matching == "token")]
        .assign(**{"Language model": lambda x: x.experiment.map(TRANSFORMERS)})
        .groupby("Language model")[METRICS]
        .agg(mean_std)
        .reindex(list(TRANSFORMERS.values()))
    )
    write_table(table, output_dir / "bert_ablation_table.html")
    bar_chart(
        long[long.experiment.isin(list(TRANSFORMERS))].assign(
            **{"Language model": lambda x: x.experiment.map(TRANSFORMERS)}
        ),
        x="Model",
        facet="Language model",
        selectors={**menus, "label": "ALL"},
        title="Language model",
    ).save(output_dir / "bert_ablation.json")

    # 4. PDF extraction, the ML model of aphp-camembert-aphp on the test notes extracted by both tools
    is_pdf_test = scores.evaluation.isin(["test-edspdf", "test-pdfbox"]) & (scores.label == "ALL")
    table = (
        scores[(scores.experiment == "aphp-camembert-aphp") & (scores.Model == "ML") & (scores.matching == "token")]
        .loc[is_pdf_test]
        .assign(**{"PDF text": lambda x: x.evaluation.map(TEST_EVALUATIONS)})
        .groupby("PDF text")[METRICS]
        .agg(mean_std)
    )
    write_table(table, output_dir / "pdf_comparison_table.html")

    # 5. Training size, the mean over seeds of the models trained on the AP-HP documents alone or with the synthetic
    # notes, against the words of these documents, with the rules as a flat line since they don't train
    rules = long[(long.experiment == "rules") & is_all].assign(**{"Training data": "Rules, no training"})
    limit_data = pd.concat(
        [
            long[long.experiment.isin(list(TRAINING_SIZES)) & is_all].assign(
                documents=lambda x: x.experiment.map(lambda name: TRAINING_SIZES[name][0]),
                **{"Training data": lambda x: x.experiment.map(lambda name: TRAINING_SIZES[name][1])},
            ),
            rules.assign(documents=min(TRAINING_DOCUMENTS)),
            rules.assign(documents=max(TRAINING_DOCUMENTS)),
        ]
    ).assign(words=lambda x: x.documents.map(TRAINING_DOCUMENTS))
    # The log axis stops a bit past the smallest and largest training sets, otherwise it would round up to 10M words
    words = TRAINING_DOCUMENTS.values()
    base = alt.Chart(limit_data).encode(
        x=alt.X(
            "words:Q",
            title="Words of the annotated AP-HP training documents",
            scale=alt.Scale(type="log", nice=False, domain=[min(words) / 1.5, max(words) * 1.5]),
            axis=alt.Axis(values=[10_000, 100_000, 1_000_000]),
        ),
        y=alt.Y("mean(value):Q", title=None, scale=alt.Scale(zero=False)),
        color=alt.Color("Model:N", sort=MODELS),
    )
    base, params = interactive_chart(base, limit_data, menus)
    # Since a line mark with points shows a point in the legend, we draw the points in their own layer, so that the
    # legend of the training data shows the dash of each line
    alt.layer(
        base.mark_line().encode(
            strokeDash=alt.StrokeDash("Training data:N", legend=alt.Legend(symbolType="stroke", labelLimit=0))
        ),
        base.mark_point(filled=True, opacity=1).encode(
            tooltip=["Model:N", "Training data:N", "documents:Q", "words:Q", alt.Tooltip("mean(value):Q", format=".1f")]
        ),
    ).properties(width=400, height=400, title="Training size").add_params(*params).save(
        output_dir / "limit_ablation.json"
    )

    # 6. Missing note types, each type scored by the model trained with and without its notes
    note_type_data = pd.concat(
        [
            scores[
                (scores.label == "ALL")
                & (scores.evaluation == f"test-{note_type}")
                & scores.experiment.isin(["aphp-camembert-aphp", f"aphp-without-{note_type}"])
            ]
            .melt(id_vars=keys, value_vars=METRICS, var_name="metric")
            .dropna(subset="value")
            .assign(
                **{
                    "Note type": name,
                    "Training": lambda x: x.experiment.eq("aphp-camembert-aphp").map({True: "with", False: "without"}),
                },
                matching=lambda x: x.matching.map(MATCHINGS),
            )
            for note_type, name in NOTE_TYPES.items()
        ]
    )
    bar_chart(
        note_type_data,
        x="Training",
        facet="Note type",
        selectors={"metric": "F1", "matching": "token", "Model": "Hybrid"},
        title="Note types missing from training",
    ).save(output_dir / "doc_type_ablation.json")

    # 7. Form answers, the scores of the form answer test set and of its unseen answers, then the encounter and person
    # shares of the form answer test set with the mean over seeds of their value and of the bounds of their 95%
    # interval, since this interval is much wider than the spread over seeds
    short = scores[
        scores.experiment.isin(["aphp-camembert-aphp", *SINGLE_MODELS])
        & scores.evaluation.isin(list(SHORT_EVALUATIONS))
        & (scores.matching == "token")
        & (scores.label == "ALL")
    ]
    table = pd.concat(
        {
            title: short[short.evaluation == evaluation].groupby("Model")[["F1", "Full redact"]].agg(mean_std)
            for evaluation, title in SHORT_EVALUATIONS.items()
        },
        axis=1,
    )
    write_table(table.reindex(MODELS), output_dir / "short_field_table.html")
    columns = [", ".join(header) for header in POPULATION.values()]
    means = (
        short[short.evaluation == "real-short-test"]
        .groupby("Model")[[f"{column}{bound}" for column in columns for bound in ("", " low", " high")]]
        .mean()
    )
    table = pd.DataFrame(
        {
            header: [
                f"{value:.1f} [{low:.1f}, {high:.1f}]"
                for value, low, high in means[[column, f"{column} low", f"{column} high"]].to_numpy()
            ]
            for header, column in zip(POPULATION.values(), columns)
        },
        index=means.index,
    )
    write_table(table.reindex(MODELS), output_dir / "short_field_population_table.html")

    # 8. Presidio, the token scores on all the test documents of the `coarse` keys, where `eds_pseudo.coarse` merged
    # our labels into the labels of Presidio, with the mean over seeds
    rows = []
    for run_dir in run_dirs:
        experiment = run_dir.name.rpartition("-")[0]
        if experiment not in ["aphp-camembert-aphp", *SINGLE_MODELS]:
            continue
        for key, metrics in json.loads((run_dir / "original-test-metrics.json").read_text()).items():
            if key not in ("coarse", "coarse-ml") or (experiment in SINGLE_MODELS and key != "coarse"):
                continue
            for label, value in metrics["token"].items():
                rows.append(
                    {
                        "Model": SINGLE_MODELS.get(experiment) or {"coarse": "Hybrid", "coarse-ml": "ML"}[key],
                        "Label": "ALL" if label == "micro" else label,
                        "F1": 100 * value["f"],
                        "Precision": 100 * value["p"],
                        "Recall": 100 * value["r"],
                    }
                )
    table = (
        pd.DataFrame(rows)
        .groupby(["Label", "Model"])[["F1", "Precision", "Recall"]]
        .mean()
        .unstack("Model")
        .reindex(columns=pd.MultiIndex.from_product([["F1", "Precision", "Recall"], MODELS]))
    )
    write_table(table.reindex([*sorted(set(table.index) - {"ALL"}), "ALL"]), output_dir / "presidio_table.html")


if __name__ == "__main__":
    app()
