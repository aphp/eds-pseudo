# Reproducibility

To guarantee the reproducibility of our models, we rely on virtual environments and VCS
tools.

## Environments

We use [uv](https://docs.astral.sh/uv/) to install the dependencies of our model, with the version constraints of
the `pyproject.toml` file:

```{: .shell data-md-color-scheme="slate" }
uv sync --extra docs
```

## Versioning

We use git to version the configs and templates used to train our models. When it generates a dataset,
`eds-pseudo prepare-data` also writes the SHA256 checksums of its input and output files to
`artifacts/dataset-info.json`.

To (re-)train a model and package it, just run:

```{: .shell data-md-color-scheme="slate" }
eds-pseudo prepare-data --config configs/config.yml
python -m edsnlp.train --config configs/config.yml
eds-pseudo evaluate --config configs/config.yml
eds-pseudo package --config configs/config.yml
```

## Results page experiments

The experiments of the [results page](results.md) train and evaluate on AP-HP notes, which are not public, but their
configs are in `experiments/<name>.yml` so that you can read what we ran. We run them from an AP-HP training
project that checks out eds-pseudo under `eds-pseudo/`, so their paths are relative to the root of that project:

| Path | Content |
| --- | --- |
| `data/dataset/aphp-clean` | AP-HP train, dev and test notes |
| `data/dataset/aphp-short-field` | AP-HP form answers (short fields) |
| `data/dataset/prepared` | Form answer examples, split by source group |
| `data/dataset/ablations` | Filtered views of the above, written by `scripts/make_ablation_data.py` |
| `data/dataset/public` | Synthetic notes written by `eds-pseudo prepare-data --config configs/config.yml` |
| `data/models/camembert-aphp` | CamemBERT model trained on AP-HP notes |

The results page compares these experiments, on seed 42 unless noted:

| Section | Experiments |
| --- | --- |
| General results | `aphp-camembert-aphp` and `public-doctobert` (seeds 42, 13, 123), `rules` (rules only, no training), `public-0.4` (model 0.4 of Hugging Face, no training) |
| Language model | `aphp-camembert-aphp` against `aphp-camembert` and `aphp-doctobert` |
| Training size | `aphp-docs-<N>` (AP-HP documents only) and `aphp-docs-<N>-synthetic` (with the synthetic notes) for N = 10 and 30 (seeds 42, 13, 123), 100, 300, 1000 and 3373 |
| Missing document types | `aphp-without-lt-ordo`, `aphp-without-cr-cons`, `aphp-without-cr-image`, `aphp-without-cr-urge` |
| PDF extraction | `aphp-camembert-aphp`, on the 340 test notes extracted by both pdfbox and edspdf |

A run of an experiment and a seed trains with `python -m edsnlp.train` (except `rules` and `public-0.4`), evaluates
with `eds-pseudo evaluate` and writes its metrics to `artifacts/<experiment>-<seed>`, with the eds-pseudo commit it ran
in `source-state/source-state.json`. Since model 0.4 holds its own copy of `eds_pseudo` 0.4, which can't share a
process with eds-pseudo 1.0, `public-0.4` first writes the predictions of the model with `scripts/predict_package.py`,
then `eds-pseudo evaluate` scores these predictions. From the root of the training project:

```{: .shell data-md-color-scheme="slate" }
python eds-pseudo/scripts/make_ablation_data.py --config eds-pseudo/configs/ablation-data.yml
# train and evaluate every run, then
python eds-pseudo/scripts/make_results_figures.py --runs_dir artifacts
```

`make_results_figures.py` refuses runs whose eds-pseudo commit is not in `origin/main`, had uncommitted changes, or
had another config than the one of their experiment at that commit. It writes the figures and tables of the results
page to `docs/assets/figures`, with the commit of each run in `runs_table.html`.

The experiments and figures of the article were made with the `experiments.py` and `analysis.py` scripts of the
[v0.4.0 release](https://github.com/aphp/eds-pseudo/tree/v0.4.0/scripts), which run with DVC and Poetry.
