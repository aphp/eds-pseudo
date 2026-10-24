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

## Article experiments

The experiments and figures of our article were made with the `experiments.py` and `analysis.py` scripts of the
[v0.4.0 release](https://github.com/aphp/eds-pseudo/tree/v0.4.0/scripts), which run with DVC and Poetry.
