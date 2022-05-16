# Quickstart

## Requirements

To use this repository, you will need to supply:

- a labelled dataset
- a HuggingFace transformers model, or use a publicly available model like `camembert-base`

In any case, you will need to modify the configuration to reflect these changes.

## Training the Pipeline

EDS-Pseudonymisation is a [spaCy project](https://spacy.io/usage/projects).
We created a single workflow that:

- partitions the data between train, valid, and test
- converts the datasets to spaCy format
- train the pipeline
- evaluates the pipeline using the test set

At AP-HP, we use Slurm to orchestrate machine-learning experiments.

!!! note "EDS Specificities"

    Because of the way our platform is configured, we need to provide and use a conda environment to train the pipeline using Slurm.

    Said environment uses `poetry.lock` file to ensure reproducibility.

## Deployment

This project merely trains the pseudonymisation pipeline,
and packages it as a pip-installable package
using standard spaCy operations.
