# Changelog

## v0.3.0 - 2023-12-01

- Refactoring and fixes to use [edsnlp](https://github.com/aphp/edsnlp) instead of spaCy.
- Renamed `eds_pseudonymisation` to `eds_pseudo` and default model name to `eds_pseudo_aphp`.
- Renamed `pipelines` to `pipes`
- New `scripts/train.py` script to train the model

## ## v0.2.0 - Unreleased

Some fixes to enable training the model:
- committed the missing script `infer.py`
- changed config default bert model to `camembert-base`
- put `config.cfg` as a dependency, not params
- default to cpu training
- allow for missing metadata (i.e. omop's `note_class_source_value`)

## v0.2.0 - 2023-05-04

Many fixes along the publication of our [article](https://arxiv.org/pdf/2303.13451.pdf):

- Tests for the rule-based components
- Code documentation and cleaning
- Experiment and analysis scripts
- Charts and tables in the Results page of our documentation

## v0.1.0 - 2022-05-13

Inception ! :tada:

### Features

- spaCy project for pseudonymisation
- Pseudonymisation-specific pipelines:
    - `pseudonymisation-rules` for rule-based pseudonymisation
    - `pseudonymisation-dates` for date detection and normalisation
    - `structured-data-matcher` for structured data detection (eg first and last name, available in the information system)
- Evaluation methodology
