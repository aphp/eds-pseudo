# Changelog

## Pending

- Added `eds_pseudo.dates_normalizer` to parse ML detected dates and extract their value and format.
- Support empty `doc._.context` field
- Update EDS-NLP to v0.10.7:
  - fix somes issues with jsonl loading
  - more transformer overriding options
  - fix out-of-memory issues (auto split transformer input depending on the available memory)
  - fixes some multiprocessing deadlock issues
  - add chunk sorting option to the lazy collection `set_processing` method
- Replace `gen_dataset/train.jsonl` with the original fictitious templates and the dataset generation script.
- Update the README with the instructions to download the public pre-trained model.
- Improve packaging to add evaluation results to the model's meta field and packaged model README (for HF)

## v0.3.0 - 2023-12-01

- Refactoring and fixes to use [edsnlp](https://github.com/aphp/edsnlp) instead of spaCy.
- Renamed `eds_pseudonymisation` to `eds_pseudo` and default model name to `eds_pseudo_aphp`.
- Renamed `pipelines` to `pipes`
- New `scripts/train.py` script to train the model

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
