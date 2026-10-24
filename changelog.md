# Changelog

## Unreleased

- Add RPPS pattern to the `pseudonymisation-rules` pipeline
- eds-pseudo can now be installed with pip and provides the `eds-pseudo` command (`prepare-data`, `evaluate`, `stats`, `experiments`, `package` and `infer`), while training now goes through the `python -m edsnlp.train` command of EDS-NLP
- `data/templates.txt` is replaced by the annotated templates of `data/templates.jsonl`, from which `eds-pseudo prepare-data` generates the synthetic dataset
- Models are now packaged with a dependency on `eds-pseudo>=1,<2` instead of a copy of the `eds_pseudo` code
- `eds_pseudo.dates` now builds on the `eds.dates` matcher of EDS-NLP, and still labels birth dates as `DATE_NAISSANCE`
- New `eds_pseudo.redact` metric, which counts the redacted characters and tokens of the gold entities whatever their predicted label, and `eds_pseudo.final`, which runs it with the exact, token and overlap NER metrics of EDS-NLP on the whole pipeline
- Poetry, DVC and the `scripts` folder are removed

## v0.4.0 - 2024-24-05

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
