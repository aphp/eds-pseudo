![Tests](https://img.shields.io/github/actions/workflow/status/aphp/eds-pseudo/tests.yml?branch=main&label=tests&style=flat-square)
[![Documentation](https://img.shields.io/github/actions/workflow/status/aphp/eds-pseudo/documentation.yml?branch=main&label=docs&style=flat-square)](https://aphp.github.io/eds-pseudo/latest/)
[![Codecov](https://img.shields.io/codecov/c/github/aphp/eds-pseudo?logo=codecov&style=flat-square)](https://codecov.io/gh/aphp/eds-pseudo)
<a href="https://github.com/psf/black" target="_blank">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Black">
</a>
<a href="https://python-poetry.org" target="_blank">
    <img src="https://img.shields.io/badge/reproducibility-poetry-blue" alt="Poetry">
</a>
<a href="https://dvc.org" target="_blank">
    <img src="https://img.shields.io/badge/reproducibility-dvc-blue" alt="DVC">
</a>

# EDS-Pseudo

This project aims at detecting identifying entities documents, and was primarily tested
on clinical reports at AP-HP's Clinical Data Warehouse (EDS).

The model is built on top of [edsnlp](https://github.com/aphp/edsnlp), and consists in a
hybrid model (rule-based + deep learning) for which we provide rules (`eds_pseudo/pipes`)
and a training recipe (`eds_pseudo/scripts/train.py`).

We also provide a small set of fictive documents (`data/gen_dataset`) to test the method.

The entities that are detected are listed below.

| Label            | Description                                                   |
|------------------|---------------------------------------------------------------|
| `ADRESSE`        | Street address, eg `33 boulevard de Picpus`                   |
| `DATE`           | Any absolute date other than a birthdate                      |
| `DATE_NAISSANCE` | Birthdate                                                     |
| `HOPITAL`        | Hospital name, eg `Hôpital Rothschild`                        |
| `IPP`            | Internal AP-HP identifier for patients, displayed as a number |
| `MAIL`           | Email address                                                 |
| `NDA`            | Internal AP-HP identifier for visits, displayed as a number   |
| `NOM`            | Any last name (patients, doctors, third parties)              |
| `PRENOM`         | Any first name (patients, doctors, etc)                       |
| `SECU`           | Social security number                                        |
| `TEL`            | Any phone number                                              |
| `VILLE`          | Any city                                                      |
| `ZIP`            | Any zip code                                                  |

## Installation

You can install EDS-Pseudo via `pip`. We recommend pinning the library version in your projects, or use a strict package manager like [Poetry](https://python-poetry.org/).

```shell
git clone https://github.com/aphp/eds-pseudo.git
cd eds-pseudo
```

And install the dependencies:

```shell
poetry install
```

## How to use

Put your data in the `data/dataset` folder (or copy `data/gen_dataset` to `data/dataset`).

```shell
python scripts/train.py --config configs/config.cfg --train.seed 43 --cpu
```

This will train a model and save it in `artifacts/model-last`. You can evaluate it on the test set (`data/dataset/test.jsonl`) with:

```shell
python scripts/evaluate.py --config configs/config.cfg
```

To package it, run:

```shell
python scripts/package.py
```

This will create a `dist/eds-pseudo-aphp-0.3.0.***.whl` file that you can install with `pip install dist/eds-pseudo-aphp-0.3.0*`.

You can use it in your code:

```python
import edsnlp

# Either from the model path directly
nlp = edsnlp.load("artifacts/model-last")

# Or from the wheel file
import eds_pseudo_aphp

nlp = eds_pseudo_aphp.load()
```

## Documentation

Visit the [documentation](https://aphp.github.io/eds-pseudo/) for more information!

## Publication

Please find our arXiv preprint at the following link: https://arxiv.org/pdf/2303.13451.pdf.

If you use EDS-Pseudo, please cite us as below:

```
@article{tannier2023development,
  title={Development and validation of a natural language processing algorithm to pseudonymize documents in the context of a clinical data warehouse},
  author={Tannier, Xavier and Wajsb{\"u}rt, Perceval and Calliger, Alice and Dura, Basile and Mouchet, Alexandre and Hilka, Martin and Bey, Romain},
  journal={arXiv preprint arXiv:2303.13451},
  year={2023}
}
```

## Documentation

Visit the [documentation](https://aphp.github.io/eds-pseudo/) for more information!

## Acknowledgement

We would like to thank [Assistance Publique – Hôpitaux de Paris](https://www.aphp.fr/)
and [AP-HP Foundation](https://fondationrechercheaphp.fr/) for funding this project.
