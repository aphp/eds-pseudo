<div>
<a target="_blank">
    <img style="display: inline" src="https://img.shields.io/github/actions/workflow/status/aphp/eds-pseudo/tests.yml?branch=main&label=tests&style=flat-square" alt="Tests">
</a>
<a href="https://aphp.github.io/eds-pseudo/latest/" target="_blank">
    <img style="display: inline" src="https://img.shields.io/github/actions/workflow/status/aphp/eds-pseudo/documentation.yml?branch=main&label=docs&style=flat-square" alt="Documentation">
</a>
<a href="https://codecov.io/gh/aphp/eds-pseudo" target="_blank">
    <img style="display: inline" src="https://img.shields.io/codecov/c/github/aphp/eds-pseudo?logo=codecov&style=flat-square" alt="Codecov">
</a>
<a href="https://github.com/psf/black" target="_blank">
    <img style="display: inline" src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Black">
</a>
<a href="https://python-poetry.org" target="_blank">
    <img style="display: inline" src="https://img.shields.io/badge/repro-poetry-blue" alt="Poetry">
</a>
<a href="https://dvc.org" target="_blank">
    <img style="display: inline" src="https://img.shields.io/badge/repro-dvc-blue" alt="DVC">
</a>
</div>

# EDS-Pseudo

This project aims at detecting identifying entities documents, and was primarily tested
on clinical reports at AP-HP's Clinical Data Warehouse (EDS).

The model is built on top of [edsnlp](https://github.com/aphp/edsnlp), and consists in a
hybrid model (rule-based + deep learning) for which we provide rules ([`eds-pseudo/pipes`](https://github.com/aphp/eds-pseudo/tree/main/eds_pseudo/pipes))
and a training recipe [`train.py`](https://github.com/aphp/eds-pseudo/blob/main/scripts/train.py).

We also provide some fictitious templates ([`templates.txt`](https://github.com/aphp/eds-pseudo/blob/main/data/templates.txt)) and a script to
generate a synthetic dataset [`generate_dataset.py`](https://github.com/aphp/eds-pseudo/blob/main/scripts/generate_dataset.py).

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

## Downloading the public pre-trained model

The public pretrained model is available on the HuggingFace model hub, and was trained
on synthetic data (see [`generate_dataset.py`](https://github.com/aphp/eds-pseudo/blob/main/scripts/generate_dataset.py)). You can also test
it directly on its huggingface hub page:
[AP-HP/eds-pseudo-public](https://huggingface.co/AP-HP/eds-pseudo-public).

```shell
pip install "edsnlp[ml]"
```

```python
import edsnlp

nlp = edsnlp.load("AP-HP/eds-pseudo-public", auto_update=True)
doc = nlp(
    "En 2015, M. Charles-François-Bienvenu "
    "Myriel était évêque de Digne. C’était un vieillard "
    "d’environ soixante-quinze ans ; il occupait le "
    "siège de Digne depuis 2006."
)

for ent in doc.ents:
    print(ent, ent.label_, str(ent._.date))
```

To apply the model on many documents using one or more GPUs, refer to the documentation
of [edsnlp](https://aphp.github.io/edsnlp/latest/tutorials/multiple-texts/).

## Installation

Clone eds-pseudo:

```shell
git clone https://github.com/aphp/eds-pseudo.git
cd eds-pseudo
```

And install the dependencies. We recommend pinning the library version in your projects, or use a strict package manager like [Poetry](https://python-poetry.org/).

```shell
poetry install
```

## How to use without machine learning

```python
import edsnlp

nlp = edsnlp.blank("eds")

# Some text cleaning
nlp.add_pipe("eds.normalizer")

# Various simple rules
nlp.add_pipe(
    "eds_pseudo.simple_rules",
    config={"pattern_keys": ["TEL", "MAIL", "SECU", "PERSON"]},
)

# Address detection
nlp.add_pipe("eds_pseudo.addresses")

# Date detection
nlp.add_pipe("eds_pseudo.dates")

# Contextual rules (requires a dict of info about the patient)
nlp.add_pipe("eds_pseudo.context")

# Apply it to a text
doc = nlp(
    "En 2015, M. Charles-François-Bienvenu "
    "Myriel était évêque de Digne. C’était un vieillard "
    "d’environ soixante-quinze ans ; il occupait le "
    "siège de Digne depuis 2006."
)

for ent in doc.ents:
    print(ent, ent.label_)

# 2015 DATE
# Charles-François-Bienvenu NOM
# Myriel PRENOM
# 2006 DATE
```

## How to train

Before training a model, you should update the
[configs/config.cfg](https://github.com/aphp/eds-pseudo/blob/main/configs/config.cfg) and
[pyproject.toml](https://github.com/aphp/eds-pseudo/blob/main/pyproject.toml) files to
fit your needs.

Put your data in the `data/dataset` folder (or edit the paths `configs/config.cfg` file to point to `data/gen_dataset/train.jsonl`).

Then, run the training script

```shell
python scripts/train.py --config configs/config.cfg --seed 43
```

This will train a model and save it in `artifacts/model-last`. You can evaluate it on the test set (defaults to `data/dataset/test.jsonl`) with:

```shell
python scripts/evaluate.py --config configs/config.cfg
```

To package it, run:

```shell
python scripts/package.py
```

This will create a `dist/eds-pseudo-aphp-***.whl` file that you can install with `pip install dist/eds-pseudo-aphp-***`.

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

Please find our publication at the following link: https://doi.org/mkfv.

If you use EDS-Pseudo, please cite us as below:

```
@article{eds_pseudo,
  title={Development and validation of a natural language processing algorithm to pseudonymize documents in the context of a clinical data warehouse},
  author={Tannier, Xavier and Wajsb{\"u}rt, Perceval and Calliger, Alice and Dura, Basile and Mouchet, Alexandre and Hilka, Martin and Bey, Romain},
  journal={Methods of Information in Medicine},
  year={2024},
  publisher={Georg Thieme Verlag KG}
}
```

## Documentation

Visit the [documentation](https://aphp.github.io/eds-pseudo/) for more information!

## Acknowledgement

We would like to thank [Assistance Publique – Hôpitaux de Paris](https://www.aphp.fr/)
and [AP-HP Foundation](https://fondationrechercheaphp.fr/) for funding this project.
