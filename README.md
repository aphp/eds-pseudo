<!-- modelcard -->
<div>

[<img style="display: inline" src="https://img.shields.io/github/actions/workflow/status/aphp/eds-pseudo/tests.yml?branch=main&label=tests&style=flat-square" alt="Tests">]()
[<img style="display: inline" src="https://img.shields.io/github/actions/workflow/status/aphp/eds-pseudo/documentation.yml?branch=main&label=docs&style=flat-square" alt="Documentation">](https://aphp.github.io/eds-pseudo/latest/)
[<img style="display: inline" src="https://img.shields.io/codecov/c/github/aphp/eds-pseudo?logo=codecov&style=flat-square" alt="Codecov">](https://codecov.io/gh/aphp/eds-pseudo)
[<img style="display: inline" src="https://img.shields.io/badge/demo%20%F0%9F%9A%80-streamlit-purple?style=flat-square" alt="Demo">](https://eds-pseudo-public.streamlit.app/)

</div>

# EDS-Pseudo

The EDS-Pseudo project aims at detecting identifying entities in clinical documents, and was primarily tested
on clinical reports at AP-HP's Clinical Data Warehouse (EDS).

The model is built on top of [edsnlp](https://github.com/aphp/edsnlp), and consists in a
hybrid model (rule-based + deep learning) for which we provide
rules ([`eds-pseudo/pipes`](https://github.com/aphp/eds-pseudo/tree/main/eds_pseudo/pipes))
and a training recipe [`configs/config.yml`](https://github.com/aphp/eds-pseudo/blob/main/configs/config.yml).

We also provide some fictitious
templates ([`templates.jsonl`](https://github.com/aphp/eds-pseudo/blob/main/data/templates.jsonl)) and a command to
generate a synthetic
dataset (`eds-pseudo prepare-data`).

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

The public pretrained model is available on the HuggingFace model hub at
[AP-HP/eds-pseudo-public](https://hf.co/AP-HP/eds-pseudo-public) and was trained on synthetic data
(see [`templates.jsonl`](https://github.com/aphp/eds-pseudo/blob/main/data/templates.jsonl)). You can also
test it directly on the **[demo](https://eds-pseudo-public.streamlit.app/)**.

1. Install the latest version of eds-pseudo

    ```shell
    pip install "eds-pseudo" -U
    ```

2. Get access to the model at [AP-HP/eds-pseudo-public](https://hf.co/AP-HP/eds-pseudo-public)
3. Create and copy a huggingface token https://huggingface.co/settings/tokens?new_token=true
4. Register the token (only once) on your machine

    ```python
    import huggingface_hub

    huggingface_hub.login(token=YOUR_TOKEN, add_to_git_credential=True)
    ```

5. Load the model

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
of [edsnlp](https://aphp.github.io/eds-pseudo/main/inference).

<!-- metrics -->

## Installation to reproduce

If you'd like to reproduce eds-pseudo's training or contribute to its development, you should first clone it:

```shell
git clone https://github.com/aphp/eds-pseudo.git
cd eds-pseudo
```

And install the dependencies. We recommend pinning the library version in your projects, or use a strict package manager
like [uv](https://docs.astral.sh/uv/).

```shell
uv sync
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
[configs/config.yml](https://github.com/aphp/eds-pseudo/blob/main/configs/config.yml) and
[pyproject.toml](https://github.com/aphp/eds-pseudo/blob/main/pyproject.toml) files to
fit your needs.

Put your data in the `data/prepared` folder (or edit the `vars` paths of the `configs/config.yml` file), or generate
the synthetic dataset there with

```shell
eds-pseudo prepare-data --config configs/config.yml
```

Then, run the training command

```shell
python -m edsnlp.train --config configs/config.yml --vars.seed=43
```

This will train a model and save it in `artifacts/model-last`. You can evaluate it on the validation set (defaults
to `data/prepared/public-dev.jsonl`) with:

```shell
eds-pseudo evaluate --config configs/config.yml
```

To package it, run:

```shell
eds-pseudo package --config configs/config.yml
```

This will create a `dist/eds_pseudo_public-***.tar.gz` file that you can install with `pip install dist/eds_pseudo_public-***`.

You can use it in your code:

```python
import edsnlp

# Either from the model path directly
nlp = edsnlp.load("artifacts/model-last")

# Or from the installed package
import eds_pseudo_public

nlp = eds_pseudo_public.load()
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

## Acknowledgement

We would like to thank [Assistance Publique – Hôpitaux de Paris](https://www.aphp.fr/)
and [AP-HP Foundation](https://fondationrechercheaphp.fr/) for funding this project.
