# Quickstart

## Deployment

This project trains our pseudonymisation pipeline, and make it pip-installable.

## Requirements

To use this repository, you will need to supply:

- A labelled dataset
- A HuggingFace transformers model, or use a publicly available model like `camembert-base`

In any case, you will need to modify the configuration to reflect these changes.

## Installation

Install the requirements by running the following command at the root of the repo

<div class="termy">

```bash
poetry install
```

</div>

## Training a model

EDS-Pseudonymisation is a [spaCy project](https://spacy.io/usage/projects).
We created a single workflow that:

- Converts the datasets to spaCy format
- Trains the pipeline
- Evaluates the pipeline using the test set
- Packages the resulting model to make it pip-installable

To add a new dataset, run

<div class="termy">

```bash
dvc import-url url/or/path/to/your/dataset data/dataset
```

</div>

To (re-)train a model and package it, just run:

<div class="termy">

```bash
dvc repro
```

</div>

You should now be able to install and publish it:

```bash
pip install dist/eds_pseudonymisation-0.2.0-*
```

## Use it

To use it, execute

```python
import eds_pseudonymisation

# Load the machine learning model
nlp = eds_pseudonymisation.load()

# Add optional rule-based components
nlp.add_pipe("eds.remove-lowercase", name="remove-lowercase")
nlp.add_pipe("eds.accents", name="accents")
nlp.add_pipe(
    "pseudonymisation-rules",
    name="pseudonymisation-rules",
    config={"pattern_keys": ["TEL", "MAIL", "SECU"]},
)
nlp.add_pipe("pseudonymisation-addresses", name="pseudonymisation-addresses")
nlp.add_pipe("structured-data-matcher", name="structured-data-matcher")

# Apply it to a text
doc = nlp(
    """En 1815, M. Charles-François-Bienvenu
Myriel était évêque de Digne. C’était un vieillard
d’environ soixante-quinze ans ; il occupait le
siège de Digne depuis 1806. """
)
for ent in doc.ents:
    print(ent, ent.label)

# 1815 DATE
# Charles-François-Bienvenu NOM
# Myriel PRENOM
# Digne VILLE
# 1806 DATE
```
