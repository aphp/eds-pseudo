# Quickstart

## Deployment

This project trains our pseudonymisation pipeline, and make it pip-installable.

## Requirements

To use this repository, you will need to supply:

- A labelled dataset
- A HuggingFace transformers model, or use a publicly available model like `camembert-base`

In any case, you will need to modify the configuration to reflect these changes.

## Training a model

EDS-Pseudonymisation is a [spaCy project](https://spacy.io/usage/projects).
We created a single workflow that:

- Converts the datasets to spaCy format
- Trains the pipeline
- Evaluates the pipeline using the test set
- Packages the resulting model to make it pip-installable

To add a new dataset, run

```bash
dvc import-url url/or/path/to/your/dataset data/dataset
```

To (re-)train a model and package it, just run:

```bash
dvc repro
```

You should now be able to install and publish it:

```bash
pip install dist/eds_pseudonymisation-0.2.0-*
```

## Use it

To use it, execute

```python
import eds_pseudonymisation

nlp = eds_pseudonymisation.load()
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
