# Training

## Requirements

To train a model, you will need to provide:

- A labelled dataset
- A HuggingFace transformers model, or a publicly available model like `camembert-base`
- Ideally, a GPU to accelerate training

In any case, you will need to modify the
[configs/config.cfg](https://github.com/aphp/eds-pseudo/blob/main/configs/config.cfg) file to
reflect these changes. This configuration already contains the rule-based components of
the previous section, feel free to add or remove them as you see fit. You may also want
to modify the [pyproject.toml](https://github.com/aphp/eds-pseudo/blob/main/pyproject.toml) file to change the name of packaged model
(defaults to `eds-pseudo-aphp`).

## DVC

We use [DVC](https://dvc.org/) to manage the training pipeline. DVC is a version control
system for data science and machine learning projects. We recommend you use it too.
First, import some data (this basically copies the data to `data/dataset`, but in a
version-controlled fashion):

```{: .shell data-md-color-scheme="slate" }
dvc import-url url/or/path/to/your/dataset data/dataset
```

and execute the following command to (re)train the model and package it

```{: .shell data-md-color-scheme="slate" }
dvc repro
```

??? note "Content of the `dvc.yaml` file"

    The above command runs the
    [`dvc.yaml`](https://github.com/aphp/eds-pseudo/blob/main/dvc.yaml) config file to
    sequentially execute :

    ```{: .shell data-md-color-scheme="slate" }
    # Trains the model, and outputs it to artifacts/model-last
    python scripts/train.py --config configs/config.cfg

    # Evaluates the model, and outputs the results to artifacts
    python scripts/evaluate.py --config configs/config.cfg

    # Packages the model
    python scripts/package.py
    ```

You should now be able to install and publish it:

```{: .shell data-md-color-scheme="slate" }
pip install dist/eds_pseudo_aphp-0.3.0-*
```

## Use it

To test it, execute

=== "Loading the packaged model"

    ```python
    import eds_pseudo_aphp

    # Load the model
    nlp = eds_pseudo_aphp.load()
    ```

=== "Loading from the folder"

    ```python
    import edsnlp

    # Load the model
    nlp = edsnlp.load("artifacts/model-last")
    ```

```python
# Apply it to a text
doc = nlp(
    "En 1815, M. Charles-François-Bienvenu "
    "Myriel était évêque de Digne. C’était un vieillard "
    "d’environ soixante-quinze ans ; il occupait le "
    "siège de Digne depuis 1806."
)
for ent in doc.ents:
    print(ent, ent.label_)

# 1815 DATE
# Charles-François-Bienvenu NOM
# Myriel PRENOM
# Digne VILLE
# 1806 DATE
```

You can also add the NER component to an existing model (this is only compatible with edsnlp, not spaCy)

```python
# Given an existing model
existing_nlp = ...

existing_nlp.add_pipe(nlp.get_pipe("ner"))
```
