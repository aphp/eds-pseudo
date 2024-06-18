# Training a custom model

If neither the rule-based model nor the public model are sufficient for your needs, you can
train your own model. This section will guide you through the process.

## Requirements

To train a model, you will need to provide:

- A labelled dataset
- A HuggingFace transformers model, or a publicly available model like `camembert-base`
- Ideally, a GPU to accelerate training

In any case, you will need to modify the
[configs/config.cfg](https://github.com/aphp/eds-pseudo/blob/main/configs/config.cfg) file to
reflect these changes. This configuration already contains the rule-based components of
the previous section, feel free to add or remove them as you see fit. The [configs/config.cfg](https://github.com/aphp/eds-pseudo/blob/main/configs/config.cfg) file also contains
the name of the package model in the `[package]` section (defaults to `eds-pseudo-public`).

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

You should now be able to install and use it:

```{: .shell data-md-color-scheme="slate" }
pip install dist/eds_pseudo_your_eds-0.3.0-*
```

## Use it

To test it, execute

=== "Loading the packaged model"

    ```python
    import eds_pseudo_your_eds

    # Load the model
    nlp = eds_pseudo_your_eds.load()
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
    "En 2015, M. Charles-François-Bienvenu "
    "Myriel était évêque de Digne. C’était un vieillard "
    "d’environ soixante-quinze ans ; il occupait le "
    "siège de Digne depuis le 2 janveir 2006."
)
for e in doc.ents:
    print(f"{e.text: <30}{e.label_: <10}{str(e._.date): <15}{e._.date_format}")

# Text                           Label      Date            Format
# -----------------------------  ---------  --------------  ---------
# 2015                           DATE       2015-??-??      %Y
# Charles-François-Bienvenu      PRENOM     None            None
# Myriel                         NOM        None            None
# Digne                          VILLE      None            None
# Digne                          VILLE      None            None
# 2 janveir 2006                 DATE       2006-01-02      %-d %B %Y
```

You can also add the NER component to an existing model (this is only compatible with edsnlp, not spaCy)

```python
# Given an existing model
existing_nlp = ...

existing_nlp.add_pipe(nlp.get_pipe("ner"), name="ner")
```

To apply the model in parallel on many documents using one or more GPUs, refer to the [Inference](/inference) page.
