# Quickstart

## Installation

First, clone the repository

```{: .shell data-md-color-scheme="slate" }
git clone https://github.com/aphp/eds-pseudo.git
cd eds-pseudo
```

And install the dependencies:

```{: .shell data-md-color-scheme="slate" }
poetry install
```

## Without machine learning

If you do not have a labelled dataset, you can still use the rule-based components of the
model.

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
    "En 2015, M. Charles-François-Bienvenu "  # (1)!
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

1. The original date is 1815, but the rule-based date detection only matches dates after
   1900 to avoid false positives.

You can observe that the model is not flawless : "Digne" is not detected as a city. This
can be alleviated by adding contextual information about the patient (see below), or by
[training a model](../training).

## Apply on multiple documents

We recommend you check out the edsnlp's tutorial on [how to process multiple documents](https://aphp.github.io/edsnlp/latest/tutorials/multiple-texts).

Assuming we have a dataframe `df` with columns `note_id`, `text` and an optional column `context`, containing information about the patient, e.g.:

| note_id | text                                                                            | context                            |
|---------|---------------------------------------------------------------------------------|------------------------------------|
| doc-1   | En 2015, M. Charles-François-Bienvenu ...                                       | {"VILLE": "DIGNE", "zip": "04070"} |
| doc-2   | Mme. Josephine-Ange Gardien est admise pour irritation des tendons fléchisseurs |                                    |
| doc-3   | josephine.ange-gardien @ test.com                                               |                                    |

We can apply the model to all the documents with the following code:

```python
import edsnlp


def converter(row):
    tokenizer = edsnlp.data.converters.get_current_tokenizer()
    doc = tokenizer(row["text"])
    doc._.note_id = row["note_id"]
    ctx = row["context"]
    if isinstance(ctx, dict):
        doc._.context = {k: v if isinstance(v, list) else [v] for k, v in ctx.items()}
    return doc


data = edsnlp.data.from_pandas(df, converter=converter)
data = data.map_pipeline(nlp)
data.to_pandas(converter="ents")
```

and we get the following dataframe:

| note_id | start | end | label  | lexical_variant                   |
|:--------|:------|:----|:-------|:----------------------------------|
| doc-1   | 3     | 7   | DATE   | 2015                              |
| doc-1   | 12    | 37  | NOM    | Charles-François-Bienvenu         |
| doc-1   | 38    | 44  | PRENOM | Myriel                            |
| doc-1   | 61    | 66  | VILLE  | Digne                             |
| doc-1   | 145   | 150 | VILLE  | Digne                             |
| doc-1   | 158   | 162 | DATE   | 2006                              |
| doc-2   | 5     | 19  | NOM    | Josephine-Ange                    |
| doc-2   | 20    | 27  | PRENOM | Gardien                           |
| doc-3   | 0     | 33  | MAIL   | josephine.ange-gardien @ test.com |
