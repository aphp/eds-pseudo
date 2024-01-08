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

If you face issues with the installation, try to lower the maximum python version to
<= 3.10 (in `pyproject.toml`).

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

# Date value and format detector
# This is useful to reinsert a new shifted date with the same format in the text
nlp.add_pipe(
    "eds_pseudo.dates_normalizer",
    config={"format": "java"}
    # java format -> will output a format like "yyyy/MM/dd"
    # strftime format -> will output a format like "%Y/%m/%d"
)

# Apply it to a text
doc = nlp(
    "En 2015, M. Charles-François-Bienvenu "
    "Myriel était évêque de Digne. C’était un vieillard "
    "d’environ soixante-quinze ans ; il occupait le "
    "siège de Digne depuis le 2 janvier 2006."
)
for e in doc.ents:
    print(f"{e.text: <30}{e.label_: <10}{str(e._.date): <15}{e._.date_format}")

# Text                          Label     Date           Format
# ----------------------------  --------  -------------  ---------
# 2015                          DATE      2015-??-??     yyyy
# Charles-François-Bienvenu     NOM       None           None
# Myriel                        PRENOM    None           None
# 2 janvier 2006                DATE      2006-01-02     d MMMM yyyy
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
| doc-2   | Mme. Ange-Gardien Josephine est admise pour irritation des tendons fléchisseurs |                                    |
| doc-3   | josephine.ange-gardien @ test.com                                               |                                    |

We can apply the model to all the documents with the following code:

```python
import edsnlp


# Function to convert a row of the dataframe to a Doc object
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
data.to_pandas(converter="ents", span_attributes=["date", "date_format"])
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
| doc-2   | 5     | 17  | NOM    | Ange-Gardien                      |
| doc-2   | 18    | 27  | PRENOM | Joséphine                         |
| doc-3   | 0     | 33  | MAIL   | josephine.ange-gardien @ test.com |
