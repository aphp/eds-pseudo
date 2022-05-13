<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: EDS-Pseudonymisation

This project aims at detecting identifying entities at AP-HP's Clinical Data Warehouse:

| Label            | Description                                                   |
| ---------------- | ------------------------------------------------------------- |
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

To run the full pipeline (download, split and format the dataset, train the pipeline and package it), simply run :

```shell
spacy project run all
```

If the pipeline detects that a command has already been run, it skips it unless its inputs have changed.

## 📋 project.yml

The [`project.yml`](project.yml) defines the data assets required by the
project, as well as the available commands and workflows. For details, see the
[spaCy projects documentation](https://spacy.io/usage/projects).

### ⏯ Commands

The following commands are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run).
Commands are only re-run if their inputs have changed.

| Command           | Description                                                                                     |
| ----------------- | ----------------------------------------------------------------------------------------------- |
| `partition`       | Split data between train, dev and test data. The development set is taken from the train split. |
| `convert`         | Convert the data to spaCy's binary format                                                       |
| `create-config`   | Create a new config with an NER pipeline component                                              |
| `train`           | Train the NER model                                                                             |
| `evaluate`        | Evaluate the model and export metrics                                                           |
| `package`         | Package the trained model as a pip package                                                      |
| `visualize-model` | Visualize the model's output interactively using Streamlit                                      |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow       | Steps                                                         |
| -------------- | ------------------------------------------------------------- |
| `all`          | `partition` &rarr; `convert` &rarr; `train` &rarr; `evaluate` |
| `prepare-data` | `partition` &rarr; `convert`                                  |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->

## Documentation

For more information, check out the [documentation](https://equipedatascience-pages.eds.aphp.fr/eds-pseudonymisation/)!
