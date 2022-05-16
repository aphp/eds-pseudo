# Overview

EDS-Pseudonymisation is a spaCy project dedicated to the creation of a Named Entity Recognition (NER)
pipeline to extract identifying entities.

## Getting started

EDS-Pseudonymisation is a [spaCy project](https://spacy.io/usage/projects).
We created a single workflow that:

- Partitions the data between train, valid, and test
- Converts the datasets to spaCy format
- Trains the pipeline
- Evaluates the pipeline using the test set

To use it, you will need to supply:

- A labelled dataset
- A HuggingFace transformers model, or use `camembert-base`

In any case, you will need to modify the configuration to reflect these changes.

## Entities

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

Take a look at the [annotation guide](annotation-guide.md) for more detail.
