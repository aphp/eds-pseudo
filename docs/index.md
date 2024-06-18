# Overview

The EDS-Pseudo project aims at detecting identifying entities in clinical documents, and was primarily tested
on clinical reports at AP-HP's Clinical Data Warehouse (EDS).

The model is built on top of [edsnlp](https://github.com/aphp/edsnlp), and consists in a
hybrid model (rule-based + deep learning) for which we provide
rules ([`eds-pseudo/pipes`](https://github.com/aphp/eds-pseudo/tree/main/eds_pseudo/pipes))
and a training recipe [`train.py`](https://github.com/aphp/eds-pseudo/blob/main/scripts/train.py).

We also provide some fictitious
templates ([`templates.txt`](https://github.com/aphp/eds-pseudo/blob/main/data/templates.txt)) and a script to
generate a synthetic
dataset [`generate_dataset.py`](https://github.com/aphp/eds-pseudo/blob/main/scripts/generate_dataset.py).

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
