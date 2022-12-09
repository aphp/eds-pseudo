# Annotation Guide

## Data Selection

We annotated around 4000 documents, selected according to the distribution of AP-HP's
Clinical Data Warehouse (CDW), to obtain a sample that is representative of the actual
documents present within the CDW.

Training data are selected among notes that were edited after August 2017, in order to
skew the model towards more recent clinical notes. The test set, however, is sampled
without any time constraints, to make sure the model performs well overall.

## Annotated Entities

We annotated clinical documents with the following entities :

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

## Software

The software used to annotate the document with personal identification entities was
LabelStudio, but any software will do.

The `convert` step takes as input either a jsonlines file (`.jsonl`) or a folder
containing Standoff files (`.ann`) from an annotation with [Brat](https://brat.nlplab.org/).

Feel free to [submit a pull request](https://github.com/aphp/eds-pseudo/pulls) if these
formats do not suit you!
