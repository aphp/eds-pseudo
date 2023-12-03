# Dataset

!!! warning "Disclaimer"

    We do not provide the dataset due to privacy and regulatory constraints. You will
    however find the description of the dataset below. We also release the code for the
    rule-based annotation system.

    You can find fictive data in the
    [`data/gen_dataset`](https://github.com/aphp/eds-pseudo/tree/main/data/gen_dataset/)
    folder to test the model.

## Format

By default, we expect the annotations to follow the format of the demo dataset [data/gen_dataset](https://github.com/aphp/eds-pseudo/tree/main/data/gen_dataset), but you can change the format by modifying the [config file](https://github.com/aphp/eds-pseudo/blob/main/configs/config.cfg), and the "Datasets" part of it in particular, or the code of the [adapter](https://github.com/aphp/eds-pseudo/blob/main/eds_pseudo/adapter.py).

## Data Selection

We annotated around 4000 documents, selected according to the distribution of AP-HP's
Clinical Data Warehouse (CDW), to obtain a sample that is representative of the actual
documents present within the CDW.

Training data are selected among notes that were edited after August 2017, in order to
skew the model towards more recent clinical notes. The test set, however, is sampled
without any time constraints, to make sure the model performs well overall.

To ensure the robustness of the model, training and test sets documents were
generated from two different PDF extraction methods:

- the legacy method, based on [PDFBox](https://pdfbox.apache.org/) with a fixed mask
- our new method [EDS-PDF](https://github.com/aphp/edspdf) with an adaptative
  (machine-learned) mask

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

## Statistics

To inspect the statistics for the latest version of our dataset, please refer to the
[latest release](/eds-pseudo/latest/dataset#statistics).

<!--

## Statistics

The following table presents the counts of annotated entities per split and per label.

--8<-- "docs/assets/figures/corpus_stats_table.html"

-->

## Software

The software tools used to annotate the documents with personal identification entities were:

- [LabelStudio](https://labelstud.io/) for the first annotation campaign
- [Metanno](https://github.com/percevalw/metanno) for the second annotation campaign
but any annotation software will do.
