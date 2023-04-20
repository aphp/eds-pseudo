![Tests](https://img.shields.io/github/actions/workflow/status/aphp/eds-pseudo/tests.yml?branch=main&label=tests&style=flat-square)
[![Documentation](https://img.shields.io/github/actions/workflow/status/aphp/eds-pseudo/documentation.yml?branch=main&label=docs&style=flat-square)](https://aphp.github.io/eds-pseudo/latest/)
[![Codecov](https://img.shields.io/codecov/c/github/aphp/eds-pseudo?logo=codecov&style=flat-square)](https://codecov.io/gh/aphp/eds-pseudo)
<a href="https://github.com/psf/black" target="_blank">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Black">
</a>
<a href="https://python-poetry.org" target="_blank">
    <img src="https://img.shields.io/badge/reproducibility-poetry-blue" alt="Poetry">
</a>
<a href="https://dvc.org" target="_blank">
    <img src="https://img.shields.io/badge/reproducibility-dvc-blue" alt="DVC">
</a>

# EDS-Pseudonymisation

This project aims at detecting identifying entities at AP-HP's Clinical Data Warehouse:

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

## Publication

Please find our arXiv preprint at the following link: https://arxiv.org/pdf/2303.13451.pdf.

If you use EDS-Pseudo, please cite us as below:

```
@article{tannier2023development,
  title={Development and validation of a natural language processing algorithm to pseudonymize documents in the context of a clinical data warehouse},
  author={Tannier, Xavier and Wajsb{\"u}rt, Perceval and Calliger, Alice and Dura, Basile and Mouchet, Alexandre and Hilka, Martin and Bey, Romain},
  journal={arXiv preprint arXiv:2303.13451},
  year={2023}
}
```

## Documentation

Visit the [documentation](https://aphp.github.io/eds-pseudo/) for more information!

## Acknowledgement

We would like to thank [Assistance Publique – Hôpitaux de Paris](https://www.aphp.fr/)
and [AP-HP Foundation](https://fondationrechercheaphp.fr/) for funding this project.
