# Results

Please find our publication at the following link: https://doi.org/mkfv.

If you use EDS-Pseudo, please cite us as below:

```
@article{eds_pseudo,
  title={Development and validation of a natural language processing algorithm to pseudonymize documents in the context of a clinical data warehouse},
  author={Tannier, Xavier and Wajsb{\"u}rt, Perceval and Calliger, Alice and Dura, Basile and Mouchet, Alexandre and Hilka, Martin and Bey, Romain},
  journal={Methods of Information in Medicine},
  year={2024},
  publisher={Georg Thieme Verlag KG}
}
```

This page scores the models of this version on AP-HP clinical notes. We rerun every experiment at each version, and
the [runs](#runs) section gives the commit and the seeds of each one (see [how we run them](reproducibility.md#results-page-experiments)).

The models are:

- **RB**, the rules alone
- **ML**, the NER model alone, trained on 3,373 AP-HP documents, 3,207 AP-HP [form answers](#form-answers),
  5,215 synthetic answers generated from them and 9,233 synthetic notes generated from the public templates
- **Hybrid**, the ML model merged with the rules, which is the model we use at AP-HP
- **Public v2**, [eds-pseudo-public-v2](https://hf.co/AP-HP/eds-pseudo-public-v2), the same hybrid pipeline trained
  on synthetic notes only, which we publish
- **Public 0.4**, [eds-pseudo-public](https://hf.co/AP-HP/eds-pseudo-public) 0.4, the previous public model
- **Presidio**, the PII detector [Presidio](https://github.com/microsoft/presidio) 2.2.366 with the French
  `fr_core_news_lg` model of spaCy, as an outside baseline. Since it predicts its own labels, the other sections only
  give its redaction scores, and the [Presidio](#comparison-with-presidio) section scores every model on its labels

The AP-HP test set holds 688 documents. Since AP-HP notes are PDF files, 340 test notes appear twice, once in their
edspdf text and once in their pdfbox text (see [PDF extraction](#impact-of-the-pdf-extraction-step)), and 8 more notes
only in their pdfbox text.

Precision, recall and F1 count the tokens with the right label. Redact is the share of identifying tokens that are
detected whatever their label (so they get altered or redacted, the kind of PII doesn't matter here, hence "Redact"),
and Full redact the share of documents where all the identifying tokens get redacted. None of these scores count
`HOPITAL`, which we only annotate to show the model what not to redact.

## General results

The table gives the scores on the 688 test documents, as the mean over the 3 seeds of ML, Hybrid and Public v2.

--8<-- "docs/assets/figures/ml_vs_rb_table.html"

The chart gives the same scores per label with their spread over seeds. Its menus choose the metric, the matching
(`token`, or `entity` where an identifier only counts when both its bounds and its label are right) and the test
set: all the test documents, the edspdf or pdfbox text of the 340 notes extracted by both, or the
[form answers](#form-answers).

*If you have trouble seeing the chart, please refresh the page.*

```vegalite
{
  "schema-url": "../assets/figures/label_scores.json"
}
```

---

## Impact of the language model

We train the hybrid pipeline from three language models, CamemBERT AP-HP (CamemBERT further trained on AP-HP notes,
our default), CamemBERT base and DoctoBERT. The table gives the scores of the hybrid pipeline on the 688 test
documents.

--8<-- "docs/assets/figures/bert_ablation_table.html"

*If you have trouble seeing the chart, please refresh the page.*

```vegalite
{
  "schema-url": "../assets/figures/bert_ablation.json"
}
```

---

## Impact of the PDF extraction step

AP-HP notes are PDF files, which we turn into text either with pdfbox and a fixed mask of the page, or with
[edspdf](https://github.com/aphp/edspdf), whose mask is learned and drops most of the header, footer and margins. Most
identifiers sit in these parts, so on the 340 test notes extracted by both, the edspdf text keeps 3,329 of the 15,887
identifiers of the pdfbox text (79% fewer, e.g. 77 phone numbers instead of 1,848).

The token scores of the ML model barely change between the two texts, but Full redact does, since a note with fewer
identifiers is more likely to get all of them redacted. 71 of the 340 edspdf notes even have no identifier left, so
they count as fully redacted.

--8<-- "docs/assets/figures/pdf_comparison_table.html"

---

## Impact of the number of training examples

This section scores the pipeline trained on **private AP-HP documents**, to show how many annotated documents another
hospital needs to train its own model. We train it on 10 to 3,373 AP-HP training documents, without the form answers,
either alone or with the 9,233 synthetic notes of the public templates, and score it on the 688 test documents. Since
the documents don't all have the same length, the x axis gives their words (2,318 for 10 documents, 1,033,612 for all
of them). Each point is the mean over 3 seeds for 10 and 30 documents and a single run above, and the rules give a flat
line since they don't train.

*If you have trouble seeing the chart, please refresh the page.*

```vegalite
{
  "schema-url": "../assets/figures/limit_ablation.json"
}
```

## Impact of the missing note types

We train the pipeline without the training notes of one type, and score it on the test documents of that type (103
prescriptions, 90 consultation reports, 102 imaging reports and 78 emergency reports), next to the pipeline trained
with every type.

*If you have trouble seeing the chart, please refresh the page.*

```vegalite
{
  "schema-url": "../assets/figures/doc_type_ablation.json"
}
```

---

## Form answers

Form answers are the free-text answers typed in the emergency and maternity forms of AP-HP (the "short fields" of the
AP-HP data), which the models read after their question, e.g. "Personne à prévenir : sa fille Marie Dupont 06 12 34 56
78". The form answer test set holds 3,339 answers sampled to represent these forms (729 emergency and 2,610 maternity
answers), and Unseen answers keeps the 2,690 of them whose answer appears in no training or development example. Full
redact is here the share of answers whose identifying tokens all get redacted.

--8<-- "docs/assets/figures/short_field_table.html"

A patient answers many questions over their encounters, so we also estimate the share of encounters and of persons
whose answers all get redacted, from the share of fully redacted test answers of each question and the questions that
each encounter or person answered in the forms. The table gives the mean over seeds of this estimate and, in brackets,
of the bounds of its 95% interval, which we get by resampling 1,000 times the number of fully redacted answers of each
question.

--8<-- "docs/assets/figures/short_field_population_table.html"

## Comparison with Presidio

Presidio predicts 5 labels that cover 9 of ours, so we merge our labels into them on the gold and the predicted
entities: `NOM` and `PRENOM` into `PERSON`, `ADRESSE`, `VILLE` and `ZIP` into `LOCATION`, `DATE` and `DATE_NAISSANCE`
into `DATE_TIME`, `MAIL` into `EMAIL_ADDRESS` and `TEL` into `PHONE_NUMBER`. Since Presidio has no label for `IPP`,
`NDA` and `SECU`, they don't count here, and the [general results](#general-results) give how much of them Presidio
redacts. The table gives the token scores on the 688 test documents, where a token counts when its merged label is
right, as the mean over the 3 seeds of ML, Hybrid and Public v2.

--8<-- "docs/assets/figures/presidio_table.html"

## Runs

Each row gives an experiment, the eds-pseudo commit it ran and its seeds.

--8<-- "docs/assets/figures/runs_table.html"
