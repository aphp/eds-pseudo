import os.path
from pathlib import Path
from typing import Dict, List, Union

import spacy
import srsly
import typer
from spacy.language import Language
from spacy.tokens import Doc, DocBin
from spacy.util import filter_spans

from edsnlp.connectors.brat import BratConnector

if not Doc.has_extension("context"):
    Doc.set_extension("context", default=dict())
if not Doc.has_extension("note_id"):
    Doc.set_extension("note_id", default=None)
if not Doc.has_extension("note_datetime"):
    Doc.set_extension("note_datetime", default=None)
if not Doc.has_extension("note_class_source_value"):
    Doc.set_extension("note_class_source_value", default=None)
if not Doc.has_extension("split"):
    Doc.set_extension("split", default=None)


def add_entities(doc: Doc, entities: List[Dict[str, Union[int, str]]]):
    """
    Add annotations as Doc entities, re-tokenizing the document if need be.

    Parameters
    ----------
    doc : Doc
        spaCy Doc object
    entities : List[Dict[str, Union[int, str]]]
        List of annotations.
    """

    ents = []

    for entity in entities:
        start, end, label = entity["start"], entity["end"], entity["label"]
        span = doc.char_span(start, end, label=label, alignment_mode="expand")
        if span:
            ents.append(span)

    doc.ents = filter_spans(ents)


def get_nlp(lang: str) -> Language:
    nlp = spacy.blank(lang)
    nlp.add_pipe("sentencizer")

    return nlp


def convert_jsonl(
      nlp: spacy.Language,
      input_path: Path,
) -> spacy.tokens.DocBin:
    db = DocBin(store_user_data=True)

    for annot in srsly.read_jsonl(input_path):
        text, note_id, note_datetime, note_class_source_value, entities, context, split = (
            annot["note_text"],
            annot["note_id"],
            annot["note_datetime"],
            annot["note_class_source_value"],
            annot.get("entities", []),
            annot.get("context", {}),
            annot.get("split", None),
        )

        doc = nlp(text)
        doc._.note_id = note_id
        # doc._.note_datetime = note_datetime
        doc._.note_class_source_value = note_class_source_value
        doc._.context = context
        doc._.split = split

        add_entities(doc, entities)

        db.add(doc)

    return db


def convert_brat(
      nlp: spacy.Language,
      input_path: Path,
) -> spacy.tokens.DocBin:
    db = DocBin(store_user_data=True)

    connector = BratConnector(input_path)
    for doc in connector.brat2docs(nlp):
        db.add(doc)

    return db


def convert(
      lang: str = typer.Option(
          "fr",
          help="Language to use",
      ),
      input_path: Path = typer.Option(
          ...,
          help="Path to the JSONL file",
      ),
      output_path: Path = typer.Option(
          ...,
          help="Path to the output spacy DocBin",
      ),
) -> None:
    nlp = get_nlp(lang)

    if os.path.isdir(input_path):
        db = convert_brat(nlp, input_path)
    else:
        db = convert_jsonl(nlp, input_path)

    typer.echo(f"The saved dataset contains {len(db)} documents.")
    db.to_disk(output_path)


if __name__ == "__main__":
    typer.run(convert)
