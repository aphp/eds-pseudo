from pathlib import Path
from typing import Dict, List, Union

import spacy
import srsly
import typer
from spacy.language import Language
from spacy.tokens import Doc, DocBin
from spacy.util import filter_spans

if not Doc.has_extension("structured_data"):
    Doc.set_extension("structured_data", default=dict())
if not Doc.has_extension("note_id"):
    Doc.set_extension("note_id", default=None)


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
    db = DocBin(store_user_data=True)

    for annot in srsly.read_jsonl(input_path):
        text, note_id, entities, context = (
            annot["note_text"],
            annot["note_id"],
            annot["entities"],
            annot["context"],
        )

        doc = nlp(text)
        doc._.note_id = note_id
        doc._.structured_data = context

        add_entities(doc, entities)

        db.add(doc)

    typer.echo(f"The saved dataset contains {len(db)} documents.")
    db.to_disk(output_path)


if __name__ == "__main__":
    typer.run(convert)
