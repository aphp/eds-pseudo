import os.path
from pathlib import Path
from typing import Dict, List, Union

import spacy
import srsly
import typer
from edsnlp.connectors.brat import BratConnector
from spacy.language import Language
from spacy.tokens import Doc, DocBin
from spacy.util import filter_spans
import xml.etree.ElementTree as ET

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


def create_doc_from_xml_and_txt(
    nlp: spacy.Language,
    file_xml_path: str,
    file_txt_path: str
) -> spacy.tokens.Doc:
    """Returns the Doc object corresponding to the annotated document.

                Args:
                    nlp: The spacy Language we use, here it is "eds"
                    file_xml_path: Path to the Inception annotations in xml format
                    file_txt_path: Path to the text file
    """
    tree = ET.parse(file_xml_path)
    root = tree.getroot()
    with open(file_txt_path, "r") as file:
        txt = file.read()
    doc = nlp(txt)
    entities = []
    for children in root:
        for child in children.iter('webanno.custom.Span'):
            if 'Span' in child.attrib:
                key = 'Span'
            elif 'Label' in child.attrib:
                key = 'Label'
            else:
                raise ValueError('No Span or Label indicated for this annotation in', file_xml_path)
            entity = dict()
            entity['start'] = int(child.attrib['begin'])
            entity['end'] = int(child.attrib['end'])
            entity['label'] = child.attrib[key]
            entities.append(entity)
    add_entities(doc, entities)
    return doc


def get_nlp(lang: str) -> Language:
    nlp = spacy.blank(lang)
    nlp.add_pipe("sentencizer")

    return nlp


def convert_inception_xml(
    nlp: spacy.Language,
    xml_path: Path,
    txt_path: Path,
    split: str
) -> spacy.tokens.DocBin:
    """Returns the spacy DocBin containing the Doc object corresponding to each document.

            Args:
                nlp: The spacy Language we use, here it is "eds"
                xml_path: Path to the directory containing the Inception annotations in xml format
                txt_path: Path to the directory containing the text files
                split: Split of the data to use

            Returns:
                The spacy DocBin containing the Doc object corresponding to each document.
    """
    db = DocBin(store_user_data=True)
    for filename_txt in os.listdir(txt_path):
        filename_xml = filename_txt.split('.txt')[0] + '.xml'
        file_path_xml = os.path.join(xml_path, filename_xml)
        file_path_txt = os.path.join(txt_path, filename_txt)
        if os.path.isfile(file_path_xml):
            if os.path.isfile(file_path_txt):
                doc=create_doc_from_xml_and_txt(nlp, file_path_xml, file_path_txt)
                doc._.note_id = filename_txt.split('.txt')[0]
                # doc._.note_datetime = note_datetime
                #the files are named in a format like this: ORBIS.xxxxxx.CLASS.xxxxx.txt
                doc._.note_class_source_value = filename_txt.split('.')[2]
                doc._.context = ""
                doc._.split = split
                db.add(doc)
    return db


def convert(
    lang: str = typer.Option(
        "fr",
        help="Language to use",
    ),
    xml_path: Path = typer.Option(
        ...,
        help="Directory to the xml files",
    ),
    txt_path: Path = typer.Option(
        ...,
        help="Directory to the txt files",
    ),
    output_path: Path = typer.Option(
        ...,
        help="Path to the output spacy DocBin",
    ),
    split: str = typer.Option(
        ...,
        help="split of the data to use"),
) -> None:
    nlp = get_nlp(lang)

    db = convert_inception_xml(nlp, xml_path, txt_path, split)

    typer.echo(f"The saved dataset contains {len(db)} documents.")
    db.to_disk(output_path)


if __name__ == "__main__":
    typer.run(convert)
