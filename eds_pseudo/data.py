import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from spacy.tokens import Doc

from edsnlp import registry
from edsnlp.core import PipelineProtocol
from edsnlp.data.converters import AttributesMappingArg, get_current_tokenizer
from edsnlp.pipes.base import SpanGetterArg, get_spans
from edsnlp.utils.span_getters import SpanSetterArg, set_spans

# Record fields that `Dict2DocConverter` copies to `doc._`, so that the metrics of `metrics.py` can group their scores
# by source, question, encounter or person
DOC_FIELDS = (
    "note_id",
    "source_id",
    "group_id",
    "question_id",
    "encounter_id",
    "person_id",
    "source",
    "question",
    "question_frequency",
    "split",
    "note_class_source_value",
    "note_datetime",
    "original_text",
    "context",
)


def text_hash(text: str) -> str:
    """
    Hash a text after collapsing spaces, casing and Unicode variants, so that `prepare-data` sees " Alice  " and
    "alice" as the same text when it keeps test texts out of training data
    """
    normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()
    return hashlib.sha256(normalized.encode()).hexdigest()


def file_hash(path: Path) -> str:
    """
    SHA256 of a file, written by `prepare-data` to `dataset-info.json` for each input and output file
    """
    with path.open("rb") as file:
        digest = hashlib.sha256()
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Check that a record has a `note_text` and entities with `start`, `end` and `label` inside that text, and drop the
    other entity keys (e.g. a copied `text`), since offsets are the only thing we trust
    """
    result = dict(record)
    if not isinstance(result.get("note_text"), str):
        raise ValueError("Each record must contain a string note_text field")
    entities = []
    for entity in result.get("entities") or []:
        entity = dict(entity)
        if set(("start", "end", "label")) - entity.keys():
            raise ValueError("Each entity must contain start, end and label")
        start, end, label = entity["start"], entity["end"], entity["label"]
        if not isinstance(start, int) or not isinstance(end, int) or not isinstance(label, str):
            raise ValueError("Entity start and end must be integers and label must be a string")
        if start < 0 or end <= start or end > len(result["note_text"]):
            raise ValueError(f"Invalid entity boundary {start}:{end} for text of length {len(result['note_text'])}")
        entities.append({"start": start, "end": end, "label": label})
    result["entities"] = entities
    return result


@registry.factory.register("eds_pseudo.dict2doc", spacy_compatible=False)
class Dict2DocConverter:
    """
    The `eds_pseudo.dict2doc` converter turns a JSON record into a Doc for the `train_data` and `val_data` readers of
    the configs and for `eds-pseudo evaluate`.

    We tokenize the text first, independently of its entities, so that the gold doc and the prediction doc of a note
    get the same tokens, since the NER metrics of EDS-NLP compare spans by token index. An entity whose bounds fall
    inside a token is stretched to the whole token, e.g. `Alice` annotated in "AliceDupont" becomes "AliceDupont".
    """

    def __init__(
        self,
        nlp: Optional[PipelineProtocol] = None,
        name: str = "pseudo_dict2doc",
        *,
        tokenizer: Any = None,
        span_setter: SpanSetterArg = {"ents": True, "*": True},
        doc_attributes: AttributesMappingArg = {field: field for field in DOC_FIELDS},
    ):
        self.tokenizer = tokenizer or (nlp.tokenizer if nlp is not None else None)
        self.span_setter = span_setter
        self.doc_attributes = doc_attributes

    def __call__(self, obj: Mapping[str, Any]) -> Doc:
        record = normalize_record(obj)
        tokenizer = get_current_tokenizer() if self.tokenizer is None else self.tokenizer
        doc = tokenizer(record["note_text"])
        for obj_name, extension_name in self.doc_attributes.items():
            if not Doc.has_extension(extension_name):
                Doc.set_extension(extension_name, default=None)
            if obj_name in record:
                doc._.set(extension_name, record[obj_name])
        spans = [
            doc.char_span(entity["start"], entity["end"], label=entity["label"], alignment_mode="expand")
            for entity in record["entities"]
        ]
        set_spans(doc, spans, span_setter=self.span_setter)
        return doc


@registry.factory.register("eds_pseudo.doc2dict", spacy_compatible=False)
class Doc2DictConverter:
    """
    The `eds_pseudo.doc2dict` converter turns a Doc back into the JSON record that `eds_pseudo.dict2doc` reads, with
    the `DOC_FIELDS` of `doc._` and the spans of `span_getter` as entities. `eds-pseudo infer` writes its predictions
    with it.
    """

    def __init__(
        self,
        *,
        span_getter: SpanGetterArg = {"ents": True},
        doc_attributes: AttributesMappingArg = {field: field for field in DOC_FIELDS},
    ):
        self.span_getter = span_getter
        self.doc_attributes = doc_attributes

    def __call__(self, doc: Doc) -> Dict[str, Any]:
        record = {"note_text": doc.text}
        for extension_name, obj_name in self.doc_attributes.items():
            if Doc.has_extension(extension_name) and doc._.get(extension_name) is not None:
                record[obj_name] = doc._.get(extension_name)
        record["entities"] = [
            {"start": span.start_char, "end": span.end_char, "label": span.label_}
            for span in get_spans(doc, self.span_getter)
        ]
        return record
