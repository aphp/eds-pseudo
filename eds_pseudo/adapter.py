import random
from typing import Any, Iterable, List, Optional

import spacy
from confit import validate_arguments
from spacy.tokens import Doc

import edsnlp
from edsnlp import registry
from edsnlp.core.pipeline import PipelineProtocol
from edsnlp.data.converters import (
    FILENAME,
    AttributesMappingArg,
    SequenceStr,
    get_current_tokenizer,
)
from edsnlp.utils.span_getters import SpanSetterArg, set_spans


@registry.factory.register("eds.pseudo_dict2doc", spacy_compatible=False)
class PseudoDict2DocConverter:
    """
    Read a JSON dictionary in the format used by EDS-Pseudo and convert it to a Doc
    object.

    Parameters
    ----------
    nlp: Optional[PipelineProtocol]
        The pipeline object (optional and likely not needed, prefer to use the
        `tokenizer` directly argument instead).
    tokenizer: Optional[spacy.tokenizer.Tokenizer]
        The tokenizer instance used to tokenize the documents. Likely not needed since
        by default it uses the current context tokenizer :

        - the tokenizer of the next pipeline run by `.map_model` in a
          [LazyCollection][edsnlp.core.lazy_collection.LazyCollection].
        - or the `eds` tokenizer by default.
    span_setter: SpanSetterArg
        The span setter to use when setting the spans in the documents. Defaults to
        setting the spans in the `ents` attribute, and creates a new span group for
        each JSON entity label.
    doc_attributes: AttributesMappingArg
        Mapping from JSON attributes to Span extensions (can be a list too).
        By default, all attributes are imported as Doc extensions with the same name.
    span_attributes: Optional[AttributesMappingArg]
        Mapping from JSON attributes to Span extensions (can be a list too).
        By default, all attributes are imported as Span extensions with the same name.
    bool_attributes: SequenceStr
        List of attributes for which missing values should be set to False.
    """

    def __init__(
        self,
        nlp: Optional[PipelineProtocol] = None,
        name: str = "pseudo_dict2doc",
        *,
        tokenizer: Optional[PipelineProtocol] = None,
        span_setter: SpanSetterArg = {"ents": True, "*": True},
        doc_attributes: AttributesMappingArg = {
            "note_class_source_value": "note_class_source_value",
            "note_datetime": "note_datetime",
            "context": "context",
        },
        bool_attributes: SequenceStr = [],
    ):
        self.tokenizer = tokenizer or (nlp.tokenizer if nlp is not None else None)
        self.span_setter = span_setter
        self.doc_attributes = doc_attributes
        self.bool_attributes = bool_attributes

    def __call__(self, obj):
        tok = get_current_tokenizer() if self.tokenizer is None else self.tokenizer
        doc = tok(obj["note_text"] or "")
        doc._.note_id = obj.get("note_id", obj.get(FILENAME))
        for obj_name, ext_name in self.doc_attributes.items():
            if not Doc.has_extension(ext_name):
                Doc.set_extension(ext_name, default=None)
            if obj_name in obj:
                doc._.set(ext_name, obj[obj_name])

        spans = []

        for ent in obj.get("entities") or ():
            ent = dict(ent)
            span = doc.char_span(
                ent.pop("start"),
                ent.pop("end"),
                label=ent.pop("label"),
                alignment_mode="expand",
            )
            spans.append(span)

        try:
            set_spans(doc, spans, span_setter=self.span_setter)
        except:
            print("Could not load document", obj)
            raise
        return doc


def subset_doc(doc: Doc, start: int, end: int) -> Doc:
    """
    Subset a doc given a start and end index.

    Parameters
    ----------
    doc: Doc
        The doc to subset
    start: int
        The start index
    end: int
        The end index

    Returns
    -------
    Doc
    """
    # TODO: review user_data copy strategy
    new_doc = doc[start:end].as_doc(copy_user_data=True)
    new_doc.user_data.update(doc.user_data)

    for name, group in doc.spans.items():
        new_doc.spans[name] = [
            spacy.tokens.Span(
                new_doc,
                max(0, span.start - start),
                min(end, span.end) - start,
                span.label,
            )
            for span in group
            if span.end > start and span.start < end
        ]

    return new_doc


@validate_arguments
class PseudoReader:
    """
    Reader that reads docs from a file or a generator, and adapts them to the pipeline.

    Parameters
    ----------
    source: Callable[..., Iterable[Doc]]
        The source of documents (e.g. `edsnlp.data.from_json(...)` or something else)
    limit: Optional[int]
        The maximum number of docs to read
    max_length: int
        The maximum length of the resulting docs
    randomize: bool
        Whether to randomize the split
    multi_sentence: bool
        Whether to split sentences across multiple docs
    filter_expr: Optional[str]
        An expression to filter the docs to generate
    """

    def __init__(
        self,
        source: Any,
        limit: Optional[int] = -1,
        max_length: int = 0,
        randomize: bool = False,
        multi_sentence: bool = True,
        filter_expr: Optional[str] = None,
    ):
        self.source = source
        self.limit = limit
        self.max_length = max_length
        self.randomize = randomize
        self.multi_sentence = multi_sentence
        self.filter_expr = filter_expr

    def __call__(self, nlp) -> List[Doc]:
        filter_fn = eval(f"lambda doc:{self.filter_expr}") if self.filter_expr else None

        blank_nlp = edsnlp.Pipeline(nlp.lang, vocab=nlp.vocab, vocab_config=None)
        blank_nlp.add_pipe("eds.normalizer")
        blank_nlp.add_pipe("eds.sentences")

        docs = blank_nlp.pipe(self.source)

        count = 0

        # Load the jsonl data from path
        if self.randomize:
            docs: List[Doc] = list(docs)
            random.shuffle(docs)

        for doc in docs:
            if 0 <= self.limit <= count:
                break
            if not (len(doc) and (filter_fn is None or filter_fn(doc))):
                continue
            count += 1

            for sub_doc in self.split_doc(doc):
                if len(sub_doc.text.strip()):
                    yield sub_doc
            else:
                continue

    def split_doc(
        self,
        doc: Doc,
    ) -> Iterable[Doc]:
        """
        Split a doc into multiple docs of max_length tokens.

        Parameters
        ----------
        doc: Doc
            The doc to split

        Returns
        -------
        Iterable[Doc]
        """
        max_length = self.max_length
        randomize = self.randomize

        if max_length <= 0:
            yield doc
        else:
            start = 0
            end = 0
            for ent in doc.ents:
                for token in ent:
                    token.is_sent_start = False
            for sent in doc.sents if doc.has_annotation("SENT_START") else (doc[:],):
                # If the sentence adds too many tokens
                if sent.end - start > max_length:
                    # But the current buffer too large
                    while sent.end - start > max_length:
                        subset_end = start + int(
                            max_length * (random.random() ** 0.3 if randomize else 1)
                        )
                        yield subset_doc(doc, start, subset_end)
                        start = subset_end
                    yield subset_doc(doc, start, sent.end)
                    start = sent.end

                if not self.multi_sentence:
                    yield subset_doc(doc, start, sent.end)
                    start = sent.end

                # Otherwise, extend the current buffer
                end = sent.end

            yield subset_doc(doc, start, end)
