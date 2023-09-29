import re
import string

from edsnlp import registry
from edsnlp.core import PipelineProtocol
from edsnlp.pipelines.base import BaseNERComponent, SpanGetterArg
from spacy.tokens import Doc


@registry.factory.register("eds_pseudo.clean")
class CleanEntities(BaseNERComponent):
    def __init__(
        self,
        nlp: PipelineProtocol,
        name: str,
        *,
        span_getter: SpanGetterArg = {"pseudo-rb": True, "pseudo-ml": True},
    ):
        """
        Removes empty entities from the document and clean entity boundaries
        """

        super().__init__(nlp, name, span_setter=span_getter)
        self.nlp = nlp
        self.name = name
        self.span_getter = span_getter

    @classmethod
    def clean_spans(cls, spans):
        new_spans = []
        for span in spans:
            if len(span.text.strip(string.punctuation)) == 0:
                continue
            m = re.match(r"^\s*(.*?)\s*$", span.text, flags=re.DOTALL)
            new_begin = m.start(1)
            new_end = m.end(1)
            new_ent = span.doc.char_span(
                span[0].idx + new_begin,
                span[0].idx + new_end,
                label=span.label_,
                alignment_mode="expand",
            )
            if new_ent is not None:
                new_spans.append(new_ent)
        return new_spans

    def __call__(self, doc: Doc) -> Doc:
        for name, spans in self.span_setter.items():
            if name == "ents":
                doc.ents = self.clean_spans(doc.ents)
            else:
                doc.spans[name] = self.clean_spans(doc.spans[name])
        return doc
