from edsnlp import registry
from edsnlp.core import PipelineProtocol
from edsnlp.pipelines.base import (
    BaseNERComponent,
    SpanGetterArg,
    SpanSetterArg,
    get_spans,
)
from edsnlp.utils.filter import filter_spans
from spacy.tokens import Doc


@registry.factory.register("eds_pseudo.merge")
class MergeEntities(BaseNERComponent):
    """
    Removes empty entities from the document and clean entity boundaries
    """

    def __init__(
        self,
        nlp: PipelineProtocol = None,
        name: str = None,
        *,
        span_getter: SpanGetterArg = {"pseudo-rb": True, "pseudo-ml": True},
        span_setter: SpanSetterArg = {"pseudo-hybrid": True},
    ):
        super().__init__(nlp, name, span_setter=span_setter)
        self.span_getter = span_getter

    def __call__(self, doc: Doc) -> Doc:
        ents = filter_spans(get_spans(doc, self.span_getter))
        for group in self.span_setter:
            doc.spans[group] = []
        for group in self.span_getter:
            if group in doc.spans:
                doc.spans[group] = filter_spans(doc.spans[group])
        return self.set_spans(doc, ents)
