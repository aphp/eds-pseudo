from edsnlp import registry
from edsnlp.core import PipelineProtocol
from edsnlp.pipelines.base import (
    BaseNERComponent,
    SpanGetterArg,
    SpanSetterArg,
    get_spans,
)
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
        ents = get_spans(doc, self.span_getter)
        return self.set_spans(doc, ents)
