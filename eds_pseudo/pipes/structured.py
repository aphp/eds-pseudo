from string import punctuation
from typing import Dict, List

from spacy.tokens import Doc, Span

from edsnlp import registry
from edsnlp.core import PipelineProtocol
from edsnlp.matchers.phrase import EDSPhraseMatcher
from edsnlp.pipes.base import BaseNERComponent, SpanSetterArg


@registry.factory.register("eds_pseudo.context")
class ContextMatcher(BaseNERComponent):
    """
    Provides a component for matching terms retrieved from the patient context, such as
    the patient name or address.

    Parameters
    ----------
    nlp : Language
        The pipeline object
    name: str
        Name of the component, not used.
    attr : str
        The default attribute to use for matching.
        Can be overridden using the `terms` and `regex` configurations.
    ignore_excluded : bool
        Whether to skip excluded tokens (requires an upstream
        pipeline to mark excluded tokens).
    """

    def __init__(
        self,
        nlp: PipelineProtocol = None,
        name: str = None,
        *,
        span_setter: SpanSetterArg = {
            "ents": True,
            "pseudo-rb": True,
            "*": True,
        },
        attr: str = "NORM",
        ignore_excluded: bool = False,
    ):
        super().__init__(nlp, name, span_setter=span_setter)

        self.nlp = nlp
        self.attr = attr
        self.ignore_excluded = ignore_excluded
        self.punct_remover = str.maketrans(punctuation, " " * len(punctuation))

    def set_extensions(self):
        if not Span.has_extension("source"):
            Span.set_extension("source", default=None)
        super().set_extensions()

    def matcher_factory(
        self,
        context: Dict[str, List[str]],
    ) -> EDSPhraseMatcher:
        matcher = EDSPhraseMatcher(
            self.nlp.vocab,
            attr=self.attr,
            ignore_excluded=self.ignore_excluded,
        )
        matcher.build_patterns(
            nlp=self.nlp,
            terms={
                k: set(v)
                | set(pat.title() for pat in v)
                | set(pat.upper() for pat in v)
                for k, v in context.items()
            },
        )
        return matcher

    def process(self, doc: Doc) -> List[Span]:
        """
        Find matching spans in doc.

        Parameters
        ----------
        doc:
            spaCy Doc object.

        Returns
        -------
        spans:
            List of Spans returned by the matchers.
        """

        context = doc._.context
        if "EMAIL" in context:
            context["MAIL"] = context.pop("EMAIL")

        if not context:
            return []

        context_patterns: Dict[str, List[str]] = {
            key: [v for v in values if len(v.translate(self.punct_remover).strip()) > 2]
            for key, values in context.items()
        }
        matcher = self.matcher_factory(context_patterns)
        matches = matcher(doc, as_spans=True)

        return list(matches)

    def __call__(self, doc: Doc) -> Doc:
        """
        Adds spans to document.

        Parameters
        ----------
        doc:
            spaCy Doc object

        Returns
        -------
        doc:
            spaCy Doc object, annotated for extracted terms.
        """
        assert doc.vocab is self.nlp.vocab
        matches = self.process(doc)

        for span in matches:
            span._.source = "structured"
            if span.label_ == "NOM_NAISS":
                span.label_ = "NOM"

        return self.set_spans(doc, matches)
