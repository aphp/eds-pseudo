from enum import Enum
from typing import Dict, List

from edsnlp.matchers.phrase import EDSPhraseMatcher
from edsnlp.matchers.regex import RegexMatcher
from edsnlp.pipelines.base import BaseComponent
from edsnlp.utils.filter import filter_spans
from spacy.language import Language
from spacy.tokens import Doc, Span

if not Doc.has_extension("structured_data"):
    Doc.set_extension("structured_data", default=dict())


class Matcher(Enum):
    regex = "regex"
    phrase = "phrase"


@Language.factory(
    "structured-data-matcher",
    default_config=dict(
        matcher=Matcher.phrase,
        attr="NORM",
        ignore_excluded=False,
    ),
)
class StructuredDataMatcher(BaseComponent):
    """
    Provides a generic matcher component.

    Parameters
    ----------
    nlp : Language
        The spaCy object.
    name: str
        Name of the component, not used.
    matcher : Matcher
        Enum, to match using regex or phrase
    attr : str
        The default attribute to use for matching.
        Can be overridden using the `terms` and `regex` configurations.
    ignore_excluded : bool
        Whether to skip excluded tokens (requires an upstream
        pipeline to mark excluded tokens).
    """

    def __init__(
        self,
        nlp: Language,
        name: str,
        matcher: Matcher,
        attr: str,
        ignore_excluded: bool,
    ):

        self.nlp = nlp

        self.attr = attr
        self.ignore_excluded = ignore_excluded

        self.matcher_factory = (
            self.phrase_matcher_factory
            if matcher == Matcher.phrase
            else self.regex_matcher_factory
        )

        self.set_extensions()

    def phrase_matcher_factory(
        self,
        structured_data: Dict[str, List[str]],
    ) -> EDSPhraseMatcher:
        matcher = EDSPhraseMatcher(
            self.nlp.vocab,
            attr=self.attr,
            ignore_excluded=self.ignore_excluded,
        )
        matcher.build_patterns(nlp=self.nlp, terms=structured_data)
        return matcher

    def regex_matcher_factory(
        self,
        structured_data: Dict[str, List[str]],
    ) -> RegexMatcher:
        matcher = RegexMatcher(
            attr=self.attr,
            ignore_excluded=self.ignore_excluded,
        )
        matcher.build_patterns(regex=structured_data)
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

        structured_data = doc._.structured_data

        if not structured_data:
            return []

        matcher = self.matcher_factory(structured_data=structured_data)
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
        matches = self.process(doc)

        for span in matches:
            if span.label_ not in doc.spans:
                doc.spans[span.label_] = []
            doc.spans[span.label_].append(span)

        ents, discarded = filter_spans(list(doc.ents) + matches, return_discarded=True)

        doc.ents = ents

        if "discarded" not in doc.spans:
            doc.spans["discarded"] = []
        doc.spans["discarded"].extend(discarded)

        return doc
