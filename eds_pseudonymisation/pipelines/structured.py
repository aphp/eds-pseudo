from enum import Enum
from string import punctuation
from typing import Callable, Dict, List, Optional, Tuple

from edsnlp.matchers.phrase import EDSPhraseMatcher
from edsnlp.matchers.regex import RegexMatcher
from edsnlp.pipelines.base import BaseComponent
from edsnlp.utils.filter import filter_spans
from spacy.language import Language
from spacy.tokens import Doc, Span

if not Doc.has_extension("structured_data"):
    Doc.set_extension("structured_data", default=dict())


class Matcher(str, Enum):
    regex = "regex"
    phrase = "phrase"


def pseudo_sort_key(span: Span) -> Tuple[int, int]:
    """
    Returns the sort key for filtering spans.
    The key differs from the default sort key because we attribute more
    importance to patterns that come from structured data such as the patient name
    or address patterns when they conflict with more general patterns.

    Parameters
    ----------
    span : Span
        Span to sort.
    Returns
    -------
    key : Tuple(int, int)
        Sort key.
    """
    if isinstance(span, tuple):
        span = span[0]
    return (
        (1 if span._.source == "structured" else 0),
        span.end - span.start,
        -span.start,
    )


@Language.factory(
    "structured-data-matcher",
    default_config=dict(
        matcher=Matcher.phrase,
        attr="NORM",
        ignore_excluded=False,
        scorer={"@scorers": "spacy.ner_scorer.v1"},
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
        scorer: Optional[Callable],
    ):

        self.nlp = nlp

        self.attr = attr
        self.ignore_excluded = ignore_excluded

        self.matcher_factory = (
            self.phrase_matcher_factory
            if matcher == Matcher.phrase
            else self.regex_matcher_factory
        )
        self.punct_remover = str.maketrans(punctuation, " " * len(punctuation))

        self.set_extensions()

        self.scorer = scorer

    def score(self, examples, **kwargs):
        return self.scorer(examples, **kwargs)

    def set_extensions(self):
        if not Span.has_extension("source"):
            Span.set_extension("source", default=None)
        super().set_extensions()

    def phrase_matcher_factory(
        self,
        structured_data: Dict[str, List[str]],
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
                for k, v in structured_data.items()
            },
        )
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
        if "EMAIL" in structured_data:
            structured_data["MAIL"] = structured_data.pop("EMAIL")

        if not structured_data:
            return []

        matcher = self.matcher_factory(
            structured_data={
                key: tuple(
                    v
                    for v in values
                    if len(v.translate(self.punct_remover).strip()) > 2
                )
                for key, values in structured_data.items()
            }
        )
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
            span._.source = "structured"
            if span.label_ == "NOM_NAISS":
                span.label_ = "NOM"
            if span.label_ not in doc.spans:
                doc.spans[span.label_] = []
            doc.spans[span.label_].append(span)

        ents, discarded = filter_spans(
            matches + list(doc.ents),
            return_discarded=True,
            sort_key=pseudo_sort_key,
        )
        self.last_ents = ents

        doc.ents = ents

        if "discarded" not in doc.spans:
            doc.spans["discarded"] = []
        doc.spans["discarded"].extend(discarded)

        return doc
