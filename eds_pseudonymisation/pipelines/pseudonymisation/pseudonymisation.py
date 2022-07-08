from itertools import chain

from edsnlp.matchers.regex import RegexMatcher
from edsnlp.utils.filter import filter_spans
from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc

from eds_pseudonymisation.utils.resources import get_cities

from .patterns import patterns

DEFAULT_CONFIG = dict(
    attr="NORM",
)


@Language.factory("pseudonymisation-rules", default_config=DEFAULT_CONFIG)
class Pseudonymisation:
    def __init__(
        self,
        nlp: Language,
        name: str,
        attr: str,
    ):

        cities = get_cities()

        self.regex_matcher = RegexMatcher(attr=attr)
        self.phrase_matcher = PhraseMatcher(vocab=nlp.vocab, attr=attr)

        self.regex_matcher.build_patterns(patterns)

        # We add Paris, Lyon and Marseille since they contain arrondissement info
        cities_patterns = list(
            nlp.pipe(list(set(cities.name) | {"Paris", "Lyon", "Marseille"}))
        )
        self.phrase_matcher.add(key="VILLE", docs=cities_patterns)

    def process(self, doc: Doc) -> Doc:

        matches = chain(
            self.regex_matcher(doc, as_spans=True),
            self.phrase_matcher(doc, as_spans=True),
        )
        matches = filter_spans(matches)

        doc.ents = matches

        return doc

    def __call__(self, doc: Doc) -> Doc:
        return self.process(doc)
