from itertools import chain
from typing import Callable, Optional, Sequence

from edsnlp.matchers.phrase import EDSPhraseMatcher
from edsnlp.matchers.regex import RegexMatcher
from edsnlp.utils.filter import filter_spans
from spacy.language import Language
from spacy.tokens import Doc

from .patterns import patterns, person_patterns

DEFAULT_CONFIG = dict(
    attr="NORM",
    scorer={"@scorers": "spacy.ner_scorer.v1"},
    pattern_keys=[*patterns, "PERSON"],
)


@Language.factory("pseudonymisation-rules", default_config=DEFAULT_CONFIG)
class Pseudonymisation:
    def __init__(
        self,
        nlp: Language,
        name: str,
        attr: str,
        scorer: Optional[Callable],
        pattern_keys: Sequence[str],
    ):

        self.scorer = scorer
        self.regex_matcher = RegexMatcher(attr=attr)
        self.phrase_matcher = EDSPhraseMatcher(vocab=nlp.vocab, attr=attr)

        self.regex_matcher.build_patterns(
            {k: v for k, v in patterns.items() if k in pattern_keys}
        )
        for key in pattern_keys:
            assert key == "PERSON" or key in patterns, "Missing pattern: {}".format(key)

        self.person_matcher = RegexMatcher(attr="TEXT")
        if "PERSON" in pattern_keys:
            self.person_matcher.build_patterns({"PERSON": person_patterns})

    def process(self, doc: Doc) -> Doc:

        matches = list(
            chain(
                list(self.regex_matcher(doc, as_spans=True)),
                self.phrase_matcher(doc, as_spans=True),
            )
        )

        for span, groups in self.person_matcher(
            doc, as_spans=True, return_groupdict=True
        ):
            first_name = groups.get("FN0") or groups.get("FN1")
            last_name = groups.get("LN0") or groups.get("LN1") or groups.get("LN2")
            if first_name is not None:
                begin = span[0].idx + span.text.index(first_name)
                matches.insert(
                    0,
                    doc.char_span(
                        begin,
                        begin + len(first_name),
                        "FIRSTNAME",
                        alignment_mode="expand",
                    ),
                )
            if last_name is not None:
                begin = span[0].idx + span.text.index(last_name)
                matches.insert(
                    0,
                    doc.char_span(
                        begin, begin + len(last_name), "LASTNAME", alignment_mode="expand"
                    ),
                )

        matches = filter_spans((*doc.ents, *matches))

        doc.ents = matches

        return doc

    def __call__(self, doc: Doc) -> Doc:
        return self.process(doc)
