from itertools import chain
from typing import Callable, Optional, Sequence

from edsnlp.matchers.regex import RegexMatcher
from edsnlp.utils.filter import filter_spans
from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc

from ...utils.resources import get_hospitals
from .patterns import common_medical_terms, patterns, person_patterns

DEFAULT_CONFIG = dict(
    attr="NORM",
    scorer={"@scorers": "spacy.ner_scorer.v1"},
    pattern_keys=[*patterns, "PERSON", "HOPITAL"]
)


@Language.factory("pseudonymisation-rules", default_config=DEFAULT_CONFIG)
class Pseudonymisation:
    def __init__(
        self,
        nlp: Language,
        name: str,
        attr: str,
        scorer: Optional[Callable],
        pattern_keys: Sequence[str]
    ):

        self.scorer = scorer
        self.regex_matcher = RegexMatcher(attr=attr)
        self.phrase_matcher = PhraseMatcher(vocab=nlp.vocab, attr=attr)

        self.regex_matcher.build_patterns(
            {k: v for k, v in patterns.items() if k in pattern_keys}
        )  # , 'ADRESSE': address_patterns})
        for key in pattern_keys:
            if key != "PERSON" and key != "HOPITAL":
                assert key in patterns, "Missing pattern: {}".format(key)

        self.person_matcher = RegexMatcher(attr="TEXT")
        if "PERSON" in pattern_keys:
            self.person_matcher.build_patterns({"PERSON": person_patterns})

        # We add Hospitals
        if "HOPITAL" in pattern_keys:
            hospitals = get_hospitals()
            self.phrase_matcher.add(
                key="HOPITAL",
                docs=[nlp.make_doc(name) for name in dict.fromkeys(hospitals)],
            )

        # We add first names
        # first_names = get_first_names()
        # self.phrase_matcher.add(key="PRENOM", docs=list(
        #     nlp.pipe(list(set(first_names) - {"MME"}))
        # ))

    def process(self, doc: Doc) -> Doc:

        regex_matches = list(self.regex_matcher(doc, as_spans=True))

        matches = list(
            chain(
                regex_matches,
                self.phrase_matcher(doc, as_spans=True),
            )
        )

        filtered_matches = []
        for match in matches:
            if match.label_ == "VILLE" or match.label_ == "PRENOM":
                if match.text.isupper() and match.text in common_medical_terms:
                    continue
                if not (match.text[0].islower() or len(match.text) == 1):
                    filtered_matches.append(match)
            else:
                filtered_matches.append(match)

        for span, groups in self.person_matcher(
            doc, as_spans=True, return_groupdict=True
        ):
            first_name = groups.get("FN0") or groups.get("FN1")
            last_name = groups.get("LN0") or groups.get("LN1") or groups.get("LN2")
            if first_name is not None:
                begin = span[0].idx + span.text.index(first_name)
                filtered_matches.insert(
                    0,
                    doc.char_span(
                        begin,
                        begin + len(first_name),
                        "PRENOM",
                        alignment_mode="expand",
                    ),
                )
            if last_name is not None:
                begin = span[0].idx + span.text.index(last_name)
                filtered_matches.insert(
                    0,
                    doc.char_span(
                        begin, begin + len(last_name), "NOM", alignment_mode="expand"
                    ),
                )

        matches = filter_spans((*doc.ents, *filtered_matches))

        doc.ents = matches

        return doc

    def score(self, examples, **kwargs):
        return self.scorer(examples, **kwargs)

    def __call__(self, doc: Doc) -> Doc:
        return self.process(doc)
