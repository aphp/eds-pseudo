from typing import Callable, Optional

from edsnlp.matchers.regex import RegexMatcher
from edsnlp.utils.filter import filter_spans
from spacy.language import Language
from spacy.tokens import Doc

from .patterns import address_patterns

DEFAULT_CONFIG = dict(
    attr="NORM",
    scorer={"@scorers": "spacy.ner_scorer.v1"},
)


@Language.factory("pseudonymisation-addresses", default_config=DEFAULT_CONFIG)
class PseudonymisationAddresses:
    def __init__(
        self,
        nlp: Language,
        name: str,
        attr: str,
        scorer: Optional[Callable],
    ):

        self.scorer = scorer
        self.regex_matcher = RegexMatcher(attr=attr)
        self.regex_matcher.build_patterns({"STREET_ADDRESS": address_patterns})


    def process(self, doc: Doc) -> Doc:

        addresses = []
        zip_codes = []
        cities = []
        filtered_matches = []
        for span, gd in self.regex_matcher.match_with_groupdict_as_spans(doc):
            print(gd)
            if gd.get("UPPER_STREET") is not None and gd.get("NUMERO") is not None:
                filtered_matches.append((span, gd))
            elif (
                gd.get("ZIP_CODE") is not None
                and gd.get("CITY") is not None
                and gd.get("NUMERO") is not None
            ):
                filtered_matches.append((span, gd))
            elif (
                gd.get("ZIP_CODE") is not None
                or gd.get("CITY") is not None
                or gd.get("TRIGGER") is not None
            ) and (
                gd.get("STREET_PIECE") is not None
                or (
                    gd.get("NUMERO") is not None
                    and gd.get("LOWER_STREET_PIECE") is not None
                )
            ):
                filtered_matches.append((span, gd))

        filtered_matches = filter_spans(filtered_matches)

        for span, gd in filtered_matches:
            addresses.append(span)
            if "ZIP_CODE" in gd:
                zip_codes.append(gd["ZIP_CODE"])
            if "CITY" in gd:
                cities.append(gd["CITY"])

        doc.spans["STREET_ADDRESS"] = addresses
        doc.spans["CITY"] = cities
        doc.spans["ZIP_CODE"] = zip_codes
        doc.ents = filter_spans((*doc.ents, *addresses, *cities, *zip_codes))

    def __call__(self, doc: Doc) -> Doc:
        self.process(doc)
        return doc
