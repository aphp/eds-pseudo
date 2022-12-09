from itertools import chain
from typing import Callable, Optional

from edsnlp.matchers.regex import RegexMatcher, create_span
from edsnlp.utils.filter import filter_spans
from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc

DEFAULT_CONFIG = dict(
    attr="NORM",
    scorer={"@scorers": "spacy.ner_scorer.v1"},
)
import re

opt_newline = r"[ ]*\n?[ ]*"
street_patterns_base = [
    "avenue",
    "villa",
    "impasse",
    "ave",
    "rues?",
    "(?<!(?i:en|au|le)[ ])cours?",
    "(?<!(?i:en|de|la)[ ])place?",
    "parvis?",
    "(?<!(?i:en)[ ])routes?",
    "alll?ee?",
    "sentier",
    "faubourg",
    r"(?<!(?i:de)[ ])passage(?![ ](?i:en|a|au|aux)\b)",
    "boulevards?",
    "bd",
    "chemin",
    "quaie?s?",
    "residences?",
    "citee?",
    "square",
]
street_patterns = [
    p.title().replace("(?I:", "(?i:").replace(r"\B", r"\b")
    for p in street_patterns_base
] + street_patterns_base
street_patterns_upper = [
    pat.upper().replace("(?I:", "(?i:").replace(r"\B", r"\b")
    for pat in street_patterns_base
]
street_name_piece = r"(?:(?:(?P<STREET_PIECE>[A-Z][A-Za-zéèà]{1,})|(?P<LOWER_STREET_PIECE>[a-zéèà]{2,})|de|du|la|a|à|le|les|des|DE|DU|LA|LE|LES|DES|D|L|d|l)\b|d'|D')"
street_name_piece_upper = (
    r"(?:(?:(?P<STREET_PIECE>[A-Z][A-Z]+)|[A-Z]\.|DE|DU|LA|LE|LES|DES|D|L)\b|D')"
)
city_name = r"(?P<VILLE>(?:(?:[A-Z][A-Z]+|sur|en|Paris)[-]?)+(?<![-])(?:[ ]*(?i:CEDEX)[ ]*\d{{2}})?)\b"
city_name_with_spaces = r"(?P<VILLE>(?:(?:[A-Z][A-Z]+|sur|en|Paris)[- ]?)+(?<![- ])(?:[ ]*(?i:CEDEX)[ ]*\d{{2}})?)\b"
address_patterns = [
    # RELAXED REGEX (requires a combination of NUMBER STREET CITY AND ZIP CODE)
    # See process() fn code
    rf"""(?x)
(?<=(?P<TRIGGER>Adresse[ ]?:)?\s*)
(
        (?i:(?<![:])\b(?:(?P<NUMERO>[0-9]\d*)[,]?(?:\s*(?:bis|a|b|ter))?[ ]+)?)
        (?:
            \b(?P<UPPER_STREET>(?:{'|'.join( street_patterns_upper)})\b[ ]+)
            \b(?:{street_name_piece_upper}[ -]*)*{street_name_piece_upper}
        |
            \b(?:{'|'.join(street_patterns)})\b[ -]+
            \b(?:{street_name_piece}[ -]*)*{street_name_piece}
        )
    |
    (?<=(?P<TRIGGER>Ville[ ]?:)){opt_newline}
)
(?=
    (?i:(?:[ ]*[à,.-])?{opt_newline}\(?(?P<ZIP>(?:\d{{2}}[ ]*?\d{{3}})|(?:[1-9]|1[0-9]|20)[èe]m?e?)\)?)
    {opt_newline}(?:{city_name})
    |
    (?i:(?:[ ]*[à,.-])?{opt_newline}\(?(?P<ZIP>(?:\d{{2}}[ ]*?\d{{3}})|(?:[1-9]|1[0-9]|20)[èe]m?e?)?\)?)
    (?:{city_name})
    |
    (?:(?:[ ]*[à,.-])?[ ]*{city_name})?
    (?i:{opt_newline}\(?(?P<ZIP>(?:\d{{2}}[ ]*?\d{{3}})\)?|(?:[1-9]|1[0-9]|20)[èe]m?e?)?)
)
""",
    # FULL REGEX (requires NUMBER STREET CITY AND ZIP CODE) but more flexible on street and city
    rf"""(?x)
(
        (?i:(?<![:])\b(?:(?P<NUMERO>[0-9]\d*)[,]?(?:\s*(?:bis|a|b|ter))?[ ]+)?)
        (?:
            \b(?:(?:{'|'.join(street_patterns+street_patterns_upper)})\b[ -]+)
            \b(?:(?:[1-9A-Za-z]+)[ -,]*\n?)*(?:[1-9A-Za-z]+)
        )
        (?P<REGEX_2>)
)
(?=
    (?i:(?:[ ]*[à,.-]|[ ]*\n?)?[ ]*\(?(?P<ZIP>(?:\d{{2}}[ ]*?\d{{3}})|(?:[1-9]|1[0-9]|20)[èe]m?e?)\)?[ ]+)
    (?:{city_name_with_spaces})
    |
    (?:(?:[ ]*[à,.-]|[ ]*\n?)?\s*{city_name_with_spaces})
    (?i:[ ]+\(?(?P<ZIP>(?:\d{{2}}[ ]*?\d{{3}})\)?|(?:[1-9]|1[0-9]|20)[èe]m?e?))
)
""",
]


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
        self.regex_matcher.build_patterns({"ADRESSE": address_patterns})

    def process(self, doc: Doc) -> Doc:

        addresses = []
        zip_codes = []
        cities = []
        filtered_matches = []
        for span, gd in self.regex_matcher.match_with_groupdict_as_spans(doc):
            if not len(span.text.strip()):
                continue
            if gd.get("REGEX_2"):
                if gd.get("ZIP") and gd.get("VILLE") and gt.get("NUMERO"):
                    filtered_matches.append((span, gd))
                else:
                    continue
            elif gd.get("UPPER_STREET") is not None and gd.get("NUMERO") is not None:
                filtered_matches.append((span, gd))
            elif (
                gd.get("ZIP") is not None
                and gd.get("VILLE") is not None
                and gd.get("NUMERO") is not None
            ):
                filtered_matches.append((span, gd))
            elif (
                gd.get("ZIP") is not None
                or gd.get("VILLE") is not None
                or gd.get("TRIGGER") is not None
            ) and (
                gd.get("STREET_PIECE") is not None
                or (
                    gd.get("NUMERO") is not None
                    and gd.get("LOWER_STREET_PIECE") is not None
                )
            ):
                filtered_matches.append((span, gd))
            else:
                continue

        filtered_matches = filter_spans(filtered_matches)

        for span, gd in filtered_matches:
            addresses.append(span)
            if "ZIP" in gd:
                zip_codes.append(gd["ZIP"])
            if "VILLE" in gd:
                cities.append(gd["VILLE"])

        doc.spans["ADRESSE"] = addresses
        doc.spans["VILLE"] = cities
        doc.spans["ZIP"] = zip_codes
        doc.ents = filter_spans((*doc.ents, *addresses, *cities, *zip_codes))

    def score(self, examples, **kwargs):
        return self.scorer(examples, **kwargs)

    def __call__(self, doc: Doc) -> Doc:
        self.process(doc)
        return doc
