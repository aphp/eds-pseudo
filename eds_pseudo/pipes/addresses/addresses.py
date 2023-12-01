from spacy.tokens import Doc

from edsnlp import registry
from edsnlp.core import PipelineProtocol
from edsnlp.matchers.regex import RegexMatcher
from edsnlp.pipes.base import BaseNERComponent, SpanSetterArg
from edsnlp.utils.filter import filter_spans

from .patterns import address_patterns


@registry.factory.register("eds_pseudo.addresses")
class PseudonymisationAddresses(BaseNERComponent):
    def __init__(
        self,
        nlp: PipelineProtocol,
        name: str,
        *,
        attr: str = "NORM",
        span_setter: SpanSetterArg = {
            "ents": True,  # base filtered span group
            "pseudo-rb": True,  # base unfiltered span group
            "ADRESSE": "ADRESSE",  # only for ADRESSE spans
            "ZIP": "ZIP",  # only for ZIP spans
            "VILLE": "VILLE",  # only for VILLE spans
        },
    ):
        super().__init__(nlp, name, span_setter=span_setter)

        self.regex_matcher = RegexMatcher(attr=attr)
        self.regex_matcher.build_patterns({"ADRESSE": address_patterns})

    def process(self, doc: Doc) -> Doc:
        addresses = []
        zip_codes = []
        cities = []
        filtered_matches = []
        for span, gd in self.regex_matcher.match_with_groupdict_as_spans(doc):
            if gd.get("UPPER_STREET") is not None and gd.get("NUMERO") is not None:
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

        filtered_matches = filter_spans(filtered_matches)

        for span, gd in filtered_matches:
            addresses.append(span)
            if "ZIP" in gd:
                zip_codes.append(gd["ZIP"])
            if "VILLE" in gd:
                cities.append(gd["VILLE"])

        return self.set_spans(doc, [*addresses, *cities, *zip_codes])

    def __call__(self, doc: Doc) -> Doc:
        self.process(doc)
        return doc
