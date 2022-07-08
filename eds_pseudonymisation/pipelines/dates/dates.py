from typing import Dict, List, Optional, Tuple

from edsnlp.pipelines.misc.dates import Dates
from edsnlp.pipelines.misc.dates.models import AbsoluteDate
from edsnlp.utils.filter import filter_spans
from pydantic import BaseModel, root_validator
from spacy.tokens import Doc, Span

if not Span.has_extension("date_string"):
    Span.set_extension("date_string", default=None)


class DateString(BaseModel):
    day: Optional[str] = None
    month: Optional[str] = None
    year: Optional[str] = None

    @root_validator(pre=True)
    def normalise_keys(cls, d: Dict[str, Optional[str]]) -> Dict[str, str]:
        normalised = {k.split("_")[0]: v for k, v in d.items() if v is not None}
        return normalised


class PseudonymisationDates(Dates):
    def parse(self, dates: List[Tuple[Span, Dict[str, str]]]) -> List[Span]:
        """
        Parse dates using the groupdict returned by the matcher.

        Parameters
        ----------
        dates : List[Tuple[Span, Dict[str, str]]]
            List of tuples containing the spans and groupdict
            returned by the matcher.

        Returns
        -------
        List[Span]
            List of processed spans, with the date parsed.
        """

        spans = []

        for span, groupdict in dates:
            if span.label_ == "absolute":
                parsed = AbsoluteDate.parse_obj(groupdict)

                span._.date = parsed
                span._.date_string = DateString.parse_obj(groupdict)

                spans.append(span)

        return spans

    def __call__(self, doc: Doc) -> Doc:
        """
        Tags dates.

        Parameters
        ----------
        doc : Doc
            spaCy Doc object

        Returns
        -------
        doc : Doc
            spaCy Doc object, annotated for dates
        """
        dates = self.process(doc)
        dates = self.parse(dates)

        dates = [Span(doc, d.start, d.end, label="DATE") for d in dates]

        ents = filter_spans(list(doc.ents) + dates, return_discarded=False)

        doc.ents = ents

        return doc
