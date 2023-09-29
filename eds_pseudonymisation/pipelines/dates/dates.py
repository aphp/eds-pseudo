import re
from typing import Dict, Iterable, List, Optional, Tuple, Union

from edsnlp.core import PipelineProtocol
from edsnlp.pipelines.base import SpanSetterArg
from edsnlp.pipelines.misc.dates.dates import DatesMatcher
from edsnlp.pipelines.misc.dates.models import AbsoluteDate
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


class PseudonymisationDates(DatesMatcher):
    def __init__(
        self,
        nlp: PipelineProtocol = None,
        name: str = None,
        *,
        absolute: Optional[List[str]],
        relative: Optional[List[str]],
        duration: Optional[List[str]],
        false_positive: Optional[List[str]],
        on_ents_only: Union[bool, str, Iterable[str]],
        detect_periods: bool,
        detect_time: bool,
        span_setter: SpanSetterArg,
        attr: str,
    ):
        super().__init__(
            nlp=nlp,
            name=name,
            absolute=absolute,
            relative=relative,
            duration=duration,
            false_positive=false_positive,
            on_ents_only=on_ents_only,
            detect_periods=detect_periods,
            detect_time=detect_time,
            span_setter=span_setter,
            attr=attr,
        )

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
        matches = self.process(doc)
        dates = []

        birth_date = None
        for date in matches:
            snippet = doc[max(0, date.start - 5) : date.end + 5].text
            if re.search(r"\b(né|n[ée]e|naissance)\b", snippet, flags=re.IGNORECASE):
                date.label_ = "DATE_NAISSANCE"
                birth_date = date._.date
            else:
                date.label_ = "DATE"
            dates.append(date)

        for date in dates:
            if date._.date == birth_date:
                date.label_ = "DATE_NAISSANCE"

        self.set_spans(doc, dates)

        return doc
