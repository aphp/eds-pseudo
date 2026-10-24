import re
from typing import Dict, Iterable, List, Optional, Union

from pydantic import BaseModel, model_validator
from spacy.tokens import Doc, Span
from typing_extensions import Literal

from edsnlp import registry
from edsnlp.core import PipelineProtocol
from edsnlp.pipes.base import SpanGetterArg, SpanSetterArg
from edsnlp.pipes.misc.dates.dates import DatesMatcher

from . import patterns

if not Span.has_extension("date_string"):
    Span.set_extension("date_string", default=None)


class DateString(BaseModel):
    """
    Keep the matched date fields before EDS-NLP converts them to integers
    """

    day: Optional[str] = None
    month: Optional[str] = None
    year: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_keys(cls, values: Dict[str, Optional[str]]) -> Dict[str, str]:
        return {key.split("_", 1)[0]: value for key, value in values.items() if value is not None}


@registry.factory.register("eds_pseudo.dates", deprecated=["pseudonymisation-dates"])
class PseudonymisationDates(DatesMatcher):
    """
    Detect absolute identifying dates for the rules-only pipeline
    """

    def __init__(
        self,
        nlp: PipelineProtocol,
        name: str = "dates",
        *,
        absolute: Optional[List[str]] = None,
        relative: Optional[List[str]] = [],
        duration: Optional[List[str]] = [],
        false_positive: Optional[List[str]] = None,
        on_ents_only: Optional[Union[bool, str, Iterable[str]]] = None,
        span_getter: Optional[SpanGetterArg] = None,
        merge_mode: Literal["intersect", "align"] = "intersect",
        detect_periods: bool = False,
        detect_time: bool = False,
        attr: str = "LOWER",
        span_setter: SpanSetterArg = {"ents": True, "pseudo-rb": True, "*": True},
    ):
        super().__init__(
            nlp=nlp,
            name=name,
            absolute=patterns.pseudo_date_pattern if absolute is None else absolute,
            relative=relative,
            duration=duration,
            false_positive=patterns.false_positive_pattern if false_positive is None else false_positive,
            on_ents_only=on_ents_only,
            span_getter=span_getter,
            merge_mode=merge_mode,
            detect_periods=detect_periods,
            detect_time=detect_time,
            attr=attr,
            span_setter=span_setter,
        )

    def parse(self, matches):
        date_strings = [DateString.model_validate(groups) for _, groups in matches]
        dates = super().parse(matches)
        for span, date_string in zip(dates, date_strings):
            span._.date_string = date_string
        return dates

    def __call__(self, doc: Doc) -> Doc:
        dates = self.process(doc)
        birth_dates = set()
        for date in dates:
            snippet = doc[max(0, date.start - 5) : date.end + 5].text
            if re.search(r"\b(né|n[ée]e|naissance)\b", snippet, flags=re.IGNORECASE):
                birth_dates.add(str(date._.date))
        for date in dates:
            date.label_ = "DATE_NAISSANCE" if str(date._.date) in birth_dates else "DATE"
        self.set_spans(doc, dates)
        return doc
