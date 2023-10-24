from typing import List, Optional

from edsnlp import registry
from edsnlp.core import PipelineProtocol
from edsnlp.pipelines.base import SpanSetterArg

from .dates import PseudonymisationDates
from .patterns import false_positive_pattern, pseudo_date_pattern


@registry.factory.register("eds_pseudo.dates")
def create_component(
    nlp: PipelineProtocol = None,
    name: str = None,
    *,
    absolute: Optional[List[str]] = None,
    false_positive: Optional[List[str]] = None,
    attr: str = "LOWER",
    span_setter: SpanSetterArg = {
        "ents": True,
        "pseudo-rb": True,
        "DATE": "DATE",
        "DATE_NAISSANCE": "DATE_NAISSANCE",
    },
):
    if absolute is None:
        absolute = pseudo_date_pattern
    if false_positive is None:
        false_positive = false_positive_pattern

    return PseudonymisationDates(
        nlp=nlp,
        name=name,
        absolute=absolute,
        relative=[],
        duration=[],
        detect_time=False,
        false_positive=false_positive,
        on_ents_only=False,
        detect_periods=False,
        span_setter=span_setter,
        attr=attr,
    )
