from typing import Callable, List, Optional

from spacy.language import Language

from .dates import PseudonymisationDates
from .patterns import false_positive_pattern, pseudo_date_pattern

DEFAULT_CONFIG = dict(
    absolute=None,
    false_positive=None,
    attr="LOWER",
    scorer={"@scorers": "spacy.ner_scorer.v1"},
)


@Language.factory("pseudonymisation-dates", default_config=DEFAULT_CONFIG)
def create_component(
    nlp: Language,
    name: str,
    absolute: Optional[List[str]],
    false_positive: Optional[List[str]],
    attr: str,
    scorer: Optional[Callable],
):

    if absolute is None:
        absolute = pseudo_date_pattern
    if false_positive is None:
        false_positive = false_positive_pattern

    return PseudonymisationDates(
        nlp,
        absolute=absolute,
        relative=[],
        duration=[],
        detect_time=False,
        false_positive=false_positive,
        on_ents_only=False,
        detect_periods=False,
        as_ents=True,
        attr=attr,
        scorer=scorer,
    )
