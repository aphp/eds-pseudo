import spacy
from pytest import fixture
from spacy.language import Language


@fixture()
def blank_nlp() -> Language:
    nlp = spacy.blank("eds")
    nlp.add_pipe("eds.remove-lowercase", name="remove-lowercase")
    nlp.add_pipe("eds.accents", name="accents")
    nlp.add_pipe("eds.sentences")
    return nlp


@fixture()
def nlp(blank_nlp) -> Language:
    blank_nlp.add_pipe("pseudonymisation-dates", name="pseudonymisation-dates")
    blank_nlp.add_pipe(
        "pseudonymisation-rules",
        name="pseudonymisation-rules",
        config={"pattern_keys": ["PHONE", "EMAIL", "SSN", "PERSON"]},
    )
    blank_nlp.add_pipe("pseudonymisation-addresses", name="pseudonymisation-addresses")
    blank_nlp.add_pipe("structured-data-matcher", name="structured-data-matcher")
    blank_nlp.add_pipe("clean-entities")
    return blank_nlp
