from pytest import fixture
from spacy.language import Language

import edsnlp


@fixture()
def blank_nlp() -> Language:
    nlp = edsnlp.blank("eds")
    nlp.add_pipe("eds.remove-lowercase", name="remove-lowercase")
    nlp.add_pipe("eds.accents", name="accents")
    nlp.add_pipe("eds.sentences")
    return nlp


@fixture()
def nlp(blank_nlp) -> Language:
    blank_nlp.add_pipe("eds_pseudo.dates", name="dates")
    blank_nlp.add_pipe(
        "eds_pseudo.simple_rules",
        name="rules",
        config={"pattern_keys": ["TEL", "MAIL", "SECU", "PERSON", "NDA"]},
    )
    blank_nlp.add_pipe("eds_pseudo.addresses", name="addresses")
    blank_nlp.add_pipe("eds_pseudo.context", name="context")
    blank_nlp.add_pipe("eds_pseudo.clean", name="cleaner")
    blank_nlp.add_pipe("eds_pseudo.merge", name="merge")
    return blank_nlp
