from typing import List

from edsnlp.utils.examples import parse_example
from pytest import fixture, mark
from spacy.language import Language
from spacy.tokens import Doc, DocBin

from eds_pseudonymisation.utils.evaluation import evaluate

examples = [
    (
        "Le patient s'appelle <ent label=NOM>Jean-Michel</ent>",
        dict(PRENOM="Jean-Michel"),
    ),
    (
        "Le père du patient s'appelle <ent label=PRENOM>Antoine</ent>",
        dict(PRENOM="Jean-Michel"),
    ),
]


@fixture
def docs(nlp: Language) -> List[Doc]:

    result = []

    for example, structured_data in examples:
        text, entities = parse_example(example)

        doc: Doc = nlp(text)
        doc._.structured_data = structured_data

        doc.ents = [
            doc.char_span(
                entity.start_char,
                entity.end_char,
                label=entity.modifiers[0].value,
            )
            for entity in entities
        ]

        result.append(doc)

    return result


@mark.parametrize("matcher", ["regex", "phrase"])
def test_evaluation(nlp: Language, docs: List[Doc], matcher: str):

    db = DocBin(store_user_data=True)

    for doc in docs:
        db.add(doc)

    nlp.add_pipe("structured-data-matcher", config=dict(matcher=matcher, attr="TEXT"))

    score = evaluate(nlp, db, merges=dict(NOM="PRENOM"))

    assert score["ents_p"] == 1.0
    assert score["ents_r"] == 0.5
