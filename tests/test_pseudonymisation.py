# ruff: noqa: E501
from edsnlp.utils.examples import parse_example

examples = [
    # fmt: off
    'mail : <ent label="EMAIL">test@example.com</ent>',
    'M. <ent label="FIRSTNAME">Gaston</ent> <ent label="LASTNAME">LAGAFFE</ent>, né le <ent label="BIRTHDATE">06/02/1993</ent>, est suivit par le Dr. <ent label="LASTNAME">Dupont</ent>',
    'Veuillez contacter le <ent label="PHONE">06 01 02 03 04</ent>',
    'Consultation\n<ent label="SSN">253072B07300123</ent>\nLe patient est venu ce jour pour consultation.',
    'Le patient est venu ce jour pour consultation.',
    # fmt: on
]


def test_pseudonymisation(nlp):
    for example in examples:
        text, expected_entities = parse_example(example=example)
        doc = nlp(text)
        expected_ents = [
            doc.char_span(
                ent.start_char,
                ent.end_char,
                label=next(m.value for m in ent.modifiers if m.key == "label"),
            )
            for ent in expected_entities
        ]
        assert set(doc.ents) == set(expected_ents)
