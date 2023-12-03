# ruff: noqa: E501
from edsnlp.utils.examples import parse_example

examples = [
    # fmt: off
    'mail : <ent label="MAIL">test@example.com</ent>',
    'M. <ent label="PRENOM">Gaston</ent> <ent label="NOM">LAGAFFE</ent>, né le <ent label="DATE_NAISSANCE">06/02/1993</ent>, est suivit par le Dr. <ent label="NOM">Dupont</ent>',
    'Veuillez contacter le <ent label="TEL">06 01 02 03 04</ent>',
    'Consultation\nNumero d\'examen: <ent label="NDA">123456789</ent>\nLe patient est venu ce jour pour consultation.',
    'Consultation\n<ent label="SECU">253072B07300123</ent>\nLe patient est venu ce jour pour consultation.',
    "Le patient est venu ce jour pour consultation.",
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
