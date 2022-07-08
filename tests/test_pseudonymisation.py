examples = [
    "mail : test@example.com",
    "Le patient vit à Paris",
]


def test_pseudonymisation(nlp):
    nlp.add_pipe("pseudonymisation-rules")

    for example in examples:
        doc = nlp(example)
        assert doc.ents
