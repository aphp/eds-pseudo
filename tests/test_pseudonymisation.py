examples = [
    "mail : test@example.com",
    "Le patient vit au \n71 rue Magelan, Paris 750012",
]


def test_pseudonymisation(nlp):
    nlp.add_pipe("eds.remove-lowercase")
    nlp.add_pipe("eds.accents")
    nlp.add_pipe("pseudonymisation-dates")
    nlp.add_pipe("pseudonymisation-rules")
    nlp.add_pipe("pseudonymisation-addresses")
    nlp.add_pipe("structured-data-matcher")

    for example in examples:
        doc = nlp(example)
        assert doc.ents
