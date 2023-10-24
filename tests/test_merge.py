def test_merge(blank_nlp):
    blank_nlp.add_pipe(
        "eds_pseudo.simple_rules",
        name="pseudo1",
        config={
            "pattern_keys": ["PERSON"],
            "span_setter": "pseudo1",
        },
    )
    blank_nlp.add_pipe(
        "eds_pseudo.simple_rules",
        name="pseudo2",
        config={
            "pattern_keys": ["MAIL"],
            "span_setter": "pseudo2",
        },
    )
    blank_nlp.add_pipe(
        "eds_pseudo.merge",
        name="merge",
        config={
            "span_getter": ["pseudo1", "pseudo2"],
            "span_setter": "pseudo3",
        },
    )
    text = "Dr. Juan a pour mail don.juan@caramail.com"
    doc = blank_nlp(text)
    assert {ent.text for ent in doc.spans["pseudo3"]} == {
        "Juan",
        "don.juan@caramail.com",
    }
