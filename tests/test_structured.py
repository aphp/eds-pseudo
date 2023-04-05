def test_structured(blank_nlp):
    text = "Marie Élise (marie.elise@mail.com) habite au 2 rue de Jouvence, Paris"
    context = {
        "ADRESSE": ["2 rue de Jouvence"],
        "VILLE": ["Paris"],
        "EMAIL": ["marie.elise@mail.com"],
        "NOM_NAISS": ["Élise"],
        "PRENOM": ["Marie"],
    }
    blank_nlp.add_pipe("structured-data-matcher")
    doc = blank_nlp.make_doc(text)
    doc._.context = context
    doc = blank_nlp(doc)
    texts = {ent.text for ent in doc.ents}
    assert texts == {
        "Marie",
        "Élise",
        "2 rue de Jouvence",
        "Paris",
        "marie.elise@mail.com",
    }
