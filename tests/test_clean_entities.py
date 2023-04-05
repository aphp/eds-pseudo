from spacy.tokens import Span


def test_clean_entities(blank_nlp):
    blank_nlp.add_pipe("clean-entities")
    text = "Elle habite au 2 rue de la paix    75002, PARIS"
    doc = blank_nlp.make_doc(text)
    doc.ents = [Span(doc, 3, 9, "ADRESSE"), Span(doc, 10, 11, "ZIP")]
    assert doc.ents[0].text == "2 rue de la paix    "
    assert doc.ents[1].text == ","
    doc = blank_nlp(doc)
    assert len(doc.ents) == 1
    assert doc.ents[0].text == "2 rue de la paix"
