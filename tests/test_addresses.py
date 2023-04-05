def test_addresses_1(nlp):
    text = "Elle habite au 2 rue de la paix 75002 PARIS"
    doc = nlp(text)
    assert len(doc.spans["ADRESSE"]) == 1
    assert len(doc.spans["VILLE"]) == 1
    assert len(doc.spans["ZIP"]) == 1
    assert doc.spans["ADRESSE"][0].text == "2 rue de la paix"
    assert doc.spans["VILLE"][0].text == "PARIS"
    assert doc.spans["ZIP"][0].text == "75002"
    assert doc.spans["ADRESSE"][0].label_ == "ADRESSE"
    assert doc.spans["VILLE"][0].label_ == "VILLE"
    assert doc.spans["ZIP"][0].label_ == "ZIP"


def test_addresses_2(nlp):
    text = "Il ira à BD FLEURY 12eme Paris, en septembre 2020."
    doc = nlp(text)
    assert len(doc.spans["ADRESSE"]) == 1
    assert len(doc.spans["VILLE"]) == 1
    assert len(doc.spans["ZIP"]) == 1
    assert doc.spans["ADRESSE"][0].text == "BD FLEURY"
    assert doc.spans["VILLE"][0].text == "Paris"
    assert doc.spans["ZIP"][0].text == "12eme"


def test_addresses_3(nlp):
    text = "Vit 12 bis avenue louis philippe 16 - 75012 Paris"
    doc = nlp(text)
    assert len(doc.spans["ADRESSE"]) == 1
    assert len(doc.spans["VILLE"]) == 1
    assert len(doc.spans["ZIP"]) == 1
    assert doc.spans["ADRESSE"][0].text == "12 bis avenue louis philippe 16"
    assert doc.spans["VILLE"][0].text == "Paris"
    assert doc.spans["ZIP"][0].text == "75012"
