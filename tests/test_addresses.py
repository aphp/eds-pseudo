def test_addresses_1(nlp):
    text = "Elle habite au 2 rue de la paix 75002 PARIS"
    doc = nlp(text)
    assert len(doc.spans["STREET_ADDRESS"]) == 1
    assert len(doc.spans["CITY"]) == 1
    assert len(doc.spans["ZIP_CODE"]) == 1
    assert doc.spans["STREET_ADDRESS"][0].text == "2 rue de la paix"
    assert doc.spans["CITY"][0].text == "PARIS"
    assert doc.spans["ZIP_CODE"][0].text == "75002"
    assert doc.spans["STREET_ADDRESS"][0].label_ == "STREET_ADDRESS"
    assert doc.spans["CITY"][0].label_ == "CITY"
    assert doc.spans["ZIP_CODE"][0].label_ == "ZIP_CODE"


def test_addresses_2(nlp):
    text = "Il ira à BD FLEURY 12eme Paris, en septembre 2020."
    doc = nlp(text)
    assert len(doc.spans["STREET_ADDRESS"]) == 1
    assert len(doc.spans["CITY"]) == 1
    assert len(doc.spans["ZIP_CODE"]) == 1
    assert doc.spans["STREET_ADDRESS"][0].text == "BD FLEURY"
    assert doc.spans["CITY"][0].text == "Paris"
    assert doc.spans["ZIP_CODE"][0].text == "12eme"


def test_addresses_3(nlp):
    text = "Vit 12 bis avenue louis philippe 16 - 75012 Paris"
    doc = nlp(text)
    assert len(doc.spans["STREET_ADDRESS"]) == 1
    assert len(doc.spans["CITY"]) == 1
    assert len(doc.spans["ZIP_CODE"]) == 1
    assert doc.spans["STREET_ADDRESS"][0].text == "12 bis avenue louis philippe 16"
    assert doc.spans["CITY"][0].text == "Paris"
    assert doc.spans["ZIP_CODE"][0].text == "75012"
