from spacy.language import Language

examples = [
    ("le 06/02/2020", dict(day="06", month="02", year="2020"), "DATE"),
    ("Le 4 aout", dict(day="4", month="aout"), "DATE"),
    ("le 0 6 1 2 2 0 2 2", dict(day="0 6", month="1 2", year="2 0 2 2"), "DATE"),
    ("né le 06/02/1993", dict(day="06", month="02", year="1993"), "DATE_NAISSANCE"),
]


def test_dates(nlp: Language):
    for example, date_string, label in examples:
        doc = nlp(example)
        assert doc.ents
        assert doc.ents[0]._.date_string.dict(exclude_none=True) == date_string
        assert doc.ents[0].label_ == label
