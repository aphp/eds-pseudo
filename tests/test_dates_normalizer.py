import pytest

import eds_pseudo.pipes.dates_normalizer.dates_normalizer  # noqa: F401
import edsnlp
from edsnlp.utils.examples import parse_example

examples = [
    "<ent norm='2014-01-06' fmt='%Y, %B, %-d'>deux mille quatorze, janvier, 6</ent>,",
    "<ent norm='1989-10-29' fmt='%y/%m/%d'>89/10/29</ent>,",
    "<ent norm='2011-12-08' fmt='%-d%b%y'>8dec11</ent>,",
    "<ent norm='2033-12-08' fmt='%-d%b%y'>8dec33</ent>,",
    "<ent norm='1989-10-29' fmt='%Y-%m-%d'>1989-10-29</ent>,",
    "<ent norm='2003-??-??' fmt='jeudi ?? %Y'>jeudi 18 20Û3</ent>,",
    "<ent norm='????-03-12' fmt='%A %-d %B'>mardi douze mars</ent>,",
    "<ent norm='2003-02-01' fmt='%d%m%y'>010203</ent>,",
    "<ent norm='2012-12-12' fmt='%d%m%y'>121212</ent>,",
    "<ent norm='2005-04-20' fmt='%d%m%y ??'>200405 10</ent>,",
    "<ent norm='1991-01-10' fmt='%d %b %Y'>10 jan quatrevingtsonzes</ent>,",
    "<ent norm='????-??-??' fmt='??????'>000000</ent>,",
    "<ent norm='????-03-??' fmt='%B'>mars</ent>,",
    "<ent norm='????-??-08' fmt='%d'>08</ent>,",
    "<ent norm='1990-??-??' fmt='%y'>90</ent>,",
    "<ent norm='????-??-??' fmt='mercredi'>mercredi</ent>,",
    "<ent norm='2003-02-01' fmt='%d%m%y????'>0102030945</ent>,",
    (
        "<ent norm='2003-02-01' fmt='dès le %d/%m/%Y ? ? ? ?'>"
        "dès le 0 1 0 2 0 3 0 9 4 5</ent>,"
    ),
    "<ent norm='2003-02-01' fmt='%-d %B %Y?'>1er fevreri 20030</ent>,",
    "<ent norm='????-09-31' fmt='%-d %b ????'>trente et un sep 4022</ent>,",
    "<ent norm='????-06-04' fmt='%-d/%-m'>4/6</ent>,",
    "<ent norm='1996-04-??' fmt='%-m/%y'>4/96</ent>,",
    "<ent norm='2010-08-04' fmt='%a %-d/%-m/%y'>lun 4/8/10</ent>,",
    "<ent norm='1958-08-03' fmt='%d/%m/%Y matin'>0 3 0 8 1 9 5 8 matin</ent>,",
    (
        "<ent norm='1998-01-10' fmt='%d %b %Y'>10 jan mil neuf "
        "cents quatrevingtdix-huit</ent>,"
    ),
    # Test dates that complete each others (only the day is missing).
    (
        "<ent norm='1978-01-10' fmt='%d'>10</ent>, "
        "<ent norm='1978-01-11' fmt='%d'>11</ent> et "
        "<ent norm='1978-01-12' fmt='%d %b %y'>12 jan 78</ent>,"
    ),
    # Test dates that complete each others (only the year is missing).
    (
        "<ent norm='1978-03-??' fmt='%B'>mars</ent>, "
        "<ent norm='1978-04-??' fmt='%B'>avril</ent> et "
        "<ent norm='1978-05-??' fmt='%B %Y'>mai soixantedix-huit</ent>,"
    ),
    (
        "<ent norm='2020-06-10' fmt='%d %B'>10 juin</ent>, "
        "<ent norm='2020-07-23' fmt='%d %B'>23 juillet</ent> et "
        "<ent norm='2020-12-13' fmt='%d %b %Y'>13 dec 2020</ent>,"
    ),
    # Test dates that could complete each others but are too far apart.
    (
        "Il est né en <ent norm='????-03-??' fmt='%B'>mars</ent>. Ensuite, en "
        "<ent norm='2023-05-??' fmt='%B %Y'>mai vingt trois</ent>, ..."
    ),
]


@pytest.fixture(scope="module")
def nlp():
    nlp = edsnlp.blank("eds")
    nlp.add_pipe("eds_pseudo.dates_normalizer")
    return nlp


@pytest.mark.parametrize("example", examples)
def test_dates_normalizer(example, nlp):
    text, entities = parse_example(example=example)

    doc = nlp.make_doc(text)
    doc.ents = [
        doc.char_span(ent.start_char, ent.end_char, label="DATE") for ent in entities
    ]
    doc = nlp(doc)
    assert len(doc.ents) == len(entities)
    for ent, parsed in zip(doc.ents, entities):
        norm = next(m.value for m in parsed.modifiers if m.key == "norm")
        date_format = next(m.value for m in parsed.modifiers if m.key == "fmt")
        assert str(ent._.date) == norm, ent
        assert str(ent._.date_format) == date_format, ent


def test_java_formatter():
    nlp = edsnlp.blank("eds")
    nlp.add_pipe("eds_pseudo.dates_normalizer", config={"format": "java"})

    example = (
        "<ent norm='2014-01-06' fmt='yyyy, MMMM\" d,  \"d'>"
        "deux mille quatorze, janvier d, ' 6</ent>,"
    )
    text, entities = parse_example(example=example)
    doc = nlp.make_doc(text)
    doc.ents = [
        doc.char_span(ent.start_char, ent.end_char, label="DATE") for ent in entities
    ]
    doc = nlp(doc)
    assert len(doc.ents) == len(entities)
    for ent, eg in zip(doc.ents, entities):
        modifiers = dict((m.key, m.value) for m in eg.modifiers)
        norm = modifiers["norm"]
        date_format = modifiers["fmt"].replace('"', "'")
        assert str(ent._.date) == norm, ent
        assert str(ent._.date_format) == date_format, ent
