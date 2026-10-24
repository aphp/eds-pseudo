def test_absolute_dates_and_normalization(blank_nlp):
    blank_nlp.add_pipe("eds_pseudo.dates")
    blank_nlp.add_pipe("eds_pseudo.dates_normalizer", config={"span_getter": "ents"})

    doc = blank_nlp("Né le 06/02/1993, revu le 0 6 1 2 2 0 2 2 puis ce jour")

    assert [(span.text, span.label_) for span in doc.ents] == [
        ("06/02/1993", "DATE_NAISSANCE"),
        ("0 6 1 2 2 0 2 2", "DATE"),
    ]
    assert doc.ents[0]._.date_string.model_dump(exclude_none=True) == {
        "day": "06",
        "month": "02",
        "year": "1993",
    }
    assert str(doc.ents[0]._.date) == "1993-02-06"
    assert doc.ents[0]._.date_format == "%d/%m/%Y"
