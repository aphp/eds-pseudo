from spacy.tokens import Span

from eds_pseudonymisation.scorer import PseudoScorer


def test_scorer(nlp):
    scorer = PseudoScorer(
        ml_spans="pseudo-ml",
        rb_spans="pseudo-rb",
        hybrid_spans="pseudo-hybrid",
        main_mode="rb",
    )
    text = "Dr. Juan a pour mail don juan@caramail.fr vit a grigny, tel 0607080910."
    doc = nlp.make_doc(text)
    doc.spans["pseudo-rb"] = [
        Span(doc, 1, 2, "NOM"),
        Span(doc, 6, 11, "MAIL"),
        Span(doc, 13, 14, "CITY"),
        Span(doc, 16, 17, "MAIL"),  # not a mail but to test the scorer
    ]
    print(doc.spans)
    main_scores = scorer(nlp, [doc, nlp.make_doc("")])
    wps = main_scores.pop("wps")
    dps = main_scores.pop("dps")

    assert main_scores == {
        "f": 0.8,
        "full": 0.5,
        "p": 6 / 7,
        "r": 0.75,
        "redact": 0.875,
    }
    assert wps > 0 and dps > 0
