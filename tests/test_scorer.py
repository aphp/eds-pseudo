from edsnlp.scorers.ner import create_ner_exact_scorer
from spacy.tokens import Span

from eds_pseudonymisation.scorer import PseudoScorer, create_pseudo_ner_redact_scorer


def test_scorer(nlp):
    ner_scorer = create_ner_exact_scorer("pseudo-rb")
    redact_scorer = create_pseudo_ner_redact_scorer("pseudo-rb")
    scorer = PseudoScorer(
        ner=ner_scorer,
        redact=redact_scorer,
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
    scores = scorer(nlp, [doc, nlp.make_doc("")])
    print(nlp(doc).spans)

    assert scores["ner"]["ents_p"] == 2 / 3
    assert scores["ner"]["ents_r"] == 2 / 4
    assert scores["ner"]["ents_f"] == 4 / 7
    assert scores["ner"]["support"] == 4
    assert scores["redact"]["redact"] == 7 / 8
    assert scores["redact"]["redact_full"] == 0.5
    assert scores["speed"]["wps"] > 0
    assert scores["speed"]["dps"] > 0
