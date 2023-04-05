import spacy
from spacy.tokens import DocBin

from eds_pseudonymisation.corpus_reader import PseudoCorpus


def test_corpus_reader(tmp_path):
    """
    We test the corpus reader by creating a Doc, storing it in a DocBin, and then
    reading it back in.

    Returns
    -------

    """

    nlp = spacy.blank("eds")
    nlp.add_pipe("eds.sentences")

    doc1 = nlp("This is a test. This is another test.")
    doc1._.note_id = "doc-1"
    doc1._.note_class_source_value = "CONSULTATION"
    doc1._.context = {}
    doc1._.split = "train"

    doc2 = nlp("Le patient est né le 02/03/2000. Il est hospitalisé depuis 2 jours.")
    doc2.ents = [doc2.char_span(21, 31, label="DATE_NAISSANCE")]
    doc2.spans["DATE_NAISSANCE"] = list(doc2.ents)
    doc2._.note_id = "doc-2"
    doc2._.note_class_source_value = "URGENCE"
    doc2._.context = {"DATE_NAISSANCE": "02/03/2000"}
    doc2._.split = "train"

    # Long document, only one sentence
    doc3 = nlp(
        "Le patient mange des pates depuis le début du confinement, "
        "il est donc un peu ballonné, mais pense revenir à un régime plus "
        "équilibré en mangeant des légumes et des fruits."
    )
    doc3._.note_class_source_value = "CONSULTATION"
    doc4 = nlp("")

    db = DocBin(store_user_data=True, docs=[doc1, doc2, doc3, doc4])
    db.to_disk(tmp_path / "test.spacy")

    read_db = list(DocBin().from_disk(tmp_path / "test.spacy").get_docs(nlp.vocab))
    assert len(read_db) == 4

    corpus = PseudoCorpus(tmp_path / "test.spacy", max_length=12)
    examples = list(corpus(nlp))
    assert len(examples) == 6
    assert examples[1].reference._.note_id == "doc-2"
    assert examples[2].reference._.note_id == "doc-2"
    assert [eg.reference.text for eg in examples] == [
        "This is a test. This is another test.",
        "Le patient est né le 02/03/2000. Il ",
        "est hospitalisé depuis 2 jours.",
        "Le patient mange des pates depuis le début du confinement, il ",
        "est donc un peu ballonné, mais pense revenir à un régime ",
        "plus équilibré en mangeant des légumes et des fruits.",
    ]

    corpus = PseudoCorpus(
        tmp_path / "test.spacy",
        max_length=1000,
        shuffle=True,
        limit=1,
        filter_expr='doc._.note_class_source_value == "CONSULTATION"',
    )
    examples = list(corpus(nlp))
    assert len(examples) == 1
