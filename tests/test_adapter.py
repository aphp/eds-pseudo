import json

import edsnlp

from eds_pseudonymisation.adapter import pseudo_dataset


def test_corpus_reader(tmp_path):
    """
    We test the corpus reader by creating a Doc, storing it in a DocBin, and then
    reading it back in.

    Returns
    -------

    """
    jsonl_data = [
        {
            "note_id": "doc-1",
            "note_class_source_value": "CONSULTATION",
            "note_text": "This is a sentence of exactly 12 words used in test. "
            "This is another test.",
            "entities": [],
            "context": {},
        },
        {
            "note_id": "doc-2",
            "note_class_source_value": "URGENCE",
            "note_text": "Le patient est né le 02/03/2000. "
            "Il est hospitalisé depuis 2 jours.",
            "entities": [
                {"start": 21, "end": 31, "label": "DATE_NAISSANCE"},
            ],
            "context": {"DATE_NAISSANCE": "02/03/2000"},
        },
        {
            "note_id": "doc-3",
            "note_class_source_value": "CONSULTATION",
            "note_text": "Le patient mange des pates depuis le début du confinement, "
            "il est donc un peu ballonné, mais pense revenir à un régime plus "
            "équilibré en mangeant des légumes et des fruits.",
            "entities": [],
            "context": {},
        },
        {
            "note_id": "doc-4",
            "note_class_source_value": "CONSULTATION",
            "note_text": "",
            "entities": [],
            "context": {},
        },
    ]

    with open(tmp_path / "test.jsonl", "w") as f:
        for line in jsonl_data:
            f.write(json.dumps(line) + "\n")

    # Long document, only one sentence
    nlp = edsnlp.blank("eds")
    nlp.add_pipe("eds.normalizer", name="normalizer")
    nlp.add_pipe("eds.sentences", name="sentencizer")
    docs = list(
        pseudo_dataset(
            tmp_path / "test.jsonl", max_length=12, multi_sentence=True, randomize=False
        )(nlp)
    )
    assert [d.text for d in docs] == [
        "This is a sentence of exactly 12 words used in test. ",
        "This is another test.",
        "Le patient est né le 02/03/2000. Il ",
        "est hospitalisé depuis 2 jours.",
        "Le patient mange des pates depuis le début du confinement, il ",
        "est donc un peu ballonné, mais pense revenir à un régime ",
        "plus équilibré en mangeant des légumes et des fruits.",
    ]

    full_docs = list(pseudo_dataset(tmp_path / "test.jsonl", max_length=0)(nlp))
    assert [d.text for d in full_docs] == [
        "This is a sentence of exactly 12 words used in test. This is another test.",
        "Le patient est né le 02/03/2000. Il est hospitalisé depuis 2 jours.",
        "Le patient mange des pates depuis le début du confinement, il est donc un "
        "peu ballonné, mais pense revenir à un régime plus équilibré en mangeant des "
        "légumes et des fruits.",
    ]

    consultation_docs = list(
        pseudo_dataset(
            tmp_path / "test.jsonl",
            max_length=1000,
            multi_sentence=False,
            limit=1,
            filter_expr='doc._.note_class_source_value == "CONSULTATION"',
        )(nlp)
    )
    assert len(consultation_docs) == 1
