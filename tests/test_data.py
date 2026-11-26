from datetime import date
from pathlib import Path

import pytest
from confit import Config

import edsnlp
from eds_pseudo.augment import format_date, format_name, generate_records, load_values
from eds_pseudo.cli import prepare_data
from eds_pseudo.data import Dict2DocConverter, normalize_record


def write_records(path, records):
    edsnlp.data.write_json(records, path, lines=True, overwrite=True)


def test_public_templates_separate_list_items_by_newlines():
    # Since confit keeps the quotes of a double-quoted YAML string ("\n" gives three characters), the config writes the
    # separators with single quotes, which confit reads as a newline
    config = Config.from_disk(Path(__file__).parents[1] / "configs/config.yml")
    templates = config["prepare-data"]["sources"][0]["generation"]["templates"]
    assert {template["separator"] for template in templates if "separator" in template} == {"\n"}


def test_conversion_stretches_entities_to_tokens(tmp_path):
    nlp = edsnlp.blank("eds")
    record = {
        "note_text": "Contact AliceDupont demain",
        "note_id": "note-1",
        "person_id": "person-1",
        "entities": [{"start": 8, "end": 13, "label": "PERSON", "note_text": "stale"}],
    }
    doc = Dict2DocConverter(tokenizer=nlp.tokenizer)(record)
    assert doc.text == record["note_text"]
    assert doc._.note_id == "note-1"
    assert doc._.person_id == "person-1"
    assert [(span.start_char, span.end_char, span.text) for span in doc.ents] == [(8, 19, "AliceDupont")]
    assert normalize_record(record)["entities"] == [{"start": 8, "end": 13, "label": "PERSON"}]

    whitespace_doc = Dict2DocConverter(tokenizer=nlp.tokenizer)(
        {"note_text": "Alice Dupont", "entities": [{"start": 0, "end": 6, "label": "PERSON"}]}
    )
    assert [(span.start_char, span.end_char, span.text) for span in whitespace_doc.ents] == [(0, 5, "Alice")]

    # A gold and a predicted record of the same note get the same tokens, whatever their entity bounds, so that the
    # later spans keep the same token indices for the NER metrics
    text = "Hospitalisationmai 2021 : vu le 9 juillet 2020"
    gold, pred = (
        Dict2DocConverter(tokenizer=nlp.tokenizer)({"note_text": text, "entities": entities})
        for entities in (
            [{"start": 15, "end": 23, "label": "DATE"}, {"start": 32, "end": 46, "label": "DATE"}],
            [{"start": 19, "end": 23, "label": "DATE"}, {"start": 32, "end": 46, "label": "DATE"}],
        )
    )
    assert [token.text for token in gold] == [token.text for token in pred]
    assert [(span.start, span.end, span.text) for span in gold.ents] == [
        (0, 2, "Hospitalisationmai 2021"),
        (5, 8, "9 juillet 2020"),
    ]
    assert [(span.start, span.end) for span in pred.ents] == [(1, 2), (5, 8)]


def test_generation_preserves_offsets_repeated_values_and_lineage(tmp_path):
    templates = [{"name": "staff-list", "weight": 1, "min_items": 2, "max_items": 2}]
    source = {
        "note_text": "Dr Alice Alice, née le 02/03/1980",
        "note_id": "seed-1",
        "entities": [
            {"start": 3, "end": 8, "label": "PRENOM"},
            {"start": 9, "end": 14, "label": "PRENOM"},
            {"start": 23, "end": 33, "label": "DATE_NAISSANCE"},
        ],
    }
    generated = generate_records([source], 1, templates, 42, "public", excluded_hashes=["unused"])[0]
    values = [generated["note_text"][entity["start"] : entity["end"]] for entity in generated["entities"]]

    assert values[0] == values[1]
    assert values[3] == values[4]
    assert generated["parent_ids"] == ["seed-1", "seed-1"]
    assert all(values)
    # Seed dates and the 2024-10-07 date written in their format, where "mars" and "AOUT" count as full month names
    dates = {
        "1995": "2024",
        "05nov": "07oct",
        "16/09": "07/10",
        "23022018": "07102024",
        "2018-2019": "2023-2024",
        "1 au 4 mars": "4 au 7 octobre",
        "3 mars 2021": "7 octobre 2024",
        "AOUT 1961": "OCTOBRE 2024",
        "juil 2019": "oct 2024",
    }
    assert {value: format_date(value, date(2024, 10, 7)) for value in dates} == dates
    assert format_name("JEAN-PAUL", lambda: "Dupont") == "DUPONT-DUPONT"

    templates = [{"name": "negative", "generator": "note", "weight": 1, "drop_entities": True}]
    negative = generate_records([source], 1, templates, 42, "public")[0]
    assert negative["entities"] == []
    assert "Alice" not in negative["note_text"]


def test_generation_balances_template_words():
    templates = [
        {"name": "note", "weight": 0.5},
        {"name": "staff-list", "weight": 0.5, "min_items": 5, "max_items": 5},
    ]
    seed = {
        "note_text": "Dr Alice",
        "entities": [{"start": 3, "end": 8, "label": "PRENOM"}],
    }
    records = generate_records([seed], 1000, templates, 42, "public")
    tokenizer = edsnlp.blank("eds").tokenizer
    words = {
        template: sum(len(tokenizer(record["note_text"])) for record in records if record["template"] == template)
        for template in ("note", "staff-list")
    }
    assert abs(words["note"] - words["staff-list"]) < 20


def test_replacement_values_preserve_shape(tmp_path):
    values_path = tmp_path / "values.jsonl"
    write_records(
        values_path,
        [
            {"label": "HOPITAL", "value": "JVR"},
            {"label": "HOPITAL", "value": "Henri Mondor"},
        ],
    )
    values = load_values(values_path)
    assert values[("HOPITAL", ("short", 1, False))][0] == ["JVR"]
    assert values[("HOPITAL", ("long", 2, False))][0] == ["Henri Mondor"]

    people_path = tmp_path / "people.jsonl"
    write_records(people_path, [{"PRENOM": "Camille", "NOM": "Moreau"}])
    templates = [{"name": "identity-field", "generator": "note", "weight": 1, "seed_templates": ["identity-field"]}]
    record = {
        "note_text": "Nom DURAND Prénom Alice-Paul",
        "template": "identity-field",
        "entities": [
            {"start": 4, "end": 10, "label": "NOM"},
            {"start": 18, "end": 28, "label": "PRENOM"},
        ],
    }
    generated = generate_records([record], 1, templates, 42, "public", people_path=people_path)[0]
    assert [generated["note_text"][item["start"] : item["end"]] for item in generated["entities"]] == [
        "MOREAU",
        "Camille",
    ]


def test_prepare_data_keeps_streams_separate(tmp_path):
    original = tmp_path / "original.jsonl"
    real = tmp_path / "real.jsonl"
    synthetic = tmp_path / "synthetic.jsonl"
    reserved = tmp_path / "reserved.jsonl"
    write_records(original, [{"note_text": "Original Alice", "entities": [], "source_id": "o1"}])
    write_records(
        real,
        [
            {"note_text": "Real one", "entities": [], "source_id": "r1"},
            {"note_text": "Real two", "entities": [], "source_id": "r2"},
        ],
    )
    write_records(
        synthetic,
        [
            {
                "note_text": "Generated Alice",
                "entities": [{"start": 10, "end": 15, "label": "PRENOM"}],
                "source_id": "s1",
                "split": "train",
            },
            {
                "note_text": "Generated Bob",
                "entities": [{"start": 10, "end": 13, "label": "PRENOM"}],
                "source_id": "s2",
                "split": "dev",
            },
        ],
    )
    write_records(reserved, [{"note_text": "Reserved text", "entities": [], "source_id": "q1"}])

    info = prepare_data(
        sources=[
            {"path": original, "source": "original", "kind": "original", "split": "train"},
            {"path": real, "source": "real-short", "kind": "real"},
            {
                "path": synthetic,
                "source": "synthetic-short",
                "kind": "synthetic",
                "generation": {"templates": [{"name": "note", "weight": 1}], "target_words": {"train": 1, "dev": 1}},
            },
        ],
        reserved_sources=[{"path": reserved, "source": "real-short-test"}],
        output_dir=tmp_path / "prepared",
        seed=13,
    )

    output_names = {path["path"].rsplit("/", 1)[-1] for path in info["outputs"]}
    assert "original-train.jsonl" in output_names
    assert "synthetic-short-train.jsonl" in output_names
    assert "synthetic-short-dev.jsonl" in output_names
    assert "real-short-test-test.jsonl" in output_names
    assert not any("combined" in name for name in output_names)
    synthetic_train = edsnlp.data.read_json(tmp_path / "prepared/synthetic-short-train.jsonl")
    assert all(record["split"] == "train" for record in synthetic_train)


@pytest.mark.parametrize(
    "train_record,reserved_record,message",
    [
        (
            {"note_text": "same text", "entities": [], "source_id": "train"},
            {"note_text": " SAME   TEXT ", "entities": [], "source_id": "reserved"},
            "Reserved text",
        ),
        (
            {"note_text": "training text", "entities": [], "source_id": "same-group"},
            {"note_text": "reserved text", "entities": [], "source_id": "same-group"},
            "Reserved text or group",
        ),
    ],
)
def test_prepare_data_rejects_reserved_leakage(tmp_path, train_record, reserved_record, message):
    train_path = tmp_path / "train.jsonl"
    reserved_path = tmp_path / "reserved.jsonl"
    write_records(train_path, [train_record])
    write_records(reserved_path, [reserved_record])
    with pytest.raises(ValueError, match=message):
        prepare_data(
            sources=[{"path": train_path, "kind": "real", "split": "train"}],
            reserved_sources=[{"path": reserved_path}],
            output_dir=tmp_path / "prepared",
        )


def test_prepare_data_checks_leakage_sources(tmp_path):
    reserved = tmp_path / "reserved.jsonl"
    original = tmp_path / "original.jsonl"
    write_records(reserved, [{"note_text": "Alice", "entities": []}])
    write_records(original, [{"note_text": " Alice  ", "entities": []}])

    with pytest.raises(ValueError, match="leakage check source"):
        prepare_data(
            sources=[],
            reserved_sources=[{"path": reserved}],
            leakage_check_sources=[{"path": original}],
            output_dir=tmp_path / "prepared",
        )


def test_prepare_data_rejects_cross_split_duplicates_and_groups(tmp_path):
    train_path = tmp_path / "train.jsonl"
    dev_path = tmp_path / "dev.jsonl"
    write_records(train_path, [{"note_text": "Duplicate text", "entities": [], "source_id": "train-group"}])
    write_records(dev_path, [{"note_text": " duplicate  text ", "entities": [], "source_id": "dev-group"}])
    with pytest.raises(ValueError, match="The same text is in two splits"):
        prepare_data(
            sources=[
                {"path": train_path, "kind": "real", "split": "train"},
                {"path": dev_path, "kind": "real", "split": "dev"},
            ],
            output_dir=tmp_path / "prepared",
        )

    write_records(dev_path, [{"note_text": "Different text", "entities": [], "source_id": "train-group"}])
    with pytest.raises(ValueError, match="is in both train and dev"):
        prepare_data(
            sources=[
                {"path": train_path, "kind": "real", "split": "train"},
                {"path": dev_path, "kind": "real", "split": "dev"},
            ],
            output_dir=tmp_path / "prepared-groups",
        )


def test_invalid_entity_boundaries_fail(tmp_path):
    record = {"note_text": "short", "entities": [{"start": 0, "end": 9, "label": "PERSON"}]}
    with pytest.raises(ValueError, match="Invalid entity boundary 0:9"):
        Dict2DocConverter(tokenizer=edsnlp.blank("eds").tokenizer)(record)
