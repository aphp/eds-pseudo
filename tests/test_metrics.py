from pathlib import Path

import pandas as pd

import edsnlp
from eds_pseudo.data import Dict2DocConverter
from eds_pseudo.metrics import (
    FinalMetric,
    PopulationMetric,
    PrivacyMetric,
    RedactMetric,
    covered,
    exact,
    partial,
    span,
)
from edsnlp.training.trainer import GenericScorer


def test_document_metrics_have_uniform_gold_prediction_interface():
    gold = [{"start": 1, "end": 5, "label": "PERSON"}]
    pred = [{"start": 0, "end": 5, "label": "OTHER"}]

    assert not exact(gold, pred)
    assert not span(gold, pred)
    assert covered(gold, pred)
    assert partial(gold, pred)


def test_label_agnostic_redaction_and_grouped_privacy():
    nlp = edsnlp.blank("eds")
    converter = Dict2DocConverter(tokenizer=nlp.tokenizer)
    gold = converter(
        {
            "note_text": "Alice Dupont Paris",
            "question_id": "q1",
            "encounter_id": "e1",
            "person_id": "p1",
            "entities": [{"start": 0, "end": 12, "label": "PERSON"}, {"start": 13, "end": 18, "label": "CITY"}],
        }
    )
    pred = converter(
        {
            "note_text": "Alice Dupont Paris",
            "question_id": "q1",
            "encounter_id": "e1",
            "person_id": "p1",
            "entities": [{"start": 0, "end": 5, "label": "WRONG_LABEL"}, {"start": 13, "end": 18, "label": "DATE"}],
        }
    )

    redaction = RedactMetric(span_getter="ents")([gold], [pred])
    privacy = PrivacyMetric(span_getter="ents")([gold], [pred])
    assert (redaction["micro"]["token_r"], redaction["micro"]["full"]) == (2 / 3, 0)
    assert (redaction["PERSON"]["token_r"], redaction["PERSON"]["full"]) == (0.5, 0)
    assert (redaction["CITY"]["token_r"], redaction["CITY"]["full"]) == (1, 1)
    assert privacy["question"] == {"token_r": 2 / 3, "full": 0, "groups": 1, "documents": 1}
    assert privacy["encounter"]["groups"] == 1
    assert privacy["person"]["groups"] == 1


def test_population_metric_ignores_predictions_before_response_text(monkeypatch):
    nlp = edsnlp.blank("eds")
    converter = Dict2DocConverter(tokenizer=nlp.tokenizer)
    gold = converter(
        {
            "note_text": "Nom: aucun",
            "original_text": "aucun",
            "source": "urgence",
            "question": "Nom",
            "entities": [],
        }
    )
    pred = converter(
        {
            "note_text": "Nom: aucun",
            "entities": [{"start": 0, "end": 3, "label": "NOM"}],
        }
    )

    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda path: pd.DataFrame([{"source": "urgence", "question": "Nom", "encounter_id": "e1", "person_id": "p1"}]),
    )
    result = PopulationMetric(span_getter="ents", population_path=Path("population.parquet"))([gold], [pred])

    assert result["document"] == {"exact": 1, "span": 1, "covered": 1, "partial": 1}
    assert result["entity_f1"] == {}
    assert result["population"]["source"]["urgence"]["covered"] == {"encounter": 1, "person": 1}


def test_final_metric_scores_clean_full_pipeline_predictions():
    nlp = edsnlp.blank("eds")
    gold = Dict2DocConverter(tokenizer=nlp.tokenizer)(
        {"note_text": "Alice", "entities": [{"start": 0, "end": 5, "label": "PERSON"}]}
    )

    result = GenericScorer(speed=False, final=FinalMetric("ents", group_fields=["person_id"]))(nlp, [gold])

    assert result["final"]["exact"]["micro"]["r"] == 0
    assert result["final"]["redact"]["micro"]["full"] == 0
    assert result["final"]["privacy"]["person"]["full"] == 0
