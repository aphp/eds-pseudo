import json

from transformers import BertConfig, BertModel, BertTokenizer

import edsnlp
from eds_pseudo.cli import evaluate, experiments, infer
from eds_pseudo.data import Dict2DocConverter
from edsnlp.metrics.ner import NerExactMetric
from edsnlp.training.trainer import GenericScorer, TrainingData, train


def test_tiny_train_evaluate_and_infer_cycle(tmp_path, monkeypatch):
    model_dir = tmp_path / "tiny-bert"
    model_dir.mkdir()
    (model_dir / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nalice\nbob\nworks\n")
    BertTokenizer(vocab_file=str(model_dir / "vocab.txt")).save_pretrained(model_dir)
    BertModel(
        BertConfig(
            vocab_size=8,
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=1,
            num_hidden_layers=1,
        )
    ).save_pretrained(model_dir)

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        "eds.ner_crf",
        config={
            "mode": "independent",
            "window": 1,
            "target_span_getter": "gold_spans",
            "span_setter": "ents",
            "embedding": {
                "@factory": "eds.text_cnn",
                "kernel_sizes": [3],
                "embedding": {"@factory": "eds.transformer", "model": str(model_dir)},
            },
        },
    )
    nlp.add_pipe("eds_pseudo.context", before="ner_crf", config={"span_setter": "pseudo-rb"})
    converter = Dict2DocConverter(span_setter=["ents", "gold_spans"])
    records = [
        {"note_text": "alice works", "entities": [{"start": 0, "end": 5, "label": "PERSON"}]},
        {"note_text": "bob works", "entities": [{"start": 0, "end": 3, "label": "PERSON"}]},
    ]
    train_stream = edsnlp.data.from_iterable(records).map(converter)
    val_stream = edsnlp.data.from_iterable(records).map(converter)
    scorer = GenericScorer(speed=False, exact=NerExactMetric(span_getter="ents"))
    saved_models = []
    to_disk = edsnlp.Pipeline.to_disk

    def record_save(pipeline, path, *args, **kwargs):
        saved_models.append(path)
        return to_disk(pipeline, path, *args, **kwargs)

    monkeypatch.setattr(edsnlp.Pipeline, "to_disk", record_save)
    train(
        nlp=nlp,
        train_data=[TrainingData(data=train_stream, batch_size="2 docs", shuffle=False, pipe_names=["ner_crf"])],
        val_data=[val_stream],
        scorer=scorer,
        max_steps=1,
        validation_interval=1,
        checkpoint_interval=1,
        output_dir=tmp_path / "training",
        output_model_dir=tmp_path / "model",
        cpu=True,
        logger=False,
    )
    assert saved_models[-1] == tmp_path / "model"
    restored = edsnlp.load(tmp_path / "model")
    assert all(
        (parameter == dict(restored.named_parameters())[name]).all() for name, parameter in nlp.named_parameters()
    )

    eval_stream = edsnlp.data.from_iterable(records).map(converter)
    run_dir = tmp_path / "runs" / "tiny-42"
    metrics = evaluate(nlp=nlp, evaluations={"dev": {"data": eval_stream, "scorer": scorer}}, output_dir=run_dir)["dev"]
    assert "exact" in metrics
    assert json.loads((run_dir / "dev-metrics.json").read_text()) == metrics
    table = experiments(runs_dir=tmp_path / "runs", output_path=tmp_path / "experiments.csv")
    assert table[0]["run"] == "tiny-42" and table[0]["dev/exact/micro/f"] == metrics["exact"]["micro"]["f"]

    input_path = tmp_path / "input.jsonl"
    record = {"note_id": "n1", "note_text": "alice works", "entities": [{"start": 0, "end": 5, "label": "PRENOM"}]}
    edsnlp.data.write_json([record], input_path, lines=True)
    infer(nlp=nlp, input_path=input_path, output_path=tmp_path / "output.jsonl", batch_size=1)
    [prediction] = edsnlp.data.read_json(tmp_path / "output.jsonl")
    assert prediction["note_id"] == "n1" and prediction["note_text"] == "alice works"
