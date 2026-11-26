"""
Write the predictions of a packaged model on JSONL records, so that `eds-pseudo evaluate` can score them through the
`predictions` of an evaluation.

We use it for model 0.4 of `AP-HP/eds-pseudo-public` (experiment `public-0.4`): its package holds its own copy of
`eds_pseudo` 0.4, which can't share a process with eds-pseudo 1.0, so this script only imports EDS-NLP and the training
job runs it apart from the evaluation.
"""

from pathlib import Path
from typing import Dict, List, Optional

from confit import Cli
from spacy.tokens import Doc

import edsnlp

app = Cli(pretty_exceptions_show_locals=False)


@app.command(name="predict")
def predict(
    model: str,
    inputs: Dict[str, Path],
    output_dir: Path,
    revision: Optional[str] = None,
    fields: List[str] = ["note_id", "context", "note_datetime", "note_class_source_value"],
    batch_size: int = 32,
) -> None:
    """
    Predict the entities of each input file and write them to `<output_dir>/<name>-predictions.jsonl`, with the
    `note_id` and `note_text` of the input records in the same order.

    Parameters
    ----------
    model
        Path or Hugging Face name of the packaged model
    inputs
        Gold JSONL files (`note_text`, `entities` and record fields) keyed by evaluation name, e.g. `original-test`
    revision
        Hugging Face revision of `model`, e.g. the commit of a release
    fields
        Record fields that we copy to `doc._` before running the model, since its rules read them (e.g. `context`)
    """
    import torch

    nlp = edsnlp.load(model, revision=revision).to("cuda" if torch.cuda.is_available() else "cpu")
    for field in [*fields, "input_text"]:
        if not Doc.has_extension(field):
            Doc.set_extension(field, default=None)

    def make_doc(record):
        doc = nlp.make_doc(record["note_text"])
        for field in fields:
            setattr(doc._, field, record.get(field))
        doc._.input_text = record["note_text"]
        return doc

    def to_record(doc):
        # The tokenizer of model 0.4 writes non-breaking spaces as spaces, so we keep the input text, on which the
        # entity offsets stay valid as long as the doc keeps its length
        if len(doc.text) != len(doc._.input_text):
            raise ValueError(f"The model changed the length of {doc._.note_id}, so its offsets don't fit its text")
        entities = [{"start": ent.start_char, "end": ent.end_char, "label": ent.label_} for ent in doc.ents]
        return {"note_id": doc._.note_id, "note_text": doc._.input_text, "entities": entities}

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, path in inputs.items():
        docs = edsnlp.data.read_json(path).map(make_doc).map_pipeline(nlp).set_processing(batch_size=batch_size)
        edsnlp.data.write_json(
            docs, output_dir / f"{name}-predictions.jsonl", converter=to_record, lines=True, overwrite=True
        )


if __name__ == "__main__":
    app()
