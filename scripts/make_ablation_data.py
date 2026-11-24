import hashlib
import json
from pathlib import Path
from typing import Dict, List

from confit import Cli

import edsnlp
from eds_pseudo.data import file_hash, text_hash

app = Cli(pretty_exceptions_show_locals=False)


def note_key(record: dict) -> str:
    """
    Id shared by the PDF extractions of one note, from a `note_id` like `train/edspdf/<id>`
    """
    return record["note_id"].rsplit("/", 1)[-1]


@app.command(name="make")
def make(
    original_train: Path,
    original_test: Path,
    real_short_train: Path,
    real_short_dev: Path,
    real_short_test: Path,
    output_dir: Path,
    sizes: List[int] = [10, 30, 100, 300, 1000],
    note_types: List[str] = ["LT-ORDO", "CR-CONS", "CR-IMAGE", "CR-URGE"],
) -> Dict[str, dict]:
    """
    Write the filtered views of the gold files used by the ablation experiments, without changing the sources.

    Original notes have `note_id` values like `train/edspdf/<id>`, and some notes exist in both PDF extractions. The
    files are:

    - `original-train-notes-<size>.jsonl`: nested subsets of `size` training documents (the 10 documents are in the
      30 documents, which are in the 100 documents, and so on)
    - `original-train-without-<type>.jsonl` and `original-test-<type>.jsonl`: training without one note type, test
      notes of that type only
    - `original-test-pdfbox.jsonl` and `original-test-edspdf.jsonl`: the test notes extracted by both tools
    - `real-short-test-answer-unseen.jsonl`: short field test rows whose answer is in no training or development row

    Counts and SHA256 of every written file go to `ablation-data-info.json`, read by experiment configs and the
    analysis.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    train = list(edsnlp.data.read_json(original_train))
    test = list(edsnlp.data.read_json(original_test))
    # We order the training documents by a hash of their id, so that a subset takes the first `size` documents and
    # holds the smaller subsets
    ordered = sorted(train, key=lambda r: hashlib.sha256(r["note_id"].encode()).digest())
    files = {f"original-train-notes-{size}": ordered[:size] for size in sizes}
    for note_type in note_types:
        slug = note_type.lower()
        files[f"original-train-without-{slug}"] = [r for r in train if r["note_class_source_value"] != note_type]
        files[f"original-test-{slug}"] = [r for r in test if r["note_class_source_value"] == note_type]
    extractions = {e: {note_key(r) for r in test if r["note_id"].split("/")[1] == e} for e in ("pdfbox", "edspdf")}
    both = extractions["pdfbox"] & extractions["edspdf"]
    for extraction in ("pdfbox", "edspdf"):
        files[f"original-test-{extraction}"] = [
            r for r in test if r["note_id"].split("/")[1] == extraction and note_key(r) in both
        ]
    # Answer-unseen rows, whose answer text appears in no short field training or development row
    seen = {
        text_hash(row.get("original_text") or row["note_text"])
        for path in (real_short_train, real_short_dev)
        for row in edsnlp.data.read_json(path)
    }
    files["real-short-test-answer-unseen"] = [
        row
        for row in edsnlp.data.read_json(real_short_test)
        if text_hash(row.get("original_text") or row["note_text"]) not in seen
    ]
    info = {}
    for name, records in files.items():
        path = output_dir / f"{name}.jsonl"
        edsnlp.data.write_json(records, path, lines=True, overwrite=True)
        info[name] = {"records": len(records), "sha256": file_hash(path)}
    (output_dir / "ablation-data-info.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    return info


if __name__ == "__main__":
    app()
