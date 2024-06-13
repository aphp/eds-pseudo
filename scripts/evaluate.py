import json
from collections import defaultdict
from pathlib import Path
from typing import List

import pandas as pd
import spacy
import torch
from confit import Cli
from confit.utils.random import set_seed

import edsnlp
from eds_pseudo.adapter import PseudoReader
from eds_pseudo.scorer import PseudoScorer
from edsnlp.core.registries import registry

app = Cli(pretty_exceptions_show_locals=False)

BASE_DIR = Path(__file__).parent.parent


def flatten_dict(d, depth=-1, path="", current_depth=0):
    if not isinstance(d, dict) or current_depth == depth:
        return {path: d}
    return {
        k: v
        for key, val in d.items()
        for k, v in flatten_dict(
            val, depth, f"{path}/{key}" if path else key, current_depth + 1
        ).items()
    }


@app.command(name="evaluate", registry=registry)
def evaluate(
    *,
    data: PseudoReader,
    model_path: Path = BASE_DIR / "artifacts/model-last",
    dataset_name: str,
    scorer: PseudoScorer,
    data_seed: int = 42,
):
    nlp = edsnlp.load(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
    with set_seed(data_seed):
        val_docs: List[spacy.tokens.Doc] = list(data(nlp))
        metrics, per_doc = scorer(nlp, val_docs, per_doc=True)

    grouped_metrics = defaultdict(lambda: {})
    for metric in metrics:
        parts = [key.strip() for key in metric["name"].split("/")]
        value = metric["value"]
        current = grouped_metrics
        for key in parts[:-1]:
            current = current.setdefault(key, {})
        current[parts[-1]] = value

    for group, data in grouped_metrics.items():
        print(group)
        print(
            pd.DataFrame.from_dict(data, orient="index").applymap(
                lambda x: f"{x * 100:.2f}"
            )
        )
    (BASE_DIR / "artifacts/test_metrics.json").write_text(json.dumps(metrics, indent=2))
    meta = json.loads((model_path / "meta.json").read_text())
    results = meta.setdefault("results", [])
    index = next(
        (
            i
            for i, res in enumerate(results)
            if res.get("dataset", {}).get("name") == dataset_name
        ),
        None,
    )
    if index is not None:
        results.pop(index)
    results.append(
        {
            "task": {"type": "token-classification"},
            "dataset": {
                "name": dataset_name,
                "type": "private",
            },
            "metrics": metrics,
        }
    )
    (model_path / "meta.json").write_text(json.dumps(meta, indent=2))
    (BASE_DIR / "artifacts/test_metrics.jsonl").write_text(
        json.dumps(metrics, indent=2)
    )
    with open(BASE_DIR / "artifacts/test_metrics_per_doc.jsonl", "w") as f:
        for doc_scores in per_doc:
            f.write(json.dumps(doc_scores, separators=(",", ":")))
            f.write("\n")


if __name__ == "__main__":
    app()
