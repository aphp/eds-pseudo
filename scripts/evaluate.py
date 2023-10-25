import json
from pathlib import Path
from typing import Callable, Iterable, List

import edsnlp
import spacy
import torch
from confit import Cli
from confit.utils.random import set_seed
from edsnlp.core.pipeline import Pipeline
from edsnlp.core.registry import registry
from spacy.tokens import Doc

import eds_pseudonymisation.adapter  # noqa: F401
from eds_pseudonymisation.scorer import PseudoScorer

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
    data: Callable[[Pipeline], Iterable[Doc]],
    model_path: Path = BASE_DIR / "artifacts/model-last",
    scorer: PseudoScorer,
    data_seed: int = 42,
):
    test_metrics_path = BASE_DIR / "artifacts/test_metrics.json"
    per_doc_path = BASE_DIR / "artifacts/test_metrics_per_doc.jsonl"
    nlp = edsnlp.load(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
    with set_seed(data_seed):
        val_docs: List[spacy.tokens.Doc] = list(data(nlp))
        scores, per_doc = scorer(nlp, val_docs, per_doc=True)
    print(json.dumps(scores, indent=2))
    test_metrics_path.write_text(json.dumps(scores, indent=2))
    with open(per_doc_path, "w") as f:
        for doc_scores in per_doc:
            f.write(json.dumps(doc_scores, separators=(",", ":")))
            f.write("\n")


if __name__ == "__main__":
    app()
