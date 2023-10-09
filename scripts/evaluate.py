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


@app.command(name="evaluate", registry=registry)
def evaluate(
    data: Callable[[Pipeline], Iterable[Doc]],
    model_path: Path = BASE_DIR / "artifacts/model-last",
    scorer: PseudoScorer = PseudoScorer(),
    data_seed: int = 42,
):
    test_metrics_path = BASE_DIR / "artifacts/evaluate_metrics.jsonl"
    nlp = edsnlp.load(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
    with set_seed(data_seed):
        val_docs: List[spacy.tokens.Doc] = list(data(nlp))
        scores = scorer(nlp, val_docs)
        scores_str = json.dumps(scores, indent=2)
        print(scores_str)
        test_metrics_path.write_text(scores_str)


if __name__ == "__main__":
    app()
