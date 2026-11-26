# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "confit==0.13.0",
#   "presidio-analyzer==2.2.366",
#   "fr_core_news_lg @ https://github.com/explosion/spacy-models/releases/download/fr_core_news_lg-3.8.0/fr_core_news_lg-3.8.0-py3-none-any.whl",
# ]
# ///
"""
Write the predictions of Presidio (`presidio-analyzer` with the `fr_core_news_lg` model of spaCy) on JSONL records, so
that `eds-pseudo evaluate` can score them through the `predictions` of an evaluation (experiment `presidio`).

Since Presidio and this spaCy model don't install next to eds-pseudo, the training job runs this script in its own
environment with `uv run`, and the script only reads the `presidio` section of the run config.
"""

import json
from pathlib import Path
from typing import Dict, List

from confit import Cli

app = Cli(pretty_exceptions_show_locals=False)


@app.command(name="presidio")
def presidio(inputs: Dict[str, Path], output_dir: Path, entities: List[str]) -> None:
    """
    Predict the `entities` of Presidio in each input file and write them to `<output_dir>/<name>-predictions.jsonl`,
    with the `note_id` and `note_text` of the input records in the same order.

    Parameters
    ----------
    inputs
        Gold JSONL files (`note_id`, `note_text` and `entities`) keyed by evaluation name, e.g. `original-test`
    entities
        Presidio entity types that we keep, e.g. `PERSON` or `PHONE_NUMBER`
    """
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    engine = NlpEngineProvider(
        nlp_configuration={"nlp_engine_name": "spacy", "models": [{"lang_code": "fr", "model_name": "fr_core_news_lg"}]}
    ).create_engine()
    registry = RecognizerRegistry(supported_languages=["fr"])
    registry.load_predefined_recognizers(languages=["fr"], nlp_engine=engine)
    analyzer = AnalyzerEngine(nlp_engine=engine, registry=registry, supported_languages=["fr"])

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, path in inputs.items():
        with open(path) as file, open(output_dir / f"{name}-predictions.jsonl", "w") as output:
            for line in file:
                record = json.loads(line)
                # Presidio returns overlapping results of different recognizers, e.g. a date inside a location, so
                # like its anonymizer we keep the best scored one and then the longest
                results = sorted(
                    analyzer.analyze(record["note_text"], language="fr", entities=entities),
                    key=lambda res: (-res.score, res.start - res.end),
                )
                kept = []
                for res in results:
                    if all(res.end <= other.start or other.end <= res.start for other in kept):
                        kept.append(res)
                record = {
                    "note_id": record["note_id"],
                    "note_text": record["note_text"],
                    "entities": [{"start": res.start, "end": res.end, "label": res.entity_type} for res in kept],
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    app()
