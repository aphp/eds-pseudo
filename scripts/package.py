from collections import defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from confit import Cli

import edsnlp
from eds_pseudo.adapter import PseudoReader  # noqa: F401
from eds_pseudo.scorer import PseudoScorer  # noqa: F401
from edsnlp.core.registries import registry

app = Cli(pretty_exceptions_show_locals=False)

DEFAULT_MODEL = Path(__file__).parent.parent / "artifacts" / "model-last"


@app.command("package", registry=registry)
def package(
    *,
    model: Optional[Path] = DEFAULT_MODEL,
    name: str,
    hf_name: str,
    **kwargs,
):
    nlp = edsnlp.load(model)
    results = nlp.meta.get("results", [])
    metrics_table_parts = ["## Metrics"] if results else []
    for result in results:
        grouped_metrics = defaultdict(lambda: {})
        for metric in result["metrics"]:
            parts = [key.strip() for key in metric["name"].split("/")]
            value = metric["value"]
            current = grouped_metrics
            for key in parts[:-1]:
                current = current.setdefault(key, {})
            current[parts[-1]] = value

        for group, data in grouped_metrics.items():
            df = pd.DataFrame.from_dict(data, orient="index")
            df.index.name = " ".join((result["dataset"]["name"], group))
            metrics_table_parts.append(
                df.applymap(lambda x: f"{x * 100:.1f}").to_markdown()
            )

    modelcard_metadata = {
        "language": ["fr" if nlp.lang == "eds" else nlp.lang],
        "pipeline_tag": "token-classification",
        "tags": ["medical", "ner", "nlp", "pseudonymisation"],
        "license": "bsd-3-clause",
        "library_name": "edsnlp",
        "model-index": [
            {
                "name": hf_name,
                "results": results,
            }
        ],
        "extra_gated_fields": {
            "Organisation": "text",
            "Intended use of the model": {
                "type": "select",
                "options": [
                    "NLP Research",
                    "Education",
                    "Commercial Product",
                    "Clinical Data Warehouse",
                    {"label": "Other", "value": "other"},
                ],
            },
        },
    }
    kwargs["readme_replacements"] = {
        "AP-HP/eds-pseudo-public": hf_name,
        "eds-pseudo-public": name,
        "<!-- metrics -->": "\n\n".join(metrics_table_parts),
        "<!-- modelcard -->": "---\n"
        + yaml.dump(
            modelcard_metadata,
            default_flow_style=False,
            sort_keys=False,
        )
        + "---",
    }
    nlp.package(name, **kwargs)


if __name__ == "__main__":
    app()
