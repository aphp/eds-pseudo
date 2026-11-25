import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from confit import Cli, Config

import edsnlp
from eds_pseudo.data import file_hash, normalize_record, text_hash
from edsnlp.core.pipeline import Pipeline
from edsnlp.core.stream import Stream
from edsnlp.training.trainer import GenericScorer

app = Cli(pretty_exceptions_show_locals=False)
# Name, Hugging Face metric type and metric file key of the scores that `package` writes in the model card
SCORES = (("Precision", "precision", "p"), ("Recall", "recall", "r"), ("F1", "f1", "f"))


@dataclass
class Evaluation:
    """
    One dataset scored by `eds-pseudo evaluate`, and the scorer that runs on it. With `predictions` (docs predicted
    in another process by `scripts/predict_package.py`, in the order of `data`), the metrics of the scorer compare
    them with `data` and no pipeline runs.
    """

    data: Stream
    scorer: GenericScorer
    predictions: Optional[Stream] = None


def source_group(record: Mapping[str, Any], group_field: Optional[str]) -> str:
    """
    Group of a record (its `group_field`, `group_id` or `source_id`, else the hash of its text), so that the records
    of one note or one person always land in the same split
    """
    for field in (group_field, "group_id", "source_id"):
        if field and record.get(field) not in (None, ""):
            return str(record[field])
    return text_hash(record["note_text"])


def assigned_split(group: str, seed: int, dev_fraction: float) -> str:
    """
    Put a group in dev with probability `dev_fraction`, from a hash of the seed and the group, so that the split of a
    group doesn't depend on the order of the records or on the other groups
    """
    value = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:16], 16) / 2**64
    return "dev" if value < dev_fraction else "train"


@app.command(name="prepare-data", registry=edsnlp.registry)
def prepare_data(
    sources: List[Dict[str, Any]],
    output_dir: Path,
    reserved_sources: List[Dict[str, Any]] = [],
    leakage_check_sources: List[Dict[str, Any]] = [],
    seed: int = 42,
    real_dev_fraction: float = 0.15,
    dataset_info_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Build the train and dev JSONL files read by the training configs, e.g. `data/prepared/public-train.jsonl` for
    `configs/config.yml`.

    1. Read the `reserved_sources` (test sets), whose texts and groups must not appear in train or dev.
    2. Check that the `leakage_check_sources`, which we read but don't rewrite, contain no reserved text or group.
    3. Split each of the `sources` into train and dev by group, or generate synthetic records from its seeds when it
       has a `generation` section, and check that no group or text ends up in two splits.
    4. Write one file per source and split.
    5. Write the counts and SHA256 of every input and output file to `dataset-info.json`.
    """
    if not 0 <= real_dev_fraction <= 1:
        raise ValueError("real_dev_fraction must be between zero and one")
    output_dir.mkdir(parents=True, exist_ok=True)
    input_info = []
    reserved_hashes = set()
    reserved_groups = set()
    prepared = defaultdict(list)

    # 1. Read the reserved test sets
    for spec in reserved_sources:
        path = Path(spec["path"])
        records = list(edsnlp.data.read_json(path).map(normalize_record))
        input_info.append({"path": str(path), "sha256": file_hash(path), "records": len(records), "role": "reserved"})
        for record in records:
            group = source_group(record, spec.get("group_field"))
            record.update(
                source=record.get("source", spec.get("source", path.stem)),
                split="test",
                group_id=group,
            )
            reserved_hashes.add(text_hash(record["note_text"]))
            reserved_groups.add(group)
            prepared[(spec.get("output_source", spec.get("source", path.stem)), "test")].append(record)

    # 2. Check the sources we don't rewrite against the reserved texts and groups
    for spec in leakage_check_sources:
        path = Path(spec["path"])
        records = list(edsnlp.data.read_json(path).map(normalize_record))
        input_info.append(
            {"path": str(path), "sha256": file_hash(path), "records": len(records), "role": "leakage-check"}
        )
        for record in records:
            group = source_group(record, spec.get("group_field"))
            if text_hash(record["note_text"]) in reserved_hashes or group in reserved_groups:
                raise ValueError(f"Reserved text or group occurs in leakage check source {path}, group {group}")

    # 3. Split or generate each source
    seen_splits = {}
    for source_idx, spec in enumerate(sources):
        path = Path(spec["path"])
        records = list(edsnlp.data.read_json(path).map(normalize_record))
        kind = spec.get("kind", "original")
        source = spec.get("source", path.stem)
        input_info.append({"path": str(path), "sha256": file_hash(path), "records": len(records), "role": kind})
        if "generation" in spec:
            from eds_pseudo.augment import generate_records

            generation = spec["generation"]
            values_path = Path(generation["values_path"]) if generation.get("values_path") else None
            people_path = Path(generation["people_path"]) if generation.get("people_path") else None
            # Since the templates sit in the config, we copy them to the dataset info, so that it tells how the records
            # were generated
            input_info.append({"source": source, "templates": generation["templates"], "role": "templates"})
            if values_path:
                input_info.append({"path": str(values_path), "sha256": file_hash(values_path), "role": "values"})
            if people_path:
                input_info.append({"path": str(people_path), "sha256": file_hash(people_path), "role": "people"})
            # We split the seeds before generating, so that no train record comes from a dev seed
            split_seeds = defaultdict(list)
            for record in records:
                group = source_group(record, spec.get("group_field"))
                split = spec.get("split") or record.get("split") or assigned_split(group, seed, real_dev_fraction)
                split_seeds[split].append({**record, "group_id": group, "split": split})
            records = []
            generated_hashes = set()
            for split, target_words in generation["target_words"].items():
                generated = generate_records(
                    seeds=split_seeds[split],
                    target_words=target_words,
                    templates=generation["templates"],
                    seed=seed + source_idx + list(generation["target_words"]).index(split),
                    document_prefix=f"{source}-{split}",
                    values_path=values_path,
                    people_path=people_path,
                    people_probability=generation.get("people_probability", 1),
                    excluded_hashes=reserved_hashes
                    | generated_hashes
                    | {
                        text_hash(record["note_text"])
                        for other_split, seed_records in split_seeds.items()
                        if other_split != split
                        for record in seed_records
                    },
                )
                records.extend({**record, "split": split} for record in generated)
                generated_hashes.update(text_hash(record["note_text"]) for record in generated)
        for record in records:
            group = source_group(record, spec.get("group_field"))
            split = spec.get("split") or record.get("split")
            normalized_hash = text_hash(record["note_text"])
            if kind in {"real", "synthetic"} and split is None:
                split = assigned_split(group, seed, real_dev_fraction)
            if split not in {"train", "dev", "test"}:
                raise ValueError(f"Source {source} needs a train, dev or test split")

            if split in {"train", "dev"} and (normalized_hash in reserved_hashes or group in reserved_groups):
                raise ValueError(f"Reserved text or group occurs in {split}, source {source}, group {group}")
            if kind in {"real", "synthetic"}:
                previous = seen_splits.setdefault(("group", group), split)
                if previous != split:
                    raise ValueError(f"Group {group} is in both {previous} and {split}")
                previous = seen_splits.setdefault(("text", normalized_hash), split)
                if previous != split:
                    raise ValueError("The same text is in two splits")
            record_source = record.get("source", source) if spec.get("preserve_source") else source
            record.update(source=record_source, split=split, group_id=group)
            prepared[(record_source, split)].append(record)

    # 4. Write one file per source and split
    outputs = []
    tokenizer = edsnlp.blank("eds").tokenizer
    for (source, split), records in sorted(prepared.items()):
        safe_source = "".join(character if character.isalnum() or character in "-_" else "-" for character in source)
        path = output_dir / f"{safe_source}-{split}.jsonl"
        edsnlp.data.write_json(records, path, lines=True, overwrite=True)
        outputs.append(
            {
                "path": str(path),
                "sha256": file_hash(path),
                "source": source,
                "split": split,
                "records": len(records),
                "words": sum(len(tokenizer(record["note_text"])) for record in records),
                "entities": sum(len(record["entities"]) for record in records),
                "labels": dict(
                    sorted(Counter(entity["label"] for record in records for entity in record["entities"]).items())
                ),
                "templates": dict(
                    sorted(Counter(record["template"] for record in records if "template" in record).items())
                ),
            }
        )

    # 5. Write the counts and SHA256 of every input and output file
    info = {
        "schema": 1,
        "seed": seed,
        "real_dev_fraction": real_dev_fraction,
        "inputs": input_info,
        "outputs": outputs,
        "reserved": {
            "text_hashes": len(reserved_hashes),
            "groups": len(reserved_groups),
        },
    }
    dataset_info_path = dataset_info_path or output_dir / "dataset-info.json"
    dataset_info_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return info


@app.command(name="evaluate", registry=edsnlp.registry)
def evaluate(
    evaluations: Dict[str, Evaluation],
    nlp: Optional[Pipeline] = None,
    output_dir: Path = Path("artifacts"),
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Score a pipeline on each dataset of `evaluations`, or the `predictions` of an evaluation when it has some, and
    write the metrics to `<output_dir>/<name>-metrics.json`, which `eds-pseudo experiments` gathers
    """
    if nlp is not None:
        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        nlp.to(device)
    results = {}
    for name, settings in evaluations.items():
        if settings.predictions is None:
            with nlp.select_pipes(enable=[]):
                docs = list(settings.data.map_pipeline(nlp))
            metrics = settings.scorer(nlp, docs)
        else:
            golds, preds = list(settings.data), list(settings.predictions)
            # The metrics pair the gold and predicted docs by position, so both files must hold the same texts
            if [doc.text for doc in golds] != [doc.text for doc in preds]:
                raise ValueError(f"The predictions of {name} don't follow the texts of its data")
            metrics = {key: metric(golds, preds) for key, metric in settings.scorer.metrics.items()}
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{name}-metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        results[name] = metrics
    return results


@app.command(name="stats", registry=edsnlp.registry)
def stats(sources: List[Path], output_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Count the records, entities, characters and labels of each file, and the texts found twice in one file or in
    several files
    """
    distributions = {}
    hashes = defaultdict(set)
    hash_counts = Counter()
    for source in sources:
        path = Path(source)
        counts = Counter()
        for record in edsnlp.data.read_json(path):
            counts["records"] += 1
            counts["entities"] += len(record["entities"])
            counts["characters"] += len(record["note_text"])
            normalized_hash = text_hash(record["note_text"])
            hashes[normalized_hash].add(str(path))
            hash_counts[normalized_hash] += 1
            counts.update(f"label:{entity['label']}" for entity in record["entities"])
        distributions[str(path)] = dict(sorted(counts.items()))
    result = {
        "sources": distributions,
        "duplicate_texts": sum(count > 1 for count in hash_counts.values()),
        "cross_source_overlaps": sum(len(paths) > 1 for paths in hashes.values()),
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return result


@app.command(name="experiments", registry=edsnlp.registry)
def experiments(
    runs_dir: Path = Path("artifacts"),
    output_path: Path = Path("artifacts/experiments.csv"),
) -> List[Dict[str, Any]]:
    """
    Gather the `<evaluation>-metrics.json` files that `evaluate` wrote in each run folder of `runs_dir` into the CSV
    `output_path`, one row per run and one `<evaluation>/<metric path>` column per score, so that we can compare runs
    side by side
    """
    import pandas as pd

    rows = []
    for run in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        row = {"run": run.name}
        for metrics_path in sorted(run.glob("*-metrics.json")):
            evaluation = metrics_path.name.removesuffix("-metrics.json")
            scores = pd.json_normalize(json.loads(metrics_path.read_text()), sep="/").iloc[0].to_dict()
            row.update({f"{evaluation}/{key}": value for key, value in scores.items()})
        if len(row) > 1:
            rows.append(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return rows


@app.command(name="package", registry=edsnlp.registry)
def package(
    pipeline: Union[Path, Pipeline],
    name: str,
    version: str = "1.0.0",
    root_dir: Path = Path("."),
    dist_dir: Path = Path("dist"),
    distributions: List[str] = ["wheel"],
    code_dependency: str = "eds-pseudo>=1,<2",
    code_check: str = "error",
    publish_index: Optional[str] = None,
    isolation: bool = True,
    skip_build_dependency_check: bool = False,
    metadata: Dict[str, Any] = {},
    readme_replacements: Dict[str, str] = {},
    card_metadata: Dict[str, Any] = {},
    results_dir: Optional[Path] = None,
    results: Dict[str, str] = {},
    label_results: List[str] = [],
):
    """
    Package a pipeline so that the package depends on `code_dependency` (`eds-pseudo>=1,<2` by default) instead of
    holding a copy of the eds_pseudo code.

    The README is the model card of the package: we write `card_metadata` as its front matter in place of
    `<!-- modelcard -->`. When `results_dir` is a run folder, we add the scores of its metric files to that front
    matter as the `model-index` that Hugging Face shows as evaluation results, and as tables in place of
    `<!-- metrics -->`.

    Parameters
    ----------
    card_metadata
        Front matter of the model card, e.g. its license, tags and gating fields
    results_dir
        Run folder with the `<evaluation>-metrics.json` files of `eds-pseudo evaluate`, e.g. the archived run of the
        packaged model
    results
        Evaluations of `results_dir` to report and the name we show for each, e.g. `{"original-test": "AP-HP test
        notes"}`. Each gets the micro NER strict match P/R/F1 and the redaction rates of the whole pipeline
    label_results
        Evaluations of `results` that also get the NER token match P/R/F1 of each label
    """
    if not hasattr(pipeline, "package"):
        pipeline = edsnlp.load(pipeline)
    front_matter = dict(card_metadata)
    if results_dir is not None:
        # The `final` key of a metric file holds the scores of the whole pipeline (rules and NER model)
        model_results = []
        rows = [
            "| Evaluation set | Documents | Precision | Recall | F1 | Redacted tokens | Fully redacted documents |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        label_tables = []
        for evaluation, dataset in results.items():
            scores = json.loads((results_dir / f"{evaluation}-metrics.json").read_text())["final"]
            exact, redact = scores["exact"]["micro"], scores["redact"]["micro"]
            metrics = [
                {"type": kind, "name": f"Exact Scores / micro / {name}", "value": round(exact[key], 4)}
                for name, kind, key in SCORES
            ]
            metrics.append({"type": "recall", "name": "Redaction / Tokens", "value": round(redact["token_r"], 4)})
            metrics.append(
                {"type": "accuracy", "name": "Redaction / Full documents", "value": round(redact["full"], 4)}
            )
            rows.append(
                f"| {dataset} | {redact['documents']:,} | {exact['p']:.3f} | {exact['r']:.3f} | {exact['f']:.3f} "
                f"| {redact['token_r']:.3f} | {redact['full']:.3f} |"
            )
            if evaluation in label_results:
                labels = {label: value for label, value in scores["token"].items() if label != "micro"}
                metrics += [
                    {"type": kind, "name": f"Token Scores / {label} / {name}", "value": round(value[key], 4)}
                    for label, value in labels.items()
                    for name, kind, key in SCORES
                ]
                label_tables.append(
                    "\n".join(
                        [
                            f"NER token match on {dataset}:",
                            "",
                            "| Label | Precision | Recall | F1 |",
                            "| --- | ---: | ---: | ---: |",
                        ]
                        + [f"| `{label}` | {v['p']:.3f} | {v['r']:.3f} | {v['f']:.3f} |" for label, v in labels.items()]
                    )
                )
            model_results.append(
                {
                    "task": {"type": "token-classification"},
                    "dataset": {"name": dataset, "type": evaluation},
                    "metrics": metrics,
                }
            )
        front_matter["model-index"] = [{"name": name, "results": model_results}]
        readme_replacements = {
            **readme_replacements,
            "<!-- metrics -->": "\n\n".join(
                [
                    "## Metrics",
                    "Scores of the whole pipeline on internal AP-HP data, which is not distributed. NER strict match "
                    "counts the entities with the same boundaries and label, NER token match counts their tokens, and "
                    "redaction counts the gold tokens (or the documents with all their gold tokens) that the pipeline "
                    "redacts whatever their predicted label.",
                    "\n".join(rows),
                    *label_tables,
                ]
            ),
        }
    if front_matter:
        readme_replacements = {
            **readme_replacements,
            # Config.to_yaml_str also writes the nested sections that confit gives as Config objects
            "<!-- modelcard -->": "---\n" + Config(front_matter).to_yaml_str() + "---",
        }
    return pipeline.package(
        name=name,
        version=version,
        root_dir=root_dir,
        dist_dir=dist_dir,
        distributions=distributions,
        code="dependency",
        code_dependency=code_dependency,
        code_check=code_check,
        publish_index=publish_index,
        isolation=isolation,
        skip_build_dependency_check=skip_build_dependency_check,
        metadata=metadata,
        readme_replacements=readme_replacements,
    )


@app.command(name="infer", registry=edsnlp.registry)
def infer(
    nlp: Union[Path, Pipeline],
    input_path: Path,
    output_path: Path,
    span_getter: Any = "ents",
    batch_size: Any = "32 docs",
) -> None:
    """
    Predict the entities of each record of `input_path` and write the records to `output_path` with their `entities`
    replaced by the predictions, keeping the `note_text` and the `DOC_FIELDS` of `eds_pseudo.data`
    """
    if not hasattr(nlp, "pipe"):
        nlp = edsnlp.load(nlp)
    # The empty span setter leaves the input entities out of the docs, so that the pipeline starts from blank docs
    docs = edsnlp.data.read_json(input_path, converter="eds_pseudo.dict2doc", tokenizer=nlp.tokenizer, span_setter={})
    edsnlp.data.write_json(
        docs.map_pipeline(nlp).set_processing(batch_size=batch_size),
        output_path,
        converter="eds_pseudo.doc2dict",
        span_getter=span_getter,
        lines=True,
        overwrite=True,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
