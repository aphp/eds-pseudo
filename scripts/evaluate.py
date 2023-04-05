# ruff: noqa: E501
import re
from pathlib import Path
from typing import Any, Dict, Optional

import srsly
import typer
from spacy import util
from spacy.cli._util import Arg, Opt, import_code, setup_gpu
from spacy.cli.evaluate import handle_scores_per_type, render_parses
from spacy.tokens import DocBin
from thinc.api import fix_random_seed
from wasabi import Printer

from eds_pseudonymisation.corpus_reader import PseudoCorpus


# fmt: off
def evaluate_cli(
      model: str = Arg(..., help="Model name or path"),
      data_path: Path = Arg(..., help="Location of binary evaluation data in .spacy format", exists=True),
      output: Optional[Path] = Opt(None, "--output", "-o", help="Output JSON file for metrics", dir_okay=False),
      docbin: Optional[Path] = Opt(None, "--docbin", help="Output Doc Bin path", dir_okay=False),
      code_path: Optional[Path] = Opt(None, "--code", "-c", help="Path to Python file with additional code (registered functions) to be imported"),
      use_gpu: int = Opt(-1, "--gpu-id", "-g", help="GPU ID or -1 for CPU"),
      gold_preproc: bool = Opt(False, "--gold-preproc", "-G", help="Use gold preprocessing"),
      displacy_path: Optional[Path] = Opt(None, "--displacy-path", "-dp", help="Directory to output rendered parses as HTML", exists=True, file_okay=False),
      displacy_limit: int = Opt(25, "--displacy-limit", "-dl", help="Limit of parses to render as HTML"),
):
    # fmt: on
    """
    Evaluate a trained pipeline. Expects a loadable spaCy pipeline and evaluation
    data in the binary .spacy format. The --gold-preproc option sets up the
    evaluation examples with gold-standard sentences and tokens for the
    predictions. Gold preprocessing helps the annotations align to the
    tokenization, and may result in sequences of more consistent length. However,
    it may reduce runtime accuracy due to train/test skew. To render a sample of
    dependency parses in a HTML file, set as output directory as the
    displacy_path argument.
    DOCS: https://spacy.io/api/cli#evaluate
    """
    import_code(code_path)
    evaluate(
        model,
        data_path,
        output=output,
        docbin=docbin,
        use_gpu=use_gpu,
        gold_preproc=gold_preproc,
        displacy_path=displacy_path,
        displacy_limit=displacy_limit,
        silent=False,
    )


def evaluate(
      model: str,
      data_path: Path,
      output: Optional[Path] = None,
      docbin: Optional[Path] = None,
      use_gpu: int = -1,
      gold_preproc: bool = False,
      displacy_path: Optional[Path] = None,
      displacy_limit: int = 25,
      silent: bool = True,
      spans_key: str = "sc",
) -> Dict[str, Any]:
    msg = Printer(no_print=silent, pretty=not silent)
    fix_random_seed()
    setup_gpu(use_gpu, silent=silent)
    data_path = util.ensure_path(data_path)
    output_path = util.ensure_path(output)
    displacy_path = util.ensure_path(displacy_path)
    if not data_path.exists():
        msg.fail("Evaluation data not found", data_path, exits=1)
    if displacy_path and not displacy_path.exists():
        msg.fail("Visualization output directory not found", displacy_path, exits=1)
    corpus = PseudoCorpus(data_path, gold_preproc=gold_preproc)
    nlp = util.load_model(model)
    # nlp.remove_pipe("dates")
    # nlp.remove_pipe("addresses")
    # nlp.remove_pipe("rules")
    # nlp.remove_pipe("structured")

    dev_dataset = [
        eg
        for eg in corpus(nlp)  # if getattr(eg.reference._, "split", "test") == "test"
    ]
    print(f"Evaluating {len(dev_dataset)} docs")

    if docbin is not None:
        output_db = DocBin(store_user_data=True)
        for doc in nlp.pipe(DocBin().from_disk(data_path).get_docs(nlp.vocab)):
            doc.user_data = {
                k: v
                for k, v in doc.user_data.items()
                if "note_id" in k or "context" in k or "split" in k
            }
            output_db.add(doc)
        output_db.to_disk(docbin)

    scores = nlp.evaluate(dev_dataset)
    metrics = {
        "TOK": "token_acc",
        "TAG": "tag_acc",
        "POS": "pos_acc",
        "MORPH": "morph_acc",
        "LEMMA": "lemma_acc",
        "UAS": "dep_uas",
        "LAS": "dep_las",
        "NER P": "ents_p",
        "NER R": "ents_r",
        "NER F": "ents_f",
        "TEXTCAT": "cats_score",
        "SENT P": "sents_p",
        "SENT R": "sents_r",
        "SENT F": "sents_f",
        "SPAN P": f"spans_{spans_key}_p",
        "SPAN R": f"spans_{spans_key}_r",
        "SPAN F": f"spans_{spans_key}_f",
        "SPEED": "speed",
    }
    results = {}
    data = {}
    for metric, key in metrics.items():
        if key in scores:
            if key == "cats_score":
                metric = metric + " (" + scores.get("cats_score_desc", "unk") + ")"
            if isinstance(scores[key], (int, float)):
                if key == "speed":
                    results[metric] = f"{scores[key]:.0f}"
                else:
                    results[metric] = f"{scores[key] * 100:.2f}"
            else:
                results[metric] = "-"
            data[re.sub(r"[\s/]", "_", key.lower())] = scores[key]

    msg.table(results, title="Results")
    data = handle_scores_per_type(scores, data, spans_key=spans_key, silent=silent)

    if displacy_path:
        factory_names = [nlp.get_pipe_meta(pipe).factory for pipe in nlp.pipe_names]
        docs = list(nlp.pipe(ex.reference.text for ex in dev_dataset[:displacy_limit]))
        render_deps = "parser" in factory_names
        render_ents = "ner" in factory_names
        render_parses(
            docs,
            displacy_path,
            model_name=model,
            limit=displacy_limit,
            deps=render_deps,
            ents=render_ents,
        )
        msg.good(f"Generated {displacy_limit} parses as HTML", displacy_path)

    if output_path is not None:
        srsly.write_json(output_path, data)
        msg.good(f"Saved results to {output_path}")
    return data


if __name__ == "__main__":
    typer.run(evaluate_cli)
