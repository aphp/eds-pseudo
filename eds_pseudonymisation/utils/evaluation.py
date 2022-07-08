from collections import Counter
from functools import partial
from itertools import chain
from timeit import default_timer as timer
from typing import Any, Dict, Optional

from spacy.language import Language
from spacy.scorer import Scorer
from spacy.tokens import Doc, DocBin, Span
from spacy.training.example import Example


def remove_ents(doc: Doc) -> Doc:
    doc.ents = []
    return doc


def merge_annotations(merges: Dict[str, str], doc: Doc) -> Doc:
    doc.ents = [
        Span(doc, ent.start, ent.end, label=merges.get(ent.label_, ent.label_))
        for ent in doc.ents
    ]
    return doc


def evaluate(
    nlp: Language,
    db: DocBin,
    *,
    merges: Dict[str, str] = dict(),
    batch_size: Optional[int] = None,
    scorer: Optional[Scorer] = None,
    component_cfg: Optional[Dict[str, Dict[str, Any]]] = None,
    scorer_cfg: Optional[Dict[str, Any]] = None,
):

    references = db.get_docs(nlp.vocab)
    predicted = db.get_docs(nlp.vocab)

    predicted = map(remove_ents, predicted)

    if batch_size is None:
        batch_size = nlp.batch_size
    if component_cfg is None:
        component_cfg = {}
    if scorer_cfg is None:
        scorer_cfg = {}
    if scorer is None:
        kwargs = dict(scorer_cfg)
        kwargs.setdefault("nlp", nlp)
        scorer = Scorer(**kwargs)

    start_time = timer()
    # this is purely for timing
    for doc in db.get_docs(nlp.vocab):
        nlp.make_doc(doc.text)

    # apply all pipeline components
    predicted = nlp.pipe(
        (pred for pred in predicted),
        batch_size=batch_size,
        component_cfg=component_cfg,
    )

    predicted = list(predicted)

    end_time = timer()

    if merges:
        merge_ann = partial(merge_annotations, merges)
        references = map(merge_ann, references)
        predicted = map(merge_ann, predicted)

    examples = (Example(pred, ref) for pred, ref in zip(predicted, references))

    results = scorer.score_spans(examples, attr="ents")

    n_words = sum(len(doc) for doc in db.get_docs(nlp.vocab))
    results["speed"] = n_words / (end_time - start_time)

    labels = chain.from_iterable(
        map(lambda doc: [e.label_ for e in doc.ents], db.get_docs(nlp.vocab))
    )
    if merges:
        labels = map(lambda label: merges.get(label, label), labels)

    counter = Counter(labels)

    for k, v in counter.items():
        results["ents_per_type"][k]["support"] = v

    return results
