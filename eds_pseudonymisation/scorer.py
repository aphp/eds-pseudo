import time
from typing import List

import spacy
from confit import validate_arguments
from edsnlp.scorers import Scorer
from tqdm import tqdm

import eds_pseudonymisation.adapter  # noqa: F401


@validate_arguments
class PseudoScorer:
    def __init__(self, **scorers: Scorer):
        self.scorers = scorers

    def __call__(self, nlp, docs):
        clean_docs: List[spacy.tokens.Doc] = [d.copy() for d in docs]
        for d in clean_docs:
            d.ents = []
            d.spans.clear()
        t0 = time.time()
        preds = list(nlp.pipe(tqdm(clean_docs)))
        duration = time.time() - t0
        scores = {
            scorer_name: scorer(docs, preds)
            for scorer_name, scorer in self.scorers.items()
        }
        scores["speed"] = dict(
            wps=sum(len(d) for d in docs) / duration,
            dps=len(docs) / duration,
        )
        return scores
