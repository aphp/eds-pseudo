from functools import partial
from pathlib import Path
from typing import Callable, Iterable

import spacy
from spacy.language import Language
from spacy.tokens import DocBin, Doc
from spacy.training.corpus import *


if not Doc.has_extension("structured_data"):
    Doc.set_extension("structured_data", default=dict())
if not Doc.has_extension("note_id"):
    Doc.set_extension("note_id", default=None)
if not Doc.has_extension("split"):
    Doc.set_extension("split", default=None)


@spacy.registry.readers("eds-pseudonymisation.Corpus.v1")
class PseudoCorpus(Corpus):
    def _make_example(
        self, nlp: "Language", reference: Doc, gold_preproc: bool
    ) -> Example:
        eg = super()._make_example(nlp, reference, gold_preproc)
        eg.predicted._.structured_data = eg.reference._.structured_data
        eg.predicted._.split = eg.reference._.split
        return eg


@util.registry.readers("eds-pseudonymisation.Corpus.v1")
def create_docbin_reader(
    path: Optional[Path],
    gold_preproc: bool,
    max_length: int = 0,
    limit: int = 0,
    augmenter: Optional[Callable] = None,
) -> Callable[["Language"], Iterable[Example]]:
    if path is None:
        raise ValueError(Errors.E913)
    util.logger.debug(f"Loading corpus from path: {path}")
    return PseudoCorpus(
        path,
        gold_preproc=gold_preproc,
        max_length=max_length,
        limit=limit,
        augmenter=augmenter,
    )
