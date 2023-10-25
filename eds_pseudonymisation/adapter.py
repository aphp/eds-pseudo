import json
import random
from pathlib import Path
from typing import List, Optional

import spacy
from edsnlp.core.registry import registry
from edsnlp.utils.filter import filter_spans
from spacy.tokens import Doc


def subset_doc(doc: Doc, start: int, end: int) -> Doc:
    """
    Subset a doc given a start and end index.

    Parameters
    ----------
    doc: Doc
        The doc to subset
    start: int
        The start index
    end: int
        The end index

    Returns
    -------
    Doc
    """
    # TODO: review user_data copy strategy
    new_doc = doc[start:end].as_doc(copy_user_data=True)
    new_doc.user_data.update(doc.user_data)

    for name, group in doc.spans.items():
        new_doc.spans[name] = [
            spacy.tokens.Span(
                new_doc,
                max(0, span.start - start),
                min(end, span.end) - start,
                span.label,
            )
            for span in group
            if span.end > start and span.start < end
        ]

    return new_doc


def split_doc(
    doc: Doc,
    max_length: int = 0,
    randomize: bool = True,
    multi_sentence: bool = True,
) -> List[Doc]:
    """
    Split a doc into multiple docs of max_length tokens.

    Parameters
    ----------
    doc: Doc
        The doc to split
    max_length: int
        The maximum length of the resulting docs
    multi_sentence: bool
        Whether to split sentences across multiple docs
    randomize: bool
        Whether to randomize the split

    Returns
    -------
    Iterable[Doc]
    """
    if max_length <= 0:
        yield doc
    else:
        start = 0
        end = 0
        for ent in doc.spans.get("pseudo-ml", ()):
            for token in ent:
                token.is_sent_start = False
        for sent in doc.sents if doc.has_annotation("SENT_START") else (doc[:],):
            # If the sentence adds too many tokens
            if sent.end - start > max_length:
                # But the current buffer too large
                while sent.end - start > max_length:
                    subset_end = start + int(
                        max_length * (random.random() ** 0.3 if randomize else 1)
                    )
                    yield subset_doc(doc, start, subset_end)
                    start = subset_end
                yield subset_doc(doc, start, sent.end)
                start = sent.end

            if not multi_sentence:
                yield subset_doc(doc, start, sent.end)
                start = sent.end

            # Otherwise, extend the current buffer
            end = sent.end

        yield subset_doc(doc, start, end)


@registry.misc.register("pseudo-dataset")
def pseudo_dataset(
    path,
    limit: Optional[int] = None,
    max_length: int = 0,
    randomize: bool = False,
    multi_sentence: bool = True,
    filter_expr: Optional[str] = None,
):
    filter_fn = eval(f"lambda doc: {filter_expr}") if filter_expr else None
    assert not (
        limit is not None and isinstance(path, dict)
    ), "Cannot use specify both global limit and path-wise limit"
    if isinstance(path, (str, Path)):
        path = [path]
    if isinstance(path, list):
        path = {single_path: (limit or 0) for single_path in path}

    def load(nlp) -> List[Doc]:

        # Initialize the docs (tokenize them)
        normalizer = nlp.get_pipe("normalizer")
        sentencizer = nlp.get_pipe("sentencizer")

        # Load the jsonl data from path
        for single_path, path_limit in path.items():
            path_count = 0
            with open(single_path, "r") as f:
                lines = f
                if randomize:
                    lines = list(lines)
                    random.shuffle(lines)

                for line in lines:
                    if path_limit > 0 and path_count >= path_limit:
                        break
                    raw = json.loads(line)
                    doc = nlp.make_doc(raw["note_text"])
                    doc._.note_id = raw["note_id"]
                    doc._.note_datetime = raw.get("note_datetime")
                    doc._.note_class_source_value = raw.get("note_class_source_value")
                    doc._.context = raw.get("context", {})
                    doc = normalizer(doc)
                    doc = sentencizer(doc)
                    if not (len(doc) and (filter_fn is None or filter_fn(doc))):
                        continue

                    path_count += 1

                    # Annotate entities from the raw data
                    ents = []
                    span_groups = {
                        "pseudo-rb": [],
                        "pseudo-ml": [],
                        "pseudo-hybrid": [],
                    }
                    for ent in raw["entities"]:
                        span = doc.char_span(
                            ent["start"],
                            ent["end"],
                            label=ent["label"],
                            alignment_mode="expand",
                        )
                        ents.append(span)
                        span_groups["pseudo-rb"].append(span)
                        span_groups["pseudo-ml"].append(span)
                        span_groups["pseudo-hybrid"].append(span)
                    doc.ents = filter_spans(ents)
                    doc.spans.update(span_groups)

                    for new_doc in split_doc(
                        doc, max_length, randomize, multi_sentence
                    ):
                        if len(new_doc.text.strip()):
                            yield new_doc
                    else:
                        continue

                assert path_count > 0, "No data found in {}".format(single_path)

    return load
