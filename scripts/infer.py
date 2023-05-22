from pathlib import Path
from typing import Optional

import spacy
import typer
from spacy.tokens import DocBin
from tqdm import tqdm


def main(
    model: Optional[Path] = typer.Option(None, help="Path to the model"),
    data: Path = typer.Option(
        ..., help="Path to the evaluation dataset, in spaCy format"
    ),
    output: Path = typer.Option(
        ..., help="Path to the output dataset, in spaCy format"
    ),
):
    """Partition the data into train/test/dev split."""

    spacy.prefer_gpu()

    nlp = spacy.load(model)

    db = DocBin().from_disk(data)
    input_docs = []
    for doc in db.get_docs(nlp.vocab):
        doc.ents = []
        input_docs.append(doc)

    print("Number of docs:", len(input_docs))

    out_db = DocBin(store_user_data=True)
    for doc in tqdm(nlp.pipe(input_docs), total=len(input_docs)):
        doc.user_data = {k: v for k, v in doc.user_data.items() if "trf_data" not in k}
        out_db.add(doc)

    out_db.to_disk(output)


if __name__ == "__main__":
    typer.run(main)
