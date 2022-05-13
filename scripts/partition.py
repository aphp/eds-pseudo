from pathlib import Path

import pandas as pd
import typer


def main(
    input_path: Path = typer.Option(..., help="Path to the input dataset"),
    output_train: Path = typer.Option(..., help="Path to the output train dataset"),
    output_dev: Path = typer.Option(..., help="Path to the output development dataset"),
    output_test: Path = typer.Option(..., help="Path to the output test dataset"),
    fraction: float = typer.Option(
        ..., help="Fraction (or number) of documents to use for development"
    ),
    seed: int = typer.Option(0, help="Random seed"),
):
    """Partition the data into train/test/dev split."""

    data = pd.read_json(input_path, lines=True)

    train = data.query("split == 'train'")
    test = data.query("split == 'edspdf'")

    train = train.drop(columns=["split"])
    test = test.drop(columns=["split"])

    if fraction >= 1:
        n = int(fraction)
        dev = train.sample(n=n, random_state=seed)
    else:
        dev = train.sample(n=None, frac=fraction, random_state=seed)

    train = train[~train.note_id.isin(dev.note_id)]

    train.to_json(output_train, orient="records", lines=True)
    dev.to_json(output_dev, orient="records", lines=True)
    test.to_json(output_test, orient="records", lines=True)


if __name__ == "__main__":
    typer.run(main)
