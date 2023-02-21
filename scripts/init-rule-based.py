from pathlib import Path
import spacy
import typer

def main(
    lang: str = typer.Option(
        ..., help="Langage of the model (`eds` is preferred)"
    ),
    output: Path = typer.Option(
        ..., help="Path to the output dataset, in spaCy format"
    ),
):
    """Partition the data into train/test/dev split."""

    nlp = spacy.blank(lang)
    nlp.add_pipe('eds.remove-lowercase', name="remove-lowercase")
    nlp.add_pipe('eds.accents', name="accents")
    #nlp.add_pipe('pseudonymisation-dates', name="pseudonymisation-dates")
    nlp.add_pipe('pseudonymisation-rules', name="pseudonymisation-rules", config={
        "pattern_keys": ["TEL","MAIL","SECU"]#,"PERSON","NDA"]
    })
    nlp.add_pipe('pseudonymisation-addresses', name="pseudonymisation-addresses")
    nlp.add_pipe('structured-data-matcher', name="structured-data-matcher")
    nlp.to_disk(output)


if __name__ == "__main__":
    typer.run(main)
