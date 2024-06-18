# Pretrained model

An even simpler option than the rule-based model, and with a better performance (although a bit more compute-intensive)
consists in using the public pretrained model.
This model is available on the HuggingFace model hub at
[AP-HP/eds-pseudo-public](https://hf.co/AP-HP/eds-pseudo-public) and was trained on synthetic data described in the
[Dataset](/dataset) page. You can also test it directly on the **[demo](https://eds-pseudo-public.streamlit.app/)**.

## Installation

1. Install the latest version of edsnlp

    ```shell
    pip install "edsnlp[ml]" -U
    ```

2. Get access to the model at [AP-HP/eds-pseudo-public](https://hf.co/AP-HP/eds-pseudo-public)
3. Create and copy a token [https://hf.co/settings/tokens?new_token=true](https://hf.co/settings/tokens?new_token=true)
4. Register the token (only once) on your machine

    ```python
    import huggingface_hub

    huggingface_hub.login(
        token=YOUR_TOKEN,
        new_session=False,
        add_to_git_credential=True,
    )
    ```
5. Load the model

    ```python
    import edsnlp

    nlp = edsnlp.load("AP-HP/eds-pseudo-public", auto_update=True)
    doc = nlp(
        "En 2015, M. Charles-François-Bienvenu "
        "Myriel était évêque de Digne. C’était un vieillard "
        "d’environ soixante-quinze ans ; il occupait le "
        "siège de Digne depuis 2006."
    )

    for ent in doc.ents:
        print(ent, ent.label_, str(ent._.date))
    ```

To apply the model in parallel on many documents using one or more GPUs, refer to the [Inference](/inference) page.
