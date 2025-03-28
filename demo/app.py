import pandas as pd
import streamlit as st
from spacy import displacy

import edsnlp

DEFAULT_TEXT = """\
En 2015, M. Charles-François-Bienvenu
Myriel était évêque de Digne. C’était un vieillard
d’environ soixante-quinze ans ; il occupait le
siège de Digne depuis 2006."""


@st.cache_resource
def load_model(use_rule_based=False):
    model_load_state = st.info("Loading model...")
    if use_rule_based:
        nlp = edsnlp.blank("eds")
        nlp.add_pipe("eds.normalizer")
        nlp.add_pipe(
            "eds_pseudo.simple_rules",
            config={"pattern_keys": ["TEL", "MAIL", "SECU", "PERSON", "RPPS"]},
        )
        nlp.add_pipe("eds_pseudo.addresses")
        nlp.add_pipe("eds_pseudo.dates")
        nlp.add_pipe("eds_pseudo.context")
    else:
        nlp = edsnlp.load("AP-HP/eds-pseudo-public")
        nlp.pipes.ner.compute_confidence_score = True
    model_load_state.empty()
    return nlp


@st.cache_data(max_entries=64)
def apply_model(text, use_rule_based):
    doc = nlp(text)
    html = displacy.render(
        doc,
        style="ent",
        options={
            "colors": {
                "ADRESSE": "#1f77b4",
                "DATE": "#aec7e8",
                "DATE_NAISSANCE": "#ff7f0e",
                "HOPITAL": "#ffbb78",
                "IPP": "#2ca02c",
                "MAIL": "#98df8a",
                "NDA": "#d62728",
                "NOM": "#ff9896",
                "PRENOM": "#9467bd",
                "SECU": "#c5b0d5",
                "TEL": "#8c564b",
                "VILLE": "#c49c94",
                "RPPS": "#e377c2",
            }
        },
    )
    html = html.replace("line-height: 2.5;", "line-height: 2.25;")
    html = (
        '<div style="padding: 10px; border: solid 2px; border-radius: 10px; '
        f'border-color: #afc6e0;">{html}</div>'
    )
    data = []
    for ent in doc.ents:
        d = dict(
            start=ent.start_char,
            end=ent.end_char,
            text=ent.text,
            label=ent.label_,
            normalized_value=str(ent._.value or ""),
        )
        if not use_rule_based:
            d["confidence_score"] = ent._.confidence_score
        data.append(d)
    return data, html


st.set_page_config(
    page_title="EDS-Pseudo Demo",
    page_icon="📄",
)

st.title("EDS-Pseudo")

st.warning(
    "You should **not** put sensitive data in the example, as this application "
    "**is not secure**."
)

st.sidebar.header("About")
st.sidebar.markdown(
    "EDS-Pseudo is a contributive effort maintained by AP-HP's Data Science team. "
    "Have a look at the "
    "[project](https://github.com/aphp/eds-pseudo/) for "
    "more information.\n\n"
    "In particular, the eds-pseudo-public model was trained on fictitious data "
    "and should be tested on your own data before considering using it.\n\n"
    "Since the data is fictitious, the model may produce errors or "
    "inaccuracies on real-world cases. Consider finetuning it on your own data or "
    "contributing to the training "
    "[templates](https://github.com/aphp/eds-pseudo/tree/main/data/templates.txt)"
    "\n\n"
    "A few links:\n"
    "- [Documentation](https://aphp.github.io/eds-pseudo/)\n"
    "- [GitHub](https://github.com/aphp/eds-pseudo/)\n"
    "- [Model card](https://huggingface.co/AP-HP/eds-pseudo-public)\n"
)
# Rule-based vs pretrained switch
use_rule_based = st.sidebar.checkbox("Use rule-based model", value=False)

nlp = load_model(use_rule_based)

st.header("Enter a text to analyse:")
text = st.text_area(
    "Modify the following text and see the model's predictions :",
    DEFAULT_TEXT,
    height=125,
    max_chars=512,
)

data, html = apply_model(text, use_rule_based)

st.header("Visualisation" + (" (rule-based model)" if use_rule_based else ""))

st.write(html, unsafe_allow_html=True)

st.header("Entities")

if data:
    df = pd.DataFrame.from_records(data)
    df.normalized_value = df.normalized_value.replace({"None": ""})
    st.dataframe(df)

else:
    st.markdown("The model did not match any entity...")
