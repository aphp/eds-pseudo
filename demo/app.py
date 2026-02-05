import datetime
import random
import re
import string
from pathlib import Path

import pandas as pd
import streamlit as st
from babel.dates import format_date
from faker import Faker
from spacy import displacy

import edsnlp
from eds_pseudo.pipes.dates_normalizer.dates_normalizer import DatesNormalizer

DEFAULT_TEXT = (
    "En 2015, M. Charles-François-Bienvenu "
    "Myriel était évêque de Digne. C’était un vieillard "
    "d’environ soixante-quinze ans ; il occupait le "
    "siège de Digne depuis 2006."
)
FAKER_LOCALE = "fr_FR"
DECES_PATH = Path("data/deces-2024-m03.txt")
MAIL_DOMAINS = [
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "outlook.com",
    "apple.com",
    "aphp.fr",
    "chu.fr",
    "chu-lyon.fr",
    "chu-montpellier.fr",
    "chu-nantes.fr",
    "aphm.fr",
    "labo-123.fr",
    "labo-paris.fr",
    "clinique-paris.fr",
    "clinique-lyon.fr",
    "medicale.fr",
    "parisseine.com",
    "univ-med.com",
]
HOSPITALS = [
    "AMBROISE PARÉ",
    "ANTOINE BÉCLÈRE",
    "ARMAND TROUSSEAU",
    "AVICENNE",
    "BICÊTRE",
    "BICHAT",
    "COCHIN",
    "HÔPITAL SAINTE-ANNE",
    "HÔPITAL SAINT-ANTOINE",
    "HÔPITAL SAINT-LOUIS",
    "HÔPITAL TENON",
    "LA PITIE-SALPÊTRIÈRE",
    "LARIBOISIÈRE",
    "NECKER",
    "ROBERT DEBRÉ",
    "SAINT-JOSEPH",
    "SAINT-VINCENT-DE-PAUL",
    "TROUSSEAU",
]
ENT_COLORS = {
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
FAKE_NAMES_SIZE = 750
FAKE_CITIES_SIZE = 300


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
        nlp.add_pipe(
            "eds_pseudo.dates_normalizer",
            name="demo_dates_normalizer",
            config={"format": "java"},
        )
        nlp.add_pipe("eds_pseudo.context")
    else:
        nlp = edsnlp.load("AP-HP/eds-pseudo-public")
        nlp.pipes.ner.compute_confidence_score = True
        if "demo_dates_normalizer" not in nlp.pipe_names:
            nlp.add_pipe(
                "eds_pseudo.dates_normalizer",
                name="demo_dates_normalizer",
                config={"format": "java"},
            )
    model_load_state.empty()
    return nlp


@st.cache_resource
def load_faker():
    return Faker(FAKER_LOCALE)


@st.cache_resource
def load_fake_names_and_cities():
    deces_names, deces_cities = load_insee_deces()
    if deces_names and deces_cities:
        names = deces_names
        cities = list(dict.fromkeys(deces_cities))
        if len(names) > FAKE_NAMES_SIZE:
            names = random.sample(names, FAKE_NAMES_SIZE)
        if len(cities) > FAKE_CITIES_SIZE:
            cities = random.sample(cities, FAKE_CITIES_SIZE)
        return names, sorted(cities)

    faker = load_faker()
    names = []
    cities = set()
    for _ in range(FAKE_NAMES_SIZE):
        first_names = tuple(
            faker.first_name() for _ in range(random.choice([1, 1, 1, 2, 2, 3]))
        )
        last_name = faker.last_name()
        names.append((first_names, last_name))
    for _ in range(FAKE_CITIES_SIZE):
        cities.add(faker.city())
    return names, sorted(cities)


def load_insee_deces(path=DECES_PATH):
    path = Path(path)
    if not path.exists():
        return [], []

    data = path.read_text(encoding="utf-8", errors="ignore")
    names = []
    cities = []

    for line in data.split("\n"):
        if not line or "/" not in line:
            continue
        name_portion, rest = line.split("/", 1)
        if "*" in name_portion:
            last_name, all_first_names = name_portion.split("*", 1)
            first_names = tuple(f.strip() for f in all_first_names.split())
            last_name = last_name.strip()
            if not first_names or not all(first_names) or not last_name:
                continue
            names.append((first_names, last_name))
            city = (
                rest.strip()
                .lstrip(string.digits)
                .replace("B033", "")
                .replace("B081", "")
            )[:30]
            if city.lower().startswith("departement") or "Préfecture" in city:
                continue
            city = re.sub(r"\d+E(?:R|ME)?\s+ARRONDISSEMENT", "", city)
            city = city.strip()
            if not city or (set(city) & set(string.digits)):
                continue
            cities.append(city)

    return names, cities


def generate_fake_phone_number():
    if random.randint(0, 2):
        ctry = 33
    else:
        ctry = random.randint(11, 99)

    if random.randint(0, 2):
        ctry = f"+{ctry:02}"
    else:
        ctry = f"({ctry:02})"

    n = "".join(
        [" " if random.randint(0, 2) else "0"]
        + [str(random.randint(1 if i == 2 else 0, 7)) for i in range(9)]
    )

    s1 = random.choice([" ", "-", ".", "", "", ""])
    s2 = random.choice([" ", ""])

    formats = [
        f"0{n[1:2]}{s1}{n[2:4]}{s1}{n[4:6]}{s1}{n[6:8]}{s1}{n[8:10]}",
        f"0{n[1:2]}{s1}{n[2:4]}{s1}{n[4:6]}{s1}{n[6:8]}{s1}{n[8:10]}",
        f"{ctry}{s2}{n[:2].strip()}{s2}{n[2:4]}{s2}{n[4:6]}{s2}{n[6:8]}{s2}{n[8:10]}",
    ]

    fake_phone = random.choice(formats)

    if not random.randint(0, 6):
        fake_phone = " ".join([char for char in fake_phone if char != " "])

    return fake_phone


def pick_fake_name(names):
    first_names, last_name = random.choice(names)
    if random.randint(0, 2):
        first_names = first_names[:1]
    num_names_to_use = random.choice([1, 1, 1, 1, 1, 2, 2, 2, 3])
    first_names = first_names[:num_names_to_use]
    return first_names, (last_name,)


def make_first_name(first_names, case=None):
    if (case in ("N", "N.") or case is None and random.randint(0, 3) == 0) and not (
        len(first_names) == 1 and first_names[0][0] == "M"
    ):
        first_name_sep = random.choice(["-", ""])
        with_dot = random.randint(0, 2) == 0 or case == "N"
        first_name = first_name_sep.join(
            [f[:1] + ("" if with_dot else ".") for f in first_names[:2]]
        )
    else:
        first_name = " ".join(first_names)
        first_name = first_name.title()

    first_name = first_name.strip()
    if case is not None:
        if case.isupper():
            return first_name.upper()
        if case.islower():
            return first_name.lower()
    return first_name


def make_last_name(last_name, case=None):
    last_name = " ".join(last_name).title()
    last_name = last_name.strip()
    if case is not None:
        if case.isupper():
            return last_name.upper()
        if case.islower():
            return last_name.lower()
    return last_name


def pick_city(cities):
    city = random.choice(cities)
    parts = [p for p in re.split(r"[\s-]+", city) if p]
    if random.randint(0, 2):
        parts = [p.lower().capitalize() for p in parts]
    if random.randint(0, 2):
        city = " ".join(parts)
    else:
        city = "-".join(parts)
    return city


def generate_random_mail(fake_names):
    first_names, last_name = pick_fake_name(fake_names)
    first_name = make_first_name(first_names, "Nn")
    last_name = make_last_name(last_name, "Nn")
    first_name = re.sub(f"[{re.escape(string.punctuation) + ' '}]", "", first_name)
    last_name = re.sub(f"[{re.escape(string.punctuation) + ' '}]", "", last_name)
    domain = random.choice(MAIL_DOMAINS)

    formats = [
        "{first_name}{num1}{sp}{sep}{sp}{last_name}{num2}{sp}@{sp}{domain}",
        "{last_name}{num1}{sp}{sep}{sp}{first_name}{num2}{sp}@{sp}{domain}",
    ]
    sp = "" if random.randint(0, 10) else " "

    fake_mail = random.choice(formats).format(
        first_name=first_name.lower()[: random.randint(1, len(first_name))],
        last_name=last_name.lower()[: random.randint(1, len(last_name))],
        domain=domain,
        sep=random.choice(["-", ".", "_", "", ""]),
        num1=random.choice(["", "", random.randint(0, 99)]),
        num2=random.choice(["", "", random.randint(0, 99)]),
        sp=sp,
    )

    if not sp:
        fake_mail = fake_mail.replace(" ", "")
    fake_mail = fake_mail.strip()
    return fake_mail


def generate_random_date(year=None, format=None):
    start_date = datetime.date(2000, 1, 1)
    end_date = datetime.date(2027, 12, 31)

    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    random_number_of_days = random.randrange(days_between_dates)
    random_date = start_date + datetime.timedelta(days=random_number_of_days)

    if year is not None:
        random_date = random_date + datetime.timedelta(
            days=(year - random_date.year) * 365
        )

    if format == "dddddddd":
        return " ".join(random_date.strftime("%d%m%Y"))

    if format is None:
        format = "dd/MM/yyyy"

    result = format_date(random_date, format=format, locale=FAKER_LOCALE)

    if random.randint(0, 20) == 0 and not (set(result) & set(string.ascii_letters)):
        result = " ".join(char for char in result if char != " ")

    if "." not in format and random.randint(0, 4):
        result = result.replace(".", "")
    return result


def generate_fake_nss(year=None):
    sex = random.choice([1, 2])

    if year is None:
        year = str(random.randint(0, 99)).zfill(2)
    else:
        year = str(year % 100).zfill(2)

    month = str(random.randint(1, 12)).zfill(2)

    department = str(random.choice(list(range(1, 96)) + [98, 99])).zfill(2)

    town = str(random.randint(1, 999)).zfill(3)

    order_num = str(random.randint(1, 999)).zfill(3)

    s1, s2, s3, s4, s5 = random.choices(["", " "], k=5)
    nss = f"{sex}{s1}{year}{s2}{month}{s3}{department}{s4}{town}{s5}{order_num}"

    if random.randint(0, 2):
        nss = " ".join([char for char in nss if char != " "])

    return nss


def generate_french_zipcode():
    metro_departments = list(range(1, 96))
    metro_departments.remove(20)
    dom_departments = [971, 972, 973, 974, 976]
    corsica_departments = ["2A", "2B"]

    choice = random.choice([*(("metro",) * 5), "dom", "corsica"])

    if choice == "metro":
        department = random.choice(metro_departments)
        zipcode = f"{department:02d}{random.randint(0, 999):03d}"
    elif choice == "dom":
        department = random.choice(dom_departments)
        zipcode = f"{department}{random.randint(0, 99):02d}"
    else:
        department = random.choice(corsica_departments)
        zipcode = f"{department}{random.randint(0, 999):03d}"

    if random.choice([True, False]):
        zipcode = f"{zipcode[:2]} {zipcode[2:]}"

    return zipcode


def detect_name_format(text):
    if text.endswith("."):
        return "N."
    if len(text) == 1:
        return "N"
    if text.isupper():
        return "NN"
    if text.istitle():
        return "Nn"
    return "Nn"


def _apply_case(original, replacement):
    if original.isupper():
        return replacement.upper()
    if original.islower():
        return replacement.lower()
    if original.istitle():
        return replacement.title()
    return replacement


def _extract_date_format(normalizer, text):
    try:
        return normalizer.extract_date(text)[2]
    except Exception:
        return None


def _render_displacy(doc_or_data):
    manual = isinstance(doc_or_data, dict)
    html = displacy.render(
        doc_or_data,
        style="ent",
        manual=manual,
        options={"colors": ENT_COLORS},
    )
    html = html.replace("line-height: 2.5;", "line-height: 2.25;")
    html = (
        '<div style="padding: 10px; border: solid 2px; border-radius: 10px; '
        f'border-color: #afc6e0;">{html}</div>'
    )
    return html


def _get_base_date(ent, normalizer):
    date_format = ent._.date_format or None
    abs_date = getattr(ent._, "date", None)

    if abs_date is None:
        try:
            abs_date, _, extracted_format = normalizer.extract_date(ent.text)
            if not date_format:
                date_format = extracted_format
        except Exception:
            return None, date_format

    if abs_date is None:
        return None, date_format

    year = getattr(abs_date, "year", None)
    if year is None:
        return None, date_format

    month = getattr(abs_date, "month", None) or 1
    day = getattr(abs_date, "day", None) or 1
    try:
        base_date = datetime.date(int(year), int(month), int(day))
    except Exception:
        base_date = datetime.date(int(year), 1, 1)
    return base_date, date_format


def _format_shifted_date(base_date, shift_days, date_format):
    shifted = base_date + datetime.timedelta(days=shift_days)
    if date_format == "dddddddd":
        return " ".join(shifted.strftime("%d%m%Y"))

    if not date_format:
        date_format = "dd/MM/yyyy"

    result = format_date(shifted, format=date_format, locale=FAKER_LOCALE)

    if random.randint(0, 20) == 0 and not (set(result) & set(string.ascii_letters)):
        result = " ".join(char for char in result if char != " ")

    if "." not in date_format and random.randint(0, 4):
        result = result.replace(".", "")
    return result


def pseudonymize_doc(doc):
    faker = load_faker()
    fake_names, fake_cities = load_fake_names_and_cities()
    normalizer = DatesNormalizer(None, format="java")
    name_cache = {}
    noun_cache = {}
    last_first_names = None
    last_last_name = None
    date_shift_days = random.choice(
        [
            random.randint(-3 * 365, -1 * 365),
            random.randint(1 * 365, 3 * 365),
        ]
    )
    birth_year = random.randint(1970, 2100)
    for ent in doc.ents:
        if ent.label_ == "DATE_NAISSANCE":
            base_date, _ = _get_base_date(ent, normalizer)
            if base_date is not None:
                birth_year = (base_date + datetime.timedelta(days=date_shift_days)).year
            break

    replacements = []
    for ent in sorted(doc.ents, key=lambda span: span.start_char):
        label = ent.label_
        text = ent.text
        norm_text = text.casefold()
        replacement = None

        if label in {"NOM", "PRENOM"}:
            if label == "NOM" and (
                norm_text in name_cache or last_last_name is not None
            ):
                if norm_text not in name_cache:
                    name_cache[norm_text] = last_last_name
                replacement = make_last_name(
                    name_cache[norm_text], detect_name_format(text)
                )
                last_last_name = None
            elif label == "PRENOM" and (
                norm_text in name_cache or last_first_names is not None
            ):
                if norm_text not in name_cache:
                    name_cache[norm_text] = last_first_names
                replacement = make_first_name(
                    name_cache[norm_text], detect_name_format(text)
                )
                last_first_names = None
            else:
                first_name, last_name = pick_fake_name(fake_names)
                last_first_names, last_last_name = first_name, last_name
                if label == "NOM":
                    name_cache[norm_text] = last_last_name
                    last_last_name = None
                    replacement = make_last_name(last_name, detect_name_format(text))
                else:
                    name_cache[norm_text] = first_name
                    last_first_names = None
                    replacement = make_first_name(first_name, detect_name_format(text))
        elif label in {"VILLE", "HOPITAL", "ADRESSE"}:
            cache_key = (label, norm_text)
            if cache_key in noun_cache:
                base_replacement = noun_cache[cache_key]
            else:
                if label == "VILLE":
                    base_replacement = pick_city(fake_cities)
                elif label == "HOPITAL":
                    base_replacement = random.choice(HOSPITALS)
                else:
                    base_replacement = faker.street_address()
                noun_cache[cache_key] = base_replacement
            replacement = _apply_case(text, base_replacement)
        elif label == "TEL":
            replacement = generate_fake_phone_number()
        elif label == "SECU":
            replacement = generate_fake_nss(birth_year)
        elif label == "ZIP":
            replacement = generate_french_zipcode()
        elif label == "DATE_NAISSANCE":
            base_date, date_format = _get_base_date(ent, normalizer)
            if base_date is not None:
                replacement = _format_shifted_date(
                    base_date, date_shift_days, date_format
                )
            else:
                replacement = generate_random_date(year=birth_year, format=date_format)
        elif label == "DATE":
            base_date, date_format = _get_base_date(ent, normalizer)
            if base_date is not None:
                replacement = _format_shifted_date(
                    base_date, date_shift_days, date_format
                )
            else:
                replacement = generate_random_date(format=date_format)
        elif label in {"MAIL", "EMAIL"}:
            replacement = generate_random_mail(fake_names)
        elif label == "IPP":
            replacement = str(random.randint(10**9, 10**10 - 1))
        elif label == "NDA":
            replacement = str(random.randint(10**8, 10**9 - 1))
        elif label == "RPPS":
            replacement = str(random.randint(10**10, 10**11 - 1))
        else:
            replacement = f"[{label}]"

        replacements.append((ent.start_char, ent.end_char, label, replacement))

    out_parts = []
    pseudo_ents = []
    cursor = 0
    out_len = 0
    for start, end, label, replacement in replacements:
        chunk = doc.text[cursor:start]
        out_parts.append(chunk)
        out_len += len(chunk)
        ent_start = out_len
        out_parts.append(replacement)
        out_len += len(replacement)
        ent_end = out_len
        pseudo_ents.append({"start": ent_start, "end": ent_end, "label": label})
        cursor = end
    out_parts.append(doc.text[cursor:])
    pseudo_text = "".join(out_parts)
    return pseudo_text, pseudo_ents


@st.cache_data(max_entries=64)
def apply_model(text, use_rule_based):
    doc = nlp(text)
    html = _render_displacy(doc)
    pseudonymized_text, pseudo_ents = pseudonymize_doc(doc)
    pseudo_html = _render_displacy({"text": pseudonymized_text, "ents": pseudo_ents})
    data = []
    for ent in doc.ents:
        d = dict(
            start=ent.start_char,
            end=ent.end_char,
            text=ent.text,
            label=ent.label_,
            normalized_value=str(ent._.value or ""),
            date_format=str(ent._.date_format),  # e.g. dd/MM/yyyy or d MMMM yyyy
        )
        if not use_rule_based:
            d["ner_confidence_score"] = ent._.ner_confidence_score
        data.append(d)
    return data, html, pseudo_html


st.set_page_config(
    page_title="EDS-Pseudo Demo",
    page_icon="📄",
    layout="wide",
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

st.header("Enter a text to pseudonymize:")
text = st.text_area(
    "Modify the following text and see the model's predictions :",
    value=DEFAULT_TEXT,
    label_visibility="collapsed",
    height=125,
    max_chars=512,
)

data, html, pseudo_html = apply_model(text, use_rule_based)

col_original, col_pseudo = st.columns(2, gap="large")
with col_original:
    st.subheader("Original text")
    st.write(html, unsafe_allow_html=True)
with col_pseudo:
    st.subheader("Pseudonymized text")
    st.write(pseudo_html, unsafe_allow_html=True)

st.subheader("Identifying entities")

if data:
    df = pd.DataFrame.from_records(data)
    df.normalized_value = df.normalized_value.replace({"None": ""})
    st.dataframe(df, use_container_width=True)

else:
    st.markdown("The model did not match any entity...")
