import collections
import datetime
import json
import random
import re
import string
import sys
from collections.abc import MutableMapping
from pathlib import Path
from random import choice

from confit import Cli
from tqdm import tqdm

import eds_pseudo.pipes.dates_normalizer.dates_normalizer
import edsnlp
from eds_pseudo.pipes.dates_normalizer.dates_normalizer import DatesNormalizer
from edsnlp.core.registries import registry

print(eds_pseudo.pipes.dates_normalizer.dates_normalizer.__file__)

collections.MutableMapping = MutableMapping

from babel.dates import format_date  # noqa: E402

app = Cli(pretty_exceptions_show_locals=False)

last_names_prefix = [
    "Noms",
    "Nom",
    "Nom de famille",
    "Noms de famille",
    "Nom de jeune fille",
]
first_names_prefix = [
    "Prénoms",
    "Prénom",
    "Prénom(s)",
]

mail_domains = [
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

titles = [
    "Dr",
    "Docteur",
    "Interne",
    "Professeur",
    "Pr",
    "M",
    "Mme",
    "Monsieur",
    "Madame",
    "",
    "",
]

label_mapping = {
    "ADDRESS": "ADRESSE",
    "ADDRESSE": "ADRESSE",
    "ADRESSE": "ADRESSE",
    "DATE": "DATE",
    "BIRTHDATE": "DATE_NAISSANCE",
    "DATE_NAISSANCE": "DATE_NAISSANCE",
    "NAISSANCE": "DATE_NAISSANCE",
    "HOSPITAL": "HOPITAL",
    "HOPITAL": "HOPITAL",
    "IPP": "IPP",
    "MAIL": "MAIL",
    "EMAIL": "MAIL",
    "NDA": "NDA",
    "LASTNAME": "NOM",
    "NOM": "NOM",
    "FIRSTNAME": "PRENOM",
    "PRENOM": "PRENOM",
    "SECU": "SECU",
    "NSS": "SECU",
    "PHONE": "TEL",
    "TELEPHONE": "TEL",
    "TEL": "TEL",
    "CITY": "VILLE",
    "VILLE": "VILLE",
    "ZIP": "ZIP",
}

hospitals = [
    "A. Béclère",
    "A. beclere",
    "A.CHENEVIER",
    "A.CHENEVIER-H.MONDOR",
    "A.Chenevier",
    "ABC",
    "AMBROISE PARE",
    "AMBROISE PARÉ",
    "ANTOINE BECLERE",
    "APR",
    "ARMAND TROUSSEAU",
    "ARMAND TROUSSEAU-LA ROCHE GUYON",
    "AVC",
    "AVICENNE",
    "Albert\nChenevier",
    "Albert Chennevier",
    "Albert-Chennevier",
    "Ambroise\nParé",
    "Ambroise Paré",
    "Ambroise paré",
    "Américain",
    "Antoine\nBéclère",
    "Antoine Béclère",
    "Antoine-Béclère",
    "Armand Briard",
    "Armand Trousseau",
    "Avicenne",
    "BCH",
    "BCT",
    "BEAUJON",
    "BICETRE",
    "BICHAT",
    "BICHAT - CLAUDE BERNARD",
    "BICHAT-CLAUDE BERNARD",
    "BICHAT-LOUIS MOURIER",
    "BICÊTRE",
    "BJN",
    "BRT",
    "Ballanger",
    "Beaujon",
    "Beclere",
    "Begin",
    "Berck",
    "Bichat",
    "Bicêtre",
    "Bluets",
    "Bois d'Amour",
    "Bouch",
    "Bretonneau",
    "Broca",
    "Béclère",
    "Bégin",
    "C. BERNARD",
    "C. Celton",
    "CCH",
    "CCL",
    "CDS Grand Ouest Massy",
    "CDS MARCEL HANRA",
    "CDS Municipal Pantin",
    "CDS POLYVALENT MIROMESNIL",
    "CFX",
    "CHB",
    "CHIC",
    "CHS ROGER PREVOT",
    "CLAUDE BERNARD",
    "COCHIN",
    "COCHIN - SAINT-VINCENT DE PAUL",
    "COCHIN SAINT VINCENT DE PAUL",
    "CSAPA Murger",
    "Caunes",
    "Centre Cardiologique\nNord",
    "Centre Cardiologique du Nord",
    "Centre Charlebourg",
    "Centre Europe",
    "Centre Hospitalier Sainte-Anne",
    "Centre Hospitalier Universitaire Bicêtre",
    "Centre Hospitalier Universitaire de Bicêtre",
    "Centre hospitalier régional",
    "Centre municipla de santé Salavador Allende",
    "Charles Foix",
    "Charles-Foix",
    "Clinique\nde Val D'or",
    "Clinique Jeanne d'Arc",
    "Clinique Pervenche",
    "Clinique Saint-Louis",
    "Clinique de Bachaumont",
    "Clinique de la Jonquière",
    "Clinique des Cèdres",
    "Clinique des Jonquières",
    "Clinique du Parc",
    "Clinique du Parc de Belleville",
    "Clinique du Pont de Sèvres",
    "Clinique du Val D'Oo",
    "Clinique du Val D'or",
    "Cochin",
    "Cochin-Saint Vincent de Paul",
    "Cognacq Jay",
    "Corentin Celton",
    "Croix Saint Simon",
    "Croix St Simon",
    "Curie",
    "DE BICETRE",
    "DE LA SOURCE",
    "DE RAMBOUILLET",
    "DEBORD BROCA",
    "DELAFONTAINE",
    "DIACONESSES / CROIX SAINT SIMON",
    "DUPUYTREN",
    "DeLafontaine",
    "Delafontaine",
    "Denfert Rochereau",
    "E. Rist",
    "E.Rist",
    "EPS Roger Prévot",
    "ERX",
    "ESQUIROL",
    "EUROPÉEN GEORGES POMPIDOU",
    "Edouard Rist",
    "Esquirol",
    "FERNAND WIDAL",
    "FOCH",
    "FONDATION A. DE ROTHSCHILD",
    "FOUGERE",
    "Fernand Vidal",
    "Fernand Widal",
    "Foch",
    "Fondation\nRothschild",
    "Fondation Rothschild",
    "Fondation Rotschild",
    "Fondation St Jean de Dieu",
    "Fougère",
    "Franco-Britannique",
    "Franquefort",
    "G GPompidou",
    "G.POMPIDOU",
    "GONESSE",
    "GPompidou",
    "Georges Clémenceau",
    "Gustave Roussy",
    "H.MONDOR",
    "HEGP",
    "HENRI MONDOR",
    "HENRY DUNANT",
    "HMN",
    "HOPITAL DE BICETRE",
    "HTD",
    "Hamburger",
    "Hdj",
    "Hendaye",
    "Henri\nMondor",
    "Henri Ey",
    "Henri MONDOR",
    "Henri Mondor",
    "Henri mondor",
    "Hopital de val d'Yerres",
    "Hopital privé des peupliers",
    "Hotel Dieu",
    "Husson mourrier",
    "Hôtel Dieu",
    "Hôtel-Dieu",
    "IGR",
    "Institut Curie",
    "Ipso Nation",
    "JEAN VERDIER",
    "JOFFRE DUPUYTREN",
    "JVR",
    "Jean Verdier",
    "KB",
    "Kremlin Bicêtre",
    "Kremlin-Bicêtre",
    "LA ROCHE GUYON",
    "LARIBOIS IERE",
    "LARIBOISIERE",
    "LARIBOISIERE FERNAND WIDAL",
    "LARIBOISIÈRE",
    "LEON BERARD",
    "LMR",
    "LOUIS MOURIER",
    "LOUIS PASTEUR-LE COUDRAY",
    "LPS",
    "LRB",
    "LUCIE ET RAYMOND AUBRAC",
    "La Guisane",
    "La Pitié",
    "La Pitié\nSalpétrière",
    "La Pitié Salpétrière",
    "La Roche-Guyon",
    "La Roseraie",
    "La Salpêtrière",
    "La Verrière",
    "La pitié",
    "Labrouste",
    "Laennec",
    "Lamalou",
    "Larib",
    "Lariboisiere",
    "Lariboisière",
    "Lariboisère",
    "Louis Mourier",
    "Luchon",
    "Léon\nBerard",
    "Léon Bérard",
    "Léopold Bellan",
    "MONDOR",
    "Marie Lannelongue",
    "Marin de\nHendaye",
    "Marin de Hendaye",
    "Max Fourestier",
    "Mignot",
    "Mondor",
    "Monfer Meil",
    "Mont Louis",
    "Montsouris",
    "NCH",
    "NCK",
    "NECKER",
    "NECKER - ENFANTS\nMALADES",
    "NECKER - ENFANTS MALADES",
    "NECKER ENFANTS MALADES",
    "Necker",
    "Necker Enfants-Malades",
    "PARIS 7 – DENIS DIDEROT",
    "PARIS OUEST SITE G POMPIDOU",
    "PAUL BROUSSE",
    "PAUL GUIRAUD",
    "PB",
    "PBR",
    "PERCY",
    "PGV",
    "PITIE",
    "PITIE SALPETRIERE",
    "PITIE-LA SALPETRIERE",
    "PITIE-SALPETRIERE",
    "PITIÉ SALPÊTRIÈRE",
    "PRIVE DE L OUEST PARISIEN",
    "PSL",
    "PVR",
    "Paris I",
    "Paris Nord",
    "Pasteur",
    "Paul Brousse",
    "Paul Guiraud",
    "Percy",
    "Pitie",
    "Pitié",
    "Pitié\nSalpêtrière",
    "Pitié Salpetrière",
    "Pitié Salpitrière",
    "Pitié Salpétrière",
    "Pitié Salpétrière Pitié",
    "Pitié Salpêtrière",
    "Pitié-Salpétrière",
    "Pitié-Salpêtrière",
    "Pitié-Salpêtrière Charles Foix",
    "Pompidou",
    "Port Royal",
    "Quinze-Vingts",
    "RDB",
    "RMB",
    "RObert DEbre",
    "RPC",
    "RTH",
    "René-Muret",
    "RobDeb",
    "Robert\nDebré",
    "Robert Ballanger",
    "Robert Debré",
    "Robert-Debré",
    "Rothschild",
    "Rotschild",
    "Rotshild",
    "SAINT ANTOINE",
    "SAINT LOUIS",
    "SAINT REMY",
    "SAINT-ANTOINE",
    "SAINT-Camille",
    "SAINT-LOUIS",
    "SAT",
    "SLS",
    "ST JOSEPH",
    "ST LOUIS",
    "Saint\nANtoine",
    "Saint\nLouis",
    "Saint Anne",
    "Saint Antoine",
    "Saint Camille",
    "Saint Joseph",
    "Saint Louis",
    "Saint Maurice",
    "Saint-Antoine",
    "Saint-Joseph",
    "Saint-Louis",
    "Saint-Maurice",
    "Saint-Michel",
    "Sainte Anne",
    "Sainte Camille",
    "Sainte Périne",
    "Salneuve",
    "Salpétrière",
    "San Antonio",
    "San Salvadour",
    "Saujon",
    "Simone Veil",
    "St ANNE",
    "St Antoine",
    "St Etienne",
    "St Joseph",
    "St Louis",
    "St Maur",
    "St Maurice",
    "St antoine",
    "St-Louis",
    "Stalingrad",
    "Ste\nAnne",
    "Ste Anne",
    "Ste Camille",
    "Stell",
    "TENON",
    "TNN",
    "TRS",
    "Tenon",
    "Tnn",
    "Trouseau",
    "Trousseau",
    "UNIVERSITAIRE\nNECKER-ENFANTS MALADES",
    "UNIVERSITAIRE HENRI MONDOR",
    "UNIVERSITAIRE NECKER-ENFANTS MALADES",
    "Universitaire Necker-Enfants malades",
    "Universitaires Paris Centre",
    "Universitaires Paris Est",
    "Universitaires Paris-Seine-Saint-Denis",
    "VAUGIRARD-GABRIEL PALLEZ",
    "VCH",
    "Val d'Hyères",
    "Ville Evrard",
    "ambroise paré",
    "antoine beclere",
    "antony",
    "assan",
    "avicenne",
    "beaujon",
    "becelre",
    "bichat",
    "bjn",
    "béclère",
    "cap bastille",
    "centre cardiologique du Nord",
    "charles Foix",
    "clinique\nde Nogent",
    "clinique Floréal",
    "clinique Floréale",
    "clinique Jeanne d'Arc",
    "clinique Mont Louis",
    "clinique Montsouris",
    "clinique Saint-Hilaire",
    "clinique bizet",
    "clinique de\nNogent",
    "clinique de Turin",
    "clinique de Villepinte",
    "clinique de la Porte de saint cloud",
    "clinique de saint cloud",
    "clinique des Peupliers",
    "clinique du Louvre",
    "clinique du Saint-Coeur",
    "clinique montsouris",
    "cochin",
    "de la Source",
    "des JOCKEYS",
    "esquirol",
    "fernand widal",
    "henri mondor",
    "hopital Suisse",
    "husson Mourrier",
    "hôpital des Bluets",
    "hôpital du Val d'Yerres",
    "institut Charcot",
    "institut Franco-Britannique",
    "jean verdi",
    "kb",
    "la\nPitié",
    "la Croix Saint Simon",
    "la Pitié",
    "la Pitié Salpétrière",
    "la Pitié-Salpêtrière",
    "la Roche Guyon",
    "la pitie",
    "la pitié",
    "lariboisiere",
    "lariboisière",
    "mondor",
    "paul  guiraud",
    "paul Guiraud",
    "paul guirout",
    "pitie salpetriere",
    "pitié salpetrière",
    "pitié salpétrière",
    "psl",
    "saint antoine",
    "saint louis",
    "salpétrière",
    "st LOUIS",
    "st antoine",
    "tenon",
    "tnn",
    "trousseau",
    "universitaire mère-enfant\nRobert Debré",
    "vaugirard",
    "Émile Roux",
    "Émile roux",
]


def generate_fake_phone_number():
    # Generating a random country code between 1 and 99
    if random.randint(0, 2):
        ctry = 33
    else:
        ctry = random.randint(11, 99)

    if random.randint(0, 2):
        ctry = f"+{ctry:02}"
    else:
        ctry = f"({ctry:02})"

    # Generating a 10-digit phone number
    n = "".join(
        [" " if random.randint(0, 2) else "0"]
        + [str(random.randint(1 if i == 2 else 0, 7)) for i in range(9)]
    )

    # Different types of separators
    s1 = random.choice([" ", "-", ".", "", "", ""])
    s2 = random.choice([" ", ""])

    # Combining country code and number with different formats
    formats = [
        f"0{n[1:2]}{s1}{n[2:4]}{s1}{n[4:6]}{s1}{n[6:8]}{s1}{n[8:10]}",
        f"0{n[1:2]}{s1}{n[2:4]}{s1}{n[4:6]}{s1}{n[6:8]}{s1}{n[8:10]}",
        f"{ctry}{s2}{n[:2].strip()}{s2}{n[2:4]}{s2}{n[4:6]}{s2}{n[6:8]}{s2}{n[8:10]}",
    ]

    # Choosing a random format from the formats list
    fake_phone = random.choice(formats)

    if not random.randint(0, 6):
        fake_phone = " ".join([char for char in fake_phone if char != " "])

    return fake_phone


def load_insee_deces(path="data/deces-2024-m03.txt"):
    path = Path(path)

    # download if path doesn't exist
    if not path.exists():
        import requests

        url = "https://www.data.gouv.fr/fr/datasets/r/227743d3-434f-4659-8b0f-6af8b1c802f3"
        response = requests.get(url)
        path.write_text(response.text)
    data = path.read_text()

    names = []
    cities = []

    for line in data.split("\n"):
        if not line:
            continue
        name_portion, rest = line.split("/", 1)
        if "*" in name_portion:
            last_name, all_first_names = name_portion.split("*")
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
    return sorted(names), sorted(cities)


def pick_fake_name(names, firstname_case=None, lastname_case=None):
    first_names, last_name = random.choice(names)
    num_names_to_use = random.choice([1, 1, 1, 1, 1, 2, 2, 2, 3])
    first_names = first_names[:num_names_to_use]
    return first_names, (last_name,)


def make_first_name(first_names, case=None):
    # Generate abbreviations but avoid ambiguous initials like M.
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
    # Generate a random email address
    first_names, last_name = pick_fake_name(fake_names)
    first_name = make_first_name(first_names, "Nn")
    last_name = make_last_name(last_name, "Nn")
    first_name = re.sub(f"[{re.escape(string.punctuation) + ' '}]", "", first_name)
    last_name = re.sub(f"[{re.escape(string.punctuation) + ' '}]", "", last_name)
    domain = random.choice(mail_domains)

    # Different types of email formats
    formats = [
        "{first_name}{num1}{sp}{sep}{sp}{last_name}{num2}{sp}@{sp}{domain}",
        "{last_name}{num1}{sp}{sep}{sp}{first_name}{num2}{sp}@{sp}{domain}",
    ]
    sp = "" if random.randint(0, 10) else " "

    # Choose a random format from the formats list
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


def generate_random_date(year=None, format=None, allow_missing_parts=False):
    # Generate a random date between the years 1900 and 2100
    start_date = datetime.date(1970, 1, 1)
    end_date = datetime.date(2100, 12, 31)

    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    random_number_of_days = random.randrange(days_between_dates)
    random_date = start_date + datetime.timedelta(days=random_number_of_days)

    if year is not None:
        random_date = random_date + datetime.timedelta(
            days=(year - random_date.year) * 365
        )
    if format is None:
        # List of date formats
        formats = [
            f"{choice(['dd', 'd'])}/{choice(['MM', 'M'])}/{choice(['YYYY', 'YY'])}",
            f"{choice(['dd', 'd'])}-{choice(['MM', 'M'])}-{choice(['YYYY', 'YY'])}",
            f"{choice(['dd', 'd'])}.{choice(['MM', 'M'])}.{choice(['YYYY', 'YY'])}",
            "dddddddd",
            f"EEE{choice([' ', ' ', ', '])}d {choice(['MMM', 'MMMM'])} YYYY",
            f"d {choice(['MMM', 'MMMM'])} YYYY",
        ]

        if allow_missing_parts:
            formats += [
                "dd/MM",
                "dd-MM",
                "dd.MM",
                f"EEE{choice([' ', ' ', ', '])}d {choice(['MMM', 'MMMM'])}",
                f"{choice(['MMM', 'MMMM'])} YYYY",
            ]

        # Choose a random format and return the date
        format = random.choice(formats)

    if format == "dddddddd":
        return " ".join(random_date.strftime("%d%m%Y"))
    result = format_date(random_date, format=format, locale="fr_FR")

    # Remove dots like "12 sept. 2012"
    if "." not in format and random.randint(0, 4):
        result = result.replace(".", "")
    return result


def generate_fake_nss(year=None):
    # 1. Generate a digit for sex (either 1 or 2)
    sex = random.choice([1, 2])

    # 2. Generate two digits for the year of birth (00-99)
    if year is None:
        year = str(random.randint(0, 99)).zfill(2)
    else:
        year = str(year % 100).zfill(2)

    # 3. Generate two digits for the month of birth (01-12)
    month = str(random.randint(1, 12)).zfill(2)

    # 4. Generate two digits for the department of birth (01-95, 98
    # for non-metropolitan France) or 99 for births outside of France
    department = str(random.choice(list(range(1, 96)) + [98, 99])).zfill(2)

    # 5. Generate three digits for the town or borough of birth (001-999)
    town = str(random.randint(1, 999)).zfill(3)

    # 6. Generate three digits for the order number of the birth certificate (001-999)
    order_num = str(random.randint(1, 999)).zfill(3)

    # Combine all parts to form the fake NSS
    s1, s2, s3, s4, s5 = random.choices(["", " "], k=5)
    nss = f"{sex}{s1}{year}{s2}{month}{s3}{department}{s4}{town}{s5}{order_num}"

    if random.randint(0, 2):
        nss = " ".join([char for char in nss if char != " "])

    return nss


def make_dummy_sample(template=None, *, fake_names):
    if template is None:
        template = random.randint(0, 7)
    first_name, last_name = pick_fake_name(fake_names)
    first_name = make_first_name(first_name)
    last_name = make_last_name(last_name)
    title = random.choice(titles)
    phone = generate_fake_phone_number()

    if random.randint(0, 2):
        title = title.upper()

    if 0 < len(title) < 4 and random.randint(0, 2):
        title = title + "."
        if random.randint(0, 3):
            title = title + " "
    else:
        title = title + " "
    if template in (0, 1, 2, 3):
        if template == 0:
            return f"{title}[{first_name}](PRENOM) [{last_name}](NOM)".strip()
        elif template == 1:
            return f"{title}[{last_name}](NOM) [{first_name}](PRENOM)".strip()
        elif template == 2:
            tel_trigger = random.choice(
                ["Tel ", "Secrétariat ", "Téléphone ", "Accueil", "Standard", ""]
            )
            return (
                f"{title}[{first_name}](PRENOM) "
                f"[{last_name}](NOM)\n{tel_trigger}[{phone}](TEL)"
            ).strip()
        elif template == 3:
            if random.randint(0, 2):
                return (
                    f"{choice(last_names_prefix)}: [{last_name}](NOM)\n"
                    f"{choice(first_names_prefix)}: [{first_name}](PRENOM)"
                ).strip()
            else:
                return (
                    f"{choice(first_names_prefix)}: [{first_name}](PRENOM)\n"
                    f"{choice(last_names_prefix)}: [{last_name}](NOM)"
                ).strip()
    elif template > 3:
        birth_year = random.randint(1970, 2000)
        birth_date = generate_random_date(birth_year)
        nss = generate_fake_nss(birth_year)

        current_year = max(2000, birth_year) + random.randint(10, 30)
        date = generate_random_date(current_year, allow_missing_parts=True)

        date_nss_template = random.randint(0, 3)
        sep = random.choice([" ", "\n"])
        date_nss_str = None
        if date_nss_template == 0:
            date_nss_str = (
                f"[{date}](DATE){sep}[{nss}](SECU){sep}[{birth_date}](BIRTHDATE)"
            ).strip()
        elif date_nss_template == 1:
            date_nss_str = (
                f"[{date}](DATE){sep}"
                f"[{birth_date}](BIRTHDATE){sep}"
                f"[{nss}](SECU)"
            ).strip()
        elif date_nss_template == 2:
            date_nss_str = (
                f"[{birth_date}](BIRTHDATE) "
                f"[{date}](DATE){sep}[{nss}](SECU)".strip()
            )
        elif date_nss_template == 3:
            date_nss_str = (
                f"[{nss}](SECU){sep}"
                f"[{birth_date}](BIRTHDATE){sep}"
                f"[{date}](DATE)"
            ).strip()

        if template in (5, 6):
            return (
                f"{date_nss_str}\n{title}[{last_name}](NOM) "
                f"[{first_name}](PRENOM)\n[{phone}](TEL)"
            ).strip()
        else:
            return (
                f"{title}[{last_name}](NOM) "
                f"[{first_name}](PRENOM)\n"
                f"[{phone}](TEL)\n{date_nss_str}"
            ).strip()


def generate_french_zipcode():
    # Define metropolitan department codes
    metro_departments = list(range(1, 96))
    metro_departments.remove(20)  # Removing 20 as Corsica is handled separately
    dom_departments = [971, 972, 973, 974, 976]
    corsica_departments = ["2A", "2B"]

    # Randomly choose a department
    choice = random.choice([*(("metro",) * 5), "dom", "corsica"])

    if choice == "metro":
        department = random.choice(metro_departments)
        zipcode = f"{department:02d}{random.randint(0, 999):03d}"

    elif choice == "dom":
        department = random.choice(dom_departments)
        zipcode = f"{department}{random.randint(0, 99):02d}"

    else:  # Corsica
        department = random.choice(corsica_departments)
        zipcode = f"{department}{random.randint(0, 999):03d}"

    # Adding space in the middle for variation
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


# noinspection RegExpRedundantEscape
def augment_sample(sample, *, fake_names, fake_cities):
    # names = []
    normalizer = DatesNormalizer(None, format="java")
    mem = {}
    last_first_names, last_last_name = None, None
    birth_year = random.randint(1970, 2010)

    def pick_replacement(match):
        nonlocal last_first_names, last_last_name
        label = label_mapping[match.group(2).upper()]
        text = match.group(1)

        if label == "NOM" or label == "PRENOM":
            if label == "NOM" and (text.lower() in mem or last_last_name is not None):
                if text.lower() not in mem:
                    mem[text.lower()] = last_last_name
                res = make_last_name(mem[text.lower()], detect_name_format(text))
                last_last_name = None
                return res
            elif label == "PRENOM" and (
                text.lower() in mem or last_first_names is not None
            ):
                if text.lower() not in mem:
                    mem[text.lower()] = last_first_names
                res = make_first_name(mem[text.lower()], detect_name_format(text))
                last_first_names = None
                return res
            else:
                last_first_names, last_last_name = (
                    first_name,
                    last_name,
                ) = pick_fake_name(fake_names)

                if label == "NOM":
                    mem[text.lower()] = last_last_name
                    last_last_name = None
                    return make_last_name(last_name, detect_name_format(text))

                elif label == "PRENOM":
                    mem[text.lower()] = first_name
                    last_first_names = None
                    return make_first_name(first_name, detect_name_format(text))

        if label == "TEL":
            return generate_fake_phone_number()

        if label == "SECU":
            return generate_fake_nss(birth_year)

        if label == "ZIP":
            return generate_french_zipcode()

        if label == "VILLE":
            return pick_city(fake_cities)

        if label == "DATE_NAISSANCE":
            date_format = normalizer.extract_date(text)[2]
            return generate_random_date(year=birth_year, format=date_format)

        if label == "DATE":
            date_format = normalizer.extract_date(text)[2]
            return generate_random_date(format=date_format)

        if label == "MAIL":
            return generate_random_mail(fake_names)

        if label == "HOPITAL":
            return random.choice(hospitals)

        return None

    def make_substitute(match):
        rep = pick_replacement(match)
        if rep is not None:
            return f"[{rep}]({match.group(2)})"
        return match.group(0)

    return re.sub(r"\[([^\]]*)\] *\(([^\)]*)\)", make_substitute, sample)


# noinspection RegExpRedundantEscape
def parse_formatted_sample(text, check_parsing_errors=False):
    new_text = ""
    offset = 0
    ents = []
    for match in re.finditer(r"\[([^\]]*)\] *\(([^\)]*)\)", text):
        new_text = new_text + text[offset : match.start(0)]
        begin = len(new_text)
        new_text = new_text + match.group(1)
        end = len(new_text)
        offset = match.end(0)
        # print(match.group(0))
        ents.append(
            {"start": begin, "end": end, "label": label_mapping[match.group(2).upper()]}
        )
    new_text = new_text + text[offset:]
    # print(new_text)
    hash_id = hash(str(text) + str(ents)) % ((sys.maxsize + 1) * 2)
    # for ent in ents:
    #     print(f"{hash_id:<20}", new_text[ent["start"]:ent["end"]], "=>", ent["label"])
    if check_parsing_errors and re.findall(
        rf"\(({'|'.join(label_mapping)}*)\)", new_text
    ):
        print("Possible parsing error: {}".format(text))
    if check_parsing_errors and re.findall(r"\[([^\]]*)\]", new_text):
        print("Possible parsing error: {}".format(text))
    return {
        "note_id": f"gen-{hash_id}",
        "note_text": new_text,
        "entities": ents,
    }


@app.command(name="generate_dataset", registry=registry)
def generate_dataset(
    seed: int = 123,
    augmentations_ratio: int = 5,
    dummy_snippets_ratio: float = 2,
    staff_list_snippets_ratio: float = 0.5,
    target_words: int = 2_000_000,
):
    """
    Generate a synthetic dataset of fictitious medical notes annotated personal
    identifiable information (PII) such as names, phone numbers, dates, etc.
    I know, there are a lot of magic numbers in this code, but feel free to clone and
    adjust them as needed.

    Parameters
    ----------
    seed: int, optional
        Random seed for reproducibility, used by confit to set the seed.
    augmentations_ratio: int
        Number of augmented samples to generate for each template.
    dummy_snippets_ratio: float
        Ratio of dummy snippets to generate.
    staff_list_snippets_ratio: float
        Ratio of staff list snippets to generate.
    target_words: int
        Target number of words in the generated dataset.
    """
    gen_dataset = []
    nlp = edsnlp.blank("eds")
    fake_names, fake_cities = load_insee_deces()
    templates = Path("data/templates.txt").read_text().split("\n\n")
    total_words = 0
    dup_freq = 0.1
    dup_max_count = 10
    dup_count_probs = [(0.5**i) for i in range(dup_max_count)]
    dup_count_probs = [p / sum(dup_count_probs) for p in dup_count_probs]

    with tqdm(total=target_words, desc="Generating dataset") as bar:
        while total_words < target_words:
            augmented_templates = [
                augment_sample(s, fake_names=fake_names, fake_cities=fake_cities)
                for i in range(augmentations_ratio)
                for s in templates
            ]
            dummy_snippets = [
                make_dummy_sample(fake_names=fake_names)
                for _ in range(int(len(templates) * dummy_snippets_ratio))
            ]
            staff_list_snippets = [
                "\n".join(
                    make_dummy_sample(random.randint(0, 3), fake_names=fake_names)
                    for _ in range(random.randint(3, 10))
                )
                for _ in range(int(len(templates) * staff_list_snippets_ratio))
            ]
            for text in (
                *templates,
                *augmented_templates,
                *dummy_snippets,
                *staff_list_snippets,
            ):
                if random.random() < dup_freq:
                    lines = text.split("\n")
                    safe_lines = [
                        idx
                        for idx, line in enumerate(lines)
                        if line.count("[") == line.count("]")
                        and (
                            line.count("[") == 0
                            or (
                                line.index("[") < line.index("]")
                                and line.rindex("[") < line.rindex("]")
                            )
                        )
                    ]
                    if safe_lines:
                        dup_idx = random.choice(safe_lines)
                        # Duplicate n times the line with geometric probability dist
                        n = random.choices(
                            range(1, 1 + dup_max_count), dup_count_probs
                        )[0]
                        lines[dup_idx:dup_idx] = [lines[dup_idx]] * n
                        text = "\n".join(lines)
                if not text.strip():
                    continue
                doc = parse_formatted_sample(text)
                note_size = len(nlp(doc["note_text"]))
                total_words += note_size
                gen_dataset.append(doc)
                bar.update(note_size)
                if total_words > target_words:
                    break
    print(
        "Generated dataset with {} samples and {} words".format(
            len(gen_dataset), total_words
        )
    )
    write_path = Path("data/gen_dataset/train.jsonl")
    write_path.write_text("\n".join([json.dumps(d) for d in gen_dataset]))


if __name__ == "__main__":
    app()
