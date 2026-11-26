import hashlib
import json
import random
import re
import unicodedata
from collections import defaultdict
from datetime import date
from itertools import accumulate
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import edsnlp
from eds_pseudo.data import normalize_record, text_hash

MONTHS = (
    "janvier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
)
SHORT_MONTHS = ("jan", "fév", "mars", "avr", "mai", "juin", "juil", "août", "sept", "oct", "nov", "déc")
# format_date compares the month word of a seed with these names without accents, so that "mars" or "aout" keep
# a full month name while "nov" or "juil" get an abbreviation
FULL_MONTHS = {unicodedata.normalize("NFKD", month).encode("ascii", "ignore").decode() for month in MONTHS}
MONTH_PATTERN = r"jan\w*|f[ée]v\w*|mars|avr\w*|mai|juin|juil\w*|ao[uû]t|sept\w*|oct\w*|nov\w*|d[ée]c\w*"
REFERENCE_DATE = date(2025, 1, 1)


def value_shape(value: str) -> tuple:
    """
    Rough shape of a value (short or long, number of words, hyphen or not), e.g. `('long', 1, True)` for
    "A.CHENEVIER-H.MONDOR", so that `pick_value` replaces a value with one that looks like it
    """
    return (
        "short" if sum(character.isalpha() for character in value) <= 4 else "long",
        min(len(value.split()), 3),
        "-" in value,
    )


def load_values(path: Optional[Path]) -> Dict[tuple, tuple]:
    """
    Read the `{label, value, weight}` lines of a values file (e.g. the hospital names of
    `data/replacement-values.jsonl`), indexed by label and shape for `pick_value`
    """
    values = defaultdict(lambda: ([], []))
    for item in map(json.loads, Path(path).read_text(encoding="utf-8").splitlines()) if path else ():
        for key in ((item["label"], value_shape(item["value"])), (item["label"], None)):
            values[key][0].append(item["value"])
            values[key][1].append(item.get("weight", 1))
    return values


def load_people(path: Optional[Path]) -> tuple:
    """
    Read the `{PRENOM, NOM, weight}` lines of an optional people file, which `replace_record` uses instead of
    Faker names when the generation config gives one
    """
    people, weights = [], []
    for item in map(json.loads, Path(path).read_text(encoding="utf-8").splitlines()) if path else ():
        people.append((item["PRENOM"], item["NOM"]))
        weights.append(item.get("weight", 1))
    return people, list(accumulate(weights))


def pick_value(label: str, original: str, fallback, context: Dict[str, Any], probability: float = 1) -> str:
    """
    Draw a value of the same label and shape as `original` from the values file and the template values, or call
    `fallback` (a Faker provider) when there is none
    """
    values = context["values"].get((label, value_shape(original))) or context["values"].get((label, None))
    return (
        context["random"].choices(values[0], cum_weights=values[1])[0]
        if values and context["random"].random() < probability
        else fallback()
    )


def format_case(original: str, value: str) -> str:
    """
    Give a generated value the casing of the seed value, e.g. "MARTIN" for "DUPONT", and keep initials as
    initials, e.g. "M." for "J."
    """
    if sum(character.isalpha() for character in original) <= 2:
        generated = iter(character.upper() for character in value if character.isalpha())
        return "".join(next(generated, character) if character.isalpha() else character for character in original)
    if original.isupper():
        return value.upper()
    if original.islower():
        return value.lower()
    return value.title() if original.istitle() else value


def format_name(original: str, provider, extra_parts: int = 0) -> str:
    """
    Replace each word of a name with a generated one and keep the separators, e.g. "Marc-Louis" for
    "Jean-Pierre", and sometimes add `extra_parts` words to a one-word name
    """
    value = re.sub(
        r"[^\W\d_]+",
        lambda match: format_case(match.group(), re.search(r"[^\W\d_]+", provider()).group()),
        original,
    )
    for _ in range(extra_parts if len(re.findall(r"[^\W\d_]+", original)) == 1 else 0):
        value += " " + format_case(original, re.search(r"[^\W\d_]+", provider()).group())
    return value


def format_digits(original: str, digits: str) -> str:
    """
    Replace the digits of an identifier or a phone number and keep its separators, e.g. "07 98 76 54 32" for
    "06 12 34 56 78"
    """
    generated = iter(digits)
    return "".join(next(generated, character) if character.isdigit() else character for character in original)


def format_date(original: str, value: date) -> str:
    """
    Write a date in the format of the seed date, e.g. "14/07/2019" for "12/03/1961" and "07/2019" for
    "03/2021", with a French month word when the seed has one
    """
    month = re.search(MONTH_PATTERN, original, flags=re.I)
    groups = list(re.finditer(r"\d+", original))
    result = original
    if month:
        is_full = unicodedata.normalize("NFKD", month.group().lower()).encode("ascii", "ignore").decode() in FULL_MONTHS
        result = (
            original[: month.start()]
            + format_case(month.group(), (MONTHS if is_full else SHORT_MONTHS)[value.month - 1])
            + original[month.end() :]
        )
        numbers = (
            [max(1, value.day - 3), value.day]
            if len(groups) == 2 and all(group.end() <= month.start() for group in groups)
            else [
                value.year if len(group.group()) >= 3 or len(groups) == 1 and group.start() > month.end() else value.day
                for group in groups
            ]
        )
    elif len(groups) >= 3:
        numbers = (
            [value.year, value.month, value.day]
            if len(groups[0].group()) == 4
            else [value.day, value.month, value.year]
        )
    elif len(groups) == 2 and all(len(group.group()) >= 3 for group in groups):
        numbers = [value.year - 1, value.year]
    elif len(groups) == 2:
        numbers = [value.month, value.year] if len(groups[1].group()) >= 3 else [value.day, value.month]
    elif groups and len(groups[0].group()) in {6, 8}:
        digits = f"{value.day:02d}{value.month:02d}{value.year:04d}"
        return digits if len(groups[0].group()) == 8 else digits[:4] + digits[-2:]
    elif groups:
        numbers = [value.year if len(groups[0].group()) >= 3 else value.day]
    else:
        return value.strftime("%d/%m/%Y")
    replacements = iter(numbers)

    def replace(group):
        number = next(replacements, None)
        if number is None:
            return group.group()
        if len(group.group()) == 2 and number > 99:
            number %= 100
        return str(number).zfill(len(group.group()))

    return re.sub(r"\d+", replace, result)


def replacement(label: str, original: str, context: Dict[str, Any]) -> str:
    """
    Generate the replacement of one entity value, from Faker, the values file or random digits depending on its
    label, e.g. an email built from the replaced first and last names and the seed domain
    """
    faker = context["faker"]
    rng = context["random"]
    birth = context["birth_date"]
    if label == "PRENOM":
        return pick_value(
            label,
            original,
            lambda: format_name(original, faker.first_name, rng.choices([0, 1, 2], [96, 3.5, 0.5])[0]),
            context,
        )
    if label == "NOM":
        return pick_value(
            label,
            original,
            lambda: format_name(original, faker.last_name, rng.choices([0, 1, 2], [99, 1, 0])[0]),
            context,
        )
    if label == "DATE_NAISSANCE":
        return format_date(original, birth)
    if label == "DATE":
        return format_date(original, faker.date_between_dates(date(1970, 1, 1), REFERENCE_DATE))
    if label == "MAIL":
        first, last = context["person"]
        local = unicodedata.normalize("NFKD", f"{first}.{last}").encode("ascii", "ignore").decode().lower()
        domain = original.rsplit("@", 1)[-1].replace(" ", "") if "@" in original else faker.free_email_domain()
        return f"{local}@{domain}"
    if label == "SECU":
        digits = f"{rng.choice('12')}{birth.year % 100:02d}{birth.month:02d}" + "".join(rng.choices("0123456789", k=10))
        return format_digits(original, digits)
    if label in {"IPP", "NDA", "TEL", "ZIP"}:
        return format_digits(
            original, "".join(rng.choices("0123456789", k=sum(character.isdigit() for character in original)))
        )
    providers = {
        "ADRESSE": faker.street_address,
        "HOPITAL": lambda: "Hôpital " + faker.last_name(),
        "VILLE": faker.city,
    }
    return pick_value(label, original, providers[label], context)


def replace_record(record: Mapping[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Replace every entity of a seed and recompute the offsets. Since a name often appears twice in a note, we
    replace the same value with the same replacement, and the birth date used for `DATE_NAISSANCE` also gives the
    year and month of the `SECU` number.
    """
    record = normalize_record(record)
    state = {
        **context,
        "birth_date": context["faker"].date_between_dates(REFERENCE_DATE.replace(year=1925), REFERENCE_DATE),
        "cache": {},
    }
    configured_people = context["people"]
    configured_person = None
    if configured_people[0] and context["random"].random() < context["people_probability"]:
        configured_person = context["random"].choices(configured_people[0], cum_weights=configured_people[1])[0]
    person = []
    for idx, label in enumerate(("PRENOM", "NOM")):
        entity = next((entity for entity in record["entities"] if entity["label"] == label), None)
        original = record["note_text"][entity["start"] : entity["end"]] if entity else ""
        if configured_person:
            value = configured_person[idx]
        elif original:
            value = replacement(label, original, state)
        else:
            value = context["faker"].first_name() if label == "PRENOM" else context["faker"].last_name()
        state["cache"][(label, original.casefold())] = value
        person.append(value)
    state["person"] = person
    text, entities, cursor = "", [], 0
    for entity in sorted(record["entities"], key=lambda item: (item["start"], item["end"])):
        text += record["note_text"][cursor : entity["start"]]
        original = record["note_text"][entity["start"] : entity["end"]]
        key = (entity["label"], original.casefold())
        value = state["cache"].setdefault(key, replacement(entity["label"], original, state))
        value = format_case(original, value) if entity["label"] in {"NOM", "PRENOM", "VILLE", "HOPITAL"} else value
        start = len(text)
        text += value
        entities.append({"start": start, "end": len(text), "label": entity["label"]})
        cursor = entity["end"]
    return normalize_record({**record, "note_text": text + record["note_text"][cursor:], "entities": entities})


def insert_text(
    record: Dict[str, Any], offset: int, value: str, copied: Sequence[Mapping[str, Any]] = ()
) -> Dict[str, Any]:
    """
    Insert `value` at `offset` and shift the entities after it, and add `copied` entities inside the inserted
    text, so that `generate_note` can repeat a line or add blank lines without breaking offsets
    """
    entities = [
        {
            **entity,
            "start": entity["start"] + (len(value) if entity["start"] >= offset else 0),
            "end": entity["end"] + (len(value) if entity["end"] > offset else 0),
        }
        for entity in record["entities"]
    ]
    entities.extend({**entity, "start": entity["start"] + offset, "end": entity["end"] + offset} for entity in copied)
    text = record["note_text"]
    return normalize_record({**record, "note_text": text[:offset] + value + text[offset:], "entities": entities})


def generate_note(
    seeds: Sequence[Mapping[str, Any]], context: Dict[str, Any], params: Mapping[str, Any]
) -> Dict[str, Any]:
    """
    Generate one record from one seed: replace its entities (or keep them with `replace: false`), then sometimes
    repeat one of its lines and add blank lines
    """
    rng = context["random"]
    record = normalize_record(rng.choice(seeds))
    if params.get("drop_entities"):
        cursor = 0
        text = ""
        for entity in sorted(record["entities"], key=lambda item: (item["start"], item["end"])):
            text += record["note_text"][cursor : entity["start"]]
            cursor = entity["end"]
        record = normalize_record({**record, "note_text": text + record["note_text"][cursor:], "entities": []})
    elif params.get("replace", True):
        record = replace_record(record, context)
    if rng.random() < params.get("duplicate_line_probability", 0):
        lines = list(re.finditer(r"[^\n]+(?:\n|$)", record["note_text"]))
        line = rng.choice(lines)
        prefix = "" if line.group().endswith("\n") else "\n"
        copied = [
            {
                **entity,
                "start": entity["start"] - line.start() + len(prefix),
                "end": entity["end"] - line.start() + len(prefix),
            }
            for entity in record["entities"]
            if line.start() <= entity["start"] and entity["end"] <= line.end()
        ]
        record = insert_text(record, line.end(), prefix + line.group(), copied)
    weights = params.get("extra_newline_weights", [1])
    for offset in reversed([match.end() for match in re.finditer(r"\n", record["note_text"])]):
        record = insert_text(record, offset, "\n" * rng.choices(range(len(weights)), weights=weights)[0])
    return record


def generate_list(
    seeds: Sequence[Mapping[str, Any]], context: Dict[str, Any], params: Mapping[str, Any]
) -> Dict[str, Any]:
    """
    Generate one record by joining between `min_items` and `max_items` replaced row seeds, e.g. a list of staff
    names and roles
    """
    rng = context["random"]
    rows = [
        replace_record(rng.choice(seeds), context) for _ in range(rng.randint(params["min_items"], params["max_items"]))
    ]
    separator = params.get("separator", "\n")
    text, entities, parents = "", [], []
    for row in rows:
        offset = len(text)
        text += (separator if text else "") + row["note_text"]
        shift = len(separator) if offset else 0
        entities.extend(
            {**entity, "start": entity["start"] + offset + shift, "end": entity["end"] + offset + shift}
            for entity in row["entities"]
        )
        parents.append(str(row.get("note_id") or row.get("source_id")))
    return normalize_record({"note_text": text, "entities": entities, "parent_ids": parents})


TEMPLATES = {"note": generate_note, "list": generate_list, "staff-list": generate_list}


def generate_records(
    seeds: Sequence[Mapping[str, Any]],
    target_words: int,
    templates: Sequence[Mapping[str, Any]],
    seed: int,
    document_prefix: str,
    values_path: Optional[Path] = None,
    people_path: Optional[Path] = None,
    people_probability: float = 1,
    excluded_hashes: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    """
    Generate synthetic records from the seeds of one split until they hold `target_words` words, following the
    `templates` of the `generation` section of a source (e.g. in `configs/config.yml`). `prepare-data` calls it once
    per split.

    1. Read the replacement values and select the seeds of each template.
    2. Give each seed group (one seed for a note template, the whole pool for a list template) its own random
       generator, seeded from `seed` and the ids of its seeds, so that the copies of a seed don't depend on the order
       of the seeds.
    3. Generate from the template that is the furthest below its share of words, and within it from the seed group
       used the least, and skip records whose text is in `excluded_hashes`.
    """
    from faker import Faker

    # 1. Read the replacement values and select the seeds of each template
    values = load_values(values_path)
    people = load_people(people_path)
    # Addresses and cities of the seeds become replacement values for the other seeds
    for record in seeds:
        for entity in record["entities"]:
            if entity["label"] in {"ADRESSE", "VILLE"}:
                value = record["note_text"][entity["start"] : entity["end"]]
                for key in ((entity["label"], value_shape(value)), (entity["label"], None)):
                    values[key][0].append(value)
                    values[key][1].append(1)
    values = {key: (items[0], list(accumulate(items[1]))) for key, items in values.items()}
    pools = {
        template["name"]: [
            record
            for record in seeds
            if record.get("template", "note") in template.get("seed_templates", ["note"])
            and (not template.get("require_entities") or record["entities"])
            and (template.get("has_entities") is None or bool(record["entities"]) == template["has_entities"])
            and (not template.get("seed_splits") or record.get("split") in template["seed_splits"])
            and len(record["note_text"].split()) <= template.get("max_seed_words", float("inf"))
            and len(record["entities"]) <= template.get("max_seed_entities", float("inf"))
        ]
        for template in templates
    }
    templates = [template for template in templates if pools[template["name"]]]
    # 2. Give each seed group its own random generator
    seed_groups, contexts = [], []
    for template in templates:
        template_groups = (
            [[record] for record in pools[template["name"]]]
            if template.get("generator", template["name"]) == "note"
            else [pools[template["name"]]]
        )
        seed_groups.append(template_groups)
        template_contexts = []
        for group in template_groups:
            group_key = (
                template["name"]
                + ":"
                + "|".join(
                    str(record.get("note_id") or record.get("source_id") or text_hash(record["note_text"]))
                    for record in group
                )
            )
            group_seed = int.from_bytes(hashlib.sha256(f"{seed}:{group_key}".encode()).digest()[:8], "big")
            faker = Faker("fr_FR")
            faker.seed_instance(group_seed)
            template_contexts.append(
                {
                    "random": random.Random(group_seed),
                    "faker": faker,
                    "values": values,
                    "people": people,
                    "people_probability": people_probability,
                    "group_id": hashlib.sha256(group_key.encode()).hexdigest()[:12],
                }
            )
        contexts.append(template_contexts)
    # 3. Generate from the template furthest below its share of words, then from its least used seed group
    tokenizer = edsnlp.blank("eds").tokenizer
    records, words, excluded_hashes = [], 0, set(excluded_hashes)
    template_words = [0] * len(templates)
    group_counts = [[0] * len(template_groups) for template_groups in seed_groups]
    while words < target_words:
        template_idx = min(range(len(templates)), key=lambda idx: template_words[idx] / templates[idx]["weight"])
        group_idx = min(range(len(seed_groups[template_idx])), key=group_counts[template_idx].__getitem__)
        template = templates[template_idx]
        context = contexts[template_idx][group_idx]
        generation = group_counts[template_idx][group_idx]
        group_counts[template_idx][group_idx] += 1
        record = TEMPLATES[template.get("generator", template["name"])](
            seed_groups[template_idx][group_idx], context, template
        )
        normalized_hash = text_hash(record["note_text"])
        if normalized_hash in excluded_hashes:
            continue
        parents = record.get("parent_ids") or [record.get("note_id") or record.get("source_id")]
        group = hashlib.sha256("|".join(map(str, parents)).encode()).hexdigest()
        record.update(
            note_id=f"{document_prefix}-{context['group_id']}-{generation}",
            source_id=group,
            group_id=group,
            template=template["name"],
            parent_ids=parents,
        )
        records.append(record)
        record_words = len(tokenizer(record["note_text"]))
        words += record_words
        template_words[template_idx] += record_words
    return records
