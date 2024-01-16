import datetime
import string
from typing import List

import regex
from spacy.tokens import Span
from typing_extensions import Literal

from edsnlp import registry
from edsnlp.core import PipelineProtocol
from edsnlp.pipes.base import BaseComponent
from edsnlp.pipes.misc.dates.models import AbsoluteDate
from edsnlp.utils.span_getters import SpanGetterArg, get_spans, validate_span_getter

from .patterns import full_regex, noun_regex, number_regex

JAVA_CHARS = set("GuyDMLdQqYwWEecFahKkHmsSAnNVzOXxZp'[]#{}")
DIGIT_TRANS = str.maketrans(string.digits, "?" * len(string.digits))


@registry.factory.register(
    "eds_pseudo.dates_normalizer",
    requires=["doc.ents", "doc.spans"],
    assigns=["token._.date", "token._.datetime", "token._.date_format"],
)
class DatesNormalizer(BaseComponent):
    """
    The `eds_pseudo.dates_normalizer` component takes date entities and parses them as
    dates using various French-specific heuristics (e.g., in an ambiguous case like
    `01/02/03`, we assume the leftmost digits represent the day, and the rightmost the
    year).

    This component also extract a closely matching date format, such that a new date
    replaced and formatted with it, looks approximately like the true one.

    Parameters
    ----------
    nlp: PipelineProtocol
        The pipeline object
    name: str = "eds.dates_normalizer"
        The name of the component
    span_getter: SpanGetterArg
        The spans to process as dates. Default to spans in `doc.ents` that have the
        "DATE" label.
    format: Literal['strftime', 'java']
        The format syntax to use for the parsed format :
        - `strftime` looks like '%d %m %Y' and is used in most non-Java languages
        - `java` looks like 'dd mm yyyy' and is used in Java-related applications like
          spark. It is also supported by the python `pendulum` library.
    """

    def __init__(
        self,
        nlp: PipelineProtocol,
        name: str = "eds.dates_normalizer",
        span_getter: SpanGetterArg = {"ents": ["DATE", "DATE_NAISSANCE"]},
        format: Literal["strftime", "java"] = "strftime",
    ):
        super().__init__(nlp=nlp, name=name)
        self.span_getter = validate_span_getter(span_getter)
        self.format = format

    def set_extensions(self):
        if not Span.has_extension("date"):
            Span.set_extension("date", default=None)
        if not Span.has_extension("datetime"):
            Span.set_extension("datetime", default=None)
        if not Span.has_extension("date_format"):
            Span.set_extension("date_format", default=None)

    def extract_date(self, s, next_date=None, next_offsets=None):
        date_conf = {}

        m = regex.search(full_regex, s)
        if m:
            date = AbsoluteDate(
                day=int(m.group("day").replace(" ", "").replace("Û", "0")),
                month=int(m.group("month").replace(" ", "").replace("Û", "0")),
                year=int(m.group("year").replace(" ", "").replace("Û", "0")),
            )
            date_offsets = sorted(
                [
                    (m.start("day"), m.end("day"), "d"),
                    (m.start("month"), m.end("month"), "m"),
                    (m.start("year"), m.end("year"), "y"),
                ]
            )
            if not m.group("spaced"):
                date_format = self.extract_format(s, date_offsets)
            else:
                date_format = "%d/%m/%Y" if self.format == "strftime" else "dd/MM/yyyy"
                date_format = (
                    self.escape(s[: date_offsets[0][0]])
                    + date_format
                    + self.escape(s[date_offsets[-1][1] :])
                )
            return date, date_offsets, date_format

        matches = []
        remaining = []

        for match in regex.finditer(noun_regex, s):
            if match.group("day_of_week"):
                remaining.append((match.start(), match.end(), "w", None))
            else:
                value = next(m for m, value in match.groupdict().items() if value)
                matches.append((match.start(), match.end(), "m", int(value[6:])))

        numbers = list(regex.finditer(number_regex, s))
        in_10_years_2_digits = 10 + datetime.datetime.now().year % 100
        for match in numbers:
            snippet = match.group()
            if snippet.replace("Û", "0").isdigit():
                snippet = snippet.replace("Û", "0")
                value = int(snippet)
            else:
                value = sum(int(m[1:]) for m, v in match.groupdict().items() if v)
            if value == 0:
                matches.append((match.start(), match.end(), ".", None))
            elif value <= 12:
                if len(snippet) == 1:
                    matches.append((match.start(), match.end(), "dm", value))
                else:
                    matches.append((match.start(), match.end(), "dmy", value))
            elif value <= 31:
                matches.append((match.start(), match.end(), "dy", value))
            elif value <= in_10_years_2_digits:
                # 2 digits like in 33 -> 2033
                matches.append((match.start(), match.end(), "y", 2000 + value))
            elif in_10_years_2_digits < value < 100:
                # TODO: arbitrate with the above rule
                matches.append((match.start(), match.end(), "y", 1900 + value))
            elif 1900 <= value <= 2100:
                matches.append((match.start(), match.end(), "y", value))
            elif 1900 <= int(snippet[:4]) <= 2100:
                value = int(snippet[:4])
                matches.append((match.start(), match.start() + 4, "y", value))
            else:
                matches.append((match.start(), match.end(), ".", value))

        matches = sorted(matches)

        pattern: List[str] = [m[2] for m in matches]  # type: ignore

        last = -1
        found = "".join(p for p in pattern if len(p) == 1)
        while len(found) != last:
            pattern = [(p.strip(found)) if len(p) > 1 else p for p in pattern]
            last = len(found)
            found = "".join(p for p in pattern if len(p) == 1)

        # In France, dates follow the d/m/y order
        if len(pattern) >= 2 and "d" in pattern[0] and "m" in pattern[1]:
            remaining = ["." if p in ("d", "m") else p for p in remaining]
            pattern = ["." if p in ("d", "m") else p for p in pattern]
            pattern[0] = "d"
            pattern[1] = "m"
        if len(pattern) >= 2 and "m" in pattern[-2] and "y" in pattern[-1]:
            remaining = ["." if p in ("m", "y") else p for p in remaining]
            pattern = ["." if p in ("m", "y") else p for p in pattern]
            pattern[-2] = "m"
            pattern[-1] = "y"
        if len(pattern) >= 3 and "y" in pattern[0] and "m" in pattern[1]:
            remaining = ["." if p in ("d", "m", "y") else p for p in remaining]
            pattern = ["." if p in ("d", "m", "y") else p for p in pattern]
            pattern[0] = "y"
            pattern[1] = "m"
            pattern[2] = "d"

        # Handle cases like [10] in "du [10] au [12/08/1995]"
        if (
            pattern
            and next_date is not None
            and "m" not in pattern
            and "y" not in pattern
            and next_date.month is not None
            and next_date.day is not None
            and next_offsets[0][2] not in "ym"
            and sum(["d" in p for p in pattern]) == 1
        ):
            if next_date.year is not None:
                date_conf["year"] = next_date.year
            date_conf["month"] = next_date.month
            found += "ym"
            pattern = ["d" if "d" in p else p for p in pattern]
        # Handle cases like [10 mai] in "du [10 mai] au [21 juin 1996]"
        elif (
            pattern
            and next_date is not None
            and "y" not in pattern
            and next_date.year is not None
            and next_date.month is not None
            and next_date.day is not None
            and next_offsets[0][2] not in "y"
            and sum(["m" in p for p in pattern]) == 1
            and sum(["d" in p for p in pattern]) == 1
        ):
            date_conf["year"] = next_date.year
            found += "y"
            pattern = ["d" if "d" in p else "m" if "m" in p else p for p in pattern]
        # Handle cases like [mai] in "de [mai] à [juin 1996]"
        elif (
            pattern
            and next_date is not None
            and "y" not in pattern
            and next_date.year is not None
            and next_date.month is not None
            and next_offsets[0][2] not in "yd"
            and sum(["m" in p for p in pattern]) == 1
        ):
            date_conf["year"] = next_date.year
            found += "y"
            pattern = ["m" if "m" in p else p for p in pattern]

        # Handle year found but missing month => remove day
        forbidden = ""
        found = set(found) | set(pattern + [m[2] for m in remaining])
        if "d" in found and "y" in found and "m" not in found:
            forbidden += "d"
            found.remove("d")

        if len(["d" in p for p in pattern]) == 1:
            pattern = ["d" if "d" in p else p for p in pattern]

        # Handle missing day => remove day of week
        if "w" in found and "d" not in found:
            forbidden += "w"

        matches = sorted(
            [
                (m[0], m[1], p if len(p) == 1 and p not in forbidden else ".", m[3])
                for m, p in (
                    *zip(matches, pattern),
                    *zip(remaining, [m[2] for m in remaining]),
                )
            ]
        )

        for m in matches:
            if m[2] == "d":
                date_conf["day"] = m[3]
            elif m[2] == "m":
                date_conf["month"] = m[3]
            elif m[2] == "y":
                date_conf["year"] = m[3]

        date = AbsoluteDate.parse_obj(date_conf)
        date_offsets = [(b, e, k) for b, e, k, v in matches]

        return date, date_offsets, self.extract_format(s, date_offsets)

    def escape(self, s):
        if self.format == "strftime":
            return s.translate(DIGIT_TRANS).replace("%", "%%") if s else ""

        # Test s against JAVA_CHARS
        if set(s).isdisjoint(JAVA_CHARS):
            return s
        return ("'" + s.translate(DIGIT_TRANS).replace("'", "") + "'") if s else ""

    def extract_format(self, s, matches):
        strftime = self.format == "strftime"
        date_format = ""
        offset = 0
        for begin, end, kind in matches:
            date_format += self.escape(s[offset:begin])
            snippet = s[begin:end].replace(" ", "")
            if kind == "d":
                if snippet.isdigit() and len(snippet) == 2:
                    date_format += "%d" if strftime else "dd"
                else:
                    date_format += "%-d" if strftime else "d"
            elif kind == "m":
                if snippet.isdigit():
                    if len(snippet) == 1:
                        date_format += "%-m" if strftime else "M"
                    else:
                        date_format += "%m" if strftime else "MM"
                else:
                    # 'mai' is the only month that has <= 3 letters,
                    # so we assume it's not intended as an abbreviation
                    if len(snippet) <= 3 and snippet != "mai":
                        date_format += "%b" if strftime else "MMM"
                    else:
                        date_format += "%B" if strftime else "MMMM"
            elif kind == "w":
                if len(snippet) <= 3:
                    date_format += "%a" if strftime else "EEE"
                else:
                    date_format += "%A" if strftime else "EEEE"
            elif kind == "y":
                if snippet.isdigit() and len(snippet) <= 2:
                    date_format += "%y" if strftime else "yy"
                else:
                    date_format += "%Y" if strftime else "yyyy"
            elif kind == ".":
                date_format += self.escape(s[begin:end])
            offset = end

        date_format += self.escape(s[offset:])

        return date_format

    def __call__(self, doc):
        spans = list(get_spans(doc, self.span_getter))
        last_date = None
        last_date_offsets = None
        last_span = None
        for span in reversed(spans):
            span: Span
            text = span.text
            if last_span and last_span.start - span.end > 3:
                last_date = last_date_offsets = None

            date, date_offsets, date_format = self.extract_date(
                text,
                last_date,
                last_date_offsets,
            )
            span._.date = date
            span._.datetime = date.to_datetime(doc._.note_datetime)
            span._.date_format = date_format

            last_span, last_date, last_date_offsets = span, date, date_offsets

        return doc
