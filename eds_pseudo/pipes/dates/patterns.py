from edsnlp.pipes.misc.dates.patterns.absolute import absolute_pattern, mode_pattern
from edsnlp.pipes.misc.dates.patterns.atomic.delimiters import raw_delimiter_pattern
from edsnlp.utils.regex import make_pattern

numeric_day_pattern = r"(?<!\d)(0 [1-9]|[12] \d|3 [01])(?!\d)"
numeric_month_pattern = r"(?<!\d)(0 [1-9]|1 [0-2])(?!\d)"
year_pattern = make_pattern(
    [r"1 9 \d \d", *(" ".join(str(year)) for year in range(2000, 2024))],
    name="year",
)

pseudo_date_pattern = [
    rf"(?<={mode_pattern}.{{,3}})?(?P<day>{numeric_day_pattern}) (?P<month>{numeric_month_pattern}) {year_pattern}",
    *absolute_pattern,
]
false_positive_pattern = make_pattern(
    [
        rf"(\d+{raw_delimiter_pattern}){{3,}}\d+(?!:\d\d)\b",
        r"\d/\d",
    ]
)
