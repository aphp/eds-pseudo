noun_regex = r"""(?xi)
# No letter before
(?<![[:alpha:]])
(?:
    # Either a day of week
    (?P<day_of_week>
        lun[[:alpha:]]*|
        mard[[:alpha:]]*|
        mer[[:alpha:]]*|
        jeu[[:alpha:]]*|
        vend[[:alpha:]]*|
        sam[[:alpha:]]*|
        dim[[:alpha:]]*
    )|
    # Or a month
    (?P<month_1>jan[[:alpha:]]*)|
    (?P<month_2>fev[[:alpha:]]*|fév[[:alpha:]]*)|
    (?P<month_3>mar[[:alpha:]]*)|
    (?P<month_4>avr[[:alpha:]]*)|
    (?P<month_5>mai)|
    (?P<month_6>jun[[:alpha:]]*|juin[[:alpha:]]*)|
    (?P<month_7>jul[[:alpha:]]*|juil[[:alpha:]]*)|
    (?P<month_8>aou[[:alpha:]]*|aoû[[:alpha:]]*)|
    (?P<month_9>sep[[:alpha:]]*(?<!sept))|
    (?P<month_10>oct[[:alpha:]]*)|
    (?P<month_11>nov[[:alpha:]]*)|
    (?P<month_12>dec[[:alpha:]]*|déc[[:alpha:]]*)
)
# Abbreviation marker
\.?
"""

# no delimiter with space between numbers, ex: d d m m y y y y
full_regex = r"""(?x)
(?P<classic>
    (?P<day>[0Û][1-9]|[12][\dÛ]|3[Û01])
    [.-/]?(?P<month>[0Û][1-9]|1[Û012])
    [.-/]?(?P<year>(?:19|2[0Û])?[\dÛ][\dÛ])
)
|
(?P<reverse>
    (?P<year>(?:19|2[0Û])[\dÛ][\dÛ])
    [.-/]?(?P<month>[0Û][1-9]|1[Û012])
    [.-/]?(?P<day>[0Û][1-9]|[12][\dÛ]|3[Û01])
)
|
(?P<spaced>
    (?P<day>[0ÛO]\ +[1-9]|[12]\ +[\dÛ]|3\ +[Û01])
    \ *[.-/]?\ *(?<!\d)(?P<month>[0ÛO]\ +[1-9]|1\ +[Û012])
    \ *[.-/]?\ *(?<!\d)(?P<year>
        (?:1\ +9|2\ +[0ÛO])\ +[\dÛ]\ +[\dÛ]
        |[\dÛ]\ +[\dÛ]
    )
)
"""

_thousands = r"""
(?:
    (?P<_1900>mil[[:alpha:]]*[-\s]?neuf[-\s]?cents?)|
    (?P<_2000>deux[-\s]?mil[[:alpha:]]*)
)
"""

_tens = r"""
(?:
    (?P<_20>vingt-?\s*et|vingt)|
    (?P<_30>trente-?\s*et|trente)|
    (?P<_40>quarante-?\s*et|quarante)|
    (?P<_50>cinquante-?\s*et|cinquante)|
    (?P<_60>soixante-?\s*et|soixante)|
    (?P<_80>quatres?-?\s*vingts?-?\s*et|quatres?-?\s*vingts?)
)
"""

_units = r"""
(?:
    (?P<_1>premier|1\s*er|un)|
    (?P<_2>deux)|
    (?P<_3>trois)|
    (?P<_4>quatres?)|
    (?P<_5>cinq)|
    (?P<_6>six)|
    (?P<_7>sept)|
    (?P<_8>huit)|
    (?P<_9>neuf)|
    (?P<_17>dix-?\s*septs?)|
    (?P<_18>dix-?\s*huits?)|
    (?P<_19>dix-?\s*neufs?)|
    (?P<_10>dix)|
    (?P<_11>onzes?)|
    (?P<_12>douzes?)|
    (?P<_13>treizes?)|
    (?P<_14>quatorzes?)|
    (?P<_15>quinzes?)|
    (?P<_16>seizes?)
)
"""

letter_numbers = "|".join(
    r"[-\s]?".join(filter(bool, (thousand, ten, unit)))
    for thousand in [_thousands, None]
    for ten in [_tens, None]
    for unit in [_units, None]
)

number_regex = rf"""(?x)
# Numbers with letters: we will sum the value of the matched groups (_value)
(?i:
    (?<![[:alpha:]])(?:{letter_numbers})
)|
# Numbers with digits (0 is sometime wrongly parsed in PDFs as Û or O)
(?:\d|(?<![[:alpha:]])[ÛO]+(?![[:alpha:]]))+
"""
