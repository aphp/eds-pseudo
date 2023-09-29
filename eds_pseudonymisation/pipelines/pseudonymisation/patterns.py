# ruff: noqa: E501


# Phones
delimiters = ["", r"\.", r"\-", " "]

phone_pattern = (
    r"("
    r"(?<!\d[ .-]{,3})\b"
    r"((?:(?:\+|00)33\s?[.]\s?(?:\(0\)\s?[.]\s?)?|0)[1-9](?:(?:[.]\d{2}){4}|\d{2}(?:[.]\d{3}){2})(?![\d])"
    r"|(?:(?:\+|00)33\s?[-]\s?(?:\(0\)\s?[-]\s?)?|0)[1-9](?:(?:[-]\d{2}){4}|\d{2}(?:[-]\d{3}){2})(?![\d])"
    r"|(?:(?:\+|00)33\s?[-]\s?(?:\(0\)\s)?|0)[1-9](?:(?:[ ]?\d{2}){4}|\d{2}(?:[ ]?\d{3}){2})(?![\d]))"
    r"\b(?![ .-]{,3}\d)"
    r")"
)

# IPP
ipp_pattern = r"(" r"(?<!\d[ .-]{,3})\b" r"(8(\d ?){9})" r"\b(?![ .-]{,3}\d)" r")"

# NDA
nda_pattern = r"""(?x)
(?<=(?:
    (?:(?i:
        (?:(?:no|n°|numero|no\s+d[e'‘]|n°\s+d[e'‘]|numero\s+d[e'‘])\s+)?
        (?:examen|demande|sejour|dossier)
    ))
|(?:Examen|Demande|Sejour)
)\s*:?\s*)
\b
(
    \d{2,}[A-Z]?[A-Z]?\d*(?:[-]\d+)?
    |\d*[A-Z]?[A-Z]?\d{2,}(?:[-]\d+)?
)
\b
(?![-/+\\_])
"""

# Mail
mail_pattern = r"""(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?: ?\. ?[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*") ?@ ?(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])? ?\. ?)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?) ?\. ?){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])"""

# NSS
nss_pattern = r"""(?x)
# No digits just before on the same line
(?<!\d[ .-/+=_]{,3})\b
(
# Sex
    (?:[1-2])[ ]?
# Year of birth
    (?:([0-9][ ]?){2})[ ]?
# Month of birth
    (?:0[ ]?[0-9]|[2-35-9][ ]?[0-9]|[14][ ]?[0-2])[ ]?
# Location of birth
    (?:
        (?:
            0[ ]?[1-9]
            |[1-8][ ]?[0-9]
            |9[ ]?[0-69]
            |2[ ]?[abAB]
        )[ ]?
        (?:
            0[ ]?0[ ]?[1-9]|0[ ]?[1-9][ ]?[0-9]|
            [1-8][ ]?([0-9][ ]?){2}|9[ ]?[0-8][ ]?[0-9]|9[ ]?9[ ]?0
        )
        |(?:9[ ]?[78][ ]?[0-9])[ ]?(?:0[ ]?[1-9]|[1-8][ ]?[0-9]|9 ?0)
    )[ ]?
# Birth number 001-999
    (?:0[ ]?0[ ]?[1-9]|0[ ]?[1-9][ ]?[0-9]|[1-9][ ]?([0-9][ ]?){2})[ ]?
# Control key
    (?:0[ ]?[1-9]|[1-8][ ]?[0-9]|9[ ]?[0-7])
|
# Temporary NSS
    [3478][ ]?(?:[0-9][ ]?){14}
)
# Not followed by digits on the same line
\b(?![ .-]{,3}\d)
"""

# PERSON (FIRSTNAME AND LASTNAME)
Xxxxx = r"[A-Z]\p{Ll}+"
XXxX_ = r"[A-Z][A-Z\p{Ll}-]"
sep = r"(?:[ ]*|-)?"
person_patterns = [
    rf"""(?x)
(?<![/+])
\b
(?:[Dd]r[.]?|[Dd]octeur|[mM]r?[.]?|[Ii]nterne[ ]?:|[Ee]xterne[ ]?:|[Mm]onsieur|[Mm]adame|[Rr].f.rent[ ]?:|[P]r[.]?|[Pp]rofesseure?|[Mm]me[.]?|[Ee]nfant|[Mm]lle)[ ]+
(?:
    (?P<LN0>[A-Z][A-Z](?:{sep}(?:ep[.]|de|[A-Z]+))*)[ ]+(?P<FN0>{Xxxxx}(?:{sep}{Xxxxx})*)
    |(?P<FN1>{Xxxxx}(?:{sep}{Xxxxx})*)[ ]+(?P<LN1>[A-Z][A-Z]+(?:{sep}(?:ep[.]|de|[A-Z]+))*)
    |(?P<LN3>{Xxxxx}(?:(?:-|[ ]de[ ]|[ ]ep[.][ ]){Xxxxx})*)[ ]+(?P<FN2>{Xxxxx}(?:-{Xxxxx})*)
    |(?P<LN2>{XXxX_}+(?:{sep}{XXxX_}+)*)
)
\b(?![/+])
""",
    rf"""
\b
(?<![/+%])
(?P<FN0>[A-Z][.])\s+(?P<LN0>{XXxX_}+(?:{sep}{XXxX_}+)*)
\b(?![/+%])
""",
]

patterns = dict(
    IPP=ipp_pattern,
    MAIL=mail_pattern,
    TEL=phone_pattern,
    NDA=nda_pattern,
    SECU=nss_pattern,
)
