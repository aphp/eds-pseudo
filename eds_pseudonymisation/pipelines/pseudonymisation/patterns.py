import re

from edsnlp.utils.regex import make_pattern

# Phones
delimiters = ["", r"\.", r"\-", " "]
# phone_pattern = make_pattern(
#    [
#        r"((\+ ?3 ?3|0 ?0 ?3 ?3)|0 ?[1-8]) ?" + d.join([r"(\d ?){2}"] * 4)
#        for d in delimiters
#    ]
# )
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
nda_pattern = (
    r"("
    r"(?<!\d[ .-]{,3})\b"
    r"(?:0 ?[159]|1 ?[0146]|2 ?[12689]|3 ?[2368]|4 ?[12479]"
    r"|5 ?3|6 ?[14689]|7 ?[2369]|8 ?[478]|9 ?[0569]|A ?G) ?(\d ?){7,8}"
    r"\b(?![ .-]{,3}\d)"
    r")"
)

# Mail
mail_pattern = r"""(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?: ?\. ?[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*") ?@ ?(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])? ?\. ?)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?) ?\. ?){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])"""  # noqa


# Zip
zip_pattern = (
    r"("
    r"(?<!\d[ .-°]{,3})\b"
    r"((?:(?:[0-8] [0-9])|(9 [0-5])|(2 [abAB]))\s?([0-9] ){2}[0-9]|"
    r"((?:(?:[0-8][0-9])|(9[0-5])|(2[abAB]))\s?([0-9]){2}[0-9]))"
    r"\b(?![ .-]{,3}\d)"
    r")"
)

# NSS
nss_pattern = (
    r"("
    r"(?<!\d[ .-]{,3})\b"
    r"(?:[1-37-8]) ?(?:([0-9] ?){2}) ?"
    r"(?:0 ?[0-9]|[2-35-9] ?[0-9]|[14] ?[0-2]) ?"
    r"(?:(?:0 ?[1-9]|[1-8] ?[0-9]|9 ?[0-69]|2 ?[abAB]) ?"
    r"(?:0 ?0 ?[1-9]|0 ?[1-9] ?[0-9]|"
    r"[1-8] ?([0-9] ?){2}|9 ?[0-8] ?[0-9]|9 ?9 ?0)|(?:9 ?[78] ?[0-9]) ?"
    r"(?:0 ?[1-9]|[1-8] ?[0-9]|9 ?0) ?)"
    r"(?:0 ?0 ?[1-9]|0 ?[1-9] ?[0-9]|[1-9] ?([0-9] ?){2})"
    r"(?:0 ?[1-9]|[1-8] ?[0-9]|9 ?[0-7])"
    r"\b(?![ .-]{,3}\d)"
    r")"
)

person_patterns = [
    r"""(?x)
(?<![/+])
\b
(?:[Dd]r[.]?|[Dd]octeur|[mM]r?[.]?|[Ii]nterne[ ]?:|[Ee]xterne[ ]?:|[Mm]onsieur|[Mm]adame|[Rr].f.rent[ ]?:|[P]r[.]?|[Pp]rofesseure?|[Mm]me[.]?|[Ee]nfant|[Mm]lle)[ ]+
(?:
    (?P<LN0>[A-Z]{2,}(?:[ ]*(?:-[ ]*)?(?:ep[.]|de|[A-Z]+))*)[ ]+(?P<FN0>[A-Z]\p{Ll}+(?:[ ]*(?:-[ ]*)?[A-Z]\p{Ll}+)*)
    |(?P<FN1>[A-Z]\p{Ll}+(?:[ ]*(?:-[ ]*)?[A-Z]\p{Ll}+)*)[ ]+(?P<LN1>[A-Z]{2,}(?:[ ]*(?:-[ ]*)?(?:ep[.]|de|[A-Z]+))*)
    |(?P<LN3>[A-Z]\p{Ll}+(?:(?:-|[ ]de[ ]|[ ]ep[.][ ])[A-Z]\p{Ll}+)*)[ ]+(?P<FN2>[A-Z]\p{Ll}+(?:-[A-Z]\p{Ll}+)*)
    |(?P<LN2>[A-Z][A-Z\p{Ll}-]+(?:[ ]*(?:-[ ]*)?[A-Z][A-Z\p{Ll}-]+)*)
)
\b(?![/+])
""",
    r"""
\b
(?<![/+%])
(?P<FN0>[A-Z][.])\s+(?P<LN0>[A-Z][A-Z\p{Ll}-]+(?:[ ]*(?:-[ ]*)?[A-Z][A-Z\p{Ll}-]+)*)
\b(?![/+%])
""",
]

common_medical_terms = {
    "EVA",
    "GE",
    "BIO",
    "SAU",
    "MEDICAL",
    "PA",
    "AVC",
    "PO",
    "OMS",
    "IVA",
    "AD",
}


street_patterns = ["rue", "route", "allée", "boulevard", "bd", "chemin"]
street_name_piece = "(?:[A-Z][A-Za-zéà]+|de|du|la|le|des)"
address_patterns = [
    rf"""(?x)
(
    (?:([1-9]\d*)\s+)?
    (?i:(?:(?:{'|'.join(map(re.escape, street_patterns))})\s+))
    (?:(?:{street_name_piece}\s+)*{street_name_piece})
)
(?=
    (?:[,]?\s*(?P<VILLE>[A-Z]+)\s+)?
    (?i:(?P<ZIP>(?:\s+\d{{2}}\s*?\d{{3}})|(?:[1-9]|1[0-9]|20)[èe]m?e?)?)
)
"""
]

patterns = dict(
    # ADRESSE=address_patterns,
    # DATE
    # DATE_NAISSANCE
    # HOPITAL
    IPP=ipp_pattern,
    MAIL=mail_pattern,
    TEL=phone_pattern,
    # NDA=nda_pattern,
    # NOM
    # PRENOM
    SECU=nss_pattern,
    # VILLE
    # ZIP=zip_pattern,
)
