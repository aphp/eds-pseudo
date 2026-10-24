from spacy.tokens import Doc

for extension in (
    "context",
    "source_id",
    "group_id",
    "question_id",
    "encounter_id",
    "person_id",
    "source",
    "split",
    "note_id",
    "note_datetime",
    "note_class_source_value",
):
    if not Doc.has_extension(extension):  # pragma: no cover
        Doc.set_extension(extension, default=None)
