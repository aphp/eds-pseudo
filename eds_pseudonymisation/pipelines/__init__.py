from spacy.tokens import Doc

if not Doc.has_extension("context"):  # pragma: no cover
    Doc.set_extension("context", default=dict())
if not Doc.has_extension("note_id"):  # pragma: no cover
    Doc.set_extension("note_id", default=None)
if not Doc.has_extension("note_datetime"):  # pragma: no cover
    Doc.set_extension("note_datetime", default=None)
if not Doc.has_extension("note_class_source_value"):  # pragma: no cover
    Doc.set_extension("note_class_source_value", default=None)
