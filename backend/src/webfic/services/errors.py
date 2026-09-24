class NotFound(Exception):
    """The resource does not exist or does not belong to the user."""


class InvalidEdit(ValueError):
    """A requested change to the text cannot be applied (e.g. the passage to replace is
    missing or ambiguous). Nothing was changed."""
