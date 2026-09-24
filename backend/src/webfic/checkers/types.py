import hashlib
from enum import StrEnum

from pydantic import BaseModel


class Confidence(StrEnum):
    CONFIRMED = "confirmed"
    SUSPECTED_REVIEW = "suspected_review"
    INSUFFICIENT_INFO = "insufficient_info"


class IssueStatus(StrEnum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    INTENTIONAL = "intentional"
    RESOLVED = "resolved"


class IssueType(StrEnum):
    """Codes follow the ConStory-Bench taxonomy (category.subtype)."""

    CHARACTER_AGE = "character.age"
    TIMELINE_DURATION = "timeline.duration"


class Evidence(BaseModel):
    chapter_number: int
    quote: str
    char_start: int
    char_end: int


class ConsistencyIssue(BaseModel):
    checker: str
    issue_type: IssueType
    confidence: Confidence
    description: str
    evidence: list[Evidence]
    fingerprint: str
    subjects: list[str] = []  # ids of the characters involved


def make_fingerprint(checker: str, *fact_ids: object) -> str:
    """Stable id for 'the same problem', so re-running keeps the author's status."""
    key = checker + "|" + "|".join(sorted(str(i) for i in fact_ids))
    return hashlib.sha256(key.encode()).hexdigest()[:32]
