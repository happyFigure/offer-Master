from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InterviewPracticeQueued:
    application_id: str | None = None

    event_type: str = "InterviewPracticeQueued"
