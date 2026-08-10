from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class InterviewPracticeDraft(BaseModel):
    application_id: str | None = None
    question: str
    context: dict[str, Any] = Field(default_factory=dict)
