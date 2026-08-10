from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class KnowledgeDocumentDraft(BaseModel):
    title: str
    source_path: str
    document_type: str = "generic"
    metadata: dict[str, Any] = Field(default_factory=dict)
