from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KnowledgeDocumentIngestionQueued:
    document_id: str

    event_type: str = "KnowledgeDocumentIngestionQueued"
