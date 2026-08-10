from __future__ import annotations


class KnowledgeRepository:
    deferred_until = "rag_phase"

    def get(self, document_id: str):
        raise NotImplementedError(
            f"Knowledge persistence is deferred until the RAG phase: {document_id}"
        )
