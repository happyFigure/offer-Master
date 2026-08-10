from __future__ import annotations


class KnowledgeService:
    deferred_until = "rag_phase"

    def queue_ingestion(self, *_args, **_kwargs):
        raise NotImplementedError("Knowledge ingestion is deferred until the RAG phase")
