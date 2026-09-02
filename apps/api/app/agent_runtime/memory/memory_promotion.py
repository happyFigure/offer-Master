from __future__ import annotations

from dataclasses import dataclass

from app.agent_runtime.memory.memory_scorer import ScoredMemoryCandidate
from app.domains.agent_memory.models import AgentMemory, AgentMemoryStatus, utc_now
from app.domains.agent_memory.repository import AgentMemoryRepository


@dataclass(frozen=True)
class MemoryPromotionResult:
    memory: AgentMemory
    created: bool


class MemoryPromotionService:
    def __init__(self, repository: AgentMemoryRepository) -> None:
        self._repository = repository

    def promote(self, candidate: ScoredMemoryCandidate, *, source_candidate_id: str) -> MemoryPromotionResult:
        existing = self._repository.find_active_memory_by_key(candidate.normalized_key)
        evidence_ids = _merge_ids(
            candidate.draft.evidence_ids,
            tuple(
                str(value)
                for value in (existing.metadata_json or {}).get("evidence_ids", [])
                if str(value).strip()
            )
            if existing is not None
            else (),
        )
        metadata = {
            **candidate.draft.metadata,
            "normalized_key": candidate.normalized_key,
            "evidence_ids": list(evidence_ids),
            "score": candidate.score,
            "promotion_mode": "automatic_low_risk",
            "source_candidate_id": source_candidate_id,
            "last_observed_at": utc_now().isoformat(),
        }

        if existing is not None:
            existing_metadata = dict(existing.metadata_json or {})
            source_candidate_ids = _merge_ids(
                tuple(str(value) for value in existing_metadata.get("source_candidate_ids", []) if str(value).strip()),
                (source_candidate_id,),
            )
            existing.metadata_json = {
                **existing_metadata,
                **metadata,
                "evidence_ids": list(evidence_ids),
                "source_candidate_ids": list(source_candidate_ids),
                "score": max(int(existing_metadata.get("score") or 0), candidate.score),
            }
            if candidate.score >= int(existing_metadata.get("score") or 0):
                existing.title = candidate.draft.title
                existing.content = candidate.draft.content
                existing.importance = max(existing.importance, candidate.score)
            existing.updated_at = utc_now()
            self._repository.update_memory(existing)
            return MemoryPromotionResult(memory=existing, created=False)

        memory = AgentMemory(
            memory_type=candidate.draft.memory_type,
            scope=candidate.draft.scope,
            title=candidate.draft.title,
            content=candidate.draft.content,
            source_type="memory_consolidation",
            source_id=source_candidate_id,
            status=AgentMemoryStatus.ACTIVE,
            importance=candidate.score,
            metadata_json=metadata,
        )
        return MemoryPromotionResult(memory=self._repository.add_memory(memory), created=True)


def _merge_ids(*groups: tuple[str, ...]) -> tuple[str, ...]:
    merged: list[str] = []
    for group in groups:
        for value in group:
            text = str(value).strip()
            if text and text not in merged:
                merged.append(text)
    return tuple(merged)
