from __future__ import annotations

from typing import Sequence

from app.agent_runtime.memory.memory_candidate_extractor import MemoryCandidateDraft
from app.agent_runtime.memory.memory_scorer import ScoredMemoryCandidate, normalized_memory_key, score_memory_candidate
from app.domains.agent_memory.models import AgentMemory, AgentMemoryStatus


def deduplicate_memory_candidates(
    candidates: Sequence[MemoryCandidateDraft],
    *,
    existing_memories: Sequence[AgentMemory],
) -> list[ScoredMemoryCandidate]:
    existing_by_key = {
        _memory_key(memory): memory
        for memory in existing_memories
        if _memory_status(memory) == AgentMemoryStatus.ACTIVE.value
    }
    grouped: dict[str, MemoryCandidateDraft] = {}
    existing_targets: dict[str, AgentMemory | None] = {}

    for candidate in candidates:
        key = normalized_memory_key(
            memory_type=candidate.memory_type,
            scope=candidate.scope,
            title=candidate.title,
        )
        current = grouped.get(key)
        if current is None:
            grouped[key] = candidate
        else:
            grouped[key] = _merge_drafts(current, candidate)
        existing_targets[key] = existing_by_key.get(key)

    result: list[ScoredMemoryCandidate] = []
    for key, draft in grouped.items():
        scored = score_memory_candidate(draft)
        existing = existing_targets.get(key)
        if existing is not None:
            scored = ScoredMemoryCandidate(
                draft=_merge_existing_evidence(draft, existing),
                score=scored.score,
                normalized_key=scored.normalized_key,
                auto_promotable=scored.auto_promotable,
                existing_memory_id=existing.id,
            )
        result.append(scored)
    return result


def _merge_drafts(first: MemoryCandidateDraft, second: MemoryCandidateDraft) -> MemoryCandidateDraft:
    evidence_ids = _merge_ids(first.evidence_ids, second.evidence_ids)
    metadata = {
        **first.metadata,
        **second.metadata,
        "evidence_ids": list(evidence_ids),
        "merged_candidate_count": int(first.metadata.get("merged_candidate_count") or 1)
        + int(second.metadata.get("merged_candidate_count") or 1),
    }
    return MemoryCandidateDraft(
        memory_type=first.memory_type,
        scope=first.scope,
        title=first.title,
        content=first.content if len(first.content) >= len(second.content) else second.content,
        importance=max(first.importance, second.importance),
        risk_level=_higher_risk(first.risk_level, second.risk_level),
        lesson_type=first.lesson_type,
        evidence_ids=evidence_ids,
        metadata=metadata,
    )


def _merge_existing_evidence(draft: MemoryCandidateDraft, existing: AgentMemory) -> MemoryCandidateDraft:
    metadata = dict(draft.metadata)
    metadata["existing_memory_id"] = existing.id
    evidence_ids = _merge_ids(
        draft.evidence_ids,
        tuple(str(value) for value in (existing.metadata_json or {}).get("evidence_ids", []) if str(value).strip()),
    )
    metadata["evidence_ids"] = list(evidence_ids)
    return MemoryCandidateDraft(
        memory_type=draft.memory_type,
        scope=draft.scope,
        title=draft.title,
        content=draft.content,
        importance=max(draft.importance, existing.importance),
        risk_level=draft.risk_level,
        lesson_type=draft.lesson_type,
        evidence_ids=evidence_ids,
        metadata=metadata,
    )


def _memory_key(memory: AgentMemory) -> str:
    metadata_key = (memory.metadata_json or {}).get("normalized_key")
    if metadata_key:
        return str(metadata_key)
    return normalized_memory_key(
        memory_type=memory.memory_type,
        scope=memory.scope,
        title=memory.title,
    )


def _merge_ids(*groups: Sequence[str]) -> tuple[str, ...]:
    merged: list[str] = []
    for group in groups:
        for value in group:
            text = str(value).strip()
            if text and text not in merged:
                merged.append(text)
    return tuple(merged)


def _higher_risk(first, second):
    ranks = {"low": 0, "medium": 1, "high": 2}
    return first if ranks[_risk_value(first)] >= ranks[_risk_value(second)] else second


def _risk_value(value) -> str:
    return str(getattr(value, "value", value))


def _memory_status(memory: AgentMemory) -> str:
    return str(getattr(memory.status, "value", memory.status))
