from __future__ import annotations

from dataclasses import dataclass

from app.agent_runtime.memory.memory_candidate_extractor import MemoryCandidateDraft
from app.domains.agent_memory.models import AgentLearningCandidateRiskLevel


@dataclass(frozen=True)
class ScoredMemoryCandidate:
    draft: MemoryCandidateDraft
    score: int
    normalized_key: str
    auto_promotable: bool
    existing_memory_id: str | None = None


def score_memory_candidate(draft: MemoryCandidateDraft) -> ScoredMemoryCandidate:
    score = max(0, min(100, draft.importance))
    source_kind = str(draft.metadata.get("source_kind") or "")
    if source_kind == "explicit_user_boundary":
        score += 5
    if source_kind == "recovered_tool_call" and len(draft.evidence_ids) >= 2:
        score += 10
    if draft.risk_level == AgentLearningCandidateRiskLevel.HIGH:
        score -= 15
    score = max(0, min(100, score))
    return ScoredMemoryCandidate(
        draft=draft,
        score=score,
        normalized_key=normalized_memory_key(
            memory_type=draft.memory_type,
            scope=draft.scope,
            title=draft.title,
        ),
        auto_promotable=(
            score >= 80
            and draft.risk_level == AgentLearningCandidateRiskLevel.LOW
            and bool(draft.evidence_ids)
        ),
    )


def normalized_memory_key(*, memory_type: str, scope: str, title: str) -> str:
    return ":".join(
        [
            _normalize_text(memory_type),
            _normalize_text(scope),
            _normalize_text(title).rstrip("。.!！?？"),
        ]
    )


def _normalize_text(value: str) -> str:
    return " ".join(str(value).split()).casefold()
