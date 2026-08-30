import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class MemoryConsolidationRulesTest(unittest.TestCase):
    def test_same_preference_is_deduplicated_and_evidence_is_merged(self):
        from app.agent_runtime.memory.memory_candidate_extractor import MemoryCandidateDraft
        from app.agent_runtime.memory.memory_deduplicator import deduplicate_memory_candidates
        from app.domains.agent_memory.models import (
            AgentLearningCandidateLessonType,
            AgentLearningCandidateRiskLevel,
        )

        first = self._draft(
            MemoryCandidateDraft,
            evidence_ids=("message-1",),
            content="投递前必须用户确认",
        )
        second = self._draft(
            MemoryCandidateDraft,
            evidence_ids=("message-2",),
            content="投递前必须用户确认",
        )

        result = deduplicate_memory_candidates([first, second], existing_memories=[])

        self.assertEqual(1, len(result))
        self.assertEqual(("message-1", "message-2"), result[0].draft.evidence_ids)
        self.assertEqual(
            "user_preference:application_submission:投递前必须用户确认",
            result[0].normalized_key,
        )
        self.assertEqual(AgentLearningCandidateLessonType.USER_PREFERENCE, result[0].draft.lesson_type)
        self.assertEqual(AgentLearningCandidateRiskLevel.HIGH, result[0].draft.risk_level)

    def test_high_risk_candidate_is_never_auto_promotable(self):
        from app.agent_runtime.memory.memory_candidate_extractor import MemoryCandidateDraft
        from app.agent_runtime.memory.memory_scorer import score_memory_candidate
        from app.domains.agent_memory.models import AgentLearningCandidateRiskLevel

        draft = self._draft(MemoryCandidateDraft, evidence_ids=("message-1",))

        scored = score_memory_candidate(draft)

        self.assertGreaterEqual(scored.score, 80)
        self.assertFalse(scored.auto_promotable)
        self.assertEqual(AgentLearningCandidateRiskLevel.HIGH, scored.draft.risk_level)

    def test_existing_active_memory_is_returned_as_merge_target(self):
        from app.agent_runtime.memory.memory_candidate_extractor import MemoryCandidateDraft
        from app.agent_runtime.memory.memory_deduplicator import deduplicate_memory_candidates
        from app.domains.agent_memory.models import AgentMemory, AgentMemoryStatus

        draft = self._draft(MemoryCandidateDraft, evidence_ids=("message-new",))
        existing = AgentMemory(
            id="memory-1",
            memory_type="user_preference",
            scope="application_submission",
            title="投递前必须用户确认",
            content="投递前必须用户确认",
            status=AgentMemoryStatus.ACTIVE,
            importance=95,
            metadata_json={
                "normalized_key": "user_preference:application_submission:投递前必须用户确认",
                "evidence_ids": ["message-old"],
            },
        )

        result = deduplicate_memory_candidates([draft], existing_memories=[existing])

        self.assertEqual(1, len(result))
        self.assertEqual("memory-1", result[0].existing_memory_id)
        self.assertEqual(("message-new", "message-old"), result[0].draft.evidence_ids)

    @staticmethod
    def _draft(draft_type, *, evidence_ids, content="投递前必须用户确认"):
        from app.domains.agent_memory.models import (
            AgentLearningCandidateLessonType,
            AgentLearningCandidateRiskLevel,
        )

        return draft_type(
            memory_type="user_preference",
            scope="application_submission",
            title="投递前必须用户确认",
            content=content,
            importance=95,
            risk_level=AgentLearningCandidateRiskLevel.HIGH,
            lesson_type=AgentLearningCandidateLessonType.USER_PREFERENCE,
            evidence_ids=evidence_ids,
            metadata={"source_kind": "explicit_user_boundary"},
        )


if __name__ == "__main__":
    unittest.main()
