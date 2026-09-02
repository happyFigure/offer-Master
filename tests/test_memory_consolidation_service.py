import sys
import unittest
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class MemoryConsolidationServiceTest(unittest.TestCase):
    def setUp(self):
        from app.db.base import Base
        import app.domains.agent_memory.models  # noqa: F401
        import app.domains.automation.models  # noqa: F401
        import app.domains.conversations.models  # noqa: F401

        self.engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

    def tearDown(self):
        self.engine.dispose()

    def test_low_risk_recovery_is_promoted_and_high_risk_preference_stays_pending(self):
        from app.agent_runtime.memory.consolidation import (
            MemoryConsolidationCommand,
            MemoryConsolidationService,
        )
        from app.domains.agent_memory.models import (
            AgentLearningCandidate,
            AgentLearningCandidateStatus,
            AgentMemory,
            AgentMemoryStatus,
        )
        from app.domains.agent_memory.repository import AgentMemoryRepository
        from app.domains.agent_memory.service import AgentLearningService
        from app.domains.automation.models import ToolCallLog, ToolCallStatus, WorkflowRun, WorkflowRunStatus
        from app.domains.conversations.models import AgentMessage, AgentMessageKind, AgentMessageRole, AgentSession

        with self.Session() as session:
            agent_session = AgentSession(id="session-1", title="记忆沉淀")
            workflow_run = WorkflowRun(
                id="workflow-1",
                workflow_type="job_discovery",
                status=WorkflowRunStatus.COMPLETED,
                current_step="final_response",
                user_goal="整理岗位线索",
            )
            user_message = AgentMessage(
                id="message-boundary",
                session_id=agent_session.id,
                role=AgentMessageRole.USER,
                message_kind=AgentMessageKind.USER_TEXT,
                content_text="投递前一定要让我确认，不要自动提交。",
                visible_content_text="投递前一定要让我确认，不要自动提交。",
            )
            failed_log = ToolCallLog(
                id="tool-log-failed",
                workflow_run_id=workflow_run.id,
                tool_name="GenericParser",
                tool_group="parser",
                status=ToolCallStatus.FAILED,
                input_payload={"url": "https://example.com"},
                error="PARSER_EMPTY",
            )
            recovered_log = ToolCallLog(
                id="tool-log-success",
                workflow_run_id=workflow_run.id,
                tool_name="GenericParser",
                tool_group="parser",
                status=ToolCallStatus.SUCCEEDED,
                input_payload={"url": "https://example.com"},
                output_payload={
                    "recovery_path": "use the list endpoint before parsing detail pages",
                    "extracted_count": 4,
                },
            )
            session.add_all([agent_session, workflow_run, user_message, failed_log, recovered_log])
            session.flush()

            service = MemoryConsolidationService(
                session=session,
                learning_service=AgentLearningService(AgentMemoryRepository(session)),
            )
            result = service.consolidate(
                MemoryConsolidationCommand(
                    session_id=agent_session.id,
                    workflow_run_id=workflow_run.id,
                    agent_run_id="agent-run-1",
                    target_scope="job_discovery",
                )
            )
            session.commit()

            memories = list(session.scalars(select(AgentMemory)).all())
            candidates = list(session.scalars(select(AgentLearningCandidate)).all())

        self.assertEqual(2, result.created_candidate_count)
        self.assertEqual(1, result.promoted_memory_count)
        self.assertEqual(1, len(memories))
        self.assertEqual(AgentMemoryStatus.ACTIVE, memories[0].status)
        self.assertEqual("tool_recovery", memories[0].memory_type)
        self.assertIn("tool-log-success", memories[0].metadata_json["evidence_ids"])
        self.assertEqual(2, len(result.created_candidate_ids))
        self.assertEqual(1, len(result.pending_candidate_ids))
        self.assertCountEqual(
            [AgentLearningCandidateStatus.APPLIED, AgentLearningCandidateStatus.PENDING_REVIEW],
            [candidate.status for candidate in candidates],
        )

    def test_repeated_consolidation_merges_into_existing_memory_without_duplicate(self):
        from app.agent_runtime.memory.consolidation import (
            MemoryConsolidationCommand,
            MemoryConsolidationService,
        )
        from app.domains.agent_memory.models import AgentMemory, AgentMemoryStatus
        from app.domains.agent_memory.repository import AgentMemoryRepository
        from app.domains.agent_memory.service import AgentLearningService
        from app.domains.automation.models import ToolCallLog, ToolCallStatus, WorkflowRun, WorkflowRunStatus

        with self.Session() as session:
            workflow_run = WorkflowRun(
                id="workflow-2",
                workflow_type="job_discovery",
                status=WorkflowRunStatus.COMPLETED,
                current_step="final_response",
                user_goal="复用解析经验",
            )
            failed_log = ToolCallLog(
                id="tool-log-failed-2",
                workflow_run_id=workflow_run.id,
                tool_name="GenericParser",
                tool_group="parser",
                status=ToolCallStatus.FAILED,
                input_payload={"url": "https://example.com"},
                error="PARSER_EMPTY",
            )
            recovered_log = ToolCallLog(
                id="tool-log-success-2",
                workflow_run_id=workflow_run.id,
                tool_name="GenericParser",
                tool_group="parser",
                status=ToolCallStatus.SUCCEEDED,
                input_payload={"url": "https://example.com"},
                output_payload={
                    "recovery_path": "use the list endpoint before parsing detail pages",
                    "extracted_count": 4,
                },
            )
            existing = AgentMemory(
                id="memory-existing",
                memory_type="tool_recovery",
                scope="parser",
                title="GenericParser 恢复经验",
                content="旧的解析恢复经验",
                source_type="memory_consolidation",
                status=AgentMemoryStatus.ACTIVE,
                importance=80,
                metadata_json={
                    "normalized_key": "tool_recovery:parser:genericparser 恢复经验",
                    "evidence_ids": ["tool-log-old"],
                    "score": 80,
                },
            )
            session.add_all([workflow_run, failed_log, recovered_log, existing])
            session.flush()

            result = MemoryConsolidationService(
                session=session,
                learning_service=AgentLearningService(AgentMemoryRepository(session)),
            ).consolidate(
                MemoryConsolidationCommand(
                    session_id=None,
                    workflow_run_id=workflow_run.id,
                    agent_run_id="agent-run-2",
                    target_scope="job_discovery",
                )
            )
            session.commit()

            memories = list(session.scalars(select(AgentMemory)).all())

        self.assertEqual(1, len(memories))
        self.assertEqual("memory-existing", result.merged_memory_ids[0])
        self.assertIn("tool-log-old", memories[0].metadata_json["evidence_ids"])
        self.assertIn("tool-log-success-2", memories[0].metadata_json["evidence_ids"])

    def test_consolidation_can_scope_messages_to_compaction_cut(self):
        from app.agent_runtime.memory.consolidation import (
            MemoryConsolidationCommand,
            MemoryConsolidationService,
        )
        from app.domains.agent_memory.models import AgentLearningCandidate, AgentLearningCandidateStatus
        from app.domains.agent_memory.repository import AgentMemoryRepository
        from app.domains.agent_memory.service import AgentLearningService
        from app.domains.automation.models import WorkflowRun, WorkflowRunStatus
        from app.domains.conversations.models import AgentMessage, AgentMessageKind, AgentMessageRole, AgentSession

        with self.Session() as session:
            agent_session = AgentSession(id="session-cut", title="只刷新被压缩旧消息")
            workflow_run = WorkflowRun(
                id="workflow-cut",
                workflow_type="agent_memory",
                status=WorkflowRunStatus.RUNNING,
                current_step="build_context",
                user_goal="压缩前先刷新记忆",
            )
            old_message = AgentMessage(
                id="message-old-boundary",
                session_id=agent_session.id,
                role=AgentMessageRole.USER,
                message_kind=AgentMessageKind.USER_TEXT,
                content_text="投递前一定要让我确认，不能自动提交。",
                visible_content_text="投递前一定要让我确认，不能自动提交。",
            )
            recent_message = AgentMessage(
                id="message-recent-integrity",
                session_id=agent_session.id,
                role=AgentMessageRole.USER,
                message_kind=AgentMessageKind.USER_TEXT,
                content_text="如果企业性质无法判断，就留空，不要编造。",
                visible_content_text="如果企业性质无法判断，就留空，不要编造。",
            )
            session.add_all([agent_session, workflow_run, old_message, recent_message])
            session.flush()

            result = MemoryConsolidationService(
                session=session,
                learning_service=AgentLearningService(AgentMemoryRepository(session)),
            ).consolidate(
                MemoryConsolidationCommand(
                    session_id=agent_session.id,
                    workflow_run_id=workflow_run.id,
                    agent_run_id="agent-run-cut",
                    target_scope="job_discovery",
                    message_ids=[old_message.id],
                )
            )
            session.commit()

            candidates = list(session.scalars(select(AgentLearningCandidate)).all())

        self.assertEqual(1, result.reviewed_message_count)
        self.assertEqual(1, result.created_candidate_count)
        self.assertEqual(AgentLearningCandidateStatus.PENDING_REVIEW, candidates[0].status)
        self.assertEqual("投递前必须用户确认", candidates[0].candidate_title)
        self.assertEqual(old_message.id, candidates[0].source_message_id)


if __name__ == "__main__":
    unittest.main()
