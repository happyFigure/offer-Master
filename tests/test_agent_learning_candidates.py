import sys
import unittest
from asyncio import run
from pathlib import Path

from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class AgentLearningCandidatesTest(unittest.TestCase):
    def setUp(self) -> None:
        from app.db.base import Base
        import app.domains.agent_memory.models  # noqa: F401
        import app.domains.automation.models  # noqa: F401
        import app.domains.conversations.models  # noqa: F401

        self.engine = create_engine(
            "sqlite+pysqlite://",
            connect_args={"check_same_thread": False},
            future=True,
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

    def tearDown(self) -> None:
        self.engine.dispose()

    def _app(self):
        from app.db.session import get_db_session
        from app.main import create_app

        app = create_app()

        def override_session():
            with self.Session() as session:
                yield session

        app.dependency_overrides[get_db_session] = override_session
        return app

    def _workflow_run(self, session):
        from app.domains.automation.models import WorkflowRun, WorkflowRunStatus

        workflow_run = WorkflowRun(
            workflow_type="url_import",
            status=WorkflowRunStatus.COMPLETED,
            current_step="final_response",
            user_goal="import a recruiting article",
        )
        session.add(workflow_run)
        session.flush()
        return workflow_run

    def test_recovered_tool_failure_creates_pending_learning_candidate(self) -> None:
        from app.agent_runtime.memory.learning_review import LearningReviewCommand, LearningReviewWorkflow
        from app.domains.agent_memory.models import AgentLearningCandidate, AgentLearningCandidateStatus
        from app.domains.agent_memory.repository import AgentMemoryRepository
        from app.domains.agent_memory.service import AgentLearningService
        from app.domains.automation.models import ToolCallLog, ToolCallStatus

        with self.Session() as session:
            workflow_run = self._workflow_run(session)
            failed_log = ToolCallLog(
                workflow_run_id=workflow_run.id,
                tool_name="WeChatArticleFetcher",
                tool_group="content_fetcher",
                status=ToolCallStatus.FAILED,
                input_payload={"domain": "mp.weixin.qq.com", "source_type": "wechat_article"},
                error="EXTRACTION_EMPTY: fetched summary page without full article body",
            )
            recovered_log = ToolCallLog(
                workflow_run_id=workflow_run.id,
                tool_name="WeChatArticleFetcher",
                tool_group="content_fetcher",
                status=ToolCallStatus.SUCCEEDED,
                input_payload={"domain": "mp.weixin.qq.com", "source_type": "wechat_article"},
                output_payload={
                    "recovery_path": "ask user to provide visible article body, then extract recruiting signals",
                    "extracted_count": 2,
                    "verified": True,
                },
            )
            session.add_all([failed_log, recovered_log])
            session.flush()

            service = AgentLearningService(AgentMemoryRepository(session))
            result = LearningReviewWorkflow(session=session, learning_service=service).review(
                LearningReviewCommand(
                    agent_run_id="agent-run-review-1",
                    session_id="session-review-1",
                    workflow_run_id=workflow_run.id,
                    target_scope="wechat_sync",
                    suggested_skill_target="wechat-recruiting-sync",
                )
            )
            session.commit()

            candidates = list(session.scalars(select(AgentLearningCandidate)).all())

        self.assertEqual(1, result.created_count)
        self.assertEqual(1, len(candidates))
        candidate = candidates[0]
        self.assertEqual(AgentLearningCandidateStatus.PENDING_REVIEW, candidate.status)
        self.assertIsNone(candidate.applied_at)
        self.assertEqual("agent-run-review-1", candidate.source_agent_run_id)
        self.assertEqual(workflow_run.id, candidate.source_workflow_run_id)
        self.assertEqual(recovered_log.id, candidate.source_tool_call_log_id)
        self.assertEqual("wechat_sync", candidate.target_scope)
        self.assertEqual("wechat-recruiting-sync", candidate.suggested_skill_target)
        self.assertIn(failed_log.id, candidate.evidence_json["tool_call_log_ids"])
        self.assertIn(recovered_log.id, candidate.evidence_json["tool_call_log_ids"])
        self.assertIn("recovery", candidate.candidate_body.lower())

    def test_transient_network_timeout_without_recovery_does_not_create_candidate(self) -> None:
        from app.agent_runtime.memory.learning_review import LearningReviewCommand, LearningReviewWorkflow
        from app.domains.agent_memory.models import AgentLearningCandidate
        from app.domains.agent_memory.repository import AgentMemoryRepository
        from app.domains.agent_memory.service import AgentLearningService
        from app.domains.automation.models import ToolCallLog, ToolCallStatus

        with self.Session() as session:
            workflow_run = self._workflow_run(session)
            session.add(
                ToolCallLog(
                    workflow_run_id=workflow_run.id,
                    tool_name="HttpFetcher",
                    tool_group="content_fetcher",
                    status=ToolCallStatus.FAILED,
                    input_payload={"domain": "example.com"},
                    error="httpx.ReadTimeout: transient timeout",
                )
            )
            session.flush()

            result = LearningReviewWorkflow(
                session=session,
                learning_service=AgentLearningService(AgentMemoryRepository(session)),
            ).review(
                LearningReviewCommand(
                    agent_run_id="agent-run-timeout",
                    session_id="session-timeout",
                    workflow_run_id=workflow_run.id,
                    target_scope="generic_fetch",
                    suggested_skill_target="generic-url-import",
                )
            )
            session.commit()

            candidates = list(session.scalars(select(AgentLearningCandidate)).all())

        self.assertEqual(0, result.created_count)
        self.assertEqual([], candidates)

    def test_learning_candidate_accepts_prefixed_agent_run_id(self) -> None:
        from app.domains.agent_memory.models import AgentLearningCandidate, AgentLearningCandidateRiskLevel
        from app.domains.agent_memory.schemas import AgentLearningCandidateCreate

        runtime_agent_run_id = "agent-run-00000000-0000-0000-0000-000000000000"

        draft = AgentLearningCandidateCreate(
            source_agent_run_id=runtime_agent_run_id,
            source_workflow_run_id="workflow-1",
            lesson_type="tool_recovery",
            target_scope="wechat_sync",
            suggested_skill_target="wechat-recruiting-sync",
            candidate_title="Recovered WeChat article extraction",
            candidate_body="Use the visible article body after the first fetch returns only a summary.",
            evidence_summary="Fetcher recovered after manual article body fallback.",
            risk_level=AgentLearningCandidateRiskLevel.MEDIUM,
        )

        self.assertEqual(runtime_agent_run_id, draft.source_agent_run_id)
        self.assertGreaterEqual(AgentLearningCandidate.__table__.c.source_agent_run_id.type.length, 64)

    def test_candidate_requires_evidence_source_risk_and_skill_target(self) -> None:
        from app.domains.agent_memory.models import AgentLearningCandidateRiskLevel
        from app.domains.agent_memory.repository import AgentMemoryRepository
        from app.domains.agent_memory.schemas import AgentLearningCandidateCreate
        from app.domains.agent_memory.service import AgentLearningService

        with self.Session() as session:
            service = AgentLearningService(AgentMemoryRepository(session))
            with self.assertRaises(ValueError):
                service.create_learning_candidate(
                    AgentLearningCandidateCreate(
                        source_workflow_run_id="workflow-1",
                        lesson_type="tool_recovery",
                        target_scope="wechat_sync",
                        suggested_skill_target="",
                        candidate_title="Recovered WeChat article extraction",
                        candidate_body="Use the visible article body after the first fetch returns only a summary.",
                        evidence_summary="",
                        risk_level=AgentLearningCandidateRiskLevel.MEDIUM,
                    )
                )

    def test_learning_candidate_api_lists_and_requires_target_skill_before_apply(self) -> None:
        from app.domains.agent_memory.models import AgentLearningCandidateRiskLevel
        from app.domains.agent_memory.repository import AgentMemoryRepository
        from app.domains.agent_memory.schemas import AgentLearningCandidateCreate
        from app.domains.agent_memory.service import AgentLearningService

        with self.Session() as session:
            candidate = AgentLearningService(AgentMemoryRepository(session)).create_learning_candidate(
                AgentLearningCandidateCreate(
                    source_agent_run_id="agent-run-api",
                    source_workflow_run_id="workflow-api",
                    source_tool_call_log_id="tool-log-api",
                    lesson_type="tool_recovery",
                    target_scope="dlmu_campus",
                    suggested_skill_target="dlmu-campus-sync",
                    candidate_title="DLMU list page parser recovery",
                    candidate_body="When campus list parsing succeeds after using the page list endpoint, save the endpoint pattern.",
                    evidence_summary="tool-log-api recovered after failed HTML parse",
                    success_evidence="2 candidate articles parsed",
                    risk_level=AgentLearningCandidateRiskLevel.LOW,
                    evidence_json={"tool_call_log_ids": ["tool-log-api"]},
                )
            )
            session.commit()

        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                list_response = await client.get("/api/v1/agent-learning/candidates")
                approve_response = await client.post(f"/api/v1/agent-learning/candidates/{candidate.id}/approve")
                apply_response = await client.post(f"/api/v1/agent-learning/candidates/{candidate.id}/apply")
                return list_response, approve_response, apply_response

        list_response, approve_response, apply_response = run(call_api())

        self.assertEqual(200, list_response.status_code)
        self.assertEqual(1, len(list_response.json()["items"]))
        self.assertEqual(200, approve_response.status_code)
        self.assertEqual("approved", approve_response.json()["status"])
        self.assertEqual(409, apply_response.status_code)
        self.assertIn("target_skill_id", apply_response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
