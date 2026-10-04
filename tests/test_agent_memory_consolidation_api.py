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


class AgentMemoryConsolidationApiTest(unittest.TestCase):
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

    def _seed_consolidation_evidence(self) -> tuple[str, str]:
        from app.domains.automation.models import ToolCallLog, ToolCallStatus, WorkflowRun, WorkflowRunStatus
        from app.domains.conversations.models import AgentMessage, AgentMessageKind, AgentMessageRole, AgentSession

        with self.Session() as session:
            agent_session = AgentSession(id="session-api-consolidation", title="API 记忆沉淀")
            workflow_run = WorkflowRun(
                id="workflow-api-consolidation",
                workflow_type="job_discovery",
                status=WorkflowRunStatus.COMPLETED,
                current_step="final_response",
                user_goal="整理岗位线索并沉淀记忆",
            )
            user_message = AgentMessage(
                id="message-api-boundary",
                session_id=agent_session.id,
                role=AgentMessageRole.USER,
                message_kind=AgentMessageKind.USER_TEXT,
                content_text="投递前一定要让我确认，不要自动提交。",
                visible_content_text="投递前一定要让我确认，不要自动提交。",
            )
            failed_log = ToolCallLog(
                id="tool-log-api-failed",
                workflow_run_id=workflow_run.id,
                tool_name="GenericParser",
                tool_group="parser",
                status=ToolCallStatus.FAILED,
                input_payload={"url": "https://example.com/jobs"},
                error="PARSER_EMPTY",
            )
            recovered_log = ToolCallLog(
                id="tool-log-api-success",
                workflow_run_id=workflow_run.id,
                tool_name="GenericParser",
                tool_group="parser",
                status=ToolCallStatus.SUCCEEDED,
                input_payload={"url": "https://example.com/jobs"},
                output_payload={
                    "recovery_path": "use the list endpoint before parsing detail pages",
                    "extracted_count": 3,
                    "verified": True,
                },
            )
            session.add_all([agent_session, workflow_run, user_message, failed_log, recovered_log])
            session.commit()
            return agent_session.id, workflow_run.id

    def test_consolidation_endpoint_returns_review_and_promotion_counts(self) -> None:
        from app.domains.agent_memory.models import AgentLearningCandidate, AgentMemory

        session_id, workflow_run_id = self._seed_consolidation_evidence()
        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.post(
                    "/api/v1/agent-memory/consolidate",
                    json={
                        "session_id": session_id,
                        "workflow_run_id": workflow_run_id,
                        "agent_run_id": "agent-run-api-consolidation",
                        "target_scope": "job_discovery",
                    },
                )

        response = run(call_api())

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual(workflow_run_id, payload["workflow_run_id"])
        self.assertEqual(1, payload["reviewed_message_count"])
        self.assertEqual(2, payload["reviewed_tool_call_count"])
        self.assertEqual(2, payload["created_candidate_count"])
        self.assertEqual(1, payload["pending_candidate_count"])
        self.assertEqual(1, payload["promoted_memory_count"])
        self.assertEqual(0, payload["merged_memory_count"])
        self.assertEqual(2, len(payload["created_candidate_ids"]))
        self.assertEqual(1, len(payload["pending_candidate_ids"]))
        self.assertEqual(1, len(payload["promoted_memory_ids"]))
        self.assertEqual([], payload["merged_memory_ids"])
        self.assertEqual([], payload["skipped_reasons"])

        with self.Session() as session:
            memories = list(session.scalars(select(AgentMemory)).all())
            candidates = list(session.scalars(select(AgentLearningCandidate)).all())

        self.assertEqual(1, len(memories))
        self.assertEqual(2, len(candidates))

    def test_consolidation_endpoint_rejects_missing_workflow_run(self) -> None:
        session_id, _workflow_run_id = self._seed_consolidation_evidence()
        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.post(
                    "/api/v1/agent-memory/consolidate",
                    json={
                        "session_id": session_id,
                        "workflow_run_id": "missing-workflow-run",
                        "target_scope": "job_discovery",
                    },
                )

        response = run(call_api())

        self.assertEqual(404, response.status_code)
        self.assertEqual("Workflow run not found", response.json()["detail"])

    def test_consolidation_endpoint_requires_session_id(self) -> None:
        _session_id, workflow_run_id = self._seed_consolidation_evidence()
        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.post(
                    "/api/v1/agent-memory/consolidate",
                    json={
                        "workflow_run_id": workflow_run_id,
                        "target_scope": "job_discovery",
                    },
                )

        response = run(call_api())

        self.assertEqual(422, response.status_code)
        self.assertEqual("Field required", response.json()["detail"][0]["msg"])


if __name__ == "__main__":
    unittest.main()
