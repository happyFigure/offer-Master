import sys
import unittest
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class DurableStateArtifactIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        import app.agent_runtime.durable_state.models  # noqa: F401
        import app.agent_runtime.external_tasks.models  # noqa: F401
        from app.db.base import Base

        self.engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_service_records_artifacts_from_result_envelope(self) -> None:
        from app.agent_runtime.durable_state.repository import SqlAlchemyDurableStateRepository
        from app.agent_runtime.durable_state.service import DurableStateService

        with self.Session() as session:
            service = DurableStateService(SqlAlchemyDurableStateRepository(session))
            service.create_task(
                task_id="task-artifact-1",
                root_workflow_run_id="workflow-artifact-1",
                conversation_session_id="session-artifact-1",
                task_type="campus_search",
                capability="external.web_search",
            )
            service.add_step(
                task_id="task-artifact-1",
                step_id="step-artifact-1",
                sequence_index=1,
                step_type="external_agent",
                executor_type="external_agent",
                executor_name="claude_sdk_agent",
                capability="external.web_search",
            )

            records = service.record_artifacts_from_result_envelope(
                task_id="task-artifact-1",
                step_id="step-artifact-1",
                result_envelope={
                    "capability": "external.web_search",
                    "executor": "claude_sdk_agent",
                    "artifacts": [
                        {"type": "url", "title": "腾讯校招", "url": "https://join.qq.com/"},
                        {"type": "screenshot", "title": "申请页截图", "url": "F:/tmp/apply.png", "mime_type": "image/png"},
                    ],
                },
            )
            session.commit()

        with self.Session() as session:
            service = DurableStateService(SqlAlchemyDurableStateRepository(session))
            artifacts = service.list_artifacts("task-artifact-1")

        self.assertEqual(2, len(records))
        self.assertEqual(["url", "screenshot"], [item.artifact_type for item in artifacts])
        self.assertEqual("https://join.qq.com/", artifacts[0].uri)
        self.assertEqual("image/png", artifacts[1].mime_type)
        self.assertEqual("claude_sdk_agent", artifacts[0].artifact_metadata["executor"])

    def test_service_syncs_external_agent_artifacts_into_unified_artifact_index(self) -> None:
        from app.agent_runtime.durable_state.repository import SqlAlchemyDurableStateRepository
        from app.agent_runtime.durable_state.schemas import AgentArtifactSourceKind
        from app.agent_runtime.durable_state.service import DurableStateService
        from app.agent_runtime.external_tasks.models import ExternalAgentArtifact, ExternalAgentTask
        from app.agent_runtime.external_tasks.schemas import ExternalAgentTaskStatus, ExternalTaskType

        with self.Session() as session:
            service = DurableStateService(SqlAlchemyDurableStateRepository(session))
            service.create_task(
                task_id="task-artifact-sync",
                root_workflow_run_id="workflow-artifact-sync",
                conversation_session_id="session-artifact-sync",
                task_type="application_orchestration",
                capability="applications.find_apply_entry",
            )
            service.add_step(
                task_id="task-artifact-sync",
                step_id="step-artifact-sync",
                sequence_index=1,
                step_type="browser_executor",
                executor_type="external_agent",
                executor_name="codex_or_multica",
                capability="applications.find_apply_entry",
            )
            session.add(
                ExternalAgentTask(
                    id="external-task-artifact-sync",
                    task_type=ExternalTaskType.FIND_APPLY_ENTRY.value,
                    status=ExternalAgentTaskStatus.SUCCEEDED.value,
                    context_pack_hash="hash-artifact-sync",
                    input_payload={"trace_id": "trace-artifact-sync"},
                    output_payload={"status": "found_opened"},
                )
            )
            session.flush()
            session.add_all(
                [
                    ExternalAgentArtifact(
                        task_id="external-task-artifact-sync",
                        artifact_type="screenshot",
                        path_or_uri="F:/tmp/apply-entry.png",
                        mime_type="image/png",
                        artifact_metadata={"title": "申请页截图", "step": "find_apply_button"},
                    ),
                    ExternalAgentArtifact(
                        task_id="external-task-artifact-sync",
                        artifact_type="url",
                        path_or_uri="https://join.qq.com/apply/1",
                        artifact_metadata={"title": "申请入口"},
                    ),
                ]
            )
            session.flush()

            synced = service.sync_external_agent_artifacts(
                task_id="task-artifact-sync",
                step_id="step-artifact-sync",
                external_task_id="external-task-artifact-sync",
            )
            session.commit()

        with self.Session() as session:
            service = DurableStateService(SqlAlchemyDurableStateRepository(session))
            artifacts = service.list_artifacts("task-artifact-sync")

        self.assertEqual(2, len(synced))
        self.assertEqual(2, len(artifacts))
        self.assertEqual([AgentArtifactSourceKind.EXTERNAL_AGENT, AgentArtifactSourceKind.EXTERNAL_AGENT], [item.source_kind for item in artifacts])
        artifacts_by_uri = {artifact.uri: artifact for artifact in artifacts}
        self.assertIn("F:/tmp/apply-entry.png", artifacts_by_uri)
        self.assertIn("https://join.qq.com/apply/1", artifacts_by_uri)
        screenshot = artifacts_by_uri["F:/tmp/apply-entry.png"]
        self.assertEqual("image/png", screenshot.mime_type)
        self.assertEqual("external-task-artifact-sync", screenshot.artifact_metadata["external_task_id"])
        self.assertEqual("find_apply_button", screenshot.artifact_metadata["raw_metadata"]["step"])


if __name__ == "__main__":
    unittest.main()
