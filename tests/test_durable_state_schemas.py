import sys
import unittest
from pathlib import Path

from pydantic import ValidationError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class DurableStateSchemasTest(unittest.TestCase):
    def test_task_state_create_expresses_main_orchestration_task(self) -> None:
        from app.agent_runtime.durable_state.schemas import AgentTaskStateCreate, AgentTaskStatus

        command = AgentTaskStateCreate(
            task_id="task-1",
            root_workflow_run_id="workflow-1",
            conversation_session_id="session-1",
            task_type="batch_application",
            capability="applications.batch_plan",
            user_goal="为我准备腾讯和京东的校招投递",
            input_payload={"company_names": ["腾讯", "京东"]},
        )

        self.assertEqual(AgentTaskStatus.CREATED, command.status)
        self.assertEqual("applications.batch_plan", command.capability)
        self.assertEqual({"company_names": ["腾讯", "京东"]}, command.input_payload)

    def test_step_state_create_expresses_executor_assignment(self) -> None:
        from app.agent_runtime.durable_state.schemas import AgentStepStateCreate, AgentStepStatus

        command = AgentStepStateCreate(
            step_id="step-1",
            task_id="task-1",
            sequence_index=1,
            step_type="external_agent",
            executor_type="external_agent",
            executor_name="claude_sdk_agent",
            capability="external.web_search",
            input_payload={"query": "腾讯 校园招聘 官网"},
        )

        self.assertEqual(AgentStepStatus.PENDING, command.status)
        self.assertEqual("claude_sdk_agent", command.executor_name)
        self.assertEqual("external.web_search", command.capability)

    def test_step_state_create_rejects_negative_sequence_index(self) -> None:
        from app.agent_runtime.durable_state.schemas import AgentStepStateCreate

        with self.assertRaises(ValidationError):
            AgentStepStateCreate(
                step_id="step-1",
                task_id="task-1",
                sequence_index=-1,
                step_type="tool_call",
                executor_type="local_tool",
                executor_name="tool_registry",
                capability="memory_search",
            )


if __name__ == "__main__":
    unittest.main()
