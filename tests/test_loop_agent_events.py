import json
import sys
import unittest
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class LoopAgentEventsTest(unittest.TestCase):
    def test_event_metadata_contains_minimum_runtime_fields(self) -> None:
        from app.agent_runtime.loop_agent.events import LoopAgentEvent, LoopAgentEventType

        event = LoopAgentEvent(
            event_type=LoopAgentEventType.TOOL_STARTED,
            session_id="session-1",
            task_id="task-1",
            run_id="run-1",
            turn_index=2,
            step_index=1,
            capability="external.web_search",
            tool_call_id="tool-call-1",
            status="running",
            summary="正在调用网页搜索",
            metadata={"query": "腾讯 校园招聘 官网 2026"},
        )

        payload = event.to_metadata_dict()

        self.assertEqual("tool_started", payload["event_type"])
        self.assertEqual("工具开始执行", payload["event_label"])
        self.assertEqual("session-1", payload["session_id"])
        self.assertEqual("task-1", payload["task_id"])
        self.assertEqual("run-1", payload["run_id"])
        self.assertEqual(2, payload["turn_index"])
        self.assertEqual(1, payload["step_index"])
        self.assertEqual("external.web_search", payload["capability"])
        self.assertEqual("tool-call-1", payload["tool_call_id"])
        self.assertEqual("running", payload["status"])
        self.assertEqual("正在调用网页搜索", payload["summary"])
        self.assertEqual({"query": "腾讯 校园招聘 官网 2026"}, payload["metadata"])
        self.assertIn("created_at", payload)

    def test_event_payload_is_json_serializable(self) -> None:
        from app.agent_runtime.loop_agent.events import LoopAgentEvent, LoopAgentEventType

        event = LoopAgentEvent(
            event_type=LoopAgentEventType.TURN_FINISHED,
            session_id="session-1",
            task_id="task-1",
            turn_index=1,
            status="succeeded",
            summary="第一轮结束",
            metadata={
                "recorded_at": datetime(2026, 8, 23, 9, 30, tzinfo=UTC),
                "visited_steps": {"search", "observe"},
                "event_type": LoopAgentEventType.TURN_FINISHED,
            },
        )

        payload = event.to_metadata_dict()
        json.dumps(payload, ensure_ascii=False)

        self.assertEqual("2026-08-23T09:30:00+00:00", payload["metadata"]["recorded_at"])
        self.assertEqual(["observe", "search"], payload["metadata"]["visited_steps"])
        self.assertEqual("turn_finished", payload["metadata"]["event_type"])

    def test_event_payload_serializes_mixed_sets_without_sort_type_errors(self) -> None:
        from app.agent_runtime.loop_agent.events import LoopAgentEvent, LoopAgentEventType

        event = LoopAgentEvent(
            event_type=LoopAgentEventType.TOOL_FINISHED,
            metadata={"mixed_values": {1, "search"}},
        )

        payload = event.to_metadata_dict()
        json.dumps(payload, ensure_ascii=False)

        self.assertEqual([1, "search"], payload["metadata"]["mixed_values"])

    def test_event_type_labels_cover_agent_loop_lifecycle(self) -> None:
        from app.agent_runtime.loop_agent.events import LoopAgentEventType

        labels = {event_type.value: event_type.label for event_type in LoopAgentEventType}

        self.assertEqual(
            {
                "task_started": "任务开始",
                "turn_started": "一轮开始",
                "model_decision": "模型决定调用能力",
                "tool_started": "工具开始执行",
                "tool_finished": "工具执行结束",
                "turn_finished": "一轮结束",
                "waiting_user": "等待用户",
                "task_finished": "任务结束",
            },
            labels,
        )

    def test_loop_agent_package_exports_event_types(self) -> None:
        from app.agent_runtime.loop_agent import LoopAgentEvent, LoopAgentEventType

        event = LoopAgentEvent(
            event_type=LoopAgentEventType.TASK_STARTED,
            session_id="session-1",
            task_id="task-1",
            summary="任务开始",
        )

        self.assertEqual("task_started", event.to_metadata_dict()["event_type"])


if __name__ == "__main__":
    unittest.main()
