import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class LoopAgentControllerTest(unittest.TestCase):
    def test_controller_stops_when_runtime_step_budget_is_exhausted(self) -> None:
        from app.agent_runtime.loop_agent.controller import LoopAgentController
        from app.agent_runtime.loop_agent.schemas import LoopAgentAction, LoopAgentDecision, LoopAgentObservation, LoopAgentStopReason

        decisions = [
            LoopAgentDecision(
                action=LoopAgentAction.CALL_TOOL,
                capability="external.web_search",
                tool_input={"query": "腾讯 校园招聘 官网"},
                reason="Need official campus recruiting source.",
            ),
            LoopAgentDecision(
                action=LoopAgentAction.CALL_TOOL,
                capability="external.web_search",
                tool_input={"query": "京东 校园招聘 官网"},
                reason="Need second company source.",
            ),
        ]
        executed = []

        def decide(trace):
            return decisions[len(trace)]

        def execute(decision):
            executed.append(decision.capability)
            return LoopAgentObservation(status="succeeded", summary="找到一个官方入口", tool_call_id="tool-call-1")

        result = LoopAgentController(max_steps=1).run(decide_next_step=decide, execute_step=execute)

        self.assertEqual(LoopAgentStopReason.BUDGET_EXHAUSTED, result.stop_reason)
        self.assertEqual(["external.web_search"], executed)
        self.assertEqual(1, result.executed_step_count)
        self.assertEqual("external.web_search", result.trace[0].capability)
        self.assertEqual("找到一个官方入口", result.trace[0].observation_summary)
        self.assertNotIn("thought", result.to_metadata_dict()["trace"][0])

    def test_controller_stops_when_executor_requires_user_action(self) -> None:
        from app.agent_runtime.loop_agent.controller import LoopAgentController
        from app.agent_runtime.loop_agent.schemas import LoopAgentAction, LoopAgentDecision, LoopAgentObservation, LoopAgentStopReason

        def decide(_trace):
            return LoopAgentDecision(
                action=LoopAgentAction.CALL_TOOL,
                capability="applications.find_apply_entry",
                tool_input={"job_id": "lead-1"},
                reason="Need browser executor to open apply page.",
            )

        def execute(_decision):
            return LoopAgentObservation(
                status="waiting_user",
                summary="需要用户登录招聘系统",
                requires_user_action=True,
                tool_call_id="tool-call-apply",
            )

        result = LoopAgentController(max_steps=3).run(decide_next_step=decide, execute_step=execute)

        self.assertEqual(LoopAgentStopReason.WAITING_USER, result.stop_reason)
        self.assertEqual(1, result.executed_step_count)
        self.assertTrue(result.requires_user_action)
        self.assertEqual("applications.find_apply_entry", result.pending_decision.capability)
        self.assertEqual("需要用户登录招聘系统", result.trace[0].observation_summary)

    def test_controller_uses_observation_suggested_next_decision_before_asking_model_again(self) -> None:
        from app.agent_runtime.loop_agent.controller import LoopAgentController
        from app.agent_runtime.loop_agent.schemas import LoopAgentAction, LoopAgentDecision, LoopAgentObservation, LoopAgentStopReason

        decide_calls = []
        executed_queries = []

        def decide(trace):
            decide_calls.append(len(trace))
            return LoopAgentDecision(
                action=LoopAgentAction.CALL_TOOL,
                capability="external.web_search",
                tool_input={"query": "腾讯 招聘"},
                reason="Need an initial search.",
            )

        def execute(decision):
            executed_queries.append(decision.tool_input["query"])
            if len(executed_queries) == 1:
                return LoopAgentObservation(
                    status="succeeded",
                    summary="结果偏题，建议换关键词再查",
                    tool_call_id="tool-call-1",
                    suggested_next_decision=LoopAgentDecision(
                        action=LoopAgentAction.CALL_TOOL,
                        capability="external.web_search",
                        tool_input={"query": "腾讯 校园招聘 官网 2026"},
                        reason="Retry with official campus recruiting terms.",
                        metadata={"source": "reflection"},
                    ),
                    metadata={"quality": "bad"},
                )
            return LoopAgentObservation(
                status="succeeded",
                summary="找到腾讯校招官网",
                tool_call_id="tool-call-2",
                suggested_next_decision=LoopAgentDecision(
                    action=LoopAgentAction.FINAL_ANSWER,
                    message="已找到腾讯校招官网入口",
                    reason="The second search found the target source.",
                ),
                metadata={"quality": "good"},
            )

        result = LoopAgentController(max_steps=3).run(decide_next_step=decide, execute_step=execute)

        self.assertEqual(LoopAgentStopReason.MODEL_FINAL, result.stop_reason)
        self.assertEqual([0], decide_calls)
        self.assertEqual(["腾讯 招聘", "腾讯 校园招聘 官网 2026"], executed_queries)
        self.assertEqual(2, result.executed_step_count)
        self.assertEqual("已找到腾讯校招官网入口", result.final_answer)
        self.assertEqual("reflection", result.trace[0].metadata["observation"]["suggested_next_decision"]["metadata"]["source"])

    def test_controller_records_lifecycle_events_for_each_loop_step(self) -> None:
        from app.agent_runtime.loop_agent.controller import LoopAgentController
        from app.agent_runtime.loop_agent.events import LoopAgentEventType
        from app.agent_runtime.loop_agent.schemas import LoopAgentAction, LoopAgentDecision, LoopAgentObservation, LoopAgentStopReason

        def decide(trace):
            if not trace:
                return LoopAgentDecision(
                    action=LoopAgentAction.CALL_TOOL,
                    capability="external.web_search",
                    tool_input={"query": "腾讯 校园招聘 官网 2026"},
                    reason="Need official source.",
                )
            return LoopAgentDecision(
                action=LoopAgentAction.FINAL_ANSWER,
                message="找到官方入口",
                reason="Search observation is sufficient.",
            )

        def execute(_decision):
            return LoopAgentObservation(status="succeeded", summary="找到官网", tool_call_id="tool-call-1")

        result = LoopAgentController(max_steps=2).run(
            decide_next_step=decide,
            execute_step=execute,
            session_id="session-1",
            task_id="task-1",
            run_id="run-1",
        )

        self.assertEqual(LoopAgentStopReason.MODEL_FINAL, result.stop_reason)
        self.assertEqual(
            [
                LoopAgentEventType.TASK_STARTED.value,
                LoopAgentEventType.TURN_STARTED.value,
                LoopAgentEventType.MODEL_DECISION.value,
                LoopAgentEventType.TOOL_STARTED.value,
                LoopAgentEventType.TOOL_FINISHED.value,
                LoopAgentEventType.TURN_FINISHED.value,
                LoopAgentEventType.TURN_STARTED.value,
                LoopAgentEventType.MODEL_DECISION.value,
                LoopAgentEventType.TASK_FINISHED.value,
            ],
            [event.event_type.value for event in result.events],
        )
        event_payloads = result.to_metadata_dict()["events"]
        self.assertEqual("session-1", event_payloads[0]["session_id"])
        self.assertEqual("external.web_search", event_payloads[3]["capability"])
        self.assertEqual("tool-call-1", event_payloads[4]["tool_call_id"])


if __name__ == "__main__":
    unittest.main()
