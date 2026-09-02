import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class OuterSessionLoopControllerTest(unittest.TestCase):
    def test_starts_active_task_and_finishes_when_inner_loop_returns_final_answer(self) -> None:
        from app.agent_runtime.loop_agent.outer_session import OuterSessionLoopController, OuterSessionStatus
        from app.agent_runtime.loop_agent.schemas import LoopAgentRunResult, LoopAgentStopReason

        requests = []

        def run_inner_loop(request):
            requests.append(request)
            return LoopAgentRunResult(
                stop_reason=LoopAgentStopReason.MODEL_FINAL,
                final_answer="已找到腾讯校招入口。",
            )

        controller = OuterSessionLoopController(run_inner_loop=run_inner_loop)

        result = controller.handle_user_message(
            session_id="session-1",
            user_message="帮我找腾讯校招入口",
            task_id="task-1",
            run_id="run-1",
        )

        self.assertEqual(OuterSessionStatus.FINISHED, result.status)
        self.assertEqual("已找到腾讯校招入口。", result.final_answer)
        self.assertFalse(result.requires_user_action)
        self.assertEqual("task-1", result.task_id)
        self.assertEqual("run-1", result.run_id)
        self.assertEqual(1, len(requests))
        self.assertFalse(requests[0].is_resume)
        self.assertEqual("帮我找腾讯校招入口", requests[0].user_message)

        state = controller.get_state("session-1")
        self.assertIsNotNone(state)
        self.assertEqual(OuterSessionStatus.FINISHED, state.status)
        self.assertEqual("task-1", state.active_task_id)
        self.assertEqual(1, state.run_count)

    def test_waiting_user_state_resumes_same_active_task_with_user_followup(self) -> None:
        from app.agent_runtime.loop_agent.outer_session import OuterSessionLoopController, OuterSessionStatus
        from app.agent_runtime.loop_agent.schemas import (
            LoopAgentAction,
            LoopAgentDecision,
            LoopAgentRunResult,
            LoopAgentStopReason,
        )

        requests = []

        def run_inner_loop(request):
            requests.append(request)
            if len(requests) == 1:
                return LoopAgentRunResult(
                    stop_reason=LoopAgentStopReason.WAITING_USER,
                    requires_user_action=True,
                    pending_decision=LoopAgentDecision(
                        action=LoopAgentAction.WAIT_USER,
                        message="请补充你的简历文本。",
                        reason="简历优化需要原始简历。",
                    ),
                )
            return LoopAgentRunResult(
                stop_reason=LoopAgentStopReason.MODEL_FINAL,
                final_answer="已根据 JD 优化简历。",
            )

        controller = OuterSessionLoopController(run_inner_loop=run_inner_loop)

        first = controller.handle_user_message(
            session_id="session-1",
            user_message="帮我根据 Java 后端 JD 优化简历",
            task_id="task-1",
            run_id="run-1",
        )
        second = controller.handle_user_message(
            session_id="session-1",
            user_message="这是我的简历文本：...",
            run_id="run-2",
        )

        self.assertEqual(OuterSessionStatus.WAITING_USER, first.status)
        self.assertTrue(first.requires_user_action)
        self.assertEqual("请补充你的简历文本。", first.waiting_message)
        self.assertEqual(OuterSessionStatus.FINISHED, second.status)
        self.assertEqual("已根据 JD 优化简历。", second.final_answer)

        self.assertEqual(2, len(requests))
        self.assertFalse(requests[0].is_resume)
        self.assertTrue(requests[1].is_resume)
        self.assertEqual("task-1", requests[1].task_id)
        self.assertEqual("帮我根据 Java 后端 JD 优化简历", requests[1].user_goal)
        self.assertEqual("这是我的简历文本：...", requests[1].user_message)
        self.assertEqual("请补充你的简历文本。", requests[1].resume_context["waiting_message"])

        state = controller.get_state("session-1")
        self.assertEqual(OuterSessionStatus.FINISHED, state.status)
        self.assertEqual("task-1", state.active_task_id)
        self.assertEqual(2, state.run_count)
        self.assertEqual(["这是我的简历文本：..."], state.user_followups)

    def test_new_message_after_finished_starts_new_active_task(self) -> None:
        from app.agent_runtime.loop_agent.outer_session import OuterSessionLoopController, OuterSessionStatus
        from app.agent_runtime.loop_agent.schemas import LoopAgentRunResult, LoopAgentStopReason

        task_ids = []

        def run_inner_loop(request):
            task_ids.append(request.task_id)
            return LoopAgentRunResult(
                stop_reason=LoopAgentStopReason.MODEL_FINAL,
                final_answer=f"完成 {request.task_id}",
            )

        controller = OuterSessionLoopController(run_inner_loop=run_inner_loop)

        first = controller.handle_user_message(
            session_id="session-1",
            user_message="先查腾讯",
            task_id="task-1",
            run_id="run-1",
        )
        second = controller.handle_user_message(
            session_id="session-1",
            user_message="再查阿里",
            task_id="task-2",
            run_id="run-2",
        )

        self.assertEqual(OuterSessionStatus.FINISHED, first.status)
        self.assertEqual(OuterSessionStatus.FINISHED, second.status)
        self.assertEqual(["task-1", "task-2"], task_ids)
        self.assertEqual("task-2", controller.get_state("session-1").active_task_id)

    def test_can_begin_and_complete_turn_separately_for_streaming_api(self) -> None:
        from app.agent_runtime.loop_agent.outer_session import OuterSessionLoopController, OuterSessionStatus
        from app.agent_runtime.loop_agent.schemas import LoopAgentRunResult, LoopAgentStopReason

        controller = OuterSessionLoopController(run_inner_loop=lambda request: self.fail("inner loop should not run"))

        request = controller.begin_user_message(
            session_id="session-1",
            user_message="帮我找腾讯校招入口",
            task_id="task-1",
            run_id="workflow-stream-1",
        )

        running_state = controller.get_state("session-1")
        self.assertEqual(OuterSessionStatus.RUNNING, running_state.status)
        self.assertEqual("workflow-stream-1", running_state.active_run_id)
        self.assertEqual(1, running_state.run_count)
        self.assertFalse(request.is_resume)

        result = controller.complete_turn(
            request,
            LoopAgentRunResult(
                stop_reason=LoopAgentStopReason.MODEL_FINAL,
                final_answer="已找到腾讯校招入口。",
            ),
        )

        self.assertEqual(OuterSessionStatus.FINISHED, result.status)
        self.assertEqual("已找到腾讯校招入口。", result.final_answer)
        self.assertEqual(OuterSessionStatus.FINISHED, controller.get_state("session-1").status)


if __name__ == "__main__":
    unittest.main()
