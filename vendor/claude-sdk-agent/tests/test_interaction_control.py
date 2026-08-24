from __future__ import annotations

import asyncio
import unittest

from src.approval_control import ApprovalRuntimeRegistry
from src.question_control import QuestionRuntimeRegistry


class InteractionControlTests(unittest.TestCase):
    def test_approval_registry_resolves_requests_and_streams_updates(self) -> None:
        async def scenario() -> None:
            registry = ApprovalRuntimeRegistry()
            stream_task = asyncio.create_task(self._collect_first_two(registry.stream_requests("session-1")))
            await asyncio.sleep(0)
            handle = await registry.create_request(
                session_id="session-1",
                run_id="run-1",
                claude_session_id="claude-1",
                tool_name="Bash",
                tool_input={"command": "echo hi"},
                tool_use_id="tool-1",
                agent_id="agent-1",
                blocked_path="",
                decision_reason="",
                title="Allow Bash?",
                display_name="Bash",
                description="Execute shell",
            )
            listing = await registry.list_requests("session-1")
            self.assertEqual(listing[0]["requestId"], handle.request_id)
            await registry.resolve_request("session-1", handle.request_id, decision="allow", reason="ok")
            items = await stream_task
            self.assertEqual(items[0]["status"], "pending")
            self.assertEqual(items[-1]["status"], "allowed")

        asyncio.run(scenario())

    def test_question_registry_creates_answers_and_streams_updates(self) -> None:
        async def scenario() -> None:
            registry = QuestionRuntimeRegistry()
            stream_task = asyncio.create_task(self._collect_first_two(registry.stream_questions("session-1")))
            await asyncio.sleep(0)
            handle = await registry.create_question(
                session_id="session-1",
                run_id="run-1",
                claude_session_id="claude-1",
                prompt="Need more detail?",
                title="Question",
                description="Follow up",
                metadata={"source": "test"},
            )
            listing = await registry.list_questions("session-1")
            self.assertEqual(listing[0]["questionId"], handle.question_id)
            await registry.answer_question("session-1", handle.question_id, answer="more detail")
            items = await stream_task
            self.assertEqual(items[0]["status"], "pending")
            self.assertEqual(items[-1]["status"], "answered")
            self.assertEqual(items[-1]["answer"], "more detail")

        asyncio.run(scenario())

    def test_question_stream_replays_only_pending_questions(self) -> None:
        async def scenario() -> None:
            registry = QuestionRuntimeRegistry()
            answered = await registry.create_question(
                session_id="session-1",
                run_id="run-1",
                claude_session_id="claude-1",
                prompt="Already answered?",
            )
            await registry.answer_question("session-1", answered.question_id, answer="done")
            pending = await registry.create_question(
                session_id="session-1",
                run_id="run-2",
                claude_session_id="claude-1",
                prompt="Still pending?",
            )

            stream = registry.stream_questions("session-1")
            item = await stream.__anext__()
            await stream.aclose()

            self.assertEqual(item["questionId"], pending.question_id)
            self.assertEqual(item["status"], "pending")

        asyncio.run(scenario())

    async def _collect_first_two(self, stream):
        items = []
        async for item in stream:
            items.append(item)
            if len(items) >= 2:
                break
        return items
