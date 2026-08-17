import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


@dataclass(frozen=True)
class MessageStub:
    id: str
    role: str
    content_text: str | None = None
    visible_content_text: str | None = None
    runtime_content_text: str | None = None
    message_kind: str | None = None
    token_estimate: int | None = None
    parent_message_id: str | None = None
    exclude_from_context: bool = False
    compacted_by_summary_id: str | None = None
    metadata_json: dict | None = None


class AgentTranscriptHygieneTest(unittest.TestCase):
    def setUp(self):
        from app.db.base import Base
        from app.domains.automation import models as automation_models  # noqa: F401
        from app.domains.conversations import models as conversation_models  # noqa: F401

        self.engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

    def tearDown(self):
        self.engine.dispose()

    def _service(self, session):
        from app.domains.conversations.repository import ConversationRepository
        from app.domains.conversations.service import ConversationService

        return ConversationService(ConversationRepository(session))

    def test_repair_keeps_adjacent_tool_call_and_result_unchanged(self):
        from app.agent_runtime.memory.transcript_hygiene import repair_tool_result_pairing

        messages = [
            MessageStub(id="call-1", role="tool_call", content_text="fetch source", token_estimate=5),
            MessageStub(
                id="result-1",
                role="tool_result",
                content_text="source saved",
                parent_message_id="call-1",
                token_estimate=5,
            ),
        ]

        result = repair_tool_result_pairing(messages)

        self.assertEqual(["call-1", "result-1"], [message.id for message in result.messages])
        self.assertEqual([], result.synthetic_error_ids)
        self.assertEqual({}, result.excluded_reasons)

    def test_repair_inserts_synthetic_error_when_tool_result_is_missing(self):
        from app.agent_runtime.memory.transcript_hygiene import repair_tool_result_pairing

        messages = [
            MessageStub(id="call-1", role="tool_call", content_text="fetch source", token_estimate=5),
            MessageStub(id="assistant-1", role="assistant", content_text="continue", token_estimate=5),
        ]

        result = repair_tool_result_pairing(messages)

        self.assertEqual(["call-1", "synthetic-error-call-1", "assistant-1"], [message.id for message in result.messages])
        self.assertEqual(["synthetic-error-call-1"], result.synthetic_error_ids)
        synthetic = result.messages[1]
        self.assertEqual("synthetic_error", synthetic.message_kind)
        self.assertIn("工具结果缺失，本轮不能判断工具成功", synthetic.visible_content_text)
        self.assertEqual("TOOL_RESULT_MISSING", synthetic.metadata_json["error_code"])

    def test_orphan_tool_result_is_excluded_with_reason(self):
        from app.agent_runtime.memory.transcript_hygiene import repair_tool_result_pairing

        messages = [
            MessageStub(id="result-1", role="tool_result", content_text="orphan result", token_estimate=5),
            MessageStub(id="user-1", role="user", content_text="next", token_estimate=5),
        ]

        result = repair_tool_result_pairing(messages)

        self.assertEqual(["user-1"], [message.id for message in result.messages])
        self.assertEqual({"result-1": "ORPHAN_TOOL_RESULT"}, result.excluded_reasons)

    def test_filter_visible_transcript_strips_runtime_content(self):
        from app.agent_runtime.memory.transcript_hygiene import filter_visible_transcript

        messages = [
            MessageStub(
                id="result-1",
                role="tool_result",
                content_text="safe summary",
                visible_content_text="safe summary",
                runtime_content_text="secret raw payload",
                token_estimate=5,
            ),
        ]

        visible = filter_visible_transcript(messages)

        self.assertEqual(1, len(visible))
        self.assertEqual("safe summary", visible[0].visible_content_text)
        self.assertIsNone(visible[0].runtime_content_text)
        self.assertNotIn("secret raw payload", visible[0].content_text or "")

    def test_context_builder_inserts_synthetic_error_for_missing_tool_result(self):
        from app.agent_runtime.memory.context_builder import ContextBuildConfig, MemoryContextBuilder
        from app.domains.conversations.models import AgentMessageRole
        from app.domains.conversations.schemas import AgentMessageCreate

        with self.Session() as session:
            service = self._service(session)
            conversation = service.create_session(title="tool hygiene", primary_intent="agent_chat")
            service.append_message(
                conversation.id,
                AgentMessageCreate(
                    role=AgentMessageRole.USER,
                    content_text="import this URL",
                    visible_content_text="import this URL",
                    token_estimate=5,
                ),
            )
            tool_call = service.append_message(
                conversation.id,
                AgentMessageCreate(
                    role=AgentMessageRole.TOOL_CALL,
                    content_text="call url_import",
                    visible_content_text="call url_import",
                    token_estimate=5,
                ),
            )
            session.commit()

            built = MemoryContextBuilder(service).build(
                conversation.id,
                new_user_message=None,
                config=ContextBuildConfig(max_recent_messages=10),
            )

        contents = "\n".join(message["content"] for message in built.llm_messages)
        self.assertIn("工具结果缺失，本轮不能判断工具成功", contents)
        self.assertIn(f"synthetic-error-{tool_call.id}", built.loaded_session_history_ids)
        self.assertEqual([f"synthetic-error-{tool_call.id}"], built.context_metadata["hygiene_synthetic_error_ids"])


if __name__ == "__main__":
    unittest.main()
