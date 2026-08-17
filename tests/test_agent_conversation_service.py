import sys
import unittest
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class AgentConversationServiceTest(unittest.TestCase):
    def setUp(self):
        from app.db.base import Base
        from app.domains.automation import models as automation_models  # noqa: F401
        from app.domains.conversations import models as conversation_models  # noqa: F401

        self.engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

    def tearDown(self):
        self.engine.dispose()

    def test_create_session_and_append_messages_updates_read_order_and_counts(self):
        from app.domains.conversations.models import AgentMessageRole, AgentSessionStatus
        from app.domains.conversations.repository import ConversationRepository
        from app.domains.conversations.schemas import AgentMessageCreate
        from app.domains.conversations.service import ConversationService

        with self.Session() as session:
            service = ConversationService(ConversationRepository(session))
            conversation = service.create_session(
                title="秋招 Agent",
                primary_intent="job_discovery",
            )

            first = service.append_message(
                conversation.id,
                AgentMessageCreate(
                    role=AgentMessageRole.USER,
                    content_text="帮我继续找 Java 后端秋招岗位",
                    visible_content_text="帮我继续找 Java 后端秋招岗位",
                    token_estimate=20,
                ),
            )
            second = service.append_message(
                conversation.id,
                AgentMessageCreate(
                    role=AgentMessageRole.ASSISTANT,
                    content_text="我会先读取最近的岗位线索。",
                    visible_content_text="我会先读取最近的岗位线索。",
                    token_estimate=14,
                ),
            )
            session.commit()

            persisted = service.get_session(conversation.id)
            messages = service.list_messages(conversation.id, limit=10)

        self.assertEqual(AgentSessionStatus.ACTIVE, persisted.status)
        self.assertEqual("job_discovery", persisted.primary_intent)
        self.assertEqual(2, persisted.message_count)
        self.assertIsNotNone(persisted.last_message_at)
        self.assertEqual([first.id, second.id], [message.id for message in messages])

    def test_context_summary_marks_old_messages_compacted_without_deleting_transcript(self):
        from app.domains.conversations.models import AgentMessageRole
        from app.domains.conversations.repository import ConversationRepository
        from app.domains.conversations.schemas import AgentContextSummaryCreate, AgentMessageCreate
        from app.domains.conversations.service import ConversationService

        with self.Session() as session:
            service = ConversationService(ConversationRepository(session))
            conversation = service.create_session(title="公众号同步", primary_intent="job_discovery")
            user_message = service.append_message(
                conversation.id,
                AgentMessageCreate(
                    role=AgentMessageRole.USER,
                    content_text="同步大连海事就业公众号",
                    visible_content_text="同步大连海事就业公众号",
                    token_estimate=12,
                ),
            )
            tool_result = service.append_message(
                conversation.id,
                AgentMessageCreate(
                    role=AgentMessageRole.TOOL_RESULT,
                    content_text="发现 4 篇候选文章",
                    runtime_content_text="article_candidate_ids=[a1,a2,a3,a4]",
                    token_estimate=18,
                ),
            )

            summary = service.create_context_summary(
                conversation.id,
                AgentContextSummaryCreate(
                    summary_text=(
                        "Progress: 已发现公众号候选文章。\n"
                        "Critical Context: 候选文章已落库，下一步抽取招聘开放信号。"
                    ),
                    summary_json={"Progress": {"Done": ["发现公众号候选文章"]}},
                    covered_message_start_id=user_message.id,
                    covered_message_end_id=tool_result.id,
                    first_kept_message_id=tool_result.id,
                    token_estimate=30,
                    created_by="deterministic_compactor",
                ),
            )
            compacted_count = service.mark_messages_compacted(
                conversation.id,
                [user_message.id],
                summary.id,
            )
            session.commit()

            persisted = service.get_session(conversation.id)
            latest_summary = service.get_latest_summary(conversation.id)
            messages = service.list_messages(conversation.id, limit=10)

        self.assertEqual(1, compacted_count)
        self.assertEqual(summary.id, persisted.last_context_summary_id)
        self.assertEqual(summary.id, latest_summary.id)
        self.assertEqual(2, len(messages))
        self.assertEqual(summary.id, messages[0].compacted_by_summary_id)
        self.assertIsNone(messages[1].compacted_by_summary_id)

    def test_message_read_schema_excludes_runtime_only_content(self):
        from app.domains.conversations.models import AgentMessageRole
        from app.domains.conversations.repository import ConversationRepository
        from app.domains.conversations.schemas import AgentMessageCreate, AgentMessageRead
        from app.domains.conversations.service import ConversationService

        with self.Session() as session:
            service = ConversationService(ConversationRepository(session))
            conversation = service.create_session(title="工具结果", primary_intent="job_discovery")
            message = service.append_message(
                conversation.id,
                AgentMessageCreate(
                    role=AgentMessageRole.TOOL_RESULT,
                    content_text="工具结果已保存。",
                    visible_content_text="工具结果已保存。",
                    runtime_content_text="raw html payload with internal details",
                    token_estimate=8,
                ),
            )
            session.commit()

            payload = AgentMessageRead.model_validate(message).model_dump()

        self.assertEqual("工具结果已保存。", payload["visible_content_text"])
        self.assertNotIn("runtime_content_text", payload)

    def test_unknown_session_errors_are_explicit(self):
        from app.domains.conversations.repository import ConversationRepository
        from app.domains.conversations.service import ConversationService

        with self.Session() as session:
            service = ConversationService(ConversationRepository(session))

            with self.assertRaisesRegex(ValueError, "Agent session not found: missing-session"):
                service.get_session("missing-session")


if __name__ == "__main__":
    unittest.main()
