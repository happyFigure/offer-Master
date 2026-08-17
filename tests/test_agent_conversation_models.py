import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class AgentConversationModelsTest(unittest.TestCase):
    def test_agent_conversation_tables_are_registered_with_context_boundaries(self):
        from app.db.base import Base
        from app.domains.automation import models as automation_models  # noqa: F401
        from app.domains.conversations.models import AgentMessageRole

        expected_tables = {
            "agent_sessions",
            "agent_messages",
            "agent_context_summaries",
        }
        self.assertTrue(expected_tables.issubset(set(Base.metadata.tables)))

        sessions = Base.metadata.tables["agent_sessions"]
        messages = Base.metadata.tables["agent_messages"]
        summaries = Base.metadata.tables["agent_context_summaries"]

        self.assertEqual(
            {
                "id",
                "title",
                "status",
                "primary_intent",
                "current_agent_run_id",
                "last_context_summary_id",
                "message_count",
                "created_at",
                "updated_at",
                "last_message_at",
                "metadata_json",
            },
            set(sessions.columns.keys()),
        )
        self.assertEqual(
            {
                "id",
                "session_id",
                "role",
                "message_kind",
                "agent_id",
                "recipient_agent_id",
                "visibility_scope",
                "content_text",
                "content_json",
                "visible_content_text",
                "runtime_content_text",
                "content_type",
                "provenance_kind",
                "agent_run_id",
                "workflow_run_id",
                "tool_call_log_id",
                "parent_message_id",
                "token_estimate",
                "exclude_from_context",
                "compacted_by_summary_id",
                "created_at",
                "metadata_json",
            },
            set(messages.columns.keys()),
        )
        self.assertEqual(
            {
                "id",
                "session_id",
                "summary_text",
                "summary_json",
                "covered_message_start_id",
                "covered_message_end_id",
                "first_kept_message_id",
                "previous_summary_id",
                "token_estimate",
                "created_at",
                "created_by",
                "metadata_json",
            },
            set(summaries.columns.keys()),
        )
        self.assertIn("tool_result", {role.value for role in AgentMessageRole})
        self.assertGreaterEqual(sessions.c.current_agent_run_id.type.length, 64)
        self.assertGreaterEqual(messages.c.agent_run_id.type.length, 64)
        self.assertEqual(
            "agent_sessions.id",
            str(next(iter(messages.c.session_id.foreign_keys)).column),
        )
        self.assertEqual(
            "agent_context_summaries.id",
            str(next(iter(messages.c.compacted_by_summary_id.foreign_keys)).column),
        )

    def test_can_persist_session_messages_and_compaction_summary(self):
        from app.db.base import Base
        from app.domains.automation import models as automation_models  # noqa: F401
        from app.domains.conversations.models import (
            AgentContextSummary,
            AgentMessage,
            AgentMessageKind,
            AgentMessageRole,
            AgentMessageVisibilityScope,
            AgentSession,
            AgentSessionStatus,
        )

        engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine, expire_on_commit=False, future=True)

        with Session() as session:
            conversation = AgentSession(
                title="公众号岗位同步",
                status=AgentSessionStatus.ACTIVE,
                primary_intent="job_discovery",
            )
            user_message = AgentMessage(
                session=conversation,
                role=AgentMessageRole.USER,
                message_kind=AgentMessageKind.USER_TEXT,
                visibility_scope=AgentMessageVisibilityScope.USER_VISIBLE,
                content_text="继续同步大连海事就业公众号",
                visible_content_text="继续同步大连海事就业公众号",
                token_estimate=18,
            )
            tool_result = AgentMessage(
                session=conversation,
                role=AgentMessageRole.TOOL_RESULT,
                message_kind=AgentMessageKind.TOOL_RESULT,
                visibility_scope=AgentMessageVisibilityScope.RUNTIME_ONLY,
                content_text="抓取到 3 篇候选文章",
                runtime_content_text="raw payload stored in article_candidates",
                token_estimate=12,
            )
            summary = AgentContextSummary(
                session=conversation,
                summary_text="Progress: 已同步公众号候选文章。\nNext Steps: 抽取招聘开放信号。",
                summary_json={"Progress": {"Done": ["同步候选文章"]}},
                covered_message_start=user_message,
                covered_message_end=tool_result,
                first_kept_message=tool_result,
                token_estimate=26,
                created_by="deterministic_compactor",
            )
            user_message.compacted_by_summary = summary
            conversation.last_context_summary = summary
            conversation.message_count = 2

            session.add(conversation)
            session.commit()

            persisted = session.query(AgentSession).one()

        self.assertEqual("公众号岗位同步", persisted.title)
        self.assertEqual(2, persisted.message_count)
        self.assertEqual(summary.id, persisted.last_context_summary_id)
        self.assertEqual(
            [AgentMessageRole.USER, AgentMessageRole.TOOL_RESULT],
            [message.role for message in persisted.messages],
        )
        self.assertEqual("deterministic_compactor", persisted.context_summaries[0].created_by)
        self.assertEqual(summary.id, persisted.messages[0].compacted_by_summary_id)
        self.assertEqual("raw payload stored in article_candidates", persisted.messages[1].runtime_content_text)

    def test_agent_conversation_migration_creates_memory_tables(self):
        migration = PROJECT_ROOT / "infra" / "migrations" / "versions" / "20260814_0006_agent_conversation_memory_tables.py"
        self.assertTrue(migration.is_file())

        from app.core.config import get_settings

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "data") as tmp_dir:
            db_path = Path(tmp_dir) / "agent_conversation_migration_check.sqlite"
            config = Config(str(PROJECT_ROOT / "alembic.ini"))
            config.set_main_option("script_location", str(PROJECT_ROOT / "infra" / "migrations"))

            with patch.dict(
                "os.environ",
                {"JOBPILOT_DATABASE_URL": f"sqlite+pysqlite:///{db_path.as_posix()}"},
                clear=False,
            ):
                get_settings.cache_clear()
                command.upgrade(config, "head")

            engine = create_engine(f"sqlite+pysqlite:///{db_path.as_posix()}", future=True)
            inspector = inspect(engine)
            self.assertTrue(
                {
                    "agent_sessions",
                    "agent_messages",
                    "agent_context_summaries",
                }.issubset(set(inspector.get_table_names()))
            )
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
