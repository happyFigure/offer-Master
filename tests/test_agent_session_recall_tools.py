import sys
import unittest
from asyncio import run
from pathlib import Path

from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class AgentSessionRecallToolsTest(unittest.TestCase):
    def setUp(self):
        from app.db.base import Base
        from app.domains.automation import models as automation_models  # noqa: F401
        from app.domains.conversations import models as conversation_models  # noqa: F401

        self.engine = create_engine(
            "sqlite+pysqlite://",
            connect_args={"check_same_thread": False},
            future=True,
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

    def tearDown(self):
        self.engine.dispose()

    def _service(self, session):
        from app.domains.conversations.repository import ConversationRepository
        from app.domains.conversations.service import ConversationService

        return ConversationService(ConversationRepository(session))

    def _app(self):
        from app.db.session import get_db_session
        from app.main import create_app

        app = create_app()

        def override_session():
            with self.Session() as session:
                yield session

        app.dependency_overrides[get_db_session] = override_session
        return app

    def _seed_session(self):
        from app.domains.conversations.models import AgentMessageRole
        from app.domains.conversations.schemas import AgentContextSummaryCreate, AgentMessageCreate

        with self.Session() as session:
            service = self._service(session)
            conversation = service.create_session(title="公众号解析复盘", primary_intent="job_discovery")
            first = service.append_message(
                conversation.id,
                AgentMessageCreate(
                    role=AgentMessageRole.USER,
                    content_text="帮我解析大连海事大学就业公众号",
                    visible_content_text="帮我解析大连海事大学就业公众号",
                    token_estimate=20,
                ),
            )
            middle = service.append_message(
                conversation.id,
                AgentMessageCreate(
                    role=AgentMessageRole.ASSISTANT,
                    content_text="公众号解析失败原因：页面只返回摘要，需要可见页面或文章列表入口。",
                    visible_content_text="公众号解析失败原因：页面只返回摘要，需要可见页面或文章列表入口。",
                    token_estimate=30,
                ),
            )
            last = service.append_message(
                conversation.id,
                AgentMessageCreate(
                    role=AgentMessageRole.USER,
                    content_text="下一步先做历史召回工具",
                    visible_content_text="下一步先做历史召回工具",
                    token_estimate=10,
                ),
            )
            summary = service.create_context_summary(
                conversation.id,
                AgentContextSummaryCreate(
                    summary_text=(
                        "Goal: 复盘公众号解析失败。\n"
                        "Retrieval Hints: 公众号 页面只返回摘要 可见页面 文章列表入口"
                    ),
                    covered_message_start_id=first.id,
                    covered_message_end_id=middle.id,
                    first_kept_message_id=last.id,
                    token_estimate=35,
                    created_by="test",
                ),
            )
            session.commit()
            return conversation.id, first.id, middle.id, last.id, summary.id

    def test_sessions_search_matches_message_visible_text(self):
        from app.agent_runtime.memory.memory_tools import sessions_search

        session_id, _first_id, middle_id, _last_id, _summary_id = self._seed_session()

        with self.Session() as session:
            result = sessions_search(session, query="解析失败", limit=10)

        self.assertEqual("sessions", result.corpus)
        self.assertEqual(1, len([item for item in result.items if item.message_id == middle_id]))
        message_hit = next(item for item in result.items if item.message_id == middle_id)
        self.assertEqual(session_id, message_hit.session_id)
        self.assertEqual("message", message_hit.source_type)
        self.assertIn("公众号解析失败原因", message_hit.excerpt)

    def test_sessions_search_matches_summary_retrieval_hints(self):
        from app.agent_runtime.memory.memory_tools import sessions_search

        session_id, _first_id, _middle_id, _last_id, summary_id = self._seed_session()

        with self.Session() as session:
            result = sessions_search(session, query="文章列表入口", limit=10)

        summary_hit = next(item for item in result.items if item.summary_id == summary_id)
        self.assertEqual(session_id, summary_hit.session_id)
        self.assertEqual("summary", summary_hit.source_type)
        self.assertIn("Retrieval Hints", summary_hit.excerpt)

    def test_sessions_history_returns_bounded_window_around_message(self):
        from app.agent_runtime.memory.memory_tools import sessions_history

        session_id, first_id, middle_id, last_id, _summary_id = self._seed_session()

        with self.Session() as session:
            window = sessions_history(
                session,
                session_key=session_id,
                around_message_id=middle_id,
                window_before=1,
                window_after=1,
            )

        self.assertEqual(session_id, window.session_id)
        self.assertEqual(middle_id, window.around_message_id)
        self.assertEqual([first_id, middle_id, last_id], [message.id for message in window.messages])
        self.assertTrue(window.truncated_before is False)
        self.assertTrue(window.truncated_after is False)

    def test_sessions_recall_api_exposes_search_and_history(self):
        session_id, _first_id, middle_id, _last_id, _summary_id = self._seed_session()
        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                search_response = await client.get(
                    "/api/v1/agent/sessions/search",
                    params={"query": "解析失败", "limit": 10},
                )
                history_response = await client.get(
                    f"/api/v1/agent/sessions/{session_id}/history",
                    params={"around_message_id": middle_id, "window_before": 1, "window_after": 1},
                )
                return search_response, history_response

        search_response, history_response = run(call_api())

        self.assertEqual(200, search_response.status_code)
        self.assertEqual(200, history_response.status_code)
        self.assertEqual("sessions", search_response.json()["corpus"])
        self.assertEqual(session_id, history_response.json()["session_id"])
        self.assertEqual(3, len(history_response.json()["messages"]))


if __name__ == "__main__":
    unittest.main()
