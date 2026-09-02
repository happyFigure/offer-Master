import sys
import unittest
from asyncio import run
from pathlib import Path

from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class AgentCompactApiTest(unittest.TestCase):
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

    def _app(self):
        from app.db.session import get_db_session
        from app.main import create_app

        app = create_app()

        def override_session():
            with self.Session() as session:
                yield session

        app.dependency_overrides[get_db_session] = override_session
        return app

    def _create_session_with_messages(self):
        from app.domains.conversations.models import AgentMessageRole
        from app.domains.conversations.repository import ConversationRepository
        from app.domains.conversations.schemas import AgentMessageCreate
        from app.domains.conversations.service import ConversationService

        with self.Session() as session:
            service = ConversationService(ConversationRepository(session))
            conversation = service.create_session(title="长对话压缩", primary_intent="agent_memory")
            messages = [
                service.append_message(
                    conversation.id,
                    AgentMessageCreate(
                        role=AgentMessageRole.USER,
                        content_text="用户目标：找 2027 届 Java 后端秋招岗位",
                        visible_content_text="用户目标：找 2027 届 Java 后端秋招岗位",
                        token_estimate=900,
                    ),
                ),
                service.append_message(
                    conversation.id,
                    AgentMessageCreate(
                        role=AgentMessageRole.ASSISTANT,
                        content_text="已确认：优先国内互联网、大厂和外企，投递前必须验证。",
                        visible_content_text="已确认：优先国内互联网、大厂和外企，投递前必须验证。",
                        token_estimate=900,
                    ),
                ),
                service.append_message(
                    conversation.id,
                    AgentMessageCreate(
                        role=AgentMessageRole.USER,
                        content_text="最近消息：下一步做记忆压缩。",
                        visible_content_text="最近消息：下一步做记忆压缩。",
                        token_estimate=100,
                    ),
                ),
                service.append_message(
                    conversation.id,
                    AgentMessageCreate(
                        role=AgentMessageRole.ASSISTANT,
                        content_text="最近回复：准备生成 summary。",
                        visible_content_text="最近回复：准备生成 summary。",
                        token_estimate=100,
                    ),
                ),
            ]
            session.commit()
            return conversation.id, [message.id for message in messages]

    def test_manual_compact_persists_summary_and_marks_old_messages(self):
        from app.domains.conversations.models import AgentContextSummary, AgentMessage, AgentSession

        app = self._app()
        session_id, message_ids = self._create_session_with_messages()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.post(
                    f"/api/v1/agent/sessions/{session_id}/compact",
                    json={"context_window": 2000, "reserve_tokens": 100, "keep_recent_tokens": 250},
                )

        response = run(call_api())

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual(2, payload["covered_message_count"])
        self.assertEqual(message_ids[2], payload["first_kept_message_id"])
        self.assertGreater(payload["token_estimate_before"], payload["token_estimate_after"])
        self.assertIn("Goal", payload["summary"]["summary_text"])
        self.assertIn("用户目标：找 2027 届 Java 后端秋招岗位", payload["summary"]["summary_text"])

        with self.Session() as session:
            stored_session = session.get(AgentSession, session_id)
            summaries = session.scalars(select(AgentContextSummary)).all()
            messages = session.scalars(select(AgentMessage).order_by(AgentMessage.created_at.asc())).all()

        self.assertEqual(1, len(summaries))
        self.assertEqual(payload["summary"]["id"], stored_session.last_context_summary_id)
        self.assertEqual(message_ids[0], summaries[0].covered_message_start_id)
        self.assertEqual(message_ids[1], summaries[0].covered_message_end_id)
        self.assertEqual(message_ids[2], summaries[0].first_kept_message_id)
        self.assertEqual(payload["summary"]["id"], messages[0].compacted_by_summary_id)
        self.assertEqual(payload["summary"]["id"], messages[1].compacted_by_summary_id)
        self.assertIsNone(messages[2].compacted_by_summary_id)
        self.assertIsNone(messages[3].compacted_by_summary_id)
        self.assertEqual(4, len(messages))

    def test_second_manual_compact_references_previous_summary_without_overwriting_it(self):
        from app.domains.conversations.models import AgentContextSummary, AgentMessageRole
        from app.domains.conversations.repository import ConversationRepository
        from app.domains.conversations.schemas import AgentMessageCreate
        from app.domains.conversations.service import ConversationService

        app = self._app()
        session_id, _message_ids = self._create_session_with_messages()

        async def compact_once():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.post(
                    f"/api/v1/agent/sessions/{session_id}/compact",
                    json={"context_window": 2000, "reserve_tokens": 100, "keep_recent_tokens": 250},
                )

        first_response = run(compact_once())
        if first_response.status_code != 200:
            self.assertEqual(200, first_response.status_code)
            return
        first_summary_id = first_response.json()["summary"]["id"]

        with self.Session() as session:
            service = ConversationService(ConversationRepository(session))
            service.append_message(
                session_id,
                AgentMessageCreate(
                    role=AgentMessageRole.USER,
                    content_text="新进展：Phase 4 已完成。",
                    visible_content_text="新进展：Phase 4 已完成。",
                    token_estimate=900,
                ),
            )
            service.append_message(
                session_id,
                AgentMessageCreate(
                    role=AgentMessageRole.ASSISTANT,
                    content_text="下一步：进入 manual compact API。",
                    visible_content_text="下一步：进入 manual compact API。",
                    token_estimate=100,
                ),
            )
            session.commit()

        second_response = run(compact_once())
        second_payload = second_response.json()

        with self.Session() as session:
            summaries = session.scalars(select(AgentContextSummary).order_by(AgentContextSummary.created_at.asc())).all()

        self.assertEqual(200, second_response.status_code)
        self.assertEqual(2, len(summaries))
        self.assertEqual(first_summary_id, summaries[0].id)
        self.assertEqual(first_summary_id, second_payload["summary"]["previous_summary_id"])
        self.assertEqual(first_summary_id, summaries[1].previous_summary_id)
        self.assertNotEqual(first_summary_id, second_payload["summary"]["id"])

    def test_manual_compact_missing_session_returns_404(self):
        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.post(
                    "/api/v1/agent/sessions/missing-session/compact",
                    json={"context_window": 2000, "reserve_tokens": 100, "keep_recent_tokens": 250},
                )

        response = run(call_api())

        self.assertEqual(404, response.status_code)
        self.assertIn("Agent session not found", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
