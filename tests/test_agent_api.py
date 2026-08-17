import sys
import unittest
from asyncio import run
from pathlib import Path
from unittest.mock import patch

from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class AgentApiTest(unittest.TestCase):
    def setUp(self):
        from app.db.base import Base
        from app.domains.agent_memory import models as agent_memory_models  # noqa: F401
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
        self._patchers = []

    def tearDown(self):
        for patcher in reversed(self._patchers):
            patcher.stop()
        self.engine.dispose()

    def _app(self, *, llm_client=None):
        from app.api.v1 import agent as agent_api
        from app.db.session import get_db_session
        from app.main import create_app

        patcher = patch.object(agent_api, "_build_agent_llm_client", return_value=llm_client)
        patcher.start()
        self._patchers.append(patcher)

        app = create_app()

        def override_session():
            with self.Session() as session:
                yield session

        app.dependency_overrides[get_db_session] = override_session
        return app

    def test_create_and_list_agent_sessions(self):
        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                created = await client.post(
                    "/api/v1/agent/sessions",
                    json={"title": "秋招规划", "primary_intent": "job_search"},
                )
                if created.status_code != 201:
                    return created, None, None
                listed = await client.get("/api/v1/agent/sessions")
                fetched = await client.get(f"/api/v1/agent/sessions/{created.json()['id']}")
                return created, listed, fetched

        created_response, listed_response, fetched_response = run(call_api())

        self.assertEqual(201, created_response.status_code)
        self.assertIsNotNone(listed_response)
        self.assertIsNotNone(fetched_response)
        self.assertEqual("秋招规划", created_response.json()["title"])
        self.assertEqual("active", created_response.json()["status"])
        self.assertEqual("job_search", created_response.json()["primary_intent"])
        self.assertEqual(0, created_response.json()["message_count"])
        self.assertEqual(200, listed_response.status_code)
        self.assertEqual(created_response.json()["id"], listed_response.json()["items"][0]["id"])
        self.assertEqual(200, fetched_response.status_code)
        self.assertEqual(created_response.json()["id"], fetched_response.json()["id"])

    def test_rename_and_delete_agent_session_hides_it_from_default_list(self):
        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                created = await client.post(
                    "/api/v1/agent/sessions",
                    json={"title": "旧会话标题", "primary_intent": "agent_chat"},
                )
                session_id = created.json()["id"]
                renamed = await client.patch(
                    f"/api/v1/agent/sessions/{session_id}",
                    json={"title": "新的会话标题"},
                )
                delete_response = await client.delete(f"/api/v1/agent/sessions/{session_id}")
                default_list = await client.get("/api/v1/agent/sessions")
                archived_list = await client.get("/api/v1/agent/sessions?include_archived=true")
                return renamed, delete_response, default_list, archived_list

        renamed_response, delete_response, default_list_response, archived_list_response = run(call_api())

        self.assertEqual(200, renamed_response.status_code)
        self.assertEqual("新的会话标题", renamed_response.json()["title"])
        self.assertEqual(204, delete_response.status_code)
        self.assertEqual([], default_list_response.json()["items"])
        self.assertEqual("archived", archived_list_response.json()["items"][0]["status"])
        self.assertEqual("新的会话标题", archived_list_response.json()["items"][0]["title"])

    def test_post_agent_message_stores_user_and_deterministic_assistant_reply(self):
        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                session_response = await client.post(
                    "/api/v1/agent/sessions",
                    json={"title": "AI 对话", "primary_intent": "agent_chat"},
                )
                if session_response.status_code != 201:
                    return session_response, None, None
                session_id = session_response.json()["id"]
                posted = await client.post(
                    f"/api/v1/agent/sessions/{session_id}/messages",
                    json={
                        "content_text": "帮我记住：我主要找 Java 后端岗位",
                        "runtime_content_text": "runtime-only prompt should not leak",
                    },
                )
                messages = await client.get(f"/api/v1/agent/sessions/{session_id}/messages")
                fetched_session = await client.get(f"/api/v1/agent/sessions/{session_id}")
                return posted, messages, fetched_session

        posted_response, messages_response, fetched_session_response = run(call_api())

        self.assertEqual(201, posted_response.status_code)
        self.assertIsNotNone(messages_response)
        self.assertIsNotNone(fetched_session_response)
        posted_payload = posted_response.json()
        self.assertEqual("帮我记住：我主要找 Java 后端岗位", posted_payload["user_message"]["content_text"])
        self.assertEqual("assistant", posted_payload["assistant_message"]["role"])
        self.assertIn("已经记录", posted_payload["assistant_message"]["content_text"])
        self.assertNotIn("runtime_content_text", posted_payload["user_message"])
        self.assertEqual(200, messages_response.status_code)
        messages = messages_response.json()["items"]
        self.assertEqual(["user", "assistant"], [message["role"] for message in messages])
        self.assertEqual("帮我记住：我主要找 Java 后端岗位", messages[0]["content_text"])
        self.assertIn("已经记录", messages[1]["content_text"])
        self.assertEqual(2, fetched_session_response.json()["message_count"])

    def test_post_agent_message_to_missing_session_returns_404(self):
        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.post(
                    "/api/v1/agent/sessions/missing-session/messages",
                    json={"content_text": "hello"},
                )

        response = run(call_api())

        self.assertEqual(404, response.status_code)
        self.assertIn("Agent session not found", response.json()["detail"])

    def test_post_agent_message_attaches_built_context_metadata_to_assistant_message(self):
        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                session_response = await client.post(
                    "/api/v1/agent/sessions",
                    json={"title": "上下文集成", "primary_intent": "agent_chat"},
                )
                session_id = session_response.json()["id"]
                return await client.post(
                    f"/api/v1/agent/sessions/{session_id}/messages",
                    json={"content_text": "这一轮应该先构建上下文"},
                )

        response = run(call_api())

        self.assertEqual(201, response.status_code)
        metadata = response.json()["assistant_message"]["metadata_json"]
        self.assertEqual("deterministic_stub", metadata["response_mode"])
        self.assertTrue(metadata["context_metadata"]["new_user_message_included"])
        self.assertEqual([], metadata["context_metadata"]["loaded_memory_ids"])
        self.assertEqual([], metadata["context_metadata"]["loaded_skill_ids"])
        self.assertTrue(metadata["context_metadata"]["agent_run_id"].startswith("agent-run-"))
        self.assertEqual("final_response", metadata["context_metadata"]["current_step"])
        self.assertIsNotNone(metadata["context_metadata"]["workflow_run_id"])

    def test_post_agent_message_returns_llm_assistant_reply_when_model_is_configured(self):
        from app.infrastructure.llm.chat_client import LLMChatCompletion

        class FakeLLMClient:
            def complete(self, *, messages):
                return LLMChatCompletion(content="模型回复：我会先确认你的 Java 秋招目标。")

        app = self._app(llm_client=FakeLLMClient())

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                session_response = await client.post(
                    "/api/v1/agent/sessions",
                    json={"title": "AI 对话", "primary_intent": "agent_chat"},
                )
                session_id = session_response.json()["id"]
                return await client.post(
                    f"/api/v1/agent/sessions/{session_id}/messages",
                    json={"content_text": "我想找 Java 后端秋招"},
                )

        response = run(call_api())

        self.assertEqual(201, response.status_code)
        assistant = response.json()["assistant_message"]
        self.assertEqual("模型回复：我会先确认你的 Java 秋招目标。", assistant["content_text"])
        self.assertEqual("llm", assistant["metadata_json"]["response_mode"])

    def test_stream_agent_message_returns_sse_tokens_and_persists_turn(self):
        class FakeStreamingLLMClient:
            def stream_complete(self, *, messages):
                self.messages = messages
                yield "你"
                yield "好"

        app = self._app(llm_client=FakeStreamingLLMClient())

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                session_response = await client.post(
                    "/api/v1/agent/sessions",
                    json={"title": "AI 瀵硅瘽", "primary_intent": "agent_chat"},
                )
                session_id = session_response.json()["id"]
                stream_response = await client.post(
                    f"/api/v1/agent/sessions/{session_id}/messages/stream",
                    json={"content_text": "你好"},
                )
                messages_response = await client.get(f"/api/v1/agent/sessions/{session_id}/messages")
                return stream_response, messages_response

        stream_response, messages_response = run(call_api())

        self.assertEqual(200, stream_response.status_code)
        self.assertIn("text/event-stream", stream_response.headers["content-type"])
        stream_text = stream_response.text
        self.assertIn("event: token", stream_text)
        self.assertIn('"content":"你"', stream_text)
        self.assertIn('"content":"好"', stream_text)
        self.assertIn("event: done", stream_text)

        messages = messages_response.json()["items"]
        self.assertEqual(["user", "assistant"], [message["role"] for message in messages])
        self.assertEqual("你好", messages[0]["content_text"])
        self.assertEqual("你好", messages[1]["content_text"])
        self.assertEqual("llm_stream", messages[1]["metadata_json"]["response_mode"])

    def test_xiaohongshu_search_uses_configured_rest_adapter_from_agent_chat(self):
        from app.core.config import get_settings

        class FakeResponse:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return {"success": True, "data": {"items": [{"title": "Tencent 2027"}]}, "message": "ok"}

        rest_calls = []

        def fake_post(url, *, json, headers, timeout):
            rest_calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
            return FakeResponse()

        with patch.dict(
            "os.environ",
            {"JOBPILOT_XIAOHONGSHU_MCP_BASE_URL": "http://127.0.0.1:18060/"},
            clear=False,
        ), patch("app.mcp_gateway.content_source_client.httpx.post", side_effect=fake_post):
            get_settings.cache_clear()
            try:
                app = self._app()

                async def call_api():
                    transport = ASGITransport(app=app)
                    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                        session_response = await client.post(
                            "/api/v1/agent/sessions",
                            json={"title": "xhs search", "primary_intent": "agent_chat"},
                        )
                        session_id = session_response.json()["id"]
                        posted = await client.post(
                            f"/api/v1/agent/sessions/{session_id}/messages",
                            json={"content_text": "\u8bf7\u5728\u5c0f\u7ea2\u4e66\u641c\u7d22 2027 \u79cb\u62db Java \u5c97\u4f4d"},
                        )
                        messages = await client.get(f"/api/v1/agent/sessions/{session_id}/messages")
                        return posted, messages

                posted_response, messages_response = run(call_api())
            finally:
                get_settings.cache_clear()

        self.assertEqual(201, posted_response.status_code)
        self.assertEqual(
            [
                {
                    "url": "http://127.0.0.1:18060/api/v1/feeds/search",
                    "json": {"keyword": "\u8bf7\u5728\u5c0f\u7ea2\u4e66\u641c\u7d22 2027 \u79cb\u62db Java \u5c97\u4f4d", "filters": None},
                    "headers": {},
                    "timeout": 30.0,
                }
            ],
            rest_calls,
        )
        messages = messages_response.json()["items"]
        self.assertEqual(["assistant", "tool_call", "tool_result", "user"], sorted(message["role"] for message in messages))
        tool_result = next(message for message in messages if message["role"] == "tool_result")
        self.assertTrue(tool_result["content_json"]["result"]["ok"])
        self.assertEqual("xiaohongshu_rest", tool_result["content_json"]["result"]["metadata"]["adapter"])

    def test_stream_agent_message_emits_approval_required_for_skill_ask_tool(self):
        self._create_approval_memory_skill()
        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                session_response = await client.post(
                    "/api/v1/agent/sessions",
                    json={"title": "approval stream", "primary_intent": "agent_chat"},
                )
                session_id = session_response.json()["id"]
                stream_response = await client.post(
                    f"/api/v1/agent/sessions/{session_id}/messages/stream",
                    json={
                        "content_text": "approval memory",
                        "requested_tool_name": "memory_search",
                        "source_type": "agent_chat",
                    },
                )
                messages_response = await client.get(f"/api/v1/agent/sessions/{session_id}/messages")
                return stream_response, messages_response

        stream_response, messages_response = run(call_api())

        self.assertEqual(200, stream_response.status_code)
        self.assertIn("event: user_message", stream_response.text)
        self.assertIn("event: approval_required", stream_response.text)
        self.assertIn('"tool_name":"memory_search"', stream_response.text)
        self.assertIn('"permission_decision":"ask"', stream_response.text)
        self.assertNotIn("event: error", stream_response.text)

        messages = messages_response.json()["items"]
        self.assertEqual(["user"], [message["role"] for message in messages])

    def test_approve_agent_approval_executes_waiting_tool_and_persists_assistant_reply(self):
        self._create_approval_memory_skill()
        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                session_response = await client.post(
                    "/api/v1/agent/sessions",
                    json={"title": "approval approve", "primary_intent": "agent_chat"},
                )
                session_id = session_response.json()["id"]
                stream_response = await client.post(
                    f"/api/v1/agent/sessions/{session_id}/messages/stream",
                    json={
                        "content_text": "approval memory",
                        "requested_tool_name": "memory_search",
                        "source_type": "agent_chat",
                    },
                )
                approval_payload = _sse_payload(stream_response.text, "approval_required")
                approve_response = await client.post(
                    f"/api/v1/agent/approvals/{approval_payload['approval']['id']}/approve",
                    json={"decision_reason": "user approved skill ask tool"},
                )
                messages_response = await client.get(f"/api/v1/agent/sessions/{session_id}/messages")
                return approve_response, messages_response

        approve_response, messages_response = run(call_api())

        self.assertEqual(200, approve_response.status_code)
        payload = approve_response.json()
        self.assertEqual("approved", payload["approval"]["status"])
        self.assertEqual("assistant", payload["assistant_message"]["role"])
        self.assertEqual("final_response", payload["context_metadata"]["current_step"])

        roles = [message["role"] for message in messages_response.json()["items"]]
        self.assertEqual(["user", "tool_call", "tool_result", "assistant"], roles)

    def _create_approval_memory_skill(self) -> None:
        from app.agent_runtime.memory.skill_repository import AgentSkillRepository
        from app.domains.agent_memory.repository import AgentMemoryRepository
        from app.domains.agent_memory.schemas import AgentSkillCreate

        with self.Session() as session:
            AgentSkillRepository(AgentMemoryRepository(session)).create_skill(
                AgentSkillCreate(
                    name="approval-memory-search",
                    title="Approval Memory Search",
                    description="Use this skill when the user asks for approval memory context.",
                    category="agent_guardrail",
                    metadata_json={"allowed_tools": ["memory_search"], "ask_tools": ["memory_search"]},
                    sections={"workflow": "Ask before memory_search for approval memory."},
                )
            )
            session.commit()


def _sse_payload(stream_text: str, event_name: str) -> dict:
    import json

    for raw_event in stream_text.split("\n\n"):
        if f"event: {event_name}" not in raw_event:
            continue
        for line in raw_event.splitlines():
            if line.startswith("data: "):
                return json.loads(line.removeprefix("data: "))
            if line.startswith("data:"):
                return json.loads(line.removeprefix("data:"))
    raise AssertionError(f"SSE event not found: {event_name}")


if __name__ == "__main__":
    unittest.main()
