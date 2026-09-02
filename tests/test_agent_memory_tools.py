import sys
import unittest
from asyncio import run
from pathlib import Path
import shutil

from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class AgentMemoryToolsTest(unittest.TestCase):
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
        self.skill_root = PROJECT_ROOT / ".tmp-agent-memory-tests" / self._testMethodName / "docs" / "agent-skills"
        shutil.rmtree(self.skill_root.parent.parent, ignore_errors=True)
        self.skill_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.engine.dispose()
        shutil.rmtree(PROJECT_ROOT / ".tmp-agent-memory-tests", ignore_errors=True)

    def _skill_repository(self, session):
        from app.agent_runtime.memory.skill_repository import AgentSkillRepository
        from app.domains.agent_memory.repository import AgentMemoryRepository

        return AgentSkillRepository(AgentMemoryRepository(session), skill_root=self.skill_root)

    def _skill_create(self, name: str = "dlmu-campus-sync"):
        from app.domains.agent_memory.schemas import AgentSkillCreate

        return AgentSkillCreate(
            name=name,
            title="DLMU campus recruiting sync",
            description="大连海事就业网 campus 页面招聘公告同步经验。",
            category="job_source_sync",
            sections={
                "when_to_use": "同步大连海事就业网招聘公告时使用。",
                "workflow": "读取 campus 列表页，提取公告链接，再解析公告正文。",
            },
        )

    def _app(self):
        from app.db.session import get_db_session
        from app.main import create_app

        app = create_app()

        def override_session():
            with self.Session() as session:
                yield session

        app.dependency_overrides[get_db_session] = override_session
        return app

    def test_memory_search_returns_structured_empty_result_when_memory_tables_are_empty(self):
        from app.agent_runtime.memory.memory_tools import memory_search

        with self.Session() as session:
            result = memory_search(session, query="Java 后端偏好", limit=10)

        self.assertEqual("memories", result.corpus)
        self.assertEqual(["agent_memories", "agent_skills"], result.searched_tables)
        self.assertEqual(["agent_memories", "agent_skills"], result.available_tables)
        self.assertEqual([], result.items)
        self.assertEqual("memory search found no matches", result.reason)

    def test_memory_search_does_not_fall_back_to_session_transcript(self):
        from app.domains.conversations.models import AgentMessageRole
        from app.domains.conversations.repository import ConversationRepository
        from app.domains.conversations.schemas import AgentMessageCreate
        from app.domains.conversations.service import ConversationService
        from app.agent_runtime.memory.memory_tools import memory_search, sessions_search

        with self.Session() as session:
            service = ConversationService(ConversationRepository(session))
            conversation = service.create_session(title="偏好", primary_intent="agent_chat")
            service.append_message(
                conversation.id,
                AgentMessageCreate(
                    role=AgentMessageRole.USER,
                    content_text="我的长期偏好是 Java 后端岗位",
                    visible_content_text="我的长期偏好是 Java 后端岗位",
                    token_estimate=10,
                ),
            )
            session.commit()

            session_result = sessions_search(session, query="Java 后端", limit=10)
            memory_result = memory_search(session, query="Java 后端", limit=10)

        self.assertEqual(1, len(session_result.items))
        self.assertEqual([], memory_result.items)

    def test_memory_get_returns_not_found_when_memory_tables_are_empty(self):
        from app.agent_runtime.memory.memory_tools import memory_get

        with self.Session() as session:
            result = memory_get(session, memory_id="missing-memory")

        self.assertFalse(result.found)
        self.assertEqual("missing-memory", result.memory_id)
        self.assertEqual("memory not found", result.reason)
        self.assertIsNone(result.content)

    def test_memory_search_and_get_returns_skill_memory(self):
        from app.agent_runtime.memory.memory_tools import memory_get, memory_search

        with self.Session() as session:
            skill = self._skill_repository(session).create_skill(self._skill_create())
            session.commit()

            search_result = memory_search(session, query="DLMU campus", limit=10)
            read_result = memory_get(session, memory_id=skill.id)

        self.assertEqual(1, len(search_result.items))
        self.assertEqual(skill.id, search_result.items[0].memory_id)
        self.assertEqual("skill", search_result.items[0].source_type)
        self.assertIn("DLMU campus", search_result.items[0].excerpt)
        self.assertTrue(read_result.found)
        self.assertEqual("skill", read_result.source_type)
        self.assertIn("# DLMU campus recruiting sync", read_result.content)

    def test_memory_api_returns_structured_empty_results(self):
        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                search_response = await client.get(
                    "/api/v1/agent-memory/search",
                    params={"query": "Java 后端偏好", "limit": 10},
                )
                get_response = await client.get("/api/v1/agent-memory/missing-memory")
                return search_response, get_response

        search_response, get_response = run(call_api())

        self.assertEqual(200, search_response.status_code)
        self.assertEqual(200, get_response.status_code)
        self.assertEqual([], search_response.json()["items"])
        self.assertEqual(["agent_memories", "agent_skills"], search_response.json()["searched_tables"])
        self.assertFalse(get_response.json()["found"])


if __name__ == "__main__":
    unittest.main()
