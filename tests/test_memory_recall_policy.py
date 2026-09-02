import sys
import unittest
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class MemoryRecallPolicyTest(unittest.TestCase):
    def setUp(self):
        from app.db.base import Base
        import app.domains.agent_memory.models  # noqa: F401

        self.engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

    def tearDown(self):
        self.engine.dispose()

    def test_recalls_relevant_active_memory_but_not_archived_memory(self):
        from app.agent_runtime.memory.recall_policy import recall_relevant_memories
        from app.domains.agent_memory.models import AgentMemory, AgentMemoryStatus

        with self.Session() as session:
            relevant = AgentMemory(
                id="memory-relevant",
                memory_type="user_preference",
                scope="application_submission",
                title="投递前必须用户确认",
                content="任何岗位最终提交前都必须等待用户确认。",
                status=AgentMemoryStatus.ACTIVE,
                importance=95,
            )
            archived = AgentMemory(
                id="memory-archived",
                memory_type="user_preference",
                scope="application_submission",
                title="旧投递偏好",
                content="旧规则不再使用。",
                status=AgentMemoryStatus.ARCHIVED,
                importance=100,
            )
            unrelated = AgentMemory(
                id="memory-unrelated",
                memory_type="tool_recovery",
                scope="content_fetcher",
                title="文章抓取恢复经验",
                content="正文为空时请求可见文章内容。",
                status=AgentMemoryStatus.ACTIVE,
                importance=90,
            )
            session.add_all([relevant, archived, unrelated])
            session.commit()

            result = recall_relevant_memories(
                session,
                query="帮我投递腾讯的 Java 岗位",
                limit=3,
                max_chars=1000,
            )

        self.assertEqual(["memory-relevant"], [item.memory_id for item in result.items])
        self.assertNotIn("memory-archived", [item.memory_id for item in result.items])

    def test_recall_respects_memory_count_and_character_budget(self):
        from app.agent_runtime.memory.recall_policy import recall_relevant_memories
        from app.domains.agent_memory.models import AgentMemory, AgentMemoryStatus

        with self.Session() as session:
            session.add_all(
                [
                    AgentMemory(
                        id=f"memory-{index}",
                        memory_type="user_preference",
                        scope="job_discovery",
                        title=f"岗位偏好 {index}",
                        content="Java 后端和 Agent 岗位优先。" * 10,
                        status=AgentMemoryStatus.ACTIVE,
                        importance=100 - index,
                    )
                    for index in range(4)
                ]
            )
            session.commit()

            result = recall_relevant_memories(
                session,
                query="Java 后端 Agent 岗位",
                limit=2,
                max_chars=120,
            )

        self.assertLessEqual(len(result.items), 2)
        self.assertLessEqual(len(result.rendered_context), 120)
        self.assertTrue(result.truncated)


if __name__ == "__main__":
    unittest.main()
