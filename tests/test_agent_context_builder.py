import sys
import unittest
import shutil
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class AgentContextBuilderTest(unittest.TestCase):
    def setUp(self):
        from app.db.base import Base
        from app.domains.agent_memory import models as agent_memory_models  # noqa: F401
        from app.domains.automation import models as automation_models  # noqa: F401
        from app.domains.conversations import models as conversation_models  # noqa: F401

        self.skill_root = PROJECT_ROOT / ".tmp-test-artifacts" / "agent-context-builder" / self._testMethodName
        shutil.rmtree(self.skill_root, ignore_errors=True)
        self.skill_root.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

    def tearDown(self):
        self.engine.dispose()
        shutil.rmtree(self.skill_root, ignore_errors=True)

    def _service(self, session):
        from app.domains.conversations.repository import ConversationRepository
        from app.domains.conversations.service import ConversationService

        return ConversationService(ConversationRepository(session))

    def _skill_repository(self, session):
        from app.agent_runtime.memory.skill_repository import AgentSkillRepository
        from app.domains.agent_memory.repository import AgentMemoryRepository

        return AgentSkillRepository(AgentMemoryRepository(session), skill_root=self.skill_root)

    def test_build_without_summary_loads_recent_messages(self):
        from app.agent_runtime.memory.context_builder import ContextBuildConfig, MemoryContextBuilder
        from app.domains.conversations.models import AgentMessageRole
        from app.domains.conversations.schemas import AgentMessageCreate

        with self.Session() as session:
            service = self._service(session)
            conversation = service.create_session(title="上下文", primary_intent="agent_chat")
            user_message = service.append_message(
                conversation.id,
                AgentMessageCreate(
                    role=AgentMessageRole.USER,
                    content_text="帮我找 Java 后端岗位",
                    visible_content_text="帮我找 Java 后端岗位",
                    token_estimate=12,
                ),
            )
            assistant_message = service.append_message(
                conversation.id,
                AgentMessageCreate(
                    role=AgentMessageRole.ASSISTANT,
                    content_text="我会先查询岗位线索",
                    visible_content_text="我会先查询岗位线索",
                    token_estimate=10,
                ),
            )
            session.commit()

            built = MemoryContextBuilder(service).build(
                conversation.id,
                new_user_message=None,
                config=ContextBuildConfig(max_recent_messages=10),
            )

        self.assertEqual([user_message.id, assistant_message.id], built.loaded_session_history_ids)
        self.assertEqual([], built.loaded_memory_ids)
        self.assertEqual([], built.loaded_skill_ids)
        self.assertEqual(["user", "assistant"], [message["role"] for message in built.llm_messages])
        self.assertEqual("帮我找 Java 后端岗位", built.llm_messages[0]["content"])
        self.assertIsNone(built.context_metadata["summary_id"])
        self.assertFalse(built.need_compaction)

    def test_build_with_latest_summary_adds_summary_context_block(self):
        from app.agent_runtime.memory.context_builder import ContextBuildConfig, MemoryContextBuilder
        from app.domains.conversations.models import AgentMessageRole
        from app.domains.conversations.schemas import AgentContextSummaryCreate, AgentMessageCreate

        with self.Session() as session:
            service = self._service(session)
            conversation = service.create_session(title="有摘要", primary_intent="agent_chat")
            old_message = service.append_message(
                conversation.id,
                AgentMessageCreate(
                    role=AgentMessageRole.USER,
                    content_text="旧消息：我只找国内秋招岗位",
                    visible_content_text="旧消息：我只找国内秋招岗位",
                    token_estimate=200,
                ),
            )
            recent_message = service.append_message(
                conversation.id,
                AgentMessageCreate(
                    role=AgentMessageRole.ASSISTANT,
                    content_text="最近回复：会保留用户确认边界",
                    visible_content_text="最近回复：会保留用户确认边界",
                    token_estimate=20,
                ),
            )
            summary = service.create_context_summary(
                conversation.id,
                AgentContextSummaryCreate(
                    summary_text="Goal: 帮用户找 2027 届 Java 后端秋招岗位。\nNext Steps: 查询岗位线索。",
                    covered_message_start_id=old_message.id,
                    covered_message_end_id=old_message.id,
                    first_kept_message_id=recent_message.id,
                    token_estimate=30,
                    created_by="test",
                ),
            )
            service.mark_messages_compacted(conversation.id, [old_message.id], summary.id)
            session.commit()

            built = MemoryContextBuilder(service).build(
                conversation.id,
                new_user_message="继续",
                config=ContextBuildConfig(max_recent_messages=10),
            )

        self.assertEqual(summary.id, built.context_metadata["summary_id"])
        self.assertEqual([recent_message.id], built.loaded_session_history_ids)
        self.assertEqual("system", built.llm_messages[0]["role"])
        self.assertIn("Latest conversation summary", built.llm_messages[0]["content"])
        self.assertIn("Goal: 帮用户找 2027 届 Java 后端秋招岗位", built.llm_messages[0]["content"])
        self.assertEqual("assistant", built.llm_messages[1]["role"])
        self.assertEqual("user", built.llm_messages[-1]["role"])
        self.assertEqual("继续", built.llm_messages[-1]["content"])

    def test_compacted_old_messages_do_not_enter_llm_messages(self):
        from app.agent_runtime.memory.context_builder import ContextBuildConfig, MemoryContextBuilder
        from app.domains.conversations.models import AgentMessageRole
        from app.domains.conversations.schemas import AgentContextSummaryCreate, AgentMessageCreate

        with self.Session() as session:
            service = self._service(session)
            conversation = service.create_session(title="不加载旧消息", primary_intent="agent_chat")
            old_message = service.append_message(
                conversation.id,
                AgentMessageCreate(
                    role=AgentMessageRole.USER,
                    content_text="旧消息不能进入模型原文",
                    visible_content_text="旧消息不能进入模型原文",
                    token_estimate=100,
                ),
            )
            recent_message = service.append_message(
                conversation.id,
                AgentMessageCreate(
                    role=AgentMessageRole.USER,
                    content_text="最近消息可以进入模型原文",
                    visible_content_text="最近消息可以进入模型原文",
                    token_estimate=10,
                ),
            )
            summary = service.create_context_summary(
                conversation.id,
                AgentContextSummaryCreate(
                    summary_text="Critical Context: 旧消息已压缩。",
                    covered_message_start_id=old_message.id,
                    covered_message_end_id=old_message.id,
                    first_kept_message_id=recent_message.id,
                    token_estimate=10,
                    created_by="test",
                ),
            )
            service.mark_messages_compacted(conversation.id, [old_message.id], summary.id)
            session.commit()

            built = MemoryContextBuilder(service).build(
                conversation.id,
                new_user_message=None,
                config=ContextBuildConfig(max_recent_messages=10),
            )

        contents = "\n".join(message["content"] for message in built.llm_messages)
        self.assertNotIn("旧消息不能进入模型原文", contents)
        self.assertIn("最近消息可以进入模型原文", contents)
        self.assertEqual([recent_message.id], built.loaded_session_history_ids)

    def test_build_marks_need_compaction_when_context_exceeds_threshold(self):
        from app.agent_runtime.memory.context_builder import ContextBuildConfig, MemoryContextBuilder
        from app.domains.conversations.models import AgentMessageRole
        from app.domains.conversations.schemas import AgentMessageCreate

        with self.Session() as session:
            service = self._service(session)
            conversation = service.create_session(title="超预算", primary_intent="agent_chat")
            message = service.append_message(
                conversation.id,
                AgentMessageCreate(
                    role=AgentMessageRole.USER,
                    content_text="很长的消息",
                    visible_content_text="很长的消息",
                    token_estimate=900,
                ),
            )
            session.commit()

            built = MemoryContextBuilder(service).build(
                conversation.id,
                new_user_message="新的长消息",
                config=ContextBuildConfig(context_window=1000, reserve_tokens=100, max_recent_messages=10),
            )

        self.assertTrue(built.need_compaction)
        self.assertEqual([message.id], built.loaded_session_history_ids)
        self.assertGreater(built.token_estimate, built.context_metadata["threshold_tokens"])

    def test_build_loads_matching_active_skill_into_runtime_context(self):
        from app.agent_runtime.memory.context_builder import ContextBuildConfig, MemoryContextBuilder
        from app.domains.agent_memory.schemas import AgentSkillCreate

        with self.Session() as session:
            service = self._service(session)
            skill_repository = self._skill_repository(session)
            conversation = service.create_session(title="agent chat", primary_intent="agent_chat")
            skill = skill_repository.create_skill(
                AgentSkillCreate(
                    name="java-job-discovery",
                    title="Java Job Discovery",
                    description="Use this skill when the user asks for Java campus recruiting leads.",
                    category="job_discovery",
                    sections={
                        "when_to_use": "Use for Java campus recruiting lead discovery.",
                        "workflow": "Search sources, import URLs, verify before application.",
                    },
                )
            )
            session.commit()

            built = MemoryContextBuilder(service, skill_repository=skill_repository).build(
                conversation.id,
                new_user_message="Java",
                config=ContextBuildConfig(max_recent_messages=10),
            )

        self.assertEqual([skill.id], built.loaded_skill_ids)
        self.assertEqual([skill.id], built.context_metadata["loaded_skill_ids"])
        skill_context_messages = [
            message
            for message in built.llm_messages
            if message.get("metadata", {}).get("source") == "agent_skill"
        ]
        self.assertEqual(1, len(skill_context_messages))
        self.assertEqual(skill.id, skill_context_messages[0]["metadata"]["skill_id"])
        self.assertIn("Java Job Discovery", skill_context_messages[0]["content"])
        self.assertIn("Search sources, import URLs, verify before application.", skill_context_messages[0]["content"])

    def test_build_creates_tool_permission_snapshot_from_loaded_skills(self):
        from app.agent_runtime.memory.context_builder import ContextBuildConfig, MemoryContextBuilder
        from app.domains.agent_memory.schemas import AgentSkillCreate

        with self.Session() as session:
            service = self._service(session)
            skill_repository = self._skill_repository(session)
            conversation = service.create_session(title="agent chat", primary_intent="agent_chat")
            skill = skill_repository.create_skill(
                AgentSkillCreate(
                    name="wechat-skill-permissions",
                    title="WeChat Skill Permissions",
                    description="Use this skill when the user asks to parse WeChat recruiting articles.",
                    category="content_source",
                    metadata_json={
                        "allowed_tools": ["weixin-articles-mcp.read_article", "ocr.extract_text"],
                        "ask_tools": ["browser.open"],
                        "disallowed_tools": ["submit_application"],
                    },
                    sections={"workflow": "Use weixin-articles-mcp.read_article."},
                )
            )
            session.commit()

            built = MemoryContextBuilder(service, skill_repository=skill_repository).build(
                conversation.id,
                new_user_message="WeChat recruiting article",
                config=ContextBuildConfig(max_recent_messages=10),
            )

        snapshot = built.context_metadata["skill_tool_permission_policy"]
        self.assertEqual([skill.id], snapshot["skill_ids"])
        self.assertEqual(["weixin-articles-mcp.read_article", "ocr.extract_text"], snapshot["allowed_tools"])
        self.assertEqual(["browser.open"], snapshot["ask_tools"])
        self.assertEqual(["submit_application"], snapshot["disallowed_tools"])
        self.assertEqual("loaded_skill_snapshot", snapshot["policy_source"])

    def test_build_merges_permission_snapshot_from_multiple_loaded_skills(self):
        from app.agent_runtime.memory.context_builder import ContextBuildConfig, MemoryContextBuilder
        from app.domains.agent_memory.schemas import AgentSkillCreate

        with self.Session() as session:
            service = self._service(session)
            skill_repository = self._skill_repository(session)
            conversation = service.create_session(title="agent chat", primary_intent="agent_chat")
            first_skill = skill_repository.create_skill(
                AgentSkillCreate(
                    name="wechat-fetch-skill",
                    title="WeChat Fetch Skill",
                    description="Use this skill when the user asks for WeChat article parsing.",
                    category="content_source",
                    metadata_json={
                        "allowed_tools": ["weixin-articles-mcp.read_article"],
                        "ask_tools": ["browser.open"],
                    },
                    sections={"workflow": "WeChat article parsing."},
                )
            )
            second_skill = skill_repository.create_skill(
                AgentSkillCreate(
                    name="wechat-ocr-skill",
                    title="WeChat OCR Skill",
                    description="Use this skill when the user asks for WeChat image OCR.",
                    category="content_source",
                    metadata_json={
                        "allowed_tools": ["ocr.extract_text", "weixin-articles-mcp.read_article"],
                        "disallowed_tools": ["submit_application"],
                    },
                    sections={"workflow": "WeChat image OCR."},
                )
            )
            session.commit()

            built = MemoryContextBuilder(service, skill_repository=skill_repository).build(
                conversation.id,
                new_user_message="WeChat article OCR",
                config=ContextBuildConfig(max_recent_messages=10, max_loaded_skills=3),
        )

        snapshot = built.context_metadata["skill_tool_permission_policy"]
        self.assertEqual(built.loaded_skill_ids, snapshot["skill_ids"])
        self.assertCountEqual(["weixin-articles-mcp.read_article", "ocr.extract_text"], snapshot["allowed_tools"])
        self.assertEqual(["browser.open"], snapshot["ask_tools"])
        self.assertEqual(["submit_application"], snapshot["disallowed_tools"])

    def test_build_does_not_load_archived_skill_into_runtime_context(self):
        from app.agent_runtime.memory.context_builder import ContextBuildConfig, MemoryContextBuilder
        from app.domains.agent_memory.schemas import AgentSkillCreate

        with self.Session() as session:
            service = self._service(session)
            skill_repository = self._skill_repository(session)
            conversation = service.create_session(title="agent chat", primary_intent="agent_chat")
            skill = skill_repository.create_skill(
                AgentSkillCreate(
                    name="archived-java-job-discovery",
                    title="Archived Java Job Discovery",
                    description="Use this skill when the user asks for Java campus recruiting leads.",
                    category="job_discovery",
                )
            )
            skill_repository.archive_skill(skill.id)
            session.commit()

            built = MemoryContextBuilder(service, skill_repository=skill_repository).build(
                conversation.id,
                new_user_message="Java",
                config=ContextBuildConfig(max_recent_messages=10),
            )

        self.assertEqual([], built.loaded_skill_ids)
        self.assertEqual([], built.context_metadata["loaded_skill_ids"])
        self.assertFalse(
            any(message.get("metadata", {}).get("source") == "agent_skill" for message in built.llm_messages)
        )


if __name__ == "__main__":
    unittest.main()
