import shutil
import sys
import unittest
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class SkillProgressiveLoadingTest(unittest.TestCase):
    def setUp(self) -> None:
        from app.db.base import Base
        import app.domains.agent_memory.models  # noqa: F401
        import app.domains.automation.models  # noqa: F401
        import app.domains.conversations.models  # noqa: F401

        self.skill_root = PROJECT_ROOT / ".tmp-test-artifacts" / "skill-progressive-loading" / self._testMethodName
        shutil.rmtree(self.skill_root, ignore_errors=True)
        self.skill_root.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

    def tearDown(self) -> None:
        self.engine.dispose()
        shutil.rmtree(PROJECT_ROOT / ".tmp-test-artifacts" / "skill-progressive-loading", ignore_errors=True)

    def _service(self, session):
        from app.domains.conversations.repository import ConversationRepository
        from app.domains.conversations.service import ConversationService

        return ConversationService(ConversationRepository(session))

    def _skill_repository(self, session):
        from app.agent_runtime.memory.skill_repository import AgentSkillRepository
        from app.domains.agent_memory.repository import AgentMemoryRepository

        return AgentSkillRepository(AgentMemoryRepository(session), skill_root=self.skill_root)

    def test_build_skill_summary_card_keeps_menu_level_information_without_body(self) -> None:
        from app.agent_runtime.memory.skill_summary_index import build_skill_summary_card
        from app.domains.agent_memory.schemas import AgentSkillCreate

        with self.Session() as session:
            repository = self._skill_repository(session)
            skill = repository.create_skill(
                AgentSkillCreate(
                    name="wechat-recruiting-import",
                    title="微信公众号招聘导入",
                    description="当用户提供微信公众号文章并希望抽取秋招公司和岗位线索时使用，不自动投递。",
                    category="content_source",
                    protected=True,
                    pinned=True,
                    metadata_json={
                        "source_types": ["wechat_article"],
                        "allowed_tools": ["weixin-articles-mcp.read_article", "ocr.extract_text"],
                        "ask_tools": ["browser.open"],
                        "disallowed_tools": ["submit_application"],
                        "security_risk_level": "medium",
                        "auto_trigger_state": "enabled",
                        "description_quality_score": 9,
                    },
                    sections={
                        "when_to_use": "用户给出公众号文章链接，要求读取正文并抽取公司、岗位、校招信息时使用。",
                        "workflow": "先读取文章，再抽取招聘信号，最后给出候选公司和来源。",
                    },
                )
            )
            document = repository.read_skill(skill.id)
            session.commit()

        card = build_skill_summary_card(document)

        self.assertEqual(skill.id, card.skill_id)
        self.assertEqual("wechat-recruiting-import", card.name)
        self.assertEqual("微信公众号招聘导入", card.title)
        self.assertEqual("content_source", card.category)
        self.assertEqual(("wechat_article",), card.source_types)
        self.assertEqual(("weixin-articles-mcp.read_article", "ocr.extract_text"), card.allowed_tools)
        self.assertEqual(("browser.open",), card.ask_tools)
        self.assertEqual(("submit_application",), card.disallowed_tools)
        self.assertEqual("medium", card.risk_level)
        self.assertTrue(card.pinned)
        self.assertTrue(card.protected)
        self.assertTrue(card.auto_load_enabled)
        self.assertEqual(9, card.description_quality_score)
        self.assertIn("公众号文章链接", card.when_to_use)
        self.assertNotIn("先读取文章，再抽取招聘信号", card.summary_text)
        self.assertIn("微信公众号招聘导入", card.summary_text)
        self.assertIn("适用场景", card.summary_text)
        self.assertIn("工具边界", card.summary_text)

    def test_select_skill_candidates_ranks_matching_skill_cards_with_reasons(self) -> None:
        from app.agent_runtime.memory.skill_candidate_selector import select_skill_candidates
        from app.agent_runtime.memory.skill_summary_index import build_skill_summary_card
        from app.domains.agent_memory.schemas import AgentSkillCreate

        with self.Session() as session:
            repository = self._skill_repository(session)
            wechat_skill = repository.create_skill(
                AgentSkillCreate(
                    name="wechat-recruiting-import",
                    title="微信公众号招聘导入",
                    description="读取微信公众号文章，抽取秋招公司和岗位线索。",
                    category="content_source",
                    metadata_json={
                        "source_types": ["wechat_article"],
                        "allowed_tools": ["weixin-articles-mcp.read_article"],
                        "auto_trigger_state": "enabled",
                    },
                    sections={
                        "when_to_use": "用户提供公众号文章链接，想提取公司、岗位、校招信息时使用。",
                    },
                )
            )
            resume_skill = repository.create_skill(
                AgentSkillCreate(
                    name="resume-tailoring",
                    title="简历优化",
                    description="根据用户简历和目标 JD 改写简历内容。",
                    category="resume",
                    metadata_json={
                        "source_types": ["resume_text", "job_description"],
                        "allowed_tools": ["resume.tailor"],
                        "auto_trigger_state": "enabled",
                    },
                    sections={
                        "when_to_use": "用户提供简历文本和目标岗位 JD，希望润色或匹配岗位时使用。",
                    },
                )
            )
            cards = [
                build_skill_summary_card(repository.read_skill(resume_skill.id)),
                build_skill_summary_card(repository.read_skill(wechat_skill.id)),
            ]
            session.commit()

        candidates = select_skill_candidates("帮我读取这篇公众号文章，提取里面的公司和岗位线索", cards, limit=2)

        self.assertEqual(2, len(candidates))
        self.assertEqual(wechat_skill.id, candidates[0].card.skill_id)
        self.assertGreater(candidates[0].score, candidates[1].score)
        self.assertIn("公众号", candidates[0].matched_terms)
        self.assertIn("文章", candidates[0].matched_terms)
        self.assertIn("命中", candidates[0].reason)
        self.assertEqual(resume_skill.id, candidates[1].card.skill_id)

    def test_select_skill_candidates_uses_source_type_aliases_for_chinese_queries(self) -> None:
        from app.agent_runtime.memory.skill_candidate_selector import select_skill_candidates
        from app.agent_runtime.memory.skill_summary_index import build_skill_summary_card
        from app.domains.agent_memory.schemas import AgentSkillCreate

        with self.Session() as session:
            repository = self._skill_repository(session)
            wechat_skill = repository.create_skill(
                AgentSkillCreate(
                    name="wechat-article-content-fetch",
                    title="WeChat Article Content Fetch",
                    description="读取微信公众号文章，抽取秋招公司和岗位线索。",
                    category="content_source",
                    metadata_json={
                        "source_types": ["wechat_article"],
                        "allowed_tools": ["weixin-articles-mcp.read_article"],
                    },
                    sections={"when_to_use": "用户提供公众号文章链接，想提取秋招信息时使用。"},
                )
            )
            xiaohongshu_skill = repository.create_skill(
                AgentSkillCreate(
                    name="xiaohongshu-content-fetch",
                    title="Xiaohongshu Content Fetch",
                    description="Use this skill when the user asks to search Xiaohongshu recruiting notes.",
                    category="content_source",
                    metadata_json={
                        "source_types": ["agent_chat", "xiaohongshu_note"],
                        "allowed_tools": ["xiaohongshu-mcp.search_feeds"],
                    },
                    sections={"workflow": "Search Xiaohongshu recruiting notes."},
                )
            )
            cards = [
                build_skill_summary_card(repository.read_skill(wechat_skill.id)),
                build_skill_summary_card(repository.read_skill(xiaohongshu_skill.id)),
            ]
            session.commit()

        candidates = select_skill_candidates("请在小红书搜索 2027 秋招 Java 岗位", cards, limit=2)

        self.assertEqual(xiaohongshu_skill.id, candidates[0].card.skill_id)
        self.assertIn("小红书", candidates[0].matched_terms)
        self.assertEqual(wechat_skill.id, candidates[1].card.skill_id)

    def test_context_builder_records_skill_candidate_menu_before_loading_body(self) -> None:
        from app.agent_runtime.memory.context_builder import ContextBuildConfig, MemoryContextBuilder
        from app.domains.agent_memory.schemas import AgentSkillCreate

        with self.Session() as session:
            service = self._service(session)
            repository = self._skill_repository(session)
            conversation = service.create_session(title="agent chat", primary_intent="agent_chat")
            resume_skill = repository.create_skill(
                AgentSkillCreate(
                    name="resume-tailoring",
                    title="简历优化",
                    description="根据用户简历和目标 JD 改写简历内容。",
                    category="resume",
                    metadata_json={"source_types": ["resume_text"], "auto_trigger_state": "enabled"},
                    sections={"when_to_use": "用户希望优化简历或匹配目标岗位时使用。"},
                )
            )
            wechat_skill = repository.create_skill(
                AgentSkillCreate(
                    name="wechat-recruiting-import",
                    title="微信公众号招聘导入",
                    description="读取微信公众号文章，抽取秋招公司和岗位线索。",
                    category="content_source",
                    metadata_json={"source_types": ["wechat_article"], "auto_trigger_state": "enabled"},
                    sections={"when_to_use": "用户提供公众号文章链接，想提取公司、岗位、校招信息时使用。"},
                )
            )
            session.commit()

            built = MemoryContextBuilder(service, skill_repository=repository).build(
                conversation.id,
                new_user_message="帮我读取这篇公众号文章，提取里面的公司和岗位线索",
                config=ContextBuildConfig(max_recent_messages=10, max_loaded_skills=2),
            )

        selection = built.context_metadata["skill_candidate_selection"]

        self.assertEqual(wechat_skill.id, selection["candidates"][0]["skill_id"])
        self.assertIn("公众号", selection["candidates"][0]["matched_terms"])
        self.assertIn("命中", selection["candidates"][0]["reason"])
        self.assertIn(resume_skill.id, [candidate["skill_id"] for candidate in selection["candidates"]])
        self.assertEqual([wechat_skill.id, resume_skill.id], built.loaded_skill_ids)

    def test_context_builder_keeps_candidate_menu_larger_than_loaded_skill_bodies_with_trace(self) -> None:
        from app.agent_runtime.memory.context_builder import ContextBuildConfig, MemoryContextBuilder
        from app.domains.agent_memory.schemas import AgentSkillCreate

        with self.Session() as session:
            service = self._service(session)
            repository = self._skill_repository(session)
            conversation = service.create_session(title="agent chat", primary_intent="agent_chat")
            wechat_skill = repository.create_skill(
                AgentSkillCreate(
                    name="wechat-recruiting-import",
                    title="微信公众号招聘导入",
                    description="读取微信公众号文章，抽取秋招公司和岗位线索。",
                    category="content_source",
                    metadata_json={"source_types": ["wechat_article"], "auto_trigger_state": "enabled"},
                    sections={"when_to_use": "用户提供公众号文章链接，想提取公司、岗位、校招信息时使用。"},
                )
            )
            company_skill = repository.create_skill(
                AgentSkillCreate(
                    name="company-signal-analysis",
                    title="公司信号分析",
                    description="分析公司招聘线索和岗位线索，整理候选公司。",
                    category="company_analysis",
                    metadata_json={"source_types": ["company", "job_lead"], "auto_trigger_state": "enabled"},
                    sections={"when_to_use": "用户希望分析公司和岗位线索质量时使用。"},
                )
            )
            session.commit()

            built = MemoryContextBuilder(service, skill_repository=repository).build(
                conversation.id,
                new_user_message="帮我读取公众号文章，提取公司和岗位线索",
                config=ContextBuildConfig(max_recent_messages=10, max_skill_candidates=2, max_loaded_skills=1),
            )

        selection = built.context_metadata["skill_candidate_selection"]
        load_trace = built.context_metadata["skill_load_trace"]

        self.assertEqual([wechat_skill.id, company_skill.id], [candidate["skill_id"] for candidate in selection["candidates"]])
        self.assertEqual([wechat_skill.id], built.loaded_skill_ids)
        self.assertEqual(1, len(load_trace["loaded_skills"]))
        self.assertEqual(wechat_skill.id, load_trace["loaded_skills"][0]["skill_id"])
        self.assertEqual("body", load_trace["loaded_skills"][0]["load_layer"])
        self.assertEqual("skill_summary_candidate", load_trace["loaded_skills"][0]["selected_by"])
        self.assertIn("命中", load_trace["loaded_skills"][0]["reason"])
        self.assertGreater(load_trace["loaded_skills"][0]["token_estimate"], 0)
        self.assertRegex(load_trace["loaded_skills"][0]["version_hash"], r"^[0-9a-f]{64}$")
        self.assertEqual(2, load_trace["candidate_count"])
        self.assertEqual(1, load_trace["loaded_count"])
        skill_context_messages = [
            message
            for message in built.llm_messages
            if message.get("metadata", {}).get("source") == "agent_skill"
        ]
        self.assertEqual(1, len(skill_context_messages))
        self.assertEqual(wechat_skill.id, skill_context_messages[0]["metadata"]["skill_id"])

    def test_parse_skill_sections_loads_only_requested_headings(self) -> None:
        from app.agent_runtime.memory.skill_section_parser import parse_skill_sections, select_skill_sections

        content = """
---
name: resume-tailoring
---

# 简历优化

## 何时使用
用户希望根据目标 JD 优化简历时使用。

## 标准流程
先通读简历，再分析 JD，最后重写项目经历。

## 输出格式
输出一份可以直接替换的中文简历片段。

## 错误处理
如果缺少 JD，要说明缺少什么信息。
""".strip()

        sections = parse_skill_sections(content)
        selected = select_skill_sections(sections, ["输出格式", "错误处理"], max_chars=500)

        self.assertEqual(["何时使用", "标准流程", "输出格式", "错误处理"], [section.heading for section in sections])
        self.assertEqual(["输出格式", "错误处理"], [section.heading for section in selected.sections])
        self.assertIn("输出一份可以直接替换的中文简历片段", selected.content)
        self.assertIn("如果缺少 JD", selected.content)
        self.assertNotIn("先通读简历", selected.content)
        self.assertFalse(selected.truncated)
        self.assertGreater(selected.token_estimate, 0)

    def test_load_skill_resource_reads_reference_inside_skill_package_and_blocks_path_escape(self) -> None:
        from app.agent_runtime.memory.skill_resource_loader import load_skill_resource
        from app.domains.agent_memory.schemas import AgentSkillCreate

        with self.Session() as session:
            repository = self._skill_repository(session)
            skill = repository.create_skill(
                AgentSkillCreate(
                    name="resume-tailoring",
                    title="简历优化",
                    description="根据简历和目标 JD 做改写。",
                    category="resume",
                    sections={"when_to_use": "用户希望优化简历时使用。"},
                )
            )
            document = repository.read_skill(skill.id)
            session.commit()

        skill_dir = Path(document.skill.file_path).parent
        reference_dir = skill_dir / "references"
        reference_dir.mkdir(parents=True, exist_ok=True)
        (reference_dir / "format.md").write_text("这是简历输出格式参考。" * 50, encoding="utf-8")
        (skill_dir.parent / "secret.md").write_text("不应该被读取", encoding="utf-8")

        result = load_skill_resource(document, "references/format.md", max_chars=40)

        self.assertEqual("resource", result.load_layer)
        self.assertEqual("references/format.md", result.resource_path)
        self.assertIn("这是简历输出格式参考", result.content)
        self.assertEqual(40, result.content_chars)
        self.assertTrue(result.truncated)
        self.assertGreater(result.token_estimate, 0)

        with self.assertRaises(PermissionError):
            load_skill_resource(document, "../secret.md")

    def test_context_builder_loads_only_relevant_skill_sections_when_query_mentions_output_format(self) -> None:
        from app.agent_runtime.memory.context_builder import ContextBuildConfig, MemoryContextBuilder
        from app.domains.agent_memory.schemas import AgentSkillCreate

        with self.Session() as session:
            service = self._service(session)
            repository = self._skill_repository(session)
            conversation = service.create_session(title="agent chat", primary_intent="agent_chat")
            skill = repository.create_skill(
                AgentSkillCreate(
                    name="resume-tailoring-output-format",
                    title="简历优化",
                    description="根据用户简历和目标 JD 改写简历内容。",
                    category="resume",
                    metadata_json={"source_types": ["resume_text", "job_description"], "auto_trigger_state": "enabled"},
                    sections={
                        "when_to_use": "用户希望根据目标 JD 优化简历时使用。",
                        "workflow": "先完整分析简历，再逐段重写项目经历。这段流程不应该在本测试里加载。",
                        "outputs": "输出一份可以直接替换的中文简历片段，并说明修改点。",
                        "error_handling": "如果缺少目标 JD，要明确说明缺少什么信息。",
                    },
                )
            )
            session.commit()

            built = MemoryContextBuilder(service, skill_repository=repository).build(
                conversation.id,
                new_user_message="简历优化这个能力最终输出格式是什么？",
                config=ContextBuildConfig(max_recent_messages=10, max_loaded_skills=1),
            )

        skill_context_messages = [
            message
            for message in built.llm_messages
            if message.get("metadata", {}).get("source") == "agent_skill"
        ]
        self.assertEqual(1, len(skill_context_messages))
        skill_message = skill_context_messages[0]
        load_trace = built.context_metadata["skill_load_trace"]

        self.assertEqual(skill.id, skill_message["metadata"]["skill_id"])
        self.assertEqual("section", skill_message["metadata"]["load_layer"])
        self.assertIn("## 输出", skill_message["content"])
        self.assertIn("输出一份可以直接替换的中文简历片段", skill_message["content"])
        self.assertIn("## 错误处理", skill_message["content"])
        self.assertIn("如果缺少目标 JD", skill_message["content"])
        self.assertNotIn("先完整分析简历", skill_message["content"])
        self.assertEqual("section", load_trace["loaded_skills"][0]["load_layer"])
        self.assertEqual(["输出", "错误处理"], load_trace["loaded_skills"][0]["selected_sections"])

    def test_context_builder_loads_referenced_skill_resource_from_selected_sections(self) -> None:
        from app.agent_runtime.memory.context_builder import ContextBuildConfig, MemoryContextBuilder
        from app.domains.agent_memory.schemas import AgentSkillCreate

        with self.Session() as session:
            service = self._service(session)
            repository = self._skill_repository(session)
            conversation = service.create_session(title="agent chat", primary_intent="agent_chat")
            skill = repository.create_skill(
                AgentSkillCreate(
                    name="resume-tailoring-resource-format",
                    title="简历优化",
                    description="根据用户简历和目标 JD 改写简历内容。",
                    category="resume",
                    metadata_json={"source_types": ["resume_text", "job_description"], "auto_trigger_state": "enabled"},
                    sections={
                        "when_to_use": "用户希望根据目标 JD 优化简历时使用。",
                        "workflow": "完整流程参考 references/internal-workflow.md，这个流程参考不应因输出格式问题被加载。",
                        "outputs": "输出时请遵循 references/resume-output-format.md。",
                        "error_handling": "如果缺少目标 JD，要明确说明缺少什么信息。",
                    },
                )
            )
            session.commit()

            skill_dir = Path(skill.file_path).parent
            references_dir = skill_dir / "references"
            references_dir.mkdir(parents=True, exist_ok=True)
            (references_dir / "resume-output-format.md").write_text(
                "# 简历输出模板\n\n请输出：修改后简历、修改原因、JD 匹配点。",
                encoding="utf-8",
            )
            (references_dir / "internal-workflow.md").write_text(
                "这是一份流程参考，不应该出现在本次上下文里。",
                encoding="utf-8",
            )

            built = MemoryContextBuilder(service, skill_repository=repository).build(
                conversation.id,
                new_user_message="简历优化这个能力最终输出格式是什么？",
                config=ContextBuildConfig(max_recent_messages=10, max_loaded_skills=1),
            )

        resource_messages = [
            message
            for message in built.llm_messages
            if message.get("metadata", {}).get("source") == "agent_skill_resource"
        ]
        resource_trace = built.context_metadata["skill_resource_load_trace"]

        self.assertEqual(1, len(resource_messages))
        self.assertEqual(skill.id, resource_messages[0]["metadata"]["skill_id"])
        self.assertEqual("references/resume-output-format.md", resource_messages[0]["metadata"]["resource_path"])
        self.assertIn("请输出：修改后简历、修改原因、JD 匹配点", resource_messages[0]["content"])
        self.assertNotIn("流程参考", resource_messages[0]["content"])
        self.assertEqual(1, resource_trace["loaded_count"])
        self.assertEqual("references/resume-output-format.md", resource_trace["loaded_resources"][0]["resource_path"])
        self.assertEqual("resource", resource_trace["loaded_resources"][0]["load_layer"])


if __name__ == "__main__":
    unittest.main()
