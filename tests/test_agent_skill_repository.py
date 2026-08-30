import shutil
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


class AgentSkillRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        from app.db.base import Base
        import app.domains.agent_memory.models  # noqa: F401
        import app.domains.automation.models  # noqa: F401
        import app.domains.conversations.models  # noqa: F401

        self.engine = create_engine(
            "sqlite+pysqlite://",
            connect_args={"check_same_thread": False},
            future=True,
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)
        self.skill_root = PROJECT_ROOT / ".tmp-agent-skill-tests" / self._testMethodName / "docs" / "agent-skills"
        shutil.rmtree(self.skill_root.parent.parent, ignore_errors=True)
        self.skill_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.engine.dispose()
        shutil.rmtree(PROJECT_ROOT / ".tmp-agent-skill-tests", ignore_errors=True)

    def _app(self):
        from fastapi import Depends
        from app.api.v1.agent_skills import get_skill_repository
        from app.db.session import get_db_session
        from app.main import create_app

        app = create_app()

        def override_session():
            with self.Session() as session:
                yield session

        def override_skill_repository(session=Depends(get_db_session)):
            from app.agent_runtime.memory.skill_repository import AgentSkillRepository
            from app.domains.agent_memory.repository import AgentMemoryRepository

            return AgentSkillRepository(AgentMemoryRepository(session), skill_root=self.skill_root)

        app.dependency_overrides[get_db_session] = override_session
        app.dependency_overrides[get_skill_repository] = override_skill_repository
        return app

    def _skill_repository(self, session):
        from app.agent_runtime.memory.skill_repository import AgentSkillRepository
        from app.domains.agent_memory.repository import AgentMemoryRepository

        return AgentSkillRepository(AgentMemoryRepository(session), skill_root=self.skill_root)

    def _skill_create(self, *, name: str = "wechat-recruiting-sync", protected: bool = False, pinned: bool = False):
        from app.domains.agent_memory.schemas import AgentSkillCreate

        return AgentSkillCreate(
            name=name,
            title="微信公众号招聘同步",
            description="从微信公众号文章中沉淀招聘同步流程经验。",
            category="job_source_sync",
            protected=protected,
            pinned=pinned,
            created_by="developer",
            sections={
                "when_to_use": "用户提供公众号名称或招聘文章时使用。",
                "inputs": "公众号名称、文章链接、可见正文。",
                "outputs": "招聘开放信号、候选文章、验证状态。",
                "workflow": "先识别来源，再抓取正文，再抽取招聘信号。",
                "tool_boundaries": "不得绕过登录、验证码或用户确认边界。",
                "confirmation_points": "正式投递前必须用户确认。",
                "error_handling": "抓取失败时返回结构化错误。",
                "verification": "投递前去企业官网验证岗位。",
                "references": "就业信息同步模块文档。",
            },
        )

    def _approved_candidate(self, skill_id: str):
        from app.domains.agent_memory.models import (
            AgentLearningCandidateLessonType,
            AgentLearningCandidateRiskLevel,
            AgentLearningCandidateStatus,
        )
        from app.domains.agent_memory.schemas import AgentLearningCandidateCreate

        return AgentLearningCandidateCreate(
            source_agent_run_id="agent-run-skill",
            source_workflow_run_id="workflow-skill",
            source_tool_call_log_id="tool-log-skill",
            lesson_type=AgentLearningCandidateLessonType.TOOL_RECOVERY,
            target_scope="wechat_sync",
            suggested_skill_target="wechat-recruiting-sync",
            target_skill_id=skill_id,
            candidate_title="公众号正文回退流程",
            candidate_body="当公众号链接只返回摘要时，要求用户提供可见正文后再抽取招聘开放信号。",
            evidence_summary="tool-log-skill 从摘要页失败后通过可见正文恢复。",
            success_evidence="extracted_count=2, verified=true",
            risk_level=AgentLearningCandidateRiskLevel.MEDIUM,
            evidence_json={"tool_call_log_ids": ["tool-log-skill"]},
            metadata_json={"candidate_status_for_test": AgentLearningCandidateStatus.APPROVED.value},
        )

    def test_create_skill_writes_markdown_file_with_required_sections(self) -> None:
        from app.domains.agent_memory.models import AgentSkillStorageType, AgentSkillUsage

        with self.Session() as session:
            repository = self._skill_repository(session)
            skill = repository.create_skill(self._skill_create())
            usage = repository.get_usage(skill.id)
            session.commit()

        skill_path = Path(skill.file_path)
        content = skill_path.read_text(encoding="utf-8")

        self.assertEqual(AgentSkillStorageType.MARKDOWN_FILE, skill.storage_type)
        self.assertTrue(skill_path.is_file())
        self.assertIn("docs", skill_path.parts)
        self.assertIn("agent-skills", skill_path.parts)
        self.assertIn("# 微信公众号招聘同步", content)
        for section in [
            "## 何时使用",
            "## 输入",
            "## 输出",
            "## 标准流程",
            "## 工具边界",
            "## 用户确认点",
            "## 错误处理",
            "## 验证方式",
            "## 关联参考文件",
            "## 历史经验",
        ]:
            self.assertIn(section, content)
        self.assertIsNotNone(usage)
        self.assertIsInstance(usage, AgentSkillUsage)
        self.assertEqual(0, usage.use_count)

    def test_import_skill_from_local_skill_md_copies_file_and_extracts_tool_metadata(self) -> None:
        source_dir = PROJECT_ROOT / ".tmp-agent-skill-tests" / self._testMethodName / "downloaded" / "xiaohongshu-skill"
        source_dir.mkdir(parents=True, exist_ok=True)
        source_file = source_dir / "SKILL.md"
        source_file.write_text(
            "\n".join(
                [
                    "---",
                    "name: xiaohongshu-recruiting-sync",
                    "description: Parse Xiaohongshu notes and images for campus recruiting signals.",
                    "source_types: [xiaohongshu_note]",
                    "required_tools: [xiaohongshu-mcp.read_note, ocr.extract_text]",
                    "---",
                    "# 小红书秋招内容解析",
                    "",
                    "Use this skill when the user provides Xiaohongshu recruiting notes.",
                ]
            ),
            encoding="utf-8",
        )

        with self.Session() as session:
            repository = self._skill_repository(session)
            skill = repository.import_skill_from_path(source_dir, category="content_source")
            usage = repository.get_usage(skill.id)
            session.commit()

        imported_path = Path(skill.file_path)
        imported_content = imported_path.read_text(encoding="utf-8")

        self.assertTrue(imported_path.is_file())
        self.assertNotEqual(source_file, imported_path)
        self.assertEqual("xiaohongshu-recruiting-sync", skill.name)
        self.assertEqual("小红书秋招内容解析", skill.title)
        self.assertEqual("content_source", skill.category)
        self.assertIn("xiaohongshu-mcp.read_note", imported_content)
        self.assertEqual(["xiaohongshu_note"], skill.metadata_json["source_types"])
        self.assertEqual(["xiaohongshu-mcp.read_note", "ocr.extract_text"], skill.metadata_json["required_tools"])
        self.assertEqual("unavailable", skill.metadata_json["availability_state"])
        self.assertEqual(0, usage.use_count)

    def test_import_skill_from_claude_code_package_stores_import_report(self) -> None:
        source_dir = PROJECT_ROOT / ".tmp-agent-skill-tests" / self._testMethodName / "downloaded" / "wechat-skill"
        (source_dir / "scripts").mkdir(parents=True, exist_ok=True)
        (source_dir / "references").mkdir()
        (source_dir / "scripts" / "fetch.py").write_text("print('ok')\n", encoding="utf-8")
        (source_dir / "references" / "usage.md").write_text("# Usage\n", encoding="utf-8")
        (source_dir / "SKILL.md").write_text(
            "\n".join(
                [
                    "---",
                    "name: wechat-recruiting-import",
                    "description: 当用户提供微信公众号文章、公众号账号或招聘汇总链接，并希望抽取秋招开放公司信号且不自动投递时使用",
                    "source_types: [wechat_article, wechat_account]",
                    "required_tools: [weixin-articles-mcp.read_article]",
                    "allowed-tools: [weixin-articles-mcp.read_article, ocr.extract_text]",
                    "disallowed-tools: [submit_application]",
                    "compatibility: [claude-code]",
                    "license: MIT",
                    "---",
                    "# 微信公众号招聘导入",
                    "",
                    "抽取公众号招聘开放信号。",
                ]
            ),
            encoding="utf-8",
        )

        with self.Session() as session:
            repository = self._skill_repository(session)
            skill = repository.import_skill_from_path(source_dir, category="content_source")
            session.commit()

        metadata = skill.metadata_json
        self.assertEqual(["wechat_article", "wechat_account"], metadata["source_types"])
        self.assertEqual(["weixin-articles-mcp.read_article"], metadata["required_tools"])
        self.assertEqual(["weixin-articles-mcp.read_article", "ocr.extract_text"], metadata["allowed_tools"])
        self.assertEqual(["submit_application"], metadata["disallowed_tools"])
        self.assertEqual(["claude-code"], metadata["compatibility"])
        self.assertEqual("MIT", metadata["license"])
        self.assertEqual(["scripts/fetch.py"], metadata["resources"]["scripts"])
        self.assertEqual(["references/usage.md"], metadata["resources"]["references"])
        self.assertRegex(metadata["version_hash"], r"^[0-9a-f]{64}$")
        self.assertGreaterEqual(metadata["description_quality_score"], 8)
        self.assertIn("allowed_tools 只是 Skill 申请权限", metadata["permission_notice"])

    def test_record_skill_use_increments_usage_counter(self) -> None:
        from app.domains.agent_memory.models import AgentSkillUsageEvent

        with self.Session() as session:
            repository = self._skill_repository(session)
            skill = repository.create_skill(self._skill_create(name="dlmu-campus-sync"))
            usage = repository.record_usage(skill.id, AgentSkillUsageEvent.USE)
            session.commit()

        self.assertEqual(1, usage.use_count)
        self.assertIsNotNone(usage.last_used_at)

    def test_protected_or_pinned_skill_blocks_background_review_patch(self) -> None:
        with self.Session() as session:
            repository = self._skill_repository(session)
            protected_skill = repository.create_skill(self._skill_create(name="protected-sync", protected=True))
            pinned_skill = repository.create_skill(self._skill_create(name="pinned-sync", pinned=True))

            with self.assertRaises(PermissionError):
                repository.append_section(
                    protected_skill.id,
                    heading="历史经验",
                    body="后台复盘不得修改 protected skill。",
                    actor="agent_review",
                )
            with self.assertRaises(PermissionError):
                repository.append_section(
                    pinned_skill.id,
                    heading="历史经验",
                    body="后台复盘不得修改 pinned skill。",
                    actor="agent_review",
                )
            protected_usage = repository.get_usage(protected_skill.id)
            pinned_usage = repository.get_usage(pinned_skill.id)
            session.commit()

        self.assertEqual(0, protected_usage.patch_count)
        self.assertEqual(0, pinned_usage.patch_count)

    def test_apply_candidate_reads_current_skill_version_and_appends_experience(self) -> None:
        from app.domains.agent_memory.models import AgentLearningCandidateStatus, AgentSkillUsageEvent
        from app.domains.agent_memory.repository import AgentMemoryRepository
        from app.domains.agent_memory.service import AgentLearningService

        with self.Session() as session:
            repository = self._skill_repository(session)
            skill = repository.create_skill(self._skill_create(name="candidate-apply-sync"))
            skill_path = Path(skill.file_path)
            original_content = skill_path.read_text(encoding="utf-8")
            learning_service = AgentLearningService(AgentMemoryRepository(session))
            candidate = learning_service.create_learning_candidate(self._approved_candidate(skill.id))
            candidate.status = AgentLearningCandidateStatus.APPROVED
            applied = learning_service.apply_candidate(candidate.id, skill_repository=repository)
            usage = repository.get_usage(skill.id)
            session.commit()

        patched_content = skill_path.read_text(encoding="utf-8")

        self.assertEqual(AgentLearningCandidateStatus.APPLIED, applied.status)
        self.assertIsNotNone(applied.applied_at)
        self.assertEqual(skill.id, applied.target_skill_id)
        self.assertIn("公众号正文回退流程", patched_content)
        self.assertIn("当公众号链接只返回摘要时", patched_content)
        self.assertIn(original_content.strip(), patched_content)
        self.assertEqual(1, usage.patch_count)
        self.assertIsNotNone(usage.last_patched_at)
        self.assertEqual(AgentSkillUsageEvent.PATCH.value, applied.metadata_json["applied_usage_event"])
        self.assertIn("previous_skill_version_hash", applied.metadata_json)
        self.assertIn("applied_skill_version_hash", applied.metadata_json)

    def test_skill_api_lists_gets_records_usage_pins_and_archives(self) -> None:
        with self.Session() as session:
            repository = self._skill_repository(session)
            skill = repository.create_skill(self._skill_create(name="api-sync"))
            session.commit()

        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                list_response = await client.get("/api/v1/agent-skills")
                get_response = await client.get(f"/api/v1/agent-skills/{skill.id}")
                usage_response = await client.post(f"/api/v1/agent-skills/{skill.id}/usage", json={"event": "use"})
                pin_response = await client.post(f"/api/v1/agent-skills/{skill.id}/pin")
                archive_response = await client.post(f"/api/v1/agent-skills/{skill.id}/archive")
                return list_response, get_response, usage_response, pin_response, archive_response

        list_response, get_response, usage_response, pin_response, archive_response = run(call_api())

        self.assertEqual(200, list_response.status_code)
        listed_names = {item["name"] for item in list_response.json()["items"]}
        self.assertEqual(4, len(listed_names))
        self.assertIn("api-sync", listed_names)
        self.assertIn("wechat-article-content-fetch", listed_names)
        self.assertIn("xiaohongshu-content-fetch", listed_names)
        self.assertIn("database-operations", listed_names)
        self.assertEqual(200, get_response.status_code)
        self.assertEqual("markdown_file", get_response.json()["skill"]["storage_type"])
        self.assertTrue(get_response.json()["content"].startswith("# "))
        self.assertEqual(200, usage_response.status_code)
        self.assertEqual(1, usage_response.json()["use_count"])
        self.assertEqual(200, pin_response.status_code)
        self.assertTrue(pin_response.json()["pinned"])
        self.assertEqual(200, archive_response.status_code)
        self.assertEqual("archived", archive_response.json()["status"])

    def test_skill_api_imports_local_skill_directory(self) -> None:
        source_dir = PROJECT_ROOT / ".tmp-agent-skill-tests" / self._testMethodName / "downloaded" / "wechat-skill"
        source_dir.mkdir(parents=True, exist_ok=True)
        (source_dir / "SKILL.md").write_text(
            "\n".join(
                [
                    "---",
                    "name: wechat-recruiting-articles",
                    "description: Read WeChat recruiting articles and extract company signals.",
                    "source_types: [wechat_article, wechat_account]",
                    "required_tools: [weixin-articles-mcp.read_article]",
                    "---",
                    "# 微信公众号招聘文章读取",
                ]
            ),
            encoding="utf-8",
        )
        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                import_response = await client.post(
                    "/api/v1/agent-skills/import",
                    json={"source_path": str(source_dir), "category": "content_source"},
                )
                list_response = await client.get("/api/v1/agent-skills")
                return import_response, list_response

        import_response, list_response = run(call_api())

        self.assertEqual(201, import_response.status_code)
        self.assertEqual("wechat-recruiting-articles", import_response.json()["name"])
        self.assertEqual(["wechat_article", "wechat_account"], import_response.json()["metadata_json"]["source_types"])
        self.assertEqual(200, list_response.status_code)
        listed_names = {item["name"] for item in list_response.json()["items"]}
        self.assertEqual(4, len(listed_names))
        self.assertIn("wechat-recruiting-articles", listed_names)
        self.assertIn("wechat-article-content-fetch", listed_names)
        self.assertIn("xiaohongshu-content-fetch", listed_names)
        self.assertIn("database-operations", listed_names)

    def test_skill_api_bootstraps_builtin_content_source_skills_on_list(self) -> None:
        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                first_response = await client.get("/api/v1/agent-skills")
                second_response = await client.get("/api/v1/agent-skills")
                return first_response, second_response

        first_response, second_response = run(call_api())

        self.assertEqual(200, first_response.status_code)
        self.assertEqual(200, second_response.status_code)
        first_names = {item["name"] for item in first_response.json()["items"]}
        second_names = [item["name"] for item in second_response.json()["items"]]

        self.assertIn("wechat-article-content-fetch", first_names)
        self.assertIn("xiaohongshu-content-fetch", first_names)
        self.assertEqual(len(second_names), len(set(second_names)))
        self.assertEqual(3, len(second_names))
        self.assertIn("database-operations", second_names)

    def test_skill_api_bootstraps_builtin_database_operations_skill_on_list(self) -> None:
        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.get("/api/v1/agent-skills")
                return response

        response = run(call_api())

        self.assertEqual(200, response.status_code)
        database_skill = next(
            item
            for item in response.json()["items"]
            if item["name"] == "database-operations"
        )
        metadata = database_skill["metadata_json"]
        self.assertEqual("available", metadata["availability_state"])
        self.assertIn("database.company_search", metadata["required_tools"])
        self.assertIn("database.company_update", metadata["ask_tools"])
        self.assertIn("database.job_lead_delete", metadata["ask_tools"])

    def test_skill_api_decorates_dependency_status_from_single_agent_tool_registry(self) -> None:
        source_dir = PROJECT_ROOT / ".tmp-agent-skill-tests" / self._testMethodName / "downloaded" / "memory-skill"
        source_dir.mkdir(parents=True, exist_ok=True)
        (source_dir / "SKILL.md").write_text(
            "\n".join(
                [
                    "---",
                    "name: memory-recall-skill",
                    "description: Use prior session memory search to answer questions about previous agent work.",
                    "required_tools: [memory_search]",
                    "allowed_tools: [memory_search, missing.optional_tool]",
                    "---",
                    "# Memory Recall Skill",
                ]
            ),
            encoding="utf-8",
        )
        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                import_response = await client.post(
                    "/api/v1/agent-skills/import",
                    json={"source_path": str(source_dir), "category": "agent_memory"},
                )
                list_response = await client.get("/api/v1/agent-skills")
                return import_response, list_response

        import_response, list_response = run(call_api())

        self.assertEqual(201, import_response.status_code)
        imported_metadata = import_response.json()["metadata_json"]
        listed_metadata = list_response.json()["items"][0]["metadata_json"]

        self.assertEqual("available", imported_metadata["availability_state"])
        self.assertEqual("available", imported_metadata["tool_dependency_state"])
        self.assertEqual(["memory_search"], imported_metadata["available_required_tools"])
        self.assertEqual([], imported_metadata["missing_required_tools"])
        self.assertEqual(["missing.optional_tool"], imported_metadata["missing_optional_tools"])
        self.assertEqual(imported_metadata["tool_dependency_state"], listed_metadata["tool_dependency_state"])


if __name__ == "__main__":
    unittest.main()
