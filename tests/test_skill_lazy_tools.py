import shutil
import sys
import textwrap
import unittest
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class SkillLazyToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        from app.db.base import Base
        import app.domains.agent_memory.models  # noqa: F401
        import app.domains.automation.models  # noqa: F401
        import app.domains.conversations.models  # noqa: F401

        self.tmp_root = PROJECT_ROOT / ".tmp-test-artifacts" / "skill-lazy-tools" / self._testMethodName
        shutil.rmtree(self.tmp_root, ignore_errors=True)
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

    def tearDown(self) -> None:
        self.engine.dispose()
        shutil.rmtree(PROJECT_ROOT / ".tmp-test-artifacts" / "skill-lazy-tools", ignore_errors=True)

    def _skill_repository(self, session):
        from app.agent_runtime.memory.skill_repository import AgentSkillRepository
        from app.domains.agent_memory.repository import AgentMemoryRepository

        return AgentSkillRepository(AgentMemoryRepository(session), skill_root=self.tmp_root / "docs" / "agent-skills")

    def _import_article_skill(self, session):
        source_dir = self.tmp_root / "downloaded" / "article-fetcher"
        (source_dir / "scripts").mkdir(parents=True)
        (source_dir / "references").mkdir()
        (source_dir / "scripts" / "fetch_article.py").write_text(
            textwrap.dedent(
                """
                import argparse
                import json

                parser = argparse.ArgumentParser()
                parser.add_argument("--url", required=True)
                parser.add_argument("--limit", type=int, default=20)
                args = parser.parse_args()

                print(json.dumps({"url": args.url, "limit": args.limit, "status": "fetched"}, ensure_ascii=False))
                """
            ).strip(),
            encoding="utf-8",
        )
        (source_dir / "scripts" / "_helper.py").write_text("print('helper')\n", encoding="utf-8")
        (source_dir / "references" / "action-schemas.json").write_text(
            textwrap.dedent(
                """
                {
                  "fetch_article": {
                    "description": "读取文章 URL 并返回结构化正文。",
                    "required": ["url"],
                    "properties": {
                      "url": {"type": "string", "description": "文章 URL"},
                      "limit": {"type": "integer", "default": 20}
                    },
                    "additionalProperties": false
                  }
                }
                """
            ).strip(),
            encoding="utf-8",
        )
        (source_dir / "SKILL.md").write_text(
            textwrap.dedent(
                """
                ---
                name: article-fetcher
                description: When the user provides an article URL and wants extracted article content as JSON without modifying files.
                allowed-tools: [skill_call]
                metadata: {"openclaw":{"actions":{"modelAllow":["fetch_article"],"executeAllow":["fetch_article"],"actionSchemasFile":"references/action-schemas.json"}}}
                ---
                # Article Fetcher

                ## 何时使用
                用户给出文章 URL，希望读取文章内容并返回结构化 JSON 时使用。
                """
            ).strip(),
            encoding="utf-8",
        )

        repository = self._skill_repository(session)
        return repository.import_skill_from_path(source_dir, category="content_source")

    def test_default_registry_exposes_my_agents_style_lazy_skill_tools(self) -> None:
        from app.agent_runtime.tool_registry import (
            SKILL_CALL_TOOL,
            SKILL_LIST_ACTIONS_TOOL,
            SKILL_LIST_TOOL,
            SKILL_READ_TOOL,
            create_default_agent_tool_registry,
        )

        registry = create_default_agent_tool_registry()

        self.assertIsNotNone(registry.get(SKILL_LIST_TOOL))
        self.assertIsNotNone(registry.get(SKILL_LIST_ACTIONS_TOOL))
        self.assertIsNotNone(registry.get(SKILL_READ_TOOL))
        self.assertIsNotNone(registry.get(SKILL_CALL_TOOL))
        self.assertTrue(registry.get(SKILL_CALL_TOOL).requires_confirmation)

    def test_skill_list_actions_read_and_call_follow_progressive_disclosure(self) -> None:
        from app.agent_runtime.tool_registry import (
            SKILL_CALL_TOOL,
            SKILL_LIST_ACTIONS_TOOL,
            SKILL_LIST_TOOL,
            SKILL_READ_TOOL,
            create_default_agent_tool_registry,
        )

        registry = create_default_agent_tool_registry()
        with self.Session() as session:
            skill = self._import_article_skill(session)
            session.commit()

            list_result = registry.get(SKILL_LIST_TOOL).handler(session, query="article URL", limit=5, offset=0)
            actions_result = registry.get(SKILL_LIST_ACTIONS_TOOL).handler(session, skill="article-fetcher")
            read_result = registry.get(SKILL_READ_TOOL).handler(session, skill="article-fetcher", include_frontmatter=False, max_chars=400)
            call_result = registry.get(SKILL_CALL_TOOL).handler(
                session,
                skill="article-fetcher",
                action="fetch_article",
                arguments={"url": "https://example.com/post", "limit": 3},
            )
            invalid_result = registry.get(SKILL_CALL_TOOL).handler(
                session,
                skill="article-fetcher",
                action="fetch_article",
                arguments={"limit": 3, "unexpected": "x"},
            )

        first_skill = list_result["result"]["skills"][0]
        action = actions_result["result"]["actions"][0]

        self.assertTrue(list_result["ok"])
        self.assertEqual(skill.id, first_skill["skill_id"])
        self.assertEqual(["fetch_article"], first_skill["actions_preview"])
        self.assertTrue(actions_result["ok"])
        self.assertEqual("fetch_article", action["action"])
        self.assertEqual(["url"], action["required"])
        self.assertEqual(["url", "limit"], action["parameter_names"])
        self.assertFalse(action["input_schema"]["additionalProperties"])
        self.assertTrue(read_result["ok"])
        self.assertTrue(read_result["result"]["content"].startswith("# Article Fetcher"))
        self.assertNotIn("metadata:", read_result["result"]["content"])
        self.assertTrue(call_result["ok"])
        self.assertEqual("fetch_article", call_result["result"]["action"])
        self.assertIn("https://example.com/post", call_result["result"]["stdout"])
        self.assertIn('"limit": 3', call_result["result"]["stdout"])
        self.assertFalse(invalid_result["ok"])
        self.assertEqual("SKILL_ACTION_ARGUMENTS_INVALID", invalid_result["error"])
        self.assertIn("url", invalid_result["result"]["message"])
        self.assertIn("unexpected", invalid_result["result"]["message"])


if __name__ == "__main__":
    unittest.main()
