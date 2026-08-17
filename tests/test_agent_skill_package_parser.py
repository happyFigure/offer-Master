import shutil
import sys
import textwrap
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class AgentSkillPackageParserTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_root = PROJECT_ROOT / ".tmp-agent-skill-package-tests" / self._testMethodName
        shutil.rmtree(self.tmp_root, ignore_errors=True)
        self.tmp_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(PROJECT_ROOT / ".tmp-agent-skill-package-tests", ignore_errors=True)

    def test_parse_claude_code_style_skill_package_report(self) -> None:
        from app.agent_runtime.memory.skill_package_parser import SkillPackageParser

        skill_dir = self.tmp_root / "wechat-recruiting-skill"
        (skill_dir / "scripts").mkdir(parents=True)
        (skill_dir / "references").mkdir()
        (skill_dir / "assets").mkdir()
        (skill_dir / "agents").mkdir()
        (skill_dir / "scripts" / "fetch_wechat.py").write_text("print('fetch')\n", encoding="utf-8")
        (skill_dir / "references" / "usage.md").write_text("# Usage\n", encoding="utf-8")
        (skill_dir / "assets" / "example.json").write_text("{}\n", encoding="utf-8")
        (skill_dir / "agents" / "openai.yaml").write_text("name: wechat-agent\n", encoding="utf-8")
        (skill_dir / "SKILL.md").write_text(
            textwrap.dedent(
                """
                ---
                name: wechat-recruiting-sync
                description: 当用户提供微信公众号文章、公众号账号或招聘汇总链接，并希望抽取秋招开放公司信号且不自动投递时使用
                source_types: [wechat_article, wechat_account]
                required_tools: [weixin-articles-mcp.read_article]
                allowed-tools: [weixin-articles-mcp.read_article, ocr.extract_text]
                ask-tools: [browser.open]
                disallowed-tools: [submit_application, read_secret]
                disable-model-invocation: false
                compatibility: [claude-code, codex]
                license: MIT
                ---
                # 微信公众号招聘同步

                用于从公众号文章里抽取公司开放秋招信号，输出候选公司和来源链接。
                """
            ).strip(),
            encoding="utf-8",
        )

        package = SkillPackageParser().parse(skill_dir)
        report = package.import_report

        self.assertEqual("wechat-recruiting-sync", package.name)
        self.assertEqual("微信公众号招聘同步", package.title)
        self.assertEqual(["wechat_article", "wechat_account"], report["source_types"])
        self.assertEqual(["weixin-articles-mcp.read_article"], report["required_tools"])
        self.assertEqual(["weixin-articles-mcp.read_article", "ocr.extract_text"], report["allowed_tools"])
        self.assertEqual(["browser.open"], report["ask_tools"])
        self.assertEqual(["submit_application", "read_secret"], report["disallowed_tools"])
        self.assertFalse(report["disable_model_invocation"])
        self.assertEqual(["claude-code", "codex"], report["compatibility"])
        self.assertEqual("MIT", report["license"])
        self.assertEqual(["scripts/fetch_wechat.py"], report["resources"]["scripts"])
        self.assertEqual(["references/usage.md"], report["resources"]["references"])
        self.assertEqual(["assets/example.json"], report["resources"]["assets"])
        self.assertEqual(["agents/openai.yaml"], report["resources"]["agents"])
        self.assertRegex(report["version_hash"], r"^[0-9a-f]{64}$")
        self.assertGreaterEqual(report["description_quality_score"], 8)
        self.assertEqual("high", report["security_risk_level"])
        self.assertEqual("unavailable", report["availability_state"])
        self.assertEqual([], report["blocking_errors"])
        self.assertIn("allowed_tools 只是 Skill 申请权限", report["permission_notice"])

    def test_generic_description_is_imported_but_auto_trigger_is_disabled(self) -> None:
        from app.agent_runtime.memory.skill_package_parser import SkillPackageParser

        skill_dir = self.tmp_root / "generic-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            textwrap.dedent(
                """
                ---
                name: generic-web-helper
                description: 处理网页
                allowed-tools: [browser.open]
                ---
                # Generic Web Helper
                """
            ).strip(),
            encoding="utf-8",
        )

        package = SkillPackageParser().parse(skill_dir)
        report = package.import_report

        self.assertLessEqual(report["description_quality_score"], 5)
        self.assertEqual("disabled", report["auto_trigger_state"])
        self.assertIn("description 过于空泛", "\n".join(report["import_warnings"]))
        self.assertEqual([], report["blocking_errors"])

    def test_clear_non_recruiting_skill_description_scores_high(self) -> None:
        from app.agent_runtime.memory.skill_package_parser import SkillPackageParser

        skill_dir = self.tmp_root / "pdf-table-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            textwrap.dedent(
                """
                ---
                name: pdf-table-extractor
                description: 当用户提供 PDF 文件并希望提取表格为 CSV，且不修改原始文件时使用
                allowed-tools: [pdf.extract_tables]
                ---
                # PDF Table Extractor
                """
            ).strip(),
            encoding="utf-8",
        )

        package = SkillPackageParser().parse(skill_dir)
        report = package.import_report

        self.assertGreaterEqual(report["description_quality_score"], 8)
        self.assertEqual("enabled", report["auto_trigger_state"])
        self.assertEqual([], report["blocking_errors"])
        self.assertEqual([], report["import_warnings"])

    def test_missing_description_blocks_import(self) -> None:
        from app.agent_runtime.memory.skill_package_parser import SkillPackageParser

        skill_dir = self.tmp_root / "missing-description-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            textwrap.dedent(
                """
                ---
                name: missing-description
                allowed-tools: [browser.open]
                ---
                # Missing Description
                """
            ).strip(),
            encoding="utf-8",
        )

        package = SkillPackageParser().parse(skill_dir)

        self.assertIn("description 缺失", "\n".join(package.import_report["blocking_errors"]))


if __name__ == "__main__":
    unittest.main()
