from pathlib import Path
from unittest import TestCase


class FrontendAgentSkillsUiTest(TestCase):
    def test_agent_skills_api_client_exposes_list_get_import_and_usage_endpoints(self) -> None:
        api_source = Path("apps/web/src/api/agentSkills.ts").read_text(encoding="utf-8")

        self.assertIn("export async function listAgentSkills", api_source)
        self.assertIn("export async function getAgentSkill", api_source)
        self.assertIn("export async function importAgentSkill", api_source)
        self.assertIn("export async function pinAgentSkill", api_source)
        self.assertIn("export async function archiveAgentSkill", api_source)
        self.assertIn('"/api/v1/agent-skills"', api_source)
        self.assertIn('"/api/v1/agent-skills/import"', api_source)

    def test_agent_skill_types_include_metadata_and_availability_state(self) -> None:
        types_source = Path("apps/web/src/types/agentSkills.ts").read_text(encoding="utf-8")

        self.assertIn("export interface AgentSkill", types_source)
        self.assertIn("export interface AgentSkillImportInput", types_source)
        self.assertIn("export interface AgentSkillMetadata", types_source)
        self.assertIn("required_tools", types_source)
        self.assertIn("allowed_tools", types_source)
        self.assertIn("ask_tools", types_source)
        self.assertIn("disallowed_tools", types_source)
        self.assertIn("description_quality_score", types_source)
        self.assertIn("security_risk_level", types_source)
        self.assertIn("source_types", types_source)
        self.assertIn("availability_state", types_source)
        self.assertIn("tool_dependency_state", types_source)
        self.assertIn("missing_required_tools", types_source)
        self.assertIn("missing_optional_tools", types_source)

    def test_app_renders_skill_management_page_with_import_and_dependency_status(self) -> None:
        app_source = Path("apps/web/src/app/App.tsx").read_text(encoding="utf-8")

        self.assertIn('"skills"', app_source)
        self.assertIn("listAgentSkills", app_source)
        self.assertIn("importAgentSkill", app_source)
        self.assertIn("SkillManagementPage", app_source)
        self.assertIn("allowed_tools", app_source)
        self.assertIn("ask_tools", app_source)
        self.assertIn("disallowed_tools", app_source)
        self.assertIn("description_quality_score", app_source)
        self.assertIn("Skill 管理", app_source)
        self.assertIn("本地 SKILL.md 或目录", app_source)
        self.assertIn("依赖工具", app_source)
        self.assertIn("availability_state", app_source)
        self.assertIn("source_types", app_source)
        self.assertIn("missingRequiredTools", app_source)
        self.assertIn("missingOptionalTools", app_source)
        self.assertIn("tool_dependency_state", app_source)

    def test_skill_management_css_has_catalog_grid_and_status_chips(self) -> None:
        css_source = Path("apps/web/src/styles/global.css").read_text(encoding="utf-8")

        self.assertIn(".skill-management-grid", css_source)
        self.assertIn(".skill-card-grid", css_source)
        self.assertIn(".skill-card", css_source)
        self.assertIn(".skill-status-chip", css_source)
        self.assertIn(".skill-dependency-list", css_source)
