from __future__ import annotations

import unittest


class SkillAsCapabilityTest(unittest.TestCase):
    def test_default_capability_registry_exposes_filesystem_skill_capability(self) -> None:
        from app.agent_runtime.agent_as_tool import create_default_agent_capability_registry

        registry = create_default_agent_capability_registry()
        capability = registry.get("skill.filesystem")

        self.assertIsNotNone(capability)
        self.assertEqual("skill", capability.kind)
        self.assertEqual("skill_executor.filesystem", capability.executor_id)
        self.assertIn("filesystem_operation", capability.candidate_profile.categories)

    def test_filesystem_request_selects_skill_capability_not_internal_tools(self) -> None:
        from app.agent_runtime.agent_as_tool import create_default_agent_capability_registry
        from app.agent_runtime.tool_candidate_selector import ToolCandidateSelector

        registry = create_default_agent_capability_registry()
        selection = ToolCandidateSelector(registry).select(
            "把刚才这个文件名字改成 刘汉卿-后端开发-AI-Agent",
            source_type="agent_chat",
            auto_executable_only=False,
        )

        self.assertIn("skill.filesystem", selection.capabilities)
        self.assertNotIn("filesystem.move_file", selection.capabilities)
        self.assertNotIn("filesystem.replace_text", selection.capabilities)
        self.assertNotIn("filesystem.read_file", selection.capabilities)

    def test_path_existence_request_selects_filesystem_skill(self) -> None:
        from app.agent_runtime.agent_as_tool import create_default_agent_capability_registry
        from app.agent_runtime.tool_candidate_selector import ToolCandidateSelector

        registry = create_default_agent_capability_registry()
        selection = ToolCandidateSelector(registry).select(
            "C:/Users/phoenix/Documents/Obsidian Vault/简历/刘汉卿-后端开发-AI-Agent平台简历.tex 你看下这个文件是否存在",
            source_type="agent_chat",
            auto_executable_only=False,
        )

        self.assertIn("skill.filesystem", selection.capabilities)
        self.assertNotIn("filesystem.path_exists", selection.capabilities)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
