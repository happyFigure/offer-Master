import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class ResourceEffectsTest(unittest.TestCase):
    def test_promotes_declared_tool_resource_effects_without_tool_specific_code(self) -> None:
        from app.agent_runtime.context.resource_effects import promote_declared_resource_effects_context

        promoted = promote_declared_resource_effects_context(
            {},
            {
                "ok": True,
                "tool_name": "resume.export_pdf",
                "resource_effects": [
                    {
                        "resource_type": "file",
                        "action": "created",
                        "operation": "export_pdf",
                        "source_path": "C:/简历/resume.tex",
                        "target_path": "C:/简历/resume.pdf",
                        "focus_path": "C:/简历/resume.pdf",
                        "aliases": ["刚才生成的 PDF"],
                    }
                ],
            },
        )

        self.assertEqual("C:/简历/resume.pdf", promoted["active_resource"]["focus_path"])
        self.assertEqual("export_pdf", promoted["active_resource"]["operation"])
        self.assertEqual("C:/简历/resume.pdf", promoted["artifact_context"]["active_artifact"]["focus_path"])
        self.assertEqual("刚才生成的 PDF", promoted["resource_effects"][-1]["aliases"][0])

    def test_builds_resource_effects_from_semantic_result_contract_template(self) -> None:
        from app.agent_runtime.context.resource_effects import promote_contract_resource_effects_context
        from app.agent_runtime.tool_registry import AgentToolSemanticProfile

        profile = AgentToolSemanticProfile(
            intent="export_resume_pdf",
            target_type="file_artifact",
            result_contract={
                "resource_effect_templates": [
                    {
                        "resource_type": "file",
                        "action": "created",
                        "operation": "export_pdf",
                        "source_path": "$input.path",
                        "target_path": "$result.result.output_path",
                        "focus_path": "$result.result.output_path",
                        "aliases": ["刚才生成的 PDF", "导出的 PDF"],
                    }
                ]
            },
        )

        promoted = promote_contract_resource_effects_context(
            {},
            {"ok": True, "result": {"output_path": "C:/简历/resume.pdf"}},
            tool_input={"path": "C:/简历/resume.tex"},
            semantic_profile=profile,
        )

        effect = promoted["resource_effects"][-1]
        self.assertEqual("C:/简历/resume.tex", effect["source_path"])
        self.assertEqual("C:/简历/resume.pdf", effect["target_path"])
        self.assertEqual("C:/简历/resume.pdf", promoted["active_resource"]["focus_path"])


if __name__ == "__main__":
    unittest.main()
