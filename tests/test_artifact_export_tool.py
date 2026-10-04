import tempfile
from pathlib import Path
from unittest import TestCase


class ArtifactExportToolTest(TestCase):
    def test_exports_sandbox_output_file_to_destination_after_confirmation(self) -> None:
        from app.agent_runtime.context.resource_effects import promote_contract_resource_effects_context
        from app.agent_runtime.sdk_agents.sandbox import SdkAgentSandboxManager
        from app.agent_runtime.tool_registry import ARTIFACT_EXPORT_TOOL, create_artifact_agent_tool_definitions

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sandbox_base = temp_path / "sdk-agent-runs"
            workspace = SdkAgentSandboxManager(base_dir=sandbox_base).prepare_workspace(run_id="run-123")
            output_file = workspace.output_dir / "resume_tailored.md"
            output_file.write_text("tailored resume", encoding="utf-8")
            destination = temp_path / "exports" / "resume_tailored.md"

            export_tool = create_artifact_agent_tool_definitions(sandbox_base_dir=sandbox_base)[0]
            result = export_tool.handler(
                None,
                artifact_uri="artifact-sandbox://run-123/output/resume_tailored.md",
                destination=str(destination),
            )

            self.assertTrue(result["ok"])
            self.assertEqual(ARTIFACT_EXPORT_TOOL, result["tool_name"])
            self.assertEqual("tailored resume", destination.read_text(encoding="utf-8"))
            self.assertEqual("artifact-sandbox://run-123/output/resume_tailored.md", result["result"]["source_uri"])
            self.assertEqual(str(destination.resolve()), result["result"]["destination_path"])
            promoted = promote_contract_resource_effects_context(
                {},
                result,
                tool_input={
                    "artifact_uri": "artifact-sandbox://run-123/output/resume_tailored.md",
                    "destination": str(destination),
                },
                semantic_profile=export_tool.semantic_profile,
            )

            self.assertEqual(str(destination.resolve()), promoted["active_resource"]["focus_path"])
            self.assertEqual("artifact_export", promoted["active_resource"]["operation"])

    def test_rejects_export_when_artifact_uri_points_outside_output_directory(self) -> None:
        from app.agent_runtime.sdk_agents.sandbox import SdkAgentSandboxManager
        from app.agent_runtime.tool_registry import create_artifact_agent_tool_definitions

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sandbox_base = temp_path / "sdk-agent-runs"
            workspace = SdkAgentSandboxManager(base_dir=sandbox_base).prepare_workspace(run_id="run-123")
            draft_file = workspace.work_dir / "draft.md"
            draft_file.write_text("intermediate draft", encoding="utf-8")
            destination = temp_path / "exports" / "draft.md"

            export_tool = create_artifact_agent_tool_definitions(sandbox_base_dir=sandbox_base)[0]
            result = export_tool.handler(
                None,
                artifact_uri="artifact-sandbox://run-123/work/draft.md",
                destination=str(destination),
            )

            self.assertFalse(result["ok"])
            self.assertEqual("ARTIFACT_SOURCE_NOT_EXPORTABLE", result["error"])
            self.assertFalse(destination.exists())

    def test_default_registry_marks_artifact_export_as_high_risk_confirmation_tool(self) -> None:
        from app.agent_runtime.guardrails import AgentToolCallContext, AgentToolNextAction, AgentToolRuntimeGuard
        from app.agent_runtime.tool_registry import ARTIFACT_EXPORT_TOOL, AgentToolRiskLevel, create_default_agent_tool_registry

        registry = create_default_agent_tool_registry()
        definition = registry.get(ARTIFACT_EXPORT_TOOL)

        self.assertIsNotNone(definition)
        self.assertEqual(AgentToolRiskLevel.HIGH, definition.risk_level)
        self.assertTrue(definition.requires_confirmation)

        blocked = AgentToolRuntimeGuard().pre_check(
            AgentToolCallContext(
                stage="maybe_tool",
                tool_name=ARTIFACT_EXPORT_TOOL,
                source_type="agent_chat",
                user_confirmed=False,
            ),
            registry=registry,
        )

        self.assertFalse(blocked.ok)
        self.assertEqual(AgentToolNextAction.REQUEST_USER_CONFIRMATION.value, blocked.next_action)
