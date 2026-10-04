import sys
import os
import shutil
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class SdkAgentRunnerAdapterTest(unittest.TestCase):
    def test_dbx_success_claim_without_runtime_mcp_call_is_failed(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentRuntimeContext, AgentTask
        from app.agent_runtime.sdk_agents.runner_adapter import OpenAISdkAgentRunnerAdapter
        from app.agent_runtime.sdk_agents.schemas import SdkAgentResultEnvelope, SdkToolTraceSummary

        class FakeRunnerClient:
            def run(self, envelope):
                return SdkAgentResultEnvelope(
                    task_id=envelope.task_id,
                    capability_id=envelope.capability_id,
                    subagent_name=envelope.subagent_name,
                    status="succeeded",
                    summary="模型声称已经完成 DBX 查询",
                    trace_summary=SdkToolTraceSummary(
                        tool_call_count=1,
                        operation_refs=["tool:dbx_list_connections#model-claimed"],
                    ),
                )

        adapter = OpenAISdkAgentRunnerAdapter(
            runner_client=FakeRunnerClient(),
            subagent_name="OfferMasterSdkAgent",
            allowed_tools=["mcp.dbx.dbx_list_connections"],
            risk_policy={
                "delegation": {
                    "tools_by_capability": {
                        "agent.dbx_readonly": ["mcp.dbx.dbx_list_connections"]
                    },
                    "subagent_names_by_capability": {"agent.dbx_readonly": "DbxReadOnlyAgent"},
                }
            },
        )

        result = adapter.run_agent_task(
            AgentTask(
                capability_id="agent.dbx_readonly",
                goal="查询 DBX",
                input_payload={"task": "查询 DBX"},
            ),
            AgentRuntimeContext(session_id="s1", run_id="r1", task_id="t1"),
        )

        self.assertEqual("failed", result.status)
        self.assertIn("真实 MCP 工具调用证据", result.summary)
        self.assertEqual(0, result.raw_result["trace_summary"]["tool_call_count"])
        self.assertEqual([], result.raw_result["trace_summary"]["operation_refs"])
        self.assertEqual("RUNTIME_MCP_EVIDENCE_MISSING", result.raw_result["diagnostics"]["error_code"])

    def test_runner_adapter_gives_each_mcp_agent_only_its_own_tools(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentRuntimeContext, AgentTask
        from app.agent_runtime.sdk_agents.runner_adapter import OpenAISdkAgentRunnerAdapter
        from app.agent_runtime.sdk_agents.schemas import SdkAgentResultEnvelope

        envelopes = []

        class FakeRunnerClient:
            def run(self, envelope):
                envelopes.append(envelope)
                return SdkAgentResultEnvelope(
                    task_id=envelope.task_id,
                    capability_id=envelope.capability_id,
                    subagent_name=envelope.subagent_name,
                    status="succeeded",
                    summary="done",
                )

        adapter = OpenAISdkAgentRunnerAdapter(
            runner_client=FakeRunnerClient(),
            subagent_name="OfferMasterSdkAgent",
            allowed_tools=["skill_list", "mcp.chrome.list_pages", "mcp.dbx.dbx_list_connections"],
            risk_policy={
                "delegation": {
                    "tools_by_capability": {
                        "agent.google_chrome": ["mcp.chrome.list_pages"],
                        "agent.dbx_readonly": ["mcp.dbx.dbx_list_connections"],
                    },
                    "subagent_names_by_capability": {
                        "agent.google_chrome": "GoogleChromeAgent",
                        "agent.dbx_readonly": "DbxReadOnlyAgent",
                    },
                }
            },
        )

        for index, capability_id in enumerate(("agent.google_chrome", "agent.dbx_readonly"), start=1):
            adapter.run_agent_task(
                AgentTask(capability_id=capability_id, goal="run task", input_payload={"task": "run task"}),
                AgentRuntimeContext(session_id="s1", run_id="r1", task_id=f"t{index}"),
            )

        self.assertEqual(["mcp.chrome.list_pages"], envelopes[0].allowed_tools)
        self.assertEqual(["mcp.dbx.dbx_list_connections"], envelopes[1].allowed_tools)
        self.assertEqual("GoogleChromeAgent", envelopes[0].subagent_name)
        self.assertEqual("DbxReadOnlyAgent", envelopes[1].subagent_name)

    def test_chrome_approval_event_uses_registry_tool_name_and_never_includes_argument_values(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentRuntimeContext, AgentTask
        from app.agent_runtime.sdk_agents.runner_adapter import OpenAISdkAgentRunnerAdapter, _result_from_interruption
        from app.agent_runtime.sdk_agents.schemas import SdkAgentResultEnvelope, SdkAgentTaskEnvelope

        envelope = SdkAgentTaskEnvelope(
            task_id="t1",
            trace_id="r1",
            capability_id="agent.google_chrome",
            subagent_name="GoogleChromeAgent",
            goal="Click the requested control.",
            allowed_tools=["mcp.chrome.click"],
        )
        interruption = types.SimpleNamespace(
            tool_name="mcp_chrome_click",
            arguments='{"selector":"#private-selector","text":"private input"}',
            call_id="call-1",
        )
        approval_result = _result_from_interruption(envelope, interruption)
        self.assertEqual("mcp.chrome.click", approval_result.approval_request.tool_name)

        events = []

        class FakeRunnerClient:
            def run(self, call_envelope):
                return SdkAgentResultEnvelope(
                    task_id=call_envelope.task_id,
                    capability_id=call_envelope.capability_id,
                    subagent_name=call_envelope.subagent_name,
                    status="needs_approval",
                    summary="waiting",
                    approval_request=approval_result.approval_request,
                )

        adapter = OpenAISdkAgentRunnerAdapter(
            runner_client=FakeRunnerClient(),
            subagent_name="OfferMasterSdkAgent",
            allowed_tools=["mcp.chrome.click"],
            risk_policy={
                "delegation": {
                    "tools_by_capability": {"agent.google_chrome": ["mcp.chrome.click"]},
                    "subagent_names_by_capability": {"agent.google_chrome": "GoogleChromeAgent"},
                }
            },
        )
        adapter.run_agent_task(
            AgentTask(capability_id="agent.google_chrome", goal="Click the requested control.", input_payload={"task": "Click the requested control."}),
            AgentRuntimeContext(
                session_id="s1",
                run_id="r1",
                task_id="t1",
                metadata={"agent_run_id": "a1", "delegation_id": "delegation:r1:child-1"},
                event_sink=events.append,
            ),
        )

        self.assertEqual("subagent_tool_waiting_approval", events[0]["event_type"])
        self.assertEqual("mcp.chrome.click", events[0]["tool_name"])
        self.assertEqual(["selector", "text"], events[0]["tool_input_keys"])
        self.assertNotIn("#private-selector", str(events[0]))
        self.assertNotIn("private input", str(events[0]))
        self.assertEqual("GoogleChromeAgent", events[0]["parent_agent_name"])
        self.assertEqual("delegation:r1:child-1", events[0]["delegation_id"])

    def test_runner_instructions_tell_child_agent_to_progressively_load_skills(self) -> None:
        from app.agent_runtime.sdk_agents.runner_adapter import _build_runner_instructions
        from app.agent_runtime.sdk_agents.schemas import SdkAgentTaskEnvelope

        instructions = _build_runner_instructions(
            SdkAgentTaskEnvelope(
                task_id="task-1",
                trace_id="trace-1",
                capability_id="skill.filesystem",
                goal="复制这个文件",
                input_payload={"user_task": "复制这个文件"},
                subagent_name="OfferMasterSdkAgent",
                allowed_tools=["skill_list", "skill_read", "filesystem.copy_file"],
            )
        )

        self.assertIn("skill_list", instructions)
        self.assertIn("skill_read", instructions)
        self.assertIn("progressively load", instructions)

    def test_runner_adapter_passes_task_envelope_to_runner_client(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentRuntimeContext, AgentTask
        from app.agent_runtime.sdk_agents.runner_adapter import OpenAISdkAgentRunnerAdapter
        from app.agent_runtime.sdk_agents.schemas import SdkAgentResultEnvelope, SdkToolTraceSummary
        from app.agent_runtime.sdk_agents.telemetry import SDK_AGENT_TELEMETRY_SCHEMA_VERSION

        seen_envelopes = []

        class FakeRunnerClient:
            def run(self, envelope):
                seen_envelopes.append(envelope)
                return SdkAgentResultEnvelope(
                    task_id=envelope.task_id,
                    capability_id=envelope.capability_id,
                    subagent_name=envelope.subagent_name,
                    status="succeeded",
                    summary="SDK runner 已完成简历改写。",
                    observation="新版简历已生成。",
                    diagnostics={"confidence": 0.88},
                    trace_summary=SdkToolTraceSummary(tool_call_count=2, retry_count=1),
                    raw_trace_ref="trace-1",
                )

        adapter = OpenAISdkAgentRunnerAdapter(
            runner_client=FakeRunnerClient(),
            subagent_name="ResumeTailorAgent",
            allowed_tools=["resume.analyze", "resume.rewrite"],
            risk_policy={"mutation_tools": "approval_required"},
            max_turns=6,
        )

        result = adapter.run_agent_task(
            AgentTask(
                capability_id="resume.tailor",
                goal="根据 JD 改简历",
                input_payload={"resume_text": "Java 项目", "job_description": "Java 后端"},
            ),
            AgentRuntimeContext(session_id="session-1", run_id="run-1", task_id="run-1:tool-1"),
        )

        self.assertEqual("succeeded", result.status)
        self.assertEqual("SDK runner 已完成简历改写。", result.summary)
        self.assertEqual(1, len(seen_envelopes))
        envelope = seen_envelopes[0]
        self.assertEqual("run-1:tool-1", envelope.task_id)
        self.assertEqual("resume.tailor", envelope.capability_id)
        self.assertEqual("ResumeTailorAgent", envelope.subagent_name)
        self.assertEqual(["resume.analyze", "resume.rewrite"], envelope.allowed_tools)
        self.assertEqual({"mutation_tools": "approval_required"}, envelope.risk_policy)
        self.assertEqual(6, envelope.max_turns)
        self.assertEqual("trace-1", result.raw_result["raw_trace_ref"])
        self.assertEqual(
            {
                "schema_version": SDK_AGENT_TELEMETRY_SCHEMA_VERSION,
                "subagent_name": "ResumeTailorAgent",
                "capability_id": "resume.tailor",
                "status": "succeeded",
                "allowed_tool_count": 2,
                "tool_call_count": 2,
                "retry_count": 1,
                "approval_required": False,
                "operation_ref_count": 0,
                "sandbox": {"enabled": False, "artifact_count": 0},
                "delegation": {"runtime_retained_tool_count": 0},
            },
            result.raw_result["metadata"]["telemetry"],
        )

    def test_runner_adapter_preserves_sdk_resource_effects_for_runtime_context(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentRuntimeContext, AgentTask
        from app.agent_runtime.sdk_agents.runner_adapter import OpenAISdkAgentRunnerAdapter

        class FakeRunnerClient:
            def run(self, envelope):
                return {
                    "status": "succeeded",
                    "summary": "SDK 子 Agent 已导出 PDF。",
                    "resource_effects": [
                        {
                            "resource_type": "file",
                            "action": "created",
                            "operation": "export_pdf",
                            "source_path": "input/resume.tex",
                            "target_path": "output/resume.pdf",
                            "focus_path": "output/resume.pdf",
                            "aliases": ["刚才导出的 PDF", "新生成的简历"],
                        }
                    ],
                }

        result = OpenAISdkAgentRunnerAdapter(
            runner_client=FakeRunnerClient(),
            subagent_name="ResumeExportAgent",
        ).run_agent_task(
            AgentTask(
                capability_id="resume.export_pdf",
                goal="把简历导出为 PDF",
                input_payload={"resume_path": "C:/简历/resume.tex"},
            ),
            AgentRuntimeContext(session_id="session-1", run_id="run-1", task_id="run-1:tool-1"),
        )

        self.assertEqual("succeeded", result.status)
        self.assertEqual("output/resume.pdf", result.raw_result["resource_effects"][0]["focus_path"])
        self.assertEqual(["刚才导出的 PDF", "新生成的简历"], result.raw_result["resource_effects"][0]["aliases"])

    def test_runner_adapter_prepares_artifact_sandbox_and_collects_outputs(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentRuntimeContext, AgentTask
        from app.agent_runtime.sdk_agents.runner_adapter import OpenAISdkAgentRunnerAdapter
        from app.agent_runtime.sdk_agents.sandbox import SdkAgentSandboxManager

        tmpdir = Path(tempfile.mkdtemp(prefix="sdk-runner-sandbox-test-"))
        self.addCleanup(lambda: shutil.rmtree(tmpdir, ignore_errors=True))
        resume = tmpdir / "resume.md"
        jd = tmpdir / "jd.txt"
        resume.write_text("old resume", encoding="utf-8")
        jd.write_text("java backend jd", encoding="utf-8")
        calls = []

        class FakeToolGateway:
            def build_tools(self, tool_names, context):
                sandbox = context.metadata["sdk_agent_sandbox"]
                output_path = Path(sandbox["output_dir"]) / "resume_tailored.md"
                output_path.write_text("tailored resume", encoding="utf-8")
                calls.append({"tool_names": list(tool_names), "sandbox": sandbox})
                return ["sdk-tool"]

        class FakeRunnerClient:
            def run(self, envelope, *, tools=None):
                calls.append({"envelope": envelope, "tools": tools})
                return {"status": "succeeded", "summary": "SDK 子 Agent 已生成新版简历。"}

        result = OpenAISdkAgentRunnerAdapter(
            runner_client=FakeRunnerClient(),
            subagent_name="ResumeTailorAgent",
            allowed_tools=["filesystem.read_file", "filesystem.write_text"],
            tool_gateway=FakeToolGateway(),
            sandbox_manager=SdkAgentSandboxManager(base_dir=tmpdir / "runs"),
        ).run_agent_task(
            AgentTask(
                capability_id="resume.tailor",
                goal="根据 JD 改简历",
                input_payload={
                    "resume_path": str(resume),
                    "job_description_path": str(jd),
                    "language": "zh-CN",
                },
            ),
            AgentRuntimeContext(session_id="session-1", run_id="run-1", task_id="run-1:tool-1"),
        )

        envelope = calls[1]["envelope"]
        envelope_json = envelope.model_dump_json()
        self.assertEqual("succeeded", result.status)
        self.assertEqual("input/resume.md", envelope.input_payload["resume_logical_path"])
        self.assertEqual("input/jd.txt", envelope.input_payload["job_description_logical_path"])
        self.assertEqual(["input/jd.txt", "input/resume.md"], envelope.input_payload["sandbox_input_paths"])
        self.assertNotIn("resume_path", envelope.input_payload)
        self.assertNotIn("job_description_path", envelope.input_payload)
        self.assertNotIn(str(resume), envelope_json)
        self.assertNotIn(str(jd), envelope_json)
        self.assertEqual(["filesystem.read_file", "filesystem.write_text"], calls[0]["tool_names"])
        self.assertEqual("old resume", (Path(calls[0]["sandbox"]["input_dir"]) / "resume.md").read_text(encoding="utf-8"))
        self.assertEqual(
            [{"logical_path": "output/resume_tailored.md", "size_bytes": len("tailored resume"), "kind": "md"}],
            result.raw_result["metadata"]["sandbox"]["artifacts"],
        )
        self.assertEqual(
            {"enabled": True, "run_id": "run-1-tool-1", "artifact_count": 1},
            result.raw_result["metadata"]["telemetry"]["sandbox"],
        )
        self.assertIn("artifact-sandbox://run-1-tool-1/output/resume_tailored.md", result.raw_result["operation_refs"])
        self.assertEqual(
            "artifact-sandbox://run-1-tool-1/output/resume_tailored.md",
            result.raw_result["resource_effects"][0]["focus_path"],
        )
        self.assertEqual("sdk_agent_artifact", result.raw_result["resource_effects"][0]["operation"])

    def test_runner_adapter_mounts_active_file_context_for_filesystem_skill(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentRuntimeContext, AgentTask
        from app.agent_runtime.sdk_agents.runner_adapter import OpenAISdkAgentRunnerAdapter
        from app.agent_runtime.sdk_agents.sandbox import SdkAgentSandboxManager

        tmpdir = Path(tempfile.mkdtemp(prefix="sdk-runner-active-file-test-"))
        self.addCleanup(lambda: shutil.rmtree(tmpdir, ignore_errors=True))
        resume = tmpdir / "刘汉卿-后端开发-AI-Agent.tex"
        resume.write_text("resume body", encoding="utf-8")
        calls = []

        class FakeToolGateway:
            def build_tools(self, tool_names, context):
                sandbox = context.metadata["sdk_agent_sandbox"]
                calls.append({"tool_names": list(tool_names), "sandbox": sandbox})
                return ["sdk-tool"]

        class FakeRunnerClient:
            def run(self, envelope, *, tools=None):
                calls.append({"envelope": envelope, "tools": tools})
                return {"status": "succeeded", "summary": "SDK 子 Agent 已处理文件。"}

        result = OpenAISdkAgentRunnerAdapter(
            runner_client=FakeRunnerClient(),
            subagent_name="OfferMasterSdkAgent",
            allowed_tools=["skill_list", "skill_read", "filesystem.read_file", "filesystem.copy_file"],
            tool_gateway=FakeToolGateway(),
            sandbox_manager=SdkAgentSandboxManager(base_dir=tmpdir / "runs"),
        ).run_agent_task(
            AgentTask(
                capability_id="skill.filesystem",
                goal="复制这个文件",
                input_payload={
                    "user_task": "把刚才这个文件复制一份，名字不要冲突",
                    "context_metadata": {"active_file": {"path": str(resume), "label": "刚才这个文件"}},
                },
            ),
            AgentRuntimeContext(session_id="session-1", run_id="run-1", task_id="run-1:tool-1"),
        )

        envelope = calls[1]["envelope"]
        envelope_json = envelope.model_dump_json()

        self.assertEqual("succeeded", result.status)
        self.assertEqual("input/刘汉卿-后端开发-AI-Agent.tex", envelope.input_payload["active_file_logical_path"])
        self.assertEqual("input/刘汉卿-后端开发-AI-Agent.tex", envelope.input_payload["context_metadata"]["active_file"]["path"])
        self.assertNotIn(str(resume), envelope_json)
        self.assertEqual("resume body", (Path(calls[0]["sandbox"]["input_dir"]) / resume.name).read_text(encoding="utf-8"))

    def test_runner_adapter_accepts_dict_runner_output(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentRuntimeContext, AgentTask
        from app.agent_runtime.sdk_agents.runner_adapter import OpenAISdkAgentRunnerAdapter

        class FakeRunnerClient:
            def run(self, envelope):
                return {
                    "task_id": envelope.task_id,
                    "capability_id": envelope.capability_id,
                    "subagent_name": envelope.subagent_name,
                    "status": "succeeded",
                    "summary": "以 dict 形式返回也可以标准化。",
                    "diagnostics": {"confidence": 0.75},
                }

        result = OpenAISdkAgentRunnerAdapter(
            runner_client=FakeRunnerClient(),
            subagent_name="ResumeTailorAgent",
        ).run_agent_task(
            AgentTask(
                capability_id="resume.tailor",
                goal="根据 JD 改简历",
                input_payload={"resume_text": "Java 项目", "job_description": "Java 后端"},
            ),
            AgentRuntimeContext(session_id="session-1", run_id="run-1", task_id="run-1:tool-1"),
        )

        self.assertEqual("succeeded", result.status)
        self.assertEqual("以 dict 形式返回也可以标准化。", result.summary)
        self.assertIn("confidence=0.75", result.observation)

    def test_runner_adapter_normalizes_structured_operation_refs_from_model_output(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentRuntimeContext, AgentTask
        from app.agent_runtime.sdk_agents.runner_adapter import OpenAISdkAgentRunnerAdapter

        class FakeRunnerClient:
            def run(self, envelope):
                return {
                    "status": "succeeded",
                    "summary": "DBX 查询已完成。",
                    "operation_refs": [
                        {"tool": "mcp.dbx.dbx_list_connections", "call_id": "call-1"},
                        {"operation_ref": "dbx-query-1"},
                    ],
                    "trace_summary": {
                        "tool_call_count": 2,
                        "operation_refs": [{"tool": "mcp.dbx.dbx_execute_query", "call_id": "call-2"}],
                    },
                }

        result = OpenAISdkAgentRunnerAdapter(
            runner_client=FakeRunnerClient(),
            subagent_name="DbxReadOnlyAgent",
        ).run_agent_task(
            AgentTask(
                capability_id="resume.tailor",
                goal="整理简历",
                input_payload={"task": "整理简历"},
            ),
            AgentRuntimeContext(session_id="session-1", run_id="run-1", task_id="run-1:tool-1"),
        )

        self.assertEqual("succeeded", result.status)
        self.assertEqual(
            ["tool:mcp.dbx.dbx_list_connections#call-1", "dbx-query-1"],
            result.raw_result["operation_refs"],
        )
        self.assertEqual(
            ["tool:mcp.dbx.dbx_execute_query#call-2"],
            result.raw_result["trace_summary"]["operation_refs"],
        )

    def test_runner_adapter_surfaces_approval_interruption_as_user_action(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentRuntimeContext, AgentTask
        from app.agent_runtime.sdk_agents.runner_adapter import OpenAISdkAgentRunnerAdapter
        from app.agent_runtime.sdk_agents.schemas import SdkAgentResultEnvelope, SdkToolApprovalRequest

        class FakeRunnerClient:
            def run(self, envelope):
                return SdkAgentResultEnvelope(
                    task_id=envelope.task_id,
                    capability_id=envelope.capability_id,
                    subagent_name=envelope.subagent_name,
                    status="needs_approval",
                    summary="等待用户确认删除文件。",
                    approval_request=SdkToolApprovalRequest(
                        approval_type="tool_call",
                        tool_name="filesystem.delete_path",
                        tool_input={"path": "data/cache/tmp.json"},
                        reason="删除文件属于高风险动作。",
                    ),
                )

        result = OpenAISdkAgentRunnerAdapter(
            runner_client=FakeRunnerClient(),
            subagent_name="FileAnalysisAgent",
        ).run_agent_task(
            AgentTask(
                capability_id="agent.file_analysis",
                goal="分析并清理缓存文件",
                input_payload={"path": "data/cache"},
            ),
            AgentRuntimeContext(session_id="session-1", run_id="run-1", task_id="run-1:tool-1"),
        )

        self.assertEqual("blocked", result.status)
        self.assertTrue(result.requires_user_action)
        self.assertEqual(["Approve or reject filesystem.delete_path"], result.next_actions)

    def test_openai_agents_runner_client_uses_sdk_runner_and_configured_env(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentRuntimeContext, AgentTask
        from app.agent_runtime.sdk_agents.runner_adapter import OpenAIAgentsSdkRunnerClient
        from app.agent_runtime.sdk_agents.schemas import SdkAgentTaskEnvelope

        calls = []

        class FakeAgent:
            def __init__(self, **payload):
                self.payload = payload

        class FakeRunner:
            @staticmethod
            def run_sync(agent, input, *, max_turns):
                calls.append(
                    {
                        "agent": agent,
                        "input": input,
                        "max_turns": max_turns,
                        "api_key": os.environ.get("OPENAI_API_KEY"),
                        "base_url": os.environ.get("OPENAI_BASE_URL"),
                        "tracing_disabled": os.environ.get("OPENAI_AGENTS_DISABLE_TRACING"),
                        "trace_sensitive_data": os.environ.get("OPENAI_AGENTS_TRACE_INCLUDE_SENSITIVE_DATA"),
                    }
                )
                return types.SimpleNamespace(final_output={"status": "succeeded", "summary": "SDK Runner 返回成功。"})

        fake_agents_module = types.SimpleNamespace(Agent=FakeAgent, Runner=FakeRunner)
        envelope = SdkAgentTaskEnvelope.from_agent_task(
            AgentTask(
                capability_id="resume.tailor",
                goal="根据 JD 改简历",
                input_payload={"resume_text": "Java 项目", "job_description": "Java 后端"},
            ),
            AgentRuntimeContext(session_id="session-1", run_id="run-1", task_id="run-1:tool-1"),
            subagent_name="ResumeTailorAgent",
            max_turns=5,
        )

        with patch.dict(sys.modules, {"agents": fake_agents_module}), patch.dict(os.environ, {}, clear=True):
            result = OpenAIAgentsSdkRunnerClient(
                model="gpt-test",
                api_key="sk-test",
                base_url="https://api.openai.example/v1",
            ).run(envelope)

            self.assertNotIn("OPENAI_API_KEY", os.environ)
            self.assertNotIn("OPENAI_BASE_URL", os.environ)
            self.assertNotIn("OPENAI_AGENTS_DISABLE_TRACING", os.environ)
            self.assertNotIn("OPENAI_AGENTS_TRACE_INCLUDE_SENSITIVE_DATA", os.environ)

        self.assertEqual("succeeded", result.status)
        self.assertEqual("SDK Runner 返回成功。", result.summary)
        self.assertEqual(1, len(calls))
        self.assertEqual("ResumeTailorAgent", calls[0]["agent"].payload["name"])
        self.assertEqual("gpt-test", calls[0]["agent"].payload["model"])
        self.assertIn('"capability_id": "resume.tailor"', calls[0]["input"])
        self.assertEqual(5, calls[0]["max_turns"])
        self.assertEqual("sk-test", calls[0]["api_key"])
        self.assertEqual("https://api.openai.example/v1", calls[0]["base_url"])
        self.assertEqual("true", calls[0]["tracing_disabled"])
        self.assertEqual("false", calls[0]["trace_sensitive_data"])

    def test_openai_agents_runner_client_passes_timeout_through_run_config(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentRuntimeContext, AgentTask
        from app.agent_runtime.sdk_agents.runner_adapter import OpenAIAgentsSdkRunnerClient
        from app.agent_runtime.sdk_agents.schemas import SdkAgentTaskEnvelope

        calls = []

        class FakeAgent:
            def __init__(self, **payload):
                self.payload = payload

        class FakeModelSettings:
            def __init__(self, **payload):
                self.payload = payload
                self.timeout = payload.get("timeout")

        class FakeRunConfig:
            def __init__(self, **payload):
                self.payload = payload
                self.model_settings = payload.get("model_settings")

        class FakeRunner:
            @staticmethod
            def run_sync(agent, input, *, max_turns, run_config=None):
                calls.append({"agent": agent, "input": input, "max_turns": max_turns, "run_config": run_config})
                return types.SimpleNamespace(final_output={"status": "succeeded", "summary": "SDK Runner 返回成功。"})

        envelope = SdkAgentTaskEnvelope.from_agent_task(
            AgentTask(
                capability_id="resume.tailor",
                goal="根据 JD 改简历",
                input_payload={"resume_text": "Java 项目", "job_description": "Java 后端"},
            ),
            AgentRuntimeContext(session_id="session-1", run_id="run-1", task_id="run-1:tool-1"),
            subagent_name="ResumeTailorAgent",
            max_turns=5,
        )

        fake_agents_module = types.SimpleNamespace(
            Agent=FakeAgent,
            Runner=FakeRunner,
            RunConfig=FakeRunConfig,
            ModelSettings=FakeModelSettings,
        )
        with patch.dict(sys.modules, {"agents": fake_agents_module}):
            result = OpenAIAgentsSdkRunnerClient(model="gpt-test", timeout_seconds=12.5).run(envelope)

        self.assertEqual("succeeded", result.status)
        self.assertIsInstance(calls[0]["run_config"], FakeRunConfig)
        self.assertIsInstance(calls[0]["run_config"].model_settings, FakeModelSettings)
        self.assertEqual(12.5, calls[0]["run_config"].model_settings.timeout)

    def test_openai_agents_runner_client_passes_timeout_through_resume_run_config(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentRuntimeContext, AgentTask
        from app.agent_runtime.sdk_agents.runner_adapter import OpenAIAgentsSdkRunnerClient
        from app.agent_runtime.sdk_agents.schemas import SdkAgentTaskEnvelope

        calls = []

        class FakeAgent:
            def __init__(self, **payload):
                self.payload = payload

        class FakeModelSettings:
            def __init__(self, **payload):
                self.payload = payload
                self.timeout = payload.get("timeout")

        class FakeRunConfig:
            def __init__(self, **payload):
                self.payload = payload
                self.model_settings = payload.get("model_settings")

        class FakeRunState:
            def approve(self, approval_request=None):
                self.approval_request = approval_request

        class FakeRunner:
            @staticmethod
            def run_sync(agent, input, *, max_turns, run_config=None):
                calls.append({"agent": agent, "input": input, "max_turns": max_turns, "run_config": run_config})
                return types.SimpleNamespace(final_output={"status": "succeeded", "summary": "SDK RunState 已恢复。"})

        envelope = SdkAgentTaskEnvelope.from_agent_task(
            AgentTask(
                capability_id="agent.file_analysis",
                goal="分析并清理缓存文件",
                input_payload={"path": "runtime"},
            ),
            AgentRuntimeContext(session_id="session-1", run_id="run-1", task_id="run-1:tool-1"),
            subagent_name="FileAnalysisAgent",
            max_turns=5,
        )

        fake_agents_module = types.SimpleNamespace(
            Agent=FakeAgent,
            Runner=FakeRunner,
            RunConfig=FakeRunConfig,
            ModelSettings=FakeModelSettings,
        )
        with patch.dict(sys.modules, {"agents": fake_agents_module}):
            result = OpenAIAgentsSdkRunnerClient(model="gpt-test", timeout_seconds=8).resume(
                envelope,
                FakeRunState(),
                approved=True,
                approval_request={"tool_name": "filesystem.delete_path", "tool_input": {"path": "runtime/tmp.json"}},
            )

        self.assertEqual("succeeded", result.status)
        self.assertIsInstance(calls[0]["run_config"], FakeRunConfig)
        self.assertIsInstance(calls[0]["run_config"].model_settings, FakeModelSettings)
        self.assertEqual(8, calls[0]["run_config"].model_settings.timeout)

    def test_openai_agents_runner_client_attaches_configured_tools(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentRuntimeContext, AgentTask
        from app.agent_runtime.sdk_agents.runner_adapter import OpenAIAgentsSdkRunnerClient
        from app.agent_runtime.sdk_agents.schemas import SdkAgentTaskEnvelope

        calls = []
        sdk_tool = object()

        class FakeAgent:
            def __init__(self, **payload):
                self.payload = payload

        class FakeRunner:
            @staticmethod
            def run_sync(agent, input, *, max_turns):
                calls.append({"agent": agent, "input": input, "max_turns": max_turns})
                return types.SimpleNamespace(final_output={"status": "succeeded", "summary": "工具已挂载。"})

        envelope = SdkAgentTaskEnvelope.from_agent_task(
            AgentTask(
                capability_id="external.web_search",
                goal="查腾讯校招官网",
                input_payload={"query": "腾讯 校招 官网"},
            ),
            AgentRuntimeContext(session_id="session-1", run_id="run-1", task_id="run-1:tool-1"),
            subagent_name="WebResearchAgent",
            allowed_tools=["external.web_search"],
            max_turns=5,
        )

        with patch.dict(sys.modules, {"agents": types.SimpleNamespace(Agent=FakeAgent, Runner=FakeRunner)}):
            result = OpenAIAgentsSdkRunnerClient(model="gpt-test").run(envelope, tools=[sdk_tool])

        self.assertEqual("succeeded", result.status)
        self.assertEqual([sdk_tool], calls[0]["agent"].payload["tools"])

    def test_openai_agents_runner_client_serializes_run_state_on_approval_interruption(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentRuntimeContext, AgentTask
        from app.agent_runtime.sdk_agents.runner_adapter import OpenAIAgentsSdkRunnerClient
        from app.agent_runtime.sdk_agents.schemas import SdkAgentTaskEnvelope

        class FakeAgent:
            def __init__(self, **payload):
                self.payload = payload

        class FakeRunState:
            def to_json(self):
                return {"cursor": "sdk-state-1", "pending": ["filesystem.delete_path"]}

        class FakeRunnerResult:
            interruptions = [types.SimpleNamespace(tool_name="filesystem.delete_path", arguments={"path": "runtime/tmp.json"})]

            def to_state(self):
                return FakeRunState()

        class FakeRunner:
            @staticmethod
            def run_sync(agent, input, *, max_turns):
                return FakeRunnerResult()

        envelope = SdkAgentTaskEnvelope.from_agent_task(
            AgentTask(
                capability_id="agent.file_analysis",
                goal="分析并清理缓存文件",
                input_payload={"path": "runtime"},
            ),
            AgentRuntimeContext(session_id="session-1", run_id="run-1", task_id="run-1:tool-1"),
            subagent_name="FileAnalysisAgent",
            allowed_tools=["filesystem.delete_path"],
            max_turns=5,
        )

        with patch.dict(sys.modules, {"agents": types.SimpleNamespace(Agent=FakeAgent, Runner=FakeRunner)}):
            result = OpenAIAgentsSdkRunnerClient(model="gpt-test").run(envelope)

        self.assertEqual("needs_approval", result.status)
        self.assertEqual({"format": "json", "value": {"cursor": "sdk-state-1", "pending": ["filesystem.delete_path"]}}, result.metadata["run_state"])
        self.assertEqual("agent.file_analysis", result.metadata["task_envelope"]["capability_id"])
        self.assertEqual("filesystem.delete_path", result.approval_request.tool_name)

    def test_runner_adapter_resumes_serialized_run_state_with_approval_decision(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentRuntimeContext
        from app.agent_runtime.sdk_agents.runner_adapter import OpenAISdkAgentRunnerAdapter

        resume_calls = []

        class FakeRunnerClient:
            def run(self, envelope):
                raise AssertionError("resume test should not start a new SDK run")

            def resume(self, envelope, run_state, *, approved, approval_request, tools=None):
                resume_calls.append(
                    {
                        "envelope": envelope,
                        "run_state": run_state,
                        "approved": approved,
                        "approval_request": approval_request,
                        "tools": tools,
                    }
                )
                return {"status": "succeeded", "summary": "SDK 子 Agent 已根据审批继续执行。"}

        approval_payload = {
            "sdk_agent_approval": True,
            "approval_request": {
                "tool_name": "filesystem.delete_path",
                "tool_input": {"path": "runtime/tmp.json"},
                "reason": "删除需要审批。",
            },
            "sdk_agent_result": {
                "metadata": {
                    "run_state": {"format": "json", "value": {"cursor": "sdk-state-1"}},
                    "task_envelope": {
                        "task_id": "run-1:tool-1",
                        "trace_id": "run-1",
                        "capability_id": "agent.file_analysis",
                        "subagent_name": "FileAnalysisAgent",
                        "goal": "分析并清理缓存文件",
                        "input_payload": {"path": "runtime"},
                        "allowed_tools": ["filesystem.delete_path"],
                        "risk_policy": {"mutation_tools": "approval_required"},
                        "constraints": [],
                        "expected_output": [],
                        "max_turns": 5,
                        "context_refs": {"session_id": "session-1", "workflow_run_id": "run-1"},
                        "metadata": {"agent_run_id": "agent-run-1"},
                    },
                }
            },
        }

        result = OpenAISdkAgentRunnerAdapter(
            runner_client=FakeRunnerClient(),
            subagent_name="FileAnalysisAgent",
            allowed_tools=["filesystem.delete_path"],
        ).resume_agent_task_after_approval(
            approval_payload,
            approved=True,
            context=AgentRuntimeContext(
                session_id="session-1",
                run_id="run-1",
                task_id="run-1:sdk-resume-1",
                metadata={"decision_reason": "allow one cleanup"},
            ),
        )

        self.assertEqual("succeeded", result.status)
        self.assertEqual("SDK 子 Agent 已根据审批继续执行。", result.summary)
        self.assertEqual(1, len(resume_calls))
        self.assertEqual("agent.file_analysis", resume_calls[0]["envelope"].capability_id)
        self.assertEqual({"format": "json", "value": {"cursor": "sdk-state-1"}}, resume_calls[0]["run_state"])
        self.assertTrue(resume_calls[0]["approved"])
        self.assertEqual("filesystem.delete_path", resume_calls[0]["approval_request"]["tool_name"])
        self.assertTrue(resume_calls[0]["approval_request"]["approved"])
        self.assertEqual("allow one cleanup", resume_calls[0]["approval_request"]["decision_reason"])

    def test_openai_agents_runner_client_resumes_real_run_state_shape(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentRuntimeContext, AgentTask
        from app.agent_runtime.sdk_agents.runner_adapter import OpenAIAgentsSdkRunnerClient
        from app.agent_runtime.sdk_agents.schemas import SdkAgentTaskEnvelope

        calls = []

        class FakeAgent:
            def __init__(self, **payload):
                self.payload = payload

        class FakeApprovalItem:
            tool_name = "filesystem.delete_path"
            arguments = {"path": "runtime/tmp.json"}

        class FakeRunState:
            def __init__(self, agent, state_json):
                self.agent = agent
                self.state_json = state_json
                self.approved_item = None

            @classmethod
            async def from_json(cls, initial_agent, state_json):
                return cls(initial_agent, state_json)

            def get_interruptions(self):
                return [FakeApprovalItem()]

            def approve(self, approval_item, always_approve=False):
                self.approved_item = approval_item

        class FakeRunner:
            @staticmethod
            def run_sync(agent, input, *, max_turns):
                calls.append({"agent": agent, "input": input, "max_turns": max_turns})
                return types.SimpleNamespace(final_output={"status": "succeeded", "summary": "SDK RunState 已恢复。"})

        envelope = SdkAgentTaskEnvelope.from_agent_task(
            AgentTask(
                capability_id="agent.file_analysis",
                goal="分析并清理缓存文件",
                input_payload={"path": "runtime"},
            ),
            AgentRuntimeContext(session_id="session-1", run_id="run-1", task_id="run-1:tool-1"),
            subagent_name="FileAnalysisAgent",
            max_turns=5,
        )

        with patch.dict(sys.modules, {"agents": types.SimpleNamespace(Agent=FakeAgent, Runner=FakeRunner, RunState=FakeRunState)}):
            result = OpenAIAgentsSdkRunnerClient(model="gpt-test").resume(
                envelope,
                {"format": "json", "value": {"cursor": "sdk-state-1"}},
                approved=True,
                approval_request={"tool_name": "filesystem.delete_path", "tool_input": {"path": "runtime/tmp.json"}},
            )

        self.assertEqual("succeeded", result.status)
        self.assertEqual("SDK RunState 已恢复。", result.summary)
        self.assertEqual(1, len(calls))
        self.assertIsInstance(calls[0]["input"], FakeRunState)
        self.assertEqual({"cursor": "sdk-state-1"}, calls[0]["input"].state_json)
        self.assertIsInstance(calls[0]["input"].approved_item, FakeApprovalItem)

    def test_openai_agents_runner_client_rejects_real_run_state_with_reason(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentRuntimeContext, AgentTask
        from app.agent_runtime.sdk_agents.runner_adapter import OpenAIAgentsSdkRunnerClient
        from app.agent_runtime.sdk_agents.schemas import SdkAgentTaskEnvelope

        calls = []

        class FakeAgent:
            def __init__(self, **payload):
                self.payload = payload

        class FakeApprovalItem:
            tool_name = "filesystem.delete_path"
            arguments = {"path": "runtime/tmp.json"}

        class FakeRunState:
            def __init__(self, agent, state_json):
                self.agent = agent
                self.state_json = state_json
                self.rejected = None

            @classmethod
            async def from_json(cls, initial_agent, state_json):
                return cls(initial_agent, state_json)

            def get_interruptions(self):
                return [FakeApprovalItem()]

            def reject(self, approval_item, always_reject=False, *, rejection_message=None):
                self.rejected = {"item": approval_item, "message": rejection_message}

        class FakeRunner:
            @staticmethod
            def run_sync(agent, input, *, max_turns):
                calls.append({"agent": agent, "input": input, "max_turns": max_turns})
                return types.SimpleNamespace(final_output={"status": "succeeded", "summary": "SDK RunState 已处理拒绝。"})

        envelope = SdkAgentTaskEnvelope.from_agent_task(
            AgentTask(capability_id="agent.file_analysis", goal="分析并清理缓存文件", input_payload={"path": "runtime"}),
            AgentRuntimeContext(session_id="session-1", run_id="run-1", task_id="run-1:tool-1"),
            subagent_name="FileAnalysisAgent",
            max_turns=5,
        )

        with patch.dict(sys.modules, {"agents": types.SimpleNamespace(Agent=FakeAgent, Runner=FakeRunner, RunState=FakeRunState)}):
            result = OpenAIAgentsSdkRunnerClient(model="gpt-test").resume(
                envelope,
                {"format": "json", "value": {"cursor": "sdk-state-1"}},
                approved=False,
                approval_request={
                    "tool_name": "filesystem.delete_path",
                    "tool_input": {"path": "runtime/tmp.json"},
                    "decision_reason": "do not delete files",
                },
            )

        self.assertEqual("succeeded", result.status)
        self.assertIsInstance(calls[0]["input"].rejected["item"], FakeApprovalItem)
        self.assertEqual("do not delete files", calls[0]["input"].rejected["message"])

    def test_openai_agents_runner_client_parses_string_arguments_from_interruption(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentRuntimeContext, AgentTask
        from app.agent_runtime.sdk_agents.runner_adapter import OpenAIAgentsSdkRunnerClient
        from app.agent_runtime.sdk_agents.schemas import SdkAgentTaskEnvelope

        class FakeAgent:
            def __init__(self, **payload):
                self.payload = payload

        class FakeRunner:
            @staticmethod
            def run_sync(agent, input, *, max_turns):
                interruption = types.SimpleNamespace(
                    tool_name="filesystem.delete_path",
                    call_id="call-delete-1",
                    arguments='{"path":"runtime/tmp.json","force":true}',
                )
                return types.SimpleNamespace(interruptions=[interruption])

        envelope = SdkAgentTaskEnvelope.from_agent_task(
            AgentTask(capability_id="agent.file_analysis", goal="分析并清理缓存文件", input_payload={"path": "runtime"}),
            AgentRuntimeContext(session_id="session-1", run_id="run-1", task_id="run-1:tool-1"),
            subagent_name="FileAnalysisAgent",
            max_turns=5,
        )

        with patch.dict(sys.modules, {"agents": types.SimpleNamespace(Agent=FakeAgent, Runner=FakeRunner)}):
            result = OpenAIAgentsSdkRunnerClient(model="gpt-test").run(envelope)

        self.assertEqual({"path": "runtime/tmp.json", "force": True}, result.approval_request.tool_input)
        self.assertEqual("call-delete-1", result.approval_request.metadata["call_id"])

    def test_runner_adapter_resume_rebuilds_tools_from_persisted_task_envelope(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentRuntimeContext
        from app.agent_runtime.sdk_agents.runner_adapter import OpenAISdkAgentRunnerAdapter

        tool_name_sets = []

        class FakeToolGateway:
            def build_tools(self, tool_names, context):
                tool_name_sets.append(list(tool_names))
                return ["sdk-tool"]

        class FakeRunnerClient:
            def run(self, envelope):
                raise AssertionError("resume test should not start a new SDK run")

            def resume(self, envelope, run_state, *, approved, approval_request, tools=None):
                return {"status": "succeeded", "summary": "SDK 子 Agent 已用原始工具集恢复。"}

        approval_payload = {
            "sdk_agent_approval": True,
            "approval_request": {"tool_name": "filesystem.delete_path", "tool_input": {"path": "runtime/tmp.json"}, "reason": "删除需要审批。"},
            "sdk_agent_result": {
                "metadata": {
                    "run_state": {"format": "json", "value": {"cursor": "sdk-state-1"}},
                    "task_envelope": {
                        "task_id": "run-1:tool-1",
                        "trace_id": "run-1",
                        "capability_id": "agent.file_analysis",
                        "subagent_name": "FileAnalysisAgent",
                        "goal": "分析并清理缓存文件",
                        "input_payload": {"path": "runtime"},
                        "allowed_tools": ["filesystem.delete_path"],
                        "risk_policy": {"mutation_tools": "approval_required"},
                        "constraints": [],
                        "expected_output": [],
                        "max_turns": 5,
                        "context_refs": {"session_id": "session-1", "workflow_run_id": "run-1"},
                        "metadata": {},
                    },
                }
            },
        }

        result = OpenAISdkAgentRunnerAdapter(
            runner_client=FakeRunnerClient(),
            subagent_name="FileAnalysisAgent",
            allowed_tools=["stale.current.config"],
            tool_gateway=FakeToolGateway(),
        ).resume_agent_task_after_approval(
            approval_payload,
            approved=True,
            context=AgentRuntimeContext(session_id="session-1", run_id="run-1", task_id="run-1:sdk-resume-1"),
        )

        self.assertEqual("succeeded", result.status)
        self.assertEqual([["filesystem.delete_path"]], tool_name_sets)

    def test_runner_executor_declares_resume_tailoring_capability(self) -> None:
        from app.agent_runtime.agent_as_tool import OPENAI_SDK_AGENT_EXECUTOR_ID
        from app.agent_runtime.sdk_agents.runner_adapter import OpenAISdkAgentRunnerAdapter, OpenAISdkAgentRunnerExecutor

        class FakeRunnerClient:
            def run(self, envelope):
                raise AssertionError("not needed")

        executor = OpenAISdkAgentRunnerExecutor(
            OpenAISdkAgentRunnerAdapter(runner_client=FakeRunnerClient(), subagent_name="ResumeTailorAgent")
        )

        capabilities = executor.capabilities()

        self.assertEqual(["resume.tailor"], [definition.capability_id for definition in capabilities])
        self.assertEqual(OPENAI_SDK_AGENT_EXECUTOR_ID, capabilities[0].executor_id)
        self.assertEqual(["resume_text", "job_description"], capabilities[0].input_schema["required"])

    def test_runner_executor_calls_runner_adapter(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentRuntimeContext, AgentTask
        from app.agent_runtime.sdk_agents.runner_adapter import OpenAISdkAgentRunnerAdapter, OpenAISdkAgentRunnerExecutor

        class FakeRunnerClient:
            def run(self, envelope):
                return {
                    "task_id": envelope.task_id,
                    "capability_id": envelope.capability_id,
                    "subagent_name": envelope.subagent_name,
                    "status": "succeeded",
                    "summary": "Runner executor 已完成。",
                }

        result = OpenAISdkAgentRunnerExecutor(
            OpenAISdkAgentRunnerAdapter(runner_client=FakeRunnerClient(), subagent_name="ResumeTailorAgent")
        ).call(
            AgentTask(
                capability_id="resume.tailor",
                goal="根据 JD 改简历",
                input_payload={"resume_text": "Java 项目", "job_description": "Java 后端"},
            ),
            AgentRuntimeContext(session_id="session-1", run_id="run-1", task_id="run-1:tool-1"),
        )

        self.assertEqual("succeeded", result.status)
        self.assertEqual("Runner executor 已完成。", result.summary)

    def test_runner_adapter_normalizes_loose_model_result_json(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentRuntimeContext, AgentTask
        from app.agent_runtime.sdk_agents.runner_adapter import OpenAISdkAgentRunnerAdapter

        class FakeRunnerClient:
            def run(self, envelope):
                return {
                    "status": "success",
                    "summary": "子 Agent 配置烟测通过。",
                    "evidence": {"file_path": "input/resume.tex", "preview": "姓名：刘汉卿"},
                    "diagnostics": [],
                    "proposed_actions": ["继续分析技术栈匹配度。"],
                }

        result = OpenAISdkAgentRunnerAdapter(
            runner_client=FakeRunnerClient(),
            subagent_name="OfferMasterSdkAgent",
        ).run_agent_task(
            AgentTask(
                capability_id="resume.tailor",
                goal="验证子 Agent 配置",
                input_payload={"smoke_test": True},
            ),
            AgentRuntimeContext(session_id="session-1", run_id="run-1", task_id="run-1:tool-1"),
        )

        self.assertEqual("succeeded", result.status)
        self.assertEqual("子 Agent 配置烟测通过。", result.summary)
        self.assertEqual([{"file_path": "input/resume.tex", "preview": "姓名：刘汉卿"}], result.raw_result["evidence"])
        self.assertEqual({}, result.raw_result["diagnostics"])
        self.assertEqual([{"description": "继续分析技术栈匹配度。"}], result.raw_result["proposed_actions"])

    def test_runner_adapter_normalizes_blocked_external_result_to_auditable_failure(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentRuntimeContext, AgentTask
        from app.agent_runtime.sdk_agents.runner_adapter import OpenAISdkAgentRunnerAdapter

        class FakeRunnerClient:
            def run(self, envelope):
                return {
                    "status": "blocked",
                    "summary": "DBX MCP 当前不可用。",
                    "diagnostics": {
                        "error_code": "MCP_TOOL_NOT_DISCOVERED",
                        "capability": "agent.dbx_readonly",
                    },
                }

        result = OpenAISdkAgentRunnerAdapter(
            runner_client=FakeRunnerClient(),
            subagent_name="DbxReadOnlyAgent",
        ).run_agent_task(
            AgentTask(
                capability_id="agent.dbx_readonly",
                goal="查询公司展览公司数",
                input_payload={"query": "公司展览公司数"},
            ),
            AgentRuntimeContext(session_id="session-1", run_id="run-1", task_id="run-1:tool-1"),
        )

        self.assertEqual("failed", result.status)
        self.assertIn("DBX MCP 当前不可用", result.summary)
        self.assertEqual("blocked", result.raw_result["diagnostics"]["raw_status"])
        self.assertEqual("MCP_TOOL_NOT_DISCOVERED", result.raw_result["diagnostics"]["error_code"])

    def test_runner_executor_can_declare_tool_backed_capability(self) -> None:
        from app.agent_runtime.agent_as_tool import (
            OPENAI_SDK_AGENT_EXECUTOR_ID,
            AgentCapabilityDefinition,
            AgentRuntimeContext,
            AgentTask,
        )
        from app.agent_runtime.sdk_agents.runner_adapter import OpenAISdkAgentRunnerAdapter, OpenAISdkAgentRunnerExecutor

        class FakeRunnerClient:
            def run(self, envelope):
                return {"status": "succeeded", "summary": f"执行 {envelope.capability_id}"}

        executor = OpenAISdkAgentRunnerExecutor(
            OpenAISdkAgentRunnerAdapter(runner_client=FakeRunnerClient(), subagent_name="WebResearchAgent"),
            additional_capabilities=[
                AgentCapabilityDefinition(
                    capability_id="external.web_search",
                    name="external.web_search",
                    description="Search through SDK sub-agent.",
                    executor_id=OPENAI_SDK_AGENT_EXECUTOR_ID,
                    input_schema={"type": "object", "required": ["query"]},
                    output_schema={"type": "object"},
                    supported_intents=("campus_recruiting_search",),
                )
            ],
        )

        capabilities = executor.capabilities()
        self.assertEqual(["external.web_search", "resume.tailor"], [definition.capability_id for definition in capabilities])

        result = executor.call(
            AgentTask(
                capability_id="external.web_search",
                goal="查腾讯校招官网",
                input_payload={"query": "腾讯 校招 官网"},
            ),
            AgentRuntimeContext(session_id="session-1", run_id="run-1", task_id="run-1:tool-1"),
        )

        self.assertEqual("succeeded", result.status)
        self.assertEqual("执行 external.web_search", result.summary)


if __name__ == "__main__":
    unittest.main()
