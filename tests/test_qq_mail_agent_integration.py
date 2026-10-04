import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class QqMailAgentIntegrationTests(unittest.TestCase):
    def test_delegation_policy_exposes_only_readonly_mail_tools_to_named_child(self) -> None:
        from app.agent_runtime.sdk_agents.delegation_policy import build_sdk_agent_delegation_policy
        from app.agent_runtime.tool_registry import (
            AgentToolDefinition,
            AgentToolRegistry,
            AgentToolRiskLevel,
            create_skill_lazy_agent_tool_definitions,
        )
        from app.core.config import Settings

        names = (
            "mcp.qq_mail.list_messages",
            "mcp.qq_mail.get_message",
            "mcp.qq_mail.send_message",
        )
        registry = AgentToolRegistry(
            create_skill_lazy_agent_tool_definitions()
            + [
                AgentToolDefinition(
                    name=name,
                    description="QQ Mail MCP tool",
                    input_schema={"type": "object", "additionalProperties": False},
                    output_schema={"type": "object"},
                    handler=lambda _session, **_arguments: {"ok": True},
                    risk_level=AgentToolRiskLevel.LOW,
                )
                for name in names
            ]
        )
        settings = Settings(_env_file=None, sdk_agent_enable_qq_mail_mcp=True)

        policy = build_sdk_agent_delegation_policy(settings, tool_registry=registry)

        capability = "agent.qq_mail_readonly"
        self.assertIn(capability, policy.exposed_capability_names)
        self.assertEqual(
            (
                "skill_list",
                "skill_list_actions",
                "skill_read",
                "mcp.qq_mail.list_messages",
                "mcp.qq_mail.get_message",
            ),
            policy.tools_by_capability[capability],
        )
        self.assertEqual("QqMailReadOnlyAgent", policy.subagent_names_by_capability[capability])
        self.assertNotIn("mcp.qq_mail.send_message", policy.internal_tool_names)

    def test_registry_blocks_non_readonly_or_unbounded_mail_arguments(self) -> None:
        from app.mcp_gateway.client import MCPToolCallResult
        from app.mcp_gateway.registry import MCPRegistryClient
        from app.mcp_gateway.registry_config import MCPServerConfig

        calls = []

        class FakeTransport:
            async def list_tools(self, _server):
                return [
                    {"name": "list_messages", "inputSchema": {"type": "object"}},
                    {"name": "get_message", "inputSchema": {"type": "object"}},
                    {"name": "send_message", "inputSchema": {"type": "object"}},
                ]

            async def call_tool(self, _server, *, tool_name, arguments):
                calls.append((tool_name, arguments))
                return MCPToolCallResult(tool_name=tool_name, ok=True, result={"messages": []})

        client = MCPRegistryClient(
            servers=(
                MCPServerConfig(
                    id="qq_mail",
                    tool_prefix="qq_mail",
                    command="python",
                    allow_tools=("list_messages", "get_message", "send_message"),
                    policy="qq_mail_read_only",
                ),
            ),
            transport=FakeTransport(),
        )
        discovered = client.discover_tools()

        self.assertEqual(
            ["qq_mail.list_messages", "qq_mail.get_message"],
            [tool["name"] for tool in discovered],
        )
        too_many = client.call_tool(tool_name="qq_mail.list_messages", arguments={"limit": 1000})
        invalid_uid = client.call_tool(tool_name="qq_mail.get_message", arguments={"uid": "../../1"})

        self.assertFalse(too_many.ok)
        self.assertEqual("QQ_MAIL_READ_ONLY_POLICY", too_many.error)
        self.assertFalse(invalid_uid.ok)
        self.assertEqual("QQ_MAIL_READ_ONLY_POLICY", invalid_uid.error)
        self.assertEqual([], calls)

    def test_declared_mail_mcp_uses_api_python_and_environment_credentials(self) -> None:
        from app.mcp_gateway.registry_config import load_mcp_server_configs

        configs = load_mcp_server_configs(
            PROJECT_ROOT / "config" / "mcp_servers.json",
            environ={
                "JOBPILOT_SDK_AGENT_ENABLE_QQ_MAIL_MCP": "true",
                "JOBPILOT_QQ_MAIL_PYTHON": "python",
                "JOBPILOT_QQ_MAIL_API_ROOT": "apps/api",
                "JOBPILOT_QQ_MAIL_USERNAME": "candidate@example.com",
                "JOBPILOT_QQ_MAIL_AUTH_CODE": "test-only-auth-code",
            },
        )
        config = next(server for server in configs if server.id == "qq_mail")

        self.assertEqual("python", config.command)
        self.assertEqual(("-m", "app.mcp_gateway.qq_mail_server"), config.args)
        self.assertEqual("apps/api", config.cwd)
        self.assertEqual(("list_messages", "get_message"), config.allow_tools)
        self.assertEqual("candidate@example.com", config.env["JOBPILOT_QQ_MAIL_USERNAME"])
        self.assertEqual("test-only-auth-code", config.env["JOBPILOT_QQ_MAIL_AUTH_CODE"])

    def test_runtime_panel_reports_qq_mail_agent_and_mcp_state(self) -> None:
        from app.api.v1.agent_runtime import _mcp_integration_statuses
        from app.core.config import Settings

        settings = Settings(
            _env_file=None,
            sdk_agent_enable_qq_mail_mcp=True,
            qq_mail_username="candidate@example.com",
            qq_mail_auth_code="test-only-auth-code",
        )
        capabilities = [
            {"id": "agent.qq_mail_readonly"},
            {"id": "mcp.qq_mail.list_messages"},
            {"id": "mcp.qq_mail.get_message"},
        ]
        client = SimpleNamespace(
            statuses=lambda: [
                {
                    "id": "qq_mail",
                    "status": "configured",
                    "detail": "Configured; tool discovery has not run.",
                    "configured_tools": ["qq_mail.list_messages", "qq_mail.get_message"],
                    "discovered_tools": [],
                }
            ]
        )

        integrations = _mcp_integration_statuses(capabilities, settings=settings, client=client)
        mail = next(item for item in integrations if item["id"] == "qq_mail")

        self.assertEqual("configured", mail["status"])
        self.assertIn("mcp.qq_mail.list_messages", mail["registered_tools"])

    def test_runtime_panel_never_claims_mail_ready_without_credentials(self) -> None:
        from app.api.v1.agent_runtime import _mcp_integration_statuses
        from app.core.config import Settings

        settings = Settings(_env_file=None, sdk_agent_enable_qq_mail_mcp=True)
        integrations = _mcp_integration_statuses([], settings=settings)
        mail = next(item for item in integrations if item["id"] == "qq_mail")

        self.assertEqual("credentials_missing", mail["status"])
        self.assertNotIn("auth_code", str(mail).lower())

    def test_sdk_runner_builds_a_distinct_mail_agent_with_scoped_tools(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentRuntimeContext, AgentTask
        from app.agent_runtime.sdk_agents.runner_adapter import OpenAISdkAgentRunnerAdapter
        from app.agent_runtime.sdk_agents.schemas import SdkAgentResultEnvelope

        envelopes = []

        class CapturingRunner:
            def run(self, envelope):
                envelopes.append(envelope)
                return SdkAgentResultEnvelope(
                    task_id=envelope.task_id,
                    capability_id=envelope.capability_id,
                    subagent_name=envelope.subagent_name,
                    status="failed",
                    summary="capture-only",
                )

        adapter = OpenAISdkAgentRunnerAdapter(
            runner_client=CapturingRunner(),
            subagent_name="OfferMasterSdkAgent",
            allowed_tools=[
                "skill_list",
                "skill_read",
                "mcp.qq_mail.list_messages",
                "mcp.qq_mail.get_message",
                "mcp.dbx.dbx_list_connections",
            ],
            risk_policy={
                "delegation": {
                    "tools_by_capability": {
                        "agent.qq_mail_readonly": [
                            "skill_list",
                            "skill_read",
                            "mcp.qq_mail.list_messages",
                            "mcp.qq_mail.get_message",
                        ]
                    },
                    "subagent_names_by_capability": {"agent.qq_mail_readonly": "QqMailReadOnlyAgent"},
                }
            },
        )
        adapter.run_agent_task(
            AgentTask(
                capability_id="agent.qq_mail_readonly",
                goal="Read recruitment notices from QQ Mail.",
                input_payload={"task": "Read recent recruiting notices."},
            ),
            AgentRuntimeContext(session_id="s-mail", run_id="r-mail", task_id="t-mail"),
        )

        self.assertEqual("QqMailReadOnlyAgent", envelopes[0].subagent_name)
        self.assertEqual(
            ["skill_list", "skill_read", "mcp.qq_mail.list_messages", "mcp.qq_mail.get_message"],
            envelopes[0].allowed_tools,
        )

    def test_mail_child_agent_is_registered_as_a_main_runtime_capability(self) -> None:
        from app.agent_runtime.agent_as_tool import (
            OPENAI_SDK_AGENT_EXECUTOR_ID,
            OPENAI_SDK_QQ_MAIL_AGENT_EXECUTOR_ID,
        )
        from app.agent_runtime.external_tasks.configured import build_agent_runtime_executor_bundle
        from app.agent_runtime.tool_registry import (
            AgentToolDefinition,
            AgentToolRegistry,
            AgentToolRiskLevel,
            create_skill_lazy_agent_tool_definitions,
        )
        from app.core.config import OPENAI_SDK_AGENT_MODE_AGENTS_SDK, Settings

        mail_definitions = [
            AgentToolDefinition(
                name=f"mcp.qq_mail.{name}",
                description="Read-only QQ Mail tool",
                input_schema={"type": "object", "additionalProperties": False},
                output_schema={"type": "object"},
                handler=lambda _session, **_arguments: {"ok": True},
                risk_level=AgentToolRiskLevel.LOW,
            )
            for name in ("list_messages", "get_message")
        ]
        registry = AgentToolRegistry(create_skill_lazy_agent_tool_definitions() + mail_definitions)
        settings = Settings(
            _env_file=None,
            external_agent_auto_dispatch=True,
            external_web_search_provider="bailian",
            openai_sdk_agent_enabled=True,
            openai_sdk_agent_mode=OPENAI_SDK_AGENT_MODE_AGENTS_SDK,
            openai_sdk_agent_api_key="test-api-key",
            openai_sdk_agent_model="test-model",
            sdk_agent_enable_qq_mail_mcp=True,
        )

        executors, capability_executor_ids = build_agent_runtime_executor_bundle(settings, tool_registry=registry)
        general_executor = executors[OPENAI_SDK_AGENT_EXECUTOR_ID]
        mail_executor = executors[OPENAI_SDK_QQ_MAIL_AGENT_EXECUTOR_ID]
        mail_capability = next(
            capability for capability in mail_executor.capabilities() if capability.capability_id == "agent.qq_mail_readonly"
        )

        self.assertEqual(OPENAI_SDK_QQ_MAIL_AGENT_EXECUTOR_ID, capability_executor_ids["agent.qq_mail_readonly"])
        self.assertEqual("QQ 邮箱只读 Agent", mail_capability.name)
        self.assertIn("IMAPS 只读", mail_capability.description)
        self.assertNotIn("agent.qq_mail_readonly", {capability.capability_id for capability in general_executor.capabilities()})
        self.assertNotIn("mcp.qq_mail.list_messages", general_executor._adapter.allowed_tools)
        self.assertEqual(OPENAI_SDK_QQ_MAIL_AGENT_EXECUTOR_ID, mail_executor._adapter.executor_id)
        self.assertEqual(
            ["skill_list", "skill_list_actions", "skill_read", "mcp.qq_mail.list_messages", "mcp.qq_mail.get_message"],
            mail_executor._adapter.allowed_tools,
        )
        self.assertEqual(
            [
                "skill_list",
                "skill_list_actions",
                "skill_read",
                "mcp.qq_mail.list_messages",
                "mcp.qq_mail.get_message",
            ],
            mail_executor._adapter.risk_policy["delegation"]["tools_by_capability"]["agent.qq_mail_readonly"],
        )

    def test_qq_mail_recruitment_skill_is_a_parseable_builtin_package(self) -> None:
        from app.agent_runtime.memory.skill_package_parser import SkillPackageParser

        skill_path = PROJECT_ROOT / "docs" / "agent-skills" / "builtin-mail" / "qq-mail-recruitment"
        package = SkillPackageParser().parse(skill_path)

        self.assertEqual("qq-mail-recruitment", package.name)
        required_tools = package.import_report["required_tools"]
        self.assertIn("mcp.qq_mail.list_messages", required_tools)
        self.assertIn("mcp.qq_mail.get_message", required_tools)
        self.assertTrue(any("non-trusted" in item.lower() or "不可信" in item for item in package.content.splitlines()))

    def test_mail_tools_open_inbox_readonly_and_fetch_with_body_peek(self) -> None:
        from app.mcp_gateway import qq_mail_server

        class FakeMailbox:
            def __init__(self):
                self.calls = []
                self.selected = None

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

            def login(self, *_args):
                return "OK", [b"authenticated"]

            def select(self, folder, readonly=False):
                self.selected = (folder, readonly)
                return "OK", [b"1"]

            def uid(self, command, *args):
                self.calls.append((command, args))
                if command == "SEARCH":
                    return "OK", [b"42"]
                header_or_message = (
                    b"From: Hiring <jobs@example.com>\r\n"
                    b"To: Candidate <candidate@example.com>\r\n"
                    b"Subject: =?utf-8?q?Interview_notice?=\r\n"
                    b"Date: Tue, 29 Sep 2026 10:00:00 +0800\r\n"
                    b"Message-ID: <mail-42@example.com>\r\n\r\n"
                    b"The interview is scheduled for tomorrow."
                )
                return "OK", [(b"* 1 FETCH (UID 42 FLAGS (\\Seen))", header_or_message)]

        mailbox = FakeMailbox()
        with patch.dict(
            "os.environ",
            {
                "JOBPILOT_QQ_MAIL_USERNAME": "test@example.com",
                "JOBPILOT_QQ_MAIL_AUTH_CODE": "test-only-auth-code",
            },
        ), patch.object(qq_mail_server.imaplib, "IMAP4_SSL", return_value=mailbox):
            listed = qq_mail_server.list_messages(limit=5, since_days=7)
            message = qq_mail_server.get_message(uid="42", max_chars=2000)

        self.assertEqual(("INBOX", True), mailbox.selected)
        self.assertTrue(listed["read_only"])
        self.assertEqual("42", listed["messages"][0]["uid"])
        self.assertTrue(message["read_only"])
        self.assertIn("interview is scheduled", message["message"]["body"])
        self.assertEqual(["\\Seen"], message["message"]["flags_before"])
        self.assertEqual(["\\Seen"], message["message"]["flags_after"])
        self.assertTrue(message["message"]["seen_state_unchanged"])
        fetch_arguments = [args for command, args in mailbox.calls if command == "FETCH"]
        body_fetches = [args for args in fetch_arguments if "BODY" in args[-1]]
        self.assertTrue(body_fetches)
        self.assertTrue(all("BODY.PEEK" in args[-1] for args in body_fetches))


if __name__ == "__main__":
    unittest.main()
