from unittest import TestCase


class AgentToolPermissionPolicyTest(TestCase):
    def test_skill_permission_policy_denies_disallowed_tool_before_allowlist(self) -> None:
        from app.agent_runtime.guardrails import AgentToolCallContext, AgentToolRuntimeGuard
        from app.agent_runtime.tool_permissions import AgentToolPermissionPolicy
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry

        registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name="submit_application",
                    description="Submit an application.",
                    input_schema={"type": "object"},
                    output_schema={"type": "object"},
                )
            ]
        )
        policy = AgentToolPermissionPolicy.from_skill_metadata(
            "skill-wechat",
            {
                "allowed_tools": ["submit_application"],
                "disallowed_tools": ["submit_application"],
            },
        )

        result = AgentToolRuntimeGuard().pre_check(
            AgentToolCallContext(stage="apply", tool_name="submit_application", source_type="application"),
            registry=registry,
            skill_permission_policy=policy,
        )

        self.assertFalse(result.ok)
        self.assertEqual("TOOL_SKILL_DENIED", result.error_code)
        self.assertEqual("stop", result.next_action)
        self.assertEqual("skill-wechat", result.error_details["skill_id"])
        self.assertEqual(["submit_application"], result.error_details["disallowed_tools"])

    def test_skill_permission_policy_asks_when_tool_is_not_in_declared_allowlist(self) -> None:
        from app.agent_runtime.guardrails import AgentToolCallContext, AgentToolRuntimeGuard
        from app.agent_runtime.tool_permissions import AgentToolPermissionPolicy
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry

        registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name="ocr.extract_text",
                    description="Extract text from an image.",
                    input_schema={"type": "object"},
                    output_schema={"type": "object"},
                )
            ]
        )
        policy = AgentToolPermissionPolicy.from_skill_metadata(
            "skill-wechat",
            {"allowed_tools": ["weixin-articles-mcp.read_article"]},
        )

        blocked = AgentToolRuntimeGuard().pre_check(
            AgentToolCallContext(stage="extract", tool_name="ocr.extract_text", source_type="wechat_article"),
            registry=registry,
            skill_permission_policy=policy,
        )
        confirmed = AgentToolRuntimeGuard().pre_check(
            AgentToolCallContext(
                stage="extract",
                tool_name="ocr.extract_text",
                source_type="wechat_article",
                user_confirmed=True,
            ),
            registry=registry,
            skill_permission_policy=policy,
        )

        self.assertFalse(blocked.ok)
        self.assertEqual("TOOL_SKILL_CONFIRMATION_REQUIRED", blocked.error_code)
        self.assertEqual("request_user_confirmation", blocked.next_action)
        self.assertTrue(blocked.retryable)
        self.assertIn("not declared in the active Skill allowed_tools", blocked.reason)
        self.assertTrue(confirmed.ok)

    def test_skill_permission_policy_allows_declared_tool_and_records_artifact(self) -> None:
        from app.agent_runtime.guardrails import AgentToolCallContext, AgentToolRuntimeGuard
        from app.agent_runtime.tool_permissions import AgentToolPermissionPolicy
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry

        registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name="weixin-articles-mcp.read_article",
                    description="Read a WeChat article.",
                    input_schema={"type": "object"},
                    output_schema={"type": "object"},
                )
            ]
        )
        policy = AgentToolPermissionPolicy.from_skill_metadata(
            "skill-wechat",
            {"allowed_tools": ["weixin-articles-mcp.read_article"]},
        )

        result = AgentToolRuntimeGuard().pre_check(
            AgentToolCallContext(
                stage="fetch",
                tool_name="weixin-articles-mcp.read_article",
                source_type="wechat_article",
            ),
            registry=registry,
            skill_permission_policy=policy,
        )

        self.assertTrue(result.ok)
        self.assertEqual("allow", result.artifacts["skill_permission_decision"])
        self.assertEqual("skill-wechat", result.artifacts["skill_id"])

    def test_skill_permission_policy_asks_for_explicit_ask_tools_even_when_allowed(self) -> None:
        from app.agent_runtime.guardrails import AgentToolCallContext, AgentToolRuntimeGuard
        from app.agent_runtime.tool_permissions import AgentToolPermissionPolicy
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry

        registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name="browser.open",
                    description="Open a visible browser page.",
                    input_schema={"type": "object"},
                    output_schema={"type": "object"},
                )
            ]
        )
        policy = AgentToolPermissionPolicy.from_skill_metadata(
            "skill-browser",
            {"allowed_tools": ["browser.open"], "ask_tools": ["browser.open"]},
        )

        result = AgentToolRuntimeGuard().pre_check(
            AgentToolCallContext(stage="browse", tool_name="browser.open", source_type="agent_chat"),
            registry=registry,
            skill_permission_policy=policy,
        )

        self.assertFalse(result.ok)
        self.assertEqual("TOOL_SKILL_CONFIRMATION_REQUIRED", result.error_code)
        self.assertEqual("ask", result.error_details["permission_decision"])

    def test_low_risk_external_web_search_can_run_outside_skill_allowlist(self) -> None:
        from app.agent_runtime.guardrails import AgentToolCallContext, AgentToolRuntimeGuard
        from app.agent_runtime.tool_permissions import AgentToolPermissionPolicy
        from app.agent_runtime.tool_registry import (
            EXTERNAL_WEB_SEARCH_TOOL,
            AgentToolDefinition,
            AgentToolRegistry,
            AgentToolRiskLevel,
        )

        registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name=EXTERNAL_WEB_SEARCH_TOOL,
                    description="Search public web results through an external agent.",
                    input_schema={"type": "object"},
                    output_schema={"type": "object"},
                    risk_level=AgentToolRiskLevel.LOW,
                    requires_confirmation=False,
                    allowed_source_types=frozenset({"agent_chat", "web_search"}),
                )
            ]
        )
        policy = AgentToolPermissionPolicy.from_skill_metadata(
            "skill-wechat",
            {"allowed_tools": ["weixin-articles-mcp.read_article"]},
        )

        result = AgentToolRuntimeGuard().pre_check(
            AgentToolCallContext(stage="search", tool_name=EXTERNAL_WEB_SEARCH_TOOL, source_type="agent_chat"),
            registry=registry,
            skill_permission_policy=policy,
        )

        self.assertTrue(result.ok)
        self.assertEqual("allow_low_risk_runtime_capability", result.artifacts["skill_permission_decision"])
        self.assertEqual("skill-wechat", result.artifacts["skill_id"])

    def test_low_risk_local_overview_tools_can_run_outside_skill_allowlist(self) -> None:
        from app.agent_runtime.guardrails import AgentToolCallContext, AgentToolRuntimeGuard
        from app.agent_runtime.tool_permissions import AgentToolPermissionPolicy
        from app.agent_runtime.tool_registry import (
            LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL,
            LOCAL_JOB_SOURCE_OVERVIEW_TOOL,
            AgentToolDefinition,
            AgentToolRegistry,
            AgentToolRiskLevel,
        )

        registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name=LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL,
                    description="Read local company database overview.",
                    input_schema={"type": "object"},
                    output_schema={"type": "object"},
                    risk_level=AgentToolRiskLevel.LOW,
                    requires_confirmation=False,
                    allowed_source_types=frozenset({"agent_chat"}),
                ),
                AgentToolDefinition(
                    name=LOCAL_JOB_SOURCE_OVERVIEW_TOOL,
                    description="Read local job source and company board overview.",
                    input_schema={"type": "object"},
                    output_schema={"type": "object"},
                    risk_level=AgentToolRiskLevel.LOW,
                    requires_confirmation=False,
                    allowed_source_types=frozenset({"agent_chat"}),
                ),
            ]
        )
        policy = AgentToolPermissionPolicy.from_skill_metadata(
            "skill-wechat",
            {"allowed_tools": ["weixin-articles-mcp.read_article"]},
        )

        for tool_name in (LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL, LOCAL_JOB_SOURCE_OVERVIEW_TOOL):
            with self.subTest(tool_name=tool_name):
                result = AgentToolRuntimeGuard().pre_check(
                    AgentToolCallContext(stage="overview", tool_name=tool_name, source_type="agent_chat"),
                    registry=registry,
                    skill_permission_policy=policy,
                )

                self.assertTrue(result.ok)
                self.assertEqual("allow_low_risk_runtime_capability", result.artifacts["skill_permission_decision"])
                self.assertEqual("skill-wechat", result.artifacts["skill_id"])

    def test_database_mutation_tools_require_explicit_confirmation(self) -> None:
        from app.agent_runtime.guardrails import AgentToolCallContext, AgentToolRuntimeGuard
        from app.agent_runtime.tool_registry import (
            DATABASE_COMPANY_UPDATE_TOOL,
            DATABASE_JOB_LEAD_DELETE_TOOL,
            create_default_agent_tool_registry,
        )

        registry = create_default_agent_tool_registry()
        for tool_name in (DATABASE_COMPANY_UPDATE_TOOL, DATABASE_JOB_LEAD_DELETE_TOOL):
            with self.subTest(tool_name=tool_name):
                blocked = AgentToolRuntimeGuard().pre_check(
                    AgentToolCallContext(stage="database_mutation", tool_name=tool_name, source_type="agent_chat"),
                    registry=registry,
                )
                confirmed = AgentToolRuntimeGuard().pre_check(
                    AgentToolCallContext(
                        stage="database_mutation",
                        tool_name=tool_name,
                        source_type="agent_chat",
                        user_confirmed=True,
                    ),
                    registry=registry,
                )

                self.assertFalse(blocked.ok)
                self.assertEqual("TOOL_USER_CONFIRMATION_REQUIRED", blocked.error_code)
                self.assertTrue(confirmed.ok)

    def test_low_risk_resume_tailor_can_run_outside_skill_allowlist(self) -> None:
        from app.agent_runtime.guardrails import AgentToolCallContext, AgentToolRuntimeGuard
        from app.agent_runtime.tool_permissions import AgentToolPermissionPolicy
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry, AgentToolRiskLevel

        registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name="resume.tailor",
                    description="Generate revised resume text from user-provided resume content and a target JD.",
                    input_schema={"type": "object"},
                    output_schema={"type": "object"},
                    risk_level=AgentToolRiskLevel.LOW,
                    requires_confirmation=False,
                    allowed_source_types=frozenset({"agent_chat"}),
                )
            ]
        )
        policy = AgentToolPermissionPolicy.from_skill_metadata(
            "skill-chat",
            {"allowed_tools": ["external.web_search"]},
        )

        result = AgentToolRuntimeGuard().pre_check(
            AgentToolCallContext(stage="resume", tool_name="resume.tailor", source_type="agent_chat"),
            registry=registry,
            skill_permission_policy=policy,
        )

        self.assertTrue(result.ok)
        self.assertEqual("allow_low_risk_runtime_capability", result.artifacts["skill_permission_decision"])
        self.assertEqual("skill-chat", result.artifacts["skill_id"])

    def test_medium_runtime_capability_without_confirmation_can_run_outside_unrelated_skill_allowlist(self) -> None:
        from app.agent_runtime.guardrails import AgentToolCallContext, AgentToolRuntimeGuard
        from app.agent_runtime.tool_permissions import AgentToolPermissionPolicy
        from app.agent_runtime.tool_registry import OFFERIO_COMPANY_JOBS_TOOL, AgentToolDefinition, AgentToolRegistry, AgentToolRiskLevel

        registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name=OFFERIO_COMPANY_JOBS_TOOL,
                    description="Sync OfferIO company aggregated campus recruiting jobs into local job leads.",
                    input_schema={"type": "object"},
                    output_schema={"type": "object"},
                    risk_level=AgentToolRiskLevel.MEDIUM,
                    requires_confirmation=False,
                    allowed_source_types=frozenset({"agent_chat"}),
                )
            ]
        )
        policy = AgentToolPermissionPolicy.from_skill_metadata(
            "skill-xiaohongshu",
            {"allowed_tools": ["xiaohongshu-mcp.search_feeds"]},
        )

        result = AgentToolRuntimeGuard().pre_check(
            AgentToolCallContext(stage="sync", tool_name=OFFERIO_COMPANY_JOBS_TOOL, source_type="agent_chat"),
            registry=registry,
            skill_permission_policy=policy,
        )

        self.assertTrue(result.ok)
        self.assertEqual("allow_low_risk_runtime_capability", result.artifacts["skill_permission_decision"])
        self.assertEqual("skill-xiaohongshu", result.artifacts["skill_id"])

    def test_skill_permission_policy_round_trips_loaded_skill_snapshot(self) -> None:
        from app.agent_runtime.tool_permissions import AgentToolPermissionPolicy

        policy = AgentToolPermissionPolicy.from_loaded_skill_metadata(
            [
                (
                    "skill-a",
                    {
                        "allowed_tools": ["tool.fetch", "tool.ocr"],
                        "ask_tools": ["browser.open"],
                    },
                ),
                (
                    "skill-b",
                    {
                        "allowed_tools": ["tool.fetch", "tool.search"],
                        "disallowed_tools": ["submit_application"],
                    },
                ),
            ]
        )
        snapshot = policy.to_metadata()
        restored = AgentToolPermissionPolicy.from_metadata_snapshot(snapshot)

        self.assertEqual(("skill-a", "skill-b"), restored.skill_ids)
        self.assertEqual(("tool.fetch", "tool.ocr", "tool.search"), restored.allowed_tools)
        self.assertEqual(("browser.open",), restored.ask_tools)
        self.assertEqual(("submit_application",), restored.disallowed_tools)
        self.assertEqual("loaded_skill_snapshot", snapshot["policy_source"])


if __name__ == "__main__":
    unittest.main()
