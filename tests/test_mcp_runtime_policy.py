import unittest


class MCPRuntimePolicyTests(unittest.TestCase):
    def test_browser_state_changing_actions_require_confirmation(self) -> None:
        from app.mcp_gateway.tool_policy import MCPToolPolicy

        policy = MCPToolPolicy.from_allowlist(["chrome.list_pages", "chrome.click", "chrome.fill"])

        self.assertFalse(policy.requires_confirmation("chrome.list_pages"))
        self.assertTrue(policy.requires_confirmation("chrome.click"))
        self.assertTrue(policy.requires_confirmation("chrome.fill"))

    def test_browser_read_tools_are_low_risk_and_navigation_is_child_visible(self) -> None:
        from app.agent_runtime.tool_registry import AgentToolRiskLevel, create_mcp_agent_tool_definitions

        definitions = create_mcp_agent_tool_definitions(
            object(),
            allowed_tool_names=["chrome.list_pages", "chrome.navigate_page", "chrome.click"],
        )
        by_name = {definition.name: definition for definition in definitions}

        self.assertEqual(AgentToolRiskLevel.LOW, by_name["mcp.chrome.list_pages"].risk_level)
        self.assertEqual(AgentToolRiskLevel.LOW, by_name["mcp.chrome.navigate_page"].risk_level)
        self.assertEqual(AgentToolRiskLevel.HIGH, by_name["mcp.chrome.click"].risk_level)
        self.assertTrue(by_name["mcp.chrome.click"].requires_confirmation)


if __name__ == "__main__":
    unittest.main()
