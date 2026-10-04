# Declarative MCP Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make MCP integration declarative and extensible, while registering DBX and Chrome DevTools MCP as controlled read/browser capabilities for the SDK child agent.

**Architecture:** A configuration-backed MCP registry owns server definitions, tool discovery, allow/deny filtering, and conversion of discovered MCP schemas into OfferMaster tool definitions. The main agent keeps coarse routing and result aggregation; the SDK child agent receives the registered MCP tools; runtime policy remains the enforcement boundary. DBX uses a dedicated read-only policy for tool names and SQL, while Chrome uses an explicit browser tool allowlist with confirmation for mutations.

**Tech Stack:** Python, Pydantic settings, JSON configuration, MCP Python SDK stdio transport, OpenAI Agents SDK tool gateway, pytest/unittest.

**Spec:** The approved chat design: declarative MCP configuration, generic registry, dynamic `list_tools` schema discovery, explicit allow/deny policy, DBX read-only enforcement, and child-agent delegation.

## Global Constraints

- Database facts must come from DBX MCP responses; connection names are discovered at runtime and never assumed.
- DBX exposes read-only discovery, schema, and query tools only; writes, transactions, connection management, batch scripts, and Redis commands are never registered.
- Chrome and DBX tools are delegated to the SDK child agent; the main runtime owns registration, policy, approval, execution, and audit boundaries.
- Secrets are read from environment variables and are never written into configuration files, prompts, or logs.
- Existing non-DBX MCP integrations keep their current behavior unless the new registry replaces their registration path.
- Every new enforcement behavior has a regression test that fails before the implementation and passes after it.

## Review Focus

- A configured server whose process is unavailable must report `unavailable` and must not invent tools.
- A DBX mutation tool supplied by configuration must be filtered before it reaches the registry or child agent.
- A DBX query with write, lock, or multiple-statement SQL must be rejected before the MCP process is called.
- A discovered tool's real input schema must reach the Agent SDK instead of an empty permissive object schema.
- A newly configured read-only MCP must be discoverable without adding a new Python factory function.

### Task 1: Declarative Server Configuration

**Files:**
- Create: `config/mcp_servers.json`
- Create: `apps/api/app/mcp_gateway/registry_config.py`
- Modify: `apps/api/app/core/config.py`
- Test: `tests/test_mcp_registry_config.py`

**Interfaces:**
- `MCPServerConfig.load(path: Path) -> tuple[MCPServerConfig, ...]`
- `MCPServerConfig(tool_prefix, transport, command, args, env, enabled, allow_tools, deny_tools, policy)`
- `Settings.mcp_registry_config_path` and `Settings.mcp_registry_configs`

- [x] Write failing tests for loading DBX/Chrome config, environment substitution, and missing-file behavior.
- [x] Run `pytest tests/test_mcp_registry_config.py -q` and verify the new tests fail because the config loader is absent.
- [x] Implement the JSON-backed configuration model and safe `${ENV_NAME}` substitution without logging secret values.
- [x] Add DBX read-only and Chrome browser entries to `config/mcp_servers.json`; keep executable commands and package args declarative.
- [x] Run the focused tests and verify they pass.

### Task 2: Dynamic MCP Discovery and Registry

**Files:**
- Modify: `apps/api/app/mcp_gateway/stdio_client.py`
- Create: `apps/api/app/mcp_gateway/registry.py`
- Modify: `apps/api/app/mcp_gateway/configured.py`
- Test: `tests/test_mcp_registry.py`

**Interfaces:**
- `MCPRegistry.discover() -> MCPRegistrySnapshot`
- `MCPRegistry.tool_definitions() -> list[DiscoveredMCPTool]`
- `StdioMCPTransport.list_tools(server) -> tuple[dict[str, Any], ...]`

- [x] Write failing tests for list-tools discovery, allow/deny filtering, and schema preservation.
- [x] Run the tests and verify failure before implementation.
- [x] Implement stdio `list_tools` using the same initialize/list session lifecycle as calls.
- [x] Implement a registry snapshot with server status, discovered tools, and structured diagnostics.
- [x] Make configured MCP client creation use registry server specs instead of hardcoded DBX/Chrome factories.
- [x] Run focused registry tests.

### Task 3: Generic Agent Tool Registration and Safety Policies

**Files:**
- Modify: `apps/api/app/agent_runtime/tool_registry.py`
- Modify: `apps/api/app/mcp_gateway/tool_policy.py`
- Modify: `apps/api/app/mcp_gateway/dbx_readonly.py`
- Test: `tests/test_mcp_registry.py`
- Test: `tests/test_agent_tool_registry.py`

- [x] Write failing tests for discovered schemas becoming `AgentToolDefinition` objects and DBX guard behavior.
- [x] Run tests to verify the missing generic registration and guard behavior.
- [x] Implement generic conversion using MCP tool descriptions and input schemas.
- [x] Keep DBX's dedicated policy as a runtime guard and explicitly classify Chrome mutation tools for approval.
- [x] Run focused tests and ensure blocked tools never invoke the gateway.

### Task 4: Runtime Panel and SDK Child-Agent Delegation

**Files:**
- Modify: `apps/api/app/api/v1/agent.py`
- Modify: `apps/api/app/api/v1/agent_runtime.py`
- Modify: `apps/api/app/agent_runtime/sdk_agents/delegation_policy.py`
- Modify: `apps/api/app/agent_runtime/sdk_agents/runner_adapter.py`
- Modify: `apps/api/app/agent_runtime/external_tasks/configured.py`
- Test: `tests/test_agent_runtime_panel_api.py`
- Test: `tests/test_sdk_agent_delegation_policy.py`
- Test: `tests/test_sdk_agent_tool_gateway.py`

- [x] Write failing tests proving DBX and Chrome discovered tools appear in the panel and child-agent allowlist.
- [x] Run tests to verify the current static probe-only behavior fails those assertions.
- [x] Register the registry snapshot once per runtime construction and expose only policy-approved tools to the SDK child agent.
- [x] Add child-agent instructions for DBX discovery order and browser stop/approval boundaries.
- [x] Run focused runtime tests.

### Task 5: End-to-End MCP Verification and Documentation

**Files:**
- Modify: `.env.example`
- Create: `docs/superpowers/mcp-registry-verification.md`
- Test: `tests/test_mcp_stdio_client.py`

- [x] Run the complete MCP/runtime test set.
- [x] Start the API and verify `/health` and `/api/v1/agent-runtime/panel` report DBX and Chrome registrations.
- [x] Call DBX `list_tools`/`dbx_list_connections` and Chrome `list_pages` through the configured bridge; record actual status without exposing credentials.
- [x] Run full backend tests, compile checks, frontend build, and `git diff --check`.
- [x] Record any external DBX connection availability limitation separately from code correctness.
