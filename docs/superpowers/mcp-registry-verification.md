# MCP Registry Verification

Date: 2026-09-23

## Runtime Design

- `config/mcp_servers.json` is the declaration point for MCP servers.
- `MCPRegistryClient` owns discovery, allow/deny filtering, schemas, and policy checks.
- The SDK child agent receives only discovered and policy-approved MCP tools.
- The runtime owns approvals and execution boundaries. Chrome mutation tools require confirmation; DBX is read-only.
- The stdio bridge keeps one persistent session per server spec. This is required for Chrome because page navigation and selected-page state live in the MCP server process.

## Verification Results

- Backend tests: `804 passed, 2 skipped, 23 subtests passed`.
- Focused MCP tests after the persistent-session fix: `12 passed`.
- API `compileall`: passed.
- Frontend production build: passed, 1802 modules transformed.
- `git diff --check`: passed; only existing line-ending warnings were emitted.
- API: `http://127.0.0.1:8000/health` returned HTTP 200.
- Frontend: `http://127.0.0.1:5174/` returned HTTP 200 and the root mount exists.
- Runtime panel: Chrome and DBX are registered through `stdio_bridge`.
- Runtime panel delegation: Chrome read/navigation tools and DBX read-only tools are exposed to `openai-sdk-agent`; Chrome click/fill/type/keypress tools remain runtime-owned and require confirmation.
- Chrome: 9 allowed tools discovered. A single session successfully executed `list_pages`, `navigate_page(pageId=1, url=http://127.0.0.1:5174/)`, then `list_pages`; the final result retained the `JobPilot` page.
- Chrome screenshot: `take_screenshot(pageId=1, format=png)` returned both text and image content successfully.
- DBX: 6 read-only tools discovered. `dbx_list_connections` returned `No connections configured in DBX.`.
- DBX policy: `DELETE FROM companies` was rejected as `DBX_READ_ONLY_POLICY` before transport execution.

## Environment Limitation

The DBX MCP server is healthy, but DBX currently has zero configured connections in `%APPDATA%\\com.dbx.app\\dbx.db`. OfferMaster does not copy its own database credentials into DBX or modify DBX connection storage automatically. Once a DBX connection is configured in the DBX application, the same discovered read-only tools can inspect its databases, tables, schemas, and SELECT queries.
