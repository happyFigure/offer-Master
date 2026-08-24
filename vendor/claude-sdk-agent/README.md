# claude-sdk-agent

`claude-sdk-agent` is a standalone OpenAI-compatible SSE adapter built on Claude Agent SDK.

## Scope

- Exposes `POST /v1/chat/completions`
- Keeps `session_id` normalization semantics aligned with `my-agents`
- Keeps UAC / TDL auth behavior aligned with `my-agents`
- Persists `frontend_session_id -> claude_session_id` mapping in `data/sessions/session-map.json`
- Runs a local Anthropic proxy so Claude CLI can reuse front-end auth context (`X-User-Id`, `uac-user-id`, `uac-user-token`, `api-key`)
- Auto-loads MCP server definitions from `../my-agents/mcps` by default
- Inspects request-scoped Claude Code workspace resources without returning settings or MCP secrets

## Setup

1. Install Claude Code CLI yourself and make sure `claude` is available to the service process.
   By default the service auto-detects the CLI from `PATH`. If a machine uses a custom install location, set `CLAUDE_SDK_AGENT_CLI_PATH=/abs/path/to/claude` before startup.
   Auto-detection only selects a CLI that can successfully return `claude --version`; otherwise the Agent SDK bundled CLI remains available as fallback.
2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Configure Claude Code / MiniMax under `config/service.json`.
   The default `config_dir` points to `../claude-code/.claude`.
   The `provider.base_url` should be the Anthropic-compatible root URL, not the OpenAI `/v1/chat/completions` path.

4. Start the service:

```bash
./start.sh
```

Default listen address: `0.0.0.0:18008`

## Notes

- `data/runtime/allow_users.json` uses the same whitelist semantics as `my-agents` and is generated automatically at startup when missing.
- If `../my-agents/config/allow_users.json` contains `x-api-key`, `claude-sdk-agent` will expose it to skills as fallback `TDL_API_KEY` when the current request does not provide a TDL API key.
- `stream=true` is the default behavior for `/v1/chat/completions`.
- Model routing follows these rules:
  - omitted `model`, `openclaw:main`, and `claude-code` all fall back to the configured default model
  - any other front-end `model` value is forwarded upstream as-is
- MCP routing follows these rules:
  - `config/service.json` `mcp.auto_load=true` loads all supported `*.json` MCP configs from `../my-agents/mcps`
  - HTTP/SSE MCP headers with empty `x-api-key` or UAC fields are filled from the current request context when possible
- Workspace runtime contract:
  - `POST /v1/runtime/workspace/inspect` accepts an ordered `workspace` object with `cwd` and `add_dirs`.
  - `cwd` is the native Claude Code project root. Project/local settings, `.mcp.json`, `CLAUDE.md`, rules, skills, commands, output styles, agents, hooks, and native `.claude/workflows/*.js` workflows are discovered from this root according to `setting_sources` and `strict_mcp_config`.
  - `add_dirs` are access roots by default; their project settings, hooks, permissions, and MCP files are not imported.
  - Non-native files such as `.flow/workflows/*.workflow.json` are not returned as Claude Code workflow resources.
  - The response separates `agentPolicy`, declared `resources`, expected `effectiveRuntime`, and SDK `observed` state. Configuration values, MCP environment variables, and auth fields are omitted.
  - `commands.items[].invoke` is the structured payload accepted for workspace commands, skills, and workflows. Workflow entries set `requiresConfirmation=false` and are sent directly through the existing chat/runtime command path; runtime tool permissions still follow the effective permission profile. Resource fingerprints force a client reconnect when project runtime files change.
- Runtime permission profiles are Agent-scoped and persisted only by the Service in `agent_configs.advanced_config.permissions`. The Service projects the selected Agent's `profile`, `runtime_key`, and `revision` into each request's `runtime_config.permissions`; this runtime consumes that immutable request snapshot and does not expose permission mutation endpoints or write permission sidecar files.
- Product profiles map to Claude Code native modes: `readonly -> plan`, `safe -> default`, `edit -> acceptEdits`, plus native `auto`, `dontAsk`, and `bypassPermissions`. The compatibility id `readonly` is displayed as native Plan Mode: only the planning phase is read-only, and approving `ExitPlanMode` allows Claude Code to continue under the selected execution permission. `fullBypass` is the only platform-extended profile; it additionally removes Agent-configured tool lists and requests the CLI dangerous bypass flag, but it cannot override workspace/managed deny or ask rules, `disableBypassPermissionsMode`, or blocking Hooks without a separate resource-loading adapter.
- Claude SDK option passthrough:
  - `config/service.json` `claude` supports JSON-configurable `ClaudeAgentOptions` such as `tools`, `allowed_tools`, `disallowed_tools`, `strict_mcp_config`, `fallback_model`, `max_turns`, `max_budget_usd`, `betas`, `settings`, `add_dirs`, `env`, `extra_args`, `agents`, `sandbox`, `plugins`, `thinking`, `effort`, `output_format`, `task_budget`, and related limits.
  - `env` is merged with the service-managed provider proxy environment; internal proxy/auth values win on key conflicts.
  - `add_dirs` is merged with the generated skill mount directory.
  - `agents` accepts JSON objects; snake_case aliases such as `max_turns` and `permission_mode` are converted to Claude SDK camelCase fields.
  - Callback/object fields such as `can_use_tool`, `hooks`, `stderr`, and `session_store` are owned by service code and are not loaded from JSON config.
- Claude CLI does not talk to the remote provider directly. It talks to this service's internal `/internal/anthropic/...` proxy, which injects request-scoped auth headers before forwarding upstream.
- When Claude Agent SDK is missing, stream mode degrades to a readable SSE error message and non-stream mode returns a JSON error body.
- If the parent process launches `claude-sdk-agent` via `sudo`, `src/main.py` will automatically drop back to the original `SUDO_USER` before initializing the service, unless `CLAUDE_SDK_AGENT_KEEP_ROOT=1` is set.
- In the same `sudo` startup path, `src/main.py` also repairs ownership/permissions under `biz-core-slaver/biz-core` for the target operating user before dropping privileges. Set `CLAUDE_SDK_AGENT_SKIP_REPO_PERMISSION_REPAIR=1` to disable that repair step.
- If `<repo>/agents/claude-sdk-agent/log/` is not writable on a machine, logging automatically falls back to a temp-directory log path and finally to stdout-only mode, so the service can still start.

## Troubleshooting

- `ProcessError: Command failed with exit code 127` means the Claude CLI process started but could not find an interpreter or a command required during initialization. Common examples are an npm-installed `claude` shim without `node` on `PATH`, or MCP/hook commands such as `uvx`/`npx` missing from the sidecar environment.
- The service augments the Claude subprocess `PATH` with the selected CLI directory, the active Python environment, and common user-level bin directories after any `sudo` user switch. An explicitly configured `claude.env.PATH` remains authoritative.
- Claude CLI stderr is logged with the `[claude-sdk][cli-stderr]` prefix and included in initialization failures. Request-scoped token/key values are redacted before logging.
