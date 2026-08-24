# Claude SDK Agent Integration

OfferMaster integrates `claude-sdk-agent` as an external execution service instead of embedding it into the main FastAPI process.

## Layout

```text
OfferMaster
  apps/api/app/agent_runtime/external_tasks/   # HTTP adapter and dispatch code
  vendor/claude-sdk-agent/                     # source snapshot for local context
  scripts/start_claude_sdk_agent.ps1           # local startup helper

Runtime service
  http://127.0.0.1:18008/v1/chat/completions
```

The vendored directory is a source snapshot. It intentionally excludes runtime state:

- `.venv/`
- `.claude/`
- `data/`
- `log/` and `logs/`
- API keys, allow-user files, sessions, checkpoints, and generated artifacts

Those paths are ignored in `.gitignore` in case the service is started from the vendored source.

## OfferMaster Tools

OfferMaster currently uses the external agent through two tool surfaces:

- `applications.find_apply_entry`: finds an official application page for a local job and stops before final submission.
- `external.web_search`: handles ordinary web/campus-recruiting search requests through a deterministic HTTP search fallback. For campus recruiting queries, known official entrances such as Tencent `join.qq.com` and JD `campus.jd.com` are prioritized before generic search results.

The external execution tool calls the OpenAI-compatible endpoint:

```text
POST http://127.0.0.1:18008/v1/chat/completions
```

## Required OfferMaster Environment

```env
JOBPILOT_EXTERNAL_AGENT_AUTO_DISPATCH=true
JOBPILOT_CLAUDE_SDK_AGENT_BASE_URL=http://127.0.0.1:18008
JOBPILOT_CLAUDE_SDK_AGENT_API_KEY=<same key as JOBPILOT_LLM_API_KEY or a valid TDL key>
JOBPILOT_CLAUDE_SDK_AGENT_MODEL=qwen-plus
JOBPILOT_CLAUDE_SDK_AGENT_TIMEOUT_SECONDS=300
```

The API key is sent as both `x-api-key` and `Authorization: Bearer ...` by `ClaudeSdkHttpExecutorAdapter`.

For local OfferMaster integration, the vendored `config/service.json` sets `auth.enabled=false`. Keep the service bound to `127.0.0.1` when using this mode. If exposing the service to another host, re-enable auth and pass valid UAC or TDL headers instead of the generic OfferMaster LLM key.

## Starting Locally

Use the helper script from the repository root:

```powershell
.\scripts\start_claude_sdk_agent.ps1
```

By default it prefers the external runtime source at `E:\zte\lastest_bizcore\claude-sdk-agent` when present, and falls back to the vendored source at `vendor\claude-sdk-agent`. It uses the E-drive Python environment created for the runtime:

```text
E:\zte\lastest_bizcore\my-agents\.venv\Scripts\python.exe
```

It also points `CLAUDE_SDK_AGENT_CLI_PATH` at the native Claude executable bundled in that E-drive environment:

```text
E:\zte\lastest_bizcore\my-agents\.venv\Lib\site-packages\claude_agent_sdk\_bundled\claude.exe
```

Do not let the service auto-select `F:\ClaudeCode\node_global\claude.CMD`; the Windows SDK refuses `.cmd` wrappers for safety and returns `CLIConnectionError`.

The script sets `TEMP`, `TMP`, and `PIP_CACHE_DIR` to E-drive paths so dependency/runtime work does not use the C drive. Prefer the external runtime path for normal use so the vendored directory remains a clean source snapshot.

The local `config/service.json` caps external-agent work with `max_turns=8` and `max_thinking_tokens=512`. This gives application-entry discovery enough room to search and summarize while still preventing indefinite runs when the upstream model emits long thinking streams.

## Windows Compatibility Patch

The upstream `claude-sdk-agent` expects symlink permissions when mounting skills/workflows. On this Windows machine, ordinary user permissions cannot create directory symlinks. The vendored snapshot includes the local compatibility patch:

- `src/skills_mount.py`: fallback to `shutil.copytree(...)` when symlink creation fails.
- `src/workflows_mount.py`: fallback to `shutil.copytree(...)` or `shutil.copy2(...)` when symlink creation fails.

This keeps startup working without administrator/developer-mode symlink privileges.

## Current Boundary

This integration enables URL discovery through `claude-sdk-agent`, while ordinary campus-recruiting search is handled by OfferMaster's own `external.web_search` HTTP fallback for lower latency and fewer model hallucinations. It does not by itself give OfferMaster unrestricted local file or desktop-control permissions. Browser/Edge automation should be added as a separate executor with explicit allow/deny rules and human approval gates.
