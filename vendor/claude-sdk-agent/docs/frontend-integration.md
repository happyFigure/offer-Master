# claude-sdk-agent Frontend Integration

## Current status

These backend capabilities are already implemented and can be integrated now:

- `POST /v1/chat/completions`
  - OpenAI-compatible SSE chat endpoint.
  - Main assistant stream may contain `[tool] {...}` shell events inside `delta.content`.
- `GET /v1/runtime/status`
  - Runtime health and SDK option snapshot.
  - Includes MCP autoload status and config directory.
- Session control APIs
  - `GET /v1/sessions/{session_id}`
  - `POST /v1/sessions/{session_id}/interrupt`
  - `GET /v1/sessions/{session_id}/checkpoints`
  - `POST /v1/sessions/{session_id}/checkpoints/{checkpoint_id}/rewind`
- Tool detail/stream APIs
  - `GET /v1/tools/{run_id}/{tool_call_id}`
  - `GET /v1/tools/{run_id}/{tool_call_id}/status/stream`
  - `GET /v1/tools/{run_id}/{tool_call_id}/output/stream`
- Task detail/stream APIs
  - `GET /v1/tasks/{run_id}`
  - `GET /v1/tasks/{run_id}/{task_id}`
  - `GET /v1/tasks/{run_id}/{task_id}/status/stream`
  - `GET /v1/tasks/{run_id}/{task_id}/output/stream`
- Hook detail/stream APIs
  - `GET /v1/sessions/{session_id}/hooks`
  - `GET /v1/sessions/{session_id}/hooks/stream`
  - `GET /v1/sessions/{session_id}/hooks/{event_id}`
- Goal detail/stream APIs
  - `GET /v1/sessions/{session_id}/goal`
  - `GET /v1/sessions/{session_id}/goal/stream`
- Artifact APIs
  - `GET /v1/runs/{run_id}/artifacts`
  - `GET /v1/artifacts/by-session?sessionId={session_id}&limit=50`
  - `GET /v1/artifacts/{artifact_id}/metadata`
  - `POST /v1/artifacts/{artifact_id}/open`
  - `GET /v1/artifacts/{artifact_id}/download`

## Runtime flags

`GET /v1/runtime/status` returns both `sdkOptions` and `featureFlags`.

Current `featureFlags` semantics:

- `auto_interrupt_on_disconnect`
  - backend behavior switch
  - when `true`, client-side SSE disconnect triggers Claude `interrupt()`
- `approval_frontend_enabled`
- `question_frontend_enabled`
- `hook_frontend_enabled`
- `checkpoint_rewind_frontend_enabled`
- `task_panel_frontend_enabled`
- `subagent_events_frontend_enabled`

The frontend-oriented flags above are currently acceptance markers for integration readiness. They do not suppress existing backend endpoints yet.

## Frontend constraints

These are implementation constraints, not optional recommendations.

- Main chat SSE is for assistant text plus lightweight coordination metadata only.
- Detailed runtime state must be fetched from dedicated APIs or dedicated SSE endpoints.
- Frontend must not treat structured shell lines as normal transcript text.
- Frontend must not replay structured shell lines back into model history.
- Frontend must tolerate duplicate shell events and idempotently merge by stable IDs.
- Frontend must tolerate out-of-order arrival between main chat SSE and dedicated SSE/detail APIs.
- Frontend must treat main chat SSE disconnect and dedicated SSE disconnect independently.
- Frontend must treat goal state as session-scoped, not run-scoped.

### Main chat SSE constraints

- `/v1/chat/completions` is the user-facing assistant stream.
- Structured shell payloads inside `delta.content` are summary/index events only.
- Do not place large detail payloads, long logs, full hook bodies, or approval/question/task detail blobs into the main transcript area.
- Use shell payload IDs such as `runId`, `toolCallId`, `taskId`, `requestId`, `questionId`, `eventId` as lookup keys only.
- For goal coordination, use `sessionId + goalId` as the stable key and treat `currentRunId` as the current execution attempt only.

### Dedicated detail channel constraints

- Tool/task output detail belongs to their own `.../output/stream` and `.../status/stream` endpoints.
- Approval/question detail belongs to their session-scoped detail and stream endpoints.
- Hook detail belongs to:
  - `GET /v1/sessions/{session_id}/hooks`
  - `GET /v1/sessions/{session_id}/hooks/stream`
  - `GET /v1/sessions/{session_id}/hooks/{event_id}`
- Goal detail belongs to:
  - `GET /v1/sessions/{session_id}/goal`
  - `GET /v1/sessions/{session_id}/goal/stream`
- If the frontend needs richer cards, timelines, raw payload inspection, or audit views, those must be sourced from these dedicated endpoints rather than the main chat SSE.

### History replay constraints

- When frontend rehydrates conversation history for display, structured shell lines should be hidden or rendered as non-transcript metadata.
- When frontend sends conversation messages back to backend, it must not include prior `[tool]`, `[task]`, `[approval]`, `[question]`, `[goal]`, `[hook]`, `[meta]` lines as assistant content.
- Backend already strips protocol shell lines from replayed assistant history, but frontend must still enforce the same rule to avoid coupling UI correctness to backend cleanup.

## Model routing

`POST /v1/chat/completions` accepts front-end `model`, but routing is normalized before the request reaches Claude SDK:

- empty `model`
- `openclaw:main`
- `claude-code`

all resolve to backend default model from `config/service.json`.

Any other `model` value is forwarded upstream as-is.

Front-end can override the upstream provider per request through `metadata.agentconfig.runtime_config.llm`.
This is used when the selected model is served by a different Anthropic-compatible endpoint than the backend default provider.

Supported fields:

- `base_url` / `api_base`: upstream endpoint or base URL. Values ending in `/v1`, `/v1/messages`, `/v1/messages/count_tokens`, or `/v1/chat/completions` are normalized to the provider base URL before the internal proxy appends `/v1/messages`.
- `api_key`: upstream API key for that provider. It is used only by the internal Anthropic proxy when calling the LLM provider; it does not replace request auth headers or SDK/tool runtime auth environment.
- `anthropic_version`: optional Anthropic version header override; defaults to backend provider config.

Example:

```json
{
  "model": "grok-4.5",
  "metadata": {
    "agentconfig": {
      "runtime_config": {
        "llm": {
          "base_url": "https://example.com/v1",
          "api_key": "<provider-api-key>"
        }
      }
    }
  }
}
```

## Existing event behavior

## Session APIs

### Get session state

`GET /v1/sessions/{session_id}`

Response fields:

- `frontendSessionId`
- `claudeSessionId`
- `model`
- `resumed`
- `createdAt`
- `lastUsedAt`
- `connected`
- `checkpoints`
- `goal`

### Interrupt current execution

`POST /v1/sessions/{session_id}/interrupt`

Current backend behavior:

- interrupts the active Claude SDK client for the session
- useful for future frontend "stop generating" button

### Stop button compatibility

`my-agents` backend chat stop behavior is effectively SSE cancellation, not a dedicated chat interrupt API.

- `_safe_sse_stream()` logs `stream-cancelled` on client disconnect
- `/v1/tools/cancel` is only for tool execution cancellation

`claude-sdk-agent` keeps that compatibility and adds a backend enhancement:

- if the chat SSE disconnects and `featureFlags.auto_interrupt_on_disconnect=true`, backend also calls Claude `interrupt()` for the active session

### List rewind checkpoints

`GET /v1/sessions/{session_id}/checkpoints`

Checkpoint fields:

- `frontend_session_id`
- `claude_session_id`
- `checkpoint_id`
- `created_at`
- `prompt_excerpt`

### Rewind files to checkpoint

`POST /v1/sessions/{session_id}/checkpoints/{checkpoint_id}/rewind`

Current backend behavior:

- rewinds tracked files for that Claude session
- requires backend config `enable_file_checkpointing=true`

### Chat SSE

`/v1/chat/completions` follows OpenAI chat completion chunks.

In addition to normal assistant text, the stream may include protocol payloads in `choices[0].delta.content`:

- `[tool] {...}`
- `[task] {...}`
- `[approval] {...}`
- `[question] {...}`
- `[goal] {...}`
- `[hook] {...}`
- `[meta] {...}`
- `[artifacts] {...}`

Frontend should parse these lines as structured runtime events and must not send them back to the model as plain history content.

### Main chat shell payloads

These shell payloads are intentionally lightweight.
Frontend should treat them as event headers and then use the returned IDs to fetch detailed data from dedicated APIs.

`[meta] {...}`

Current minimum payload:

```json
{
  "runId": "run-123",
  "sessionId": "session:sess_xxx:user:10154402",
  "model": "MiniMax-M2.7"
}
```

Usage:

- initialize current `runId`
- bind the chat turn to one execution run before task/approval/question details arrive

### Artifact Panel

Claude SDK Agent artifacts use the same frontend contract as MyAgents. The feature is disabled by default and is enabled per chat request:

```json
{
  "metadata": {
    "agentconfig": {
      "artifacts_enabled": true
    }
  }
}
```

Compatible forms are also accepted: `artifactsEnabled: true` and `artifacts: {"enabled": true}`.

When enabled, the backend records files reported by Claude Agent SDK file-edit events for the current run. It does not perform filesystem snapshot diffing. The source field is `sdk_affected_files`; changed files are resolved under the effective Claude workspace `cwd` and `add_dirs`. Files outside those roots are ignored and listed in the run `errors` array.

For streaming chat, the main SSE may emit:

```text
[artifacts] {"type":"artifacts.updated","sessionId":"session-1","runId":"run-abc","artifactCount":1,"summary":{"created":0,"modified":1,"deleted":0,"artifactCount":1,"truncated":false},"url":"http://host/v1/runs/run-abc/artifacts"}
```

Frontend behavior:

- Treat `[artifacts]` as transport metadata, not assistant transcript text.
- Refresh the artifact panel through `url` or `GET /v1/runs/{runId}/artifacts`.
- For non-streaming requests, read `x-agent-run-id` and `x-artifacts-count` when artifacts are enabled.
- Do not implement file-format preview rules in the panel. Use `availableActions`; `open` delegates to the local PC default application through the backend.

Artifact records follow `agent.artifacts/v1`:

```json
{
  "artifactId": "art_xxx",
  "sessionId": "session-1",
  "runId": "run-abc",
  "path": "/workspace/report.md",
  "relativePath": "report.md",
  "root": "/workspace",
  "rootRole": "workspace",
  "kind": "file",
  "changeType": "modified",
  "source": "sdk_affected_files",
  "confidence": "high",
  "size": 123,
  "mtimeNs": 1760000000000000000,
  "mimeType": "text/markdown",
  "availableActions": ["open", "download"]
}
```

`[tool] {...}`

Current minimum payload:

```json
{
  "phase": "start",
  "runId": "run-123",
  "toolCallId": "tool-1",
  "name": "bash",
  "display_name": "bash",
  "status": "running",
  "toolType": "claude_task",
  "arguments": {
    "command": "echo hi"
  }
}
```

Usage:

- create/update the tool shell card in chat
- use `runId + toolCallId` to subscribe to:
  - `GET /v1/tools/{run_id}/{tool_call_id}/status/stream`
  - `GET /v1/tools/{run_id}/{tool_call_id}/output/stream`

`[task] {...}`

Current minimum payload:

```json
{
  "phase": "start",
  "runId": "run-123",
  "taskId": "task-1",
  "taskType": "bash",
  "status": "running",
  "toolCallId": "tool-1",
  "name": "bash task"
}
```

Optional fields:

- `log`
- `result`

Usage:

- create/update task list items for the current run
- use `runId + taskId` to subscribe to:
  - `GET /v1/tasks/{run_id}/{task_id}/status/stream`
  - `GET /v1/tasks/{run_id}/{task_id}/output/stream`

`[approval] {...}`

Current minimum payload:

```json
{
  "sessionId": "session:sess_xxx:user:10154402",
  "runId": "run-123",
  "requestId": "req-1",
  "status": "pending",
  "toolName": "Bash"
}
```

Possible detailed fields from approval APIs:

- `claudeSessionId`
- `toolInput`
- `toolUseId`
- `agentId`
- `blockedPath`
- `decisionReason`
- `title`
- `displayName`
- `description`

Usage:

- create/update approval card
- use `sessionId + requestId` to fetch:
  - `GET /v1/sessions/{session_id}/approvals/{request_id}`
- submit decision:
  - `POST /v1/sessions/{session_id}/approvals/{request_id}`

`[question] {...}`

Current minimum payload:

```json
{
  "sessionId": "session:sess_xxx:user:10154402",
  "runId": "run-123",
  "questionId": "question-1",
  "requestId": "request-1",
  "status": "pending",
  "prompt": "Need more detail?"
}
```

Possible detailed fields from question APIs:

- `claudeSessionId`
- `title`
- `description`
- `metadata`

Usage:

- create/update question card
- use `sessionId + questionId` to fetch:
  - `GET /v1/sessions/{session_id}/questions/{question_id}`
- submit answer:
  - `POST /v1/sessions/{session_id}/questions/{question_id}`

`[goal] {...}`

Current minimum payload:

```json
{
  "sessionId": "session:sess_xxx:user:10154402",
  "goalId": "goal-1",
  "status": "active",
  "currentRunId": "run-123",
  "condition": "finish deployment",
  "turnCount": 1,
  "pendingApproval": false,
  "pendingQuestion": false,
  "stopHookActive": true
}
```

Possible extra summary fields:

- `lastReason`

Usage:

- create/update a session-scoped goal card
- treat `goalId` as the stable goal identity
- treat `currentRunId` as the current execution attempt only
- fetch detail from:
  - `GET /v1/sessions/{session_id}/goal`
- subscribe to:
  - `GET /v1/sessions/{session_id}/goal/stream`

Frontend constraints:

- a goal is not a tool and not a task
- a goal belongs to the session, not to a specific run
- do not key goal state by `runId` alone
- when `pendingApproval=true` or `pendingQuestion=true`, render the goal as waiting on user interaction

`[hook] {...}`

Current minimum payload:

```json
{
  "eventId": "hook-1",
  "sessionId": "session:sess_xxx:user:10154402",
  "runId": "run-123",
  "claudeSessionId": "claude-1",
  "hookEventName": "Notification",
  "phase": "hook_response",
  "source": "sdk_stream",
  "status": "completed"
}
```

Possible extra summary fields:

- `toolName`
- `toolUseId`
- `agentId`
- `agentType`
- `title`
- `notificationType`

Usage:

- create/update lightweight hook entries in chat
- use `sessionId + eventId` to fetch:
  - `GET /v1/sessions/{session_id}/hooks/{event_id}`
- subscribe to:
  - `GET /v1/sessions/{session_id}/hooks/stream`

### Goal detail stream

`GET /v1/sessions/{session_id}/goal/stream`

SSE events:

- `event: goal`
  - payload fields:
    - `sessionId`
    - `goalId`
    - `currentRunId`
    - `status`
    - `command`
    - `condition`
    - `createdAt`
    - `updatedAt`
    - `completedAt`
    - `clearedAt`
    - `lastReason`
    - `stopHookActive`
    - `turnCount`
    - `pendingApproval`
    - `pendingQuestion`
    - `metadata`
- `event: end`
  - payload fields:
    - `sessionId`
    - `streamType`
    - `streamStatus`

Recommended frontend state mapping:

- `active`
  - goal is currently being pursued
- `waiting_input`
  - goal is blocked on approval or question response
- `completed`
  - goal finished successfully
- `cleared`
  - user explicitly cleared the goal
- `failed`
  - reserved for future backend use

### Tool status stream

`GET /v1/tools/{run_id}/{tool_call_id}/status/stream`

SSE events:

- `event: status`
  - payload fields:
    - `runId`
    - `toolCallId`
    - `name`
    - `displayName`
    - `toolType`
    - `arguments`
    - `status`
    - `startedAt`
    - `finishedAt`
- `event: end`
  - payload fields:
    - `runId`
    - `toolCallId`
    - `status`
    - `streamStatus`
    - `finishedAt`

### Tool output stream

`GET /v1/tools/{run_id}/{tool_call_id}/output/stream`

SSE events:

- `event: system`
- `event: stdout`
- `event: stderr`
- `event: end`

Chunk payload fields:

- `sequence`
- `stream`
- `text`
- `timestamp`

### Task status stream

`GET /v1/tasks/{run_id}/{task_id}/status/stream`

SSE events:

- `event: status`
  - payload fields:
    - `runId`
    - `taskId`
    - `description`
    - `taskType`
    - `toolCallId`
    - `status`
    - `startedAt`
    - `finishedAt`
    - `metadata`
- `event: end`
  - payload fields:
    - `runId`
    - `taskId`
    - `status`
    - `streamStatus`
    - `finishedAt`

### Task output stream

`GET /v1/tasks/{run_id}/{task_id}/output/stream`

SSE events:

- `event: system`
- `event: stdout`
- `event: stderr`
- `event: end`

Chunk payload fields:

- `sequence`
- `stream`
- `text`
- `timestamp`

## Frontend work needed next

These capabilities require frontend development.

### 1. Approval / permission UI

Planned backend contract:

- SSE stream for pending permission requests
- `POST` API for approve / deny

Recommended contract:

- `GET /v1/sessions/{session_id}/approvals`
- `GET /v1/sessions/{session_id}/approvals/stream`
- `GET /v1/sessions/{session_id}/approvals/{request_id}`
- `POST /v1/sessions/{session_id}/approvals/{request_id}`

Suggested request body:

```json
{
  "decision": "allow"
}
```

or

```json
{
  "decision": "deny",
  "reason": "optional text"
}
```

Frontend requirement:

- Show blocking approval card
- Allow user approve/deny per request
- Preserve request order
- `request_id` source:
  - primary: approval SSE event payload `requestId`
  - fallback: `GET /v1/sessions/{session_id}/approvals`
- enable `featureFlags.approval_frontend_enabled` after integration acceptance

### 2. AskUserQuestion / interactive follow-up

Current backend contract:

- SSE stream for question events
- `POST` API to submit user answer

Recommended contract:

- `GET /v1/sessions/{session_id}/questions`
- `GET /v1/sessions/{session_id}/questions/stream`
- `GET /v1/sessions/{session_id}/questions/{question_id}`
- `POST /v1/sessions/{session_id}/questions/{question_id}`

Suggested request body:

```json
{
  "answer": "user reply"
}
```

Frontend requirement:

- Render question card or inline follow-up prompt
- Support free-text answer submission
- `question_id` / `request_id` source:
  - primary: question SSE event payload `questionId` / `requestId`
  - fallback: `GET /v1/sessions/{session_id}/questions`
- enable `featureFlags.question_frontend_enabled` after integration acceptance

Current note:

- backend APIs and SSE are implemented
- current deployed Claude SDK version does not yet emit a native AskUserQuestion control request in this service path
- frontend can integrate against the contract now; server-side producer wiring can be enabled once the SDK/runtime exposes that event source

### 3. File checkpoint / rewind

Planned backend contract:

- `GET /v1/sessions/{session_id}/checkpoints`
- `POST /v1/sessions/{session_id}/checkpoints/{checkpoint_id}/rewind`

Frontend requirement:

- Show checkpoint timeline
- Ask user to confirm rewind
- Refresh file/task/tool state after rewind
- enable `featureFlags.checkpoint_rewind_frontend_enabled` after integration acceptance

### 4. Task panel

Current backend contract already exists:

- `GET /v1/tasks/{run_id}`
- `GET /v1/tasks/{run_id}/{task_id}`
- `GET /v1/tasks/{run_id}/{task_id}/status/stream`
- `GET /v1/tasks/{run_id}/{task_id}/output/stream`

Frontend requirement:

- show per-run task list
- subscribe to task status and output SSE
- correlate task entries with tool shells shown in the main chat stream
- enable `featureFlags.task_panel_frontend_enabled` after integration acceptance

### 5. Subagent visualization

Possible future contract:

- main chat SSE `[subagent] {...}`
- or dedicated SSE:
  - `GET /v1/sessions/{session_id}/subagents/stream`

Frontend requirement:

- group output by subagent
- show subagent start / stop / status
- enable `featureFlags.subagent_events_frontend_enabled` after integration acceptance

## Backend-only roadmap

These do not require frontend changes and can continue independently:

- move from `query()+resume` to long-lived `ClaudeSDKClient`
- checkpoint metadata persistence and rewind backend
- hook-based policy / audit module
- custom MCP / business tool bridging

## Workspace runtime discovery

Use `POST /v1/runtime/workspace/inspect` before starting a chat or when the ordered workspace selection changes:

```json
{
  "workspace": {
    "source": "agent",
    "cwd": "/srv/project",
    "add_dirs": ["/srv/shared"]
  }
}
```

The response intentionally has separate layers:

- `agentPolicy`: Agent-level settings and hard runtime limits.
- `resources`: redacted declarations found in the primary workspace, additional access roots, and Agent mounts.
- `effectiveRuntime`: expected Claude Code resolution before connecting.
- `observed`: commands and MCP status reported by the connected SDK; unavailable during static preview.
- `conflicts`: name collisions and Agent policy caps with explicit resolution reasons.

Only `cwd` is a native project configuration root. `addDirs` do not contribute settings, hooks, permissions, or MCP declarations. The frontend should label them as additional access directories rather than equivalent workspaces.

Resource discovery follows Claude Code's official workspace resource boundary. Native workflow rows only come from `.claude/workflows/*.js`; non-native files such as `.flow/workflows/*.workflow.json` are not returned as Claude Code workflow resources. Output styles are exposed under `resources.outputStyles`.

For dynamic command suggestions, merge the existing built-in command list with active `commands.items`. Send the selected item's `invoke` object as `metadata.runtimeCommand`; put user arguments in `invoke.args.text`. Do not reconstruct the protocol from display text. Workflow entries use `kind=workflow`, `requiresConfirmation=false`, and `launchMode=sdk_immediate`; the frontend can run them without a start-confirmation modal, while tool-level permission prompts continue through the existing approval flow.

### Agent-scoped runtime permissions

The permission profile belongs to the Service-owned Agent configuration, not to a session or to the Claude Runtime process. Its only persistent source is:

```text
agent_configs.advanced_config.permissions
```

The `/permissions` Slash Command reads and updates the profile through the Service API:

```http
GET /api/sessions/runtime-permissions?agentId=asc_agent_config_id
PUT /api/sessions/runtime-permissions?agentId=asc_agent_config_id
```

Update it with optimistic concurrency:

```json
{
  "profile": "safe",
  "expectedRevision": 3
}
```

The response includes `scope=agent`, `runtimeKey`, `revision`, and `current`. A stale `expectedRevision` returns HTTP 409 with `expectedRevision` and `actualRevision`. Omitting `expectedRevision` preserves backward-compatible last-write-wins behavior for older clients. Different Agent rows hold independent profiles; sessions using the same Agent intentionally read the same latest profile on their next operation.

The Service projects the persisted value into `metadata.agentconfig.runtime_config.permissions` for chat and into root `runtime_config.permissions` for workspace inspection and checkpoint recovery. Persisted Service state overrides client drafts. The Claude Runtime only consumes this request-scoped snapshot; direct `GET/PUT /v1/runtime/permissions` endpoints and `permission-profile*.json` persistence do not exist.

Permission is independent of the Advanced System Config master switch because it is managed by the Slash Command. Skill isolation is different: it belongs to Advanced System Config and is omitted when `advancedConfig.enabled=false`.

The profile mapping is:

- `readonly -> plan`
- `safe -> default`
- `edit -> acceptEdits`
- `auto -> auto`
- `dontAsk -> dontAsk`
- `bypass -> bypassPermissions`
- `fullBypass -> bypassPermissions` plus cleared Agent tool lists and the CLI dangerous bypass request

Except for `fullBypass`, these entries must retain Claude Code's native semantics. In particular, the persisted compatibility id `readonly` means native Plan Mode, not permanent read-only access: the planning phase uses read-only tools, while approving `ExitPlanMode` exits Plan Mode and can continue with file modifications under the selected execution permission. Frontends must label it **规划模式**, not **只读**.

`fullBypass` is a product profile, not a stronger Claude Code native permission mode. Claude Code still evaluates blocking Hooks, deny/ask rules, and `disableBypassPermissionsMode` before or around native bypass handling. Keep `fullBypass` warnings visible, and use `effectiveRuntime.permission.fullBypass.enforcementStatus` and `limitations` from workspace inspection instead of claiming unconditional authorization. Selectively ignoring those settings while retaining project commands, Skills, and CLAUDE.md requires a separate resource-loading adapter; disabling `setting_sources` is not equivalent because it also removes those resources.

Frontend labels must resolve the product `profile` before falling back to `permissionMode`, because both bypass profiles report `bypassPermissions`:

- `bypass`: **Claude Code 原生绕过**; keep the Agent tool allow/deny lists.
- `fullBypass`: **清除 Agent 限制并绕过**; clear the Agent tool allow/deny lists, but do not claim that workspace or Claude Code hard protections are disabled.

Workspace inspection is read-only. It reports the Agent runtime key/revision, concrete permission rules and their sources, settings default modes that were ignored by Agent mode precedence, and the native evaluation order. It never writes either the Agent profile or workspace settings. A profile revision is part of the SDK client signature, so a successful profile update rebuilds the affected Agent session client on its next runtime operation.

## Important integration note

The frontend must not replay protocol shell lines such as:

- `[tool] {...}`
- `[task] {...}`
- `[approval] {...}`
- `[question] {...}`
- `[goal] {...}`
- `[hook] {...}`
- `[meta] {...}`

back into the model context as plain assistant history. Backend already strips these lines from replayed assistant history, but frontend should still treat them as transport metadata rather than user-visible transcript content.

## Main chat structured shells

The main `/v1/chat/completions` SSE may include lightweight shell lines in assistant content.
These are transport metadata for frontend coordination.

Current shell formats:

- `[tool] {...}`
- `[task] {...}`
- `[approval] {...}`
- `[question] {...}`
- `[goal] {...}`
- `[hook] {...}`
- `[meta] {...}`

Emission rule:

- `[tool] {...}` is always enabled because the current frontend already depends on it
- `[task] {...}` is emitted only when `featureFlags.task_panel_frontend_enabled=true`
- `[approval] {...}` is emitted only when `featureFlags.approval_frontend_enabled=true`
- `[question] {...}` is emitted only when `featureFlags.question_frontend_enabled=true`
- `[goal] {...}` is emitted whenever a goal state change is produced for the session
- `[hook] {...}` is emitted only when `featureFlags.hook_frontend_enabled=true`
- `[meta] {...}` is emitted only when any of the structured frontend flags above are enabled

Recommended frontend behavior:

- parse shell lines from the main chat SSE
- store IDs from shell payloads
- fetch detailed task / approval / question data from dedicated APIs using those IDs
- fetch detailed goal data from dedicated APIs using `sessionId`
- fetch detailed hook data from dedicated APIs using those IDs
- do not render shell lines as normal assistant text

Recommended parser rule:

- if `delta.content` line starts with `[tool] `
  - parse trailing JSON as tool shell payload
- if `delta.content` line starts with `[task] `
  - parse trailing JSON as task shell payload
- if `delta.content` line starts with `[approval] `
  - parse trailing JSON as approval shell payload
- if `delta.content` line starts with `[question] `
  - parse trailing JSON as question shell payload
- if `delta.content` line starts with `[goal] `
  - parse trailing JSON as goal shell payload
- if `delta.content` line starts with `[hook] `
  - parse trailing JSON as hook shell payload
- if `delta.content` line starts with `[meta] `
  - parse trailing JSON as meta shell payload
- otherwise treat it as normal assistant text

Recommended frontend execution flow:

1. open chat SSE
2. read `[meta]` first if present and store `runId`
3. when `[tool]` arrives, create tool shell card
4. when `[task]` arrives, create/update task item and optionally subscribe to task SSE
5. when `[approval]` arrives, fetch full approval payload if needed and show approval card
6. when `[question]` arrives, fetch full question payload if needed and show follow-up card
7. when `[goal]` arrives, fetch full goal payload if needed and update the session goal card
8. when `[hook]` arrives, fetch full hook payload if needed and update hook timeline
9. on future history replay, never send shell lines back as plain assistant content

## ID sources

- `session_id`
  - frontend-provided request field
- `run_id`
  - main chat SSE `[tool] {...}` payload `runId`
  - task/tool status payloads also include `runId`
- `tool_call_id`
  - main chat SSE `[tool] {...}` payload `toolCallId`
- `task_id`
  - task SSE payload `taskId`
  - fallback: `GET /v1/tasks/{run_id}`
- `request_id`
  - approval SSE payload `requestId`
  - question SSE payload `requestId`
- `question_id`
  - question SSE payload `questionId`
  - fallback: `GET /v1/sessions/{session_id}/questions`
- `goal_id`
  - main chat SSE `[goal] {...}` payload `goalId`
  - fallback: `GET /v1/sessions/{session_id}/goal`
- `checkpoint_id`
  - `GET /v1/sessions/{session_id}/checkpoints`
