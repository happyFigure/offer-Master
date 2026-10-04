# Filesystem Model Autonomy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 让模型在读取文件后可以继续自主选择 copy/rename 并生成文件名，同时让路由层只选择能力、runtime 只执行安全校验和审批。

**Architecture:** 保留 `skill.filesystem` 作为粗粒度能力路由；由 `ToolChoiceLoopRunner` 驱动连续模型决策，工具结果和可恢复参数错误都写回模型上下文；`FilesystemSkillExecutor` 和 runtime guard 负责 Schema、路径、审批、脚本和 postcheck。

**Tech Stack:** Python 3.11、Pydantic、SQLAlchemy、pytest、React/Vite。

**Spec:** `C:/Users/phoenix/Documents/Obsidian Vault/秋招助手/开发/开发目标/给文件命名有问题/OfferMaster路由与模型自治修改方案.md`

## Global Constraints

- 模型负责理解文件内容、选择下一步工具和创作文件名。
- 路由层只决定 filesystem、browser、database 或 external agent。
- runtime 只负责 Schema 校验、路径边界、权限、审批、执行和 postcheck。
- 禁止通过关键词、正则或 assistant 普通文本猜测 dst 或文件名。
- `content_based` 命名必须提交结构化 `name_intent`。
- 写操作必须经过用户确认。

## Review Focus

- `read_file` 成功后仍有复制/重命名目标时，模型必须获得下一次决策机会。
- 缺少 `dst`、`name_intent` 等模型参数时，runtime 必须返回可恢复错误而不是直接向用户结束。
- 普通文本中的伪工具调用不得被执行。
- 路由层缺少具体 operation 时不得自行猜测 operation。
- 文件名安全校验和真实执行后的 postcheck 不能被连续 loop 绕过。

---

### Task 1: Make tool observations recoverable

**Files:**
- Modify: `apps/api/app/agent_runtime/loop_agent/tool_choice_runner.py`
- Test: `tests/test_loop_agent_tool_choice_runner.py`

**Interfaces:**
- Consumes `AgentToolDefinition` schemas and model tool calls.
- Produces `LoopAgentObservation(status="failed", metadata.recoverable=True)` for model-fixable input errors so the controller continues.

- [ ] Write a failing test proving a missing required tool field is returned to the model and the second model call can provide the field.
- [ ] Run the focused test and confirm it fails because the current runner stops with `WAITING_USER` or `STEP_FAILED` before the second model call.
- [ ] Change model-fixable schema errors to a recoverable failed observation with `error_code`, `missing_required_fields`, `recoverable=True`, and `next_action="continue_model_loop"`.
- [ ] Keep genuine user input gaps as `waiting_user` only when the tool definition marks the missing value as user-owned or the model has exhausted recoveries.
- [ ] Add structured continuation logging metadata without including file contents.
- [ ] Run the focused runner tests and the full runner test file.

### Task 2: Remove textual tool-call recovery

**Files:**
- Modify: `apps/api/app/agent_runtime/loop_agent/tool_choice_runner.py`
- Modify: `tests/test_loop_agent_tool_choice_runner.py`
- Modify: `tests/test_frontend_agent_chat_api.py` if its assertions describe the removed path.

**Interfaces:**
- Only provider-native structured `tool_calls` may produce `LoopAgentDecision(action=CALL_TOOL)`.
- Plain assistant text remains a final answer and is never parsed into executable arguments.

- [ ] Replace textual-recovery tests with a regression test asserting a plain-text `Tool call` response is not executed.
- [ ] Run the focused tests and confirm the old tests fail or expose the old recovery behavior.
- [ ] Remove textual parsing helpers, regexes, and metadata from the production loop runner.
- [ ] Update user-facing timeline expectations to report that plain text was not executed, if that UI contract remains.
- [ ] Run all loop runner and API timeline tests.

### Task 3: Keep capability routing coarse-grained

**Files:**
- Modify: `apps/api/app/agent_runtime/routing/capability_routing_middleware.py`
- Test: `tests/test_capability_routing_middleware.py`

**Interfaces:**
- Filesystem routing returns `skill.filesystem` as the capability.
- Structured `filesystem_operation` and `operation_intent` are passed through unchanged when present.
- Missing operation is not converted by routing into a guessed operation or filename.

- [ ] Add a failing route test for a filesystem request with no structured operation; assert the decision preserves the filesystem capability and marks the missing operation as recoverable model work.
- [ ] Run the focused middleware test and confirm the current implementation returns a clarification route.
- [ ] Change the decision to preserve the coarse filesystem route and attach `structured_operation_required` metadata without inventing an operation.
- [ ] Ensure explicit browser, database, and external-agent routes retain their current behavior.
- [ ] Run the routing test file.

### Task 4: Enforce structured content-based naming at runtime

**Files:**
- Modify: `apps/api/app/agent_runtime/skills/filesystem_executor.py`
- Modify: `apps/api/app/agent_runtime/context/file_operation_policy.py`
- Test: `tests/test_filesystem_skill_executor.py`
- Test: `tests/test_file_operation_policy.py`

**Interfaces:**
- `name_policy=content_based` requires model-provided `name_intent`.
- Runtime rejects placeholders and unsafe paths but never invents a semantic filename.

- [ ] Add failing tests for `content_based` copy and rename without `name_intent`; assert a recoverable structured error and no script execution.
- [ ] Run the focused tests and confirm the current executor/policy either returns an ambiguous generic error or executes the wrong fallback.
- [ ] Implement `CONTENT_BASED_NAME_REQUIRED` and preserve `next_action="continue_model_loop"`.
- [ ] Keep concrete model-provided names flowing through normalization, approval, script execution, and postcheck.
- [ ] Run the filesystem executor and policy test files.

### Task 5: Cover the full read-then-name workflow

**Files:**
- Modify: `apps/api/app/agent_runtime/graph_factory.py` only where the current state machine prevents continuation.
- Modify: `tests/test_agent_runtime_graph.py`.
- Modify: `tests/test_agent_api.py` only where response-mode expectations are stale.

**Interfaces:**
- A model can call `read_file`, receive content, then call `copy_file` or `rename_file` with `name_intent` in the same run.
- The final response reports the actual postchecked path.

- [ ] Add or repair a graph regression test with two native model tool calls: read, then copy/rename with a content-based name.
- [ ] Run it before implementation and record the failing stage, response mode, and missing state transition.
- [ ] Implement only the state transition or continuation metadata needed to return to the model.
- [ ] Update stale tests to explicit structured arguments instead of old prose inference.
- [ ] Run `tests/test_agent_runtime_graph.py` and then the full Python suite.

### Task 6: Verify build, stale-link removal, and documentation

**Files:**
- Modify: `C:/Users/phoenix/Documents/Obsidian Vault/秋招助手/开发/开发目标/给文件命名有问题/OfferMaster文件命名开发记录.md`
- Modify: `C:/Users/phoenix/Documents/Obsidian Vault/秋招助手/开发/开发目标/给文件命名有问题/OfferMaster文件命名问题根因分析.md` only if the implementation evidence changes the diagnosis.

- [ ] Run production-code grep proving removed legacy symbols are absent.
- [ ] Run Python compileall, full pytest, frontend build, and git diff check.
- [ ] Record the actual implementation decisions, test evidence, and any remaining gaps in the development log.
- [ ] Restart backend and frontend only after code verification, then run real frontend regression for filesystem and non-filesystem routes.
