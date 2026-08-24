from __future__ import annotations

import asyncio
import hashlib
import json
import importlib
import inspect
import logging
import os
import sys
import time
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Mapping, Sequence

from ..approval_control import ApprovalRuntimeRegistry
from ..api.payload import build_system_prompt, extract_artifacts_enabled, extract_runtime_command, extract_session_id
from ..artifacts import ClaudeArtifactService
from ..hook_control import HookRuntimeRegistry
from ..multimodal.sdk_input import build_runtime_command_prompt_input, build_sdk_prompt_input
from ..config import ClaudeSettings, McpSettings, ProviderSettings, SkillUsageAuditSettings
from ..mcp_loader import build_mcp_servers
from ..model_routing import resolve_effective_model
from ..permission_profile import (
    PermissionOptions,
    permission_options_from_runtime_config,
)
from ..provider.context_store import ProxyContextStore
from ..question_control import QuestionRuntimeRegistry
from ..session.checkpoint_store import SessionCheckpointStore
from ..session.goal_store import SessionGoalStore
from ..session.models import SessionGoal
from ..session.store import SessionMappingStore
from ..skill_usage_audit import SkillUsageAuditor
from ..stream_types import RuntimeStreamEvent
from ..task_control import TaskControlContext, TaskRuntimeRegistry
from ..tool_control import ToolControlContext, ToolRuntimeRegistry
from ..workspace_runtime import inspect_workspace_runtime, merge_observed_runtime
from .client_pool import ClaudeClientPool
from .hooks import build_sdk_hooks, hook_stream_payload
from .stream_adapter import (
    assistant_text,
    content_tool_results,
    content_tool_starts,
    event_to_text_delta,
    extract_session_id as extract_claude_session_id,
    final_result_error_text,
    is_assistant_message,
    is_compact_boundary,
    is_final_result,
    item_tool_file_paths,
    stream_event_tool_start,
    task_message_payload,
)

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.environ.get(name, "") or default))
    except (TypeError, ValueError):
        return default


_TASK_TERMINAL_STATUSES = {
    "completed",
    "failed",
    "failure",
    "error",
    "stopped",
    "killed",
    "cancelled",
    "canceled",
    "ended",
}
_SUBTASK_WAIT_TIMEOUT_SECONDS = _env_float("CLAUDE_SUBTASK_WAIT_TIMEOUT_SECONDS", 3600.0)
_SUBTASK_WAIT_POLL_SECONDS = max(0.1, _env_float("CLAUDE_SUBTASK_WAIT_POLL_SECONDS", 1.0))
_SUBTASK_WAIT_HEARTBEAT_SECONDS = max(1.0, _env_float("CLAUDE_SUBTASK_WAIT_HEARTBEAT_SECONDS", 15.0))
_TASK_STREAM_WAITING_LOG = "主响应流已结束，正在等待后台子任务完成。"
_TASK_STREAM_STILL_WAITING_LOG = "主响应流已结束，仍在等待后台子任务完成。"
_TASK_STREAM_ENDED_LOG = "主响应流已结束，等待后台子任务超时，未收到该子任务的最终完成状态。"
_SUBTASK_TIMEOUT_TEXT = (
    "后台子任务仍未返回最终完成状态，本轮已达到等待上限。"
    "当前回复不是最终汇总；如需继续，请发送“继续”，我会基于同一 Claude Code 会话继续检查。"
)
_SUBTASK_SUMMARY_PROMPT = (
    "后台子任务已经全部结束。请不要再启动新的子任务、工具调用或后台任务；"
    "请基于刚才各子任务返回的结果，直接给用户输出最终汇总。"
    "汇总需要说明：已完成什么、每个子任务的关键发现、仍未完成或不确定的事项、下一步建议。"
)
_DESTRUCTIVE_COMMAND_SYSTEM_WARNING = (
    "安全要求：禁止执行可能破坏系统、用户目录或挂载视图的高危删除/清理命令。"
    "不要对根目录、系统目录、用户主目录、挂载目录或平台文件视图执行递归删除、批量删除或 find -delete，"
    "包括但不限于 /、/home、/root、/etc、/usr、/var、/tmp 的全量清理以及 /.ft、/.ft/e 及其子路径。"
    "尤其不要执行 rm -rf /.ft/e/*，该路径可能是根文件系统的挂载视图，不是普通缓存目录。"
    "清理磁盘空间前必须先定位具体大文件或明确的工作区临时目录，只能删除用户明确指定且确认安全的项目文件。"
)
_SENSITIVE_ENV_MARKERS = ("TOKEN", "KEY", "SECRET", "PASSWORD", "AUTHORIZATION")
_CLI_STDERR_MAX_CHARS = 4000


@dataclass(slots=True)
class _SessionExecutionContext:
    run_id: str
    claude_session_id: str
    last_checkpoint_id: str = ""


@dataclass(frozen=True, slots=True)
class _WorkspaceExecution:
    cwd: Path
    add_dirs: tuple[Path, ...] = ()
    source: str = "agent_default"
    configured: bool = False


@dataclass(frozen=True, slots=True)
class _RuntimeProviderConfig:
    base_url: str
    anthropic_version: str
    api_token: str
    source: str = "service"


class ClaudeSdkRuntime:
    def __init__(
        self,
        settings: ClaudeSettings,
        mcp_settings: McpSettings,
        provider: ProviderSettings,
        session_store: SessionMappingStore,
        checkpoint_store: SessionCheckpointStore,
        proxy_contexts: ProxyContextStore,
        client_pool: ClaudeClientPool,
        tool_registry: ToolRuntimeRegistry | None = None,
        task_registry: TaskRuntimeRegistry | None = None,
        approval_registry: ApprovalRuntimeRegistry | None = None,
        question_registry: QuestionRuntimeRegistry | None = None,
        goal_store: SessionGoalStore | None = None,
        hook_registry: HookRuntimeRegistry | None = None,
        workflow_mount_root: Path | None = None,
        skill_usage_audit: SkillUsageAuditSettings | None = None,
        artifact_service: ClaudeArtifactService | None = None,
    ) -> None:
        self._settings = settings
        self._mcp_settings = mcp_settings
        self._provider = provider
        self._session_store = session_store
        self._checkpoint_store = checkpoint_store
        self._proxy_contexts = proxy_contexts
        self._client_pool = client_pool
        self._tool_registry = tool_registry
        self._task_registry = task_registry
        self._approval_registry = approval_registry or ApprovalRuntimeRegistry()
        self._question_registry = question_registry or QuestionRuntimeRegistry()
        self._goal_store = goal_store or SessionGoalStore()
        self._hook_registry = hook_registry or HookRuntimeRegistry()
        self._workflow_mount_root = workflow_mount_root
        self._artifact_service = artifact_service
        self._skill_usage_audit = skill_usage_audit or SkillUsageAuditSettings(
            enabled=False,
            base_url="",
            endpoint="",
            timeout_sec=3.0,
        )
        self._session_contexts: dict[str, _SessionExecutionContext] = {}

    async def stream_events(
        self,
        payload: Mapping[str, Any],
        *,
        request_headers: Mapping[str, str],
        current_user: Any | None,
        fallback_tdl_api_key: str = "",
        proxy_base_url: str,
        skill_mount_root,
        run_id: str | None = None,
    ) -> AsyncIterator[RuntimeStreamEvent]:
        frontend_session_id = extract_session_id(payload) or f"ephemeral:{run_id or uuid.uuid4().hex}"
        runtime_command = extract_runtime_command(payload)
        artifact_started_at = time.time()
        artifacts_enabled = extract_artifacts_enabled(payload)
        if runtime_command is None:
            recovery_goal = await self._goal_store.get(frontend_session_id)
            recovery_client = await self._client_pool.get(frontend_session_id) if frontend_session_id else None
            if recovery_goal is not None and _goal_needs_recovery_decision(recovery_goal, has_active_client=recovery_client is not None):
                if recovery_goal.status != "paused":
                    recovery_goal = await self._goal_store.pause(frontend_session_id, reason="process_interrupted") or recovery_goal
                yield RuntimeStreamEvent(
                    kind="command",
                    payload=_goal_runtime_command_payload(
                        status="running",
                        phase="goal_recovering",
                        message="检测到当前会话存在未完成目标，正在继续推进该目标。",
                        run_id=str(run_id or "").strip(),
                    ),
                )
                await self._goal_store.mark_run_started(frontend_session_id, str(run_id or "").strip())
                yield RuntimeStreamEvent(
                    kind="command",
                    payload=_goal_runtime_command_payload(
                        status="completed",
                        phase="goal_resumed",
                        message="目标已恢复，将继续围绕该目标推进。",
                        result="goal_resumed",
                        run_id=str(run_id or "").strip(),
                    ),
                )
        goal_command_text = ""
        if runtime_command is not None and runtime_command.command_id == "goal":
            goal_command_text = _normalize_goal_command_text(str(runtime_command.args.get("text") or ""))
        if runtime_command is not None and runtime_command.command_id == "goal" and not goal_command_text:
            async for event in self._handle_goal_command(
                frontend_session_id,
                runtime_command,
                run_id=str(run_id or "").strip(),
            ):
                yield event
            return
        if runtime_command is not None and runtime_command.command_id == "goal":
            if goal_command_text.lower() == "clear":
                await self._goal_store.clear(frontend_session_id)
            else:
                await self._goal_store.set(frontend_session_id, goal_command_text)

        sdk = self._load_sdk()
        bridge = _ToolEventBridge(self._tool_registry, self._task_registry, run_id=run_id)
        active_goal = await self._goal_store.get(frontend_session_id)
        goal_tasks: list[Mapping[str, Any]] = []
        existing = await self._session_store.get(frontend_session_id) if frontend_session_id else None
        existing_client = await self._client_pool.get(frontend_session_id) if frontend_session_id else None
        is_new_session = existing is None and existing_client is None
        claude_session_id = (
            existing_client.claude_session_id
            if existing_client and existing_client.claude_session_id
            else existing.claude_session_id
            if existing and existing.claude_session_id
            else _build_claude_session_id(frontend_session_id)
        )
        effective_model = resolve_effective_model(payload.get("model"), self._settings.default_model)
        if active_goal is not None and runtime_command is None:
            active_goal = await self._goal_store.mark_run_started(frontend_session_id, str(run_id or "").strip()) or active_goal
        if runtime_command is not None and runtime_command.command_id == "goal":
            sdk_prompt = build_runtime_command_prompt_input(_goal_sdk_command_text(runtime_command, goal_command_text))
        elif runtime_command is not None:
            sdk_prompt = build_runtime_command_prompt_input(runtime_command.prompt_text)
        else:
            sdk_prompt = await build_sdk_prompt_input(
                payload,
                include_history=is_new_session,
                request_headers=request_headers,
                timeout_sec=self._provider.request_timeout_sec,
                attachment_text_char_limit=self._settings.attachment_text_char_limit,
            )
        system_prompt = _build_sdk_system_prompt(self._settings, payload)
        provider_config = self._resolve_provider_config(payload, request_headers)
        proxy_context = await self._proxy_contexts.create(
            upstream_base_url=provider_config.base_url,
            anthropic_version=provider_config.anthropic_version,
            x_user_id=self._resolve_x_user_id(request_headers, current_user),
            api_token=provider_config.api_token,
            model=effective_model,
            request_headers={str(key): str(value) for key, value in request_headers.items()},
            ttl_sec=self._provider.proxy_context_ttl_sec,
        )
        proxy_env = self._build_proxy_env(
            proxy_base_url,
            proxy_context.proxy_token,
            request_headers=request_headers,
            current_user=current_user,
            fallback_tdl_api_key=fallback_tdl_api_key,
        )
        mcp_servers = build_mcp_servers(
            self._mcp_settings,
            request_headers=request_headers,
            fallback_tdl_api_key=fallback_tdl_api_key,
        )
        permission_options = _runtime_permission_options(
            payload,
            self._settings,
            fallback_runtime_key=_permission_runtime_key(request_headers),
        )
        workspace = _resolve_workspace_execution(payload, self._settings)
        skill_platform_catalog = _runtime_skill_platform_catalog(payload)
        effective_skill_mount_root = (
            None if skill_platform_catalog == "exclude" else skill_mount_root
        )
        effective_workspace_add_dirs = _sdk_add_dirs(
            self._settings,
            effective_skill_mount_root,
            workspace.add_dirs,
        )
        workspace_runtime = self._inspect_workspace_runtime(
            workspace,
            permission_options=permission_options,
            mcp_servers=mcp_servers,
            skill_mount_root=skill_mount_root,
            skill_platform_catalog=skill_platform_catalog,
        )
        signature = _session_signature(
            claude_session_id=claude_session_id,
            model=effective_model,
            env=proxy_env,
            system_prompt=system_prompt,
            mcp_servers=mcp_servers,
            resumed=bool(existing or existing_client),
            permission_mode=permission_options.permission_mode,
            allowed_tools=permission_options.allowed_tools,
            disallowed_tools=permission_options.disallowed_tools,
            permission_profile=permission_options.profile,
            permission_runtime_key=permission_options.runtime_key,
            permission_revision=permission_options.revision,
            permission_full_bypass=permission_options.full_bypass,
            cwd=str(workspace.cwd),
            add_dirs=effective_workspace_add_dirs,
            workspace_fingerprint=str(workspace_runtime.get("fingerprint") or ""),
            skill_platform_catalog=skill_platform_catalog,
            provider_base_url=provider_config.base_url,
            provider_api_token_digest=_secret_digest(provider_config.api_token),
        )
        emitted_text = False
        compact_boundary_seen = False
        auto_compact_request_id = _auto_compact_request_id(frontend_session_id, run_id)
        emit_runtime_command_card = runtime_command is not None
        recent_items: list[str] = []
        turn_affected_files: list[str] = []
        artifacts_finalized = False
        goal_stop_hook_status = ""
        logger.info(
            "[claude-sdk] begin frontend_session_id=%s claude_session_id=%s model=%s provider=%s resumed=%s prompt_chars=%s workspace_add_dirs=%s",
            frontend_session_id or "-",
            claude_session_id or "-",
            effective_model or "-",
            provider_config.source,
            bool(existing_client or (existing and existing.claude_session_id)),
            sdk_prompt.prompt_chars,
            len(workspace.add_dirs),
        )
        self._session_contexts[frontend_session_id] = _SessionExecutionContext(
            run_id=str(run_id or "").strip(),
            claude_session_id=claude_session_id,
        )
        if active_goal is not None and runtime_command is not None and runtime_command.command_id == "goal":
            await self._goal_store.mark_run_started(frontend_session_id, str(run_id or "").strip())
        record = await self._client_pool.get_or_create(
            frontend_session_id,
            claude_session_id=claude_session_id,
            model=effective_model,
            resumed=bool(existing or existing_client),
            signature=signature,
            workspace_cwd=str(workspace.cwd),
            workspace_add_dirs=[str(path) for path in workspace.add_dirs],
            workspace_source=workspace.source,
            workspace_configured=workspace.configured,
            workspace_runtime=workspace_runtime,
            factory=lambda: self._create_connected_client(
                sdk=sdk,
                model=effective_model,
                system_prompt=system_prompt,
                claude_session_id=claude_session_id,
                resume_session_id=existing.claude_session_id if existing else "",
                proxy_env=proxy_env,
                skill_mount_root=effective_skill_mount_root,
                workspace_cwd=workspace.cwd,
                workspace_add_dirs=workspace.add_dirs,
                mcp_servers=mcp_servers,
                permission_options=permission_options,
                can_use_tool=self._build_can_use_tool_callback(
                    sdk_module=sdk["module"],
                    frontend_session_id=frontend_session_id,
                ),
                hooks=build_sdk_hooks(
                    sdk["module"],
                    frontend_session_id=frontend_session_id,
                    registry=self._hook_registry,
                    execution_resolver=self._resolve_hook_execution,
                ),
            ),
        )
        record.workspace_runtime = merge_observed_runtime(
            workspace_runtime,
            server_info=record.server_info,
        )
        session_mapping_committed = not is_new_session

        async def persist_session_mapping() -> None:
            nonlocal session_mapping_committed
            if not frontend_session_id or not claude_session_id:
                return
            await self._session_store.put(
                frontend_session_id,
                claude_session_id,
                model=effective_model,
                workspace_cwd=str(workspace.cwd),
                workspace_add_dirs=[str(path) for path in workspace.add_dirs],
                workspace_source=workspace.source,
                workspace_configured=workspace.configured,
            )
            session_mapping_committed = True

        if not is_new_session:
            await persist_session_mapping()
        skill_usage_headers = _skill_usage_request_headers(
            request_headers,
            frontend_session_id=frontend_session_id,
            run_id=str(run_id or "").strip(),
        )
        skill_usage_auditor = SkillUsageAuditor(
            settings=self._skill_usage_audit,
            skill_mount_root=skill_mount_root,
            base_context=_skill_usage_audit_context(
                payload=payload,
                request_headers=request_headers,
                current_user=current_user,
                proxy_base_url=proxy_base_url,
                frontend_session_id=frontend_session_id,
                run_id=str(run_id or "").strip(),
            ),
            request_headers=skill_usage_headers,
        )
        if runtime_command is not None and runtime_command.kind == "skill":
            skill_usage_auditor.record_workflow_skill_command(runtime_command.command_id)

        async def finalize_artifacts(status: str) -> RuntimeStreamEvent | None:
            nonlocal artifacts_finalized
            if artifacts_finalized or not artifacts_enabled or self._artifact_service is None:
                return None
            artifacts_finalized = True
            try:
                record = await self._artifact_service.save_run_from_paths(
                    session_id=frontend_session_id,
                    run_id=str(run_id or "").strip(),
                    workspace_cwd=workspace.cwd,
                    workspace_add_dirs=workspace.add_dirs,
                    affected_files=turn_affected_files,
                    started_at=artifact_started_at,
                    status=status,
                )
            except Exception as exc:
                logger.warning("[artifacts] finalize failed run=%s err=%s", run_id or "-", exc)
                return None
            summary = record.get("summary") if isinstance(record, Mapping) else {}
            logger.info(
                "[artifacts] finalized run=%s status=%s count=%s source=sdk_affected_files",
                run_id or "-",
                status,
                summary.get("artifactCount") if isinstance(summary, Mapping) else "-",
            )
            return RuntimeStreamEvent(kind="artifacts", payload=record)

        async with record.lock:
            try:
                if emit_runtime_command_card:
                    yield RuntimeStreamEvent(
                        kind="command",
                        payload=runtime_command.shell_payload(
                            status="running",
                            phase="start",
                            message=f"正在执行 {runtime_command.command}",
                        ),
                    )
                async def process_runtime_item(item: Any) -> AsyncIterator[RuntimeStreamEvent]:
                    nonlocal claude_session_id, emitted_text, compact_boundary_seen, turn_affected_files, goal_stop_hook_status
                    item_summary = _summarize_stream_item(item)
                    recent_items.append(item_summary)
                    if len(recent_items) > 8:
                        recent_items.pop(0)
                    logger.info("[claude-sdk] item %s", item_summary)
                    extracted_claude_session_id = extract_claude_session_id(item)
                    if extracted_claude_session_id:
                        claude_session_id = extracted_claude_session_id
                    if extracted_claude_session_id and frontend_session_id:
                        if not is_new_session:
                            await persist_session_mapping()
                        previous_context = self._session_contexts.get(frontend_session_id)
                        self._session_contexts[frontend_session_id] = _SessionExecutionContext(
                            run_id=str(run_id or "").strip(),
                            claude_session_id=extracted_claude_session_id,
                            last_checkpoint_id=str(getattr(previous_context, "last_checkpoint_id", "") or ""),
                        )
                    recorded_goal_status = await self._record_hook_event(frontend_session_id, item)
                    if recorded_goal_status:
                        goal_stop_hook_status = recorded_goal_status
                    await self._record_checkpoint(frontend_session_id, claude_session_id, item, prompt=_first_text_block(sdk_prompt.content_blocks))
                    turn_affected_files = _merge_file_lists(turn_affected_files, item_tool_file_paths(item))
                    await self._record_checkpoint_files(frontend_session_id, turn_affected_files)
                    stream_start = stream_event_tool_start(item)
                    if stream_start is not None:
                        skill_usage_auditor.observe_tool_start(stream_start, cwd=workspace.cwd)
                    for tool_start in content_tool_starts(item):
                        skill_usage_auditor.observe_tool_start(tool_start, cwd=workspace.cwd)
                    async for bridge_event in bridge.handle_item(item):
                        event = bridge_event
                        if event.kind == "task" and isinstance(event.payload, Mapping):
                            goal_tasks.append(dict(event.payload))
                        yield event
                    for tool_result in content_tool_results(item):
                        skill_usage_auditor.observe_tool_result(tool_result)
                    if is_compact_boundary(item) and not compact_boundary_seen:
                        compact_boundary_seen = True
                        if not emit_runtime_command_card:
                            yield RuntimeStreamEvent(
                                kind="command",
                                payload=_compact_runtime_command_payload(
                                    request_id=auto_compact_request_id,
                                    status="running",
                                    phase="compact_start",
                                    message="检测到 Claude Code 开始自动压缩上下文。",
                                    source="claude-code",
                                ),
                            )
                        payload = (
                            runtime_command.shell_payload(
                                status="boundary",
                                phase="compact_boundary",
                                message="已生成压缩边界，正在完成上下文压缩。",
                                result="compact_boundary",
                            )
                            if emit_runtime_command_card
                            else _compact_runtime_command_payload(
                                request_id=auto_compact_request_id,
                                status="boundary",
                                phase="compact_boundary",
                                message="检测到 Claude Code 正在自动压缩上下文。",
                                result="compact_boundary",
                                source="claude-code",
                            )
                        )
                        yield RuntimeStreamEvent(kind="command", payload=payload)
                    delta = event_to_text_delta(item)
                    if delta:
                        emitted_text = True
                        yield RuntimeStreamEvent(kind="text", text=delta)
                        return
                    if not emitted_text and is_assistant_message(item):
                        text = assistant_text(item)
                        if text:
                            emitted_text = True
                            logger.info("[claude-sdk] yielded assistant message text chars=%s", len(text))
                            yield RuntimeStreamEvent(kind="text", text=text)
                            return
                    if not emitted_text and is_final_result(item):
                        text = assistant_text(item) or final_result_error_text(item)
                        if text:
                            emitted_text = True
                            logger.info("[claude-sdk] yielded final assistant text chars=%s", len(text))
                            yield RuntimeStreamEvent(kind="text", text=text)

                await record.client.query(sdk_prompt.as_stream(), session_id=record.claude_session_id)
                async for item in record.client.receive_response():
                    async for event in process_runtime_item(item):
                        yield event
                if bridge.has_open_tasks():
                    async for event in bridge.waiting_open_tasks():
                        if event.kind == "task" and isinstance(event.payload, Mapping):
                            goal_tasks.append(dict(event.payload))
                        yield event
                    deadline = time.monotonic() + _SUBTASK_WAIT_TIMEOUT_SECONDS
                    followup_items = _open_task_message_stream(record.client)
                    pending_item: asyncio.Task[Any] | None = None
                    next_heartbeat_at = time.monotonic() + _SUBTASK_WAIT_HEARTBEAT_SECONDS
                    try:
                        while bridge.has_open_tasks() and time.monotonic() < deadline:
                            remaining = max(0.0, deadline - time.monotonic())
                            if remaining <= 0:
                                break
                            if pending_item is None:
                                pending_item = asyncio.create_task(anext(followup_items))

                            wait_seconds = min(_SUBTASK_WAIT_POLL_SECONDS, remaining)
                            now = time.monotonic()
                            if now < next_heartbeat_at:
                                wait_seconds = min(wait_seconds, next_heartbeat_at - now)
                            done, _ = await asyncio.wait({pending_item}, timeout=max(0.0, wait_seconds))
                            if not done:
                                if time.monotonic() >= next_heartbeat_at:
                                    async for event in bridge.waiting_open_tasks(log_text=_TASK_STREAM_STILL_WAITING_LOG):
                                        if event.kind == "task" and isinstance(event.payload, Mapping):
                                            goal_tasks.append(dict(event.payload))
                                        yield event
                                    next_heartbeat_at = time.monotonic() + _SUBTASK_WAIT_HEARTBEAT_SECONDS
                                continue

                            item_task = pending_item
                            pending_item = None
                            try:
                                item = item_task.result()
                            except StopAsyncIteration:
                                followup_items = _open_task_message_stream(record.client)
                                continue
                            async for event in process_runtime_item(item):
                                yield event
                    finally:
                        if pending_item is not None and not pending_item.done():
                            pending_item.cancel()
                            try:
                                await pending_item
                            except asyncio.CancelledError:
                                pass
                    if bridge.has_open_tasks():
                        if active_goal is not None:
                            paused_goal = await self._goal_store.pause(
                                frontend_session_id,
                                reason="subtask_wait_timeout",
                            )
                            active_goal = paused_goal or active_goal
                            yield RuntimeStreamEvent(
                                kind="command",
                                payload=_goal_runtime_command_payload(
                                    status="paused",
                                    phase="goal_paused",
                                    message="后台子任务等待超时，目标已暂停等待继续。",
                                    result="subtask_wait_timeout",
                                    run_id=str(run_id or "").strip(),
                                ),
                            )
                        if emitted_text:
                            yield RuntimeStreamEvent(kind="text", text=f"\n\n{_SUBTASK_TIMEOUT_TEXT}")
                        else:
                            yield RuntimeStreamEvent(kind="text", text=_SUBTASK_TIMEOUT_TEXT)
                        emitted_text = True
                        async for event in bridge.finish_open_tasks():
                            if event.kind == "task" and isinstance(event.payload, Mapping):
                                goal_tasks.append(dict(event.payload))
                            yield event
                    else:
                        already_emitted_text = emitted_text
                        emitted_text = False
                        await record.client.query(
                            build_runtime_command_prompt_input(_SUBTASK_SUMMARY_PROMPT).as_stream(),
                            session_id=record.claude_session_id,
                        )
                        async for item in record.client.receive_response():
                            async for event in process_runtime_item(item):
                                yield event
                        emitted_text = already_emitted_text or emitted_text
                if is_new_session:
                    await persist_session_mapping()
                if emit_runtime_command_card:
                    completed_message = "命令执行完成，本次命令没有文本输出。" if not emitted_text else "命令执行完成。"
                    completed_phase = "end"
                    completed_result = ""
                    if runtime_command.command_id == "compact" and compact_boundary_seen:
                        completed_message = "上下文压缩完成，后续对话将使用压缩后的上下文。"
                        completed_phase = "compact_complete"
                        completed_result = "compact_boundary"
                    elif runtime_command.command_id == "context":
                        completed_message = "上下文状态已更新。" if emitted_text else "未获取到上下文状态输出。"
                        completed_phase = "context_report"
                        completed_result = "context_report" if emitted_text else ""
                    elif runtime_command.command_id == "usage":
                        completed_message = "用量信息已更新。" if emitted_text else "未获取到用量信息输出。"
                        completed_phase = "usage_report"
                        completed_result = "usage_report" if emitted_text else ""
                    elif runtime_command.command_id == "goal":
                        goal_action = str(runtime_command.args.get("text") or "").strip().lower()
                        completed_phase = "goal_set"
                        completed_result = "goal_set"
                        completed_message = "目标已设置，Claude 将围绕该目标继续推进。"
                        if not goal_action:
                            completed_phase = "goal_status"
                            completed_result = "goal_status"
                            completed_message = "目标状态已读取。" if emitted_text else "未获取到目标状态输出。"
                        elif goal_action == "clear":
                            completed_phase = "goal_cleared"
                            completed_result = "goal_cleared"
                            completed_message = "目标已清除。"
                        elif goal_stop_hook_status == "completed":
                            completed_phase = "goal_completed"
                            completed_result = "goal_completed"
                            completed_message = "目标条件已由 Claude Code Stop hook 确认完成。"
                    yield RuntimeStreamEvent(
                        kind="command",
                        payload=runtime_command.shell_payload(
                            status="completed",
                            phase=completed_phase,
                            message=completed_message,
                            result=completed_result,
                        ),
                    )
                elif compact_boundary_seen:
                    yield RuntimeStreamEvent(
                        kind="command",
                        payload=_compact_runtime_command_payload(
                            request_id=auto_compact_request_id,
                            status="completed",
                            phase="compact_complete",
                            message="上下文压缩完成，后续对话将使用压缩后的上下文。",
                            result="compact_boundary",
                            source="claude-code",
                        ),
                    )
                if active_goal is not None and active_goal.status != "paused" and not goal_stop_hook_status:
                    await self._record_goal_run_observation(
                        frontend_session_id,
                        run_id=str(run_id or "").strip(),
                        emitted_text=emitted_text,
                        task_events=goal_tasks,
                    )
                artifact_event = await finalize_artifacts("completed")
                if artifact_event is not None:
                    yield artifact_event
            except Exception as exc:
                if active_goal is not None and active_goal.status != "paused" and not goal_stop_hook_status:
                    await self._goal_store.record_run_result(
                        frontend_session_id,
                        run_id=str(run_id or "").strip(),
                        status="running",
                        summary=str(exc),
                        tasks=goal_tasks,
                    )
                if _is_ignorable_terminal_exception(exc):
                    if emitted_text:
                        if is_new_session:
                            await persist_session_mapping()
                        logger.warning("[claude-sdk] ignored terminal exception after successful output: %s", exc)
                        artifact_event = await finalize_artifacts("completed")
                        if artifact_event is not None:
                            yield artifact_event
                        return
                    recovered_text = self._recover_text_from_session(
                        sdk["module"],
                        session_id=claude_session_id,
                    )
                    if recovered_text:
                        if is_new_session:
                            await persist_session_mapping()
                        logger.warning(
                            "[claude-sdk] recovered assistant text from session transcript after terminal exception: %s",
                            exc,
                        )
                        yield RuntimeStreamEvent(kind="text", text=recovered_text)
                        artifact_event = await finalize_artifacts("completed")
                        if artifact_event is not None:
                            yield artifact_event
                        return
                if is_new_session and not session_mapping_committed:
                    removed_record = await self._client_pool.remove(frontend_session_id)
                    if removed_record is not None:
                        disconnect = getattr(removed_record.client, "disconnect", None)
                        if callable(disconnect):
                            try:
                                await disconnect()
                            except Exception as disconnect_exc:
                                logger.warning(
                                    "[claude-sdk] failed to clean up uncommitted client frontend_session_id=%s err=%s",
                                    frontend_session_id or "-",
                                    disconnect_exc,
                                )
                    self._session_contexts.pop(frontend_session_id, None)
                logger.error(
                    "[claude-sdk] stream failed frontend_session_id=%s claude_session_id=%s recent_items=%s err=%s",
                    frontend_session_id or "-",
                    claude_session_id or "-",
                    recent_items,
                    exc,
                )
                if emit_runtime_command_card:
                    yield RuntimeStreamEvent(
                        kind="command",
                        payload=runtime_command.shell_payload(
                            status="failed",
                            phase="error",
                            error=str(exc),
                        ),
                    )
                artifact_event = await finalize_artifacts("failed")
                if artifact_event is not None:
                    yield artifact_event
                raise

    async def stream_text(
        self,
        payload: Mapping[str, Any],
        *,
        request_headers: Mapping[str, str],
        current_user: Any | None,
        fallback_tdl_api_key: str = "",
        proxy_base_url: str,
        skill_mount_root,
        run_id: str | None = None,
    ) -> AsyncIterator[str]:
        async for event in self.stream_events(
            payload,
            request_headers=request_headers,
            current_user=current_user,
            fallback_tdl_api_key=fallback_tdl_api_key,
            proxy_base_url=proxy_base_url,
            skill_mount_root=skill_mount_root,
            run_id=run_id,
        ):
            if event.kind == "text" and event.text:
                yield event.text

    def _load_sdk(self) -> dict[str, Any]:
        try:
            module = importlib.import_module("claude_agent_sdk")
        except ModuleNotFoundError as exc:
            raise RuntimeError("Missing dependency: install `claude-agent-sdk` first.") from exc
        client_cls = getattr(module, "ClaudeSDKClient", None)
        options_cls = getattr(module, "ClaudeAgentOptions", None)
        if client_cls is None or options_cls is None:
            raise RuntimeError("Unsupported claude-agent-sdk version: missing ClaudeSDKClient/ClaudeAgentOptions.")
        return {"module": module, "ClaudeSDKClient": client_cls, "ClaudeAgentOptions": options_cls}

    def _resolve_x_user_id(self, request_headers: Mapping[str, str], current_user: Any | None) -> str:
        candidates = [
            getattr(current_user, "emp_id", None),
            request_headers.get("x-user-id"),
            request_headers.get("uac-user-id"),
            request_headers.get("x-uac-user-id"),
        ]
        for value in candidates:
            text = str(value or "").strip()
            if text:
                return text
        return "anonymous"

    def _resolve_provider_config(self, payload: Mapping[str, Any], request_headers: Mapping[str, str]) -> _RuntimeProviderConfig:
        llm_config = _runtime_llm_config(payload)
        runtime_base_url = _normalize_provider_base_url(
            _runtime_llm_value(llm_config, "api_base", "apiBase", "base_url", "baseUrl")
        )
        runtime_api_key = _runtime_llm_value(llm_config, "api_key", "apiKey", "anthropic_auth_token", "anthropicAuthToken")
        runtime_version = _runtime_llm_value(llm_config, "anthropic_version", "anthropicVersion")
        uses_runtime_provider = bool(runtime_base_url or runtime_api_key or runtime_version)
        source = "runtime_config" if uses_runtime_provider else "service"
        api_token = (
            runtime_api_key
            if uses_runtime_provider
            else self._resolve_api_token(request_headers)
        ) or self._provider.api_key
        return _RuntimeProviderConfig(
            base_url=runtime_base_url or self._provider.base_url,
            anthropic_version=runtime_version or self._provider.anthropic_version,
            api_token=api_token,
            source=source,
        )

    def _resolve_api_token(self, request_headers: Mapping[str, str]) -> str:
        for key in ("api-key", "x-api-key"):
            value = request_headers.get(key)
            text = str(value or "").strip()
            if text:
                return text
        for key in ("uac-user-token", "x-uac-user-token"):
            value = request_headers.get(key)
            text = str(value or "").strip()
            if text:
                return text
        return self._provider.api_key

    def _resolve_hook_execution(self, frontend_session_id: str) -> tuple[str, str]:
        execution = self._session_contexts.get(frontend_session_id)
        if execution is None:
            return "", ""
        return execution.run_id, execution.claude_session_id

    def _build_proxy_env(
        self,
        proxy_base_url: str,
        proxy_token: str,
        *,
        request_headers: Mapping[str, str],
        current_user: Any | None,
        fallback_tdl_api_key: str,
    ) -> dict[str, str]:
        no_proxy_value = "127.0.0.1,localhost,10.2.67.41"
        env = {
            "ANTHROPIC_BASE_URL": proxy_base_url.rstrip("/"),
            "ANTHROPIC_AUTH_TOKEN": proxy_token,
            "ANTHROPIC_API_KEY": proxy_token,
            "HTTP_PROXY": "",
            "HTTPS_PROXY": "",
            "ALL_PROXY": "",
            "http_proxy": "",
            "https_proxy": "",
            "all_proxy": "",
            "NO_PROXY": no_proxy_value,
            "no_proxy": no_proxy_value,
        }
        if isinstance(self._settings.env, Mapping):
            for key, value in self._settings.env.items():
                name = str(key or "").strip()
                if name:
                    env.setdefault(name, str(value))
        env.update(
            _request_auth_env(
                request_headers,
                current_user,
                api_token=self._resolve_api_token(request_headers),
                fallback_tdl_api_key=fallback_tdl_api_key,
            )
        )
        return env

    def _build_can_use_tool_callback(
        self,
        *,
        sdk_module: Any,
        frontend_session_id: str,
    ):
        allow_cls = getattr(sdk_module, "PermissionResultAllow")
        deny_cls = getattr(sdk_module, "PermissionResultDeny")

        async def _callback(tool_name: str, tool_input: dict[str, Any], permission_context: Any):
            execution = self._session_contexts.get(frontend_session_id) or _SessionExecutionContext(run_id="", claude_session_id="")
            tool_name_text = str(tool_name or "").strip()
            if _is_ask_user_question_tool(tool_name):
                question = await self._question_registry.create_question(
                    session_id=frontend_session_id,
                    run_id=execution.run_id,
                    claude_session_id=execution.claude_session_id,
                    prompt=_ask_user_question_prompt(tool_input),
                    title=_ask_user_question_title(tool_input),
                    description=_ask_user_question_description(tool_input),
                    metadata={
                        "toolName": str(tool_name or "").strip(),
                        "toolInput": dict(tool_input or {}),
                        "toolUseId": str(getattr(permission_context, "tool_use_id", "") or ""),
                        "agentId": str(getattr(permission_context, "agent_id", "") or ""),
                    },
                )
                answer = await question.wait_for_answer()
                return allow_cls(updated_input=_ask_user_question_updated_input(tool_input, answer.answer))

            request = await self._approval_registry.create_request(
                session_id=frontend_session_id,
                run_id=execution.run_id,
                claude_session_id=execution.claude_session_id,
                tool_name=tool_name_text or "tool",
                tool_input=dict(tool_input or {}),
                tool_use_id=str(getattr(permission_context, "tool_use_id", "") or ""),
                agent_id=str(getattr(permission_context, "agent_id", "") or ""),
                blocked_path=str(getattr(permission_context, "blocked_path", "") or ""),
                decision_reason=str(getattr(permission_context, "decision_reason", "") or ""),
                title=str(getattr(permission_context, "title", "") or ""),
                display_name=str(getattr(permission_context, "display_name", "") or ""),
                description=str(getattr(permission_context, "description", "") or ""),
            )
            decision = await request.wait_for_decision()
            if decision.decision == "allow":
                return allow_cls()
            return deny_cls(message=decision.reason, interrupt=decision.interrupt)

        return _callback

    def _build_cli_stderr_callback(
        self,
        proxy_env: Mapping[str, str],
        stderr_lines: deque[str],
    ):
        sensitive_values = [
            str(value)
            for key, value in proxy_env.items()
            if any(marker in str(key).upper() for marker in _SENSITIVE_ENV_MARKERS)
            and len(str(value)) >= 4
        ]

        def _callback(line: str) -> None:
            sanitized = _sanitize_cli_stderr_line(line, sensitive_values)
            if not sanitized:
                return
            stderr_lines.append(sanitized)
            logger.warning("[claude-sdk][cli-stderr] %s", sanitized)

        return _callback

    def _build_options(self, options_cls: type[Any], **candidate_values: Any) -> Any:
        supported = set()
        accepts_var_kwargs = False
        try:
            parameters = inspect.signature(options_cls).parameters
            supported.update(parameters.keys())
            accepts_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())
        except (TypeError, ValueError):
            pass
        annotations = getattr(options_cls, "__annotations__", {})
        if isinstance(annotations, dict):
            supported.update(annotations.keys())
        kwargs = {}
        for key, value in candidate_values.items():
            if value in (None, ""):
                continue
            if supported and not accepts_var_kwargs and key not in supported:
                continue
            kwargs[key] = value
        return options_cls(**kwargs)

    @contextmanager
    def _sdk_env(self):
        old_config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
        os.environ["CLAUDE_CONFIG_DIR"] = str(self._settings.config_dir)
        try:
            yield
        finally:
            if old_config_dir is None:
                os.environ.pop("CLAUDE_CONFIG_DIR", None)
            else:
                os.environ["CLAUDE_CONFIG_DIR"] = old_config_dir

    async def _create_connected_client(
        self,
        *,
        sdk: Mapping[str, Any],
        model: str,
        system_prompt: Any,
        claude_session_id: str,
        resume_session_id: str,
        force_resume: bool = False,
        proxy_env: Mapping[str, str],
        skill_mount_root,
        workspace_cwd: Path | None = None,
        workspace_add_dirs: Sequence[Path] | None = None,
        mcp_servers: Mapping[str, Any],
        permission_mode: str | None = None,
        permission_options: PermissionOptions | None = None,
        can_use_tool: Any,
        hooks: Mapping[str, Any],
    ) -> tuple[Any, Mapping[str, Any] | None]:
        resume_option = resume_session_id if force_resume else ("" if self._settings.continue_conversation else resume_session_id)
        session_id_option = "" if force_resume or self._settings.continue_conversation or resume_session_id else claude_session_id
        continue_conversation_option = False if force_resume else self._settings.continue_conversation
        effective_permission = permission_options or permission_options_from_runtime_config(
            self._settings,
            None,
        )
        stderr_lines: deque[str] = deque(maxlen=20)
        options = self._build_options(
            sdk["ClaudeAgentOptions"],
            tools=self._settings.tools,
            allowed_tools=effective_permission.allowed_tools,
            model=model,
            fallback_model=self._settings.fallback_model,
            cwd=str(workspace_cwd or self._settings.workdir),
            permission_mode=permission_mode or effective_permission.permission_mode,
            system_prompt=system_prompt,
            strict_mcp_config=self._settings.strict_mcp_config,
            continue_conversation=continue_conversation_option,
            resume=resume_option,
            session_id=session_id_option,
            max_turns=self._settings.max_turns,
            max_budget_usd=self._settings.max_budget_usd,
            disallowed_tools=effective_permission.disallowed_tools,
            betas=self._settings.betas,
            permission_prompt_tool_name=self._settings.permission_prompt_tool_name,
            settings=self._settings.settings,
            include_partial_messages=self._settings.include_partial_messages,
            cli_path=self._settings.cli_path,
            env=_sdk_process_env(self._settings, proxy_env),
            add_dirs=_sdk_add_dirs(self._settings, skill_mount_root, workspace_add_dirs),
            max_buffer_size=self._settings.max_buffer_size,
            mcp_servers=dict(mcp_servers),
            can_use_tool=can_use_tool,
            setting_sources=self._settings.setting_sources,
            skills=self._settings.skills_filter,
            include_hook_events=self._settings.include_hook_events,
            hooks=dict(hooks),
            user=self._settings.user,
            fork_session=self._settings.fork_session,
            agents=_sdk_agents(sdk.get("module"), self._settings.agents),
            sandbox=self._settings.sandbox,
            plugins=self._settings.plugins,
            max_thinking_tokens=self._settings.max_thinking_tokens,
            thinking=self._settings.thinking,
            effort=self._settings.effort,
            output_format=self._settings.output_format,
            enable_file_checkpointing=self._settings.enable_file_checkpointing,
            session_store_flush=self._settings.session_store_flush,
            load_timeout_ms=self._settings.load_timeout_ms,
            task_budget=self._settings.task_budget,
            stderr=self._build_cli_stderr_callback(proxy_env, stderr_lines),
            extra_args=_permission_extra_args(
                _sdk_extra_args(self._settings),
                full_bypass=effective_permission.full_bypass,
            ),
        )
        client = sdk["ClaudeSDKClient"](options=options)
        try:
            with self._sdk_env():
                await client.connect()
                server_info = await client.get_server_info()
        except Exception as exc:
            exit_code = getattr(exc, "exit_code", None)
            stderr_summary = " | ".join(stderr_lines)
            if stderr_summary or exit_code == 127:
                cli_path = str(self._settings.cli_path or "SDK bundled/auto-detected CLI")
                detail = f": {stderr_summary}" if stderr_summary else ""
                hint = ""
                if exit_code == 127:
                    hint = (
                        " Verify the CLI interpreter and PATH entries required by "
                        "Claude Code MCP/hook commands."
                    )
                raise RuntimeError(
                    f"Claude Code CLI failed to initialize (cli={cli_path}, exit_code={exit_code})"
                    f"{detail}.{hint}"
                ) from exc
            raise
        return client, server_info

    async def _record_checkpoint(
        self,
        frontend_session_id: str,
        claude_session_id: str,
        item: Any,
        *,
        prompt: str,
    ) -> None:
        if not self._settings.enable_file_checkpointing:
            return
        if type(item).__name__ != "UserMessage":
            return
        checkpoint_id = str(getattr(item, "uuid", "") or "").strip()
        if not checkpoint_id:
            return
        if not str(claude_session_id or "").strip():
            execution = self._session_contexts.get(frontend_session_id)
            claude_session_id = str(getattr(execution, "claude_session_id", "") or "").strip()
        await self._checkpoint_store.put(
            frontend_session_id,
            claude_session_id,
            checkpoint_id,
            prompt_excerpt=_truncate(prompt, 160),
        )
        context = self._session_contexts.get(frontend_session_id)
        if context is not None:
            context.last_checkpoint_id = checkpoint_id

    async def _record_checkpoint_files(self, frontend_session_id: str, files: list[str]) -> None:
        if not files:
            return
        context = self._session_contexts.get(frontend_session_id)
        checkpoint_id = str(getattr(context, "last_checkpoint_id", "") or "").strip()
        if not checkpoint_id:
            return
        await self._checkpoint_store.update_metadata(
            frontend_session_id,
            checkpoint_id,
            affected_files=files,
        )

    def _recover_text_from_session(self, sdk_module: Any, *, session_id: str) -> str:
        getter = getattr(sdk_module, "get_session_messages", None)
        if not callable(getter) or not session_id:
            return ""
        try:
            messages = getter(session_id=session_id, directory=str(self._settings.config_dir))
        except Exception as exc:
            logger.warning("[claude-sdk] failed to recover session transcript session_id=%s err=%s", session_id, exc)
            return ""
        if not isinstance(messages, list):
            return ""
        for item in reversed(messages):
            if str(getattr(item, "type", "") or "").strip().lower() != "assistant":
                continue
            message = getattr(item, "message", None)
            text = assistant_text(_SessionMessageAdapter(message))
            if text:
                return text
        return ""

    async def runtime_snapshot(self) -> Mapping[str, Any]:
        payload = dict(await self._client_pool.runtime_snapshot(include_sessions=True))
        goal_runtime = dict(await self._goal_store.runtime_snapshot(include_sessions=True))
        payload["goalRuntime"] = _normalize_goal_runtime_snapshot(goal_runtime, payload.get("sessions"))
        return payload

    async def inspect_workspace(
        self,
        payload: Mapping[str, Any],
        *,
        request_headers: Mapping[str, str] | None = None,
        fallback_tdl_api_key: str = "",
        skill_mount_root: Path | None = None,
    ) -> Mapping[str, Any]:
        workspace = _resolve_workspace_execution(payload, self._settings)
        permission_options = _runtime_permission_options(
            payload,
            self._settings,
            fallback_runtime_key=_permission_runtime_key(request_headers),
        )
        mcp_servers = build_mcp_servers(
            self._mcp_settings,
            request_headers=request_headers or {},
            fallback_tdl_api_key=fallback_tdl_api_key,
        )
        skill_platform_catalog = _runtime_skill_platform_catalog(payload)
        return self._inspect_workspace_runtime(
            workspace,
            permission_options=permission_options,
            mcp_servers=mcp_servers,
            skill_mount_root=skill_mount_root,
            skill_platform_catalog=skill_platform_catalog,
        )

    def _inspect_workspace_runtime(
        self,
        workspace: _WorkspaceExecution,
        *,
        permission_options: PermissionOptions,
        mcp_servers: Mapping[str, Any],
        skill_mount_root: Path | None,
        skill_platform_catalog: str = "include",
    ) -> dict[str, Any]:
        env = self._settings.env if isinstance(self._settings.env, Mapping) else {}
        additional_memory = str(env.get("CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD") or "").strip().lower()
        return inspect_workspace_runtime(
            cwd=workspace.cwd,
            add_dirs=workspace.add_dirs,
            source=workspace.source,
            configured=workspace.configured,
            setting_sources=effective_workspace_setting_sources(self._settings),
            strict_mcp_config=self._settings.strict_mcp_config,
            permission_profile=permission_options.profile,
            permission_mode=permission_options.permission_mode,
            allowed_tools=permission_options.allowed_tools,
            disallowed_tools=permission_options.disallowed_tools,
            permission_full_bypass=permission_options.full_bypass,
            permission_runtime_key=permission_options.runtime_key,
            permission_revision=permission_options.revision,
            agent_config_dir=self._settings.config_dir,
            skill_mount_root=skill_mount_root,
            skill_platform_catalog=skill_platform_catalog,
            workflow_mount_root=self._workflow_mount_root,
            agent_mcp_names=list(mcp_servers),
            additional_directories_claude_md=additional_memory in {"1", "true", "yes", "on"},
        )

    async def _handle_goal_command(
        self,
        frontend_session_id: str,
        runtime_command: Any,
        *,
        run_id: str,
    ) -> AsyncIterator[RuntimeStreamEvent]:
        yield RuntimeStreamEvent(
            kind="command",
            payload=runtime_command.shell_payload(
                status="running",
                phase="start",
                message="正在处理本地目标模式。",
            ),
        )
        goal_text = _normalize_goal_command_text(str(runtime_command.args.get("text") or ""))
        action = goal_text.lower()
        if not goal_text:
            goal = await self._goal_store.get(frontend_session_id)
            if goal is None:
                message = "当前没有活动目标。"
            else:
                message = f"当前目标：{goal.objective}\n状态：{_goal_status_label(goal.status)}"
            yield RuntimeStreamEvent(kind="text", text=message)
            yield RuntimeStreamEvent(
                kind="command",
                payload=runtime_command.shell_payload(
                    status="completed",
                    phase="goal_status",
                    message=message,
                    result="goal_status",
                ),
            )
            return
        if action == "clear":
            goal = await self._goal_store.clear(frontend_session_id)
            message = "目标已清除。" if goal is not None else "当前没有可清除的活动目标。"
            yield RuntimeStreamEvent(kind="text", text=message)
            yield RuntimeStreamEvent(
                kind="command",
                payload=runtime_command.shell_payload(
                    status="completed",
                    phase="goal_cleared",
                    message=message,
                    result="goal_cleared",
                ),
            )
            return
        goal = await self._goal_store.set(frontend_session_id, goal_text)
        message = f"目标已设置：{goal.objective}"
        yield RuntimeStreamEvent(kind="text", text=message)
        yield RuntimeStreamEvent(
            kind="command",
            payload=runtime_command.shell_payload(
                status="completed",
                phase="goal_set",
                message=message,
                result="goal_set",
            ),
        )

    async def _record_goal_run_observation(
        self,
        frontend_session_id: str,
        *,
        run_id: str,
        emitted_text: bool,
        task_events: list[Mapping[str, Any]],
    ) -> None:
        tasks = await self._goal_tasks_for_run(run_id, task_events)
        await self._goal_store.record_run_result(
            frontend_session_id,
            run_id=run_id,
            status="running",
            summary=_goal_run_observation_summary(tasks, emitted_text=emitted_text),
            tasks=tasks,
        )

    async def _goal_tasks_for_run(self, run_id: str, task_events: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        if self._task_registry is not None and run_id:
            tasks = await self._task_registry.list_run_tasks(run_id)
            if tasks:
                return tasks
        return task_events

    async def approvals_snapshot(self) -> Mapping[str, Any]:
        return await self._approval_registry.runtime_snapshot(include_sessions=True)

    async def questions_snapshot(self) -> Mapping[str, Any]:
        return await self._question_registry.runtime_snapshot(include_sessions=True)

    async def hooks_snapshot(self) -> Mapping[str, Any]:
        return await self._hook_registry.runtime_snapshot(include_sessions=True)

    async def get_session_state(self, frontend_session_id: str) -> Mapping[str, Any] | None:
        record = await self._client_pool.get(frontend_session_id)
        if record is None:
            return None
        if not record.lock.locked():
            getter = getattr(record.client, "get_mcp_status", None)
            if callable(getter):
                try:
                    mcp_status = await asyncio.wait_for(getter(), timeout=3.0)
                except Exception as exc:
                    logger.info("[workspace] failed to observe MCP status session_id=%s err=%s", frontend_session_id, exc)
                else:
                    record.workspace_runtime = merge_observed_runtime(
                        record.workspace_runtime,
                        server_info=record.server_info,
                        mcp_status=mcp_status if isinstance(mcp_status, Mapping) else None,
                    )
        payload = dict(record.snapshot())
        payload["checkpoints"] = [item.to_dict() for item in await self._checkpoint_store.list(frontend_session_id)]
        return payload

    async def interrupt_session(self, frontend_session_id: str) -> bool:
        record = await self._client_pool.get(frontend_session_id)
        goal = await self._goal_store.get(frontend_session_id)
        if record is None and goal is None:
            return False
        await self._approval_registry.cancel_session(frontend_session_id, reason="session interrupted")
        await self._question_registry.cancel_session(frontend_session_id, reason="session interrupted")
        if goal is not None:
            await self._goal_store.pause(frontend_session_id, reason="user_interrupt")
        self._session_contexts.pop(frontend_session_id, None)
        if record is None:
            return True
        await self._client_pool.remove(frontend_session_id)
        try:
            await record.client.interrupt()
        finally:
            disconnect = getattr(record.client, "disconnect", None)
            if callable(disconnect):
                await disconnect()
        return True

    async def _run_native_goal_clear(
        self,
        payload: Mapping[str, Any],
        *,
        request_headers: Mapping[str, str],
        current_user: Any | None,
        fallback_tdl_api_key: str,
        proxy_base_url: str,
        skill_mount_root,
        run_id: str | None,
    ) -> AsyncIterator[RuntimeStreamEvent]:
        clear_payload = dict(payload)
        metadata = dict(clear_payload.get("metadata") or {})
        metadata["runtimeCommand"] = {
            "source": "claude-code",
            "commandId": "goal",
            "command": "/goal",
            "args": {"text": "clear"},
            "displayName": "目标模式",
            "requestId": f"cmd-goal-clear-{uuid.uuid4().hex}",
        }
        clear_payload["metadata"] = metadata
        clear_payload["messages"] = [{"role": "user", "content": "/goal clear"}]
        async for event in self.stream_events(
            clear_payload,
            request_headers=request_headers,
            current_user=current_user,
            fallback_tdl_api_key=fallback_tdl_api_key,
            proxy_base_url=proxy_base_url,
            skill_mount_root=skill_mount_root,
            run_id=run_id,
        ):
            yield event

    async def rewind_session(self, frontend_session_id: str, checkpoint_id: str) -> bool:
        result = await self.rewind_checkpoint(frontend_session_id, checkpoint_id)
        return bool(result.get("ok"))

    async def rewind_checkpoint(
        self,
        frontend_session_id: str,
        checkpoint_id: str,
        *,
        request_headers: Mapping[str, str] | None = None,
        current_user: Any | None = None,
        fallback_tdl_api_key: str = "",
        proxy_base_url: str = "",
        skill_mount_root: Any | None = None,
        runtime_config: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        session_id = str(frontend_session_id or "").strip()
        selected_checkpoint_id = str(checkpoint_id or "").strip()
        if not self._settings.enable_file_checkpointing:
            return {
                "ok": False,
                "status": "disabled",
                "error": "file checkpointing is disabled",
                "sessionId": session_id,
                "checkpointId": selected_checkpoint_id,
            }
        checkpoint = await self._checkpoint_store.get(session_id, selected_checkpoint_id)
        if checkpoint is None:
            return {
                "ok": False,
                "status": "checkpoint_not_found",
                "error": "checkpoint not found",
                "sessionId": session_id,
                "checkpointId": selected_checkpoint_id,
            }
        session_state = await self._session_store.get(session_id)
        recovered_claude_session_id = (
            checkpoint.claude_session_id
            or (session_state.claude_session_id if session_state else "")
            or _build_claude_session_id(session_id)
        )
        if recovered_claude_session_id and not checkpoint.claude_session_id:
            await self._checkpoint_store.update_metadata(
                session_id,
                checkpoint.checkpoint_id,
                claude_session_id=recovered_claude_session_id,
            )
            checkpoint = await self._checkpoint_store.get(session_id, selected_checkpoint_id) or checkpoint
        record = await self._client_pool.get(session_id)
        if record is not None and record.claude_session_id != recovered_claude_session_id:
            await self._client_pool.disconnect(session_id)
            record = None
        if record is None:
            try:
                record = await self._recover_checkpoint_client(
                    frontend_session_id=session_id,
                    claude_session_id=recovered_claude_session_id,
                    model=session_state.model if session_state else "",
                    request_headers=request_headers,
                    current_user=current_user,
                    fallback_tdl_api_key=fallback_tdl_api_key,
                    proxy_base_url=proxy_base_url,
                    skill_mount_root=skill_mount_root,
                    runtime_config=runtime_config,
                    force_resume=True,
                )
            except Exception as exc:  # pragma: no cover - SDK recovery depends on local runtime process state
                logger.exception(
                    "checkpoint rewind session recovery failed",
                    extra={
                        "frontend_session_id": session_id,
                        "checkpoint_id": selected_checkpoint_id,
                        "claude_session_id": recovered_claude_session_id,
                    },
                )
                return {
                    "ok": False,
                    "status": "session_reconnect_failed",
                    "error": str(exc) or "failed to recover session",
                    "sessionId": session_id,
                    "checkpointId": selected_checkpoint_id,
                    "claudeSessionId": recovered_claude_session_id,
                }
        if record is None:
            return {
                "ok": False,
                "status": "session_not_found",
                "error": "session not found",
                "sessionId": session_id,
                "checkpointId": selected_checkpoint_id,
                "claudeSessionId": recovered_claude_session_id,
            }
        if record.lock.locked():
            return {
                "ok": False,
                "status": "busy",
                "error": "session is currently running",
                "sessionId": session_id,
                "checkpointId": selected_checkpoint_id,
                "claudeSessionId": record.claude_session_id,
            }
        async with record.lock:
            try:
                applied_checkpoint_id = await self._rewind_files_with_duplicate_fallback(
                    record.client,
                    session_id=session_id,
                    checkpoint=checkpoint,
                    selected_checkpoint_id=selected_checkpoint_id,
                )
            except Exception as exc:
                logger.exception(
                    "checkpoint rewind failed",
                    extra={
                        "frontend_session_id": session_id,
                        "checkpoint_id": selected_checkpoint_id,
                        "claude_session_id": record.claude_session_id,
                    },
                )
                return {
                    "ok": False,
                    "status": "rewind_failed",
                    "error": str(exc) or "checkpoint rewind failed",
                    "sessionId": session_id,
                    "checkpointId": selected_checkpoint_id,
                    "claudeSessionId": record.claude_session_id,
                }
        rewound_at = time.time()
        await self._checkpoint_store.update_metadata(
            session_id,
            selected_checkpoint_id,
            rewound_checkpoint_id=applied_checkpoint_id,
            rewound_at=rewound_at,
        )
        if applied_checkpoint_id != selected_checkpoint_id:
            await self._checkpoint_store.update_metadata(
                session_id,
                applied_checkpoint_id,
                rewound_checkpoint_id=applied_checkpoint_id,
                rewound_at=rewound_at,
            )
        return {
            "ok": True,
            "status": "completed",
            "sessionId": session_id,
            "checkpointId": applied_checkpoint_id,
            "requestedCheckpointId": selected_checkpoint_id,
            "claudeSessionId": record.claude_session_id,
            "rewoundAt": rewound_at,
            "rewoundCheckpointId": applied_checkpoint_id,
            "action": "rewind",
        }

    async def _recover_checkpoint_client(
        self,
        *,
        frontend_session_id: str,
        claude_session_id: str,
        model: str,
        request_headers: Mapping[str, str] | None,
        current_user: Any | None,
        fallback_tdl_api_key: str,
        proxy_base_url: str,
        skill_mount_root: Any | None,
        runtime_config: Mapping[str, Any] | None,
        force_resume: bool = False,
    ) -> Any | None:
        resolved_claude_session_id = str(claude_session_id or "").strip()
        if not resolved_claude_session_id:
            return None
        sdk = self._load_sdk()
        headers = {str(key): str(value) for key, value in dict(request_headers or {}).items()}
        effective_model = resolve_effective_model(model, self._settings.default_model)
        system_prompt = _build_sdk_system_prompt(self._settings, {})
        runtime_payload = {"runtime_config": dict(runtime_config or {})}
        provider_config = self._resolve_provider_config(runtime_payload, headers)
        proxy_context = await self._proxy_contexts.create(
            upstream_base_url=provider_config.base_url,
            anthropic_version=provider_config.anthropic_version,
            x_user_id=self._resolve_x_user_id(headers, current_user),
            api_token=provider_config.api_token,
            model=effective_model,
            request_headers=headers,
            ttl_sec=self._provider.proxy_context_ttl_sec,
        )
        proxy_env = self._build_proxy_env(
            proxy_base_url,
            proxy_context.proxy_token,
            request_headers=headers,
            current_user=current_user,
            fallback_tdl_api_key=fallback_tdl_api_key,
        )
        mcp_servers = build_mcp_servers(
            self._mcp_settings,
            request_headers=headers,
            fallback_tdl_api_key=fallback_tdl_api_key,
        )
        permission_options = _runtime_permission_options(
            runtime_payload,
            self._settings,
            fallback_runtime_key=_permission_runtime_key(headers),
        )
        skill_platform_catalog = _runtime_skill_platform_catalog(runtime_payload)
        effective_skill_mount_root = (
            None if skill_platform_catalog == "exclude" else skill_mount_root
        )
        session_workspace = await self._session_store.get(frontend_session_id)
        workspace = _resolve_workspace_execution_values(
            session_workspace.workspace_cwd if session_workspace is not None else "",
            session_workspace.workspace_add_dirs if session_workspace is not None else [],
            self._settings,
            source=session_workspace.workspace_source if session_workspace is not None else "session",
            configured=session_workspace.workspace_configured if session_workspace is not None else False,
        )
        effective_workspace_add_dirs = _sdk_add_dirs(
            self._settings,
            effective_skill_mount_root,
            workspace.add_dirs,
        )
        workspace_runtime = self._inspect_workspace_runtime(
            workspace,
            permission_options=permission_options,
            mcp_servers=mcp_servers,
            skill_mount_root=skill_mount_root,
            skill_platform_catalog=skill_platform_catalog,
        )
        signature = _session_signature(
            claude_session_id=resolved_claude_session_id,
            model=effective_model,
            env=proxy_env,
            system_prompt=system_prompt,
            mcp_servers=mcp_servers,
            resumed=True,
            permission_mode=permission_options.permission_mode,
            allowed_tools=permission_options.allowed_tools,
            disallowed_tools=permission_options.disallowed_tools,
            permission_profile=permission_options.profile,
            permission_runtime_key=permission_options.runtime_key,
            permission_revision=permission_options.revision,
            permission_full_bypass=permission_options.full_bypass,
            cwd=str(workspace.cwd),
            add_dirs=effective_workspace_add_dirs,
            workspace_fingerprint=str(workspace_runtime.get("fingerprint") or ""),
            provider_base_url=provider_config.base_url,
            provider_api_token_digest=_secret_digest(provider_config.api_token),
        )
        self._session_contexts[frontend_session_id] = _SessionExecutionContext(
            run_id="",
            claude_session_id=resolved_claude_session_id,
        )
        record = await self._client_pool.get_or_create(
            frontend_session_id,
            claude_session_id=resolved_claude_session_id,
            model=effective_model,
            resumed=True,
            signature=signature,
            workspace_cwd=str(workspace.cwd),
            workspace_add_dirs=[str(path) for path in workspace.add_dirs],
            workspace_source=workspace.source,
            workspace_configured=workspace.configured,
            workspace_runtime=workspace_runtime,
            factory=lambda: self._create_connected_client(
                sdk=sdk,
                model=effective_model,
                system_prompt=system_prompt,
                claude_session_id=resolved_claude_session_id,
                resume_session_id=resolved_claude_session_id,
                force_resume=force_resume,
                proxy_env=proxy_env,
                skill_mount_root=effective_skill_mount_root,
                workspace_cwd=workspace.cwd,
                workspace_add_dirs=workspace.add_dirs,
                mcp_servers=mcp_servers,
                permission_options=permission_options,
                can_use_tool=self._build_can_use_tool_callback(
                    sdk_module=sdk["module"],
                    frontend_session_id=frontend_session_id,
                ),
                hooks=build_sdk_hooks(
                    sdk["module"],
                    frontend_session_id=frontend_session_id,
                    registry=self._hook_registry,
                    execution_resolver=self._resolve_hook_execution,
                ),
            ),
        )
        record.workspace_runtime = merge_observed_runtime(
            workspace_runtime,
            server_info=record.server_info,
        )
        return record

    async def _rewind_files_with_duplicate_fallback(
        self,
        client: Any,
        *,
        session_id: str,
        checkpoint: Any,
        selected_checkpoint_id: str,
    ) -> str:
        candidates = [selected_checkpoint_id]
        prompt_key = _checkpoint_prompt_key(checkpoint)
        if prompt_key:
            for item in await self._checkpoint_store.list_raw(session_id):
                if item.checkpoint_id != selected_checkpoint_id and _checkpoint_prompt_key(item) == prompt_key:
                    candidates.append(item.checkpoint_id)
        last_error: Exception | None = None
        for candidate_id in candidates:
            try:
                await client.rewind_files(candidate_id)
                return candidate_id
            except Exception as exc:
                last_error = exc
                if "No file checkpoint found" not in str(exc):
                    raise
                await self._checkpoint_store.mark_unavailable(
                    session_id,
                    candidate_id,
                    reason="no_file_checkpoint",
                )
                logger.info(
                    "checkpoint rewind candidate had no file checkpoint",
                    extra={"frontend_session_id": session_id, "checkpoint_id": candidate_id},
                )
        if last_error is not None:
            raise RuntimeError("该节点没有可回滚的文件快照，已从列表中移除；请刷新后选择包含文件改动的节点。") from last_error
        raise RuntimeError("checkpoint rewind failed")

    async def list_checkpoints(self, frontend_session_id: str) -> list[Mapping[str, Any]]:
        return [item.to_dict() for item in await self._checkpoint_store.list(frontend_session_id)]

    async def checkpoint_snapshot(self, frontend_session_id: str) -> Mapping[str, Any]:
        session_id = str(frontend_session_id or "").strip()
        record = await self._client_pool.get(session_id)
        mapping = await self._session_store.get(session_id)
        workspace_path = (
            str(record.workspace_cwd or "").strip()
            if record is not None
            else str(mapping.workspace_cwd or "").strip()
            if mapping is not None
            else ""
        ) or str(self._settings.workdir)
        workspace_add_dirs = (
            list(record.workspace_add_dirs)
            if record is not None
            else list(mapping.workspace_add_dirs)
            if mapping is not None
            else []
        )
        return {
            "ok": True,
            "supported": True,
            "enabled": bool(self._settings.enable_file_checkpointing),
            "sessionId": session_id,
            "workspacePath": workspace_path,
            "workspaceAddDirs": workspace_add_dirs,
            "checkpoints": [
                {
                    **item.to_dict(),
                    "workspace_path": workspace_path,
                    "workspacePath": workspace_path,
                }
                for item in await self._checkpoint_store.list(session_id)
            ],
        }

    async def disconnect_all(self) -> None:
        await self._client_pool.disconnect_all()

    async def list_approvals(self, frontend_session_id: str) -> list[Mapping[str, Any]]:
        return await self._approval_registry.list_requests(frontend_session_id)

    async def get_approval(self, frontend_session_id: str, request_id: str) -> Mapping[str, Any] | None:
        return await self._approval_registry.get_request(frontend_session_id, request_id)

    async def stream_approvals(self, frontend_session_id: str) -> AsyncIterator[Mapping[str, Any]]:
        async for payload in self._approval_registry.stream_requests(frontend_session_id):
            yield payload

    async def respond_approval(
        self,
        frontend_session_id: str,
        request_id: str,
        *,
        decision: str,
        reason: str = "",
    ) -> Mapping[str, Any] | None:
        return await self._approval_registry.resolve_request(
            frontend_session_id,
            request_id,
            decision=decision,
            reason=reason,
            interrupt=False,
        )

    async def list_questions(self, frontend_session_id: str) -> list[Mapping[str, Any]]:
        return await self._question_registry.list_questions(frontend_session_id)

    async def get_question(self, frontend_session_id: str, question_id: str) -> Mapping[str, Any] | None:
        return await self._question_registry.get_question(frontend_session_id, question_id)

    async def stream_questions(self, frontend_session_id: str) -> AsyncIterator[Mapping[str, Any]]:
        async for payload in self._question_registry.stream_questions(frontend_session_id):
            yield payload

    async def list_hooks(self, frontend_session_id: str) -> list[Mapping[str, Any]]:
        return await self._hook_registry.list_events(frontend_session_id)

    async def get_hook(self, frontend_session_id: str, event_id: str) -> Mapping[str, Any] | None:
        return await self._hook_registry.get_event(frontend_session_id, event_id)

    async def stream_hooks(self, frontend_session_id: str) -> AsyncIterator[Mapping[str, Any]]:
        async for payload in self._hook_registry.stream_events(frontend_session_id):
            yield payload

    async def answer_question(self, frontend_session_id: str, question_id: str, *, answer: str) -> Mapping[str, Any] | None:
        return await self._question_registry.answer_question(frontend_session_id, question_id, answer=answer)

    async def create_question(
        self,
        frontend_session_id: str,
        *,
        run_id: str,
        prompt: str,
        title: str = "",
        description: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        execution = self._session_contexts.get(frontend_session_id) or _SessionExecutionContext(run_id=run_id, claude_session_id="")
        question = await self._question_registry.create_question(
            session_id=frontend_session_id,
            run_id=execution.run_id or run_id,
            claude_session_id=execution.claude_session_id,
            prompt=prompt,
            title=title,
            description=description,
            metadata=metadata,
        )
        return question.snapshot()

    async def _record_hook_event(self, frontend_session_id: str, item: Any) -> str | None:
        payload = hook_stream_payload(item)
        if payload is None:
            return None
        run_id, claude_session_id = self._resolve_hook_execution(frontend_session_id)
        await self._hook_registry.record_event(
            session_id=frontend_session_id,
            run_id=run_id,
            claude_session_id=str(payload.get("claudeSessionId") or claude_session_id or "").strip(),
            event_id=str(payload.get("eventId") or "").strip(),
            hook_event_name=str(payload.get("hookEventName") or "").strip(),
            phase=str(payload.get("phase") or "").strip(),
            source=str(payload.get("source") or "").strip(),
            status=str(payload.get("status") or "").strip(),
            matcher=str(payload.get("matcher") or "").strip(),
            tool_name=str(payload.get("toolName") or "").strip(),
            tool_use_id=str(payload.get("toolUseId") or "").strip(),
            agent_id=str(payload.get("agentId") or "").strip(),
            agent_type=str(payload.get("agentType") or "").strip(),
            title=str(payload.get("title") or "").strip(),
            notification_type=str(payload.get("notificationType") or "").strip(),
            data=payload.get("data") if isinstance(payload.get("data"), Mapping) else None,
            output=payload.get("output") if isinstance(payload.get("output"), Mapping) else None,
            outcome=str(payload.get("outcome") or "").strip(),
            exit_code=payload.get("exitCode") if isinstance(payload.get("exitCode"), int) else None,
        )
        return await self._record_goal_stop_hook_result(frontend_session_id, run_id=run_id, payload=payload)

    async def _record_goal_stop_hook_result(
        self,
        frontend_session_id: str,
        *,
        run_id: str,
        payload: Mapping[str, Any],
    ) -> str | None:
        result = _goal_stop_hook_result(payload)
        if result is None:
            return None
        goal = await self._goal_store.get(frontend_session_id)
        if goal is None:
            return None
        command = str(result.get("command") or "").strip()
        if command and _collapse_space(command) != _collapse_space(goal.objective):
            return None
        status = str(result.get("status") or "").strip() or "running"
        summary = str(result.get("summary") or "").strip()
        await self._goal_store.record_run_result(
            frontend_session_id,
            run_id=run_id,
            status=status,
            summary=summary,
            tasks=[],
        )
        return status


def _is_ignorable_terminal_exception(exc: Exception) -> bool:
    text = str(exc or "").strip().lower()
    if text == "claude code returned an error result: success":
        return True
    if text.startswith("claude code returned an error result: success;"):
        return True
    return False


def _open_task_message_stream(client: Any) -> AsyncIterator[Any]:
    receiver = getattr(client, "receive_messages", None)
    if callable(receiver):
        return receiver().__aiter__()
    return client.receive_response().__aiter__()


def _merge_file_lists(left: list[str], right: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for item in [*left, *right]:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        merged.append(text)
    return merged


def _build_sdk_system_prompt(settings: ClaudeSettings, payload: Mapping[str, Any]) -> Any:
    dynamic_prompt = build_system_prompt(payload)
    append_parts = [
        part
        for part in (_DESTRUCTIVE_COMMAND_SYSTEM_WARNING, settings.system_prompt_append, dynamic_prompt)
        if str(part or "").strip()
    ]
    append_text = "\n\n".join(str(part).strip() for part in append_parts if str(part).strip()).strip()
    if settings.system_prompt_file is not None:
        return {"type": "file", "path": str(settings.system_prompt_file)}
    if settings.system_prompt_preset:
        preset: dict[str, Any] = {
            "type": "preset",
            "preset": settings.system_prompt_preset,
        }
        if append_text:
            preset["append"] = append_text
        return preset
    return append_text or None


def _normalize_goal_command_text(text: str) -> str:
    value = str(text or "").strip()
    while value.lower().startswith("/goal"):
        rest = value[5:]
        if rest and not rest[0].isspace():
            break
        value = rest.strip()
    return value


def _goal_sdk_command_text(runtime_command: Any, goal_text: str) -> str:
    command = str(getattr(runtime_command, "command", "") or "").strip() or "/goal"
    if not command.startswith("/"):
        command = f"/{command}"
    clean_goal = str(goal_text or "").strip()
    return f"{command} {clean_goal}".strip()


def _goal_runtime_command_payload(
    *,
    status: str,
    phase: str,
    message: str = "",
    result: str = "",
    run_id: str = "",
) -> dict[str, Any]:
    clean_run_id = str(run_id or "").strip()
    return {
        "source": "claude-code",
        "commandId": "goal",
        "command": "/goal",
        "args": {},
        "displayName": "目标模式",
        "requestId": f"cmd-goal-recovery-{clean_run_id or uuid.uuid4().hex}",
        "status": status,
        "phase": phase,
        **({"message": message} if message else {}),
        **({"result": result} if result else {}),
        **({"runId": clean_run_id} if clean_run_id else {}),
    }


def _auto_compact_request_id(frontend_session_id: str, run_id: str | None) -> str:
    clean_run_id = str(run_id or "").strip()
    if clean_run_id:
        return f"cmd-auto-compact-{clean_run_id}"
    clean_session_id = str(frontend_session_id or "").strip()
    if clean_session_id:
        return f"cmd-auto-compact-{clean_session_id}"
    return f"cmd-auto-compact-{uuid.uuid4().hex}"


def _compact_runtime_command_payload(
    *,
    request_id: str,
    status: str,
    phase: str,
    message: str,
    result: str = "",
    source: str = "claude-code",
) -> dict[str, Any]:
    return {
        "source": source,
        "commandId": "compact",
        "command": "/compact",
        "args": {},
        "displayName": "压缩上下文",
        "requestId": request_id,
        "status": status,
        "phase": phase,
        "message": message,
        **({"result": result} if result else {}),
    }


def _normalize_goal_runtime_snapshot(
    goal_runtime: Mapping[str, Any],
    client_sessions: Any,
) -> dict[str, Any]:
    payload = dict(goal_runtime)
    sessions = payload.get("sessions")
    if not isinstance(sessions, list):
        payload["activeGoalNum"] = 0
        return payload

    active_session_ids: set[str] = set()
    if isinstance(client_sessions, list):
        for item in client_sessions:
            if not isinstance(item, Mapping):
                continue
            frontend_session_id = str(item.get("frontendSessionId") or item.get("frontend_session_id") or "").strip()
            if frontend_session_id:
                active_session_ids.add(frontend_session_id)

    normalized_sessions: list[Any] = []
    for item in sessions:
        if not isinstance(item, Mapping):
            normalized_sessions.append(item)
            continue
        goal = dict(item)
        frontend_session_id = str(goal.get("frontendSessionId") or goal.get("frontend_session_id") or "").strip()
        if goal.get("status") == "running" and frontend_session_id and frontend_session_id not in active_session_ids:
            goal["status"] = "paused"
            goal["activeRunId"] = ""
            goal["active_run_id"] = ""
            pause_reason = str(goal.get("pauseReason") or goal.get("pause_reason") or "process_interrupted").strip()
            goal["pauseReason"] = pause_reason
            goal["pause_reason"] = pause_reason
        normalized_sessions.append(goal)

    payload["sessions"] = normalized_sessions
    payload["activeGoalNum"] = sum(
        1
        for item in normalized_sessions
        if isinstance(item, Mapping) and item.get("status") == "running"
    )
    payload["goalSessionNum"] = len([item for item in normalized_sessions if isinstance(item, Mapping)])
    return payload


def _goal_status_label(status: str) -> str:
    value = str(status or "").strip().lower()
    if value == "completed":
        return "已完成"
    if value == "failed":
        return "失败"
    if value == "cleared":
        return "已清除"
    if value == "paused":
        return "已暂停"
    return "进行中"


def _goal_needs_recovery_decision(goal: SessionGoal, *, has_active_client: bool) -> bool:
    status = str(goal.status or "").strip().lower()
    if status == "paused":
        return True
    if status != "running":
        return False
    return not has_active_client


def _latest_user_text(payload: Mapping[str, Any]) -> str:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return ""
    for item in reversed(messages):
        if not isinstance(item, Mapping):
            continue
        if str(item.get("role") or "").strip().lower() != "user":
            continue
        content = item.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, Mapping) and str(block.get("type") or "").strip().lower() in {"text", "input_text"}:
                    parts.append(str(block.get("text") or ""))
            return "\n".join(part for part in parts if part).strip()
    return ""


def _goal_run_observation_summary(tasks: list[Mapping[str, Any]], *, emitted_text: bool) -> str:
    if tasks:
        completed = sum(1 for task in tasks if str(task.get("status") or "").strip().lower() in {"completed", "success", "succeeded", "done"})
        failed = sum(1 for task in tasks if str(task.get("status") or "").strip().lower() in {"failed", "error", "cancelled", "canceled", "stopped", "killed", "timeout"})
        return f"本轮任务观测：完成 {completed}/{len(tasks)}，失败 {failed}；目标保持进行中，等待 Claude Code/Stop hook 权威确认。"
    return "本轮已有文本输出，目标保持进行中。" if emitted_text else "本轮未收到任务终态，目标保持进行中。"


def _goal_stop_hook_result(payload: Mapping[str, Any]) -> Mapping[str, str] | None:
    hook_event = str(payload.get("hookEventName") or "").strip().lower()
    title = str(payload.get("title") or "").strip().lower()
    data = payload.get("data") if isinstance(payload.get("data"), Mapping) else {}
    data_hook = str(data.get("hookEvent") or data.get("hookName") or "").strip().lower()
    if "stop" not in {hook_event, title, data_hook}:
        return None

    command = str(data.get("command") or payload.get("command") or "").strip()
    stdout = str(data.get("stdout") or payload.get("stdout") or "").strip()
    stderr = str(data.get("stderr") or payload.get("outcome") or "").strip()
    output = payload.get("output") if isinstance(payload.get("output"), Mapping) else {}
    parsed = output if isinstance(output.get("ok"), bool) else _json_object_from_text(stdout)

    if isinstance(parsed, Mapping) and isinstance(parsed.get("ok"), bool):
        ok = bool(parsed.get("ok"))
        reason = str(parsed.get("reason") or parsed.get("summary") or "").strip()
        summary = f"Stop hook ok={'true' if ok else 'false'}"
        if reason:
            summary = f"{summary}: {reason}"
        if stderr and not ok:
            summary = f"{summary}；stderr={stderr}"
        return {
            "status": "completed" if ok else "running",
            "summary": summary,
            "command": command,
        }

    exit_code = payload.get("exitCode")
    if stderr or (isinstance(exit_code, int) and exit_code != 0):
        summary = stderr or f"Stop hook exited with code {exit_code}"
        return {
            "status": "failed",
            "summary": summary,
            "command": command,
        }
    return None


def _json_object_from_text(text: str) -> Mapping[str, Any] | None:
    value = str(text or "").strip()
    if not value:
        return None
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, Mapping) else None
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for index in range(len(value) - 1, -1, -1):
        if value[index] != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(value[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, Mapping):
            return parsed
    return None


def _collapse_space(text: str) -> str:
    return " ".join(str(text or "").split())


def _sdk_extra_args(settings: ClaudeSettings) -> dict[str, str | None]:
    args: dict[str, str | None] = {}
    if isinstance(settings.extra_args, Mapping):
        for key, value in settings.extra_args.items():
            name = str(key or "").strip().lstrip("-")
            if name:
                args[name] = None if value is None else str(value)
    if not settings.enable_file_checkpointing:
        return args
    args["replay-user-messages"] = None
    return args


def _permission_extra_args(
    values: Mapping[str, str | None],
    *,
    full_bypass: bool,
) -> dict[str, str | None]:
    args = dict(values)
    if full_bypass:
        args["dangerously-skip-permissions"] = None
    return args


def effective_workspace_setting_sources(settings: ClaudeSettings) -> list[str] | None:
    if settings.setting_sources is not None:
        return list(settings.setting_sources)
    # Claude Agent SDK enables user/project sources when the skills option is
    # set. Mirror that transport-level default in the inspection contract.
    if settings.skills_filter is not None:
        return ["user", "project"]
    return None


def _permission_runtime_key(headers: Mapping[str, str] | None) -> str:
    for name, value in dict(headers or {}).items():
        if str(name or "").strip().lower() == "x-agent-runtime-id":
            return str(value or "").strip()
    return ""


def _resolve_workspace_execution(
    payload: Mapping[str, Any],
    settings: ClaudeSettings,
) -> _WorkspaceExecution:
    workspace = _workspace_config_from_payload(payload)
    if workspace is None:
        return _resolve_workspace_execution_values(
            "",
            [],
            settings,
            source="agent_default",
            configured=False,
        )
    paths = _workspace_path_values(workspace.get("paths"))
    cwd = str(
        workspace.get("cwd")
        or workspace.get("working_directory")
        or workspace.get("workingDirectory")
        or (paths[0] if paths else "")
        or ""
    ).strip()
    raw_add_dirs = (
        workspace.get("add_dirs")
        or workspace.get("addDirs")
        or workspace.get("additional_directories")
        or workspace.get("additionalDirectories")
    )
    add_dirs = _workspace_path_values(raw_add_dirs)
    if not add_dirs and paths:
        add_dirs = paths[1:] if cwd else paths
    source = str(workspace.get("source") or workspace.get("scope") or "request").strip() or "request"
    return _resolve_workspace_execution_values(
        cwd,
        add_dirs,
        settings,
        source=source,
        configured=True,
    )


def _resolve_workspace_execution_values(
    cwd: Any,
    add_dirs: Sequence[Any] | None,
    settings: ClaudeSettings,
    *,
    source: str = "agent_default",
    configured: bool = False,
) -> _WorkspaceExecution:
    resolved_cwd = _resolve_workspace_directory(cwd or settings.workdir, settings.workdir)
    resolved_add_dirs: list[Path] = []
    seen = {str(resolved_cwd)}
    for item in add_dirs or []:
        resolved = _resolve_workspace_directory(item, settings.workdir)
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        resolved_add_dirs.append(resolved)
    return _WorkspaceExecution(
        cwd=resolved_cwd,
        add_dirs=tuple(resolved_add_dirs),
        source=str(source or "agent_default").strip() or "agent_default",
        configured=bool(configured),
    )


def _workspace_config_from_payload(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        agent_config = metadata.get("agentconfig") or metadata.get("agentConfig")
        if isinstance(agent_config, Mapping):
            workspace = (
                agent_config.get("workspace")
                or agent_config.get("workspace_config")
                or agent_config.get("workspaceConfig")
            )
            if isinstance(workspace, Mapping):
                return workspace
        workspace = metadata.get("workspace")
        if isinstance(workspace, Mapping):
            return workspace
    workspace = payload.get("workspace")
    return workspace if isinstance(workspace, Mapping) else None


def _workspace_path_values(value: Any) -> list[str]:
    items = value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else [value]
    return list(dict.fromkeys(
        str(item or "").strip()
        for item in items
        if str(item or "").strip()
    ))


def _resolve_workspace_directory(value: Any, default_base: Path) -> Path:
    text = str(value or "").strip()
    path = Path(text).expanduser() if text else default_base
    if not path.is_absolute():
        path = default_base / path
    resolved = path.resolve()
    if resolved.is_file():
        logger.info("[workspace] file path normalized to parent directory path=%s", resolved)
        resolved = resolved.parent
    if not resolved.exists():
        raise ValueError(f"workspace directory does not exist: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"workspace path is not a directory: {resolved}")
    return resolved


def _sdk_add_dirs(
    settings: ClaudeSettings,
    skill_mount_root: Any,
    workspace_add_dirs: Sequence[Any] | None = None,
) -> list[str]:
    paths: list[str] = []
    for item in [*(settings.add_dirs or []), *(workspace_add_dirs or [])]:
        text = str(item or "").strip()
        if text and text not in paths:
            paths.append(text)
    mount = str(skill_mount_root or "").strip()
    if mount and mount not in paths:
        paths.append(mount)
    return paths


def _runtime_skill_platform_catalog(payload: Mapping[str, Any]) -> str:
    """Read the shared Agent Skill policy from root or metadata runtime config."""

    for runtime_config in _runtime_config_sections(payload):
        skills = runtime_config.get("skills")
        if not isinstance(skills, Mapping):
            continue
        value = str(
            skills.get("platform_catalog")
            or skills.get("platformCatalog")
            or ""
        ).strip().lower()
        if value in {"include", "exclude"}:
            return value
    return "include"


def _runtime_permission_options(
    payload: Mapping[str, Any],
    settings: ClaudeSettings,
    *,
    fallback_runtime_key: str = "",
) -> PermissionOptions:
    for runtime_config in _runtime_config_sections(payload):
        permissions = runtime_config.get("permissions")
        if isinstance(permissions, Mapping):
            return permission_options_from_runtime_config(
                settings,
                permissions,
                fallback_runtime_key=fallback_runtime_key,
            )
    return permission_options_from_runtime_config(
        settings,
        None,
        fallback_runtime_key=fallback_runtime_key,
    )


def _runtime_config_sections(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    runtime_configs: list[Mapping[str, Any]] = []
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        agent_config = metadata.get("agentconfig") or metadata.get("agentConfig")
        if isinstance(agent_config, Mapping):
            nested = agent_config.get("runtime_config") or agent_config.get("runtimeConfig")
            if isinstance(nested, Mapping):
                runtime_configs.append(nested)
    root_config = payload.get("runtime_config") or payload.get("runtimeConfig")
    if isinstance(root_config, Mapping):
        runtime_configs.append(root_config)
    return runtime_configs


def _runtime_llm_config(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    for runtime_config in _runtime_config_sections(payload):
        llm = runtime_config.get("llm")
        if isinstance(llm, Mapping):
            return llm
    return {}


def _runtime_llm_value(llm_config: Mapping[str, Any], *keys: str) -> str:
    if not isinstance(llm_config, Mapping):
        return ""
    for key in keys:
        value = str(llm_config.get(key) or "").strip()
        if value:
            return value
    return ""


def _normalize_provider_base_url(value: str) -> str:
    text = str(value or "").strip().rstrip("/")
    if not text:
        return ""
    lowered = text.lower()
    for suffix in ("/v1/messages/count_tokens", "/v1/messages", "/v1/chat/completions", "/v1"):
        if lowered.endswith(suffix):
            return text[: -len(suffix)].rstrip("/")
    return text


def _sdk_process_env(settings: ClaudeSettings, proxy_env: Mapping[str, str]) -> dict[str, str]:
    env: dict[str, str] = {}
    if isinstance(settings.env, Mapping):
        for key, value in settings.env.items():
            name = str(key or "").strip()
            if name:
                env[name] = str(value)
    env.update({str(key): str(value) for key, value in proxy_env.items()})
    if not any(key.upper() == "PATH" for key in env):
        process_path = _sdk_process_path(settings)
        if process_path:
            env["PATH"] = process_path
    return env


def _sdk_process_path(settings: ClaudeSettings) -> str:
    home = Path.home()
    candidates: list[Path] = []
    if settings.cli_path:
        candidates.append(Path(settings.cli_path).expanduser().parent)
    candidates.extend(
        [
            Path(sys.executable).resolve().parent,
            home / ".local" / "bin",
            home / ".npm-global" / "bin",
            home / ".n" / "bin",
            home / "bin",
        ]
    )
    for pattern in (".nvm/versions/node/*/bin", ".local/opt/node-*/bin"):
        candidates.extend(sorted(home.glob(pattern), reverse=True))
    app_data = str(os.environ.get("APPDATA") or "").strip()
    if app_data:
        candidates.append(Path(app_data) / "npm")

    inherited = [
        Path(item)
        for item in str(os.environ.get("PATH") or "").split(os.pathsep)
        if str(item).strip()
    ]
    candidates.extend(inherited)

    resolved: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        text = str(candidate).strip()
        if not text:
            continue
        key = os.path.normcase(os.path.abspath(text))
        if key in seen:
            continue
        seen.add(key)
        resolved.append(text)
    return os.pathsep.join(resolved)


def _sanitize_cli_stderr_line(line: str, sensitive_values: list[str]) -> str:
    text = str(line or "").strip()
    for value in sensitive_values:
        text = text.replace(value, "***")
    if len(text) > _CLI_STDERR_MAX_CHARS:
        text = text[:_CLI_STDERR_MAX_CHARS] + "..."
    return text


def _sdk_agents(sdk_module: Any, raw_agents: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(raw_agents, Mapping) or not raw_agents:
        return None
    agent_cls = getattr(sdk_module, "AgentDefinition", None)
    if agent_cls is None:
        types_module = getattr(sdk_module, "types", None)
        agent_cls = getattr(types_module, "AgentDefinition", None)
    if agent_cls is None:
        return None
    agents: dict[str, Any] = {}
    for name, raw in raw_agents.items():
        agent_name = str(name or "").strip()
        if not agent_name or not isinstance(raw, Mapping):
            continue
        payload = _normalize_agent_definition(dict(raw))
        try:
            agents[agent_name] = agent_cls(**payload)
        except TypeError:
            logger.warning("[claude-sdk] skipped invalid agent definition name=%s", agent_name)
    return agents or None


def _normalize_agent_definition(raw: Mapping[str, Any]) -> dict[str, Any]:
    aliases = {
        "disallowed_tools": "disallowedTools",
        "mcp_servers": "mcpServers",
        "initial_prompt": "initialPrompt",
        "max_turns": "maxTurns",
        "permission_mode": "permissionMode",
    }
    out: dict[str, Any] = {}
    for key, value in raw.items():
        name = aliases.get(str(key), str(key))
        out[name] = value
    return out


def _build_claude_session_id(frontend_session_id: str | None) -> str:
    text = str(frontend_session_id or "").strip()
    if text:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"claude-sdk-agent:{text}"))
    return str(uuid.uuid4())


def _skill_usage_audit_context(
    *,
    payload: Mapping[str, Any],
    request_headers: Mapping[str, str],
    current_user: Any | None,
    proxy_base_url: str,
    frontend_session_id: str,
    run_id: str,
) -> Mapping[str, Any]:
    user_id = _first_non_empty(
        getattr(current_user, "emp_id", None),
        _metadata_userinfo_value(payload, "employNum", "employ_num", "employeeNum", "employee_num", "empId", "emp_id"),
        request_headers.get("x-user-id"),
        request_headers.get("uac-user-id"),
        request_headers.get("x-uac-user-id"),
    ) or "unknown"
    base_url = str(proxy_base_url or "").strip().rstrip("/")
    endpoint = "/v1/chat/completions"
    return {
        "agent_id": "claude_sdk_agent",
        "agent_name": "Claude Code",
        "operator_user_id": user_id,
        "agent_base_url": base_url,
        "agent_endpoint": endpoint,
        "url": f"{base_url}{endpoint}" if base_url else endpoint,
        "request_id": request_headers.get("x-request-id") or run_id or str(uuid.uuid4()),
        "session_id_hash": _hash_session_id(frontend_session_id),
        "entry": "chat",
    }


def _skill_usage_request_headers(
    request_headers: Mapping[str, str],
    *,
    frontend_session_id: str,
    run_id: str,
) -> Mapping[str, str]:
    headers = {str(key): str(value) for key, value in dict(request_headers or {}).items()}
    if frontend_session_id and not _has_header(headers, "x-session-id"):
        headers["x-session-id"] = frontend_session_id
    if run_id and not _has_header(headers, "x-request-id"):
        headers["x-request-id"] = run_id
    return headers


def _has_header(headers: Mapping[str, str], name: str) -> bool:
    expected = name.lower()
    return any(str(key or "").strip().lower() == expected and str(value or "").strip() for key, value in headers.items())


def _metadata_userinfo_value(payload: Mapping[str, Any], *keys: str) -> str:
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return ""
    userinfo = metadata.get("userinfo") or metadata.get("userInfo")
    if not isinstance(userinfo, Mapping):
        return ""
    for key in keys:
        value = str(userinfo.get(key) or "").strip()
        if value:
            return value
    return ""


def _hash_session_id(session_id: str) -> str:
    text = str(session_id or "").strip()
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _secret_digest(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _request_auth_env(
    request_headers: Mapping[str, str],
    current_user: Any | None,
    *,
    api_token: str,
    fallback_tdl_api_key: str = "",
) -> dict[str, str]:
    user_id = _first_non_empty(
        getattr(current_user, "emp_id", None),
        request_headers.get("x-user-id"),
        request_headers.get("uac-user-id"),
        request_headers.get("x-uac-user-id"),
    )
    user_token = _first_non_empty(
        request_headers.get("uac-user-token"),
        request_headers.get("x-uac-user-token"),
    )
    env: dict[str, str] = {}
    if user_id:
        env.update(
            {
                "USER": user_id,
                "UAC_USER_ID": user_id,
                "VISION_MODEL_USER_ID": user_id,
                "coclaw_empno": user_id,
                "RDCLOUD_EMP_NO": user_id,
            }
        )
    if user_token:
        env.update(
            {
                "UAC_USER_TOKEN": user_token,
                "VISION_MODEL_USER_TOKEN": user_token,
                "coclaw_token": user_token,
                "RDCLOUD_AUTH_TOKEN": user_token,
            }
        )
    if api_token:
        env.setdefault("VISION_MODEL_API_KEY", api_token)
        if str(api_token).strip().startswith("tdl_"):
            env["TDL_API_KEY"] = api_token
    if "TDL_API_KEY" not in env:
        fallback = str(fallback_tdl_api_key or "").strip()
        if fallback.startswith("tdl_"):
            env["TDL_API_KEY"] = fallback
    return env


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _first_text_block(blocks: list[Mapping[str, Any]]) -> str:
    for block in blocks:
        if not isinstance(block, Mapping):
            continue
        if str(block.get("type") or "").strip().lower() != "text":
            continue
        text = str(block.get("text") or "").strip()
        if text:
            return text
    return ""

def _is_ask_user_question_tool(tool_name: str) -> bool:
    return str(tool_name or "").strip().lower() == "askuserquestion"


def _ask_user_question_items(tool_input: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    if not isinstance(tool_input, Mapping):
        return []
    questions = tool_input.get("questions")
    if not isinstance(questions, list):
        return []
    return [item for item in questions if isinstance(item, Mapping)]


def _ask_user_question_prompt(tool_input: Mapping[str, Any] | None) -> str:
    prompts = [
        str(item.get("question") or "").strip()
        for item in _ask_user_question_items(tool_input)
        if str(item.get("question") or "").strip()
    ]
    if prompts:
        return "\n".join(prompts)
    return "Claude Code 需要补充信息才能继续。"


def _ask_user_question_title(tool_input: Mapping[str, Any] | None) -> str:
    items = _ask_user_question_items(tool_input)
    first_header = str(items[0].get("header") or "").strip() if items else ""
    if first_header:
        return first_header
    return "需要补充信息"


def _ask_user_question_description(tool_input: Mapping[str, Any] | None) -> str:
    count = len(_ask_user_question_items(tool_input))
    if count > 1:
        return f"Claude Code 请求你回答 {count} 个问题后继续。"
    return "Claude Code 请求你回答后继续。"


def _ask_user_question_updated_input(tool_input: Mapping[str, Any] | None, answer: str) -> dict[str, Any]:
    updated = dict(tool_input or {})
    answer_text = str(answer or "")
    answers: dict[str, str] = {}
    for item in _ask_user_question_items(updated):
        question_text = str(item.get("question") or "").strip()
        if question_text:
            answers[question_text] = answer_text
    if not answers:
        answers["answer"] = answer_text
    updated["answers"] = answers
    return updated


_SESSION_SIGNATURE_VOLATILE_ENV_KEYS = {
    # Internal proxy token generated per request by ProxyContextStore. Including it
    # in the client-pool signature prevents SDK client reuse across turns.
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_API_KEY",
}


def _session_signature_env(env: Mapping[str, str]) -> dict[str, str]:
    return {
        str(key): str(value)
        for key, value in sorted(env.items())
        if str(key) not in _SESSION_SIGNATURE_VOLATILE_ENV_KEYS
    }


def _session_signature(
    *,
    claude_session_id: str,
    model: str,
    env: Mapping[str, str],
    system_prompt: Any,
    mcp_servers: Mapping[str, Any],
    resumed: bool,
    permission_mode: str,
    allowed_tools: list[str] | None = None,
    disallowed_tools: list[str] | None = None,
    permission_profile: str = "",
    permission_runtime_key: str = "",
    permission_revision: int = 0,
    permission_full_bypass: bool = False,
    cwd: str = "",
    add_dirs: Sequence[str] | None = None,
    workspace_fingerprint: str = "",
    skill_platform_catalog: str = "include",
    provider_base_url: str = "",
    provider_api_token_digest: str = "",
) -> str:
    payload = {
        "claude_session_id": claude_session_id,
        "model": model,
        "env": _session_signature_env(env),
        "system_prompt": system_prompt,
        "mcp_servers": dict(sorted(mcp_servers.items())),
        "resumed": resumed,
        "permission_mode": permission_mode,
        "allowed_tools": list(allowed_tools or []),
        "disallowed_tools": list(disallowed_tools or []),
        "permission_profile": permission_profile,
        "permission_runtime_key": permission_runtime_key,
        "permission_revision": int(permission_revision),
        "permission_full_bypass": bool(permission_full_bypass),
        "cwd": str(cwd or "").strip(),
        "add_dirs": [str(path) for path in (add_dirs or [])],
        "workspace_fingerprint": str(workspace_fingerprint or "").strip(),
        "skill_platform_catalog": str(skill_platform_catalog or "include").strip().lower(),
        "provider_base_url": str(provider_base_url or "").strip().rstrip("/"),
        "provider_api_token_digest": str(provider_api_token_digest or "").strip(),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def _checkpoint_prompt_key(checkpoint: Any) -> str:
    return " ".join(str(getattr(checkpoint, "prompt_excerpt", "") or "").split())


def _truncate(text: str, limit: int) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


class _SessionMessageAdapter:
    def __init__(self, message: Any) -> None:
        self.message = message


class _ToolEventBridge:
    def __init__(
        self,
        registry: ToolRuntimeRegistry | None,
        task_registry: TaskRuntimeRegistry | None,
        *,
        run_id: str | None,
    ) -> None:
        self._registry = registry
        self._task_registry = task_registry
        self._run_id = str(run_id or "").strip()
        self._contexts: dict[str, ToolControlContext] = {}
        self._task_contexts: dict[str, TaskControlContext] = {}
        self._tool_meta: dict[str, dict[str, Any]] = {}
        self._task_to_tool: dict[str, str] = {}
        self._task_meta: dict[str, dict[str, Any]] = {}
        self._finished: set[str] = set()
        self._finished_tasks: set[str] = set()

    async def handle_item(self, item: Any) -> AsyncIterator[RuntimeStreamEvent]:
        event = stream_event_tool_start(item)
        if event is not None:
            shell_event = await self._start_tool(
                tool_call_id=event["toolCallId"],
                name=event["name"],
                arguments=event["arguments"],
                tool_type=event["toolType"],
            )
            if shell_event is not None:
                yield RuntimeStreamEvent(kind="tool", payload=shell_event)
        for event in content_tool_starts(item):
            shell_event = await self._start_tool(
                tool_call_id=event["toolCallId"],
                name=event["name"],
                arguments=event["arguments"],
                tool_type=event["toolType"],
            )
            if shell_event is not None:
                yield RuntimeStreamEvent(kind="tool", payload=shell_event)
        task_payload = task_message_payload(item)
        if task_payload is not None:
            task_event = self._task_shell_payload(item, task_payload)
            if task_event is not None:
                yield RuntimeStreamEvent(kind="task", payload=task_event)
            async for shell_event in self._handle_task_payload(item, task_payload):
                yield RuntimeStreamEvent(kind="tool", payload=shell_event)
        for event in content_tool_results(item):
            shell_event = await self._finish_tool(
                tool_call_id=event["toolCallId"],
                status=str(event["status"] or "completed"),
                result=str(event["result"] or ""),
            )
            if shell_event is not None:
                yield RuntimeStreamEvent(kind="tool", payload=shell_event)

    async def _handle_task_payload(self, item: Any, payload: Mapping[str, Any]) -> AsyncIterator[Mapping[str, Any]]:
        tool_call_id = str(payload.get("toolCallId") or "").strip()
        task_id = self._task_id_from_item_payload(item, payload)
        task_type = self._task_type_from_item_payload(item, payload)
        if tool_call_id and task_id:
            self._task_to_tool[task_id] = tool_call_id
        if not tool_call_id and task_id:
            tool_call_id = self._task_to_tool.get(task_id, "")
        status = str(payload.get("status") or "running").strip() or "running"
        name = str(payload.get("name") or "").strip()
        log_text = str(payload.get("log") or "").strip()
        self._remember_task(
            task_id=task_id,
            task_type=task_type or "task",
            tool_call_id=tool_call_id or "",
            name=name or log_text or task_id or "task",
            status=status,
        )
        await self._track_task(
            task_id=task_id,
            description=name or log_text or task_id or "task",
            task_type=task_type or "task",
            tool_call_id=tool_call_id or None,
            status=status,
            log_text=log_text,
            metadata={"phase": str(payload.get("phase") or ""), "result": str(payload.get("result") or "")},
        )
        if payload.get("phase") == "start":
            shell_event = await self._start_tool(
                tool_call_id=tool_call_id or task_id,
                name=name or "task",
                arguments={},
                tool_type="claude_task",
            )
            if shell_event is not None:
                yield shell_event
        elif payload.get("phase") == "update":
            if (tool_call_id or task_id) not in self._tool_meta:
                shell_event = await self._start_tool(
                    tool_call_id=tool_call_id or task_id,
                    name=name or "task",
                    arguments={},
                    tool_type="claude_task",
                )
                if shell_event is not None:
                    yield shell_event
            shell_event = await self._update_tool(
                tool_call_id=tool_call_id or task_id,
                status=status,
                name=name,
                partial_result=log_text,
            )
            if shell_event is not None:
                yield shell_event
        elif payload.get("phase") == "end":
            if (tool_call_id or task_id) not in self._tool_meta:
                shell_event = await self._start_tool(
                    tool_call_id=tool_call_id or task_id,
                    name=name or "task",
                    arguments={},
                    tool_type="claude_task",
                )
                if shell_event is not None:
                    yield shell_event
            shell_event = await self._finish_tool(
                tool_call_id=tool_call_id or task_id,
                status=status,
                result=str(payload.get("result") or log_text or ""),
                name=name,
            )
            if shell_event is not None:
                yield shell_event
        if log_text:
            await self._emit_output(tool_call_id or task_id, log_text)

    def has_open_tasks(self) -> bool:
        return any(task_id not in self._finished_tasks for task_id in self._task_meta)

    async def waiting_open_tasks(self, *, log_text: str = _TASK_STREAM_WAITING_LOG) -> AsyncIterator[RuntimeStreamEvent]:
        for task_id, meta in list(self._task_meta.items()):
            if task_id in self._finished_tasks:
                continue
            payload = {
                "phase": "waiting_subtasks",
                "runId": self._run_id,
                "taskId": task_id,
                "taskType": str(meta.get("taskType") or "task"),
                "toolCallId": str(meta.get("toolCallId") or self._task_to_tool.get(task_id, "")),
                "name": str(meta.get("name") or task_id),
                "status": "running",
                "log": log_text,
            }
            yield RuntimeStreamEvent(kind="task", payload=payload)

    async def finish_open_tasks(self, *, log_text: str = _TASK_STREAM_ENDED_LOG) -> AsyncIterator[RuntimeStreamEvent]:
        for task_id, meta in list(self._task_meta.items()):
            if task_id in self._finished_tasks:
                continue
            payload = {
                "phase": "end",
                "runId": self._run_id,
                "taskId": task_id,
                "taskType": str(meta.get("taskType") or "task"),
                "toolCallId": str(meta.get("toolCallId") or self._task_to_tool.get(task_id, "")),
                "name": str(meta.get("name") or task_id),
                "status": "ended",
                "log": log_text,
                "result": log_text,
            }
            await self._track_task(
                task_id=task_id,
                description=payload["name"],
                task_type=payload["taskType"],
                tool_call_id=payload["toolCallId"] or None,
                status="ended",
                log_text=log_text,
                metadata={"phase": "end", "result": log_text},
            )
            self._finished_tasks.add(task_id)
            yield RuntimeStreamEvent(kind="task", payload=payload)

    def _remember_task(
        self,
        *,
        task_id: str,
        task_type: str,
        tool_call_id: str,
        name: str,
        status: str,
    ) -> None:
        task_id = str(task_id or "").strip()
        if not task_id:
            return
        normalized_status = str(status or "").strip().lower()
        if normalized_status in _TASK_TERMINAL_STATUSES:
            self._finished_tasks.add(task_id)
        previous = self._task_meta.get(task_id, {})
        self._task_meta[task_id] = {
            "taskType": task_type or previous.get("taskType") or "task",
            "toolCallId": tool_call_id or previous.get("toolCallId") or "",
            "name": name or previous.get("name") or task_id,
        }

    async def _start_tool(
        self,
        *,
        tool_call_id: str,
        name: str,
        arguments: Mapping[str, Any],
        tool_type: str,
    ) -> Mapping[str, Any] | None:
        tool_call_id = str(tool_call_id or "").strip()
        if not tool_call_id:
            return None
        display_name = name or tool_type or "tool"
        meta = self._tool_meta.get(tool_call_id)
        if meta is not None:
            meta["arguments"] = dict(arguments or meta.get("arguments") or {})
            meta["name"] = name or meta.get("name") or display_name
            if self._registry is not None and self._run_id:
                context = self._contexts.get(tool_call_id)
                if context is not None and arguments:
                    await context.update_arguments(arguments)
            return None
        self._tool_meta[tool_call_id] = {
            "name": name or display_name,
            "display_name": display_name,
            "tool_type": tool_type,
            "arguments": dict(arguments or {}),
        }
        if self._registry is not None and self._run_id:
            context = await self._registry.start_tool(
                run_id=self._run_id,
                tool_call_id=tool_call_id,
                name=name or display_name,
                display_name=display_name,
                tool_type=tool_type,
                arguments=arguments,
            )
            self._contexts[tool_call_id] = context
        return {
            "phase": "start",
            "runId": self._run_id,
            "toolCallId": tool_call_id,
            "name": name or display_name,
            "display_name": display_name,
            "status": "running",
            "toolType": tool_type,
            "arguments": dict(arguments or {}),
        }

    async def _update_tool(
        self,
        *,
        tool_call_id: str,
        status: str,
        name: str = "",
        partial_result: str = "",
    ) -> Mapping[str, Any] | None:
        tool_call_id = str(tool_call_id or "").strip()
        if not tool_call_id:
            return None
        if tool_call_id not in self._tool_meta:
            shell_event = await self._start_tool(
                tool_call_id=tool_call_id,
                name=name or "tool",
                arguments={},
                tool_type="claude_task",
            )
            if shell_event is not None:
                pass
        meta = self._tool_meta.get(tool_call_id, {})
        context = self._contexts.get(tool_call_id)
        if context is not None:
            await context.update_status(status or "running")
        payload = {
            "phase": "update",
            "runId": self._run_id,
            "toolCallId": tool_call_id,
            "name": str(meta.get("name") or name or "tool"),
            "status": status or "running",
        }
        if partial_result:
            payload["partialResult"] = partial_result
        return payload

    async def _finish_tool(
        self,
        *,
        tool_call_id: str,
        status: str,
        result: str,
        name: str = "",
    ) -> Mapping[str, Any] | None:
        tool_call_id = str(tool_call_id or "").strip()
        if not tool_call_id:
            return None
        if tool_call_id in self._finished:
            return None
        if tool_call_id not in self._tool_meta:
            await self._start_tool(
                tool_call_id=tool_call_id,
                name=name or "tool",
                arguments={},
                tool_type="claude_task",
            )
        meta = self._tool_meta.get(tool_call_id, {})
        if result:
            await self._emit_output(tool_call_id, result)
        if self._registry is not None and self._run_id:
            await self._registry.finish_tool(run_id=self._run_id, tool_call_id=tool_call_id, status=status)
        self._finished.add(tool_call_id)
        return {
            "phase": "end",
            "runId": self._run_id,
            "toolCallId": tool_call_id,
            "name": str(meta.get("name") or name or "tool"),
            "display_name": str(meta.get("display_name") or meta.get("name") or name or "tool"),
            "status": status or "completed",
            "result": result,
        }

    async def _emit_output(self, tool_call_id: str, text: str) -> None:
        tool_call_id = str(tool_call_id or "").strip()
        if not text or not tool_call_id:
            return
        context = self._contexts.get(tool_call_id)
        if context is not None:
            await context.emit_output("system", text)

    async def _track_task(
        self,
        *,
        task_id: str,
        description: str,
        task_type: str,
        tool_call_id: str | None,
        status: str,
        log_text: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        task_id = str(task_id or "").strip()
        if not task_id or self._task_registry is None or not self._run_id:
            return
        context = self._task_contexts.get(task_id)
        if context is None:
            context = await self._task_registry.start_task(
                run_id=self._run_id,
                task_id=task_id,
                description=description or task_id,
                task_type=task_type or "task",
                tool_call_id=tool_call_id,
                metadata=metadata,
            )
            self._task_contexts[task_id] = context
        else:
            await context.update_status(status or "running", metadata=metadata)
        if log_text:
            await context.emit_output("system", log_text)
        if str(status or "").strip().lower() in _TASK_TERMINAL_STATUSES:
            await self._task_registry.finish_task(
                run_id=self._run_id,
                task_id=task_id,
                status=status,
                metadata=metadata,
            )

    def _task_shell_payload(self, item: Any, payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
        task_id = self._task_id_from_item_payload(item, payload)
        if not task_id:
            return None
        task_type = self._task_type_from_item_payload(item, payload) or "task"
        tool_call_id = str(payload.get("toolCallId") or self._task_to_tool.get(task_id, "") or "").strip()
        body: dict[str, Any] = {
            "phase": str(payload.get("phase") or "").strip() or "update",
            "runId": self._run_id,
            "taskId": task_id,
            "taskType": task_type,
            "status": str(payload.get("status") or "running").strip() or "running",
            "toolCallId": tool_call_id,
            "name": str(payload.get("name") or "").strip() or task_id,
        }
        log_text = str(payload.get("log") or "").strip()
        if log_text:
            body["log"] = log_text
        result = str(payload.get("result") or "").strip()
        if result:
            body["result"] = result
        return body

    @staticmethod
    def _task_id_from_item_payload(item: Any, payload: Mapping[str, Any]) -> str:
        return str(getattr(item, "task_id", "") or payload.get("taskId") or payload.get("task_id") or "").strip()

    @staticmethod
    def _task_type_from_item_payload(item: Any, payload: Mapping[str, Any]) -> str:
        return str(getattr(item, "task_type", "") or payload.get("taskType") or payload.get("task_type") or "").strip()


def _summarize_stream_item(item: Any) -> str:
    item_type = type(item).__name__
    session_id = extract_claude_session_id(item) or "-"
    delta = event_to_text_delta(item)
    if delta:
        return f"type={item_type} session_id={session_id} delta_chars={len(delta)}"
    text = assistant_text(item)
    if text:
        return f"type={item_type} session_id={session_id} text_chars={len(text)}"
    event = getattr(item, "event", None)
    if isinstance(event, dict):
        event_type = str(event.get("type") or "-")
        return f"type={item_type} session_id={session_id} event={event_type}"
    subtype = getattr(item, "subtype", None)
    if subtype is not None:
        return f"type={item_type} session_id={session_id} subtype={subtype}"
    result = getattr(item, "result", None)
    if result is not None:
        return f"type={item_type} session_id={session_id} result={str(result)[:80]}"
    return f"type={item_type} session_id={session_id}"
