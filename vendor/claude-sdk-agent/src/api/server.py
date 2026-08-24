from __future__ import annotations

import asyncio
import json
import logging
import os
import string
from pathlib import Path
import uuid
from contextlib import asynccontextmanager, suppress
from typing import Any, AsyncIterator, Mapping, Sequence

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from ..claude_sdk.runtime import ClaudeSdkRuntime, effective_workspace_setting_sources
from ..claude_sdk.client_pool import ClaudeClientPool
from ..artifacts import ArtifactStore, ClaudeArtifactService
from ..artifacts.opener import ArtifactOpenError, open_file_with_default_app
from ..auth.uac_authz_middleware import install_uac_authz_inject_env_middleware
from ..config import _claude_cli_version, load_settings
from ..context_compact import (
    ContextCompactError,
    ContextCompactor,
    context_compact_error_payload,
)
from ..hook_control import HookRuntimeRegistry, hook_shell_payload
from ..provider.context_store import ProxyContextStore
from ..provider.proxy import install_provider_routes
from ..runtime_files import ensure_audit_links, ensure_runtime_files
from ..session.checkpoint_store import SessionCheckpointStore
from ..session.goal_store import SessionGoalStore
from ..session.store import SessionMappingStore
from ..skills_mount import sync_skill_mount
from ..stream_types import RuntimeStreamEvent
from ..task_control import TaskOutputChunk, TaskRuntimeRegistry
from ..tool_control import ToolOutputChunk, ToolRuntimeRegistry
from ..model_routing import resolve_effective_model
from ..permission_profile import permission_snapshot
from ..workflows_mount import sync_workflow_mount
from .openai_compat import completion_payload, done_chunk, new_chat_id, protocol_shell_chunk, stream_chunk, tool_shell_chunk
from .payload import normalize_session_id_in_payload

logger = logging.getLogger(__name__)
PROJECT_FILE_MAX_BYTES = 100 * 1024 * 1024
PROJECT_FILE_READ_CHUNK_BYTES = 1024 * 1024


def _project_files_root(root: Path) -> Path:
    return root.parent / "my-agents" / "data" / "project-files"


def _sanitize_project_file_segment(value: str) -> str:
    text = "".join(ch for ch in str(value or "").strip() if ch.isalnum() or ch in {"-", "_"})
    return text or "unknown"


def _sanitize_project_filename(filename: str) -> str:
    name = Path(str(filename or "").replace("\x00", "")).name.strip()
    return name or "untitled"


def _project_file_path(
    root: Path,
    *,
    owner_id: str,
    project_id: str,
    file_id: str,
    filename: str,
) -> Path:
    return (
        _project_files_root(root)
        / _sanitize_project_file_segment(owner_id)
        / _sanitize_project_file_segment(project_id)
        / _sanitize_project_file_segment(file_id)
        / _sanitize_project_filename(filename)
    )


def _ensure_managed_project_file_path(root: Path, raw_path: str) -> Path:
    resolved = Path(str(raw_path or "")).expanduser().resolve()
    project_root = _project_files_root(root).resolve()
    if not resolved.is_relative_to(project_root):
        raise HTTPException(status_code=400, detail="Project file path is outside managed storage root")
    return resolved


def _cleanup_empty_project_file_dirs(root: Path, path: Path) -> None:
    project_root = _project_files_root(root).resolve()
    parent = path.parent
    while parent != project_root and parent.is_relative_to(project_root) and parent.exists():
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


async def _write_project_file(path: Path, content: bytes) -> None:
    def write_sync() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    await asyncio.to_thread(write_sync)


async def _read_project_file_upload(file: UploadFile) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(PROJECT_FILE_READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > PROJECT_FILE_MAX_BYTES:
            raise HTTPException(status_code=413, detail="Project file size must not exceed 100MB")
        chunks.append(chunk)
    return b"".join(chunks)


async def _delete_project_file_path(root: Path, raw_path: str) -> bool:
    path = _ensure_managed_project_file_path(root, raw_path)

    def delete_sync() -> bool:
        deleted = False
        if path.exists():
            if path.is_file():
                path.unlink()
                deleted = True
            else:
                raise HTTPException(status_code=400, detail="Project file path is not a file")
        _cleanup_empty_project_file_dirs(root, path)
        return deleted

    return await asyncio.to_thread(delete_sync)


def _runtime_project_file_path(root: Path, path: Path) -> str:
    my_agents_root = root.parent / "my-agents"
    try:
        return str(path.resolve().relative_to(my_agents_root.resolve()))
    except ValueError:
        return str(path.resolve())


def create_app(root: Path | None = None) -> FastAPI:
    app_root = (root or Path(__file__).resolve().parents[2]).resolve()
    settings = load_settings(app_root)
    claude_cli_version = _claude_cli_version(settings.claude.cli_path) if settings.claude.cli_path else ()
    logger.info(
        "[startup] claude cli path=%s version=%s",
        settings.claude.cli_path or "-",
        ".".join(str(part) for part in claude_cli_version) if claude_cli_version else "-",
    )
    ensure_runtime_files(settings)
    audit_links = ensure_audit_links(settings)

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        try:
            yield
        finally:
            runtime = getattr(app.state, "runtime", None)
            if runtime is not None:
                await runtime.disconnect_all()

    app = FastAPI(title="claude-sdk-agent", lifespan=_lifespan)
    skill_mount_root, skill_names = sync_skill_mount(settings.skills)
    workflow_mount_root, workflow_names = sync_workflow_mount(settings.workflows)
    app.state.settings = settings
    app.state.audit_links = audit_links
    app.state.skill_mount_root = skill_mount_root
    app.state.skill_names = skill_names
    app.state.workflow_mount_root = workflow_mount_root
    app.state.workflow_names = workflow_names
    app.state.session_store = SessionMappingStore(settings.sessions.mapping_path)
    app.state.checkpoint_store = SessionCheckpointStore(settings.sessions.checkpoints_path)
    app.state.goal_store = SessionGoalStore(settings.sessions.goals_path)
    app.state.proxy_contexts = ProxyContextStore()
    app.state.client_pool = ClaudeClientPool()
    app.state.context_compactor = ContextCompactor(
        settings.provider,
        default_model=settings.claude.default_model,
    )
    app.state.tool_registry = ToolRuntimeRegistry()
    app.state.task_registry = TaskRuntimeRegistry()
    app.state.hook_registry = HookRuntimeRegistry()
    app.state.artifact_service = ClaudeArtifactService(
        ArtifactStore(app_root / "data" / "artifacts"),
        runtime_root=app_root,
    )
    app.state.runtime = ClaudeSdkRuntime(
        settings.claude,
        settings.mcp,
        settings.provider,
        app.state.session_store,
        app.state.checkpoint_store,
        app.state.proxy_contexts,
        app.state.client_pool,
        app.state.tool_registry,
        app.state.task_registry,
        goal_store=app.state.goal_store,
        hook_registry=app.state.hook_registry,
        workflow_mount_root=app.state.workflow_mount_root,
        skill_usage_audit=settings.skill_usage_audit,
        artifact_service=app.state.artifact_service,
    )
    install_uac_authz_inject_env_middleware(app, settings)
    install_provider_routes(app, settings.provider, app.state.proxy_contexts)

    @app.get("/healthz")
    async def healthz() -> Mapping[str, object]:
        return {"ok": True, "service": "claude-sdk-agent", "port": settings.server.port}

    @app.post("/v1/context/compact")
    async def compact_context(request: Request) -> JSONResponse:
        try:
            payload = await request.json()
        except Exception:
            error = ContextCompactError(
                status_code=400,
                code="invalid_request",
                message="Request body must be valid JSON",
            )
            return JSONResponse(
                context_compact_error_payload(error),
                status_code=error.status_code,
            )
        if not isinstance(payload, Mapping):
            error = ContextCompactError(
                status_code=400,
                code="invalid_request",
                message="Request body must be a JSON object",
            )
            return JSONResponse(
                context_compact_error_payload(error),
                status_code=error.status_code,
            )
        try:
            summary = await app.state.context_compactor.compact(
                payload,
                request_headers=_build_forward_headers(request),
                current_user=getattr(request.state, "current_user", None),
            )
        except ContextCompactError as error:
            return JSONResponse(
                context_compact_error_payload(error),
                status_code=error.status_code,
            )
        except Exception as exc:
            logger.exception("[context-compact] unexpected failure type=%s", type(exc).__name__)
            error = ContextCompactError(
                status_code=500,
                code="internal_error",
                message="Context compaction failed",
            )
            return JSONResponse(
                context_compact_error_payload(error),
                status_code=error.status_code,
            )
        return JSONResponse({"summary": summary})

    @app.get("/v1/runtime/status")
    async def runtime_status() -> Mapping[str, object]:
        tool_runtime = await app.state.tool_registry.runtime_snapshot(include_runs=True)
        task_runtime = await app.state.task_registry.runtime_snapshot(include_runs=True)
        session_runtime = await app.state.runtime.runtime_snapshot()
        permissions = permission_snapshot(
            settings.claude,
            source="startup_default",
        )
        return {
            "ok": True,
            "service": "claude-sdk-agent",
            "runtime": "claude-sdk-agent",
            "runtimeContract": {
                "schemaVersion": "claude-code.runtime/v2",
                "agentPolicyKey": "agentRuntime",
                "sessionEffectiveKey": "sessionRuntime.sessions[].workspaceRuntime",
                "workspaceInspectEndpoint": "/v1/runtime/workspace/inspect",
            },
            "agentTaskNum": int(tool_runtime.get("agentTaskNum") or 0),
            "workdir": str(settings.claude.workdir),
            "config_dir": str(settings.claude.config_dir),
            "claude_cli": {
                "path": settings.claude.cli_path or "",
                "version": ".".join(str(part) for part in claude_cli_version) if claude_cli_version else "",
            },
            "project_files": {
                "supported": True,
                "root": str(_project_files_root(app_root).resolve()),
                "writable": os.access(_project_files_root(app_root).parent, os.W_OK)
                if _project_files_root(app_root).parent.exists()
                else os.access(app_root, os.W_OK),
            },
            "auth_enabled": settings.auth.enabled,
            "provider_base_url": settings.provider.base_url,
            "skill_mount_root": str(app.state.skill_mount_root),
            "skill_count": len(app.state.skill_names),
            "workflow_mount_root": str(app.state.workflow_mount_root),
            "workflow_count": len(app.state.workflow_names),
            "workflow_names": list(app.state.workflow_names),
            "mcp_config_dir": str(settings.mcp.config_dir),
            "auditLinks": [
                {
                    "name": item.name,
                    "path": item.path,
                    "target": item.target,
                    "status": item.status,
                }
                for item in app.state.audit_links
            ],
            "toolRuntime": tool_runtime,
            "taskRuntime": task_runtime,
            "sessionRuntime": session_runtime,
            "approvalRuntime": await app.state.runtime.approvals_snapshot(),
            "questionRuntime": await app.state.runtime.questions_snapshot(),
            "hookRuntime": await app.state.runtime.hooks_snapshot(),
            "permissionRuntime": permissions,
            "agentRuntime": {
                "scope": "agent",
                "settingSources": settings.claude.setting_sources,
                "effectiveSettingSources": effective_workspace_setting_sources(settings.claude),
                "permission": permissions.get("current", {}),
                "strictMcpConfig": settings.claude.strict_mcp_config,
                "skillsFilter": settings.claude.skills_filter,
                "skillMountRoot": str(app.state.skill_mount_root),
                "workflowMountRoot": str(app.state.workflow_mount_root),
                "mcpConfigDir": str(settings.mcp.config_dir),
            },
            "sdkOptions": {
                "tools": settings.claude.tools,
                "allowed_tools": permissions.get("current", {}).get("allowedTools"),
                "disallowed_tools": permissions.get("current", {}).get("disallowedTools"),
                "permission_mode": permissions.get("current", {}).get("permissionMode"),
                "strict_mcp_config": settings.claude.strict_mcp_config,
                "setting_sources": settings.claude.setting_sources,
                "skills_filter": settings.claude.skills_filter,
                "system_prompt_preset": settings.claude.system_prompt_preset,
                "system_prompt_file": str(settings.claude.system_prompt_file) if settings.claude.system_prompt_file else "",
                "continue_conversation": settings.claude.continue_conversation,
                "max_turns": settings.claude.max_turns,
                "max_budget_usd": settings.claude.max_budget_usd,
                "fallback_model": settings.claude.fallback_model,
                "betas": settings.claude.betas,
                "permission_prompt_tool_name": settings.claude.permission_prompt_tool_name,
                "settings": settings.claude.settings,
                "add_dirs": [str(item) for item in settings.claude.add_dirs or []],
                "extra_args": settings.claude.extra_args,
                "max_buffer_size": settings.claude.max_buffer_size,
                "user": settings.claude.user,
                "include_partial_messages": settings.claude.include_partial_messages,
                "include_hook_events": settings.claude.include_hook_events,
                "fork_session": settings.claude.fork_session,
                "agents": settings.claude.agents,
                "sandbox": settings.claude.sandbox,
                "plugins": settings.claude.plugins,
                "max_thinking_tokens": settings.claude.max_thinking_tokens,
                "thinking": settings.claude.thinking,
                "effort": settings.claude.effort,
                "output_format": settings.claude.output_format,
                "enable_file_checkpointing": settings.claude.enable_file_checkpointing,
                "attachment_text_char_limit": settings.claude.attachment_text_char_limit,
                "workflow_target_dir": str(settings.workflows.target_dir),
                "session_store_flush": settings.claude.session_store_flush,
                "load_timeout_ms": settings.claude.load_timeout_ms,
                "task_budget": settings.claude.task_budget,
            },
            "featureFlags": {
                "auto_interrupt_on_disconnect": settings.features.auto_interrupt_on_disconnect,
                "approval_frontend_enabled": settings.features.approval_frontend_enabled,
                "question_frontend_enabled": settings.features.question_frontend_enabled,
                "hook_frontend_enabled": settings.features.hook_frontend_enabled,
                "checkpoint_rewind_frontend_enabled": settings.features.checkpoint_rewind_frontend_enabled,
                "task_panel_frontend_enabled": settings.features.task_panel_frontend_enabled,
                "subagent_events_frontend_enabled": settings.features.subagent_events_frontend_enabled,
            },
            "mcpOptions": {
                "auto_load": settings.mcp.auto_load,
            },
        }

    @app.post("/v1/runtime/workspace/inspect")
    async def inspect_runtime_workspace(request: Request) -> JSONResponse:
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        if not isinstance(payload, Mapping):
            return JSONResponse(
                {"ok": False, "error": "request body must be a JSON object"},
                status_code=400,
            )
        try:
            snapshot = await app.state.runtime.inspect_workspace(
                payload,
                request_headers=_build_forward_headers(request),
                fallback_tdl_api_key=str(getattr(request.state, "shared_tdl_api_key", "") or ""),
                skill_mount_root=app.state.skill_mount_root,
            )
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        return JSONResponse({"ok": True, "workspaceRuntime": snapshot})

    @app.get("/v1/project-files/status")
    async def project_files_status() -> Mapping[str, Any]:
        project_root = _project_files_root(app_root)
        writable_parent = project_root if project_root.exists() else project_root.parent
        return {
            "ok": True,
            "supported": True,
            "root": str(project_root.resolve()),
            "runtimeBase": str((app_root.parent / "my-agents").resolve()),
            "writable": os.access(writable_parent, os.W_OK),
        }

    @app.post("/v1/project-files")
    async def upload_project_file(
        owner_id: str = Form(...),
        project_id: str = Form(...),
        file_id: str = Form(...),
        file: UploadFile = File(...),
    ) -> Mapping[str, Any]:
        filename = _sanitize_project_filename(file.filename or "")
        content = await _read_project_file_upload(file)
        if not content:
            raise HTTPException(status_code=400, detail="File content is empty")
        target_path = _project_file_path(
            app_root,
            owner_id=owner_id,
            project_id=project_id,
            file_id=file_id,
            filename=filename,
        )
        await _write_project_file(target_path, content)
        resolved_path = target_path.resolve()
        return {
            "ok": True,
            "fileId": _sanitize_project_file_segment(file_id),
            "filename": filename,
            "size": len(content),
            "localPath": str(resolved_path),
            "runtimePath": _runtime_project_file_path(app_root, resolved_path),
            "storageRoot": str(_project_files_root(app_root).resolve()),
        }

    @app.delete("/v1/project-files/{file_id}")
    async def delete_project_file(file_id: str, request: Request) -> Mapping[str, Any]:
        local_path = str(request.query_params.get("localPath") or request.query_params.get("local_path") or "").strip()
        if not local_path:
            raise HTTPException(status_code=400, detail="localPath is required")
        deleted = await _delete_project_file_path(app_root, local_path)
        return {
            "ok": True,
            "fileId": _sanitize_project_file_segment(file_id),
            "deleted": deleted,
        }

    @app.get("/v1/sessions/{frontend_session_id}")
    async def get_session_state(frontend_session_id: str) -> JSONResponse:
        state = await app.state.runtime.get_session_state(frontend_session_id)
        if state is None:
            return JSONResponse({"ok": False, "error": "session not found", "sessionId": frontend_session_id}, status_code=404)
        return JSONResponse({"ok": True, "session": state})

    @app.post("/v1/sessions/{frontend_session_id}/interrupt")
    async def interrupt_session(frontend_session_id: str) -> JSONResponse:
        ok = await app.state.runtime.interrupt_session(frontend_session_id)
        if not ok:
            return JSONResponse({"ok": False, "error": "session not found", "sessionId": frontend_session_id}, status_code=404)
        return JSONResponse({"ok": True, "sessionId": frontend_session_id, "action": "interrupt"})

    @app.get("/v1/sessions/{frontend_session_id}/checkpoints")
    async def list_session_checkpoints(frontend_session_id: str) -> JSONResponse:
        snapshot = await app.state.runtime.checkpoint_snapshot(frontend_session_id)
        return JSONResponse(dict(snapshot))

    @app.get("/v1/sessions/{frontend_session_id}/approvals")
    async def list_session_approvals(frontend_session_id: str) -> JSONResponse:
        return JSONResponse({"ok": True, "approvals": await app.state.runtime.list_approvals(frontend_session_id)})

    @app.get("/v1/sessions/{frontend_session_id}/approvals/stream")
    async def stream_session_approvals(frontend_session_id: str) -> StreamingResponse:
        return StreamingResponse(
            _stream_approvals(app, frontend_session_id=frontend_session_id),
            media_type="text/event-stream",
        )

    @app.get("/v1/sessions/{frontend_session_id}/approvals/{request_id}")
    async def get_session_approval(frontend_session_id: str, request_id: str) -> JSONResponse:
        approval = await app.state.runtime.get_approval(frontend_session_id, request_id)
        if approval is None:
            return JSONResponse(
                {"ok": False, "error": "approval not found", "sessionId": frontend_session_id, "requestId": request_id},
                status_code=404,
            )
        return JSONResponse({"ok": True, "approval": approval})

    @app.post("/v1/sessions/{frontend_session_id}/approvals/{request_id}")
    async def respond_session_approval(frontend_session_id: str, request_id: str, request: Request) -> JSONResponse:
        body = await request.json()
        decision = str(body.get("decision") or "").strip().lower()
        reason = str(body.get("reason") or "").strip()
        if decision not in {"allow", "deny"}:
            return JSONResponse(
                {"ok": False, "error": "decision must be allow or deny", "requestId": request_id},
                status_code=400,
            )
        approval = await app.state.runtime.respond_approval(
            frontend_session_id,
            request_id,
            decision=decision,
            reason=reason,
        )
        if approval is None:
            return JSONResponse(
                {"ok": False, "error": "approval not found", "sessionId": frontend_session_id, "requestId": request_id},
                status_code=404,
            )
        return JSONResponse({"ok": True, "approval": approval})

    @app.get("/v1/sessions/{frontend_session_id}/questions")
    async def list_session_questions(frontend_session_id: str) -> JSONResponse:
        return JSONResponse({"ok": True, "questions": await app.state.runtime.list_questions(frontend_session_id)})

    @app.get("/v1/sessions/{frontend_session_id}/questions/stream")
    async def stream_session_questions(frontend_session_id: str) -> StreamingResponse:
        return StreamingResponse(
            _stream_questions(app, frontend_session_id=frontend_session_id),
            media_type="text/event-stream",
        )

    @app.get("/v1/sessions/{frontend_session_id}/questions/{question_id}")
    async def get_session_question(frontend_session_id: str, question_id: str) -> JSONResponse:
        question = await app.state.runtime.get_question(frontend_session_id, question_id)
        if question is None:
            return JSONResponse(
                {"ok": False, "error": "question not found", "sessionId": frontend_session_id, "questionId": question_id},
                status_code=404,
            )
        return JSONResponse({"ok": True, "question": question})

    @app.post("/v1/sessions/{frontend_session_id}/questions/{question_id}")
    async def answer_session_question(frontend_session_id: str, question_id: str, request: Request) -> JSONResponse:
        body = await request.json()
        answer = str(body.get("answer") or "")
        question = await app.state.runtime.answer_question(frontend_session_id, question_id, answer=answer)
        if question is None:
            return JSONResponse(
                {"ok": False, "error": "question not found", "sessionId": frontend_session_id, "questionId": question_id},
                status_code=404,
            )
        return JSONResponse({"ok": True, "question": question})

    @app.get("/v1/sessions/{frontend_session_id}/hooks")
    async def list_session_hooks(frontend_session_id: str) -> JSONResponse:
        return JSONResponse({"ok": True, "hooks": await app.state.runtime.list_hooks(frontend_session_id)})

    @app.get("/v1/sessions/{frontend_session_id}/hooks/stream")
    async def stream_session_hooks(frontend_session_id: str) -> StreamingResponse:
        return StreamingResponse(
            _stream_hooks(app, frontend_session_id=frontend_session_id),
            media_type="text/event-stream",
        )

    @app.get("/v1/sessions/{frontend_session_id}/hooks/{event_id}")
    async def get_session_hook(frontend_session_id: str, event_id: str) -> JSONResponse:
        hook = await app.state.runtime.get_hook(frontend_session_id, event_id)
        if hook is None:
            return JSONResponse(
                {"ok": False, "error": "hook not found", "sessionId": frontend_session_id, "eventId": event_id},
                status_code=404,
            )
        return JSONResponse({"ok": True, "hook": hook})

    @app.post("/v1/sessions/{frontend_session_id}/checkpoints/{checkpoint_id}/rewind")
    async def rewind_session(frontend_session_id: str, checkpoint_id: str, request: Request) -> JSONResponse:
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        runtime_config = payload.get("runtime_config") or payload.get("runtimeConfig")
        result = await app.state.runtime.rewind_checkpoint(
            frontend_session_id,
            checkpoint_id,
            request_headers=_build_forward_headers(request),
            current_user=getattr(request.state, "current_user", None),
            fallback_tdl_api_key=str(getattr(request.state, "shared_tdl_api_key", "") or ""),
            proxy_base_url=_proxy_base_url(request),
            skill_mount_root=app.state.skill_mount_root,
            runtime_config=runtime_config if isinstance(runtime_config, Mapping) else None,
        )
        if result.get("ok"):
            return JSONResponse(dict(result))
        status = str(result.get("status") or "").strip()
        if status == "busy":
            return JSONResponse(dict(result), status_code=409)
        if status == "disabled":
            return JSONResponse(dict(result), status_code=400)
        if status == "session_reconnect_failed":
            return JSONResponse(dict(result), status_code=409)
        if status in {"session_not_found", "checkpoint_not_found"}:
            return JSONResponse(
                dict(result),
                status_code=404,
            )
        return JSONResponse(dict(result), status_code=500)

    @app.get("/v1/tools/{run_id}/{tool_call_id}")
    async def get_tool_status(run_id: str, tool_call_id: str) -> JSONResponse:
        tool = await app.state.tool_registry.get_tool(run_id=run_id, tool_call_id=tool_call_id)
        if tool is None:
            return JSONResponse(
                {"ok": False, "error": "tool not found", "runId": run_id, "toolCallId": tool_call_id},
                status_code=404,
            )
        return JSONResponse({"ok": True, "tool": tool})

    @app.get("/v1/tools/{run_id}/{tool_call_id}/status/stream")
    async def stream_tool_status(run_id: str, tool_call_id: str) -> StreamingResponse:
        return StreamingResponse(
            _stream_tool_status(app, run_id=run_id, tool_call_id=tool_call_id),
            media_type="text/event-stream",
        )

    @app.get("/v1/tools/{run_id}/{tool_call_id}/output/stream")
    async def stream_tool_output(run_id: str, tool_call_id: str) -> StreamingResponse:
        return StreamingResponse(
            _stream_tool_output(app, run_id=run_id, tool_call_id=tool_call_id),
            media_type="text/event-stream",
        )

    @app.get("/v1/tasks/{run_id}")
    async def list_run_tasks(run_id: str) -> JSONResponse:
        return JSONResponse({"ok": True, "tasks": await app.state.task_registry.list_run_tasks(run_id)})

    @app.get("/v1/tasks/{run_id}/{task_id}")
    async def get_task_status(run_id: str, task_id: str) -> JSONResponse:
        task = await app.state.task_registry.get_task(run_id=run_id, task_id=task_id)
        if task is None:
            return JSONResponse(
                {"ok": False, "error": "task not found", "runId": run_id, "taskId": task_id},
                status_code=404,
            )
        return JSONResponse({"ok": True, "task": task})

    @app.get("/v1/tasks/{run_id}/{task_id}/status/stream")
    async def stream_task_status(run_id: str, task_id: str) -> StreamingResponse:
        return StreamingResponse(
            _stream_task_status(app, run_id=run_id, task_id=task_id),
            media_type="text/event-stream",
        )

    @app.get("/v1/tasks/{run_id}/{task_id}/output/stream")
    async def stream_task_output(run_id: str, task_id: str) -> StreamingResponse:
        return StreamingResponse(
            _stream_task_output(app, run_id=run_id, task_id=task_id),
            media_type="text/event-stream",
        )

    @app.get("/v1/runs/{run_id}/artifacts")
    async def run_artifacts(request: Request, run_id: str) -> JSONResponse:
        record = await request.app.state.artifact_service.get_run(run_id)
        if record is None:
            return JSONResponse(
                {"ok": False, "error": "artifacts not found", "runId": run_id},
                status_code=404,
            )
        return JSONResponse({"ok": True, "run": record})

    @app.get("/v1/artifacts/by-session")
    async def session_artifacts(request: Request) -> JSONResponse:
        session_id = str(
            request.query_params.get("sessionId")
            or request.query_params.get("session_id")
            or ""
        ).strip()
        if not session_id:
            return JSONResponse({"ok": False, "error": "sessionId is required"}, status_code=400)
        limit_raw = request.query_params.get("limit")
        try:
            limit = int(limit_raw) if limit_raw is not None else 50
        except ValueError:
            limit = 50
        payload = await request.app.state.artifact_service.list_session_artifacts(session_id, limit=limit)
        return JSONResponse({"ok": True, "session": payload})

    @app.get("/v1/artifacts/{artifact_id}/metadata")
    async def artifact_metadata(request: Request, artifact_id: str) -> JSONResponse:
        artifact = await request.app.state.artifact_service.get_artifact(artifact_id)
        if artifact is None:
            return JSONResponse(
                {"ok": False, "error": "artifact not found", "artifactId": artifact_id},
                status_code=404,
            )
        return JSONResponse({"ok": True, "artifact": artifact})

    @app.post("/v1/artifacts/{artifact_id}/open")
    async def artifact_open(request: Request, artifact_id: str) -> JSONResponse:
        artifact = await request.app.state.artifact_service.get_artifact(artifact_id)
        if artifact is None:
            return JSONResponse(
                {"ok": False, "error": "artifact not found", "artifactId": artifact_id},
                status_code=404,
            )
        path = await request.app.state.artifact_service.resolve_artifact_file(artifact_id)
        if path is None:
            return JSONResponse(
                {"ok": False, "error": "artifact file not available", "artifactId": artifact_id},
                status_code=404,
            )
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        requested_path = ""
        if isinstance(payload, Mapping):
            requested_path = str(payload.get("path") or "").strip()
        if requested_path:
            try:
                requested = Path(requested_path).expanduser().resolve()
            except OSError:
                requested = Path()
            if requested != path:
                return JSONResponse(
                    {"ok": False, "error": "artifact path mismatch", "artifactId": artifact_id},
                    status_code=409,
                )
        try:
            await asyncio.to_thread(open_file_with_default_app, path)
        except FileNotFoundError:
            return JSONResponse(
                {"ok": False, "error": "artifact file not available", "artifactId": artifact_id},
                status_code=404,
            )
        except ArtifactOpenError as exc:
            return JSONResponse(
                {"ok": False, "error": str(exc), "artifactId": artifact_id},
                status_code=501,
            )
        return JSONResponse({"ok": True, "artifactId": artifact_id, "artifact": artifact, "path": path.as_posix()})

    @app.get("/v1/artifacts/{artifact_id}/download", response_model=None)
    async def artifact_download(request: Request, artifact_id: str) -> FileResponse | JSONResponse:
        artifact = await request.app.state.artifact_service.get_artifact(artifact_id)
        if artifact is None:
            return JSONResponse(
                {"ok": False, "error": "artifact not found", "artifactId": artifact_id},
                status_code=404,
            )
        path = await request.app.state.artifact_service.resolve_artifact_file(artifact_id)
        if path is None:
            return JSONResponse(
                {"ok": False, "error": "artifact file not available", "artifactId": artifact_id},
                status_code=404,
            )
        return FileResponse(path, filename=path.name)

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(request: Request) -> JSONResponse | StreamingResponse:
        payload = await request.json()
        normalize_session_id_in_payload(payload)
        stream_flag = payload.get("stream") is not False
        payload["stream"] = stream_flag
        effective_model = resolve_effective_model(payload.get("model"), settings.claude.default_model)
        payload["model"] = effective_model
        session_id = str(payload.get("session_id") or "").strip()
        run_id = f"run-{uuid.uuid4().hex}"
        payload["_run_id"] = run_id
        await app.state.tool_registry.register_run(run_id, session_id=session_id or None)
        await app.state.task_registry.register_run(run_id)
        logger.info(
            "[chat] request stream=%s model=%s session_id=%s run_id=%s messages=%s",
            stream_flag,
            effective_model or "-",
            session_id or "-",
            run_id,
            _message_count(payload),
        )
        logger.info("[chat] payload_sanitized %s", _format_chat_payload_for_log(payload))
        if stream_flag:
            return StreamingResponse(
                _stream_chat_response(
                    app=app,
                    runtime=app.state.runtime,
                    payload=payload,
                    model_name=effective_model,
                    request_headers=_build_forward_headers(request),
                    current_user=getattr(request.state, "current_user", None),
                    fallback_tdl_api_key=str(getattr(request.state, "shared_tdl_api_key", "") or ""),
                    proxy_base_url=_proxy_base_url(request),
                    request_base_url=str(request.base_url).rstrip("/"),
                    skill_mount_root=app.state.skill_mount_root,
                    run_id=run_id,
                ),
                media_type="text/event-stream",
            )
        try:
            content, artifact_record = await _collect_text_and_artifacts(
                app.state.runtime.stream_events(
                    payload,
                    request_headers=_build_forward_headers(request),
                    current_user=getattr(request.state, "current_user", None),
                    fallback_tdl_api_key=str(getattr(request.state, "shared_tdl_api_key", "") or ""),
                    proxy_base_url=_proxy_base_url(request),
                    skill_mount_root=app.state.skill_mount_root,
                    run_id=run_id,
                )
            )
        except Exception as exc:
            await app.state.tool_registry.finish_run(run_id, status="failed")
            logger.exception("[chat] non-stream request failed: %s", exc)
            return JSONResponse(
                {
                    "error": {
                        "message": "runtime execution failed",
                        "type": type(exc).__name__,
                        "detail": str(exc),
                    }
                },
                status_code=502,
            )
        await app.state.tool_registry.finish_run(run_id, status="completed")
        logger.info("[chat] non-stream completed model=%s session_id=%s chars=%s", effective_model or "-", session_id or "-", len(content))
        return JSONResponse(
            completion_payload(chat_id=new_chat_id(), model=effective_model, content=content),
            headers=dict(_artifact_response_headers(artifact_record)),
        )

    return app


async def _collect_text(stream: AsyncIterator[RuntimeStreamEvent]) -> str:
    text, _ = await _collect_text_and_artifacts(stream)
    return text


async def _collect_text_and_artifacts(stream: AsyncIterator[RuntimeStreamEvent]) -> tuple[str, Mapping[str, Any] | None]:
    parts: list[str] = []
    terminal_task_issues: list[dict[str, str]] = []
    artifact_record: Mapping[str, Any] | None = None
    async for event in stream:
        if event.kind == "text" and event.text:
            parts.append(event.text)
        elif event.kind == "task" and isinstance(event.payload, Mapping):
            issue = _task_terminal_issue(event.payload)
            if issue is not None:
                terminal_task_issues.append(issue)
        elif event.kind == "artifacts" and isinstance(event.payload, Mapping):
            artifact_record = dict(event.payload)
    if not parts and terminal_task_issues:
        parts.append(_empty_response_task_issue_message(terminal_task_issues))
    return "".join(parts), artifact_record


def _build_forward_headers(request: Request) -> Mapping[str, str]:
    blocked = {"host", "content-length", "transfer-encoding", "connection", "keep-alive", "upgrade", "proxy-connection"}
    out: dict[str, str] = {}
    for key, value in request.headers.items():
        name = str(key or "").strip()
        if name and name.lower() not in blocked:
            out[name] = str(value)
    return out


def _proxy_base_url(request: Request) -> str:
    return str(request.base_url).rstrip("/") + "/internal/anthropic"


def _message_count(payload: Mapping[str, Any]) -> int:
    messages = payload.get("messages")
    return len(messages) if isinstance(messages, list) else 0


def _format_chat_payload_for_log(payload: Mapping[str, Any]) -> str:
    sanitized = _sanitize_value_for_log(payload)
    return json.dumps(sanitized, ensure_ascii=False, sort_keys=True)


def _sanitize_value_for_log(value: Any, *, key: str = "") -> Any:
    if isinstance(value, Mapping):
        return {str(item_key): _sanitize_value_for_log(item_value, key=str(item_key)) for item_key, item_value in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_sanitize_value_for_log(item, key=key) for item in value]
    if isinstance(value, (bytes, bytearray)):
        return f"<binary:{len(value)} bytes>"
    if isinstance(value, str):
        return _sanitize_string_for_log(value, key=key)
    return value


def _sanitize_string_for_log(value: str, *, key: str = "") -> str:
    text = str(value or "")
    lower_key = str(key or "").strip().lower()
    if _is_sensitive_log_key(lower_key):
        return _mask_secret(text)
    stripped = text.strip()
    if stripped.startswith("data:") and ";base64," in stripped:
        prefix, encoded = stripped.split(",", 1)
        return f"{prefix},<base64:{len(encoded)} chars>"
    if _looks_like_base64_blob(stripped):
        return f"<base64:{len(stripped)} chars>"
    if len(text) > 400:
        return f"{text[:240]}...<truncated:{len(text) - 280} chars>...{text[-40:]}"
    return text


def _is_sensitive_log_key(key: str) -> bool:
    return key in {
        "api-key",
        "api_key",
        "authorization",
        "proxy-authorization",
        "token",
        "access_token",
        "refresh_token",
        "uac-user-token",
        "x-api-key",
        "x-uac-user-token",
    } or any(marker in key for marker in ("token", "secret", "password"))


def _mask_secret(value: str) -> str:
    text = str(value or "")
    if not text:
        return ""
    if len(text) <= 8:
        return "<redacted>"
    return f"{text[:4]}...{text[-4:]}<redacted>"


def _looks_like_base64_blob(value: str) -> bool:
    if len(value) < 256 or " " in value or "\t" in value:
        return False
    alphabet = set(string.ascii_letters + string.digits + "+/=_-\n\r")
    if any(char not in alphabet for char in value):
        return False
    return any(char in value for char in ("=", "+", "/", "_", "-"))


_TASK_TERMINAL_FAILURE_STATUSES = {"failed", "stopped", "killed", "cancelled", "canceled"}
_TASK_MISSING_COMPLETION_MARKER = "No completion record was found"


def _task_terminal_issue(payload: Mapping[str, Any]) -> dict[str, str] | None:
    status = str(payload.get("status") or "").strip().lower()
    log = str(payload.get("log") or payload.get("result") or "").strip()
    if status not in _TASK_TERMINAL_FAILURE_STATUSES and _TASK_MISSING_COMPLETION_MARKER not in log:
        return None

    return {
        "name": str(payload.get("name") or payload.get("taskType") or payload.get("taskId") or "后台子任务").strip(),
        "status": status or "unknown",
        "log": log,
    }


def _empty_response_task_issue_message(issues: Sequence[Mapping[str, str]]) -> str:
    lines = ["未能生成回复：后台子任务没有返回可用的完成记录。", ""]
    for issue in issues[:6]:
        name = str(issue.get("name") or "后台子任务").strip()
        status = str(issue.get("status") or "unknown").strip()
        log = str(issue.get("log") or "").strip()
        if log:
            lines.append(f"- {name}：{status}，{log}")
        else:
            lines.append(f"- {name}：{status}")
    if len(issues) > 6:
        lines.append(f"- 另有 {len(issues) - 6} 个子任务没有返回可用完成记录。")
    lines.extend(["", "请重新发起该请求；如果仍复现，需要检查 sidecar 进程是否中途重启或 Claude 后台任务记录是否丢失。"])
    return "\n".join(lines)


async def _stream_chat_response(
    *,
    app: FastAPI,
    runtime: ClaudeSdkRuntime,
    payload: Mapping[str, Any],
    model_name: str,
    request_headers: Mapping[str, str],
    current_user: Any | None,
    fallback_tdl_api_key: str,
    proxy_base_url: str,
    skill_mount_root: Path,
    run_id: str,
    request_base_url: str = "",
) -> AsyncIterator[bytes]:
    chat_id = new_chat_id()
    session_id = str(payload.get("session_id") or "").strip()
    yield stream_chunk(chat_id=chat_id, model=model_name, role="assistant")
    artifact_base_url = request_base_url or _public_base_url_from_proxy(proxy_base_url)
    structured_shells_enabled = any(
        (
            app.state.settings.features.task_panel_frontend_enabled,
            app.state.settings.features.approval_frontend_enabled,
            app.state.settings.features.question_frontend_enabled,
        )
    )
    if run_id and structured_shells_enabled:
        yield protocol_shell_chunk(
            chat_id=chat_id,
            model=model_name,
            tag="meta",
            payload={"runId": run_id, "sessionId": session_id or "", "model": model_name},
        )
    stream_queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
    producer_tasks: list[asyncio.Task[None]] = []
    if session_id and app.state.settings.features.approval_frontend_enabled:
        producer_tasks.append(asyncio.create_task(
            _forward_session_events(
                runtime.stream_approvals(session_id),
                stream_queue,
                kind="approval",
            )
        ))
    if session_id and app.state.settings.features.question_frontend_enabled:
        producer_tasks.append(asyncio.create_task(
            _forward_session_events(
                runtime.stream_questions(session_id),
                stream_queue,
                kind="question",
            )
        ))
    if session_id and app.state.settings.features.hook_frontend_enabled:
        producer_tasks.append(asyncio.create_task(
            _forward_session_events(
                runtime.stream_hooks(session_id),
                stream_queue,
                kind="hook",
            )
        ))
    producer_tasks.append(asyncio.create_task(_forward_runtime_events(
        runtime.stream_events(
            payload,
            request_headers=request_headers,
            current_user=current_user,
            fallback_tdl_api_key=fallback_tdl_api_key,
            proxy_base_url=proxy_base_url,
            skill_mount_root=skill_mount_root,
            run_id=run_id,
        ),
        stream_queue,
    )))
    try:
        chunk_count = 0
        assistant_text_parts: list[str] = []
        terminal_task_issues: list[dict[str, str]] = []
        runtime_done = False
        while not runtime_done:
            item_kind, item_payload = await stream_queue.get()
            if item_kind == "session_event":
                event_kind, payload_item = item_payload
                yield _session_protocol_shell_chunk(
                    kind=str(event_kind),
                    payload=payload_item,
                    chat_id=chat_id,
                    model_name=model_name,
                )
                continue
            if item_kind == "runtime_done":
                runtime_done = True
                continue
            if item_kind == "runtime_cancelled":
                raise asyncio.CancelledError()
            if item_kind == "runtime_error":
                raise item_payload
            if item_kind != "runtime_event":
                continue

            event = item_payload
            if event.kind == "text" and event.text:
                chunk_count += 1
                assistant_text_parts.append(event.text)
                yield stream_chunk(chat_id=chat_id, model=model_name, content=event.text)
            elif event.kind == "tool" and isinstance(event.payload, Mapping):
                yield tool_shell_chunk(chat_id=chat_id, model=model_name, payload=dict(event.payload))
            elif event.kind == "task" and isinstance(event.payload, Mapping) and app.state.settings.features.task_panel_frontend_enabled:
                task_payload = dict(event.payload)
                issue = _task_terminal_issue(task_payload)
                if issue is not None:
                    terminal_task_issues.append(issue)
                yield protocol_shell_chunk(chat_id=chat_id, model=model_name, tag="task", payload=task_payload)
            elif event.kind == "command" and isinstance(event.payload, Mapping):
                yield protocol_shell_chunk(chat_id=chat_id, model=model_name, tag="command", payload=dict(event.payload))
            elif event.kind == "artifacts" and isinstance(event.payload, Mapping):
                yield _artifact_protocol_chunk(
                    event.payload,
                    chat_id=chat_id,
                    model_name=model_name,
                    request_base_url=artifact_base_url,
                )
        await asyncio.sleep(0)
        while True:
            try:
                item_kind, item_payload = stream_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item_kind == "session_event":
                event_kind, payload_item = item_payload
                yield _session_protocol_shell_chunk(
                    kind=str(event_kind),
                    payload=payload_item,
                    chat_id=chat_id,
                    model_name=model_name,
                )
            elif item_kind == "runtime_cancelled":
                raise asyncio.CancelledError()
            elif item_kind == "runtime_error":
                raise item_payload
            elif item_kind == "runtime_event":
                event = item_payload
                if event.kind == "artifacts" and isinstance(event.payload, Mapping):
                    yield _artifact_protocol_chunk(
                        event.payload,
                        chat_id=chat_id,
                        model_name=model_name,
                        request_base_url=artifact_base_url,
                    )
        if chunk_count == 0 and terminal_task_issues:
            fallback_content = _empty_response_task_issue_message(terminal_task_issues)
            chunk_count += 1
            assistant_text_parts.append(fallback_content)
            yield stream_chunk(chat_id=chat_id, model=model_name, content=fallback_content)
        if structured_shells_enabled:
            final_text = "".join(assistant_text_parts)
            yield protocol_shell_chunk(
                chat_id=chat_id,
                model=model_name,
                tag="run_result",
                payload={
                    "runId": run_id,
                    "sessionId": session_id or "",
                    "status": "completed",
                    "phase": "end",
                    "final": True,
                    "textChars": len(final_text),
                    "finalText": final_text,
                },
            )
        await app.state.tool_registry.finish_run(run_id, status="completed")
        logger.info("[chat] stream completed model=%s session_id=%s run_id=%s chunks=%s", model_name or "-", session_id or "-", run_id, chunk_count)
        yield stream_chunk(chat_id=chat_id, model=model_name, finish_reason="stop")
        yield done_chunk()
    except asyncio.CancelledError:
        await app.state.tool_registry.finish_run(run_id, status="cancelled")
        if session_id and app.state.settings.features.auto_interrupt_on_disconnect:
            try:
                interrupted = await runtime.interrupt_session(session_id)
                logger.info(
                    "[chat] stream cancelled by client model=%s session_id=%s run_id=%s interrupted=%s",
                    model_name or "-",
                    session_id,
                    run_id,
                    interrupted,
                )
            except Exception as exc:
                logger.warning(
                    "[chat] client disconnect interrupt failed model=%s session_id=%s run_id=%s err=%s",
                    model_name or "-",
                    session_id,
                    run_id,
                    exc,
                )
        raise
    except Exception as exc:
        await app.state.tool_registry.finish_run(run_id, status="failed")
        logger.exception("[chat] stream failed: %s", exc)
        message = f"runtime execution failed: {type(exc).__name__}: {exc}"
        yield stream_chunk(chat_id=chat_id, model=model_name, content=message)
        yield stream_chunk(chat_id=chat_id, model=model_name, finish_reason="stop")
        yield done_chunk()
    finally:
        for task in producer_tasks:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


def _tool_output_chunk_sse(chunk: ToolOutputChunk) -> bytes:
    payload = {
        "sequence": chunk.sequence,
        "stream": chunk.stream,
        "text": chunk.text,
        "timestamp": chunk.timestamp,
    }
    lines = [
        f"event: {chunk.stream}",
        f"data: {json.dumps(payload, ensure_ascii=False)}",
        "",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def _tool_status_sse(payload: Mapping[str, Any]) -> bytes:
    lines = ["event: status", f"data: {json.dumps(payload, ensure_ascii=False)}", "", ""]
    return "\n".join(lines).encode("utf-8")


async def _tool_stream_end_sse(app: FastAPI, *, run_id: str, tool_call_id: str) -> bytes:
    tool = await app.state.tool_registry.get_tool(run_id=run_id, tool_call_id=tool_call_id)
    status = "unknown"
    finished_at = None
    if isinstance(tool, Mapping):
        status = str(tool.get("status") or "unknown")
        finished_at = tool.get("finishedAt")
    payload = {
        "runId": run_id,
        "toolCallId": tool_call_id,
        "status": status,
        "streamStatus": "closed",
    }
    if finished_at is not None:
        payload["finishedAt"] = finished_at
    lines = ["event: end", f"data: {json.dumps(payload, ensure_ascii=False)}", "", ""]
    return "\n".join(lines).encode("utf-8")


async def _stream_tool_output(app: FastAPI, *, run_id: str, tool_call_id: str) -> AsyncIterator[bytes]:
    try:
        async for chunk in app.state.tool_registry.stream_tool_output(run_id=run_id, tool_call_id=tool_call_id):
            yield _tool_output_chunk_sse(chunk)
    except KeyError:
        payload = {"error": "tool not found", "runId": run_id, "toolCallId": tool_call_id}
        lines = ["event: error", f"data: {json.dumps(payload, ensure_ascii=False)}", "", ""]
        yield "\n".join(lines).encode("utf-8")
        return
    yield await _tool_stream_end_sse(app, run_id=run_id, tool_call_id=tool_call_id)


async def _stream_tool_status(app: FastAPI, *, run_id: str, tool_call_id: str) -> AsyncIterator[bytes]:
    try:
        async for payload in app.state.tool_registry.stream_tool_status(run_id=run_id, tool_call_id=tool_call_id):
            yield _tool_status_sse(payload)
    except KeyError:
        payload = {"error": "tool not found", "runId": run_id, "toolCallId": tool_call_id}
        lines = ["event: error", f"data: {json.dumps(payload, ensure_ascii=False)}", "", ""]
        yield "\n".join(lines).encode("utf-8")
        return
    yield await _tool_stream_end_sse(app, run_id=run_id, tool_call_id=tool_call_id)


def _task_output_chunk_sse(chunk: TaskOutputChunk) -> bytes:
    payload = {
        "sequence": chunk.sequence,
        "stream": chunk.stream,
        "text": chunk.text,
        "timestamp": chunk.timestamp,
    }
    lines = [
        f"event: {chunk.stream}",
        f"data: {json.dumps(payload, ensure_ascii=False)}",
        "",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def _task_status_sse(payload: Mapping[str, Any]) -> bytes:
    lines = ["event: status", f"data: {json.dumps(payload, ensure_ascii=False)}", "", ""]
    return "\n".join(lines).encode("utf-8")


async def _task_stream_end_sse(app: FastAPI, *, run_id: str, task_id: str) -> bytes:
    task = await app.state.task_registry.get_task(run_id=run_id, task_id=task_id)
    status = "unknown"
    finished_at = None
    if isinstance(task, Mapping):
        status = str(task.get("status") or "unknown")
        finished_at = task.get("finishedAt")
    payload = {
        "runId": run_id,
        "taskId": task_id,
        "status": status,
        "streamStatus": "closed",
    }
    if finished_at is not None:
        payload["finishedAt"] = finished_at
    lines = ["event: end", f"data: {json.dumps(payload, ensure_ascii=False)}", "", ""]
    return "\n".join(lines).encode("utf-8")


async def _stream_task_output(app: FastAPI, *, run_id: str, task_id: str) -> AsyncIterator[bytes]:
    try:
        async for chunk in app.state.task_registry.stream_task_output(run_id=run_id, task_id=task_id):
            yield _task_output_chunk_sse(chunk)
    except KeyError:
        payload = {"error": "task not found", "runId": run_id, "taskId": task_id}
        lines = ["event: error", f"data: {json.dumps(payload, ensure_ascii=False)}", "", ""]
        yield "\n".join(lines).encode("utf-8")
        return
    yield await _task_stream_end_sse(app, run_id=run_id, task_id=task_id)


def _session_event_sse(*, event: str, payload: Mapping[str, Any]) -> bytes:
    lines = [f"event: {event}", f"data: {json.dumps(payload, ensure_ascii=False)}", "", ""]
    return "\n".join(lines).encode("utf-8")


async def _stream_approvals(app: FastAPI, *, frontend_session_id: str) -> AsyncIterator[bytes]:
    async for payload in app.state.runtime.stream_approvals(frontend_session_id):
        yield _session_event_sse(event="approval", payload=payload)
    yield _session_event_sse(
        event="end",
        payload={"sessionId": frontend_session_id, "streamType": "approvals", "streamStatus": "closed"},
    )


async def _stream_questions(app: FastAPI, *, frontend_session_id: str) -> AsyncIterator[bytes]:
    async for payload in app.state.runtime.stream_questions(frontend_session_id):
        yield _session_event_sse(event="question", payload=payload)
    yield _session_event_sse(
        event="end",
        payload={"sessionId": frontend_session_id, "streamType": "questions", "streamStatus": "closed"},
    )


async def _stream_hooks(app: FastAPI, *, frontend_session_id: str) -> AsyncIterator[bytes]:
    async for payload in app.state.runtime.stream_hooks(frontend_session_id):
        yield _session_event_sse(event="hook", payload=payload)
    yield _session_event_sse(
        event="end",
        payload={"sessionId": frontend_session_id, "streamType": "hooks", "streamStatus": "closed"},
    )


async def _forward_session_events(
    source: AsyncIterator[Mapping[str, Any]],
    event_queue: asyncio.Queue[tuple[str, Any]],
    *,
    kind: str,
) -> None:
    try:
        async for payload in source:
            await event_queue.put(("session_event", (kind, payload)))
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("[chat] failed to forward %s session events", kind)


async def _forward_runtime_events(
    source: AsyncIterator[RuntimeStreamEvent],
    event_queue: asyncio.Queue[tuple[str, Any]],
) -> None:
    try:
        async for event in source:
            await event_queue.put(("runtime_event", event))
    except asyncio.CancelledError as exc:
        await event_queue.put(("runtime_cancelled", exc))
    except Exception as exc:
        await event_queue.put(("runtime_error", exc))
    else:
        await event_queue.put(("runtime_done", None))


def _session_protocol_shell_chunk(
    *,
    kind: str,
    payload: Mapping[str, Any],
    chat_id: str,
    model_name: str,
) -> bytes:
    encoded_payload = hook_shell_payload(payload) if kind == "hook" else dict(payload)
    return protocol_shell_chunk(
        chat_id=chat_id,
        model=model_name,
        tag=kind,
        payload=encoded_payload,
    )


def _artifact_protocol_chunk(
    record: Mapping[str, Any],
    *,
    chat_id: str,
    model_name: str,
    request_base_url: str,
) -> bytes:
    return protocol_shell_chunk(
        chat_id=chat_id,
        model=model_name,
        tag="artifacts",
        payload=_artifact_notification_payload(record, request_base_url=request_base_url),
    )


def _artifact_notification_payload(record: Mapping[str, Any], *, request_base_url: str) -> dict[str, Any]:
    run_id = str(record.get("runId") or "")
    summary = record.get("summary")
    summary_payload = dict(summary) if isinstance(summary, Mapping) else {}
    return {
        "type": "artifacts.updated",
        "sessionId": str(record.get("sessionId") or ""),
        "runId": run_id,
        "artifactCount": int(summary_payload.get("artifactCount") or 0),
        "summary": summary_payload,
        "url": f"{request_base_url}/v1/runs/{run_id}/artifacts" if request_base_url and run_id else "",
    }


def _artifact_response_headers(record: Mapping[str, Any] | None) -> Mapping[str, str]:
    if not isinstance(record, Mapping):
        return {}
    summary = record.get("summary")
    artifact_count = 0
    if isinstance(summary, Mapping):
        artifact_count = int(summary.get("artifactCount") or 0)
    return {
        "x-agent-run-id": str(record.get("runId") or ""),
        "x-artifacts-count": str(artifact_count),
    }


def _public_base_url_from_proxy(proxy_base_url: str) -> str:
    text = str(proxy_base_url or "").rstrip("/")
    suffix = "/internal/anthropic"
    if text.endswith(suffix):
        return text[: -len(suffix)]
    return text


async def _stream_task_status(app: FastAPI, *, run_id: str, task_id: str) -> AsyncIterator[bytes]:
    try:
        async for payload in app.state.task_registry.stream_task_status(run_id=run_id, task_id=task_id):
            yield _task_status_sse(payload)
    except KeyError:
        payload = {"error": "task not found", "runId": run_id, "taskId": task_id}
        lines = ["event: error", f"data: {json.dumps(payload, ensure_ascii=False)}", "", ""]
        yield "\n".join(lines).encode("utf-8")
        return
    yield await _task_stream_end_sse(app, run_id=run_id, task_id=task_id)
