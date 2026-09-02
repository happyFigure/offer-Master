from __future__ import annotations

import json
import tempfile
import unittest
import asyncio
from pathlib import Path
from typing import Any, AsyncIterator, Mapping

from fastapi.testclient import TestClient

from src.api import server as server_module
from src.api.server import _format_chat_payload_for_log, _read_project_file_upload, _stream_chat_response, create_app
from src.auth.uac_authz_middleware import _should_bypass_auth
from src.stream_types import RuntimeStreamEvent


def _write_service_json(
    root: Path,
    *,
    auth_enabled: bool,
    auto_interrupt_on_disconnect: bool = True,
    approval_frontend_enabled: bool = False,
    question_frontend_enabled: bool = False,
    hook_frontend_enabled: bool = False,
    task_panel_frontend_enabled: bool = False,
) -> None:
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "data" / "sessions").mkdir(parents=True, exist_ok=True)
    (root / "shared").mkdir(parents=True, exist_ok=True)
    payload = {
        "server": {"host": "127.0.0.1", "port": 18008},
        "claude": {
            "workdir": ".",
            "config_dir": ".",
            "default_model": "MiniMax-RAN3",
            "permission_mode": "acceptEdits",
            "setting_sources": ["project", "local"],
            "skills_filter": ["demo-skill"],
            "system_prompt_preset": "claude_code",
            "include_hook_events": True,
            "enable_file_checkpointing": True,
            "attachment_text_char_limit": 123456,
        },
        "provider": {
            "base_url": "http://127.0.0.1:9999",
            "anthropic_version": "2023-06-01",
            "request_timeout_sec": 30.0,
        },
        "auth": {
            "enabled": auth_enabled,
            "uac_auth_url": "http://127.0.0.1:9998/auth",
            "allow_users_path": "data/runtime/allow_users.json",
            "shared_tdl_api_key_path": "shared/allow_users.json",
        },
        "sessions": {"mapping_path": "data/sessions/session-map.json"},
        "workflows": {"source_dirs": ["shared/workflows"], "target_dir": ".claude/workflows"},
        "mcp": {"config_dir": "shared/mcps", "auto_load": True},
        "features": {
            "auto_interrupt_on_disconnect": auto_interrupt_on_disconnect,
            "approval_frontend_enabled": approval_frontend_enabled,
            "question_frontend_enabled": question_frontend_enabled,
            "hook_frontend_enabled": hook_frontend_enabled,
            "checkpoint_rewind_frontend_enabled": False,
            "task_panel_frontend_enabled": task_panel_frontend_enabled,
            "subagent_events_frontend_enabled": False,
        },
    }
    (root / "config" / "service.json").write_text(json.dumps(payload), encoding="utf-8")
    (root / "shared" / "allow_users.json").write_text(
        json.dumps({"allow_users": ["10154402"], "x-api-key": "tdl_shared_key"}),
        encoding="utf-8",
    )
    (root / "shared" / "mcps").mkdir(parents=True, exist_ok=True)
    (root / "shared" / "workflows").mkdir(parents=True, exist_ok=True)
    (root / "shared" / "workflows" / "demo.js").write_text("export const meta = {}", encoding="utf-8")


class _ChunkedUpload:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)

    async def read(self, _size: int = -1) -> bytes:
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


class _FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.interrupts: list[str] = []
        self.approvals = [
            {
                "sessionId": "session-1",
                "runId": "run-1",
                "requestId": "req-1",
                "status": "pending",
                "toolName": "Bash",
            }
        ]
        self.questions = [
            {
                "sessionId": "session-1",
                "runId": "run-1",
                "questionId": "question-1",
                "requestId": "request-1",
                "status": "pending",
                "prompt": "Need more detail?",
            }
        ]
        self.hooks = [
            {
                "sessionId": "session-1",
                "runId": "run-1",
                "eventId": "hook-1",
                "claudeSessionId": "claude-1",
                "hookEventName": "Notification",
                "phase": "hook_response",
                "source": "sdk_stream",
                "status": "completed",
                "title": "notice",
                "data": {"message": "hello"},
                "output": {},
            }
        ]

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
        self.calls.append(
            {
                "payload": dict(payload),
                "request_headers": dict(request_headers),
                "proxy_base_url": proxy_base_url,
                "skill_mount_root": str(skill_mount_root),
                "current_user": current_user,
                "fallback_tdl_api_key": fallback_tdl_api_key,
                "run_id": run_id,
            }
        )
        yield RuntimeStreamEvent(
            kind="tool",
            payload={
                "phase": "start",
                "runId": run_id,
                "toolCallId": "tool-1",
                "name": "bash",
                "display_name": "bash",
                "status": "running",
                "toolType": "claude_task",
                "arguments": {"command": "echo hi"},
            },
        )
        yield RuntimeStreamEvent(
            kind="task",
            payload={
                "phase": "start",
                "runId": run_id,
                "taskId": "task-1",
                "taskType": "bash",
                "status": "running",
                "toolCallId": "tool-1",
                "name": "bash task",
            },
        )
        yield RuntimeStreamEvent(kind="text", text="hello ")
        yield RuntimeStreamEvent(kind="text", text="world")

    async def runtime_snapshot(self) -> Mapping[str, Any]:
        return {"connectedSessions": 1, "sessions": [{"frontendSessionId": "session-1"}]}

    async def inspect_workspace(
        self,
        payload: Mapping[str, Any],
        **_: Any,
    ) -> Mapping[str, Any]:
        workspace = payload.get("workspace") if isinstance(payload.get("workspace"), Mapping) else {}
        return {
            "schemaVersion": "claude-code.workspace-runtime/v1",
            "workspace": {
                "cwd": str(workspace.get("cwd") or ""),
                "addDirs": list(workspace.get("add_dirs") or []),
            },
            "resources": {"commands": {"detectedCount": 1, "activeCount": 1, "items": []}},
        }

    async def approvals_snapshot(self) -> Mapping[str, Any]:
        return {"pendingApprovalNum": 1, "approvalSessionNum": 1}

    async def questions_snapshot(self) -> Mapping[str, Any]:
        return {"pendingQuestionNum": 1, "questionSessionNum": 1}

    async def get_session_state(self, frontend_session_id: str) -> Mapping[str, Any] | None:
        if frontend_session_id != "session-1":
            return None
        return {"frontendSessionId": frontend_session_id, "claudeSessionId": "claude-1", "checkpoints": [{"checkpoint_id": "cp-1"}]}

    async def interrupt_session(self, frontend_session_id: str) -> bool:
        self.interrupts.append(frontend_session_id)
        return frontend_session_id == "session-1"

    async def list_checkpoints(self, frontend_session_id: str) -> list[Mapping[str, Any]]:
        if frontend_session_id != "session-1":
            return []
        return [{"checkpoint_id": "cp-1"}]

    async def checkpoint_snapshot(self, frontend_session_id: str) -> Mapping[str, Any]:
        return {
            "ok": True,
            "supported": True,
            "enabled": True,
            "sessionId": frontend_session_id,
            "workspacePath": "/tmp/claude-workspace",
            "checkpoints": await self.list_checkpoints(frontend_session_id),
        }

    async def list_approvals(self, frontend_session_id: str) -> list[Mapping[str, Any]]:
        if frontend_session_id != "session-1":
            return []
        return [dict(item) for item in self.approvals]

    async def get_approval(self, frontend_session_id: str, request_id: str) -> Mapping[str, Any] | None:
        if frontend_session_id != "session-1":
            return None
        for item in self.approvals:
            if item["requestId"] == request_id:
                return dict(item)
        return None

    async def stream_approvals(self, frontend_session_id: str) -> AsyncIterator[Mapping[str, Any]]:
        for item in await self.list_approvals(frontend_session_id):
            yield item

    async def respond_approval(
        self,
        frontend_session_id: str,
        request_id: str,
        *,
        decision: str,
        reason: str = "",
    ) -> Mapping[str, Any] | None:
        if frontend_session_id != "session-1":
            return None
        for item in self.approvals:
            if item["requestId"] == request_id:
                item["status"] = "allowed" if decision == "allow" else "denied"
                item["decision"] = decision
                item["reason"] = reason
                return dict(item)
        return None

    async def list_questions(self, frontend_session_id: str) -> list[Mapping[str, Any]]:
        if frontend_session_id != "session-1":
            return []
        return [dict(item) for item in self.questions]

    async def get_question(self, frontend_session_id: str, question_id: str) -> Mapping[str, Any] | None:
        if frontend_session_id != "session-1":
            return None
        for item in self.questions:
            if item["questionId"] == question_id:
                return dict(item)
        return None

    async def stream_questions(self, frontend_session_id: str) -> AsyncIterator[Mapping[str, Any]]:
        for item in await self.list_questions(frontend_session_id):
            yield item

    async def answer_question(
        self,
        frontend_session_id: str,
        question_id: str,
        *,
        answer: str,
    ) -> Mapping[str, Any] | None:
        if frontend_session_id != "session-1":
            return None
        for item in self.questions:
            if item["questionId"] == question_id:
                item["status"] = "answered"
                item["answer"] = answer
                return dict(item)
        return None

    async def list_hooks(self, frontend_session_id: str) -> list[Mapping[str, Any]]:
        if frontend_session_id != "session-1":
            return []
        return [dict(item) for item in self.hooks]

    async def get_hook(self, frontend_session_id: str, event_id: str) -> Mapping[str, Any] | None:
        if frontend_session_id != "session-1":
            return None
        for item in self.hooks:
            if item["eventId"] == event_id:
                return dict(item)
        return None

    async def stream_hooks(self, frontend_session_id: str) -> AsyncIterator[Mapping[str, Any]]:
        for item in await self.list_hooks(frontend_session_id):
            yield item

    async def rewind_session(self, frontend_session_id: str, checkpoint_id: str) -> bool:
        return frontend_session_id == "session-1" and checkpoint_id == "cp-1"

    async def rewind_checkpoint(self, frontend_session_id: str, checkpoint_id: str, **_: Any) -> Mapping[str, Any]:
        if frontend_session_id == "session-1" and checkpoint_id == "cp-1":
            return {
                "ok": True,
                "status": "completed",
                "sessionId": frontend_session_id,
                "checkpointId": checkpoint_id,
                "action": "rewind",
            }
        return {
            "ok": False,
            "status": "checkpoint_not_found",
            "error": "checkpoint not found",
            "sessionId": frontend_session_id,
            "checkpointId": checkpoint_id,
        }

    async def disconnect_all(self) -> None:
        return None


class _ArtifactRuntime(_FakeRuntime):
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
        async for event in super().stream_events(
            payload,
            request_headers=request_headers,
            current_user=current_user,
            fallback_tdl_api_key=fallback_tdl_api_key,
            proxy_base_url=proxy_base_url,
            skill_mount_root=skill_mount_root,
            run_id=run_id,
        ):
            yield event
        yield RuntimeStreamEvent(
            kind="artifacts",
            payload={
                "sessionId": str(payload.get("session_id") or ""),
                "runId": str(run_id or ""),
                "runtime": "claude-sdk-agent",
                "summary": {"created": 0, "modified": 1, "deleted": 0, "artifactCount": 1, "truncated": False},
                "artifacts": [],
            },
        )


class _CancellingRuntime(_FakeRuntime):
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
        self.calls.append(
            {
                "payload": dict(payload),
                "request_headers": dict(request_headers),
                "proxy_base_url": proxy_base_url,
                "skill_mount_root": str(skill_mount_root),
                "current_user": current_user,
                "fallback_tdl_api_key": fallback_tdl_api_key,
                "run_id": run_id,
            }
        )
        yield RuntimeStreamEvent(kind="text", text="partial")
        raise asyncio.CancelledError()


class _BlockingApprovalRuntime(_FakeRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()

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
        self.calls.append(
            {
                "payload": dict(payload),
                "request_headers": dict(request_headers),
                "proxy_base_url": proxy_base_url,
                "skill_mount_root": str(skill_mount_root),
                "current_user": current_user,
                "fallback_tdl_api_key": fallback_tdl_api_key,
                "run_id": run_id,
            }
        )
        await self.release.wait()
        yield RuntimeStreamEvent(kind="text", text="done")


class _TaskIssueOnlyRuntime(_FakeRuntime):
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
        self.calls.append(
            {
                "payload": dict(payload),
                "request_headers": dict(request_headers),
                "proxy_base_url": proxy_base_url,
                "skill_mount_root": str(skill_mount_root),
                "current_user": current_user,
                "fallback_tdl_api_key": fallback_tdl_api_key,
                "run_id": run_id,
            }
        )
        yield RuntimeStreamEvent(
            kind="task",
            payload={
                "phase": "end",
                "runId": run_id,
                "taskId": "task-1",
                "taskType": "task",
                "status": "stopped",
                "name": "检查 UI 入口文件",
                "log": 'No completion record was found for background agent "检查 UI 入口文件".',
            },
        )


class ServerTests(unittest.TestCase):
    def test_project_file_upload_accepts_exact_size_limit(self) -> None:
        old_max = server_module.PROJECT_FILE_MAX_BYTES
        old_chunk = server_module.PROJECT_FILE_READ_CHUNK_BYTES
        server_module.PROJECT_FILE_MAX_BYTES = 4
        server_module.PROJECT_FILE_READ_CHUNK_BYTES = 2
        try:
            content = asyncio.run(_read_project_file_upload(_ChunkedUpload([b"ab", b"cd"])))
        finally:
            server_module.PROJECT_FILE_MAX_BYTES = old_max
            server_module.PROJECT_FILE_READ_CHUNK_BYTES = old_chunk

        self.assertEqual(content, b"abcd")

    def test_project_file_upload_rejects_over_size_limit(self) -> None:
        old_max = server_module.PROJECT_FILE_MAX_BYTES
        old_chunk = server_module.PROJECT_FILE_READ_CHUNK_BYTES
        server_module.PROJECT_FILE_MAX_BYTES = 4
        server_module.PROJECT_FILE_READ_CHUNK_BYTES = 2
        try:
            with self.assertRaises(Exception) as captured:
                asyncio.run(_read_project_file_upload(_ChunkedUpload([b"ab", b"cd", b"e"])))
        finally:
            server_module.PROJECT_FILE_MAX_BYTES = old_max
            server_module.PROJECT_FILE_READ_CHUNK_BYTES = old_chunk

        self.assertEqual(getattr(captured.exception, "status_code", None), 413)
        self.assertEqual(getattr(captured.exception, "detail", ""), "Project file size must not exceed 100MB")

    def test_app_startup_creates_runtime_allow_users_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_service_json(root, auth_enabled=False)
            runtime_allow_users = root / "data" / "runtime" / "allow_users.json"
            self.assertFalse(runtime_allow_users.exists())
            create_app(root=root)
            self.assertTrue(runtime_allow_users.exists())
            self.assertEqual(json.loads(runtime_allow_users.read_text(encoding="utf-8")), {"allow_users": []})

    def test_internal_anthropic_path_bypasses_outer_auth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_service_json(root, auth_enabled=True)
            app = create_app(root=root)
            client = TestClient(app)
            request = client.build_request("POST", "/internal/anthropic/v1/messages")
            scope = {
                "type": "http",
                "method": "POST",
                "path": "/internal/anthropic/v1/messages",
                "headers": request.headers.raw,
                "client": ("10.137.58.137", 12345),
                "scheme": "http",
                "server": ("10.137.58.137", 18008),
                "query_string": b"",
                "root_path": "",
                "http_version": "1.1",
            }
            from starlette.requests import Request as StarletteRequest

            self.assertTrue(_should_bypass_auth(StarletteRequest(scope)))

    def test_streaming_chat_returns_sse_and_normalizes_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_service_json(root, auth_enabled=False)
            app = create_app(root=root)
            fake_runtime = _FakeRuntime()
            app.state.runtime = fake_runtime
            client = TestClient(app)

            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hello"}], "user": "session-1"},
            )

            self.assertEqual(response.status_code, 200)
            self.assertIn("chat.completion.chunk", response.text)
            self.assertIn("hello ", response.text)
            self.assertIn("[tool] ", response.text)
            self.assertNotIn("[task] ", response.text)
            self.assertNotIn("[approval] ", response.text)
            self.assertNotIn("[question] ", response.text)
            self.assertNotIn("[meta] ", response.text)
            self.assertEqual(fake_runtime.calls[0]["payload"]["session_id"], "session-1")
            self.assertTrue(fake_runtime.calls[0]["proxy_base_url"].endswith("/internal/anthropic"))
            self.assertTrue(str(fake_runtime.calls[0]["run_id"]).startswith("run-"))
            self.assertEqual(fake_runtime.calls[0]["fallback_tdl_api_key"], "tdl_shared_key")

    def test_streaming_chat_emits_artifacts_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_service_json(root, auth_enabled=False)
            app = create_app(root=root)
            app.state.runtime = _ArtifactRuntime()
            client = TestClient(app)

            response = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hello"}],
                    "user": "session-1",
                    "metadata": {"agentconfig": {"artifacts_enabled": True}},
                },
            )

            self.assertEqual(response.status_code, 200)
            self.assertIn("[artifacts] ", response.text)
            self.assertIn("artifacts.updated", response.text)
            self.assertIn("/v1/runs/run-", response.text)

    def test_streaming_chat_injects_structured_shells_only_when_feature_flags_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_service_json(
                root,
                auth_enabled=False,
                approval_frontend_enabled=True,
                question_frontend_enabled=True,
                hook_frontend_enabled=True,
                task_panel_frontend_enabled=True,
            )
            app = create_app(root=root)
            fake_runtime = _FakeRuntime()
            app.state.runtime = fake_runtime
            client = TestClient(app)

            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hello"}], "user": "session-1"},
            )

            self.assertEqual(response.status_code, 200)
            self.assertIn("[meta] ", response.text)
            self.assertIn("[task] ", response.text)
            self.assertIn("[approval] ", response.text)
            self.assertIn("[question] ", response.text)
            self.assertIn("[hook] ", response.text)
            self.assertIn("[run_result] ", response.text)
            self.assertIn('\\"finalText\\":\\"hello world\\"', response.text)

    def test_streaming_chat_emits_visible_message_for_empty_task_issue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_service_json(root, auth_enabled=False, task_panel_frontend_enabled=True)
            app = create_app(root=root)
            app.state.runtime = _TaskIssueOnlyRuntime()
            client = TestClient(app)

            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "汇总了么"}], "user": "session-1"},
            )

            self.assertEqual(response.status_code, 200)
            self.assertIn("[task] ", response.text)
            self.assertIn("未能生成回复", response.text)
            self.assertIn("后台子任务没有返回可用的完成记录", response.text)
            self.assertIn("检查 UI 入口文件", response.text)

    def test_streaming_chat_emits_approval_while_runtime_waits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_service_json(root, auth_enabled=False, approval_frontend_enabled=True)
            app = create_app(root=root)
            runtime = _BlockingApprovalRuntime()
            app.state.runtime = runtime

            async def scenario() -> str:
                stream = _stream_chat_response(
                    app=app,
                    runtime=runtime,
                    payload={"messages": [{"role": "user", "content": "hello"}], "session_id": "session-1"},
                    model_name="MiniMax-RAN3",
                    request_headers={},
                    current_user=None,
                    fallback_tdl_api_key="",
                    proxy_base_url="http://testserver/internal/anthropic",
                    skill_mount_root=root,
                    run_id="run-1",
                )
                chunks: list[str] = []
                chunks.append((await anext(stream)).decode("utf-8"))
                chunks.append((await anext(stream)).decode("utf-8"))
                chunks.append((await asyncio.wait_for(anext(stream), timeout=1.0)).decode("utf-8"))
                runtime.release.set()
                async for chunk in stream:
                    chunks.append(chunk.decode("utf-8"))
                return "".join(chunks)

            response_text = asyncio.run(scenario())
            self.assertIn("[meta] ", response_text)
            self.assertIn("[approval] ", response_text)
            self.assertIn('\\"requestId\\":\\"req-1\\"', response_text)
            self.assertIn("done", response_text)

    def test_streaming_chat_passes_shared_tdl_api_key_from_runtime_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_service_json(root, auth_enabled=False)
            app = create_app(root=root)
            fake_runtime = _FakeRuntime()
            app.state.runtime = fake_runtime
            client = TestClient(app)

            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hello"}], "user": "session-1"},
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(fake_runtime.calls[0]["fallback_tdl_api_key"], "tdl_shared_key")

    def test_chat_request_logging_redacts_tokens_and_base64(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_service_json(root, auth_enabled=False)
            app = create_app(root=root)
            app.state.runtime = _FakeRuntime()
            client = TestClient(app)

            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "check attachment"},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + ("A" * 512)}},
                            {
                                "type": "file_url",
                                "file_url": {
                                    "url": "https://files.example.local/demo.pdf",
                                    "mime_type": "application/pdf",
                                    "filename": "demo.pdf",
                                },
                            },
                        ],
                    }
                ],
                "user": "session-1",
                "api-key": "tdl_demo_secret_token",
                "uac-user-token": "uac-secret-token",
            }

            with self.assertLogs("src.api.server", level="INFO") as captured:
                response = client.post("/v1/chat/completions", json=payload)

            self.assertEqual(response.status_code, 200)
            combined = "\n".join(captured.output)
            self.assertIn("[chat] payload_sanitized", combined)
            self.assertIn("https://files.example.local/demo.pdf", combined)
            self.assertIn("data:image/png;base64,<base64:512 chars>", combined)
            self.assertIn("tdl_...oken<redacted>", combined)
            self.assertIn("uac-...oken<redacted>", combined)
            self.assertNotIn("tdl_demo_secret_token", combined)
            self.assertNotIn("uac-secret-token", combined)

    def test_format_chat_payload_for_log_truncates_long_text(self) -> None:
        logged = _format_chat_payload_for_log(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "x" * 520}],
                    }
                ]
            }
        )

        self.assertIn("<truncated:", logged)

    def test_alias_model_names_fall_back_to_default_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_service_json(root, auth_enabled=False)
            app = create_app(root=root)
            fake_runtime = _FakeRuntime()
            app.state.runtime = fake_runtime
            client = TestClient(app)

            for requested in ("openclaw:main", "claude-code", ""):
                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": "hello"}],
                        "user": "session-1",
                        "model": requested,
                    },
                )
                self.assertEqual(response.status_code, 200)

            routed_models = [call["payload"]["model"] for call in fake_runtime.calls]
            self.assertEqual(routed_models, ["MiniMax-RAN3", "MiniMax-RAN3", "MiniMax-RAN3"])

    def test_custom_model_name_is_forwarded_to_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_service_json(root, auth_enabled=False)
            app = create_app(root=root)
            fake_runtime = _FakeRuntime()
            app.state.runtime = fake_runtime
            client = TestClient(app)

            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hello"}], "user": "session-1", "model": "MiniMax-RAN3-Custom"},
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(fake_runtime.calls[0]["payload"]["model"], "MiniMax-RAN3-Custom")

    def test_stream_cancel_triggers_runtime_interrupt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_service_json(root, auth_enabled=False)
            app = create_app(root=root)
            runtime = _CancellingRuntime()

            async def scenario() -> None:
                await app.state.tool_registry.register_run("run-1", session_id="session-1")
                stream = _stream_chat_response(
                    app=app,
                    runtime=runtime,
                    payload={"session_id": "session-1", "messages": [{"role": "user", "content": "hello"}]},
                    model_name="MiniMax-RAN3",
                    request_headers={},
                    current_user=None,
                    fallback_tdl_api_key="",
                    proxy_base_url="http://127.0.0.1/internal/anthropic",
                    skill_mount_root=root,
                    run_id="run-1",
                )
                first = await anext(stream)
                self.assertIn(b"chat.completion.chunk", first)
                with self.assertRaises(asyncio.CancelledError):
                    while True:
                        await anext(stream)

            asyncio.run(scenario())
            self.assertEqual(runtime.interrupts, ["session-1"])

    def test_stream_cancel_does_not_interrupt_when_feature_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_service_json(root, auth_enabled=False, auto_interrupt_on_disconnect=False)
            app = create_app(root=root)
            runtime = _CancellingRuntime()

            async def scenario() -> None:
                await app.state.tool_registry.register_run("run-1", session_id="session-1")
                stream = _stream_chat_response(
                    app=app,
                    runtime=runtime,
                    payload={"session_id": "session-1", "messages": [{"role": "user", "content": "hello"}]},
                    model_name="MiniMax-RAN3",
                    request_headers={},
                    current_user=None,
                    fallback_tdl_api_key="",
                    proxy_base_url="http://127.0.0.1/internal/anthropic",
                    skill_mount_root=root,
                    run_id="run-1",
                )
                await anext(stream)
                with self.assertRaises(asyncio.CancelledError):
                    while True:
                        await anext(stream)

            asyncio.run(scenario())
            self.assertEqual(runtime.interrupts, [])

    def test_runtime_status_exposes_sdk_option_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_service_json(root, auth_enabled=False)
            app = create_app(root=root)
            client = TestClient(app)

            response = client.get("/v1/runtime/status")
            body = response.json()

            self.assertEqual(response.status_code, 200)
            self.assertEqual(body["sdkOptions"]["setting_sources"], ["project", "local"])
            self.assertEqual(body["sdkOptions"]["skills_filter"], ["demo-skill"])
            self.assertEqual(body["sdkOptions"]["system_prompt_preset"], "claude_code")
            self.assertEqual(body["runtime"], "claude-sdk-agent")
            self.assertEqual(body["runtimeContract"]["schemaVersion"], "claude-code.runtime/v2")
            self.assertEqual(body["agentRuntime"]["scope"], "agent")
            self.assertEqual(body["agentRuntime"]["settingSources"], ["project", "local"])
            self.assertEqual(body["agentTaskNum"], 0)
            self.assertEqual(body["toolRuntime"]["agentTaskNum"], 0)
            self.assertTrue(body["sdkOptions"]["include_hook_events"])
            self.assertTrue(body["sdkOptions"]["enable_file_checkpointing"])
            self.assertEqual(body["sdkOptions"]["attachment_text_char_limit"], 123456)
            self.assertEqual(body["workflow_count"], 1)
            self.assertEqual(body["workflow_names"], ["demo.js"])
            self.assertEqual(Path(str(body["workflow_mount_root"])).parts[-2:], (".claude", "workflows"))
            self.assertEqual(Path(str(body["sdkOptions"]["workflow_target_dir"])).parts[-2:], (".claude", "workflows"))
            self.assertTrue(body["featureFlags"]["auto_interrupt_on_disconnect"])
            self.assertTrue(body["mcpOptions"]["auto_load"])
            self.assertEqual(Path(str(body["mcp_config_dir"])).parts[-2:], ("shared", "mcps"))
            self.assertFalse(body["featureFlags"]["approval_frontend_enabled"])
            self.assertFalse(body["featureFlags"]["question_frontend_enabled"])
            self.assertFalse(body["featureFlags"]["hook_frontend_enabled"])
            self.assertFalse(body["featureFlags"]["checkpoint_rewind_frontend_enabled"])
            self.assertFalse(body["featureFlags"]["task_panel_frontend_enabled"])
            self.assertFalse(body["featureFlags"]["subagent_events_frontend_enabled"])
            self.assertEqual(body["approvalRuntime"]["pendingApprovalNum"], 0)
            self.assertEqual(body["questionRuntime"]["pendingQuestionNum"], 0)
            self.assertEqual(body["hookRuntime"]["hookEventNum"], 0)

    def test_workspace_inspect_endpoint_returns_request_scoped_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            additional = root / "additional"
            workspace.mkdir()
            additional.mkdir()
            _write_service_json(root, auth_enabled=False)
            app = create_app(root=root)
            app.state.runtime = _FakeRuntime()
            client = TestClient(app)

            response = client.post(
                "/v1/runtime/workspace/inspect",
                json={
                    "workspace": {
                        "source": "agent",
                        "cwd": str(workspace),
                        "add_dirs": [str(additional)],
                    }
                },
            )
            body = response.json()

            self.assertEqual(response.status_code, 200)
            self.assertTrue(body["ok"])
            self.assertEqual(body["workspaceRuntime"]["workspace"]["cwd"], str(workspace))
            self.assertEqual(body["workspaceRuntime"]["workspace"]["addDirs"], [str(additional)])

    def test_runtime_permission_mutation_endpoints_are_not_exposed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_service_json(root, auth_enabled=False)
            app = create_app(root=root)
            client = TestClient(app)

            read_response = client.get(
                "/v1/runtime/permissions",
                headers={"x-agent-runtime-id": "agent-a"},
            )
            write_response = client.put(
                "/v1/runtime/permissions",
                headers={"x-agent-runtime-id": "agent-b"},
                json={"profile": "fullBypass", "expectedRevision": 0},
            )

            self.assertEqual(read_response.status_code, 404)
            self.assertEqual(write_response.status_code, 404)

    def test_runtime_status_reports_active_chat_run_as_agent_task_num(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_service_json(root, auth_enabled=False)
            app = create_app(root=root)
            client = TestClient(app)

            asyncio.run(app.state.tool_registry.register_run("run-1", session_id="session-1"))

            response = client.get("/v1/runtime/status")
            body = response.json()

            self.assertEqual(response.status_code, 200)
            self.assertEqual(body["agentTaskNum"], 1)
            self.assertEqual(body["toolRuntime"]["agentTaskNum"], 1)
            self.assertEqual(body["toolRuntime"]["activeRuns"][0]["sessionId"], "session-1")

    def test_session_control_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_service_json(root, auth_enabled=False)
            app = create_app(root=root)
            app.state.runtime = _FakeRuntime()
            client = TestClient(app)

            session_response = client.get("/v1/sessions/session-1")
            checkpoints_response = client.get("/v1/sessions/session-1/checkpoints")
            interrupt_response = client.post("/v1/sessions/session-1/interrupt")
            rewind_response = client.post("/v1/sessions/session-1/checkpoints/cp-1/rewind")

            self.assertEqual(session_response.status_code, 200)
            self.assertTrue(session_response.json()["ok"])
            self.assertEqual(checkpoints_response.status_code, 200)
            self.assertTrue(checkpoints_response.json()["enabled"])
            self.assertEqual(checkpoints_response.json()["checkpoints"][0]["checkpoint_id"], "cp-1")
            self.assertEqual(interrupt_response.status_code, 200)
            self.assertTrue(interrupt_response.json()["ok"])
            self.assertEqual(rewind_response.status_code, 200)
            self.assertTrue(rewind_response.json()["ok"])
            self.assertEqual(rewind_response.json()["status"], "completed")

    def test_approval_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_service_json(root, auth_enabled=False)
            app = create_app(root=root)
            app.state.runtime = _FakeRuntime()
            client = TestClient(app)

            list_response = client.get("/v1/sessions/session-1/approvals")
            get_response = client.get("/v1/sessions/session-1/approvals/req-1")
            stream_response = client.get("/v1/sessions/session-1/approvals/stream")
            respond_response = client.post(
                "/v1/sessions/session-1/approvals/req-1",
                json={"decision": "allow"},
            )

            self.assertEqual(list_response.status_code, 200)
            self.assertEqual(list_response.json()["approvals"][0]["requestId"], "req-1")
            self.assertEqual(get_response.status_code, 200)
            self.assertTrue(get_response.json()["ok"])
            self.assertIn("event: approval", stream_response.text)
            self.assertIn('"requestId": "req-1"', stream_response.text)
            self.assertEqual(respond_response.status_code, 200)
            self.assertEqual(respond_response.json()["approval"]["decision"], "allow")

    def test_question_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_service_json(root, auth_enabled=False)
            app = create_app(root=root)
            app.state.runtime = _FakeRuntime()
            client = TestClient(app)

            list_response = client.get("/v1/sessions/session-1/questions")
            get_response = client.get("/v1/sessions/session-1/questions/question-1")
            stream_response = client.get("/v1/sessions/session-1/questions/stream")
            answer_response = client.post(
                "/v1/sessions/session-1/questions/question-1",
                json={"answer": "Here is more detail"},
            )

            self.assertEqual(list_response.status_code, 200)
            self.assertEqual(list_response.json()["questions"][0]["requestId"], "request-1")
            self.assertEqual(get_response.status_code, 200)
            self.assertTrue(get_response.json()["ok"])
            self.assertIn("event: question", stream_response.text)
            self.assertIn('"questionId": "question-1"', stream_response.text)
            self.assertEqual(answer_response.status_code, 200)
            self.assertEqual(answer_response.json()["question"]["status"], "answered")

    def test_hook_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_service_json(root, auth_enabled=False)
            app = create_app(root=root)
            app.state.runtime = _FakeRuntime()
            client = TestClient(app)

            list_response = client.get("/v1/sessions/session-1/hooks")
            get_response = client.get("/v1/sessions/session-1/hooks/hook-1")
            stream_response = client.get("/v1/sessions/session-1/hooks/stream")

            self.assertEqual(list_response.status_code, 200)
            self.assertEqual(list_response.json()["hooks"][0]["eventId"], "hook-1")
            self.assertEqual(get_response.status_code, 200)
            self.assertTrue(get_response.json()["ok"])
            self.assertIn("event: hook", stream_response.text)
            self.assertIn('"eventId": "hook-1"', stream_response.text)

    def test_non_stream_chat_returns_openai_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_service_json(root, auth_enabled=False)
            app = create_app(root=root)
            app.state.runtime = _FakeRuntime()
            client = TestClient(app)

            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hello"}], "user": "session-2", "stream": False},
            )

            body = response.json()
            self.assertEqual(response.status_code, 200)
            self.assertEqual(body["object"], "chat.completion")
            self.assertEqual(body["choices"][0]["message"]["content"], "hello world")

    def test_non_stream_chat_returns_task_issue_message_for_empty_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_service_json(root, auth_enabled=False, task_panel_frontend_enabled=True)
            app = create_app(root=root)
            app.state.runtime = _TaskIssueOnlyRuntime()
            client = TestClient(app)

            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "汇总了么"}], "user": "session-2", "stream": False},
            )

            body = response.json()
            content = body["choices"][0]["message"]["content"]
            self.assertEqual(response.status_code, 200)
            self.assertIn("未能生成回复", content)
            self.assertIn("后台子任务没有返回可用的完成记录", content)
            self.assertIn("检查 UI 入口文件", content)

    def test_non_stream_chat_returns_artifact_headers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_service_json(root, auth_enabled=False)
            app = create_app(root=root)
            app.state.runtime = _ArtifactRuntime()
            client = TestClient(app)

            response = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hello"}],
                    "user": "session-2",
                    "stream": False,
                    "metadata": {"agentconfig": {"artifacts_enabled": True}},
                },
            )

            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.headers["x-agent-run-id"].startswith("run-"))
            self.assertEqual(response.headers["x-artifacts-count"], "1")

    def test_missing_auth_headers_returns_readable_streaming_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_service_json(root, auth_enabled=True)
            app = create_app(root=root)
            client = TestClient(app)

            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hello"}]},
            )

            self.assertEqual(response.status_code, 200)
            self.assertIn("鉴权失败：缺少请求头 uac-user-id/uac-user-token", response.text)

    def test_tool_status_and_output_stream_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_service_json(root, auth_enabled=False)
            app = create_app(root=root)
            client = TestClient(app)

            async def seed() -> None:
                await app.state.tool_registry.register_run("run-1", session_id="session-1")
                context = await app.state.tool_registry.start_tool(
                    run_id="run-1",
                    tool_call_id="tool-1",
                    name="bash",
                    display_name="bash",
                    tool_type="claude_task",
                    arguments={"command": "echo hi"},
                )
                await context.emit_output("system", "running")
                await app.state.tool_registry.finish_tool(run_id="run-1", tool_call_id="tool-1", status="completed")

            asyncio.run(seed())

            status_response = client.get("/v1/tools/run-1/tool-1/status/stream")
            output_response = client.get("/v1/tools/run-1/tool-1/output/stream")
            get_response = client.get("/v1/tools/run-1/tool-1")

            self.assertEqual(get_response.status_code, 200)
            self.assertTrue(get_response.json()["ok"])
            self.assertIn("event: status", status_response.text)
            self.assertIn('"status": "completed"', status_response.text)
            self.assertIn("event: end", status_response.text)
            self.assertIn("event: system", output_response.text)
            self.assertIn('"text": "running"', output_response.text)
            self.assertIn("streamStatus", output_response.text)

    def test_task_status_and_output_stream_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_service_json(root, auth_enabled=False)
            app = create_app(root=root)
            client = TestClient(app)

            async def seed() -> None:
                await app.state.task_registry.register_run("run-2")
                context = await app.state.task_registry.start_task(
                    run_id="run-2",
                    task_id="task-1",
                    description="run bash",
                    task_type="bash",
                    tool_call_id="tool-1",
                    metadata={"phase": "start"},
                )
                await context.emit_output("system", "task running")
                await app.state.task_registry.finish_task(
                    run_id="run-2",
                    task_id="task-1",
                    status="completed",
                    metadata={"phase": "end"},
                )

            asyncio.run(seed())

            list_response = client.get("/v1/tasks/run-2")
            get_response = client.get("/v1/tasks/run-2/task-1")
            status_response = client.get("/v1/tasks/run-2/task-1/status/stream")
            output_response = client.get("/v1/tasks/run-2/task-1/output/stream")

            self.assertEqual(list_response.status_code, 200)
            self.assertEqual(len(list_response.json()["tasks"]), 1)
            self.assertEqual(get_response.status_code, 200)
            self.assertTrue(get_response.json()["ok"])
            self.assertIn("event: status", status_response.text)
            self.assertIn('"status": "completed"', status_response.text)
            self.assertIn("event: system", output_response.text)
            self.assertIn('"text": "task running"', output_response.text)
            self.assertIn("streamStatus", output_response.text)
