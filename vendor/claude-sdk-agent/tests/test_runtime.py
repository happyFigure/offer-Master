from __future__ import annotations

import unittest
import asyncio
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from src.claude_sdk import runtime as runtime_module
from src.claude_sdk.runtime import ClaudeSdkRuntime
from src.claude_sdk.runtime import (
    _DESTRUCTIVE_COMMAND_SYSTEM_WARNING,
    _SessionExecutionContext,
    _SessionMessageAdapter,
    _ToolEventBridge,
    _build_claude_session_id,
    _build_sdk_system_prompt,
    _is_ignorable_terminal_exception,
    _permission_extra_args,
    _runtime_permission_options,
    _request_auth_env,
    _resolve_workspace_execution,
    _runtime_skill_platform_catalog,
    _sdk_add_dirs,
    _sdk_process_env,
    _session_signature,
)
from src.hook_control import HookRuntimeRegistry
from src.claude_sdk.client_pool import ClaudeClientPool
from src.claude_sdk.hooks import build_sdk_hooks, hook_stream_payload
from src.question_control import QuestionRuntimeRegistry
from src.claude_sdk.stream_adapter import assistant_text, content_tool_results, content_tool_starts, final_result_error_text, item_tool_file_paths, stream_event_tool_start, stringify_tool_content, task_message_payload
from src.config import ClaudeSettings, McpSettings, ProviderSettings
from src.provider.context_store import ProxyContextStore
from src.session.checkpoint_store import SessionCheckpointStore
from src.session.goal_store import SessionGoalStore
from src.session.store import SessionMappingStore
from src.task_control import TaskRuntimeRegistry
from src.tool_control import ToolRuntimeRegistry


class _TextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _AssistantMessage:
    def __init__(self, content) -> None:
        self.content = content


class _SessionTranscriptMessage:
    def __init__(self, content) -> None:
        self.content = content


class _StreamEvent:
    def __init__(self, event) -> None:
        self.event = event


class _ToolBlock:
    def __init__(self, *, block_type: str, tool_id: str, name: str, input_data=None, content=None, is_error=None) -> None:
        self.type = block_type
        self.id = tool_id
        self.tool_use_id = tool_id
        self.name = name
        self.input = input_data or {}
        self.content = content
        self.is_error = is_error


class _DummyOptions:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class PermissionOptionTests(unittest.TestCase):
    def test_full_bypass_adds_explicit_dangerous_cli_flag_without_mutating_config(self) -> None:
        configured = {"replay-user-messages": None}

        native = _permission_extra_args(configured, full_bypass=False)
        full = _permission_extra_args(configured, full_bypass=True)

        self.assertEqual(native, {"replay-user-messages": None})
        self.assertEqual(
            full,
            {
                "replay-user-messages": None,
                "dangerously-skip-permissions": None,
            },
        )
        self.assertEqual(configured, {"replay-user-messages": None})

    def test_request_permission_config_takes_precedence_over_header_identity(self) -> None:
        settings = ClaudeSettings(
            workdir=Path("/workspace"),
            config_dir=Path("/config"),
            default_model="test",
            permission_mode="default",
            cli_path=None,
            setting_sources=None,
            skills_filter=None,
            system_prompt_preset=None,
            system_prompt_append="",
            system_prompt_file=None,
            include_hook_events=False,
            enable_file_checkpointing=False,
            attachment_text_char_limit=1024,
            allowed_tools=["Read"],
            disallowed_tools=["Bash"],
        )
        options = _runtime_permission_options(
            {
                "metadata": {
                    "agentconfig": {
                        "runtime_config": {
                            "permissions": {
                                "profile": "fullBypass",
                                "runtime_key": "agent-from-service",
                                "revision": 7,
                            }
                        }
                    }
                },
                "runtime_config": {
                    "permissions": {"profile": "safe", "runtime_key": "client-draft"}
                },
            },
            settings,
            fallback_runtime_key="header-agent",
        )

        self.assertEqual(options.profile, "fullBypass")
        self.assertEqual(options.runtime_key, "agent-from-service")
        self.assertEqual(options.revision, 7)
        self.assertIsNone(options.allowed_tools)
        self.assertIsNone(options.disallowed_tools)


class SkillRuntimePolicyTests(unittest.TestCase):
    def test_reads_shared_skill_policy_from_root_and_metadata_runtime_config(self) -> None:
        self.assertEqual(
            _runtime_skill_platform_catalog(
                {"runtime_config": {"skills": {"platform_catalog": "exclude"}}}
            ),
            "exclude",
        )
        self.assertEqual(
            _runtime_skill_platform_catalog(
                {
                    "metadata": {
                        "agentconfig": {
                            "runtime_config": {
                                "skills": {"platformCatalog": "exclude"}
                            }
                        }
                    }
                }
            ),
            "exclude",
        )
        self.assertEqual(_runtime_skill_platform_catalog({}), "include")

    def test_excluding_platform_mount_keeps_workspace_add_dirs(self) -> None:
        settings = ClaudeSettings(
            workdir=Path("/workspace"),
            config_dir=Path("/config"),
            default_model="test",
            permission_mode="default",
            cli_path=None,
            setting_sources=None,
            skills_filter=None,
            system_prompt_preset=None,
            system_prompt_append="",
            system_prompt_file=None,
            include_hook_events=False,
            enable_file_checkpointing=False,
            attachment_text_char_limit=1024,
            add_dirs=[Path("/configured")],
        )

        self.assertEqual(
            _sdk_add_dirs(settings, None, ["/workspace-extra"]),
            ["/configured", "/workspace-extra"],
        )


@dataclass
class _DummyAgentDefinition:
    description: str
    prompt: str
    tools: list[str] | None = None
    disallowedTools: list[str] | None = None
    model: str | None = None
    skills: list[str] | None = None
    memory: str | None = None
    mcpServers: list | None = None
    initialPrompt: str | None = None
    maxTurns: int | None = None
    background: bool | None = None
    effort: str | int | None = None
    permissionMode: str | None = None


class _DummyHookMatcher:
    def __init__(self, matcher=None, hooks=None, timeout=None) -> None:
        self.matcher = matcher
        self.hooks = list(hooks or [])
        self.timeout = timeout


class _DummyClient:
    last_options_kwargs: dict = {}

    def __init__(self, *, options) -> None:
        self.options = options
        self.__class__.last_options_kwargs = dict(getattr(options, "kwargs", {}) or {})

    async def connect(self) -> None:
        return None

    async def get_server_info(self):
        return {"ok": True}


class _Exit127Error(RuntimeError):
    exit_code = 127


class _FailingConnectClient(_DummyClient):
    async def connect(self) -> None:
        self.options.kwargs["stderr"](
            "/usr/bin/env: node: No such file or directory token=secret-token"
        )
        raise _Exit127Error("Command failed with exit code 127")


class _QueryCapturingDummyClient(_DummyClient):
    last_query_messages: list[dict] = []

    async def query(self, prompt, session_id: str = "default") -> None:  # type: ignore[no-untyped-def]
        self.__class__.last_query_messages = []
        async for message in prompt:
            self.__class__.last_query_messages.append(message)

    async def receive_response(self):  # type: ignore[no-untyped-def]
        if False:
            yield None
        return


class _FirstQueryFailsDummyClient(_QueryCapturingDummyClient):
    query_attempts = 0
    query_messages_by_attempt: list[list[dict]] = []
    disconnect_calls = 0

    async def query(self, prompt, session_id: str = "default") -> None:  # type: ignore[no-untyped-def]
        messages: list[dict] = []
        async for message in prompt:
            messages.append(message)
        self.__class__.query_attempts += 1
        self.__class__.query_messages_by_attempt.append(messages)
        if self.__class__.query_attempts == 1:
            raise RuntimeError("first query failed")

    async def disconnect(self) -> None:
        self.__class__.disconnect_calls += 1


class _CompactBoundaryDummyClient(_QueryCapturingDummyClient):
    async def receive_response(self):  # type: ignore[no-untyped-def]
        yield type("SystemMessage", (), {"subtype": "compact_boundary"})()


class _ContextReportDummyClient(_QueryCapturingDummyClient):
    queried = False

    async def query(self, prompt, session_id: str = "default") -> None:  # type: ignore[no-untyped-def]
        self.__class__.queried = True
        await super().query(prompt, session_id=session_id)

    async def receive_response(self):  # type: ignore[no-untyped-def]
        yield type(
            "AssistantMessage",
            (),
            {"content": [{"type": "text", "text": "Context usage: 42%"}]},
        )()


class _GoalSetDummyClient(_QueryCapturingDummyClient):
    async def receive_response(self):  # type: ignore[no-untyped-def]
        yield type(
            "AssistantMessage",
            (),
            {"content": [{"type": "text", "text": "Goal set: 做完 Runtime Command P0"}]},
        )()


class _GoalRecoveryDummyClient(_DummyClient):
    query_messages: list[list[dict]] = []
    query_session_ids: list[str] = []

    async def query(self, prompt, session_id: str = "default") -> None:  # type: ignore[no-untyped-def]
        self.__class__.query_session_ids.append(session_id)
        messages = []
        async for message in prompt:
            messages.append(message)
        self.__class__.query_messages.append(messages)

    async def receive_response(self):  # type: ignore[no-untyped-def]
        yield type(
            "AssistantMessage",
            (),
            {"content": [{"type": "text", "text": "恢复后的普通消息已执行。"}]},
        )()


class _GoalStopHookDummyClient(_QueryCapturingDummyClient):
    async def receive_response(self):  # type: ignore[no-untyped-def]
        yield type(
            "AssistantMessage",
            (),
            {"content": [{"type": "text", "text": "目标执行完成。"}]},
        )()
        yield type(
            "AttachmentMessage",
            (),
            {
                "attachment": {
                    "type": "hook_response",
                    "hookName": "Stop",
                    "hookEvent": "Stop",
                    "toolUseID": "hook-stop-1",
                    "command": "做完 Runtime Command P0",
                    "stdout": '{"ok": true, "reason": "目标条件已满足"}',
                    "exitCode": 0,
                }
            },
        )()


class _GoalStopHookJsonNoiseDummyClient(_QueryCapturingDummyClient):
    async def receive_response(self):  # type: ignore[no-untyped-def]
        yield type(
            "AssistantMessage",
            (),
            {"content": [{"type": "text", "text": "目标执行完成。"}]},
        )()
        yield type(
            "AttachmentMessage",
            (),
            {
                "attachment": {
                    "type": "hook_non_blocking_error",
                    "hookName": "Stop",
                    "hookEvent": "Stop",
                    "toolUseID": "hook-stop-2",
                    "command": "做完 Runtime Command P0",
                    "stderr": "JSON validation failed",
                    "stdout": '\n{"ok": true, "reason": "目标条件已满足"}',
                    "exitCode": 1,
                }
            },
        )()


class _SubtaskSummaryDummyClient(_DummyClient):
    query_messages: list[list[dict]] = []

    def __init__(self, *, options) -> None:
        super().__init__(options=options)
        self.query_count = 0
        self.receive_count = 0

    async def query(self, prompt, session_id: str = "default") -> None:  # type: ignore[no-untyped-def]
        self.query_count += 1
        messages = []
        async for message in prompt:
            messages.append(message)
        self.__class__.query_messages.append(messages)

    async def receive_response(self):  # type: ignore[no-untyped-def]
        self.receive_count += 1
        if self.query_count == 1 and self.receive_count == 1:
            yield type(
                "AssistantMessage",
                (),
                {"content": [{"type": "text", "text": "子任务已启动，我会等待完成后汇总。"}]},
            )()
            yield type(
                "TaskStartedMessage",
                (),
                {
                    "task_id": "task-1",
                    "description": "检查 BFF 入口文件",
                    "uuid": "u1",
                    "session_id": "s1",
                    "tool_use_id": "tool-1",
                    "task_type": "local_agent",
                },
            )()
            return
        if self.query_count == 1 and self.receive_count == 2:
            yield type(
                "TaskNotificationMessage",
                (),
                {
                    "task_id": "task-1",
                    "status": "completed",
                    "output_file": "",
                    "summary": "BFF 检查完成",
                    "uuid": "u2",
                    "session_id": "s1",
                    "tool_use_id": "tool-1",
                    "usage": None,
                },
            )()
            return
        if self.query_count == 2:
            yield type(
                "AssistantMessage",
                (),
                {"content": [{"type": "text", "text": "最终汇总：BFF 检查完成。"}]},
            )()


class _SubtaskReceiveMessagesDummyClient(_DummyClient):
    query_messages: list[list[dict]] = []

    def __init__(self, *, options) -> None:
        super().__init__(options=options)
        self.query_count = 0
        self.receive_response_count = 0
        self.receive_messages_count = 0

    async def query(self, prompt, session_id: str = "default") -> None:  # type: ignore[no-untyped-def]
        self.query_count += 1
        messages = []
        async for message in prompt:
            messages.append(message)
        self.__class__.query_messages.append(messages)

    async def receive_response(self):  # type: ignore[no-untyped-def]
        self.receive_response_count += 1
        if self.query_count == 1 and self.receive_response_count == 1:
            yield type(
                "AssistantMessage",
                (),
                {"content": [{"type": "text", "text": "子任务已启动，我会等待完成后汇总。"}]},
            )()
            yield type(
                "TaskStartedMessage",
                (),
                {
                    "task_id": "task-1",
                    "description": "检查 UI 入口文件",
                    "uuid": "u1",
                    "session_id": "s1",
                    "tool_use_id": "tool-1",
                    "task_type": "local_agent",
                },
            )()
            yield type(
                "ResultMessage",
                (),
                {"session_id": "s1", "subtype": "success", "is_error": False},
            )()
            return
        if self.query_count == 2:
            yield type(
                "AssistantMessage",
                (),
                {"content": [{"type": "text", "text": "最终汇总：UI 检查完成。"}]},
            )()

    async def receive_messages(self):  # type: ignore[no-untyped-def]
        self.receive_messages_count += 1
        await asyncio.sleep(0.03)
        yield type(
            "TaskUpdatedMessage",
            (),
            {
                "task_id": "task-1",
                "patch": {
                    "task_id": "task-1",
                    "task_type": "local_agent",
                    "tool_use_id": "tool-1",
                    "description": "检查 UI 入口文件",
                    "status": "completed",
                    "result": "UI 检查完成",
                },
                "status": "completed",
                "uuid": "u2",
                "session_id": "s1",
            },
        )()


class _SubtaskNeverFinishesDummyClient(_SubtaskReceiveMessagesDummyClient):
    async def receive_messages(self):  # type: ignore[no-untyped-def]
        self.receive_messages_count += 1
        while True:
            await asyncio.sleep(1)
            if False:
                yield None
class _NoFileCheckpointClient:
    async def rewind_files(self, checkpoint_id: str) -> None:
        raise Exception("No file checkpoint found for this message.")


class _DummyAllow:
    def __init__(self, *, updated_input=None) -> None:
        self.behavior = "allow"
        self.updated_input = updated_input


class _DummyDeny:
    def __init__(self, *, message: str = "", interrupt: bool = False) -> None:
        self.behavior = "deny"
        self.message = message
        self.interrupt = interrupt


class RuntimeTests(unittest.TestCase):
    def test_assistant_text_reads_sdk_text_blocks_without_type_field(self) -> None:
        item = _AssistantMessage([_TextBlock("hello"), _TextBlock("world")])
        self.assertEqual(assistant_text(item), "hello\nworld")

    def test_assistant_text_reads_session_message_adapter(self) -> None:
        item = _SessionMessageAdapter({"content": [{"text": "answer"}]})
        self.assertEqual(assistant_text(item), "answer")

    def test_final_result_error_text_reports_max_turns(self) -> None:
        item = type(
            "ResultMessage",
            (),
            {"session_id": "session-1", "subtype": "error_max_turns", "is_error": True},
        )()

        self.assertIn("达到最大回合数", final_result_error_text(item))

    def test_build_claude_session_id_is_stable_for_frontend_session(self) -> None:
        self.assertEqual(
            _build_claude_session_id("session-1"),
            _build_claude_session_id("session-1"),
        )
        self.assertNotEqual(
            _build_claude_session_id("session-1"),
            _build_claude_session_id("session-2"),
        )

    def test_build_sdk_system_prompt_injects_destructive_command_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = _build_sdk_system_prompt(
                ClaudeSettings(
                    workdir=root,
                    config_dir=root / ".claude",
                    default_model="MiniMax-RAN3",
                    permission_mode="acceptEdits",
                    cli_path=None,
                    setting_sources=["project", "local"],
                    skills_filter="all",
                    system_prompt_preset="claude_code",
                    system_prompt_append="custom guidance",
                    system_prompt_file=None,
                    include_hook_events=False,
                    enable_file_checkpointing=True,
                    attachment_text_char_limit=256000,
                ),
                {"messages": [{"role": "user", "content": "hello"}]},
            )
        self.assertEqual(prompt["type"], "preset")
        self.assertEqual(prompt["preset"], "claude_code")
        self.assertIn(_DESTRUCTIVE_COMMAND_SYSTEM_WARNING, prompt["append"])
        self.assertIn("custom guidance", prompt["append"])

    def test_resolve_workspace_execution_normalizes_file_and_additional_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            project.mkdir()
            project_file = project / "README.md"
            project_file.write_text("demo", encoding="utf-8")
            additional = root / "additional"
            additional.mkdir()
            settings = ClaudeSettings(
                workdir=root,
                config_dir=root / ".claude",
                default_model="MiniMax-RAN3",
                permission_mode="acceptEdits",
                cli_path=None,
                setting_sources=["project", "local"],
                skills_filter="all",
                system_prompt_preset="claude_code",
                system_prompt_append="",
                system_prompt_file=None,
                include_hook_events=False,
                enable_file_checkpointing=True,
                attachment_text_char_limit=256000,
            )

            workspace = _resolve_workspace_execution(
                {
                    "metadata": {
                        "agentconfig": {
                            "workspace": {
                                "cwd": str(project_file),
                                "add_dirs": [str(additional), str(project)],
                            }
                        }
                    }
                },
                settings,
            )

        self.assertEqual(workspace.cwd, project.resolve())
        self.assertEqual(workspace.add_dirs, (additional.resolve(),))

    def test_rewind_marks_no_file_checkpoint_candidates_unavailable(self) -> None:
        async def scenario(root: Path) -> None:
            store = SessionCheckpointStore(root / "checkpoints.json")
            await store.put("session-1", "claude-1", "cp-1", prompt_excerpt="same prompt")
            await store.put("session-1", "claude-1", "cp-2", prompt_excerpt="same   prompt")
            checkpoint = await store.get("session-1", "cp-1")
            self.assertIsNotNone(checkpoint)

            runtime = ClaudeSdkRuntime.__new__(ClaudeSdkRuntime)
            runtime._checkpoint_store = store  # type: ignore[attr-defined]

            with self.assertRaisesRegex(RuntimeError, "没有可回滚的文件快照"):
                await runtime._rewind_files_with_duplicate_fallback(  # type: ignore[attr-defined]
                    _NoFileCheckpointClient(),
                    session_id="session-1",
                    checkpoint=checkpoint,
                    selected_checkpoint_id="cp-1",
                )

            raw_items = await store.list_raw("session-1")
            self.assertEqual([item.unavailable_reason for item in raw_items], ["no_file_checkpoint", "no_file_checkpoint"])
            self.assertEqual(await store.list("session-1"), [])

        with tempfile.TemporaryDirectory() as tmp:
            asyncio.run(scenario(Path(tmp)))

    def test_ignorable_terminal_exception_matches_success_bug(self) -> None:
        self.assertTrue(
            _is_ignorable_terminal_exception(
                Exception("Claude Code returned an error result: success")
            )
        )
        self.assertTrue(
            _is_ignorable_terminal_exception(
                Exception("Claude Code returned an error result: success; exit code 1")
            )
        )

    def test_ignorable_terminal_exception_does_not_hide_real_errors(self) -> None:
        self.assertFalse(
            _is_ignorable_terminal_exception(
                Exception("Claude Code returned an error result: rate_limit")
            )
        )
        self.assertFalse(_is_ignorable_terminal_exception(Exception("boom")))

    def test_stream_event_tool_start_detects_content_block(self) -> None:
        item = _StreamEvent(
            {
                "type": "content_block_start",
                "content_block": {"type": "tool_use", "id": "tool-1", "name": "bash", "input": {"command": "echo hi"}},
            }
        )
        payload = stream_event_tool_start(item)
        self.assertEqual(payload["toolCallId"], "tool-1")
        self.assertEqual(payload["name"], "bash")
        self.assertEqual(payload["arguments"]["command"], "echo hi")

    def test_content_tool_helpers_cover_tool_use_and_results(self) -> None:
        item = _SessionMessageAdapter(
            {
                "content": [
                    _ToolBlock(block_type="tool_use", tool_id="tool-1", name="Write", input_data={"file_path": "/tmp/demo.txt"}),
                    _ToolBlock(block_type="tool_result", tool_id="tool-1", name="bash", content=[{"text": "ok"}], is_error=False),
                ]
            }
        )
        starts = content_tool_starts(item)
        results = content_tool_results(item)
        self.assertEqual(starts[0]["toolCallId"], "tool-1")
        self.assertEqual(item_tool_file_paths(item), ["/tmp/demo.txt"])
        self.assertEqual(results[0]["status"], "completed")
        self.assertEqual(stringify_tool_content([{"text": "a"}, {"text": "b"}]), "a\nb")

    def test_task_message_payload_maps_task_messages(self) -> None:
        task = type(
            "TaskProgressMessage",
            (),
            {"task_id": "task-1", "description": "doing", "usage": {"x": 1}, "uuid": "u", "session_id": "s", "tool_use_id": "tool-1", "last_tool_name": "bash"},
        )()
        payload = task_message_payload(task)
        self.assertEqual(payload["phase"], "update")
        self.assertEqual(payload["taskId"], "task-1")
        self.assertEqual(payload["toolCallId"], "tool-1")
        self.assertEqual(payload["name"], "doing")

    def test_task_updated_payload_uses_patch_identity(self) -> None:
        task = type(
            "TaskUpdatedMessage",
            (),
            {
                "patch": {
                    "task_id": "task-2",
                    "task_type": "local_agent",
                    "tool_use_id": "tool-2",
                    "description": "检查 UI 入口",
                    "status": "failed",
                    "error": "模型未配置或已禁用: claude-opus-4-8",
                },
                "uuid": "u",
                "session_id": "s",
            },
        )()
        payload = task_message_payload(task)
        self.assertEqual(payload["phase"], "end")
        self.assertEqual(payload["taskId"], "task-2")
        self.assertEqual(payload["taskType"], "local_agent")
        self.assertEqual(payload["toolCallId"], "tool-2")
        self.assertEqual(payload["name"], "检查 UI 入口")
        self.assertEqual(payload["status"], "failed")
        self.assertIn("claude-opus-4-8", payload["result"])

    def test_task_notification_payload_keeps_structured_completed_status(self) -> None:
        task = type(
            "TaskNotificationMessage",
            (),
            {
                "task_id": "task-3",
                "task_type": "local_agent",
                "tool_use_id": "tool-3",
                "description": "检查 my-agents 入口",
                "status": "completed",
                "summary": 'API Error: 400 {"result":false,"failReason":"模型未配置或已禁用: claude-opus-4-8","data":null}',
                "uuid": "u",
                "session_id": "s",
            },
        )()
        payload = task_message_payload(task)
        self.assertEqual(payload["phase"], "end")
        self.assertEqual(payload["taskId"], "task-3")
        self.assertEqual(payload["toolCallId"], "tool-3")
        self.assertEqual(payload["name"], "检查 my-agents 入口")
        self.assertEqual(payload["status"], "completed")
        self.assertIn("API Error", payload["result"])

    def test_request_auth_env_maps_uac_headers_to_subprocess_env(self) -> None:
        current_user = type("User", (), {"emp_id": "10154402"})()
        env = _request_auth_env(
            {
                "uac-user-id": "10154402",
                "uac-user-token": "uac-token-1",
            },
            current_user,
            api_token="uac-token-1",
        )
        self.assertEqual(env["USER"], "10154402")
        self.assertEqual(env["UAC_USER_ID"], "10154402")
        self.assertEqual(env["VISION_MODEL_USER_ID"], "10154402")
        self.assertEqual(env["UAC_USER_TOKEN"], "uac-token-1")
        self.assertEqual(env["VISION_MODEL_USER_TOKEN"], "uac-token-1")
        self.assertEqual(env["coclaw_token"], "uac-token-1")
        self.assertEqual(env["VISION_MODEL_API_KEY"], "uac-token-1")

    def test_request_auth_env_maps_tdl_api_key_without_uac_token(self) -> None:
        env = _request_auth_env(
            {
                "x-user-id": "10154402",
            },
            None,
            api_token="tdl_demo_key",
        )
        self.assertEqual(env["USER"], "10154402")
        self.assertEqual(env["TDL_API_KEY"], "tdl_demo_key")
        self.assertEqual(env["VISION_MODEL_API_KEY"], "tdl_demo_key")
        self.assertNotIn("UAC_USER_TOKEN", env)

    def test_request_auth_env_uses_fallback_tdl_api_key_when_request_does_not_provide_one(self) -> None:
        env = _request_auth_env(
            {
                "uac-user-id": "10154402",
                "uac-user-token": "uac-token-1",
            },
            None,
            api_token="uac-token-1",
            fallback_tdl_api_key="tdl_shared_key",
        )
        self.assertEqual(env["TDL_API_KEY"], "tdl_shared_key")
        self.assertEqual(env["VISION_MODEL_API_KEY"], "uac-token-1")

    def test_sdk_process_env_preserves_explicit_path(self) -> None:
        settings = ClaudeSettings(
            workdir=Path("/tmp"),
            config_dir=Path("/tmp/.claude"),
            default_model="MiniMax-RAN3",
            permission_mode="default",
            cli_path="/opt/claude/bin/claude",
            setting_sources=[],
            skills_filter=None,
            system_prompt_preset=None,
            system_prompt_append="",
            system_prompt_file=None,
            include_hook_events=False,
            enable_file_checkpointing=False,
            attachment_text_char_limit=96000,
            env={"PATH": "/custom/bin"},
        )

        env = _sdk_process_env(settings, {"ANTHROPIC_BASE_URL": "http://127.0.0.1"})

        self.assertEqual(env["PATH"], "/custom/bin")

    def test_session_signature_ignores_internal_proxy_auth_token(self) -> None:
        common = {
            "claude_session_id": "claude-session-1",
            "model": "MiniMax-M2.7",
            "system_prompt": "prompt",
            "mcp_servers": {},
            "resumed": True,
            "permission_mode": "default",
            "allowed_tools": ["Read"],
            "disallowed_tools": [],
            "permission_profile": "safe",
        }

        first = _session_signature(
            env={
                "ANTHROPIC_BASE_URL": "http://127.0.0.1:18080/provider",
                "ANTHROPIC_AUTH_TOKEN": "request-token-a",
            },
            **common,
        )
        second = _session_signature(
            env={
                "ANTHROPIC_BASE_URL": "http://127.0.0.1:18080/provider",
                "ANTHROPIC_AUTH_TOKEN": "request-token-b",
            },
            **common,
        )

        self.assertEqual(first, second)

    def test_session_signature_keeps_stable_env_values(self) -> None:
        common = {
            "claude_session_id": "claude-session-1",
            "model": "MiniMax-M2.7",
            "system_prompt": "prompt",
            "mcp_servers": {},
            "resumed": True,
            "permission_mode": "default",
            "allowed_tools": ["Read"],
            "disallowed_tools": [],
            "permission_profile": "safe",
        }

        first = _session_signature(
            env={"ANTHROPIC_BASE_URL": "http://127.0.0.1:18080/provider-a"},
            **common,
        )
        second = _session_signature(
            env={"ANTHROPIC_BASE_URL": "http://127.0.0.1:18080/provider-b"},
            **common,
        )

        self.assertNotEqual(first, second)

    def test_session_signature_changes_with_workspace(self) -> None:
        common = {
            "claude_session_id": "claude-session-1",
            "model": "MiniMax-M2.7",
            "env": {},
            "system_prompt": "prompt",
            "mcp_servers": {},
            "resumed": True,
            "permission_mode": "default",
        }

        first = _session_signature(
            cwd="/srv/project-a",
            add_dirs=["/srv/project-b"],
            **common,
        )
        second = _session_signature(
            cwd="/srv/project-b",
            add_dirs=["/srv/project-a"],
            **common,
        )

        self.assertNotEqual(first, second)

    def test_session_signature_changes_with_workspace_resource_fingerprint(self) -> None:
        common = {
            "claude_session_id": "claude-session-1",
            "model": "MiniMax-M2.7",
            "env": {},
            "system_prompt": "prompt",
            "mcp_servers": {},
            "resumed": True,
            "permission_mode": "default",
            "cwd": "/srv/project-a",
            "add_dirs": [],
        }

        first = _session_signature(workspace_fingerprint="resource-v1", **common)
        second = _session_signature(workspace_fingerprint="resource-v2", **common)

        self.assertNotEqual(first, second)

    def test_session_signature_changes_with_platform_skill_policy(self) -> None:
        common = {
            "claude_session_id": "claude-session-1",
            "model": "MiniMax-M2.7",
            "env": {},
            "system_prompt": "prompt",
            "mcp_servers": {},
            "resumed": True,
            "permission_mode": "default",
            "cwd": "/srv/project-a",
            "add_dirs": [],
        }

        included = _session_signature(skill_platform_catalog="include", **common)
        excluded = _session_signature(skill_platform_catalog="exclude", **common)

        self.assertNotEqual(included, excluded)

    def test_can_use_tool_routes_ask_user_question_to_question_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            question_registry = QuestionRuntimeRegistry()
            runtime = ClaudeSdkRuntime(
                ClaudeSettings(
                    workdir=root,
                    config_dir=root / ".claude",
                    default_model="MiniMax-RAN3",
                    permission_mode="default",
                    cli_path="/usr/bin/claude",
                    setting_sources=[],
                    skills_filter="all",
                    system_prompt_preset="claude_code",
                    system_prompt_append="",
                    system_prompt_file=None,
                    include_hook_events=False,
                    enable_file_checkpointing=False,
                    attachment_text_char_limit=96000,
                ),
                McpSettings(config_dir=root / "mcps", auto_load=False),
                ProviderSettings(
                    base_url="http://127.0.0.1:18081",
                    anthropic_version="2023-06-01",
                    api_key="demo",
                    request_timeout_sec=30.0,
                ),
                SessionMappingStore(root / "data" / "sessions" / "session-map.json"),
                SessionCheckpointStore(root / "data" / "sessions" / "checkpoints.json"),
                ProxyContextStore(),
                client_pool=None,  # type: ignore[arg-type]
                question_registry=question_registry,
            )

            async def scenario() -> None:
                runtime._session_contexts["session-1"] = _SessionExecutionContext(
                    run_id="run-1",
                    claude_session_id="claude-1",
                )
                callback = runtime._build_can_use_tool_callback(
                    sdk_module=type(
                        "SdkModule",
                        (),
                        {"PermissionResultAllow": _DummyAllow, "PermissionResultDeny": _DummyDeny},
                    ),
                    frontend_session_id="session-1",
                )
                task = asyncio.create_task(
                    callback(
                        "AskUserQuestion",
                        {
                            "questions": [
                                {
                                    "question": "文件路径和内容是什么?",
                                    "header": "文件信息",
                                    "options": [
                                        {"label": "Other", "description": "自己填写"},
                                        {"label": "跳过", "description": "不创建"},
                                    ],
                                    "multiSelect": False,
                                }
                            ]
                        },
                        type("PermissionContext", (), {"tool_use_id": "tool-1", "agent_id": "agent-1"})(),
                    )
                )
                for _ in range(10):
                    questions = await question_registry.list_questions("session-1")
                    if questions:
                        break
                    await asyncio.sleep(0.01)
                self.assertEqual(len(questions), 1)
                self.assertEqual(questions[0]["prompt"], "文件路径和内容是什么?")
                self.assertEqual(questions[0]["metadata"]["toolName"], "AskUserQuestion")
                await question_registry.answer_question(
                    "session-1",
                    str(questions[0]["questionId"]),
                    answer="/tmp/demo.txt, hello",
                )
                response = await task
                self.assertEqual(response.behavior, "allow")
                self.assertEqual(
                    response.updated_input["answers"],
                    {"文件路径和内容是什么?": "/tmp/demo.txt, hello"},
                )

            asyncio.run(scenario())

    def test_connected_client_does_not_treat_frontend_session_as_os_user(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / ".claude"
            config_dir.mkdir(parents=True, exist_ok=True)
            runtime = ClaudeSdkRuntime(
                ClaudeSettings(
                    workdir=root,
                    config_dir=config_dir,
                    default_model="MiniMax-RAN3",
                    permission_mode="acceptEdits",
                    cli_path="/usr/bin/claude",
                    setting_sources=["project", "local"],
                    skills_filter="all",
                    system_prompt_preset="claude_code",
                    system_prompt_append="",
                    system_prompt_file=None,
                    include_hook_events=False,
                    enable_file_checkpointing=False,
                    attachment_text_char_limit=96000,
                ),
                McpSettings(config_dir=root / "mcps", auto_load=False),
                ProviderSettings(
                    base_url="http://127.0.0.1:18081",
                    anthropic_version="2023-06-01",
                    api_key="demo",
                    request_timeout_sec=30.0,
                ),
                SessionMappingStore(root / "data" / "sessions" / "session-map.json"),
                SessionCheckpointStore(root / "data" / "sessions" / "checkpoints.json"),
                ProxyContextStore(),
                client_pool=None,  # type: ignore[arg-type]
            )

            async def scenario() -> None:
                client, _ = await runtime._create_connected_client(
                    sdk={"ClaudeSDKClient": _DummyClient, "ClaudeAgentOptions": _DummyOptions},
                    model="MiniMax-RAN3",
                    system_prompt="system",
                    claude_session_id="claude-1",
                    resume_session_id="",
                    proxy_env={"ANTHROPIC_BASE_URL": "http://127.0.0.1/internal/anthropic"},
                    skill_mount_root=root,
                    mcp_servers={},
                    can_use_tool=None,
                    hooks={},
                )
                self.assertNotIn("user", client.options.kwargs)
                self.assertNotIn("can_use_tool", client.options.kwargs)

            asyncio.run(scenario())

    def test_connected_client_reports_sanitized_cli_stderr_on_exit_127(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / ".claude"
            config_dir.mkdir(parents=True, exist_ok=True)
            runtime = ClaudeSdkRuntime(
                ClaudeSettings(
                    workdir=root,
                    config_dir=config_dir,
                    default_model="MiniMax-RAN3",
                    permission_mode="acceptEdits",
                    cli_path="/usr/bin/claude",
                    setting_sources=[],
                    skills_filter=None,
                    system_prompt_preset=None,
                    system_prompt_append="",
                    system_prompt_file=None,
                    include_hook_events=False,
                    enable_file_checkpointing=False,
                    attachment_text_char_limit=96000,
                ),
                McpSettings(config_dir=root / "mcps", auto_load=False),
                ProviderSettings(
                    base_url="http://127.0.0.1:18081",
                    anthropic_version="2023-06-01",
                    api_key="demo",
                    request_timeout_sec=30.0,
                ),
                SessionMappingStore(root / "data" / "sessions" / "session-map.json"),
                SessionCheckpointStore(root / "data" / "sessions" / "checkpoints.json"),
                ProxyContextStore(),
                client_pool=None,  # type: ignore[arg-type]
            )

            async def scenario() -> None:
                with self.assertLogs("src.claude_sdk.runtime", level="WARNING") as captured:
                    with self.assertRaises(RuntimeError) as raised:
                        await runtime._create_connected_client(
                            sdk={
                                "ClaudeSDKClient": _FailingConnectClient,
                                "ClaudeAgentOptions": _DummyOptions,
                            },
                            model="MiniMax-RAN3",
                            system_prompt="system",
                            claude_session_id="claude-1",
                            resume_session_id="",
                            proxy_env={
                                "ANTHROPIC_BASE_URL": "http://127.0.0.1/internal/anthropic",
                                "ANTHROPIC_AUTH_TOKEN": "secret-token",
                            },
                            skill_mount_root=root,
                            mcp_servers={},
                            can_use_tool=None,
                            hooks={},
                        )

                message = str(raised.exception)
                self.assertIn("exit_code=127", message)
                self.assertIn("node: No such file or directory", message)
                self.assertIn("token=***", message)
                self.assertNotIn("secret-token", message)
                self.assertNotIn("secret-token", "\n".join(captured.output))

            asyncio.run(scenario())

    def test_connected_client_passes_configured_claude_sdk_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / ".claude"
            config_dir.mkdir(parents=True, exist_ok=True)
            workspace = root / "workspace"
            workspace.mkdir()
            workspace_extra = root / "workspace-extra"
            workspace_extra.mkdir()
            runtime = ClaudeSdkRuntime(
                ClaudeSettings(
                    workdir=root,
                    config_dir=config_dir,
                    default_model="MiniMax-RAN3",
                    permission_mode="acceptEdits",
                    cli_path="/usr/bin/claude",
                    setting_sources=["project", "local"],
                    skills_filter="all",
                    system_prompt_preset="claude_code",
                    system_prompt_append="",
                    system_prompt_file=None,
                    include_hook_events=True,
                    enable_file_checkpointing=True,
                    attachment_text_char_limit=96000,
                    tools={"type": "preset", "preset": "claude_code"},
                    allowed_tools=["Read"],
                    disallowed_tools=["Bash"],
                    strict_mcp_config=True,
                    continue_conversation=True,
                    max_turns=3,
                    max_budget_usd=0.5,
                    fallback_model="MiniMax-Fallback",
                    betas=["context-1m-2025-08-07"],
                    permission_prompt_tool_name="mcp__approval__ask",
                    settings=str(root / "flag-settings.json"),
                    add_dirs=[root / "extra"],
                    env={"CUSTOM_ENV": "custom"},
                    extra_args={"debug": "1"},
                    max_buffer_size=123,
                    user="user-1",
                    include_partial_messages=False,
                    fork_session=True,
                    agents={"reviewer": {"description": "Review code", "prompt": "Be strict", "max_turns": 2}},
                    sandbox={"enabled": False},
                    plugins=[{"type": "local", "path": "./plugins/demo"}],
                    max_thinking_tokens=4096,
                    thinking={"type": "adaptive"},
                    effort="high",
                    output_format={"type": "json_schema", "schema": {"type": "object"}},
                    session_store_flush="eager",
                    load_timeout_ms=1000,
                    task_budget={"total": 10000},
                ),
                McpSettings(config_dir=root / "mcps", auto_load=False),
                ProviderSettings(
                    base_url="http://127.0.0.1:18081",
                    anthropic_version="2023-06-01",
                    api_key="demo",
                    request_timeout_sec=30.0,
                ),
                SessionMappingStore(root / "data" / "sessions" / "session-map.json"),
                SessionCheckpointStore(root / "data" / "sessions" / "checkpoints.json"),
                ProxyContextStore(),
                client_pool=None,  # type: ignore[arg-type]
            )

            async def scenario() -> None:
                client, _ = await runtime._create_connected_client(
                    sdk={
                        "module": type("SdkModule", (), {"AgentDefinition": _DummyAgentDefinition}),
                        "ClaudeSDKClient": _DummyClient,
                        "ClaudeAgentOptions": _DummyOptions,
                    },
                    model="MiniMax-RAN3",
                    system_prompt="system",
                    claude_session_id="claude-1",
                    resume_session_id="",
                    proxy_env={"ANTHROPIC_BASE_URL": "http://127.0.0.1/internal/anthropic"},
                    skill_mount_root=root / "skill-mount",
                    workspace_cwd=workspace,
                    workspace_add_dirs=[workspace_extra],
                    mcp_servers={"demo": {"type": "http", "url": "http://127.0.0.1/mcp"}},
                    can_use_tool=None,
                    hooks={},
                )
                options = client.options.kwargs
                self.assertEqual(options["tools"], {"type": "preset", "preset": "claude_code"})
                self.assertEqual(options["allowed_tools"], ["Read"])
                self.assertEqual(options["disallowed_tools"], ["Bash"])
                self.assertTrue(options["strict_mcp_config"])
                self.assertTrue(options["continue_conversation"])
                self.assertNotIn("session_id", options)
                self.assertNotIn("resume", options)
                self.assertEqual(options["max_turns"], 3)
                self.assertEqual(options["max_budget_usd"], 0.5)
                self.assertEqual(options["fallback_model"], "MiniMax-Fallback")
                self.assertEqual(options["betas"], ["context-1m-2025-08-07"])
                self.assertEqual(options["permission_prompt_tool_name"], "mcp__approval__ask")
                self.assertEqual(options["settings"], str(root / "flag-settings.json"))
                self.assertEqual(options["cwd"], str(workspace))
                self.assertEqual(
                    options["add_dirs"],
                    [str(root / "extra"), str(workspace_extra), str(root / "skill-mount")],
                )
                self.assertEqual(options["env"]["CUSTOM_ENV"], "custom")
                self.assertEqual(options["env"]["ANTHROPIC_BASE_URL"], "http://127.0.0.1/internal/anthropic")
                self.assertIn("PATH", options["env"])
                self.assertIn("/usr/bin", options["env"]["PATH"].split(os.pathsep))
                self.assertEqual(options["extra_args"], {"debug": "1", "replay-user-messages": None})
                self.assertEqual(options["max_buffer_size"], 123)
                self.assertEqual(options["user"], "user-1")
                self.assertFalse(options["include_partial_messages"])
                self.assertTrue(options["fork_session"])
                self.assertIsInstance(options["agents"]["reviewer"], _DummyAgentDefinition)
                self.assertEqual(options["agents"]["reviewer"].maxTurns, 2)
                self.assertEqual(options["sandbox"], {"enabled": False})
                self.assertEqual(options["plugins"], [{"type": "local", "path": "./plugins/demo"}])
                self.assertEqual(options["max_thinking_tokens"], 4096)
                self.assertEqual(options["thinking"], {"type": "adaptive"})
                self.assertEqual(options["effort"], "high")
                self.assertEqual(options["output_format"], {"type": "json_schema", "schema": {"type": "object"}})
                self.assertTrue(options["enable_file_checkpointing"])
                self.assertEqual(options["session_store_flush"], "eager")
                self.assertEqual(options["load_timeout_ms"], 1000)
                self.assertEqual(options["task_budget"], {"total": 10000})

            asyncio.run(scenario())

    def test_create_connected_client_force_resume_ignores_continue_conversation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = ClaudeSdkRuntime(
                ClaudeSettings(
                    workdir=root,
                    config_dir=root / ".claude",
                    default_model="MiniMax-RAN3",
                    permission_mode="acceptEdits",
                    cli_path="/usr/bin/claude",
                    setting_sources=["project", "local"],
                    skills_filter="all",
                    system_prompt_preset="claude_code",
                    system_prompt_append="",
                    system_prompt_file=None,
                    include_hook_events=True,
                    enable_file_checkpointing=True,
                    attachment_text_char_limit=96000,
                    continue_conversation=True,
                ),
                McpSettings(config_dir=root / "mcps", auto_load=False),
                ProviderSettings(
                    base_url="http://127.0.0.1:18081",
                    anthropic_version="2023-06-01",
                    api_key="demo",
                    request_timeout_sec=30.0,
                ),
                SessionMappingStore(root / "data" / "sessions" / "session-map.json"),
                SessionCheckpointStore(root / "data" / "sessions" / "checkpoints.json"),
                ProxyContextStore(),
                client_pool=None,  # type: ignore[arg-type]
            )

            async def scenario() -> None:
                client, _ = await runtime._create_connected_client(
                    sdk={
                        "module": type("SdkModule", (), {"AgentDefinition": _DummyAgentDefinition}),
                        "ClaudeSDKClient": _DummyClient,
                        "ClaudeAgentOptions": _DummyOptions,
                    },
                    model="MiniMax-RAN3",
                    system_prompt="system",
                    claude_session_id="claude-1",
                    resume_session_id="claude-1",
                    force_resume=True,
                    proxy_env={},
                    skill_mount_root=root,
                    mcp_servers={},
                    can_use_tool=None,
                    hooks={},
                )
                options = client.options.kwargs
                self.assertFalse(options["continue_conversation"])
                self.assertEqual(options["resume"], "claude-1")
                self.assertNotIn("session_id", options)

            asyncio.run(scenario())

    def test_stream_events_sends_structured_user_message_to_official_sdk_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / ".claude"
            config_dir.mkdir(parents=True, exist_ok=True)
            workspace = root / "project-workspace"
            workspace.mkdir()
            workspace_extra = root / "project-extra"
            workspace_extra.mkdir()
            runtime = ClaudeSdkRuntime(
                ClaudeSettings(
                    workdir=root,
                    config_dir=config_dir,
                    default_model="MiniMax-RAN3",
                    permission_mode="acceptEdits",
                    cli_path="/usr/bin/claude",
                    setting_sources=["project", "local"],
                    skills_filter="all",
                    system_prompt_preset="claude_code",
                    system_prompt_append="",
                    system_prompt_file=None,
                    include_hook_events=False,
                    enable_file_checkpointing=False,
                    attachment_text_char_limit=96000,
                ),
                McpSettings(config_dir=root / "mcps", auto_load=False),
                ProviderSettings(
                    base_url="http://127.0.0.1:18081",
                    anthropic_version="2023-06-01",
                    api_key="demo",
                    request_timeout_sec=30.0,
                ),
                SessionMappingStore(root / "data" / "sessions" / "session-map.json"),
                SessionCheckpointStore(root / "data" / "sessions" / "checkpoints.json"),
                ProxyContextStore(),
                client_pool=None,  # type: ignore[arg-type]
            )
            runtime._client_pool = ClaudeClientPool()
            _ContextReportDummyClient.queried = False
            _ContextReportDummyClient.last_query_messages = []
            runtime._load_sdk = lambda: {  # type: ignore[method-assign]
                "module": type(
                    "SdkModule",
                    (),
                    {"PermissionResultAllow": _DummyAllow, "PermissionResultDeny": _DummyDeny},
                ),
                "ClaudeSDKClient": _QueryCapturingDummyClient,
                "ClaudeAgentOptions": _DummyOptions,
            }
            image_data_url = "data:image/png;base64,iVBORw0KGgo="

            async def scenario() -> None:
                payload = {
                    "session_id": "session-1",
                    "metadata": {
                        "agentconfig": {
                            "workspace": {
                                "cwd": str(workspace),
                                "add_dirs": [str(workspace_extra)],
                            }
                        }
                    },
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "analyze this image"},
                                {"type": "image_url", "image_url": {"url": image_data_url}},
                            ],
                        }
                    ],
                }
                collected = []
                async for event in runtime.stream_events(
                    payload,
                    request_headers={"x-user-id": "10154402"},
                    current_user=None,
                    proxy_base_url="http://127.0.0.1/internal/anthropic",
                    skill_mount_root=root,
                    run_id="run-1",
                ):
                    collected.append(event)
                self.assertEqual(collected, [])
                mapping = await runtime._session_store.get("session-1")
                self.assertIsNotNone(mapping)
                self.assertEqual(mapping.workspace_cwd, str(workspace.resolve()))
                self.assertEqual(mapping.workspace_add_dirs, [str(workspace_extra.resolve())])

            asyncio.run(scenario())

            self.assertEqual(len(_QueryCapturingDummyClient.last_query_messages), 1)
            message = _QueryCapturingDummyClient.last_query_messages[0]
            self.assertEqual(message["type"], "user")
            content = message["message"]["content"]
            self.assertEqual(content[0]["type"], "text")
            self.assertEqual(content[1]["type"], "image")
            self.assertEqual(content[1]["source"]["type"], "base64")
            self.assertEqual(_QueryCapturingDummyClient.last_options_kwargs["cwd"], str(workspace.resolve()))
            self.assertEqual(
                _QueryCapturingDummyClient.last_options_kwargs["add_dirs"],
                [str(workspace_extra.resolve()), str(root)],
            )

    def test_first_query_failure_replays_initial_history_on_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / ".claude"
            config_dir.mkdir(parents=True, exist_ok=True)
            runtime = ClaudeSdkRuntime(
                ClaudeSettings(
                    workdir=root,
                    config_dir=config_dir,
                    default_model="MiniMax-RAN3",
                    permission_mode="acceptEdits",
                    cli_path="/usr/bin/claude",
                    setting_sources=["project", "local"],
                    skills_filter="all",
                    system_prompt_preset="claude_code",
                    system_prompt_append="",
                    system_prompt_file=None,
                    include_hook_events=False,
                    enable_file_checkpointing=False,
                    attachment_text_char_limit=96000,
                ),
                McpSettings(config_dir=root / "mcps", auto_load=False),
                ProviderSettings(
                    base_url="http://127.0.0.1:18081",
                    anthropic_version="2023-06-01",
                    api_key="demo",
                    request_timeout_sec=30.0,
                ),
                SessionMappingStore(root / "data" / "sessions" / "session-map.json"),
                SessionCheckpointStore(root / "data" / "sessions" / "checkpoints.json"),
                ProxyContextStore(),
                client_pool=None,  # type: ignore[arg-type]
            )
            runtime._client_pool = ClaudeClientPool()
            _FirstQueryFailsDummyClient.query_attempts = 0
            _FirstQueryFailsDummyClient.query_messages_by_attempt = []
            _FirstQueryFailsDummyClient.disconnect_calls = 0
            runtime._load_sdk = lambda: {  # type: ignore[method-assign]
                "module": type(
                    "SdkModule",
                    (),
                    {"PermissionResultAllow": _DummyAllow, "PermissionResultDeny": _DummyDeny},
                ),
                "ClaudeSDKClient": _FirstQueryFailsDummyClient,
                "ClaudeAgentOptions": _DummyOptions,
            }

            async def scenario() -> None:
                payload = {
                    "session_id": "session-import-retry",
                    "messages": [
                        {"role": "user", "content": "old imported question"},
                        {"role": "assistant", "content": "old imported answer"},
                        {"role": "user", "content": "continue now"},
                    ],
                }
                with self.assertRaisesRegex(RuntimeError, "first query failed"):
                    async for _ in runtime.stream_events(
                        payload,
                        request_headers={"x-user-id": "10154402"},
                        current_user=None,
                        proxy_base_url="http://127.0.0.1/internal/anthropic",
                        skill_mount_root=root,
                        run_id="run-1",
                    ):
                        pass

                self.assertIsNone(await runtime._session_store.get("session-import-retry"))
                self.assertIsNone(await runtime._client_pool.get("session-import-retry"))

                async for _ in runtime.stream_events(
                    payload,
                    request_headers={"x-user-id": "10154402"},
                    current_user=None,
                    proxy_base_url="http://127.0.0.1/internal/anthropic",
                    skill_mount_root=root,
                    run_id="run-2",
                ):
                    pass

                self.assertIsNotNone(await runtime._session_store.get("session-import-retry"))

            asyncio.run(scenario())

            self.assertEqual(_FirstQueryFailsDummyClient.query_attempts, 2)
            self.assertEqual(_FirstQueryFailsDummyClient.disconnect_calls, 1)
            self.assertEqual(len(_FirstQueryFailsDummyClient.query_messages_by_attempt), 2)
            for messages in _FirstQueryFailsDummyClient.query_messages_by_attempt:
                rendered = repr(messages)
                self.assertIn("old imported question", rendered)
                self.assertIn("old imported answer", rendered)
                self.assertIn("continue now", rendered)

    def test_stream_events_uses_request_runtime_llm_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / ".claude"
            config_dir.mkdir(parents=True, exist_ok=True)
            proxy_store = ProxyContextStore()
            runtime = ClaudeSdkRuntime(
                ClaudeSettings(
                    workdir=root,
                    config_dir=config_dir,
                    default_model="MiniMax-RAN3",
                    permission_mode="acceptEdits",
                    cli_path="/usr/bin/claude",
                    setting_sources=["project", "local"],
                    skills_filter="all",
                    system_prompt_preset="claude_code",
                    system_prompt_append="",
                    system_prompt_file=None,
                    include_hook_events=False,
                    enable_file_checkpointing=False,
                    attachment_text_char_limit=96000,
                ),
                McpSettings(config_dir=root / "mcps", auto_load=False),
                ProviderSettings(
                    base_url="http://127.0.0.1:18081",
                    anthropic_version="2023-06-01",
                    api_key="service-token",
                    request_timeout_sec=30.0,
                ),
                SessionMappingStore(root / "data" / "sessions" / "session-map.json"),
                SessionCheckpointStore(root / "data" / "sessions" / "checkpoints.json"),
                proxy_store,
                client_pool=None,  # type: ignore[arg-type]
            )
            runtime._client_pool = ClaudeClientPool()
            _QueryCapturingDummyClient.last_query_messages = []
            _QueryCapturingDummyClient.last_options_kwargs = {}
            runtime._load_sdk = lambda: {  # type: ignore[method-assign]
                "module": type(
                    "SdkModule",
                    (),
                    {"PermissionResultAllow": _DummyAllow, "PermissionResultDeny": _DummyDeny},
                ),
                "ClaudeSDKClient": _QueryCapturingDummyClient,
                "ClaudeAgentOptions": _DummyOptions,
            }

            async def scenario() -> None:
                payload = {
                    "session_id": "session-llm",
                    "model": "pp-omni-glm-5.2",
                    "metadata": {
                        "agentconfig": {
                            "runtime_config": {
                                "llm": {
                                    "api_base": "https://wxai-icf.zx.zte.com.cn/v1/messages",
                                    "api_key": "runtime-token",
                                }
                            }
                        }
                    },
                    "messages": [{"role": "user", "content": "hello"}],
                }
                collected = []
                async for event in runtime.stream_events(
                    payload,
                    request_headers={
                        "x-user-id": "10154402",
                        "x-api-key": "request-header-token",
                        "uac-user-token": "uac-token",
                    },
                    current_user=None,
                    proxy_base_url="http://127.0.0.1/internal/anthropic",
                    skill_mount_root=root,
                    run_id="run-llm",
                ):
                    collected.append(event)
                self.assertEqual(collected, [])
                sdk_env = _QueryCapturingDummyClient.last_options_kwargs["env"]
                proxy_token = sdk_env["ANTHROPIC_AUTH_TOKEN"]
                self.assertEqual(proxy_token, sdk_env["ANTHROPIC_API_KEY"])
                ctx = await proxy_store.get(proxy_token)
                self.assertIsNotNone(ctx)
                self.assertEqual(ctx.upstream_base_url, "https://wxai-icf.zx.zte.com.cn")
                self.assertEqual(ctx.api_token, "runtime-token")
                self.assertEqual(ctx.model, "pp-omni-glm-5.2")
                self.assertEqual(sdk_env["VISION_MODEL_API_KEY"], "request-header-token")
                self.assertEqual(sdk_env["UAC_USER_TOKEN"], "uac-token")

                grok_payload = {
                    "session_id": "session-grok",
                    "model": "grok-4.5",
                    "metadata": {
                        "agentconfig": {
                            "runtime_config": {
                                "llm": {
                                    "base_url": "https://api.krill-ai.com/v1",
                                    "api_key": "grok-runtime-token",
                                }
                            }
                        }
                    },
                    "messages": [{"role": "user", "content": "hello"}],
                }
                grok_events = []
                async for event in runtime.stream_events(
                    grok_payload,
                    request_headers={
                        "x-user-id": "10154402",
                        "x-api-key": "request-header-token",
                        "uac-user-token": "uac-token",
                    },
                    current_user=None,
                    proxy_base_url="http://127.0.0.1/internal/anthropic",
                    skill_mount_root=root,
                    run_id="run-grok",
                ):
                    grok_events.append(event)
                self.assertEqual(grok_events, [])
                grok_sdk_env = _QueryCapturingDummyClient.last_options_kwargs["env"]
                grok_proxy_token = grok_sdk_env["ANTHROPIC_AUTH_TOKEN"]
                self.assertEqual(grok_proxy_token, grok_sdk_env["ANTHROPIC_API_KEY"])
                grok_ctx = await proxy_store.get(grok_proxy_token)
                self.assertIsNotNone(grok_ctx)
                self.assertEqual(grok_ctx.upstream_base_url, "https://api.krill-ai.com")
                self.assertEqual(grok_ctx.api_token, "grok-runtime-token")
                self.assertEqual(grok_ctx.model, "grok-4.5")
                self.assertEqual(grok_sdk_env["VISION_MODEL_API_KEY"], "request-header-token")
                self.assertEqual(grok_sdk_env["UAC_USER_TOKEN"], "uac-token")

            asyncio.run(scenario())

    def test_stream_events_does_not_inject_active_goal_into_system_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / ".claude"
            config_dir.mkdir(parents=True, exist_ok=True)
            goal_store = SessionGoalStore(root / "data" / "sessions" / "goals.json")
            asyncio.run(goal_store.set("session-1", "完成四个子任务并汇总"))
            runtime = ClaudeSdkRuntime(
                ClaudeSettings(
                    workdir=root,
                    config_dir=config_dir,
                    default_model="MiniMax-RAN3",
                    permission_mode="acceptEdits",
                    cli_path="/usr/bin/claude",
                    setting_sources=["project", "local"],
                    skills_filter="all",
                    system_prompt_preset="claude_code",
                    system_prompt_append="",
                    system_prompt_file=None,
                    include_hook_events=False,
                    enable_file_checkpointing=False,
                    attachment_text_char_limit=96000,
                ),
                McpSettings(config_dir=root / "mcps", auto_load=False),
                ProviderSettings(
                    base_url="http://127.0.0.1:18081",
                    anthropic_version="2023-06-01",
                    api_key="demo",
                    request_timeout_sec=30.0,
                ),
                SessionMappingStore(root / "data" / "sessions" / "session-map.json"),
                SessionCheckpointStore(root / "data" / "sessions" / "checkpoints.json"),
                ProxyContextStore(),
                client_pool=None,  # type: ignore[arg-type]
                goal_store=goal_store,
            )
            runtime._client_pool = ClaudeClientPool()
            _QueryCapturingDummyClient.last_query_messages = []
            _QueryCapturingDummyClient.last_options_kwargs = {}
            runtime._load_sdk = lambda: {  # type: ignore[method-assign]
                "module": type(
                    "SdkModule",
                    (),
                    {"PermissionResultAllow": _DummyAllow, "PermissionResultDeny": _DummyDeny},
                ),
                "ClaudeSDKClient": _QueryCapturingDummyClient,
                "ClaudeAgentOptions": _DummyOptions,
            }

            async def scenario() -> None:
                payload = {
                    "session_id": "session-1",
                    "messages": [{"role": "user", "content": "继续"}],
                }
                collected = []
                async for event in runtime.stream_events(
                    payload,
                    request_headers={"x-user-id": "10154402"},
                    current_user=None,
                    proxy_base_url="http://127.0.0.1/internal/anthropic",
                    skill_mount_root=root,
                    run_id="run-1",
                ):
                    collected.append(event)
                self.assertEqual([event.payload.get("phase") for event in collected], ["goal_recovering", "goal_resumed"])

            asyncio.run(scenario())

            system_prompt = _QueryCapturingDummyClient.last_options_kwargs["system_prompt"]
            self.assertEqual(system_prompt["type"], "preset")
            self.assertNotIn("当前会话处于本地 goal 模式", system_prompt.get("append", ""))
            self.assertNotIn("完成四个子任务并汇总", system_prompt.get("append", ""))
            goal = asyncio.run(goal_store.get("session-1"))
            self.assertIsNotNone(goal)
            self.assertEqual(goal.status, "running")

    def test_stream_events_sets_goal_then_executes_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / ".claude"
            config_dir.mkdir(parents=True, exist_ok=True)
            runtime = ClaudeSdkRuntime(
                ClaudeSettings(
                    workdir=root,
                    config_dir=config_dir,
                    default_model="MiniMax-RAN3",
                    permission_mode="acceptEdits",
                    cli_path="/usr/bin/claude",
                    setting_sources=["project", "local"],
                    skills_filter="all",
                    system_prompt_preset="claude_code",
                    system_prompt_append="",
                    system_prompt_file=None,
                    include_hook_events=False,
                    enable_file_checkpointing=False,
                    attachment_text_char_limit=96000,
                ),
                McpSettings(config_dir=root / "mcps", auto_load=False),
                ProviderSettings(
                    base_url="http://127.0.0.1:18081",
                    anthropic_version="2023-06-01",
                    api_key="demo",
                    request_timeout_sec=30.0,
                ),
                SessionMappingStore(root / "data" / "sessions" / "session-map.json"),
                SessionCheckpointStore(root / "data" / "sessions" / "checkpoints.json"),
                ProxyContextStore(),
                client_pool=None,  # type: ignore[arg-type]
                goal_store=SessionGoalStore(root / "data" / "sessions" / "goals.json"),
            )
            runtime._client_pool = ClaudeClientPool()
            _GoalSetDummyClient.last_query_messages = []
            runtime._load_sdk = lambda: {  # type: ignore[method-assign]
                "module": type(
                    "SdkModule",
                    (),
                    {"PermissionResultAllow": _DummyAllow, "PermissionResultDeny": _DummyDeny},
                ),
                "ClaudeSDKClient": _GoalSetDummyClient,
                "ClaudeAgentOptions": _DummyOptions,
            }

            async def scenario() -> list[RuntimeStreamEvent]:
                payload = {
                    "session_id": "session-1",
                    "metadata": {
                        "runtimeCommand": {
                            "source": "claude-code",
                            "commandId": "goal",
                            "command": "/goal",
                            "args": {"text": "做完 Runtime Command P0"},
                            "displayName": "目标模式",
                            "requestId": "cmd-goal-1",
                        }
                    },
                    "messages": [
                        {"role": "user", "content": "/goal 做完 Runtime Command P0"},
                    ],
                }
                collected = []
                async for event in runtime.stream_events(
                    payload,
                    request_headers={"x-user-id": "10154402"},
                    current_user=None,
                    proxy_base_url="http://127.0.0.1/internal/anthropic",
                    skill_mount_root=root,
                    run_id="run-1",
                ):
                    collected.append(event)
                return collected

            events = asyncio.run(scenario())

            self.assertEqual(len(_GoalSetDummyClient.last_query_messages), 1)
            message = _GoalSetDummyClient.last_query_messages[0]
            self.assertEqual(message["type"], "user")
            self.assertEqual(message["message"]["content"], [{"type": "text", "text": "/goal 做完 Runtime Command P0"}])
            self.assertEqual([event.kind for event in events], ["command", "text", "command"])
            self.assertEqual(events[0].payload["status"], "running")
            self.assertIn("Goal set: 做完 Runtime Command P0", events[1].text)
            self.assertEqual(events[2].payload["status"], "completed")
            self.assertEqual(events[2].payload["phase"], "goal_set")
            self.assertEqual(events[2].payload["result"], "goal_set")
            goal = asyncio.run(runtime._goal_store.get("session-1"))
            self.assertIsNotNone(goal)
            self.assertEqual(goal.objective, "做完 Runtime Command P0")
            self.assertEqual(goal.status, "running")

    def test_stream_events_marks_goal_completed_from_stop_hook(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / ".claude"
            config_dir.mkdir(parents=True, exist_ok=True)
            goal_store = SessionGoalStore(root / "data" / "sessions" / "goals.json")
            runtime = ClaudeSdkRuntime(
                ClaudeSettings(
                    workdir=root,
                    config_dir=config_dir,
                    default_model="MiniMax-RAN3",
                    permission_mode="acceptEdits",
                    cli_path="/usr/bin/claude",
                    setting_sources=["project", "local"],
                    skills_filter="all",
                    system_prompt_preset="claude_code",
                    system_prompt_append="",
                    system_prompt_file=None,
                    include_hook_events=False,
                    enable_file_checkpointing=False,
                    attachment_text_char_limit=96000,
                ),
                McpSettings(config_dir=root / "mcps", auto_load=False),
                ProviderSettings(
                    base_url="http://127.0.0.1:18081",
                    anthropic_version="2023-06-01",
                    api_key="demo",
                    request_timeout_sec=30.0,
                ),
                SessionMappingStore(root / "data" / "sessions" / "session-map.json"),
                SessionCheckpointStore(root / "data" / "sessions" / "checkpoints.json"),
                ProxyContextStore(),
                client_pool=None,  # type: ignore[arg-type]
                goal_store=goal_store,
            )
            runtime._client_pool = ClaudeClientPool()
            runtime._load_sdk = lambda: {  # type: ignore[method-assign]
                "module": type(
                    "SdkModule",
                    (),
                    {"PermissionResultAllow": _DummyAllow, "PermissionResultDeny": _DummyDeny},
                ),
                "ClaudeSDKClient": _GoalStopHookDummyClient,
                "ClaudeAgentOptions": _DummyOptions,
            }

            async def scenario() -> list[RuntimeStreamEvent]:
                payload = {
                    "session_id": "session-1",
                    "metadata": {
                        "runtimeCommand": {
                            "source": "claude-code",
                            "commandId": "goal",
                            "command": "/goal",
                            "args": {"text": "做完 Runtime Command P0"},
                            "displayName": "目标模式",
                            "requestId": "cmd-goal-1",
                        }
                    },
                    "messages": [{"role": "user", "content": "/goal 做完 Runtime Command P0"}],
                }
                collected = []
                async for event in runtime.stream_events(
                    payload,
                    request_headers={"x-user-id": "10154402"},
                    current_user=None,
                    proxy_base_url="http://127.0.0.1/internal/anthropic",
                    skill_mount_root=root,
                    run_id="run-1",
                ):
                    collected.append(event)
                return collected

            events = asyncio.run(scenario())

            self.assertEqual(events[-1].payload["phase"], "goal_completed")
            self.assertEqual(events[-1].payload["result"], "goal_completed")
            goal = asyncio.run(goal_store.get("session-1"))
            self.assertIsNotNone(goal)
            self.assertEqual(goal.status, "completed")
            self.assertEqual(goal.last_run_id, "run-1")
            self.assertIn("Stop hook ok=true", goal.last_summary)

    def test_stream_events_marks_goal_completed_from_noisy_stop_hook_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / ".claude"
            config_dir.mkdir(parents=True, exist_ok=True)
            goal_store = SessionGoalStore(root / "data" / "sessions" / "goals.json")
            runtime = ClaudeSdkRuntime(
                ClaudeSettings(
                    workdir=root,
                    config_dir=config_dir,
                    default_model="MiniMax-RAN3",
                    permission_mode="acceptEdits",
                    cli_path="/usr/bin/claude",
                    setting_sources=["project", "local"],
                    skills_filter="all",
                    system_prompt_preset="claude_code",
                    system_prompt_append="",
                    system_prompt_file=None,
                    include_hook_events=False,
                    enable_file_checkpointing=False,
                    attachment_text_char_limit=96000,
                ),
                McpSettings(config_dir=root / "mcps", auto_load=False),
                ProviderSettings(
                    base_url="http://127.0.0.1:18081",
                    anthropic_version="2023-06-01",
                    api_key="demo",
                    request_timeout_sec=30.0,
                ),
                SessionMappingStore(root / "data" / "sessions" / "session-map.json"),
                SessionCheckpointStore(root / "data" / "sessions" / "checkpoints.json"),
                ProxyContextStore(),
                client_pool=None,  # type: ignore[arg-type]
                goal_store=goal_store,
            )
            runtime._client_pool = ClaudeClientPool()
            runtime._load_sdk = lambda: {  # type: ignore[method-assign]
                "module": type(
                    "SdkModule",
                    (),
                    {"PermissionResultAllow": _DummyAllow, "PermissionResultDeny": _DummyDeny},
                ),
                "ClaudeSDKClient": _GoalStopHookJsonNoiseDummyClient,
                "ClaudeAgentOptions": _DummyOptions,
            }

            async def scenario() -> list[RuntimeStreamEvent]:
                payload = {
                    "session_id": "session-1",
                    "metadata": {
                        "runtimeCommand": {
                            "source": "claude-code",
                            "commandId": "goal",
                            "command": "/goal",
                            "args": {"text": "做完 Runtime Command P0"},
                            "displayName": "目标模式",
                            "requestId": "cmd-goal-1",
                        }
                    },
                    "messages": [{"role": "user", "content": "/goal 做完 Runtime Command P0"}],
                }
                collected = []
                async for event in runtime.stream_events(
                    payload,
                    request_headers={"x-user-id": "10154402"},
                    current_user=None,
                    proxy_base_url="http://127.0.0.1/internal/anthropic",
                    skill_mount_root=root,
                    run_id="run-1",
                ):
                    collected.append(event)
                return collected

            events = asyncio.run(scenario())

            self.assertEqual(events[-1].payload["phase"], "goal_completed")
            goal = asyncio.run(goal_store.get("session-1"))
            self.assertIsNotNone(goal)
            self.assertEqual(goal.status, "completed")
            self.assertIn("目标条件已满足", goal.last_summary)

    def test_stream_events_requests_summary_after_background_tasks_finish(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / ".claude"
            config_dir.mkdir(parents=True, exist_ok=True)
            runtime = ClaudeSdkRuntime(
                ClaudeSettings(
                    workdir=root,
                    config_dir=config_dir,
                    default_model="MiniMax-RAN3",
                    permission_mode="acceptEdits",
                    cli_path="/usr/bin/claude",
                    setting_sources=["project", "local"],
                    skills_filter="all",
                    system_prompt_preset="claude_code",
                    system_prompt_append="",
                    system_prompt_file=None,
                    include_hook_events=False,
                    enable_file_checkpointing=False,
                    attachment_text_char_limit=96000,
                ),
                McpSettings(config_dir=root / "mcps", auto_load=False),
                ProviderSettings(
                    base_url="http://127.0.0.1:18081",
                    anthropic_version="2023-06-01",
                    api_key="demo",
                    request_timeout_sec=30.0,
                ),
                SessionMappingStore(root / "data" / "sessions" / "session-map.json"),
                SessionCheckpointStore(root / "data" / "sessions" / "checkpoints.json"),
                ProxyContextStore(),
                client_pool=None,  # type: ignore[arg-type]
            )
            runtime._client_pool = ClaudeClientPool()
            _SubtaskSummaryDummyClient.query_messages = []
            runtime._load_sdk = lambda: {  # type: ignore[method-assign]
                "module": type(
                    "SdkModule",
                    (),
                    {"PermissionResultAllow": _DummyAllow, "PermissionResultDeny": _DummyDeny},
                ),
                "ClaudeSDKClient": _SubtaskSummaryDummyClient,
                "ClaudeAgentOptions": _DummyOptions,
            }

            async def scenario():
                payload = {
                    "session_id": "session-1",
                    "messages": [{"role": "user", "content": "检查 BFF 并汇总"}],
                }
                collected = []
                async for event in runtime.stream_events(
                    payload,
                    request_headers={"x-user-id": "10154402"},
                    current_user=None,
                    proxy_base_url="http://127.0.0.1/internal/anthropic",
                    skill_mount_root=root,
                    run_id="run-1",
                ):
                    collected.append(event)
                return collected

            events = asyncio.run(scenario())

            self.assertEqual(len(_SubtaskSummaryDummyClient.query_messages), 2)
            summary_prompt = _SubtaskSummaryDummyClient.query_messages[1][0]["message"]["content"][0]["text"]
            self.assertIn("后台子任务已经全部结束", summary_prompt)
            self.assertIn("请不要再启动新的子任务", summary_prompt)
            self.assertIn("最终汇总：BFF 检查完成。", "".join(event.text or "" for event in events))
            task_events = [event for event in events if event.kind == "task"]
            self.assertTrue(any(event.payload.get("phase") == "waiting_subtasks" for event in task_events))
            self.assertTrue(any(event.payload.get("status") == "completed" for event in task_events))

    def test_stream_events_waits_for_receive_messages_task_updates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / ".claude"
            config_dir.mkdir(parents=True, exist_ok=True)
            runtime = ClaudeSdkRuntime(
                ClaudeSettings(
                    workdir=root,
                    config_dir=config_dir,
                    default_model="MiniMax-RAN3",
                    permission_mode="acceptEdits",
                    cli_path="/usr/bin/claude",
                    setting_sources=["project", "local"],
                    skills_filter="all",
                    system_prompt_preset="claude_code",
                    system_prompt_append="",
                    system_prompt_file=None,
                    include_hook_events=False,
                    enable_file_checkpointing=False,
                    attachment_text_char_limit=96000,
                ),
                McpSettings(config_dir=root / "mcps", auto_load=False),
                ProviderSettings(
                    base_url="http://127.0.0.1:18081",
                    anthropic_version="2023-06-01",
                    api_key="demo",
                    request_timeout_sec=30.0,
                ),
                SessionMappingStore(root / "data" / "sessions" / "session-map.json"),
                SessionCheckpointStore(root / "data" / "sessions" / "checkpoints.json"),
                ProxyContextStore(),
                client_pool=None,  # type: ignore[arg-type]
            )
            runtime._client_pool = ClaudeClientPool()
            _SubtaskReceiveMessagesDummyClient.query_messages = []
            runtime._load_sdk = lambda: {  # type: ignore[method-assign]
                "module": type(
                    "SdkModule",
                    (),
                    {"PermissionResultAllow": _DummyAllow, "PermissionResultDeny": _DummyDeny},
                ),
                "ClaudeSDKClient": _SubtaskReceiveMessagesDummyClient,
                "ClaudeAgentOptions": _DummyOptions,
            }

            async def scenario():
                payload = {
                    "session_id": "session-1",
                    "messages": [{"role": "user", "content": "检查 UI 并汇总"}],
                }
                collected = []
                async for event in runtime.stream_events(
                    payload,
                    request_headers={"x-user-id": "10154402"},
                    current_user=None,
                    proxy_base_url="http://127.0.0.1/internal/anthropic",
                    skill_mount_root=root,
                    run_id="run-1",
                ):
                    collected.append(event)
                return collected

            events = asyncio.run(scenario())

            self.assertEqual(len(_SubtaskReceiveMessagesDummyClient.query_messages), 2)
            self.assertIn("最终汇总：UI 检查完成。", "".join(event.text or "" for event in events))
            task_events = [event for event in events if event.kind == "task"]
            self.assertTrue(any(event.payload.get("phase") == "waiting_subtasks" for event in task_events))
            self.assertTrue(any(event.payload.get("status") == "completed" for event in task_events))

    def test_stream_events_pauses_goal_when_background_tasks_time_out(self) -> None:
        old_timeout = runtime_module._SUBTASK_WAIT_TIMEOUT_SECONDS
        old_poll = runtime_module._SUBTASK_WAIT_POLL_SECONDS
        old_heartbeat = runtime_module._SUBTASK_WAIT_HEARTBEAT_SECONDS
        runtime_module._SUBTASK_WAIT_TIMEOUT_SECONDS = 0.05
        runtime_module._SUBTASK_WAIT_POLL_SECONDS = 0.01
        runtime_module._SUBTASK_WAIT_HEARTBEAT_SECONDS = 0.02
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config_dir = root / ".claude"
                config_dir.mkdir(parents=True, exist_ok=True)
                goal_store = SessionGoalStore(root / "data" / "sessions" / "goals.json")
                asyncio.run(goal_store.set("session-1", "等待后台子任务完成并汇总"))
                runtime = ClaudeSdkRuntime(
                    ClaudeSettings(
                        workdir=root,
                        config_dir=config_dir,
                        default_model="MiniMax-RAN3",
                        permission_mode="acceptEdits",
                        cli_path="/usr/bin/claude",
                        setting_sources=["project", "local"],
                        skills_filter="all",
                        system_prompt_preset="claude_code",
                        system_prompt_append="",
                        system_prompt_file=None,
                        include_hook_events=False,
                        enable_file_checkpointing=False,
                        attachment_text_char_limit=96000,
                    ),
                    McpSettings(config_dir=root / "mcps", auto_load=False),
                    ProviderSettings(
                        base_url="http://127.0.0.1:18081",
                        anthropic_version="2023-06-01",
                        api_key="demo",
                        request_timeout_sec=30.0,
                    ),
                    SessionMappingStore(root / "data" / "sessions" / "session-map.json"),
                    SessionCheckpointStore(root / "data" / "sessions" / "checkpoints.json"),
                    ProxyContextStore(),
                    client_pool=None,  # type: ignore[arg-type]
                    goal_store=goal_store,
                )
                runtime._client_pool = ClaudeClientPool()
                _SubtaskNeverFinishesDummyClient.query_messages = []
                runtime._load_sdk = lambda: {  # type: ignore[method-assign]
                    "module": type(
                        "SdkModule",
                        (),
                        {"PermissionResultAllow": _DummyAllow, "PermissionResultDeny": _DummyDeny},
                    ),
                    "ClaudeSDKClient": _SubtaskNeverFinishesDummyClient,
                    "ClaudeAgentOptions": _DummyOptions,
                }

                async def scenario():
                    payload = {
                        "session_id": "session-1",
                        "messages": [{"role": "user", "content": "继续"}],
                    }
                    collected = []
                    async for event in runtime.stream_events(
                        payload,
                        request_headers={"x-user-id": "10154402"},
                        current_user=None,
                        proxy_base_url="http://127.0.0.1/internal/anthropic",
                        skill_mount_root=root,
                        run_id="run-1",
                    ):
                        collected.append(event)
                    return collected

                events = asyncio.run(scenario())

                self.assertIn("当前回复不是最终汇总", "".join(event.text or "" for event in events))
                self.assertTrue(
                    any(
                        event.kind == "command"
                        and event.payload.get("phase") == "goal_paused"
                        and event.payload.get("result") == "subtask_wait_timeout"
                        for event in events
                    )
                )
                goal = asyncio.run(goal_store.get("session-1"))
                self.assertIsNotNone(goal)
                self.assertEqual(goal.status, "paused")
                self.assertEqual(goal.pause_reason, "subtask_wait_timeout")
        finally:
            runtime_module._SUBTASK_WAIT_TIMEOUT_SECONDS = old_timeout
            runtime_module._SUBTASK_WAIT_POLL_SECONDS = old_poll
            runtime_module._SUBTASK_WAIT_HEARTBEAT_SECONDS = old_heartbeat

    def test_stream_events_waits_for_task_updates_after_result_message(self) -> None:
        old_poll = runtime_module._SUBTASK_WAIT_POLL_SECONDS
        old_heartbeat = runtime_module._SUBTASK_WAIT_HEARTBEAT_SECONDS
        old_timeout = runtime_module._SUBTASK_WAIT_TIMEOUT_SECONDS
        runtime_module._SUBTASK_WAIT_POLL_SECONDS = 0.01
        runtime_module._SUBTASK_WAIT_HEARTBEAT_SECONDS = 0.01
        runtime_module._SUBTASK_WAIT_TIMEOUT_SECONDS = 1.0
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config_dir = root / ".claude"
                config_dir.mkdir(parents=True, exist_ok=True)
                runtime = ClaudeSdkRuntime(
                    ClaudeSettings(
                        workdir=root,
                        config_dir=config_dir,
                        default_model="MiniMax-RAN3",
                        permission_mode="acceptEdits",
                        cli_path="/usr/bin/claude",
                        setting_sources=["project", "local"],
                        skills_filter="all",
                        system_prompt_preset="claude_code",
                        system_prompt_append="",
                        system_prompt_file=None,
                        include_hook_events=False,
                        enable_file_checkpointing=False,
                        attachment_text_char_limit=96000,
                    ),
                    McpSettings(config_dir=root / "mcps", auto_load=False),
                    ProviderSettings(
                        base_url="http://127.0.0.1:18081",
                        anthropic_version="2023-06-01",
                        api_key="demo",
                        request_timeout_sec=30.0,
                    ),
                    SessionMappingStore(root / "data" / "sessions" / "session-map.json"),
                    SessionCheckpointStore(root / "data" / "sessions" / "checkpoints.json"),
                    ProxyContextStore(),
                    client_pool=None,  # type: ignore[arg-type]
                )
                runtime._client_pool = ClaudeClientPool()
                _SubtaskReceiveMessagesDummyClient.query_messages = []
                runtime._load_sdk = lambda: {  # type: ignore[method-assign]
                    "module": type(
                        "SdkModule",
                        (),
                        {"PermissionResultAllow": _DummyAllow, "PermissionResultDeny": _DummyDeny},
                    ),
                    "ClaudeSDKClient": _SubtaskReceiveMessagesDummyClient,
                    "ClaudeAgentOptions": _DummyOptions,
                }

                async def scenario():
                    payload = {
                        "session_id": "session-1",
                        "messages": [{"role": "user", "content": "检查 UI 并汇总"}],
                    }
                    collected = []
                    async for event in runtime.stream_events(
                        payload,
                        request_headers={"x-user-id": "10154402"},
                        current_user=None,
                        proxy_base_url="http://127.0.0.1/internal/anthropic",
                        skill_mount_root=root,
                        run_id="run-1",
                    ):
                        collected.append(event)
                    return collected

                events = asyncio.run(scenario())
        finally:
            runtime_module._SUBTASK_WAIT_POLL_SECONDS = old_poll
            runtime_module._SUBTASK_WAIT_HEARTBEAT_SECONDS = old_heartbeat
            runtime_module._SUBTASK_WAIT_TIMEOUT_SECONDS = old_timeout

        self.assertEqual(len(_SubtaskReceiveMessagesDummyClient.query_messages), 2)
        self.assertIn("最终汇总：UI 检查完成。", "".join(event.text or "" for event in events))
        task_events = [event for event in events if event.kind == "task"]
        self.assertTrue(
            any(
                event.payload.get("log") == "主响应流已结束，仍在等待后台子任务完成。"
                for event in task_events
            )
        )
        self.assertTrue(any(event.payload.get("status") == "completed" for event in task_events))

    def test_stream_events_emits_goal_status_from_local_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / ".claude"
            config_dir.mkdir(parents=True, exist_ok=True)
            goal_store = SessionGoalStore(root / "data" / "sessions" / "goals.json")
            asyncio.run(goal_store.set("session-1", "完成并汇总子任务"))
            runtime = ClaudeSdkRuntime(
                ClaudeSettings(
                    workdir=root,
                    config_dir=config_dir,
                    default_model="MiniMax-RAN3",
                    permission_mode="acceptEdits",
                    cli_path="/usr/bin/claude",
                    setting_sources=["project", "local"],
                    skills_filter="all",
                    system_prompt_preset="claude_code",
                    system_prompt_append="",
                    system_prompt_file=None,
                    include_hook_events=False,
                    enable_file_checkpointing=False,
                    attachment_text_char_limit=96000,
                ),
                McpSettings(config_dir=root / "mcps", auto_load=False),
                ProviderSettings(
                    base_url="http://127.0.0.1:18081",
                    anthropic_version="2023-06-01",
                    api_key="demo",
                    request_timeout_sec=30.0,
                ),
                SessionMappingStore(root / "data" / "sessions" / "session-map.json"),
                SessionCheckpointStore(root / "data" / "sessions" / "checkpoints.json"),
                ProxyContextStore(),
                client_pool=None,  # type: ignore[arg-type]
                goal_store=goal_store,
            )
            runtime._client_pool = ClaudeClientPool()
            _GoalSetDummyClient.last_query_messages = []

            async def scenario() -> list[RuntimeStreamEvent]:
                payload = {
                    "session_id": "session-1",
                    "messages": [{"role": "user", "content": "/goal"}],
                }
                collected = []
                async for event in runtime.stream_events(
                    payload,
                    request_headers={"x-user-id": "10154402"},
                    current_user=None,
                    proxy_base_url="http://127.0.0.1/internal/anthropic",
                    skill_mount_root=root,
                    run_id="run-1",
                ):
                    collected.append(event)
                return collected

            events = asyncio.run(scenario())

            self.assertEqual(_GoalSetDummyClient.last_query_messages, [])
            self.assertEqual([event.kind for event in events], ["command", "text", "command"])
            self.assertIn("当前目标：完成并汇总子任务", events[1].text)
            self.assertEqual(events[2].payload["phase"], "goal_status")

    def test_interrupt_marks_running_goal_paused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / ".claude"
            config_dir.mkdir(parents=True, exist_ok=True)
            goal_store = SessionGoalStore(root / "data" / "sessions" / "goals.json")
            asyncio.run(goal_store.set("session-1", "持续完成目标"))
            runtime = ClaudeSdkRuntime(
                ClaudeSettings(
                    workdir=root,
                    config_dir=config_dir,
                    default_model="MiniMax-RAN3",
                    permission_mode="acceptEdits",
                    cli_path="/usr/bin/claude",
                    setting_sources=["project", "local"],
                    skills_filter="all",
                    system_prompt_preset="claude_code",
                    system_prompt_append="",
                    system_prompt_file=None,
                    include_hook_events=False,
                    enable_file_checkpointing=False,
                    attachment_text_char_limit=96000,
                ),
                McpSettings(config_dir=root / "mcps", auto_load=False),
                ProviderSettings(
                    base_url="http://127.0.0.1:18081",
                    anthropic_version="2023-06-01",
                    api_key="demo",
                    request_timeout_sec=30.0,
                ),
                SessionMappingStore(root / "data" / "sessions" / "session-map.json"),
                SessionCheckpointStore(root / "data" / "sessions" / "checkpoints.json"),
                ProxyContextStore(),
                ClaudeClientPool(),
                goal_store=goal_store,
            )

            class InterruptClient:
                def __init__(self) -> None:
                    self.interrupted = False
                    self.disconnected = False

                async def interrupt(self) -> None:
                    self.interrupted = True

                async def disconnect(self) -> None:
                    self.disconnected = True

            async def scenario() -> None:
                client = InterruptClient()
                async def factory():
                    return client, None

                await runtime._client_pool.get_or_create(
                    "session-1",
                    claude_session_id="claude-session-1",
                    model="MiniMax-RAN3",
                    resumed=False,
                    signature="sig",
                    factory=factory,
                )
                self.assertTrue(await runtime.interrupt_session("session-1"))
                self.assertTrue(client.interrupted)
                self.assertTrue(client.disconnected)
                self.assertIsNone(await runtime._client_pool.get("session-1"))

            asyncio.run(scenario())
            goal = asyncio.run(goal_store.get("session-1"))
            self.assertIsNotNone(goal)
            self.assertEqual(goal.status, "paused")
            self.assertEqual(goal.pause_reason, "user_interrupt")

    def test_interrupt_pauses_running_goal_without_connected_client(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / ".claude"
            config_dir.mkdir(parents=True, exist_ok=True)
            goal_store = SessionGoalStore(root / "data" / "sessions" / "goals.json")
            asyncio.run(goal_store.set("session-1", "持续完成目标"))
            runtime = ClaudeSdkRuntime(
                ClaudeSettings(
                    workdir=root,
                    config_dir=config_dir,
                    default_model="MiniMax-RAN3",
                    permission_mode="acceptEdits",
                    cli_path="/usr/bin/claude",
                    setting_sources=["project", "local"],
                    skills_filter="all",
                    system_prompt_preset="claude_code",
                    system_prompt_append="",
                    system_prompt_file=None,
                    include_hook_events=False,
                    enable_file_checkpointing=False,
                    attachment_text_char_limit=96000,
                ),
                McpSettings(config_dir=root / "mcps", auto_load=False),
                ProviderSettings(
                    base_url="http://127.0.0.1:18081",
                    anthropic_version="2023-06-01",
                    api_key="demo",
                    request_timeout_sec=30.0,
                ),
                SessionMappingStore(root / "data" / "sessions" / "session-map.json"),
                SessionCheckpointStore(root / "data" / "sessions" / "checkpoints.json"),
                ProxyContextStore(),
                ClaudeClientPool(),
                goal_store=goal_store,
            )

            self.assertTrue(asyncio.run(runtime.interrupt_session("session-1")))
            goal = asyncio.run(goal_store.get("session-1"))
            self.assertIsNotNone(goal)
            self.assertEqual(goal.status, "paused")
            self.assertEqual(goal.pause_reason, "user_interrupt")

    def test_interrupt_unknown_session_without_client_returns_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / ".claude"
            config_dir.mkdir(parents=True, exist_ok=True)
            runtime = ClaudeSdkRuntime(
                ClaudeSettings(
                    workdir=root,
                    config_dir=config_dir,
                    default_model="MiniMax-RAN3",
                    permission_mode="acceptEdits",
                    cli_path="/usr/bin/claude",
                    setting_sources=["project", "local"],
                    skills_filter="all",
                    system_prompt_preset="claude_code",
                    system_prompt_append="",
                    system_prompt_file=None,
                    include_hook_events=False,
                    enable_file_checkpointing=False,
                    attachment_text_char_limit=96000,
                ),
                McpSettings(config_dir=root / "mcps", auto_load=False),
                ProviderSettings(
                    base_url="http://127.0.0.1:18081",
                    anthropic_version="2023-06-01",
                    api_key="demo",
                    request_timeout_sec=30.0,
                ),
                SessionMappingStore(root / "data" / "sessions" / "session-map.json"),
                SessionCheckpointStore(root / "data" / "sessions" / "checkpoints.json"),
                ProxyContextStore(),
                ClaudeClientPool(),
                goal_store=SessionGoalStore(root / "data" / "sessions" / "goals.json"),
            )

            self.assertFalse(asyncio.run(runtime.interrupt_session("missing-session")))

    def test_runtime_snapshot_marks_running_goal_without_client_paused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / ".claude"
            config_dir.mkdir(parents=True, exist_ok=True)
            goal_store = SessionGoalStore(root / "data" / "sessions" / "goals.json")
            asyncio.run(goal_store.set("session-1", "持续完成目标"))
            asyncio.run(goal_store.mark_run_started("session-1", "stale-run"))
            runtime = ClaudeSdkRuntime(
                ClaudeSettings(
                    workdir=root,
                    config_dir=config_dir,
                    default_model="MiniMax-RAN3",
                    permission_mode="acceptEdits",
                    cli_path="/usr/bin/claude",
                    setting_sources=["project", "local"],
                    skills_filter="all",
                    system_prompt_preset="claude_code",
                    system_prompt_append="",
                    system_prompt_file=None,
                    include_hook_events=False,
                    enable_file_checkpointing=False,
                    attachment_text_char_limit=96000,
                ),
                McpSettings(config_dir=root / "mcps", auto_load=False),
                ProviderSettings(
                    base_url="http://127.0.0.1:18081",
                    anthropic_version="2023-06-01",
                    api_key="demo",
                    request_timeout_sec=30.0,
                ),
                SessionMappingStore(root / "data" / "sessions" / "session-map.json"),
                SessionCheckpointStore(root / "data" / "sessions" / "checkpoints.json"),
                ProxyContextStore(),
                ClaudeClientPool(),
                goal_store=goal_store,
            )

            snapshot = asyncio.run(runtime.runtime_snapshot())
            goal_runtime = snapshot["goalRuntime"]
            self.assertEqual(goal_runtime["activeGoalNum"], 0)
            self.assertEqual(goal_runtime["sessions"][0]["status"], "paused")
            self.assertEqual(goal_runtime["sessions"][0]["activeRunId"], "")
            self.assertEqual(goal_runtime["sessions"][0]["pauseReason"], "process_interrupted")

    def test_paused_goal_next_message_resumes_same_session_without_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / ".claude"
            config_dir.mkdir(parents=True, exist_ok=True)
            goal_store = SessionGoalStore(root / "data" / "sessions" / "goals.json")
            asyncio.run(goal_store.set("session-1", "持续完成目标"))
            asyncio.run(goal_store.pause("session-1", reason="user_interrupt"))
            asyncio.run(SessionMappingStore(root / "data" / "sessions" / "session-map.json").put("session-1", "claude-session-1", model="MiniMax-RAN3"))
            runtime = ClaudeSdkRuntime(
                ClaudeSettings(
                    workdir=root,
                    config_dir=config_dir,
                    default_model="MiniMax-RAN3",
                    permission_mode="acceptEdits",
                    cli_path="/usr/bin/claude",
                    setting_sources=["project", "local"],
                    skills_filter="all",
                    system_prompt_preset="claude_code",
                    system_prompt_append="",
                    system_prompt_file=None,
                    include_hook_events=False,
                    enable_file_checkpointing=False,
                    attachment_text_char_limit=96000,
                ),
                McpSettings(config_dir=root / "mcps", auto_load=False),
                ProviderSettings(
                    base_url="http://127.0.0.1:18081",
                    anthropic_version="2023-06-01",
                    api_key="demo",
                    request_timeout_sec=30.0,
                ),
                SessionMappingStore(root / "data" / "sessions" / "session-map.json"),
                SessionCheckpointStore(root / "data" / "sessions" / "checkpoints.json"),
                ProxyContextStore(),
                client_pool=None,  # type: ignore[arg-type]
                question_registry=QuestionRuntimeRegistry(),
                goal_store=goal_store,
            )
            runtime._client_pool = ClaudeClientPool()
            _GoalRecoveryDummyClient.query_messages = []
            _GoalRecoveryDummyClient.query_session_ids = []
            runtime._load_sdk = lambda: {  # type: ignore[method-assign]
                "module": type(
                    "SdkModule",
                    (),
                    {"PermissionResultAllow": _DummyAllow, "PermissionResultDeny": _DummyDeny},
                ),
                "ClaudeSDKClient": _GoalRecoveryDummyClient,
                "ClaudeAgentOptions": _DummyOptions,
            }

            async def scenario() -> list[RuntimeStreamEvent]:
                collected = []
                async for event in runtime.stream_events(
                    {"session_id": "session-1", "messages": [{"role": "user", "content": "继续做"}]},
                    request_headers={"x-user-id": "10154402"},
                    current_user=None,
                    proxy_base_url="http://127.0.0.1/internal/anthropic",
                    skill_mount_root=root,
                    run_id="run-1",
                ):
                    collected.append(event)
                return collected

            events = asyncio.run(scenario())

            self.assertEqual(len(asyncio.run(runtime.list_questions("session-1"))), 0)
            self.assertEqual(_GoalRecoveryDummyClient.query_session_ids, ["claude-session-1"])
            sent_text = _GoalRecoveryDummyClient.query_messages[0][0]["message"]["content"][0]["text"]
            self.assertEqual(sent_text, "继续做")
            self.assertIn("恢复后的普通消息已执行。", "".join(event.text or "" for event in events))
            goal = asyncio.run(goal_store.get("session-1"))
            self.assertIsNotNone(goal)
            self.assertEqual(goal.status, "running")
            self.assertEqual(goal.active_run_id, "")

    def test_paused_goal_arbitrary_message_does_not_clear_goal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / ".claude"
            config_dir.mkdir(parents=True, exist_ok=True)
            goal_store = SessionGoalStore(root / "data" / "sessions" / "goals.json")
            asyncio.run(goal_store.set("session-1", "持续完成目标"))
            asyncio.run(goal_store.pause("session-1", reason="user_interrupt"))
            asyncio.run(SessionMappingStore(root / "data" / "sessions" / "session-map.json").put("session-1", "claude-session-1", model="MiniMax-RAN3"))
            runtime = ClaudeSdkRuntime(
                ClaudeSettings(
                    workdir=root,
                    config_dir=config_dir,
                    default_model="MiniMax-RAN3",
                    permission_mode="acceptEdits",
                    cli_path="/usr/bin/claude",
                    setting_sources=["project", "local"],
                    skills_filter="all",
                    system_prompt_preset="claude_code",
                    system_prompt_append="",
                    system_prompt_file=None,
                    include_hook_events=False,
                    enable_file_checkpointing=False,
                    attachment_text_char_limit=96000,
                ),
                McpSettings(config_dir=root / "mcps", auto_load=False),
                ProviderSettings(
                    base_url="http://127.0.0.1:18081",
                    anthropic_version="2023-06-01",
                    api_key="demo",
                    request_timeout_sec=30.0,
                ),
                SessionMappingStore(root / "data" / "sessions" / "session-map.json"),
                SessionCheckpointStore(root / "data" / "sessions" / "checkpoints.json"),
                ProxyContextStore(),
                client_pool=None,  # type: ignore[arg-type]
                question_registry=QuestionRuntimeRegistry(),
                goal_store=goal_store,
            )
            runtime._client_pool = ClaudeClientPool()
            _GoalRecoveryDummyClient.query_messages = []
            _GoalRecoveryDummyClient.query_session_ids = []
            runtime._load_sdk = lambda: {  # type: ignore[method-assign]
                "module": type(
                    "SdkModule",
                    (),
                    {"PermissionResultAllow": _DummyAllow, "PermissionResultDeny": _DummyDeny},
                ),
                "ClaudeSDKClient": _GoalRecoveryDummyClient,
                "ClaudeAgentOptions": _DummyOptions,
            }

            async def scenario() -> list[RuntimeStreamEvent]:
                collected = []
                async for event in runtime.stream_events(
                    {"session_id": "session-1", "messages": [{"role": "user", "content": "先回答这个问题"}]},
                    request_headers={"x-user-id": "10154402"},
                    current_user=None,
                    proxy_base_url="http://127.0.0.1/internal/anthropic",
                    skill_mount_root=root,
                    run_id="run-1",
                ):
                    collected.append(event)
                return collected

            events = asyncio.run(scenario())

            self.assertEqual(len(asyncio.run(runtime.list_questions("session-1"))), 0)
            sent_texts = [
                message["message"]["content"][0]["text"]
                for query in _GoalRecoveryDummyClient.query_messages
                for message in query
            ]
            self.assertEqual(sent_texts, ["先回答这个问题"])
            self.assertIn("恢复后的普通消息已执行。", "".join(event.text or "" for event in events))
            goal = asyncio.run(goal_store.get("session-1"))
            self.assertIsNotNone(goal)
            self.assertEqual(goal.status, "running")
            self.assertEqual(goal.objective, "持续完成目标")

    def test_running_goal_without_active_client_continue_message_resumes_goal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / ".claude"
            config_dir.mkdir(parents=True, exist_ok=True)
            goal_store = SessionGoalStore(root / "data" / "sessions" / "goals.json")
            asyncio.run(goal_store.set("session-1", "持续完成目标"))
            asyncio.run(goal_store.mark_run_started("session-1", "stale-run"))
            asyncio.run(SessionMappingStore(root / "data" / "sessions" / "session-map.json").put("session-1", "claude-session-1", model="MiniMax-RAN3"))
            runtime = ClaudeSdkRuntime(
                ClaudeSettings(
                    workdir=root,
                    config_dir=config_dir,
                    default_model="MiniMax-RAN3",
                    permission_mode="acceptEdits",
                    cli_path="/usr/bin/claude",
                    setting_sources=["project", "local"],
                    skills_filter="all",
                    system_prompt_preset="claude_code",
                    system_prompt_append="",
                    system_prompt_file=None,
                    include_hook_events=False,
                    enable_file_checkpointing=False,
                    attachment_text_char_limit=96000,
                ),
                McpSettings(config_dir=root / "mcps", auto_load=False),
                ProviderSettings(
                    base_url="http://127.0.0.1:18081",
                    anthropic_version="2023-06-01",
                    api_key="demo",
                    request_timeout_sec=30.0,
                ),
                SessionMappingStore(root / "data" / "sessions" / "session-map.json"),
                SessionCheckpointStore(root / "data" / "sessions" / "checkpoints.json"),
                ProxyContextStore(),
                client_pool=None,  # type: ignore[arg-type]
                question_registry=QuestionRuntimeRegistry(),
                goal_store=goal_store,
            )
            runtime._client_pool = ClaudeClientPool()
            _GoalRecoveryDummyClient.query_messages = []
            _GoalRecoveryDummyClient.query_session_ids = []
            runtime._load_sdk = lambda: {  # type: ignore[method-assign]
                "module": type(
                    "SdkModule",
                    (),
                    {"PermissionResultAllow": _DummyAllow, "PermissionResultDeny": _DummyDeny},
                ),
                "ClaudeSDKClient": _GoalRecoveryDummyClient,
                "ClaudeAgentOptions": _DummyOptions,
            }

            async def scenario() -> list[RuntimeStreamEvent]:
                collected = []
                async for event in runtime.stream_events(
                    {"session_id": "session-1", "messages": [{"role": "user", "content": "继续"}]},
                    request_headers={"x-user-id": "10154402"},
                    current_user=None,
                    proxy_base_url="http://127.0.0.1/internal/anthropic",
                    skill_mount_root=root,
                    run_id="run-2",
                ):
                    collected.append(event)
                return collected

            events = asyncio.run(scenario())

            self.assertEqual(len(asyncio.run(runtime.list_questions("session-1"))), 0)
            self.assertEqual(_GoalRecoveryDummyClient.query_session_ids, ["claude-session-1"])
            sent_text = _GoalRecoveryDummyClient.query_messages[0][0]["message"]["content"][0]["text"]
            self.assertEqual(sent_text, "继续")
            self.assertIn("恢复后的普通消息已执行。", "".join(event.text or "" for event in events))
            goal = asyncio.run(goal_store.get("session-1"))
            self.assertIsNotNone(goal)
            self.assertEqual(goal.status, "running")
            self.assertEqual(goal.active_run_id, "")

    def test_running_goal_without_active_client_arbitrary_message_resumes_goal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / ".claude"
            config_dir.mkdir(parents=True, exist_ok=True)
            goal_store = SessionGoalStore(root / "data" / "sessions" / "goals.json")
            asyncio.run(goal_store.set("session-1", "持续完成目标"))
            asyncio.run(goal_store.mark_run_started("session-1", "stale-run"))
            asyncio.run(SessionMappingStore(root / "data" / "sessions" / "session-map.json").put("session-1", "claude-session-1", model="MiniMax-RAN3"))
            runtime = ClaudeSdkRuntime(
                ClaudeSettings(
                    workdir=root,
                    config_dir=config_dir,
                    default_model="MiniMax-RAN3",
                    permission_mode="acceptEdits",
                    cli_path="/usr/bin/claude",
                    setting_sources=["project", "local"],
                    skills_filter="all",
                    system_prompt_preset="claude_code",
                    system_prompt_append="",
                    system_prompt_file=None,
                    include_hook_events=False,
                    enable_file_checkpointing=False,
                    attachment_text_char_limit=96000,
                ),
                McpSettings(config_dir=root / "mcps", auto_load=False),
                ProviderSettings(
                    base_url="http://127.0.0.1:18081",
                    anthropic_version="2023-06-01",
                    api_key="demo",
                    request_timeout_sec=30.0,
                ),
                SessionMappingStore(root / "data" / "sessions" / "session-map.json"),
                SessionCheckpointStore(root / "data" / "sessions" / "checkpoints.json"),
                ProxyContextStore(),
                client_pool=None,  # type: ignore[arg-type]
                question_registry=QuestionRuntimeRegistry(),
                goal_store=goal_store,
            )
            runtime._client_pool = ClaudeClientPool()
            _GoalRecoveryDummyClient.query_messages = []
            _GoalRecoveryDummyClient.query_session_ids = []
            runtime._load_sdk = lambda: {  # type: ignore[method-assign]
                "module": type(
                    "SdkModule",
                    (),
                    {"PermissionResultAllow": _DummyAllow, "PermissionResultDeny": _DummyDeny},
                ),
                "ClaudeSDKClient": _GoalRecoveryDummyClient,
                "ClaudeAgentOptions": _DummyOptions,
            }

            async def scenario() -> list[RuntimeStreamEvent]:
                collected = []
                async for event in runtime.stream_events(
                    {"session_id": "session-1", "messages": [{"role": "user", "content": "补充一个测试要求"}]},
                    request_headers={"x-user-id": "10154402"},
                    current_user=None,
                    proxy_base_url="http://127.0.0.1/internal/anthropic",
                    skill_mount_root=root,
                    run_id="run-3",
                ):
                    collected.append(event)
                return collected

            events = asyncio.run(scenario())

            self.assertEqual(len(asyncio.run(runtime.list_questions("session-1"))), 0)
            self.assertEqual(_GoalRecoveryDummyClient.query_session_ids, ["claude-session-1"])
            sent_text = _GoalRecoveryDummyClient.query_messages[0][0]["message"]["content"][0]["text"]
            self.assertEqual(sent_text, "补充一个测试要求")
            self.assertIn("恢复后的普通消息已执行。", "".join(event.text or "" for event in events))
            goal = asyncio.run(goal_store.get("session-1"))
            self.assertIsNotNone(goal)
            self.assertEqual(goal.status, "running")
            self.assertEqual(goal.active_run_id, "")

    def test_stream_events_emits_compact_boundary_and_completion_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / ".claude"
            config_dir.mkdir(parents=True, exist_ok=True)
            runtime = ClaudeSdkRuntime(
                ClaudeSettings(
                    workdir=root,
                    config_dir=config_dir,
                    default_model="MiniMax-RAN3",
                    permission_mode="acceptEdits",
                    cli_path="/usr/bin/claude",
                    setting_sources=["project", "local"],
                    skills_filter="all",
                    system_prompt_preset="claude_code",
                    system_prompt_append="",
                    system_prompt_file=None,
                    include_hook_events=False,
                    enable_file_checkpointing=False,
                    attachment_text_char_limit=96000,
                ),
                McpSettings(config_dir=root / "mcps", auto_load=False),
                ProviderSettings(
                    base_url="http://127.0.0.1:18081",
                    anthropic_version="2023-06-01",
                    api_key="demo",
                    request_timeout_sec=30.0,
                ),
                SessionMappingStore(root / "data" / "sessions" / "session-map.json"),
                SessionCheckpointStore(root / "data" / "sessions" / "checkpoints.json"),
                ProxyContextStore(),
                client_pool=None,  # type: ignore[arg-type]
            )
            runtime._client_pool = ClaudeClientPool()
            runtime._load_sdk = lambda: {  # type: ignore[method-assign]
                "module": type(
                    "SdkModule",
                    (),
                    {"PermissionResultAllow": _DummyAllow, "PermissionResultDeny": _DummyDeny},
                ),
                "ClaudeSDKClient": _CompactBoundaryDummyClient,
                "ClaudeAgentOptions": _DummyOptions,
            }

            async def scenario() -> list[RuntimeStreamEvent]:
                payload = {
                    "session_id": "session-1",
                    "metadata": {
                        "runtimeCommand": {
                            "source": "claude-code",
                            "commandId": "compact",
                            "command": "/compact",
                            "displayName": "压缩上下文",
                            "requestId": "cmd-compact-1",
                        }
                    },
                }
                collected = []
                async for event in runtime.stream_events(
                    payload,
                    request_headers={"x-user-id": "10154402"},
                    current_user=None,
                    proxy_base_url="http://127.0.0.1/internal/anthropic",
                    skill_mount_root=root,
                    run_id="run-1",
                ):
                    collected.append(event)
                return collected

            events = asyncio.run(scenario())

            self.assertEqual([event.kind for event in events], ["command", "command", "command"])
            self.assertEqual(events[0].payload["status"], "running")
            self.assertEqual(events[1].payload["status"], "boundary")
            self.assertEqual(events[1].payload["phase"], "compact_boundary")
            self.assertEqual(events[1].payload["result"], "compact_boundary")
            self.assertEqual(events[2].payload["status"], "completed")
            self.assertEqual(events[2].payload["phase"], "compact_complete")
            self.assertEqual(events[2].payload["result"], "compact_boundary")

    def test_stream_events_sends_context_command_to_claude_and_emits_card_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / ".claude"
            config_dir.mkdir(parents=True, exist_ok=True)
            runtime = ClaudeSdkRuntime(
                ClaudeSettings(
                    workdir=root,
                    config_dir=config_dir,
                    default_model="MiniMax-RAN3",
                    permission_mode="acceptEdits",
                    cli_path="/usr/bin/claude",
                    setting_sources=["project", "local"],
                    skills_filter="all",
                    system_prompt_preset="claude_code",
                    system_prompt_append="",
                    system_prompt_file=None,
                    include_hook_events=False,
                    enable_file_checkpointing=False,
                    attachment_text_char_limit=96000,
                ),
                McpSettings(config_dir=root / "mcps", auto_load=False),
                ProviderSettings(
                    base_url="http://127.0.0.1:18081",
                    anthropic_version="2023-06-01",
                    api_key="demo",
                    request_timeout_sec=30.0,
                ),
                SessionMappingStore(root / "data" / "sessions" / "session-map.json"),
                SessionCheckpointStore(root / "data" / "sessions" / "checkpoints.json"),
                ProxyContextStore(),
                client_pool=None,  # type: ignore[arg-type]
            )
            runtime._client_pool = ClaudeClientPool()
            runtime._load_sdk = lambda: {  # type: ignore[method-assign]
                "module": type(
                    "SdkModule",
                    (),
                    {"PermissionResultAllow": _DummyAllow, "PermissionResultDeny": _DummyDeny},
                ),
                "ClaudeSDKClient": _ContextReportDummyClient,
                "ClaudeAgentOptions": _DummyOptions,
            }

            async def scenario() -> list[RuntimeStreamEvent]:
                payload = {
                    "session_id": "session-1",
                    "metadata": {
                        "runtimeCommand": {
                            "source": "claude-code",
                            "commandId": "context",
                            "command": "/context",
                            "displayName": "查看上下文",
                            "requestId": "cmd-context-1",
                        }
                    },
                    "messages": [
                        {"role": "user", "content": "wrapped history should not be sent"},
                    ],
                }
                collected = []
                async for event in runtime.stream_events(
                    payload,
                    request_headers={"x-user-id": "10154402"},
                    current_user=None,
                    proxy_base_url="http://127.0.0.1/internal/anthropic",
                    skill_mount_root=root,
                    run_id="run-1",
                ):
                    collected.append(event)
                return collected

            _ContextReportDummyClient.queried = False
            _ContextReportDummyClient.last_query_messages = []
            events = asyncio.run(scenario())

            self.assertTrue(_ContextReportDummyClient.queried)
            self.assertIn("/context", repr(_ContextReportDummyClient.last_query_messages))
            self.assertEqual([event.kind for event in events], ["command", "text", "command"])
            self.assertEqual(events[0].payload["status"], "running")
            self.assertEqual(events[0].payload["phase"], "start")
            self.assertIn("Context usage: 42%", events[1].text)
            self.assertEqual(events[2].payload["status"], "completed")
            self.assertEqual(events[2].payload["phase"], "context_report")
            self.assertEqual(events[2].payload["result"], "context_report")

    def test_build_sdk_hooks_records_callback_payload(self) -> None:
        registry = HookRuntimeRegistry()
        execution = ("run-1", "claude-1")

        async def scenario() -> None:
            hooks = build_sdk_hooks(
                type("SdkModule", (), {"HookMatcher": _DummyHookMatcher}),
                frontend_session_id="session-1",
                registry=registry,
                execution_resolver=lambda session_id: execution if session_id == "session-1" else ("", ""),
            )
            matcher = hooks["Notification"][0]
            callback = matcher.hooks[0]
            await callback(
                {
                    "hook_event_name": "Notification",
                    "session_id": "claude-1",
                    "message": "hello",
                    "notification_type": "info",
                },
                None,
                {"signal": None},
            )
            items = await registry.list_events("session-1")
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["runId"], "run-1")
            self.assertEqual(items[0]["hookEventName"], "Notification")
            self.assertEqual(items[0]["source"], "sdk_callback")
            self.assertEqual(items[0]["output"]["continue"], True)

        asyncio.run(scenario())

    def test_hook_stream_payload_maps_hook_event_message(self) -> None:
        item = type(
            "HookEventMessage",
            (),
            {
                "subtype": "hook_response",
                "hook_event_name": "PreToolUse",
                "session_id": "claude-1",
                "uuid": "uuid-1",
                "data": {
                    "callback_id": "callback-1",
                    "tool_name": "Bash",
                    "tool_use_id": "tool-1",
                    "agent_id": "agent-1",
                    "outcome": "success",
                    "output": {"continue": True},
                },
            },
        )()
        payload = hook_stream_payload(item)
        self.assertEqual(payload["eventId"], "callback-1")
        self.assertEqual(payload["hookEventName"], "PreToolUse")
        self.assertEqual(payload["source"], "sdk_stream")
        self.assertEqual(payload["output"]["continue"], True)

    def test_hook_stream_payload_maps_stop_hook_attachment(self) -> None:
        item = type(
            "AttachmentMessage",
            (),
            {
                "session_id": "claude-1",
                "uuid": "uuid-1",
                "attachment": {
                    "type": "hook_non_blocking_error",
                    "hookName": "Stop",
                    "hookEvent": "Stop",
                    "toolUseID": "tool-stop-1",
                    "stderr": "JSON validation failed",
                    "stdout": '{"ok": true, "reason": "done"}',
                    "exitCode": 1,
                },
            },
        )()
        payload = hook_stream_payload(item)
        self.assertEqual(payload["eventId"], "tool-stop-1")
        self.assertEqual(payload["hookEventName"], "Stop")
        self.assertEqual(payload["source"], "sdk_attachment")
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["data"]["stdout"], '{"ok": true, "reason": "done"}')

    def test_tool_event_bridge_tracks_tasks_into_task_registry(self) -> None:
        async def scenario() -> None:
            task_registry = TaskRuntimeRegistry()
            tool_registry = ToolRuntimeRegistry()
            await task_registry.register_run("run-1")
            bridge = _ToolEventBridge(tool_registry, task_registry, run_id="run-1")
            started = type(
                "TaskStartedMessage",
                (),
                {
                    "task_id": "task-1",
                    "description": "run bash",
                    "uuid": "u1",
                    "session_id": "s1",
                    "tool_use_id": "tool-1",
                    "task_type": "bash",
                },
            )()
            progressed = type(
                "TaskProgressMessage",
                (),
                {
                    "task_id": "task-1",
                    "description": "still running",
                    "usage": {"x": 1},
                    "uuid": "u2",
                    "session_id": "s1",
                    "tool_use_id": "tool-1",
                    "last_tool_name": "bash",
                },
            )()
            notified = type(
                "TaskNotificationMessage",
                (),
                {
                    "task_id": "task-1",
                    "status": "completed",
                    "output_file": "",
                    "summary": "done",
                    "uuid": "u3",
                    "session_id": "s1",
                    "tool_use_id": "tool-1",
                    "usage": None,
                },
            )()

            async for _ in bridge.handle_item(started):
                pass
            async for _ in bridge.handle_item(progressed):
                pass
            async for _ in bridge.handle_item(notified):
                pass

            task = await task_registry.get_task(run_id="run-1", task_id="task-1")
            self.assertIsNotNone(task)
            self.assertEqual(task["status"], "completed")
            self.assertEqual(task["toolCallId"], "tool-1")

            output_chunks = []
            async for chunk in task_registry.stream_task_output(run_id="run-1", task_id="task-1"):
                output_chunks.append(chunk.text)
            self.assertIn("still running", "\n".join(output_chunks))

            asyncio.run(scenario())

    def test_tool_event_bridge_ends_open_tasks_when_runtime_stream_finishes(self) -> None:
        async def scenario() -> None:
            task_registry = TaskRuntimeRegistry()
            tool_registry = ToolRuntimeRegistry()
            await task_registry.register_run("run-1")
            bridge = _ToolEventBridge(tool_registry, task_registry, run_id="run-1")
            started = type(
                "TaskStartedMessage",
                (),
                {
                    "task_id": "task-1",
                    "description": "parallel analysis",
                    "uuid": "u1",
                    "session_id": "s1",
                    "tool_use_id": "tool-1",
                    "task_type": "local_agent",
                },
            )()

            async for _ in bridge.handle_item(started):
                pass
            ended_events = []
            async for event in bridge.finish_open_tasks():
                ended_events.append(event)

            self.assertEqual(len(ended_events), 1)
            self.assertEqual(ended_events[0].kind, "task")
            self.assertEqual(ended_events[0].payload["taskId"], "task-1")
            self.assertEqual(ended_events[0].payload["status"], "ended")
            self.assertIn("主响应流已结束", ended_events[0].payload["log"])

            task = await task_registry.get_task(run_id="run-1", task_id="task-1")
            self.assertIsNotNone(task)
            self.assertEqual(task["status"], "ended")

        asyncio.run(scenario())

    def test_tool_event_bridge_reports_waiting_without_ending_open_tasks(self) -> None:
        async def scenario() -> None:
            task_registry = TaskRuntimeRegistry()
            tool_registry = ToolRuntimeRegistry()
            await task_registry.register_run("run-1")
            bridge = _ToolEventBridge(tool_registry, task_registry, run_id="run-1")
            started = type(
                "TaskStartedMessage",
                (),
                {
                    "task_id": "task-1",
                    "description": "parallel analysis",
                    "uuid": "u1",
                    "session_id": "s1",
                    "tool_use_id": "tool-1",
                    "task_type": "local_agent",
                },
            )()

            async for _ in bridge.handle_item(started):
                pass
            waiting_events = []
            async for event in bridge.waiting_open_tasks():
                waiting_events.append(event)

            self.assertEqual(len(waiting_events), 1)
            self.assertEqual(waiting_events[0].kind, "task")
            self.assertEqual(waiting_events[0].payload["phase"], "waiting_subtasks")
            self.assertEqual(waiting_events[0].payload["taskId"], "task-1")
            self.assertEqual(waiting_events[0].payload["status"], "running")
            self.assertIn("正在等待后台子任务完成", waiting_events[0].payload["log"])

            task = await task_registry.get_task(run_id="run-1", task_id="task-1")
            self.assertIsNotNone(task)
            self.assertEqual(task["status"], "running")

        asyncio.run(scenario())

    def test_tool_event_bridge_uses_patch_task_identity_for_updates(self) -> None:
        async def scenario() -> None:
            task_registry = TaskRuntimeRegistry()
            tool_registry = ToolRuntimeRegistry()
            await task_registry.register_run("run-1")
            bridge = _ToolEventBridge(tool_registry, task_registry, run_id="run-1")
            updated = type(
                "TaskUpdatedMessage",
                (),
                {
                    "patch": {
                        "task_id": "task-2",
                        "task_type": "local_agent",
                        "tool_use_id": "tool-2",
                        "description": "检查 sidecar 入口",
                        "status": "failed",
                        "error": "模型未配置或已禁用: claude-opus-4-8",
                    },
                    "uuid": "u2",
                    "session_id": "s1",
                },
            )()

            events = []
            async for event in bridge.handle_item(updated):
                events.append(event)

            task = await task_registry.get_task(run_id="run-1", task_id="task-2")
            self.assertIsNotNone(task)
            self.assertEqual(task["status"], "failed")
            self.assertEqual(task["toolCallId"], "tool-2")
            self.assertEqual(task["description"], "检查 sidecar 入口")
            self.assertTrue(any(event.kind == "task" and event.payload["taskId"] == "task-2" for event in events))

        asyncio.run(scenario())

    def test_permission_callback_creates_approval_request_and_waits_for_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / ".claude"
            config_dir.mkdir(parents=True, exist_ok=True)
            runtime = ClaudeSdkRuntime(
                ClaudeSettings(
                    workdir=root,
                    config_dir=config_dir,
                    default_model="MiniMax-RAN3",
                    permission_mode="default",
                    cli_path="/usr/bin/claude",
                    setting_sources=["project", "local"],
                    skills_filter="all",
                    system_prompt_preset="claude_code",
                    system_prompt_append="",
                    system_prompt_file=None,
                    include_hook_events=False,
                    enable_file_checkpointing=False,
                    attachment_text_char_limit=96000,
                ),
                McpSettings(config_dir=root / "mcps", auto_load=False),
                ProviderSettings(
                    base_url="http://127.0.0.1:18081",
                    anthropic_version="2023-06-01",
                    api_key="demo",
                    request_timeout_sec=30.0,
                ),
                SessionMappingStore(root / "data" / "sessions" / "session-map.json"),
                SessionCheckpointStore(root / "data" / "sessions" / "checkpoints.json"),
                ProxyContextStore(),
                client_pool=None,  # type: ignore[arg-type]
            )
            runtime._session_contexts["session-1"] = type("Execution", (), {"run_id": "run-1", "claude_session_id": "claude-1"})()

            async def scenario() -> None:
                callback = runtime._build_can_use_tool_callback(
                    sdk_module=type(
                        "SdkModule",
                        (),
                        {"PermissionResultAllow": _DummyAllow, "PermissionResultDeny": _DummyDeny},
                    ),
                    frontend_session_id="session-1",
                )
                task = asyncio.create_task(
                    callback(
                        "Bash",
                        {"command": "echo hi"},
                        type(
                            "PermissionContext",
                            (),
                            {
                                "tool_use_id": "tool-1",
                                "agent_id": "agent-1",
                                "blocked_path": "",
                                "decision_reason": "",
                                "title": "Allow Bash?",
                                "display_name": "Bash",
                                "description": "Execute shell",
                            },
                        )(),
                    )
                )
                await asyncio.sleep(0)
                approvals = await runtime.list_approvals("session-1")
                self.assertEqual(len(approvals), 1)
                self.assertEqual(approvals[0]["runId"], "run-1")
                self.assertEqual(approvals[0]["toolUseId"], "tool-1")
                await runtime.respond_approval("session-1", approvals[0]["requestId"], decision="allow")
                result = await task
                self.assertEqual(result.behavior, "allow")

            asyncio.run(scenario())
