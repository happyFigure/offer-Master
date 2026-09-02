from __future__ import annotations

import unittest

from src.api.payload import build_initial_prompt, extract_runtime_command, normalize_session_id_in_payload, sanitize_incoming_assistant_content


class PayloadTests(unittest.TestCase):
    def test_normalize_session_id_prefers_root_fields(self) -> None:
        payload = {"user": "demo-session", "messages": [{"role": "user", "content": "hello"}]}
        normalize_session_id_in_payload(payload)
        self.assertEqual(payload["session_id"], "demo-session")

    def test_normalize_session_id_falls_back_to_metadata(self) -> None:
        payload = {
            "metadata": {"userId": "meta-session"},
            "messages": [{"role": "user", "content": "hello"}],
        }
        normalize_session_id_in_payload(payload)
        self.assertEqual(payload["session_id"], "meta-session")

    def test_build_initial_prompt_splits_history_and_current_message(self) -> None:
        payload = {
            "messages": [
                {"role": "user", "content": "old question"},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "new question"},
            ]
        }
        prompt = build_initial_prompt(payload)
        self.assertIn("[Chat messages since your last reply - for context]", prompt)
        self.assertIn("assistant: old answer", prompt)
        self.assertIn("[Current message - respond to this]", prompt)
        self.assertTrue(prompt.endswith("new question"))

    def test_build_initial_prompt_strips_tool_shell_lines_from_assistant_history(self) -> None:
        payload = {
            "messages": [
                {"role": "user", "content": "old question"},
                {
                    "role": "assistant",
                    "content": (
                        "normal answer\n"
                        "[tool] {\"phase\":\"start\"}\n"
                        "[task] {\"taskId\":\"task-1\"}\n"
                        "[approval] {\"requestId\":\"req-1\"}\n"
                        "[question] {\"questionId\":\"question-1\"}\n"
                        "[meta] {\"runId\":\"run-1\"}\n"
                        "[artifacts] {\"runId\":\"run-1\"}\n"
                        "[runtime-retry] note\n"
                        "final answer"
                    ),
                },
                {"role": "user", "content": "new question"},
            ]
        }
        prompt = build_initial_prompt(payload)
        self.assertIn("assistant: normal answer\nfinal answer", prompt)
        self.assertNotIn("[tool]", prompt)
        self.assertNotIn("[task]", prompt)
        self.assertNotIn("[approval]", prompt)
        self.assertNotIn("[question]", prompt)
        self.assertNotIn("[meta]", prompt)
        self.assertNotIn("[artifacts]", prompt)
        self.assertNotIn("[runtime-retry]", prompt)

    def test_extract_runtime_command_accepts_all_p0_commands(self) -> None:
        for command_id, command in {
            "compact": "/compact",
            "context": "/context",
            "usage": "/usage",
            "goal": "/goal",
        }.items():
            runtime_command = extract_runtime_command(
                {
                    "metadata": {
                        "runtimeCommand": {
                            "source": "claude-code",
                            "commandId": command_id,
                            "command": command,
                            "args": {"text": "clear"} if command_id == "goal" else {},
                            "displayName": command,
                            "requestId": "cmd-1",
                        }
                    }
                }
            )
            self.assertIsNotNone(runtime_command)
            self.assertEqual(runtime_command.command_id, command_id)
            self.assertEqual(runtime_command.command, command)

    def test_extract_runtime_command_rejects_invalid_commands(self) -> None:
        self.assertIsNone(extract_runtime_command({"runtime_command": {"source": "claude-code", "commandId": "clear", "command": "/clear"}}))
        self.assertIsNone(extract_runtime_command({"runtime_command": {"source": "other", "commandId": "compact", "command": "/compact"}}))
        self.assertIsNone(extract_runtime_command({"runtime_command": {"source": "claude-code", "commandId": "compact", "command": "/compact", "args": {"text": "now"}}}))
        self.assertIsNone(extract_runtime_command({"runtime_command": {"source": "claude-code", "commandId": "permissions", "command": "/permissions"}}))
        self.assertIsNone(extract_runtime_command({"messages": [{"role": "user", "content": "/permissions"}]}))

    def test_extract_runtime_command_accepts_declared_workspace_command_with_arguments(self) -> None:
        runtime_command = extract_runtime_command(
            {
                "metadata": {
                    "runtimeCommand": {
                        "source": "claude-code",
                        "kind": "command",
                        "commandId": "omni.sdt",
                        "command": "/omni.sdt",
                        "args": {"text": "RAN-12345"},
                        "displayName": "omni.sdt",
                        "requestId": "cmd-workspace-1",
                    }
                }
            }
        )

        self.assertIsNotNone(runtime_command)
        self.assertEqual(runtime_command.command_id, "omni.sdt")
        self.assertEqual(runtime_command.prompt_text, "/omni.sdt RAN-12345")

    def test_extract_runtime_command_accepts_declared_workflow_with_arguments(self) -> None:
        runtime_command = extract_runtime_command(
            {
                "metadata": {
                    "runtimeCommand": {
                        "source": "claude-code",
                        "kind": "workflow",
                        "commandId": "audit-routes",
                        "command": "/audit-routes",
                        "args": {"text": "src/api"},
                        "displayName": "audit-routes",
                        "requestId": "cmd-workflow-1",
                    }
                }
            }
        )

        self.assertIsNotNone(runtime_command)
        self.assertEqual(runtime_command.command_id, "audit-routes")
        self.assertEqual(runtime_command.prompt_text, "/audit-routes src/api")

    def test_extract_runtime_command_rejects_unsafe_dynamic_command_name(self) -> None:
        self.assertIsNone(
            extract_runtime_command(
                {
                    "runtime_command": {
                        "source": "claude-code",
                        "kind": "skill",
                        "commandId": "bad command",
                        "command": "/bad command",
                    }
                }
            )
        )

    def test_extract_runtime_command_rejects_plan_in_current_environment(self) -> None:
        self.assertIsNone(
            extract_runtime_command(
                {
                    "metadata": {
                        "runtimeCommand": {
                            "source": "claude-code",
                            "commandId": "plan",
                            "command": "/plan",
                            "args": {},
                        }
                    }
                }
            )
        )
        self.assertIsNone(
            extract_runtime_command(
                {
                    "messages": [
                        {"role": "user", "content": "/plan 写一下简历"},
                    ],
                }
            )
        )

    def test_extract_runtime_command_fallback_rejects_non_argument_commands_with_args(self) -> None:
        self.assertIsNone(
            extract_runtime_command(
                {
                    "messages": [
                        {"role": "user", "content": "/usage now"},
                    ],
                }
            )
        )

    def test_sanitize_incoming_assistant_content_only_removes_protocol_lines(self) -> None:
        text = (
            "line1\n"
            "  [tool] {\"a\":1}\n"
            "[task] {\"taskId\":\"task-1\"}\n"
            "[approval] {\"requestId\":\"req-1\"}\n"
            "[question] {\"questionId\":\"question-1\"}\n"
            "[meta] {\"runId\":\"run-1\"}\n"
            "[artifacts] {\"runId\":\"run-1\"}\n"
            "[runtime-retry] x\n"
            "line2"
        )
        self.assertEqual(sanitize_incoming_assistant_content(text), "line1\nline2")
