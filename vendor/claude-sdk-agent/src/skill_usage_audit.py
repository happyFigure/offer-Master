from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import httpx

from .config import SkillUsageAuditSettings

logger = logging.getLogger(__name__)

_SCRIPT_EXTENSIONS = {".py", ".sh", ".js", ".ts", ".mjs", ".cjs", ".bash", ".zsh", ".fish"}
_SHELL_OPERATORS = {"&&", "||", "|", ";", "&", "(", ")", "{", "}", "<", ">", "2>", "1>"}
_REDIRECT_OPERATORS = {"<", ">", ">>", "1>", "1>>", "2>", "2>>"}
_EXECUTOR_NAMES = {
    "python",
    "python3",
    "py",
    "bash",
    "sh",
    "zsh",
    "fish",
    "node",
    "deno",
    "bun",
    "ruby",
    "perl",
    "php",
}
_SHELL_ASSIGNMENT_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_SHELL_VAR_RE = re.compile(r"\$(?:([A-Za-z_][A-Za-z0-9_]*)|\{([^}]+)\})")


@dataclass(frozen=True)
class SkillUsageAttribution:
    skill_name: str
    skill_dir: Path
    script_path: Path
    action: str


@dataclass(frozen=True)
class _PendingUsage:
    tool_call_id: str
    tool_name: str
    attribution: SkillUsageAttribution
    invocation: str


class SkillUsageAuditor:
    """Attribute successful Claude Code Bash tool calls to mounted Skill scripts."""

    def __init__(
        self,
        *,
        settings: SkillUsageAuditSettings,
        skill_mount_root: Path | None,
        base_context: Mapping[str, Any] | None,
        request_headers: Mapping[str, str] | None,
    ) -> None:
        self._settings = settings
        self._base_context = dict(base_context or {})
        self._request_headers = _forward_headers(request_headers)
        self._skills = _discover_mounted_skills(skill_mount_root)
        self._pending: dict[str, _PendingUsage] = {}
        self._recorded_tool_calls: set[str] = set()
        self._tasks: set[asyncio.Task[None]] = set()

    def observe_tool_start(self, event: Mapping[str, Any], *, cwd: Path) -> None:
        if not self._enabled:
            return
        tool_call_id = str(event.get("toolCallId") or "").strip()
        if not tool_call_id or tool_call_id in self._recorded_tool_calls:
            return
        tool_name = str(event.get("name") or "").strip()
        if tool_name.lower() != "bash":
            return
        arguments = event.get("arguments")
        if not isinstance(arguments, Mapping):
            return
        command = str(arguments.get("command") or arguments.get("cmd") or "").strip()
        if not command:
            return
        attribution = _attribute_skill_command(command=command, cwd=cwd, skills=self._skills)
        if attribution is None:
            return
        self._pending[tool_call_id] = _PendingUsage(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            attribution=attribution,
            invocation="claude_code_bash_attributed",
        )

    def observe_tool_result(self, event: Mapping[str, Any]) -> None:
        if not self._enabled:
            return
        tool_call_id = str(event.get("toolCallId") or "").strip()
        if not tool_call_id or tool_call_id in self._recorded_tool_calls:
            return
        pending = self._pending.pop(tool_call_id, None)
        if pending is None:
            return
        status = str(event.get("status") or "").strip().lower()
        if status not in {"completed", "complete", "done", "success", "succeeded"}:
            return
        self._recorded_tool_calls.add(tool_call_id)
        task = asyncio.create_task(self._record_usage(pending))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def record_workflow_skill_command(self, skill_name: str) -> None:
        if not self._enabled:
            return
        normalized = str(skill_name or "").strip().lower()
        if not normalized:
            return
        key = f"skill-command:{normalized}"
        if key in self._recorded_tool_calls:
            return
        for mounted_name, skill_dir in self._skills:
            if mounted_name.lower() != normalized:
                continue
            if (skill_dir / "scripts").exists():
                return
            self._recorded_tool_calls.add(key)
            pending = _PendingUsage(
                tool_call_id=key,
                tool_name="Skill",
                attribution=SkillUsageAttribution(
                    skill_name=mounted_name,
                    skill_dir=skill_dir,
                    script_path=skill_dir / "SKILL.md",
                    action="workflow",
                ),
                invocation="claude_code_skill_command_workflow",
            )
            task = asyncio.create_task(self._record_usage(pending))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            return

    @property
    def _enabled(self) -> bool:
        return bool(self._settings.enabled and self._settings.base_url and self._settings.endpoint and self._skills)

    async def _record_usage(self, pending: _PendingUsage) -> None:
        attribution = pending.attribution
        context = dict(self._base_context)
        context.update(
            {
                "skill_action": attribution.action,
                "skill_invocation": pending.invocation,
                "skill_name": attribution.skill_name,
                "tool_name": pending.tool_name,
                "tool_call_id": pending.tool_call_id,
                "runtime": "claude-code",
            }
        )
        payload = {
            "skillDir": str(attribution.skill_dir),
            "auditContext": context,
        }
        url = _join_url(self._settings.base_url, self._settings.endpoint)
        try:
            async with httpx.AsyncClient(timeout=max(0.2, float(self._settings.timeout_sec or 3.0)), trust_env=False) as client:
                response = await client.post(url, json=payload, headers=self._request_headers)
            if response.status_code >= 400:
                logger.warning(
                    "[skill-usage-audit] post failed skill=%s status=%s body=%s",
                    attribution.skill_name,
                    response.status_code,
                    _truncate(response.text, 240),
                )
                return
            try:
                body = response.json()
            except ValueError:
                body = {}
            if isinstance(body, Mapping) and body.get("ok") is False:
                logger.warning(
                    "[skill-usage-audit] post rejected skill=%s body=%s",
                    attribution.skill_name,
                    _truncate(json.dumps(body, ensure_ascii=False), 240),
                )
        except Exception as exc:
            logger.warning(
                "[skill-usage-audit] post exception skill=%s error=%s: %s",
                attribution.skill_name,
                type(exc).__name__,
                _truncate(str(exc), 240),
            )


def _discover_mounted_skills(skill_mount_root: Path | None) -> list[tuple[str, Path]]:
    if skill_mount_root is None:
        return []
    skills_root = Path(skill_mount_root) / ".claude" / "skills"
    if not skills_root.is_dir():
        return []
    out: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for entry in sorted(skills_root.iterdir(), key=lambda item: item.name):
        if not entry.is_dir() and not entry.is_symlink():
            continue
        resolved = _resolve_path(entry)
        if not (resolved / "SKILL.md").is_file():
            continue
        key = resolved.as_posix()
        if key in seen:
            continue
        seen.add(key)
        out.append((entry.name, resolved))
    return out


def _attribute_skill_command(
    *,
    command: str,
    cwd: Path,
    skills: Sequence[tuple[str, Path]],
) -> SkillUsageAttribution | None:
    if not command or not skills:
        return None
    candidates = _command_path_candidates(command, _resolve_path(cwd))
    if not candidates:
        return None
    matches: list[SkillUsageAttribution] = []
    for candidate in candidates:
        script_path = _resolve_path(candidate)
        if not _is_likely_script_path(script_path):
            continue
        for skill_name, skill_dir in skills:
            scripts_dir = _resolve_path(skill_dir) / "scripts"
            if _is_relative_to(script_path, scripts_dir):
                matches.append(
                    SkillUsageAttribution(
                        skill_name=skill_name,
                        skill_dir=_resolve_path(skill_dir),
                        script_path=script_path,
                        action=_infer_action(script_path),
                    )
                )
    if not matches:
        return None
    return max(matches, key=lambda item: len(item.skill_dir.parts))


def _command_path_candidates(command: str, cwd: Path) -> list[Path]:
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        tokens = command.split()
    shell_env = _shell_assignments(tokens)
    candidates: list[Path] = []
    seen: set[str] = set()
    for idx, token in enumerate(tokens):
        text = str(token or "").strip()
        if not text or text in _SHELL_OPERATORS or text.startswith("-"):
            continue
        if not _looks_executed_path_token(tokens, idx):
            continue
        if not _looks_like_path_token(text):
            continue
        text = _expand_shell_vars(text, shell_env)
        path = Path(text.replace("\\", "/"))
        if not path.is_absolute():
            path = cwd / path
        resolved = _resolve_path(path)
        key = resolved.as_posix()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(resolved)
    return candidates


def _shell_assignments(tokens: Sequence[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for token in tokens:
        match = _SHELL_ASSIGNMENT_RE.match(str(token or "").strip())
        if match:
            env[match.group(1)] = match.group(2)
    return env


def _expand_shell_vars(text: str, env: Mapping[str, str]) -> str:
    def _replace(match: re.Match[str]) -> str:
        name = str(match.group(1) or match.group(2) or "")
        if name in env:
            return env[name]
        return os.environ.get(name, match.group(0))

    return _SHELL_VAR_RE.sub(_replace, text)


def _looks_like_path_token(text: str) -> bool:
    normalized = text.replace("\\", "/")
    return "/" in normalized or Path(normalized).suffix.lower() in _SCRIPT_EXTENSIONS


def _looks_executed_path_token(tokens: Sequence[str], idx: int) -> bool:
    previous = _previous_meaningful_token(tokens, idx)
    if previous in _REDIRECT_OPERATORS:
        return False
    if previous and _is_executor_token(previous):
        return True
    return idx == 0 or previous in {"&&", "||", ";", "|"}


def _previous_meaningful_token(tokens: Sequence[str], idx: int) -> str:
    pos = idx - 1
    while pos >= 0:
        text = str(tokens[pos] or "").strip()
        if text:
            return text
        pos -= 1
    return ""


def _is_executor_token(token: str) -> bool:
    name = Path(str(token or "").replace("\\", "/")).name.lower()
    if name in _EXECUTOR_NAMES:
        return True
    return bool(re.fullmatch(r"python\d*(?:\.\d+)?", name))


def _is_likely_script_path(path: Path) -> bool:
    return path.suffix.lower() in _SCRIPT_EXTENSIONS or path.exists()


def _infer_action(script_path: Path) -> str:
    stem = script_path.stem.strip()
    return stem or "run"


def _resolve_path(path: Path) -> Path:
    try:
        return path.expanduser().resolve()
    except OSError:
        return path.expanduser().absolute()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _join_url(base_url: str, endpoint: str) -> str:
    base = str(base_url or "").strip().rstrip("/")
    path = str(endpoint or "").strip()
    if not path:
        return base
    if not path.startswith("/"):
        path = "/" + path
    return f"{base}{path}"


def _forward_headers(headers: Mapping[str, str] | None) -> dict[str, str]:
    allow = {
        "uac-user-id",
        "uac-user-token",
        "api-key",
        "x-api-key",
        "x-user-id",
        "x-uac-user-id",
        "x-session-id",
        "x-chat-session-id",
        "x-request-id",
    }
    out: dict[str, str] = {}
    for key, value in dict(headers or {}).items():
        name = str(key or "").strip()
        if name.lower() in allow and str(value or "").strip():
            out[name] = str(value)
    return out


def _truncate(value: str, max_len: int) -> str:
    text = str(value or "")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
