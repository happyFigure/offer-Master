from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Mapping

from .models import SessionGoal


class SessionGoalStore:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._lock = asyncio.Lock()
        self._memory: dict[str, object] = {}

    async def get(self, frontend_session_id: str) -> SessionGoal | None:
        session_id = str(frontend_session_id or "").strip()
        if not session_id:
            return None
        async with self._lock:
            data = self._read_all()
        item = data.get(session_id)
        if not isinstance(item, dict):
            return None
        goal = _goal_from_dict(session_id, item)
        if goal.status == "cleared":
            return None
        return goal

    async def set(self, frontend_session_id: str, objective: str) -> SessionGoal:
        session_id = str(frontend_session_id or "").strip()
        clean_objective = str(objective or "").strip()
        now = time.time()
        goal = SessionGoal(
            frontend_session_id=session_id,
            objective=clean_objective,
            status="running",
            created_at=now,
            updated_at=now,
        )
        async with self._lock:
            data = self._read_all()
            data[session_id] = goal.to_dict()
            self._write_all(data)
        return goal

    async def clear(self, frontend_session_id: str) -> SessionGoal | None:
        session_id = str(frontend_session_id or "").strip()
        if not session_id:
            return None
        now = time.time()
        async with self._lock:
            data = self._read_all()
            item = data.get(session_id)
            if not isinstance(item, dict):
                return None
            goal = _goal_from_dict(session_id, item)
            goal.status = "cleared"
            goal.updated_at = now
            data[session_id] = goal.to_dict()
            self._write_all(data)
            return goal

    async def mark_run_started(self, frontend_session_id: str, run_id: str) -> SessionGoal | None:
        session_id = str(frontend_session_id or "").strip()
        if not session_id:
            return None
        async with self._lock:
            data = self._read_all()
            item = data.get(session_id)
            if not isinstance(item, dict):
                return None
            goal = _goal_from_dict(session_id, item)
            if goal.status == "cleared":
                return None
            goal.status = "running"
            goal.active_run_id = str(run_id or "").strip()
            goal.pause_reason = ""
            goal.paused_at = 0.0
            goal.updated_at = time.time()
            data[session_id] = goal.to_dict()
            self._write_all(data)
            return goal

    async def pause(self, frontend_session_id: str, *, reason: str = "user_interrupt") -> SessionGoal | None:
        session_id = str(frontend_session_id or "").strip()
        if not session_id:
            return None
        now = time.time()
        async with self._lock:
            data = self._read_all()
            item = data.get(session_id)
            if not isinstance(item, dict):
                return None
            goal = _goal_from_dict(session_id, item)
            if goal.status not in {"running", "paused"}:
                return None
            goal.status = "paused"
            goal.active_run_id = ""
            goal.pause_reason = str(reason or "user_interrupt").strip()
            goal.paused_at = now
            goal.updated_at = now
            data[session_id] = goal.to_dict()
            self._write_all(data)
            return goal

    async def record_run_result(
        self,
        frontend_session_id: str,
        *,
        run_id: str,
        status: str,
        summary: str = "",
        tasks: list[Mapping[str, Any]] | None = None,
    ) -> SessionGoal | None:
        session_id = str(frontend_session_id or "").strip()
        if not session_id:
            return None
        async with self._lock:
            data = self._read_all()
            item = data.get(session_id)
            if not isinstance(item, dict):
                return None
            goal = _goal_from_dict(session_id, item)
            if goal.status == "cleared":
                return None
            goal.status = _normalize_status(status)
            goal.active_run_id = ""
            goal.last_run_id = str(run_id or "").strip()
            goal.last_summary = str(summary or "").strip()
            goal.tasks = [_task_snapshot(task) for task in tasks or []]
            if goal.status != "paused":
                goal.pause_reason = ""
                goal.paused_at = 0.0
            goal.updated_at = time.time()
            data[session_id] = goal.to_dict()
            self._write_all(data)
            return goal

    async def runtime_snapshot(self, *, include_sessions: bool = False) -> Mapping[str, Any]:
        async with self._lock:
            data = self._read_all()
        goals = []
        for session_id, item in data.items():
            if not isinstance(item, dict):
                continue
            goal = _goal_from_dict(str(session_id), item)
            if goal.status == "cleared":
                continue
            goals.append(goal.to_dict())
        payload: dict[str, Any] = {
            "activeGoalNum": sum(1 for goal in goals if goal.get("status") == "running"),
            "goalSessionNum": len(goals),
        }
        if include_sessions:
            payload["sessions"] = goals
        return payload

    def _read_all(self) -> dict[str, object]:
        if self._path is None:
            return dict(self._memory)
        if not self._path.exists():
            return {}
        text = self._path.read_text(encoding="utf-8").strip()
        if not text:
            return {}
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}

    def _write_all(self, data: dict[str, object]) -> None:
        if self._path is None:
            self._memory = dict(data)
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(self._path)


def _goal_from_dict(session_id: str, item: Mapping[str, Any]) -> SessionGoal:
    return SessionGoal(
        frontend_session_id=str(item.get("frontend_session_id") or session_id).strip(),
        objective=str(item.get("objective") or "").strip(),
        status=_normalize_status(str(item.get("status") or "running")),
        created_at=float(item.get("created_at") or 0.0),
        updated_at=float(item.get("updated_at") or 0.0),
        active_run_id=str(item.get("active_run_id") or "").strip(),
        last_run_id=str(item.get("last_run_id") or "").strip(),
        last_summary=str(item.get("last_summary") or "").strip(),
        pause_reason=str(item.get("pause_reason") or "").strip(),
        paused_at=float(item.get("paused_at") or 0.0),
        tasks=[_task_snapshot(task) for task in item.get("tasks") or [] if isinstance(task, Mapping)],
    )


def _normalize_status(status: str) -> str:
    value = str(status or "").strip().lower()
    if value in {"completed", "failed", "cleared", "paused"}:
        return value
    return "running"


def _task_snapshot(task: Mapping[str, Any]) -> dict[str, object]:
    return {
        "runId": str(task.get("runId") or task.get("run_id") or "").strip(),
        "taskId": str(task.get("taskId") or task.get("task_id") or "").strip(),
        "description": str(task.get("description") or "").strip(),
        "taskType": str(task.get("taskType") or task.get("task_type") or "").strip(),
        "toolCallId": str(task.get("toolCallId") or task.get("tool_call_id") or "").strip(),
        "status": str(task.get("status") or "").strip(),
        "startedAt": float(task.get("startedAt") or task.get("started_at") or 0.0),
        "finishedAt": float(task.get("finishedAt") or task.get("finished_at") or 0.0),
        "metadata": dict(task.get("metadata") or {}) if isinstance(task.get("metadata"), Mapping) else {},
    }
