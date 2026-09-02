from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Mapping


@dataclass
class _GoalRecord:
    session_id: str
    goal_id: str
    current_run_id: str
    status: str
    command: str
    condition: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    cleared_at: float | None = None
    last_reason: str = ""
    stop_hook_active: bool = False
    turn_count: int = 0
    pending_approval: bool = False
    pending_question: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class GoalRuntimeRegistry:
    def __init__(self) -> None:
        self._sessions: dict[str, _GoalRecord] = {}
        self._session_subscribers: dict[str, set[asyncio.Queue[Mapping[str, Any]]]] = {}
        self._lock = asyncio.Lock()

    async def runtime_snapshot(self, *, include_sessions: bool = False) -> Mapping[str, Any]:
        async with self._lock:
            records = list(self._sessions.values())
        payload: dict[str, Any] = {
            "goalSessionNum": len(records),
            "activeGoalNum": sum(1 for item in records if item.status in {"active", "waiting_input"}),
        }
        if include_sessions:
            payload["sessions"] = [_goal_snapshot(item) for item in records]
        return payload

    async def start_goal(
        self,
        *,
        session_id: str,
        run_id: str,
        condition: str,
        command: str,
    ) -> Mapping[str, Any]:
        normalized_session = str(session_id or "").strip()
        now = time.time()
        async with self._lock:
            existing = self._sessions.get(normalized_session)
            if existing is None:
                record = _GoalRecord(
                    session_id=normalized_session,
                    goal_id=f"goal-{uuid.uuid4().hex}",
                    current_run_id=str(run_id or "").strip(),
                    status="active",
                    command=str(command or "").strip(),
                    condition=str(condition or "").strip(),
                    created_at=now,
                    updated_at=now,
                )
                self._sessions[normalized_session] = record
            else:
                record = existing
                record.current_run_id = str(run_id or record.current_run_id).strip() or record.current_run_id
                record.status = "active"
                record.command = str(command or record.command).strip() or record.command
                record.condition = str(condition or record.condition).strip() or record.condition
                record.updated_at = now
                record.completed_at = None
                record.cleared_at = None
                record.last_reason = ""
                record.stop_hook_active = False
                record.pending_approval = False
                record.pending_question = False
        payload = _goal_snapshot(record)
        await self._notify_session_subscribers(normalized_session, payload)
        return payload

    async def get_goal(self, session_id: str) -> Mapping[str, Any] | None:
        async with self._lock:
            record = self._sessions.get(session_id)
        if record is None:
            return None
        return _goal_snapshot(record)

    async def clear_goal(
        self,
        session_id: str,
        *,
        run_id: str = "",
        reason: str = "",
    ) -> Mapping[str, Any] | None:
        async with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                return None
            record.current_run_id = str(run_id or record.current_run_id).strip() or record.current_run_id
            record.status = "cleared"
            record.updated_at = time.time()
            record.cleared_at = record.updated_at
            record.last_reason = str(reason or record.last_reason).strip()
            record.stop_hook_active = False
            record.pending_approval = False
            record.pending_question = False
        payload = _goal_snapshot(record)
        await self._notify_session_subscribers(session_id, payload)
        return payload

    async def query_goal(self, session_id: str, *, run_id: str = "") -> Mapping[str, Any] | None:
        async with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                return None
            if run_id:
                record.current_run_id = str(run_id).strip()
            record.updated_at = time.time()
        payload = _goal_snapshot(record)
        await self._notify_session_subscribers(session_id, payload)
        return payload

    async def mark_waiting(
        self,
        session_id: str,
        *,
        run_id: str = "",
        reason: str = "",
        pending_approval: bool | None = None,
        pending_question: bool | None = None,
    ) -> Mapping[str, Any] | None:
        async with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                return None
            record.current_run_id = str(run_id or record.current_run_id).strip() or record.current_run_id
            record.status = "waiting_input"
            record.updated_at = time.time()
            if reason:
                record.last_reason = str(reason).strip()
            if pending_approval is not None:
                record.pending_approval = bool(pending_approval)
            if pending_question is not None:
                record.pending_question = bool(pending_question)
        payload = _goal_snapshot(record)
        await self._notify_session_subscribers(session_id, payload)
        return payload

    async def mark_active(
        self,
        session_id: str,
        *,
        run_id: str = "",
        reason: str = "",
        stop_hook_active: bool | None = None,
        pending_approval: bool | None = None,
        pending_question: bool | None = None,
        increment_turn: bool = False,
    ) -> Mapping[str, Any] | None:
        async with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                return None
            record.current_run_id = str(run_id or record.current_run_id).strip() or record.current_run_id
            record.status = "active"
            record.updated_at = time.time()
            if reason:
                record.last_reason = str(reason).strip()
            if stop_hook_active is not None:
                record.stop_hook_active = bool(stop_hook_active)
            if pending_approval is not None:
                record.pending_approval = bool(pending_approval)
            if pending_question is not None:
                record.pending_question = bool(pending_question)
            if increment_turn:
                record.turn_count += 1
        payload = _goal_snapshot(record)
        await self._notify_session_subscribers(session_id, payload)
        return payload

    async def mark_completed(
        self,
        session_id: str,
        *,
        run_id: str = "",
        reason: str = "",
    ) -> Mapping[str, Any] | None:
        async with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                return None
            record.current_run_id = str(run_id or record.current_run_id).strip() or record.current_run_id
            record.status = "completed"
            record.updated_at = time.time()
            record.completed_at = record.updated_at
            record.last_reason = str(reason or record.last_reason).strip()
            record.stop_hook_active = False
            record.pending_approval = False
            record.pending_question = False
        payload = _goal_snapshot(record)
        await self._notify_session_subscribers(session_id, payload)
        return payload

    async def stream_goals(self, session_id: str) -> AsyncIterator[Mapping[str, Any]]:
        queue: asyncio.Queue[Mapping[str, Any]] = asyncio.Queue()
        async with self._lock:
            record = self._sessions.get(session_id)
            self._session_subscribers.setdefault(session_id, set()).add(queue)
        if record is not None:
            await queue.put(_goal_snapshot(record))
        try:
            while True:
                payload = await queue.get()
                yield payload
        finally:
            async with self._lock:
                self._session_subscribers.get(session_id, set()).discard(queue)

    async def _notify_session_subscribers(self, session_id: str, payload: Mapping[str, Any]) -> None:
        async with self._lock:
            subscribers = list(self._session_subscribers.get(session_id, set()))
        for queue in subscribers:
            await queue.put(payload)


def goal_shell_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    body = {
        "sessionId": str(payload.get("sessionId") or "").strip(),
        "goalId": str(payload.get("goalId") or "").strip(),
        "status": str(payload.get("status") or "").strip(),
    }
    for key in ("currentRunId", "condition", "lastReason"):
        value = str(payload.get(key) or "").strip()
        if value:
            body[key] = value
    for key in ("turnCount",):
        if isinstance(payload.get(key), int):
            body[key] = payload.get(key)
    for key in ("pendingApproval", "pendingQuestion", "stopHookActive"):
        if isinstance(payload.get(key), bool):
            body[key] = payload.get(key)
    return body


def _goal_snapshot(record: _GoalRecord) -> Mapping[str, Any]:
    return {
        "sessionId": record.session_id,
        "goalId": record.goal_id,
        "currentRunId": record.current_run_id,
        "status": record.status,
        "command": record.command,
        "condition": record.condition,
        "createdAt": record.created_at,
        "updatedAt": record.updated_at,
        "completedAt": record.completed_at,
        "clearedAt": record.cleared_at,
        "lastReason": record.last_reason,
        "stopHookActive": record.stop_hook_active,
        "turnCount": record.turn_count,
        "pendingApproval": record.pending_approval,
        "pendingQuestion": record.pending_question,
        "metadata": dict(record.metadata),
    }
