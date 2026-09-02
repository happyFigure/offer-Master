from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Mapping


@dataclass
class _HookRecord:
    session_id: str
    run_id: str
    event_id: str
    claude_session_id: str
    hook_event_name: str
    phase: str
    source: str
    status: str
    matcher: str
    tool_name: str
    tool_use_id: str
    agent_id: str
    agent_type: str
    title: str
    notification_type: str
    data: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    outcome: str = ""
    exit_code: int | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class HookRuntimeRegistry:
    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, _HookRecord]] = {}
        self._session_subscribers: dict[str, set[asyncio.Queue[Mapping[str, Any]]]] = {}
        self._lock = asyncio.Lock()

    async def runtime_snapshot(self, *, include_sessions: bool = False) -> Mapping[str, Any]:
        async with self._lock:
            sessions = {
                session_id: len(records)
                for session_id, records in self._sessions.items()
                if records
            }
        payload: dict[str, Any] = {
            "hookEventNum": sum(sessions.values()),
            "hookSessionNum": len(sessions),
        }
        if include_sessions:
            payload["sessions"] = [
                {"sessionId": session_id, "hookEventNum": count}
                for session_id, count in sessions.items()
            ]
        return payload

    async def record_event(
        self,
        *,
        session_id: str,
        run_id: str,
        claude_session_id: str,
        event_id: str | None,
        hook_event_name: str,
        phase: str,
        source: str,
        status: str,
        matcher: str = "",
        tool_name: str = "",
        tool_use_id: str = "",
        agent_id: str = "",
        agent_type: str = "",
        title: str = "",
        notification_type: str = "",
        data: Mapping[str, Any] | None = None,
        output: Mapping[str, Any] | None = None,
        outcome: str = "",
        exit_code: int | None = None,
    ) -> Mapping[str, Any]:
        normalized_session_id = str(session_id or "").strip()
        normalized_event_id = str(event_id or "").strip() or f"hook-{uuid.uuid4().hex}"
        now = time.time()
        async with self._lock:
            session_records = self._sessions.setdefault(normalized_session_id, {})
            record = session_records.get(normalized_event_id)
            if record is None:
                record = _HookRecord(
                    session_id=normalized_session_id,
                    run_id=str(run_id or "").strip(),
                    event_id=normalized_event_id,
                    claude_session_id=str(claude_session_id or "").strip(),
                    hook_event_name=str(hook_event_name or "").strip(),
                    phase=str(phase or "").strip() or "event",
                    source=str(source or "").strip() or "unknown",
                    status=str(status or "").strip() or "completed",
                    matcher=str(matcher or "").strip(),
                    tool_name=str(tool_name or "").strip(),
                    tool_use_id=str(tool_use_id or "").strip(),
                    agent_id=str(agent_id or "").strip(),
                    agent_type=str(agent_type or "").strip(),
                    title=str(title or "").strip(),
                    notification_type=str(notification_type or "").strip(),
                    data=dict(data or {}),
                    output=dict(output or {}),
                    outcome=str(outcome or "").strip(),
                    exit_code=exit_code,
                    created_at=now,
                    updated_at=now,
                )
                session_records[normalized_event_id] = record
            else:
                record.run_id = str(run_id or record.run_id).strip() or record.run_id
                record.claude_session_id = str(claude_session_id or record.claude_session_id).strip() or record.claude_session_id
                record.hook_event_name = str(hook_event_name or record.hook_event_name).strip() or record.hook_event_name
                record.phase = str(phase or record.phase).strip() or record.phase
                record.source = str(source or record.source).strip() or record.source
                record.status = str(status or record.status).strip() or record.status
                record.matcher = str(matcher or record.matcher).strip() or record.matcher
                record.tool_name = str(tool_name or record.tool_name).strip() or record.tool_name
                record.tool_use_id = str(tool_use_id or record.tool_use_id).strip() or record.tool_use_id
                record.agent_id = str(agent_id or record.agent_id).strip() or record.agent_id
                record.agent_type = str(agent_type or record.agent_type).strip() or record.agent_type
                record.title = str(title or record.title).strip() or record.title
                record.notification_type = str(notification_type or record.notification_type).strip() or record.notification_type
                if isinstance(data, Mapping):
                    record.data = dict(data)
                if isinstance(output, Mapping):
                    record.output = dict(output)
                record.outcome = str(outcome or record.outcome).strip() or record.outcome
                record.exit_code = exit_code if exit_code is not None else record.exit_code
                record.updated_at = now
        payload = _hook_snapshot(record)
        await self._notify_session_subscribers(normalized_session_id, payload)
        return payload

    async def list_events(self, session_id: str) -> list[Mapping[str, Any]]:
        async with self._lock:
            records = list(self._sessions.get(session_id, {}).values())
        records.sort(key=lambda item: (item.created_at, item.event_id))
        return [_hook_snapshot(record) for record in records]

    async def get_event(self, session_id: str, event_id: str) -> Mapping[str, Any] | None:
        async with self._lock:
            record = self._sessions.get(session_id, {}).get(event_id)
        if record is None:
            return None
        return _hook_snapshot(record)

    async def stream_events(self, session_id: str) -> AsyncIterator[Mapping[str, Any]]:
        queue: asyncio.Queue[Mapping[str, Any]] = asyncio.Queue()
        async with self._lock:
            session_records = list(self._sessions.get(session_id, {}).values())
            self._session_subscribers.setdefault(session_id, set()).add(queue)
        session_records.sort(key=lambda item: (item.created_at, item.event_id))
        for record in session_records:
            await queue.put(_hook_snapshot(record))
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


def hook_shell_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    body = {
        "eventId": str(payload.get("eventId") or "").strip(),
        "sessionId": str(payload.get("sessionId") or "").strip(),
        "runId": str(payload.get("runId") or "").strip(),
        "claudeSessionId": str(payload.get("claudeSessionId") or "").strip(),
        "hookEventName": str(payload.get("hookEventName") or "").strip(),
        "phase": str(payload.get("phase") or "").strip(),
        "source": str(payload.get("source") or "").strip(),
        "status": str(payload.get("status") or "").strip(),
    }
    for key in ("toolName", "toolUseId", "agentId", "agentType", "title", "notificationType"):
        value = str(payload.get(key) or "").strip()
        if value:
            body[key] = value
    return body


def _hook_snapshot(record: _HookRecord) -> Mapping[str, Any]:
    return {
        "sessionId": record.session_id,
        "runId": record.run_id,
        "eventId": record.event_id,
        "claudeSessionId": record.claude_session_id,
        "hookEventName": record.hook_event_name,
        "phase": record.phase,
        "source": record.source,
        "status": record.status,
        "matcher": record.matcher,
        "toolName": record.tool_name,
        "toolUseId": record.tool_use_id,
        "agentId": record.agent_id,
        "agentType": record.agent_type,
        "title": record.title,
        "notificationType": record.notification_type,
        "data": dict(record.data),
        "output": dict(record.output),
        "outcome": record.outcome,
        "exitCode": record.exit_code,
        "createdAt": record.created_at,
        "updatedAt": record.updated_at,
    }
