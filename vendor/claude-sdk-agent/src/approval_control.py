from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Mapping


@dataclass(frozen=True)
class ApprovalDecision:
    decision: str
    reason: str
    interrupt: bool
    timestamp: float


@dataclass
class _ApprovalRecord:
    session_id: str
    run_id: str
    request_id: str
    claude_session_id: str
    tool_name: str
    tool_input: Mapping[str, Any]
    tool_use_id: str
    agent_id: str
    blocked_path: str
    decision_reason: str
    title: str
    display_name: str
    description: str
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    decided_at: float | None = None
    decision: str = ""
    reason: str = ""
    interrupt: bool = False
    response_future: asyncio.Future[ApprovalDecision] | None = None
    subscribers: set[asyncio.Queue[Mapping[str, Any] | object]] = field(default_factory=set)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class ApprovalRequestContext:
    def __init__(self, record: _ApprovalRecord) -> None:
        self._record = record

    @property
    def request_id(self) -> str:
        return self._record.request_id

    def snapshot(self) -> Mapping[str, Any]:
        return _approval_snapshot(self._record)

    async def wait_for_decision(self) -> ApprovalDecision:
        future = self._record.response_future
        if future is None:
            raise RuntimeError("approval request is not awaiting a decision")
        return await future


class ApprovalRuntimeRegistry:
    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, _ApprovalRecord]] = {}
        self._session_subscribers: dict[str, set[asyncio.Queue[Mapping[str, Any]]]] = {}
        self._lock = asyncio.Lock()

    async def runtime_snapshot(self, *, include_sessions: bool = False) -> Mapping[str, Any]:
        async with self._lock:
            sessions = {
                session_id: sum(1 for record in records.values() if record.status == "pending")
                for session_id, records in self._sessions.items()
                if records
            }
        payload: dict[str, Any] = {
            "pendingApprovalNum": sum(sessions.values()),
            "approvalSessionNum": len(sessions),
        }
        if include_sessions:
            payload["sessions"] = [
                {"sessionId": session_id, "pendingApprovalNum": pending_count}
                for session_id, pending_count in sessions.items()
            ]
        return payload

    async def create_request(
        self,
        *,
        session_id: str,
        run_id: str,
        claude_session_id: str,
        tool_name: str,
        tool_input: Mapping[str, Any] | None,
        tool_use_id: str,
        agent_id: str,
        blocked_path: str,
        decision_reason: str,
        title: str,
        display_name: str,
        description: str,
    ) -> ApprovalRequestContext:
        request_id = f"req-{uuid.uuid4().hex}"
        record = _ApprovalRecord(
            session_id=session_id,
            run_id=run_id,
            request_id=request_id,
            claude_session_id=claude_session_id,
            tool_name=tool_name,
            tool_input=dict(tool_input or {}),
            tool_use_id=tool_use_id,
            agent_id=agent_id,
            blocked_path=blocked_path,
            decision_reason=decision_reason,
            title=title,
            display_name=display_name or tool_name,
            description=description,
            response_future=asyncio.get_running_loop().create_future(),
        )
        async with self._lock:
            session_records = self._sessions.setdefault(session_id, {})
            session_records[request_id] = record
        await self._notify(record)
        return ApprovalRequestContext(record)

    async def list_requests(self, session_id: str) -> list[Mapping[str, Any]]:
        async with self._lock:
            records = list(self._sessions.get(session_id, {}).values())
        records.sort(key=lambda item: item.created_at)
        return [_approval_snapshot(record) for record in records]

    async def get_request(self, session_id: str, request_id: str) -> Mapping[str, Any] | None:
        record = await self._get_record(session_id, request_id)
        if record is None:
            return None
        return _approval_snapshot(record)

    async def resolve_request(
        self,
        session_id: str,
        request_id: str,
        *,
        decision: str,
        reason: str = "",
        interrupt: bool = False,
    ) -> Mapping[str, Any] | None:
        record = await self._get_record(session_id, request_id)
        if record is None:
            return None
        normalized = str(decision or "").strip().lower()
        if normalized not in {"allow", "deny"}:
            raise ValueError("decision must be allow or deny")
        async with record.lock:
            if record.status != "pending":
                return _approval_snapshot(record)
            record.status = "allowed" if normalized == "allow" else "denied"
            record.decision = normalized
            record.reason = str(reason or "").strip()
            record.interrupt = bool(interrupt)
            record.decided_at = time.time()
            future = record.response_future
            subscribers = list(record.subscribers)
        if future is not None and not future.done():
            future.set_result(
                ApprovalDecision(
                    decision=normalized,
                    reason=record.reason,
                    interrupt=record.interrupt,
                    timestamp=record.decided_at or time.time(),
                )
            )
        payload = _approval_snapshot(record)
        for queue in subscribers:
            await queue.put(payload)
        await self._notify_session_subscribers(record.session_id, payload)
        return payload

    async def cancel_session(self, session_id: str, *, reason: str) -> int:
        async with self._lock:
            records = list(self._sessions.get(session_id, {}).values())
        cancelled = 0
        for record in records:
            if record.status != "pending":
                continue
            result = await self.resolve_request(
                session_id,
                record.request_id,
                decision="deny",
                reason=reason,
                interrupt=True,
            )
            if result is not None:
                cancelled += 1
        return cancelled

    async def stream_requests(self, session_id: str) -> AsyncIterator[Mapping[str, Any]]:
        queue: asyncio.Queue[Mapping[str, Any]] = asyncio.Queue()
        async with self._lock:
            session_records = list(self._sessions.get(session_id, {}).values())
            self._session_subscribers.setdefault(session_id, set()).add(queue)
        session_records.sort(key=lambda item: item.created_at)
        for record in session_records:
            await queue.put(_approval_snapshot(record))
        try:
            while True:
                item = await queue.get()
                yield item
        finally:
            async with self._lock:
                self._session_subscribers.get(session_id, set()).discard(queue)

    async def _get_record(self, session_id: str, request_id: str) -> _ApprovalRecord | None:
        async with self._lock:
            return self._sessions.get(session_id, {}).get(request_id)

    async def _notify(self, record: _ApprovalRecord) -> None:
        async with record.lock:
            subscribers = list(record.subscribers)
            payload = _approval_snapshot(record)
        for queue in subscribers:
            await queue.put(payload)
        await self._notify_session_subscribers(record.session_id, payload)

    async def _notify_session_subscribers(self, session_id: str, payload: Mapping[str, Any]) -> None:
        async with self._lock:
            subscribers = list(self._session_subscribers.get(session_id, set()))
        for queue in subscribers:
            await queue.put(payload)


def _approval_snapshot(record: _ApprovalRecord) -> Mapping[str, Any]:
    return {
        "sessionId": record.session_id,
        "runId": record.run_id,
        "requestId": record.request_id,
        "claudeSessionId": record.claude_session_id,
        "toolName": record.tool_name,
        "toolInput": dict(record.tool_input),
        "toolUseId": record.tool_use_id,
        "agentId": record.agent_id,
        "blockedPath": record.blocked_path,
        "decisionReason": record.decision_reason,
        "title": record.title,
        "displayName": record.display_name,
        "description": record.description,
        "status": record.status,
        "decision": record.decision,
        "reason": record.reason,
        "interrupt": record.interrupt,
        "createdAt": record.created_at,
        "decidedAt": record.decided_at,
    }
