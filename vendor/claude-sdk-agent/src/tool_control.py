from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Deque, Mapping

_STREAM_END = object()
_STATUS_END = object()
_STREAM_NAMES = {"stdout", "stderr", "system"}


@dataclass(frozen=True)
class ToolOutputChunk:
    sequence: int
    stream: str
    text: str
    timestamp: float


@dataclass
class _ToolRecord:
    run_id: str
    tool_call_id: str
    name: str
    display_name: str
    tool_type: str
    arguments: Mapping[str, Any]
    status: str = "running"
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    output_sequence: int = 0
    output_closed: bool = False
    output_buffer: Deque[ToolOutputChunk] = field(default_factory=deque)
    output_subscribers: set[asyncio.Queue[ToolOutputChunk | object]] = field(default_factory=set)
    status_subscribers: set[asyncio.Queue[Mapping[str, Any] | object]] = field(default_factory=set)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass
class _RunRecord:
    run_id: str
    session_id: str | None
    status: str = "running"
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    tools: dict[str, _ToolRecord] = field(default_factory=dict)


class ToolControlContext:
    def __init__(self, record: _ToolRecord, *, max_buffer_chunks: int) -> None:
        self._record = record
        self._max_buffer_chunks = max(1, int(max_buffer_chunks))

    async def emit_output(self, stream: str, text: str) -> None:
        normalized = str(stream or "system").strip().lower() or "system"
        if normalized not in _STREAM_NAMES:
            normalized = "system"
        if not text:
            return
        async with self._record.lock:
            self._record.output_sequence += 1
            chunk = ToolOutputChunk(
                sequence=self._record.output_sequence,
                stream=normalized,
                text=text,
                timestamp=time.time(),
            )
            self._record.output_buffer.append(chunk)
            while len(self._record.output_buffer) > self._max_buffer_chunks:
                self._record.output_buffer.popleft()
            subscribers = list(self._record.output_subscribers)
        for queue in subscribers:
            await queue.put(chunk)

    async def update_arguments(self, arguments: Mapping[str, Any]) -> None:
        async with self._record.lock:
            self._record.arguments = dict(arguments)
            snapshot = self.snapshot()
            subscribers = list(self._record.status_subscribers)
        for queue in subscribers:
            await queue.put(snapshot)

    async def update_status(self, status: str) -> None:
        async with self._record.lock:
            self._record.status = str(status or "running").strip() or "running"
            snapshot = self.snapshot()
            subscribers = list(self._record.status_subscribers)
        for queue in subscribers:
            await queue.put(snapshot)

    async def notify_status(self) -> None:
        snapshot = self.snapshot()
        async with self._record.lock:
            subscribers = list(self._record.status_subscribers)
        for queue in subscribers:
            await queue.put(snapshot)

    async def close_output(self) -> None:
        async with self._record.lock:
            if self._record.output_closed:
                return
            self._record.output_closed = True
            subscribers = list(self._record.output_subscribers)
        for queue in subscribers:
            await queue.put(_STREAM_END)

    async def close_status(self) -> None:
        async with self._record.lock:
            subscribers = list(self._record.status_subscribers)
        for queue in subscribers:
            await queue.put(_STATUS_END)

    def snapshot(self) -> Mapping[str, Any]:
        return {
            "runId": self._record.run_id,
            "toolCallId": self._record.tool_call_id,
            "name": self._record.name,
            "displayName": self._record.display_name,
            "toolType": self._record.tool_type,
            "arguments": dict(self._record.arguments),
            "status": self._record.status,
            "startedAt": self._record.started_at,
            "finishedAt": self._record.finished_at,
        }


class ToolRuntimeRegistry:
    def __init__(self, *, max_output_buffer_chunks: int = 4000) -> None:
        self._max_output_buffer_chunks = max(32, int(max_output_buffer_chunks))
        self._runs: dict[str, _RunRecord] = {}
        self._lock = asyncio.Lock()

    async def register_run(self, run_id: str, *, session_id: str | None = None) -> None:
        if not run_id:
            return
        async with self._lock:
            existing = self._runs.get(run_id)
            if existing is not None:
                existing.session_id = session_id or existing.session_id
                existing.status = "running"
                existing.finished_at = None
                return
            self._runs[run_id] = _RunRecord(run_id=run_id, session_id=session_id)

    async def finish_run(self, run_id: str, *, status: str) -> None:
        if not run_id:
            return
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            run.status = str(status or "completed").strip() or "completed"
            run.finished_at = time.time()

    async def runtime_snapshot(self, *, include_runs: bool = False) -> Mapping[str, Any]:
        async with self._lock:
            active_runs = [
                {
                    "runId": run.run_id,
                    "sessionId": run.session_id or "",
                    "status": run.status,
                    "createdAt": run.created_at,
                    "toolsCount": len(run.tools),
                }
                for run in self._runs.values()
                if run.status == "running"
            ]
        snapshot: dict[str, Any] = {
            "agentTaskNum": len(active_runs),
        }
        if include_runs:
            snapshot["activeRuns"] = active_runs
        return snapshot

    async def start_tool(
        self,
        *,
        run_id: str,
        tool_call_id: str,
        name: str,
        display_name: str,
        tool_type: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> ToolControlContext:
        async with self._lock:
            run = self._runs.setdefault(run_id, _RunRecord(run_id=run_id, session_id=None))
            record = run.tools.get(tool_call_id)
            if record is None:
                record = _ToolRecord(
                    run_id=run_id,
                    tool_call_id=tool_call_id,
                    name=name,
                    display_name=display_name,
                    tool_type=tool_type,
                    arguments=dict(arguments or {}),
                )
                run.tools[tool_call_id] = record
            else:
                record.name = name or record.name
                record.display_name = display_name or record.display_name
                record.tool_type = tool_type or record.tool_type
                record.arguments = dict(arguments or record.arguments)
                record.status = "running"
                if record.finished_at is not None:
                    record.finished_at = None
                    record.output_closed = False
        context = ToolControlContext(record, max_buffer_chunks=self._max_output_buffer_chunks)
        await context.notify_status()
        return context

    async def finish_tool(self, *, run_id: str, tool_call_id: str, status: str) -> None:
        record = await self._get_tool_record(run_id, tool_call_id)
        if record is None:
            return
        record.status = str(status or "completed").strip() or "completed"
        record.finished_at = time.time()
        context = ToolControlContext(record, max_buffer_chunks=self._max_output_buffer_chunks)
        await context.notify_status()
        await context.close_status()
        await context.close_output()

    async def get_tool(self, *, run_id: str, tool_call_id: str) -> Mapping[str, Any] | None:
        record = await self._get_tool_record(run_id, tool_call_id)
        if record is None:
            return None
        return ToolControlContext(record, max_buffer_chunks=self._max_output_buffer_chunks).snapshot()

    async def stream_tool_output(self, *, run_id: str, tool_call_id: str) -> AsyncIterator[ToolOutputChunk]:
        record = await self._get_tool_record(run_id, tool_call_id)
        if record is None:
            raise KeyError(f"tool not found: run={run_id} tool={tool_call_id}")
        queue: asyncio.Queue[ToolOutputChunk | object] = asyncio.Queue()
        async with record.lock:
            for item in list(record.output_buffer):
                await queue.put(item)
            if record.output_closed:
                await queue.put(_STREAM_END)
            else:
                record.output_subscribers.add(queue)
        try:
            while True:
                item = await queue.get()
                if item is _STREAM_END:
                    break
                if isinstance(item, ToolOutputChunk):
                    yield item
        finally:
            async with record.lock:
                record.output_subscribers.discard(queue)

    async def stream_tool_status(self, *, run_id: str, tool_call_id: str) -> AsyncIterator[Mapping[str, Any]]:
        record = await self._get_tool_record(run_id, tool_call_id)
        if record is None:
            raise KeyError(f"tool not found: run={run_id} tool={tool_call_id}")
        queue: asyncio.Queue[Mapping[str, Any] | object] = asyncio.Queue()
        snapshot = ToolControlContext(record, max_buffer_chunks=self._max_output_buffer_chunks).snapshot()
        async with record.lock:
            await queue.put(snapshot)
            if record.finished_at is not None:
                await queue.put(_STATUS_END)
            else:
                record.status_subscribers.add(queue)
        try:
            while True:
                item = await queue.get()
                if item is _STATUS_END:
                    break
                if isinstance(item, Mapping):
                    yield item
        finally:
            async with record.lock:
                record.status_subscribers.discard(queue)

    async def _get_tool_record(self, run_id: str, tool_call_id: str) -> _ToolRecord | None:
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return None
            return run.tools.get(tool_call_id)
