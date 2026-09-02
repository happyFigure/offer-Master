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
class TaskOutputChunk:
    sequence: int
    stream: str
    text: str
    timestamp: float


@dataclass
class _TaskRecord:
    run_id: str
    task_id: str
    description: str
    task_type: str
    tool_call_id: str | None
    status: str = "running"
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    output_sequence: int = 0
    output_closed: bool = False
    output_buffer: Deque[TaskOutputChunk] = field(default_factory=deque)
    output_subscribers: set[asyncio.Queue[TaskOutputChunk | object]] = field(default_factory=set)
    status_subscribers: set[asyncio.Queue[Mapping[str, Any] | object]] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class TaskControlContext:
    def __init__(self, record: _TaskRecord, *, max_buffer_chunks: int) -> None:
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
            chunk = TaskOutputChunk(
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

    async def update_status(self, status: str, *, metadata: Mapping[str, Any] | None = None) -> None:
        async with self._record.lock:
            self._record.status = str(status or "running").strip() or "running"
            if isinstance(metadata, Mapping):
                self._record.metadata.update(dict(metadata))
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
            "taskId": self._record.task_id,
            "description": self._record.description,
            "taskType": self._record.task_type,
            "toolCallId": self._record.tool_call_id or "",
            "status": self._record.status,
            "startedAt": self._record.started_at,
            "finishedAt": self._record.finished_at,
            "metadata": dict(self._record.metadata),
        }


class TaskRuntimeRegistry:
    def __init__(self, *, max_output_buffer_chunks: int = 4000) -> None:
        self._max_output_buffer_chunks = max(32, int(max_output_buffer_chunks))
        self._runs: dict[str, dict[str, _TaskRecord]] = {}
        self._lock = asyncio.Lock()

    async def register_run(self, run_id: str) -> None:
        if not run_id:
            return
        async with self._lock:
            self._runs.setdefault(run_id, {})

    async def runtime_snapshot(self, *, include_runs: bool = False) -> Mapping[str, Any]:
        async with self._lock:
            runs = {
                run_id: len(tasks)
                for run_id, tasks in self._runs.items()
                if tasks
            }
        payload: dict[str, Any] = {"taskRunNum": len(runs)}
        if include_runs:
            payload["activeTaskRuns"] = [
                {"runId": run_id, "tasksCount": count}
                for run_id, count in runs.items()
            ]
        return payload

    async def start_task(
        self,
        *,
        run_id: str,
        task_id: str,
        description: str,
        task_type: str,
        tool_call_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TaskControlContext:
        async with self._lock:
            tasks = self._runs.setdefault(run_id, {})
            record = tasks.get(task_id)
            if record is None:
                record = _TaskRecord(
                    run_id=run_id,
                    task_id=task_id,
                    description=description,
                    task_type=task_type,
                    tool_call_id=tool_call_id,
                    metadata=dict(metadata or {}),
                )
                tasks[task_id] = record
            else:
                record.description = description or record.description
                record.task_type = task_type or record.task_type
                record.tool_call_id = tool_call_id or record.tool_call_id
                if isinstance(metadata, Mapping):
                    record.metadata.update(dict(metadata))
                record.status = "running"
                record.finished_at = None
        context = TaskControlContext(record, max_buffer_chunks=self._max_output_buffer_chunks)
        await context.notify_status()
        return context

    async def finish_task(self, *, run_id: str, task_id: str, status: str, metadata: Mapping[str, Any] | None = None) -> None:
        record = await self._get_task_record(run_id, task_id)
        if record is None:
            return
        record.status = str(status or "completed").strip() or "completed"
        record.finished_at = time.time()
        if isinstance(metadata, Mapping):
            record.metadata.update(dict(metadata))
        context = TaskControlContext(record, max_buffer_chunks=self._max_output_buffer_chunks)
        await context.notify_status()
        await context.close_status()
        await context.close_output()

    async def get_task(self, *, run_id: str, task_id: str) -> Mapping[str, Any] | None:
        record = await self._get_task_record(run_id, task_id)
        if record is None:
            return None
        return TaskControlContext(record, max_buffer_chunks=self._max_output_buffer_chunks).snapshot()

    async def list_run_tasks(self, run_id: str) -> list[Mapping[str, Any]]:
        async with self._lock:
            tasks = list(self._runs.get(run_id, {}).values())
        return [
            TaskControlContext(record, max_buffer_chunks=self._max_output_buffer_chunks).snapshot()
            for record in tasks
        ]

    async def stream_task_output(self, *, run_id: str, task_id: str) -> AsyncIterator[TaskOutputChunk]:
        record = await self._get_task_record(run_id, task_id)
        if record is None:
            raise KeyError(f"task not found: run={run_id} task={task_id}")
        queue: asyncio.Queue[TaskOutputChunk | object] = asyncio.Queue()
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
                if isinstance(item, TaskOutputChunk):
                    yield item
        finally:
            async with record.lock:
                record.output_subscribers.discard(queue)

    async def stream_task_status(self, *, run_id: str, task_id: str) -> AsyncIterator[Mapping[str, Any]]:
        record = await self._get_task_record(run_id, task_id)
        if record is None:
            raise KeyError(f"task not found: run={run_id} task={task_id}")
        queue: asyncio.Queue[Mapping[str, Any] | object] = asyncio.Queue()
        snapshot = TaskControlContext(record, max_buffer_chunks=self._max_output_buffer_chunks).snapshot()
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

    async def _get_task_record(self, run_id: str, task_id: str) -> _TaskRecord | None:
        async with self._lock:
            tasks = self._runs.get(run_id)
            if tasks is None:
                return None
            return tasks.get(task_id)
