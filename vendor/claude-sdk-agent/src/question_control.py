from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Mapping


@dataclass(frozen=True)
class QuestionAnswer:
    answer: str
    timestamp: float


@dataclass
class _QuestionRecord:
    session_id: str
    run_id: str
    question_id: str
    request_id: str
    claude_session_id: str
    title: str
    prompt: str
    description: str
    metadata: dict[str, Any]
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    answered_at: float | None = None
    answer: str = ""
    response_future: asyncio.Future[QuestionAnswer] | None = None
    subscribers: set[asyncio.Queue[Mapping[str, Any] | object]] = field(default_factory=set)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class QuestionRequestContext:
    def __init__(self, record: _QuestionRecord) -> None:
        self._record = record

    @property
    def question_id(self) -> str:
        return self._record.question_id

    @property
    def request_id(self) -> str:
        return self._record.request_id

    def snapshot(self) -> Mapping[str, Any]:
        return _question_snapshot(self._record)

    async def wait_for_answer(self) -> QuestionAnswer:
        future = self._record.response_future
        if future is None:
            raise RuntimeError("question request is not awaiting an answer")
        return await future


class QuestionRuntimeRegistry:
    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, _QuestionRecord]] = {}
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
            "pendingQuestionNum": sum(sessions.values()),
            "questionSessionNum": len(sessions),
        }
        if include_sessions:
            payload["sessions"] = [
                {"sessionId": session_id, "pendingQuestionNum": pending_count}
                for session_id, pending_count in sessions.items()
            ]
        return payload

    async def create_question(
        self,
        *,
        session_id: str,
        run_id: str,
        claude_session_id: str,
        prompt: str,
        title: str = "",
        description: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> QuestionRequestContext:
        question_id = f"question-{uuid.uuid4().hex}"
        request_id = f"request-{uuid.uuid4().hex}"
        record = _QuestionRecord(
            session_id=session_id,
            run_id=run_id,
            question_id=question_id,
            request_id=request_id,
            claude_session_id=claude_session_id,
            title=title,
            prompt=prompt,
            description=description,
            metadata=dict(metadata or {}),
            response_future=asyncio.get_running_loop().create_future(),
        )
        async with self._lock:
            session_records = self._sessions.setdefault(session_id, {})
            session_records[question_id] = record
        await self._notify(record)
        return QuestionRequestContext(record)

    async def list_questions(self, session_id: str) -> list[Mapping[str, Any]]:
        async with self._lock:
            records = list(self._sessions.get(session_id, {}).values())
        records.sort(key=lambda item: item.created_at)
        return [_question_snapshot(record) for record in records]

    async def get_question(self, session_id: str, question_id: str) -> Mapping[str, Any] | None:
        record = await self._get_record(session_id, question_id)
        if record is None:
            return None
        return _question_snapshot(record)

    async def answer_question(self, session_id: str, question_id: str, *, answer: str) -> Mapping[str, Any] | None:
        record = await self._get_record(session_id, question_id)
        if record is None:
            return None
        async with record.lock:
            if record.status != "pending":
                return _question_snapshot(record)
            record.status = "answered"
            record.answer = str(answer or "")
            record.answered_at = time.time()
            future = record.response_future
            subscribers = list(record.subscribers)
        if future is not None and not future.done():
            future.set_result(QuestionAnswer(answer=record.answer, timestamp=record.answered_at or time.time()))
        payload = _question_snapshot(record)
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
            async with record.lock:
                record.status = "cancelled"
                record.answer = str(reason or "").strip()
                record.answered_at = time.time()
                future = record.response_future
                subscribers = list(record.subscribers)
            if future is not None and not future.done():
                future.cancel()
            payload = _question_snapshot(record)
            for queue in subscribers:
                await queue.put(payload)
            await self._notify_session_subscribers(record.session_id, payload)
            cancelled += 1
        return cancelled

    async def stream_questions(self, session_id: str) -> AsyncIterator[Mapping[str, Any]]:
        queue: asyncio.Queue[Mapping[str, Any]] = asyncio.Queue()
        async with self._lock:
            session_records = list(self._sessions.get(session_id, {}).values())
            self._session_subscribers.setdefault(session_id, set()).add(queue)
        session_records.sort(key=lambda item: item.created_at)
        for record in session_records:
            if record.status == "pending":
                await queue.put(_question_snapshot(record))
        try:
            while True:
                item = await queue.get()
                yield item
        finally:
            async with self._lock:
                self._session_subscribers.get(session_id, set()).discard(queue)

    async def _get_record(self, session_id: str, question_id: str) -> _QuestionRecord | None:
        async with self._lock:
            return self._sessions.get(session_id, {}).get(question_id)

    async def _notify(self, record: _QuestionRecord) -> None:
        async with record.lock:
            subscribers = list(record.subscribers)
            payload = _question_snapshot(record)
        for queue in subscribers:
            await queue.put(payload)
        await self._notify_session_subscribers(record.session_id, payload)

    async def _notify_session_subscribers(self, session_id: str, payload: Mapping[str, Any]) -> None:
        async with self._lock:
            subscribers = list(self._session_subscribers.get(session_id, set()))
        for queue in subscribers:
            await queue.put(payload)


def _question_snapshot(record: _QuestionRecord) -> Mapping[str, Any]:
    return {
        "sessionId": record.session_id,
        "runId": record.run_id,
        "questionId": record.question_id,
        "requestId": record.request_id,
        "claudeSessionId": record.claude_session_id,
        "title": record.title,
        "prompt": record.prompt,
        "description": record.description,
        "metadata": dict(record.metadata),
        "status": record.status,
        "answer": record.answer,
        "createdAt": record.created_at,
        "answeredAt": record.answered_at,
    }
