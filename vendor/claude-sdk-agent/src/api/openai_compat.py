from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator


def new_chat_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def stream_chunk(*, chat_id: str, model: str, content: str | None = None, role: str | None = None, finish_reason: str | None = None) -> bytes:
    delta: dict[str, str] = {}
    if role:
        delta["role"] = role
    if content:
        delta["content"] = content
    payload = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def protocol_shell_chunk(*, chat_id: str, model: str, tag: str, payload: dict[str, Any]) -> bytes:
    event_payload = f"[{tag}] {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n"
    return stream_chunk(chat_id=chat_id, model=model, content=event_payload)


def tool_shell_chunk(*, chat_id: str, model: str, payload: dict[str, Any]) -> bytes:
    return protocol_shell_chunk(chat_id=chat_id, model=model, tag="tool", payload=payload)


def done_chunk() -> bytes:
    return b"data: [DONE]\n\n"


def completion_payload(*, chat_id: str, model: str, content: str) -> dict[str, object]:
    return {
        "id": chat_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


async def single_text_sse(*, chat_id: str, model: str, content: str) -> AsyncIterator[bytes]:
    yield stream_chunk(chat_id=chat_id, model=model, role="assistant")
    if content:
        yield stream_chunk(chat_id=chat_id, model=model, content=content)
    yield stream_chunk(chat_id=chat_id, model=model, finish_reason="stop")
    yield done_chunk()
