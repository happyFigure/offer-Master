from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.infrastructure.llm.client import LLMRuntimeConfig, build_llm_runtime_config


@dataclass(frozen=True)
class LLMChatCompletion:
    content: str
    usage: dict[str, Any] = field(default_factory=dict)
    raw_response: dict[str, Any] = field(default_factory=dict)


class LLMChatClient:
    def __init__(
        self,
        config: LLMRuntimeConfig | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._config = config or build_llm_runtime_config()
        self._client = client

    def complete(self, *, messages: list[dict[str, Any]]) -> LLMChatCompletion:
        payload = {
            "model": self._config.model,
            "temperature": 0.2,
            "messages": _normalize_messages(messages),
        }
        response_payload = self._post_chat_completion(payload)
        return _parse_chat_completion(response_payload)

    def stream_complete(self, *, messages: list[dict[str, Any]]) -> Iterator[str]:
        payload = {
            "model": self._config.model,
            "temperature": 0.2,
            "stream": True,
            "messages": _normalize_messages(messages),
        }
        last_error: Exception | None = None

        for _ in range(self._config.max_retries + 1):
            try:
                yield from self._stream_chat_completion(payload)
                return
            except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
                last_error = exc

        raise RuntimeError("LLM streaming chat completion request failed") from last_error

    def _post_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        endpoint = f"{self._config.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self._config.api_key}"}
        last_error: Exception | None = None

        for _ in range(self._config.max_retries + 1):
            try:
                if self._client is not None:
                    response = self._client.post(endpoint, json=payload, headers=headers)
                else:
                    with httpx.Client(timeout=self._config.timeout_seconds) as client:
                        response = client.post(endpoint, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
                last_error = exc

        raise RuntimeError("LLM chat completion request failed") from last_error

    def _stream_chat_completion(self, payload: dict[str, Any]) -> Iterator[str]:
        endpoint = f"{self._config.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self._config.api_key}"}

        if self._client is not None:
            with self._client.stream("POST", endpoint, json=payload, headers=headers) as response:
                response.raise_for_status()
                yield from _parse_chat_completion_stream(response.iter_lines())
            return

        with httpx.Client(timeout=self._config.timeout_seconds) as client:
            with client.stream("POST", endpoint, json=payload, headers=headers) as response:
                response.raise_for_status()
                yield from _parse_chat_completion_stream(response.iter_lines())


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized = []
    for message in messages:
        role = str(message.get("role") or "user")
        if role not in {"system", "user", "assistant", "tool"}:
            role = "assistant"
        content = message.get("content")
        if content is None:
            continue
        normalized.append({"role": role, "content": str(content)})
    if not normalized:
        raise ValueError("LLM chat completion requires at least one message")
    return normalized


def _parse_chat_completion(response_payload: dict[str, Any]) -> LLMChatCompletion:
    choices = response_payload.get("choices")
    if not choices:
        raise RuntimeError("LLM response did not contain choices")
    message = choices[0].get("message", {})
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("LLM response did not contain assistant text content")
    usage = response_payload.get("usage")
    return LLMChatCompletion(
        content=content,
        usage=usage if isinstance(usage, dict) else {},
        raw_response=response_payload,
    )


def _parse_chat_completion_stream(lines: Iterator[str | bytes]) -> Iterator[str]:
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        stripped = line.strip()
        if not stripped or stripped.startswith(":"):
            continue
        if not stripped.startswith("data:"):
            continue

        data = stripped.removeprefix("data:").strip()
        if data == "[DONE]":
            break

        payload = json.loads(data)
        choices = payload.get("choices")
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str) and content:
            yield content
