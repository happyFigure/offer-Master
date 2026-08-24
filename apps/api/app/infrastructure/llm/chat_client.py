from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.infrastructure.llm.client import LLMRuntimeConfig, build_llm_runtime_config


@dataclass(frozen=True)
class LLMToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMChatCompletion:
    content: str
    tool_calls: list[LLMToolCall] = field(default_factory=list)
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

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMChatCompletion:
        payload = {
            "model": self._config.model,
            "temperature": 0.2,
            "messages": _normalize_messages(messages),
        }
        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
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
    tool_calls = _parse_tool_calls(message.get("tool_calls"))
    if not isinstance(content, str):
        content = ""
    if not content.strip() and not tool_calls:
        raise RuntimeError("LLM response did not contain assistant text content or tool calls")
    usage = response_payload.get("usage")
    return LLMChatCompletion(
        content=content,
        tool_calls=tool_calls,
        usage=usage if isinstance(usage, dict) else {},
        raw_response=response_payload,
    )


def _parse_tool_calls(raw_tool_calls: Any) -> list[LLMToolCall]:
    if not isinstance(raw_tool_calls, list):
        return []
    parsed: list[LLMToolCall] = []
    for index, raw_call in enumerate(raw_tool_calls):
        if not isinstance(raw_call, dict):
            continue
        function = raw_call.get("function") if isinstance(raw_call.get("function"), dict) else {}
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        raw_arguments = function.get("arguments")
        arguments: dict[str, Any]
        if isinstance(raw_arguments, dict):
            arguments = raw_arguments
        elif isinstance(raw_arguments, str) and raw_arguments.strip():
            loaded = json.loads(raw_arguments)
            arguments = loaded if isinstance(loaded, dict) else {}
        else:
            arguments = {}
        parsed.append(
            LLMToolCall(
                id=str(raw_call.get("id") or f"tool-call-{index}"),
                name=name,
                arguments=arguments,
            )
        )
    return parsed


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
