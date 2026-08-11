from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import httpx
from pydantic import ValidationError

from app.domains.jobs.providers.social_lead import ExtractedJobLead
from app.infrastructure.llm.client import LLMRuntimeConfig, build_llm_runtime_config


class LLMJobLeadExtractor:
    def __init__(
        self,
        config: LLMRuntimeConfig | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._config = config or build_llm_runtime_config()
        self._client = client

    def extract(
        self,
        raw_content: str,
        source_context: Mapping[str, Any],
    ) -> list[ExtractedJobLead]:
        payload = self._build_payload(raw_content, source_context)
        response_payload = self._post_chat_completion(payload)
        return self._parse_response(response_payload)

    def _build_payload(
        self,
        raw_content: str,
        source_context: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "model": self._config.model,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You extract China campus recruiting job leads for a CS student. "
                        "Return only JSON with an items array. Each item may include "
                        "company_name, title, city, job_direction, graduation_year, "
                        "source_url, apply_url, job_type, salary_text, jd_text, skills, "
                        "deadline as YYYY-MM-DD, confidence_score from 0 to 100. "
                        "Prefer Java/backend/Agent/LLM/RAG/AI engineering roles. "
                        "Do not invent URLs or deadlines when absent."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "source_context": dict(source_context),
                            "raw_content": raw_content,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        }

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

        raise RuntimeError("LLM job lead extraction request failed") from last_error

    def _parse_response(self, response_payload: Mapping[str, Any]) -> list[ExtractedJobLead]:
        choices = response_payload.get("choices")
        if not choices:
            raise RuntimeError("LLM response did not contain choices")
        message = choices[0].get("message", {})
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("LLM response did not contain text content")

        data = _loads_json_content(content)
        items = data if isinstance(data, list) else data.get("items", [])
        if not isinstance(items, list):
            raise RuntimeError("LLM response items must be a list")

        leads: list[ExtractedJobLead] = []
        for item in items:
            try:
                leads.append(ExtractedJobLead.model_validate(item))
            except ValidationError:
                continue
        return leads


def _loads_json_content(content: str) -> Any:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return json.loads(stripped)
