from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from app.agent_runtime.memory.compaction import CompactionPlan, build_deterministic_summary, build_summary_prompt


REQUIRED_SUMMARY_SECTIONS = (
    "Goal",
    "Constraints & Preferences",
    "Progress",
    "Key Decisions",
    "Next Steps",
    "Critical Context",
    "Retrieval Hints",
)


@dataclass(frozen=True)
class SummaryProviderResult:
    summary_text: str
    summary_json: dict[str, Any]
    created_by: str
    metadata_json: dict[str, Any]


class SummaryProvider(Protocol):
    def summarize(self, plan: CompactionPlan) -> SummaryProviderResult:
        ...


class SummaryProviderError(RuntimeError):
    pass


class DeterministicSummaryProvider:
    name = "deterministic"
    created_by = "deterministic_compactor"

    def summarize(self, plan: CompactionPlan) -> SummaryProviderResult:
        summary_text = build_deterministic_summary(plan)
        covered_message_ids = [message.id for message in plan.messages_to_summarize]
        return SummaryProviderResult(
            summary_text=summary_text,
            summary_json={
                "Goal": "Preserve older conversation context for future Agent turns.",
                "Progress": {
                    "compacted_message_count": len(plan.messages_to_summarize),
                    "covered_message_ids": covered_message_ids,
                },
                "Key Decisions": {
                    "first_kept_message_id": plan.first_kept_message_id,
                    "previous_summary_id": None,
                },
                "Retrieval Hints": {
                    "covered_message_start_id": plan.messages_to_summarize[0].id if plan.messages_to_summarize else None,
                    "covered_message_end_id": plan.messages_to_summarize[-1].id if plan.messages_to_summarize else None,
                },
            },
            created_by=self.created_by,
            metadata_json={
                "summary_provider": self.name,
                "summary_provider_mode": "deterministic_fallback",
            },
        )


class LLMSummaryProvider:
    name = "llm"
    created_by = "llm_summary_provider"

    def __init__(self, *, llm_client: Any) -> None:
        self._llm_client = llm_client

    def summarize(self, plan: CompactionPlan) -> SummaryProviderResult:
        completion = self._llm_client.complete(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You write compact, faithful Agent memory summaries. "
                        "Return only valid JSON with the required section keys. "
                        "Do not invent facts. Keep concrete ids and next actions."
                    ),
                },
                {
                    "role": "user",
                    "content": _llm_summary_user_prompt(plan),
                },
            ],
            tools=None,
            tool_choice=None,
        )
        summary_json = _parse_llm_summary_json(str(getattr(completion, "content", "")))
        return SummaryProviderResult(
            summary_text=_render_summary_json(summary_json),
            summary_json=summary_json,
            created_by=self.created_by,
            metadata_json={
                "summary_provider": self.name,
                "summary_provider_mode": "llm_structured",
                "llm_usage": dict(getattr(completion, "usage", {}) or {}),
            },
        )


class HybridSummaryProvider:
    name = "hybrid"
    created_by = "hybrid_summary_provider"

    def __init__(self, *, primary: SummaryProvider, fallback: SummaryProvider | None = None) -> None:
        self._primary = primary
        self._fallback = fallback or DeterministicSummaryProvider()

    def summarize(self, plan: CompactionPlan) -> SummaryProviderResult:
        try:
            primary_result = self._primary.summarize(plan)
        except Exception as exc:
            fallback_result = self._fallback.summarize(plan)
            return SummaryProviderResult(
                summary_text=fallback_result.summary_text,
                summary_json=fallback_result.summary_json,
                created_by=self.created_by,
                metadata_json={
                    **fallback_result.metadata_json,
                    "summary_provider": self.name,
                    "summary_provider_mode": "fallback",
                    "primary_provider": _provider_name(self._primary),
                    "fallback_provider": _provider_name(self._fallback),
                    "fallback_reason": _short_error(exc),
                },
            )

        return SummaryProviderResult(
            summary_text=primary_result.summary_text,
            summary_json=primary_result.summary_json,
            created_by=self.created_by,
            metadata_json={
                **primary_result.metadata_json,
                "summary_provider": self.name,
                "summary_provider_mode": "primary",
                "primary_provider": _provider_name(self._primary),
            },
        )


def _llm_summary_user_prompt(plan: CompactionPlan) -> str:
    section_list = "\n".join(f'- "{section}"' for section in REQUIRED_SUMMARY_SECTIONS)
    return f"""Compress this older Agent transcript into structured memory.

Return JSON only. The top-level object must contain all required keys:
{section_list}

Each value should be a short list of concrete facts, decisions, constraints, progress, or retrieval hints.
Preserve message ids, tool names, failure reasons, business entities, and user preferences when present.

Source material:
{build_summary_prompt(plan)}
""".strip()


def _parse_llm_summary_json(content: str) -> dict[str, Any]:
    if not content.strip():
        raise SummaryProviderError("LLM summary output is empty")

    raw_json = _extract_json_object(content)
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise SummaryProviderError(f"LLM summary output is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise SummaryProviderError("LLM summary output must be a JSON object")

    missing = [section for section in REQUIRED_SUMMARY_SECTIONS if not _has_summary_value(payload.get(section))]
    if missing:
        raise SummaryProviderError(f"LLM summary output is missing required summary sections: {', '.join(missing)}")
    return payload


def _extract_json_object(content: str) -> str:
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.IGNORECASE | re.DOTALL)
    if fenced_match:
        return fenced_match.group(1).strip()

    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise SummaryProviderError("LLM summary output does not contain a JSON object")
    return content[start : end + 1].strip()


def _has_summary_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(_has_summary_value(item) for item in value)
    if isinstance(value, dict):
        return any(_has_summary_value(item) for item in value.values())
    return True


def _render_summary_json(summary_json: dict[str, Any]) -> str:
    blocks = []
    for section in REQUIRED_SUMMARY_SECTIONS:
        blocks.append(f"{section}:\n{_render_summary_value(summary_json.get(section))}")
    return "\n\n".join(blocks).strip()


def _render_summary_value(value: Any) -> str:
    if isinstance(value, list):
        rendered_items = [_render_summary_item(item) for item in value if _has_summary_value(item)]
        return "\n".join(f"- {item}" for item in rendered_items) or "- None"
    if isinstance(value, dict):
        rendered_items = [f"{key}: {_render_summary_item(item)}" for key, item in value.items() if _has_summary_value(item)]
        return "\n".join(f"- {item}" for item in rendered_items) or "- None"
    if isinstance(value, str):
        stripped = value.strip()
        return f"- {stripped}" if stripped else "- None"
    if value is None:
        return "- None"
    return f"- {value}"


def _render_summary_item(value: Any) -> str:
    if isinstance(value, dict):
        return "; ".join(f"{key}: {_render_summary_item(item)}" for key, item in value.items())
    if isinstance(value, list):
        return "; ".join(_render_summary_item(item) for item in value)
    return str(value).strip()


def _provider_name(provider: SummaryProvider) -> str:
    return str(getattr(provider, "name", provider.__class__.__name__))


def _short_error(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    return message[:500]


__all__ = [
    "DeterministicSummaryProvider",
    "HybridSummaryProvider",
    "LLMSummaryProvider",
    "REQUIRED_SUMMARY_SECTIONS",
    "SummaryProvider",
    "SummaryProviderError",
    "SummaryProviderResult",
]
