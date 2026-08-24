from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from app.agent_runtime.tool_registry import OFFERIO_COMPANY_JOBS_SOURCE_NAME
from app.agent_runtime.understanding.schemas import EntityFrame, IntentFrame, fallback_intent_frame


INTENT_DETECTOR_SYSTEM_PROMPT = """你是 OfferMaster 的意图识别器。
你只能输出 JSON，不要输出 Markdown，不要回答用户问题，不要调用工具。
不要猜系统不存在的能力，不要截断中文公司名。

输出 JSON 字段：
- intent: normal_chat | memory_lookup | campus_recruiting_search | local_company_database_overview | local_job_source_overview | offerio_company_jobs_sync | application_entry_discovery | job_match_analysis | resume_tailoring | external_agent_task
- confidence: 0 到 1
- needs_external_info: boolean
- risk_level: low | medium | high | critical
- entities: object，包含 company_names/job_titles/locations/source_names/urls/job_ids/keywords/time_range
- candidate_intents: string[]
- reason: 简短中文原因
""".strip()


class DeterministicIntentMatcher:
    """High-precision command matcher, not an open-ended semantic router."""

    _OFFERIO_SYNC_RE = re.compile(
        r"(?=.*offerio)(?=.*公司聚合岗位库)(?=.*(?:更新|同步|刷新))(?=.*岗位)",
        re.IGNORECASE,
    )
    _LOCAL_COMPANY_DATABASE_OVERVIEW_RE = re.compile(
        r"(?=.*(?:数据库|本地库|企业库|公司库|库里|数据库中))(?=.*(?:企业|公司))(?=.*(?:多少|数量|总数|统计|概览|有哪些|列表|看|查))",
        re.IGNORECASE,
    )
    _LOCAL_COMPANY_LIST_RE = re.compile(
        r"(?=.*(?:企业|公司))(?=.*(?:有哪些|列表|列出|展示|多少|数量|总数|\d{1,3}\s*(?:个|家)|[零〇一二两三四五六七八九十百]{1,6}\s*(?:个|家)))",
        re.IGNORECASE,
    )
    _LOCAL_JOB_SOURCE_OVERVIEW_RE = re.compile(
        r"(?=.*(?:岗位来源|岗位信息源|信息源|来源库|岗位展览|公司展览|开放岗位来源库|开放岗位公司库|公司聚合岗位库))(?=.*(?:多少|数量|总数|统计|概览|有哪些|列表|看|查))",
        re.IGNORECASE,
    )

    def match(self, message: str) -> IntentFrame | None:
        normalized = _normalize_message(message)
        if self._LOCAL_JOB_SOURCE_OVERVIEW_RE.search(normalized):
            return IntentFrame(
                intent="local_job_source_overview",
                confidence=1.0,
                needs_external_info=False,
                risk_level="low",
                entities=EntityFrame(keywords=["岗位来源", "岗位展览"]),
                candidate_intents=["local_job_source_overview"],
                reason="matched_local_job_source_overview_question",
            )
        if self._LOCAL_COMPANY_DATABASE_OVERVIEW_RE.search(normalized) or self._LOCAL_COMPANY_LIST_RE.search(normalized):
            return IntentFrame(
                intent="local_company_database_overview",
                confidence=1.0,
                needs_external_info=False,
                risk_level="low",
                entities=EntityFrame(keywords=["企业数量", "本地数据库"]),
                candidate_intents=["local_company_database_overview"],
                reason="matched_local_company_database_overview_question",
            )
        if self._OFFERIO_SYNC_RE.search(normalized):
            return IntentFrame(
                intent="offerio_company_jobs_sync",
                confidence=1.0,
                needs_external_info=True,
                risk_level="medium",
                entities=EntityFrame(source_names=[OFFERIO_COMPANY_JOBS_SOURCE_NAME]),
                candidate_intents=["offerio_company_jobs_sync"],
                reason="matched_explicit_offerio_company_jobs_sync_command",
            )
        return None


class HybridIntentDetector:
    def __init__(self, *, llm_client: Any | None = None, matcher: DeterministicIntentMatcher | None = None) -> None:
        self._llm_client = llm_client
        self._matcher = matcher or DeterministicIntentMatcher()

    def detect(self, message: str) -> IntentFrame:
        matched = self._matcher.match(message)
        if matched is not None:
            return matched
        if self._llm_client is None:
            return fallback_intent_frame("fallback_no_intent_llm_client")

        try:
            completion = self._llm_client.complete(messages=_intent_messages(message))
            payload = _extract_json_object(completion.content)
            return IntentFrame.model_validate(payload)
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError, ValidationError):
            return fallback_intent_frame("fallback_invalid_llm_json")


def _intent_messages(message: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": INTENT_DETECTOR_SYSTEM_PROMPT},
        {"role": "user", "content": str(message).strip()},
    ]


def _normalize_message(message: str) -> str:
    return re.sub(r"\s+", " ", str(message)).strip()


def _extract_json_object(content: str) -> dict[str, Any]:
    text = str(content).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.IGNORECASE | re.DOTALL)
    if fenced is not None:
        text = fenced.group(1).strip()
    elif not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Intent detector JSON must be an object")
    return parsed
