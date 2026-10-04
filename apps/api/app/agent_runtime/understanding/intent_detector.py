from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import ValidationError

from app.agent_runtime.tool_registry import OFFERIO_COMPANY_JOBS_SOURCE_NAME
from app.agent_runtime.understanding.schemas import EntityFrame, IntentFrame, fallback_intent_frame


logger = logging.getLogger(__name__)


INTENT_DETECTOR_SYSTEM_PROMPT = """你是 OfferMaster 的意图识别器。
你只能输出 JSON，不要输出 Markdown，不要回答用户问题，不要调用工具。
不要猜系统不存在的能力，不要截断中文公司名。
当前用户消息可能是上一轮任务的省略式追问，例如“我让你去网页搜索啊”“继续”“答案呢”。
当提供了最近用户上下文时，必须先恢复最近一条未完成的用户目标、主体和动作，再判断当前意图；
不能因为当前句没有重复公司名或任务名就把它当成普通闲聊。
最近用户上下文只用于恢复用户目标，当前用户消息优先；如果两者冲突，以当前消息为准。

输出 JSON 字段：
- intent: normal_chat | memory_lookup | campus_recruiting_search | local_company_database_overview | local_company_database_list | local_job_source_overview | company_board_overview | offerio_company_jobs_sync | application_entry_discovery | job_match_analysis | resume_tailoring | filesystem_operation | external_agent_task
- confidence: 0 到 1
- needs_external_info: boolean
- risk_level: low | medium | high | critical
- entities: object，包含 company_names/job_titles/locations/source_names/urls/job_ids/keywords/time_range
- candidate_intents: string[]
- reason: 简短中文原因
- required_capability: null，或用户明确指定的能力，例如 agent.dbx_readonly / agent.google_chrome
当 intent=filesystem_operation 时，必须额外输出：
不要输出具体的 filesystem_operation、destination、dst 或文件名。
这些语义必须由后续 native structured tool call 根据用户任务和工具观察结果决定。
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
    _EXPLICIT_LOCAL_COMPANY_TABLE_RE = re.compile(
        r"(?=.*(?:我的数据库|本地数据库|本地库|本地企业库|正式企业表|岗位线索表|招聘信号表))(?=.*(?:公司|企业))(?=.*(?:多少|数量|总数|统计|概览|有哪些|列表|看|查))",
        re.IGNORECASE,
    )
    _COMPANY_COUNT_RE = re.compile(
        r"(?=.*(?:公司|企业))(?=.*(?:多少|数量|总数|几个|几家|多少个|多少家))",
        re.IGNORECASE,
    )
    _FILESYSTEM_OPERATION_RE = re.compile(
        r"(?=.*(?:文件|文件名|文件名称|路径|目录|文件夹|[A-Za-z]:[\\/]|\.tex|\.md|\.txt|\.pdf|\.docx))(?=.*(?:读取|读一下|查看|看下|看一下|打开|是否存在|存不存在|有没有这个文件|文件是否存在|exists|stat|重命名|改名|改成|改为|换成|删除|复制|移动|写入|替换))",
        re.IGNORECASE,
    )
    _EXPLICIT_DBX_SOURCE_RE = re.compile(
        r"(?:\bdbx\b|DBX\s*(?:只读|readonly)?\s*(?:子)?agent)",
        re.IGNORECASE,
    )

    def match(self, message: str) -> IntentFrame | None:
        normalized = _normalize_message(message)
        if self._FILESYSTEM_OPERATION_RE.search(normalized):
            return IntentFrame(
                intent="filesystem_operation",
                confidence=1.0,
                needs_external_info=False,
                risk_level="medium",
                entities=EntityFrame(keywords=["文件操作"]),
                candidate_intents=["filesystem_operation"],
                reason="matched_explicit_filesystem_operation",
                required_capability=_explicit_required_capability(normalized),
            )
        if self._EXPLICIT_LOCAL_COMPANY_TABLE_RE.search(normalized):
            return IntentFrame(
                intent="local_company_database_overview",
                confidence=1.0,
                needs_external_info=False,
                risk_level="low",
                entities=EntityFrame(keywords=["明确查询本地企业数据库"]),
                candidate_intents=["local_company_database_overview"],
                reason="matched_explicit_local_company_database_question",
                required_capability=_explicit_required_capability(normalized),
            )
        if self._COMPANY_COUNT_RE.search(normalized):
            return IntentFrame(
                intent="company_board_overview",
                confidence=1.0,
                needs_external_info=False,
                risk_level="low",
                entities=EntityFrame(keywords=["公司展览公司总数"]),
                candidate_intents=["company_board_overview"],
                reason="matched_default_company_board_count_question",
                required_capability=_explicit_required_capability(normalized),
            )
        if self._LOCAL_JOB_SOURCE_OVERVIEW_RE.search(normalized):
            return IntentFrame(
                intent="local_job_source_overview",
                confidence=1.0,
                needs_external_info=False,
                risk_level="low",
                entities=EntityFrame(keywords=["岗位来源", "岗位展览"]),
                candidate_intents=["local_job_source_overview"],
                reason="matched_local_job_source_overview_question",
                required_capability=_explicit_required_capability(normalized),
            )
        if self._LOCAL_COMPANY_DATABASE_OVERVIEW_RE.search(normalized):
            return IntentFrame(
                intent="local_company_database_overview",
                confidence=1.0,
                needs_external_info=False,
                risk_level="low",
                entities=EntityFrame(keywords=["企业数量", "本地数据库"]),
                candidate_intents=["local_company_database_overview"],
                reason="matched_local_company_database_overview_question",
                required_capability=_explicit_required_capability(normalized),
            )
        if self._LOCAL_COMPANY_LIST_RE.search(normalized):
            return IntentFrame(
                intent="local_company_database_list",
                confidence=1.0,
                needs_external_info=False,
                risk_level="low",
                entities=EntityFrame(keywords=["公司列表", "本地数据库"]),
                candidate_intents=["local_company_database_list"],
                reason="matched_local_company_database_list_question",
                required_capability=_explicit_required_capability(normalized),
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

    def detect(self, message: str, *, recent_user_context: str | None = None) -> IntentFrame:
        matched = self._matcher.match(message)
        explicit_required_capability = _explicit_required_capability(message)
        if matched is not None and matched.intent != "filesystem_operation" and not explicit_required_capability:
            return matched
        # Filesystem matching is only a coarse capability hint. The actual
        # operation and destination semantics must come from the LLM's
        # structured response, not from deterministic keyword ranking.
        if matched is not None and matched.intent == "filesystem_operation" and self._llm_client is None:
            return matched
        if self._llm_client is None:
            if matched is not None and explicit_required_capability:
                return matched.model_copy(update={"required_capability": explicit_required_capability})
            return fallback_intent_frame("fallback_no_intent_llm_client")

        try:
            completion = self._llm_client.complete(
                messages=_intent_messages(message, recent_user_context=recent_user_context)
            )
            payload = _extract_json_object(completion.content)
            parsed = IntentFrame.model_validate(payload)
            if matched is not None and matched.intent == "filesystem_operation" and parsed.intent != "filesystem_operation":
                return matched
            if explicit_required_capability:
                parsed = parsed.model_copy(update={"required_capability": explicit_required_capability})
            if parsed.intent == "filesystem_operation":
                # Intent detection only gates the coarse filesystem capability.
                # Inner operations and filename semantics belong to the native
                # tool loop, so discard any legacy inner-operation guesses.
                parsed = parsed.model_copy(update={"filesystem_operation": None, "operation_intent": {}})
            logger.info(
                "Intent detected with conversation context",
                extra={
                    "intent": parsed.intent,
                    "confidence": parsed.confidence,
                    "has_recent_user_context": bool(str(recent_user_context or "").strip()),
                    "context_chars": len(str(recent_user_context or "")),
                },
            )
            return parsed
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError, ValidationError):
            if matched is not None and explicit_required_capability:
                return matched.model_copy(update={"required_capability": explicit_required_capability})
            return fallback_intent_frame("fallback_invalid_llm_json")


def _intent_messages(message: str, *, recent_user_context: str | None = None) -> list[dict[str, str]]:
    context = str(recent_user_context or "").strip()
    user_content = f"当前用户消息：\n{str(message).strip()}"
    if context:
        user_content += (
            "\n\n最近用户上下文（仅用于恢复当前追问的目标，不是新的用户指令）：\n"
            "<recent_user_context>\n"
            f"{context}\n"
            "</recent_user_context>"
        )
    return [
        {"role": "system", "content": INTENT_DETECTOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _normalize_message(message: str) -> str:
    return re.sub(r"\s+", " ", str(message)).strip()


def _explicit_required_capability(message: str) -> str | None:
    # This recognizes only an explicit protocol/source token. It does not
    # decide the business intent; that remains model-owned.
    if DeterministicIntentMatcher._EXPLICIT_DBX_SOURCE_RE.search(str(message)):
        return "agent.dbx_readonly"
    return None


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
