from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from app.domains.agent_memory.models import (
    AgentLearningCandidateLessonType,
    AgentLearningCandidateRiskLevel,
)
from app.domains.automation.models import ToolCallLog, ToolCallStatus
from app.domains.conversations.models import AgentMessage


@dataclass(frozen=True)
class MemoryCandidateDraft:
    memory_type: str
    scope: str
    title: str
    content: str
    importance: int
    risk_level: AgentLearningCandidateRiskLevel
    lesson_type: AgentLearningCandidateLessonType
    evidence_ids: tuple[str, ...]
    metadata: dict[str, Any]


def extract_memory_candidates(
    *,
    messages: Sequence[AgentMessage],
    tool_logs: Sequence[ToolCallLog],
) -> list[MemoryCandidateDraft]:
    drafts: list[MemoryCandidateDraft] = []
    drafts.extend(_extract_user_preferences(messages))
    drafts.extend(_extract_tool_recovery_lessons(tool_logs))
    return drafts


def _extract_user_preferences(messages: Sequence[AgentMessage]) -> list[MemoryCandidateDraft]:
    drafts: list[MemoryCandidateDraft] = []
    for message in messages:
        if _enum_value(message.role) != "user":
            continue
        content = (message.visible_content_text or message.content_text or "").strip()
        if not content:
            continue

        if _contains_application_confirmation_boundary(content):
            drafts.append(
                MemoryCandidateDraft(
                    memory_type="user_preference",
                    scope="application_submission",
                    title="投递前必须用户确认",
                    content=(
                        "用户明确要求任何岗位最终提交前必须先获得用户确认；"
                        "Agent 可以准备材料和填写草稿，但不能自动完成最终提交。"
                    ),
                    importance=95,
                    risk_level=AgentLearningCandidateRiskLevel.HIGH,
                    lesson_type=AgentLearningCandidateLessonType.USER_PREFERENCE,
                    evidence_ids=(message.id,),
                    metadata={
                        "source_kind": "explicit_user_boundary",
                        "observed_text": _truncate(_redact_text(content), 500),
                    },
                )
            )

        if _contains_data_integrity_boundary(content):
            drafts.append(
                MemoryCandidateDraft(
                    memory_type="user_preference",
                    scope="data_integrity",
                    title="不确定的字段必须留空",
                    content=(
                        "当企业性质或其他业务字段无法从可靠来源判断时，"
                        "应留空或标记为待确认，不能编造或猜测。"
                    ),
                    importance=88,
                    risk_level=AgentLearningCandidateRiskLevel.MEDIUM,
                    lesson_type=AgentLearningCandidateLessonType.USER_PREFERENCE,
                    evidence_ids=(message.id,),
                    metadata={
                        "source_kind": "explicit_user_boundary",
                        "observed_text": _truncate(_redact_text(content), 500),
                    },
                )
            )
    return drafts


def _extract_tool_recovery_lessons(tool_logs: Sequence[ToolCallLog]) -> list[MemoryCandidateDraft]:
    drafts: list[MemoryCandidateDraft] = []
    for index, failed_log in enumerate(tool_logs):
        if _enum_value(failed_log.status) != ToolCallStatus.FAILED.value:
            continue
        recovered_log = _find_later_recovery(failed_log, tool_logs[index + 1 :])
        if recovered_log is None:
            continue

        output_payload = recovered_log.output_payload or {}
        recovery_path = _string_value(output_payload.get("recovery_path"))
        if _is_timeout_only(failed_log.error) and not recovery_path:
            continue
        if not _is_learning_worthy(output_payload):
            continue

        failed_error = _truncate(_redact_text(failed_log.error or "unknown tool failure"), 240)
        recovery_path = _truncate(_redact_text(recovery_path), 600)
        drafts.append(
            MemoryCandidateDraft(
                memory_type="tool_recovery",
                scope=failed_log.tool_group or "tool_recovery",
                title=f"{failed_log.tool_name} 恢复经验",
                content=(
                    f"{failed_log.tool_name} 失败时不要重复失败路径。"
                    f"建议执行恢复路径：{recovery_path or '复用成功工具调用的参数和顺序'}。"
                    f"本次失败原因：{failed_error}。"
                ),
                importance=75,
                risk_level=_tool_recovery_risk(failed_log),
                lesson_type=AgentLearningCandidateLessonType.TOOL_RECOVERY,
                evidence_ids=(failed_log.id, recovered_log.id),
                metadata={
                    "source_kind": "recovered_tool_call",
                    "tool_name": failed_log.tool_name,
                    "tool_group": failed_log.tool_group,
                    "failed_error": failed_error,
                    "recovery_path": recovery_path or None,
                    "failed_input": _safe_json(failed_log.input_payload or {}),
                    "recovered_output_keys": sorted(str(key) for key in output_payload),
                },
            )
        )
    return drafts


def _find_later_recovery(failed_log: ToolCallLog, later_logs: Sequence[ToolCallLog]) -> ToolCallLog | None:
    for log in later_logs:
        if log.tool_name != failed_log.tool_name or log.tool_group != failed_log.tool_group:
            continue
        if _enum_value(log.status) == ToolCallStatus.SUCCEEDED.value:
            return log
    return None


def _is_learning_worthy(output_payload: dict[str, Any]) -> bool:
    if output_payload.get("verified") is True:
        return True
    try:
        if int(output_payload.get("extracted_count") or 0) > 0:
            return True
    except (TypeError, ValueError):
        pass
    return bool(_string_value(output_payload.get("recovery_path")))


def _contains_application_confirmation_boundary(content: str) -> bool:
    normalized = content.casefold()
    return (
        "投递前" in normalized
        and any(marker in normalized for marker in ("确认", "授权", "同意"))
        and any(
            marker in normalized
            for marker in ("不要自动提交", "不能自动提交", "不自动提交", "必须让我确认", "先让我确认")
        )
    )


def _contains_data_integrity_boundary(content: str) -> bool:
    normalized = content.casefold()
    uncertainty_marker = any(
        marker in normalized
        for marker in ("判断不了", "无法判断", "不知道", "不确定", "无法确认", "不清楚")
    )
    integrity_marker = any(
        marker in normalized
        for marker in ("留空", "不要编造", "不要乱填", "不编造", "不要猜")
    )
    return uncertainty_marker and integrity_marker


def _is_timeout_only(error: str | None) -> bool:
    normalized = (error or "").casefold()
    return "timeout" in normalized or "timed out" in normalized


def _tool_recovery_risk(log: ToolCallLog) -> AgentLearningCandidateRiskLevel:
    normalized = f"{log.tool_name} {log.tool_group}".casefold()
    if any(marker in normalized for marker in ("submit", "application", "投递")):
        return AgentLearningCandidateRiskLevel.HIGH
    if any(marker in normalized for marker in ("login", "auth", "content", "fetch", "article")):
        return AgentLearningCandidateRiskLevel.MEDIUM
    return AgentLearningCandidateRiskLevel.LOW


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if _is_sensitive_key(str(key)) else _safe_json(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_safe_json(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _redact_text(value: str) -> str:
    return value.replace("sk-", "[REDACTED]-").strip()


def _is_sensitive_key(key: str) -> bool:
    normalized = key.casefold().replace("-", "_")
    return normalized in {
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "password",
        "secret",
        "token",
    } or any(marker in normalized for marker in ("access_token", "refresh_token", "session_cookie"))


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _string_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _truncate(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    return value[:max_length].rstrip() + "..."
