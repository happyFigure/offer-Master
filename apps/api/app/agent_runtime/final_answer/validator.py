from __future__ import annotations

from dataclasses import dataclass

from app.agent_runtime.final_answer.contracts import AnswerContract
from app.agent_runtime.final_answer.observations import ToolObservation


_PROCESSING_INTENTS = {
    "summarize_document",
    "extract_document_information",
    "analyze_document",
    "compare_information",
    "rewrite_content",
    "answer_from_tool_observation",
}
_STATUS_ONLY_MARKERS = (
    "已读取",
    "读取完成",
    "工具执行成功",
    "filesystem Skill 已完成",
    "Skill 已完成",
    "已完成处理",
)
_RAW_PROTOCOL_MARKERS = (
    "Tool call:",
    "Tool result:",
    "```json",
    "```text",
    "\\documentclass",
    "\\section{",
)


@dataclass(frozen=True)
class FinalAnswerValidation:
    passed: bool
    reason: str
    retry_instruction: str | None = None


def validate_final_answer(
    final_response: str,
    *,
    contract: AnswerContract,
    observations: list[ToolObservation],
) -> FinalAnswerValidation:
    """Check whether a synthesized answer satisfies the user's final goal.

    The validator stays intentionally lightweight and deterministic. It does
    not try to judge answer quality deeply; it catches the two regressions that
    break the agent loop most often: status-only answers and raw tool dumps.
    """

    response = str(final_response or "").strip()
    if not response:
        return FinalAnswerValidation(False, "empty_answer", "请根据工具 observation 生成面向用户的最终回答，不要返回空内容。")

    if not contract.allow_raw_tool_output and any(marker in response for marker in _RAW_PROTOCOL_MARKERS):
        return FinalAnswerValidation(
            False,
            "raw_tool_output",
            "上一次回答包含原始工具协议、代码块或文件全文。请只给用户需要的总结、分析或结论。",
        )

    if contract.answer_intent in _PROCESSING_INTENTS and _looks_status_only(response):
        return FinalAnswerValidation(
            False,
            "tool_status_only",
            "上一次回答只说明工具执行状态，没有完成用户目标。请基于 observation 正面回答用户问题。",
        )

    if observations and contract.answer_intent in _PROCESSING_INTENTS and _echoes_large_observation(response, observations):
        return FinalAnswerValidation(
            False,
            "echoed_observation",
            "上一次回答过度复述工具 observation。请归纳加工后回答，不要原样粘贴证据。",
        )

    return FinalAnswerValidation(True, "passed")


def _looks_status_only(response: str) -> bool:
    if any(marker in response for marker in _STATUS_ONLY_MARKERS) and len(response) <= 80:
        return True
    return response.strip(" 。.！!") in _STATUS_ONLY_MARKERS


def _echoes_large_observation(response: str, observations: list[ToolObservation]) -> bool:
    if len(response) < 600:
        return False
    for observation in observations:
        evidence = observation.evidence.strip()
        if len(evidence) >= 600 and evidence[:500] in response:
            return True
    return False
