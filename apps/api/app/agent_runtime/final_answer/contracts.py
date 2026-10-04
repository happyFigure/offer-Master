from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.agent_runtime.goal.schemas import GoalState


_SUMMARY_MARKERS = ("总结", "概括", "摘要", "归纳", "提炼", "主要内容", "讲了什么", "说了什么", "核心内容")
_EXTRACT_MARKERS = ("提取", "列出", "找出", "抽取", "有哪些", "整理出")
_ANALYZE_MARKERS = ("分析", "评价", "适合", "匹配", "问题", "建议", "优化点")
_COMPARE_MARKERS = ("对比", "比较", "差异", "区别")
_REWRITE_MARKERS = ("改写", "润色", "优化", "重写")
_SHOW_MARKERS = ("显示", "展示", "全文", "原文", "读出来", "读一下", "读取", "文件内容", "内容是什么", "告诉我这个文件的内容")
_SHOW_MARKERS_LOWER = ("cat", "show", "display")


@dataclass(frozen=True)
class AnswerContract:
    """Contract for the final user-facing answer after tool execution.

    Tool execution tells us what evidence was gathered or what side effect ran.
    This contract tells the final-answer step what the user actually wanted to
    receive, so a successful tool result cannot accidentally become the answer.
    """

    answer_intent: str
    evidence_actions: list[str] = field(default_factory=list)
    answer_policy: dict[str, Any] = field(default_factory=dict)
    completion_check: dict[str, Any] = field(default_factory=dict)
    allow_raw_tool_output: bool = False
    source: str = "answer_contract_v1"

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "answer_intent": self.answer_intent,
            "evidence_actions": list(self.evidence_actions),
            "answer_policy": dict(self.answer_policy),
            "completion_check": dict(self.completion_check),
            "allow_raw_tool_output": self.allow_raw_tool_output,
            "source": self.source,
        }

    @classmethod
    def from_metadata_dict(cls, payload: dict[str, Any] | None) -> "AnswerContract | None":
        if not isinstance(payload, dict):
            return None
        return cls(
            answer_intent=str(payload.get("answer_intent") or "answer_from_tool_observation"),
            evidence_actions=[str(item) for item in payload.get("evidence_actions") or [] if str(item or "").strip()],
            answer_policy=dict(payload.get("answer_policy") or {}) if isinstance(payload.get("answer_policy"), dict) else {},
            completion_check=dict(payload.get("completion_check") or {}) if isinstance(payload.get("completion_check"), dict) else {},
            allow_raw_tool_output=bool(payload.get("allow_raw_tool_output") or False),
            source=str(payload.get("source") or "answer_contract_v1"),
        )


def build_answer_contract(
    *,
    user_message: str,
    goal_state: GoalState | None,
    context_metadata: dict[str, Any] | None,
) -> AnswerContract:
    """Build a generic final-answer contract from goal state and context.

    This is intentionally generic: a new skill only needs to return an
    observation. The final-answer layer still knows whether the user asked to
    summarize, analyze, extract, rewrite, compare, or show raw content.
    """

    metadata = context_metadata if isinstance(context_metadata, dict) else {}
    answer_intent = _answer_intent_from_goal_or_text(user_message, goal_state)
    allow_raw_tool_output = answer_intent == "show_content"

    answer_policy = dict(goal_state.answer_policy) if goal_state is not None else {}
    answer_policy.setdefault("language", "zh-CN")
    if not allow_raw_tool_output:
        answer_policy["do_not_echo_full_content"] = True

    return AnswerContract(
        answer_intent=answer_intent,
        evidence_actions=_evidence_actions(goal_state=goal_state, context_metadata=metadata),
        answer_policy=answer_policy,
        completion_check={
            "kind": "final_answer_quality",
            "must_address_user_goal": True,
            "tool_status_is_not_enough": True,
        },
        allow_raw_tool_output=allow_raw_tool_output,
    )


def _answer_intent_from_goal_or_text(user_message: str, goal_state: GoalState | None) -> str:
    if goal_state is not None and goal_state.answer_intent:
        return goal_state.answer_intent

    text = str(user_message or "")
    lowered = text.lower()
    if any(marker in text for marker in _SUMMARY_MARKERS):
        return "summarize_document"
    if any(marker in text for marker in _EXTRACT_MARKERS):
        return "extract_document_information"
    if any(marker in text for marker in _ANALYZE_MARKERS):
        return "analyze_document"
    if any(marker in text for marker in _COMPARE_MARKERS):
        return "compare_information"
    if any(marker in text for marker in _REWRITE_MARKERS):
        return "rewrite_content"
    if any(marker in lowered for marker in _SHOW_MARKERS_LOWER) or any(marker in text for marker in _SHOW_MARKERS):
        return "show_content"
    return "answer_from_tool_observation"


def _evidence_actions(*, goal_state: GoalState | None, context_metadata: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    if goal_state is not None and goal_state.expected_operation:
        actions.append(goal_state.expected_operation)

    structured_operation = str(context_metadata.get("filesystem_operation") or "").strip()
    if structured_operation:
        actions.append(structured_operation)

    # Preserve order while removing duplicates; the first action is usually the
    # runtime's strongest signal about what evidence was gathered.
    deduped: list[str] = []
    for action in actions:
        if action not in deduped:
            deduped.append(action)
    return deduped
