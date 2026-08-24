from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.agent_runtime.reflection.evaluator import ReflectionEvaluator
from app.agent_runtime.reflection.schemas import (
    CapabilityResultEvaluationSpec,
    ReflectionDecision,
    ReflectionNextAction,
    ReflectionQuality,
)


@dataclass(frozen=True)
class CapabilityResultEvaluationRequest:
    capability: Any
    tool_input: dict[str, Any]
    result_payload: dict[str, Any]
    expected_entities: dict[str, Any] = field(default_factory=dict)
    task_goal: str = ""
    attempt_index: int = 1


class CapabilityResultEvaluator:
    """Evaluate a tool result using the capability's own acceptance standard."""

    def __init__(self, *, llm_client: Any | None = None) -> None:
        self._llm_client = llm_client

    def evaluate(self, request: CapabilityResultEvaluationRequest) -> ReflectionDecision | None:
        spec = getattr(request.capability, "result_evaluation", None)
        if spec is None:
            return None

        decision: ReflectionDecision | None = None
        rule_evaluator_id = str(getattr(spec, "rule_evaluator_id", "") or "").strip()
        if rule_evaluator_id == "campus_recruiting_web_search_rules":
            decision = ReflectionEvaluator().evaluate_web_search_result(
                tool_input=request.tool_input,
                result_payload=request.result_payload,
                expected_company_names=_expected_company_names(request.expected_entities),
            )

        if decision is not None and not _needs_model_evaluation(decision):
            return _with_capability_evaluation_metadata(decision, spec, request, decision_source="rules")

        if self._llm_client is not None:
            model_decision = self._evaluate_with_model(request, spec, rule_decision=decision)
            if model_decision is not None:
                return _with_capability_evaluation_metadata(
                    model_decision,
                    spec,
                    request,
                    decision_source="model",
                    rule_decision=decision,
                )

        if decision is not None:
            return _with_capability_evaluation_metadata(decision, spec, request, decision_source="rules")
        return None

    def _evaluate_with_model(
        self,
        request: CapabilityResultEvaluationRequest,
        spec: CapabilityResultEvaluationSpec,
        *,
        rule_decision: ReflectionDecision | None,
    ) -> ReflectionDecision | None:
        try:
            completion = self._llm_client.complete(messages=_model_evaluation_messages(request, spec, rule_decision))
        except Exception:
            return None
        data = _extract_json_object(str(getattr(completion, "content", completion) or ""))
        if data is None:
            return None
        return _model_decision_from_payload(data)


def _needs_model_evaluation(decision: ReflectionDecision) -> bool:
    return decision.quality in {ReflectionQuality.PARTIAL, ReflectionQuality.UNKNOWN}


def _model_evaluation_messages(
    request: CapabilityResultEvaluationRequest,
    spec: CapabilityResultEvaluationSpec,
    rule_decision: ReflectionDecision | None,
) -> list[dict[str, str]]:
    capability_id = str(getattr(request.capability, "capability_id", getattr(request.capability, "name", "")))
    capability_name = str(getattr(request.capability, "name", capability_id))
    capability_description = str(getattr(request.capability, "description", ""))
    rule_summary = None
    if rule_decision is not None:
        rule_summary = {
            "quality": rule_decision.quality.value,
            "next_action": rule_decision.next_action.value,
            "confidence": rule_decision.confidence,
            "reason": rule_decision.reason,
            "suggested_input_patch": dict(rule_decision.suggested_input_patch),
        }
    user_content = {
        "任务目标": request.task_goal,
        "能力编号": capability_id,
        "能力名称": capability_name,
        "能力说明": capability_description,
        "本次输入": request.tool_input,
        "本次结果": request.result_payload,
        "期望命中的实体": request.expected_entities,
        "第几次尝试": request.attempt_index,
        "规则层初判": rule_summary,
        "好结果标准": list(spec.good_result_criteria),
        "坏结果标准": list(spec.bad_result_criteria),
        "不确定结果标准": list(spec.uncertain_result_criteria),
        "如果需要重试": spec.retry_instruction,
        "参考例子": [dict(item) for item in spec.model_guidance_examples],
    }
    return [
        {
            "role": "system",
            "content": (
                "你是 OfferMaster 的工具结果验收员。请只判断这次工具结果是否完成了该能力的目标，"
                "不要补充执行新搜索。必须只返回一个 JSON 对象。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请根据下面的能力验收标准判断工具结果。\n"
                "返回 JSON 字段：quality、next_action、confidence、reason、suggested_input_patch。\n"
                "quality 只能是 good、partial、bad、unsafe、unknown。\n"
                "next_action 只能是 continue、retry、replan、ask_user、stop。\n"
                "suggested_input_patch 如果不需要修改输入就返回空对象。\n\n"
                f"{_compact_json(user_content)}"
            ),
        },
    ]


def _extract_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _model_decision_from_payload(payload: dict[str, Any]) -> ReflectionDecision | None:
    quality = _parse_quality(payload.get("quality"))
    if quality is None:
        return None
    next_action = _parse_next_action(payload.get("next_action"), quality)
    suggested_input_patch = payload.get("suggested_input_patch")
    if not isinstance(suggested_input_patch, dict):
        suggested_input_patch = {}
    reason = str(payload.get("reason") or "model evaluated tool result against capability acceptance standard")
    return ReflectionDecision(
        quality=quality,
        next_action=next_action,
        confidence=_clamp_confidence(payload.get("confidence")),
        reason=reason,
        suggested_input_patch=dict(suggested_input_patch),
        metadata={
            "model_evaluation": {
                "raw_quality": str(payload.get("quality") or ""),
                "raw_next_action": str(payload.get("next_action") or ""),
            }
        },
    )


def _parse_quality(value: Any) -> ReflectionQuality | None:
    try:
        return ReflectionQuality(str(value).strip().lower())
    except ValueError:
        return None


def _parse_next_action(value: Any, quality: ReflectionQuality) -> ReflectionNextAction:
    try:
        return ReflectionNextAction(str(value).strip().lower())
    except ValueError:
        if quality == ReflectionQuality.GOOD:
            return ReflectionNextAction.CONTINUE
        if quality == ReflectionQuality.UNSAFE:
            return ReflectionNextAction.STOP
        return ReflectionNextAction.RETRY


def _clamp_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, confidence))


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _expected_company_names(expected_entities: dict[str, Any]) -> list[str]:
    company_names = expected_entities.get("company_names") if isinstance(expected_entities, dict) else None
    if not isinstance(company_names, list):
        return []
    return [str(name).strip() for name in company_names if str(name).strip()]


def _with_capability_evaluation_metadata(
    decision: ReflectionDecision,
    spec: Any,
    request: CapabilityResultEvaluationRequest,
    *,
    decision_source: str,
    rule_decision: ReflectionDecision | None = None,
) -> ReflectionDecision:
    capability_metadata: dict[str, Any] = {
        "capability_id": str(getattr(request.capability, "capability_id", getattr(request.capability, "name", ""))),
        "executor_id": str(getattr(request.capability, "executor_id", "")),
        "evaluator_id": str(getattr(spec, "evaluator_id", "")),
        "rule_evaluator_id": str(getattr(spec, "rule_evaluator_id", "") or ""),
        "attempt_index": request.attempt_index,
        "decision_source": decision_source,
    }
    if rule_decision is not None:
        capability_metadata["rule_quality"] = rule_decision.quality.value
        capability_metadata["rule_next_action"] = rule_decision.next_action.value
        capability_metadata["rule_confidence"] = rule_decision.confidence
    metadata = {
        **dict(decision.metadata),
        "capability_result_evaluation": capability_metadata,
    }
    return ReflectionDecision(
        quality=decision.quality,
        next_action=decision.next_action,
        confidence=decision.confidence,
        reason=decision.reason,
        suggested_input_patch=dict(decision.suggested_input_patch),
        metadata=metadata,
    )


__all__ = ["CapabilityResultEvaluationRequest", "CapabilityResultEvaluator"]
