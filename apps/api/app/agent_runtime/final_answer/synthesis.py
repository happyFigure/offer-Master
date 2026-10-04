from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.agent_runtime.final_answer.contracts import AnswerContract, build_answer_contract
from app.agent_runtime.final_answer.observations import ToolObservation, extract_tool_observations
from app.agent_runtime.final_answer.validator import validate_final_answer
from app.agent_runtime.goal.schemas import GoalState
from app.agent_runtime.state import AgentState


GOAL_STATE_METADATA_KEY = "goal_state"
GENERIC_FINAL_ANSWER_RESPONSE_MODE = "llm_tool_observation_final_answer"
DETERMINISTIC_TOOL_SUMMARY_NAMES = frozenset(
    {
        "local.company_database_overview",
        "local.job_source_overview",
        "database.company_list",
        "database.company_profile",
        "database.company_search",
        "offerio.sync_company_jobs",
        "applications.find_apply_entry",
        "skill.filesystem",
    }
)


@dataclass(frozen=True)
class FinalAnswerSynthesisRequest:
    messages: list[dict[str, Any]]
    response_mode: str
    contract: AnswerContract
    observations: list[ToolObservation]


def build_tool_observation_final_answer_messages(state: AgentState) -> tuple[list[dict[str, Any]], str] | None:
    request = build_tool_observation_final_answer_request(state)
    if request is None:
        return None
    return request.messages, request.response_mode


def complete_tool_observation_final_answer(state: AgentState, *, llm_client: Any | None) -> tuple[str, str] | None:
    """Generate, validate, and optionally retry the generic final answer.

    Streaming callers intentionally use this non-streaming finalization path for
    tool observations. We need the full answer before showing it so the validator
    can prevent raw tool dumps or status-only messages from reaching the user.
    """

    if llm_client is None:
        return None
    request = build_tool_observation_final_answer_request(state)
    if request is None:
        return None

    try:
        completion = llm_client.complete(messages=request.messages)
    except Exception:
        return None

    validation = validate_final_answer(completion.content, contract=request.contract, observations=request.observations)
    if validation.passed:
        return completion.content, request.response_mode

    try:
        retry_completion = llm_client.complete(
            messages=build_retry_messages(
                request,
                bad_response=completion.content,
                reason=validation.reason,
                retry_instruction=validation.retry_instruction,
            )
        )
    except Exception:
        return None

    retry_validation = validate_final_answer(retry_completion.content, contract=request.contract, observations=request.observations)
    if retry_validation.passed:
        return retry_completion.content, f"{request.response_mode}_retry"
    return _tool_observation_validation_fallback(request), f"{request.response_mode}_validation_fallback"


def build_tool_observation_final_answer_request(state: AgentState) -> FinalAnswerSynthesisRequest | None:
    """Build the generic final-answer request after one or more tool results.

    my-agent keeps tool results in the loop as observations. OfferMaster still
    controls execution in runtime, so this adapter recreates the same final
    step: evidence from tools goes in, a user-facing answer comes out.
    """

    observations = extract_tool_observations(state)
    if not observations:
        return None
    if not any(observation.ok for observation in observations):
        # Failed tools should stay in the existing error/recovery lane. This
        # generic finalizer is for successful evidence that still needs to be
        # turned into the answer the user actually requested.
        return None

    goal_state = GoalState.from_metadata_dict(state.context_metadata.get(GOAL_STATE_METADATA_KEY))
    contract = build_answer_contract(user_message=state.user_message, goal_state=goal_state, context_metadata=state.context_metadata)
    if _should_keep_deterministic_tool_summary(
        goal_state=goal_state,
        contract=contract,
        requested_tool_name=state.requested_tool_name,
    ):
        return None
    if contract.allow_raw_tool_output:
        return None

    observation_payload = [observation.to_prompt_dict() for observation in observations]
    contract_payload = contract.to_metadata_dict()
    instruction = (
        "你是 OfferMaster 的主 Agent，正在做工具执行后的最终回答收口。"
        "工具结果只是 observation，不是最终回答。"
        "必须回到用户原始目标，根据 AnswerContract 和 ToolObservation 生成最终用户回答。"
        "不要把 read_file 的原始内容直接当成最终回答；不要原样展示全文。"
        "不要展示内部 Tool call、Tool result、JSON 协议、运行时字段或工具 envelope。"
        "如果 AnswerContract 要求总结、分析、提取、对比或改写，必须对 observation 加工后回答，不能只说工具成功。"
        "除非 AnswerContract 明确允许 raw output，否则不要原样贴出文件全文或工具原始输出。"
        "用中文回答，结论清晰，必要时给出要点。"
    )
    user_payload = {
        "user_message": state.user_message,
        "requested_tool_name": state.requested_tool_name,
        "AnswerContract": contract_payload,
        "ToolObservations": observation_payload,
    }
    return FinalAnswerSynthesisRequest(
        messages=[
            {"role": "system", "content": instruction, "metadata": {"source": "tool_observation_final_answer"}},
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False, indent=2),
                "metadata": {"source": "tool_observation_final_answer"},
            },
        ],
        response_mode=GENERIC_FINAL_ANSWER_RESPONSE_MODE,
        contract=contract,
        observations=observations,
    )


def _should_keep_deterministic_tool_summary(
    *,
    goal_state: GoalState | None,
    contract: AnswerContract,
    requested_tool_name: str | None = None,
) -> bool:
    if str(requested_tool_name or "").strip() in DETERMINISTIC_TOOL_SUMMARY_NAMES:
        return True
    if goal_state is None:
        return False
    if goal_state.target_type != "tool_result" or goal_state.answer_intent:
        return False
    # Existing local workflows such as company overview and OfferIO sync already
    # have deterministic business summaries. The generic LLM finalizer should
    # only take over when the user asked for extra processing like summarize,
    # analyze, extract, compare, or rewrite.
    return contract.answer_intent == "answer_from_tool_observation"


def _tool_observation_validation_fallback(request: FinalAnswerSynthesisRequest) -> str:
    tool_names = ", ".join(observation.tool_name for observation in request.observations) or "工具"
    return f"{tool_names} 已返回结果，但最终回答没有通过目标校验。为避免把工具日志误当答案返回，请重新说明需要总结、分析、提取还是展示原文。"


def build_retry_messages(request: FinalAnswerSynthesisRequest, *, bad_response: str, reason: str, retry_instruction: str | None) -> list[dict[str, Any]]:
    """Create one bounded retry prompt when the first final answer fails validation."""

    retry_payload = {
        "reason": reason,
        "bad_response": str(bad_response or ""),
        "retry_instruction": retry_instruction or "请重新生成最终回答。",
    }
    return [
        *request.messages,
        {
            "role": "user",
            "content": "上一次最终回答没有满足 AnswerContract：\n" + json.dumps(retry_payload, ensure_ascii=False, indent=2),
            "metadata": {"source": "tool_observation_final_answer_retry"},
        },
    ]
