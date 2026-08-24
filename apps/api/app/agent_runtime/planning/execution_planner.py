from __future__ import annotations

import json
import re
from typing import Any

from app.agent_runtime.planning.schemas import ExecutionPlan, blocked_execution_plan, fallback_execution_plan


EXECUTION_PLANNER_SYSTEM_PROMPT = """
你是 OfferMaster 的 Execution Planner，只负责把用户请求转换成下一步可执行 action，不负责直接回答业务内容。

只能输出 JSON，不要输出 Markdown，不要解释。

输出格式：
{
  "mode": "direct_answer | simple_tool_call | bounded_react | plan_execute | blocked",
  "confidence": 0.0,
  "risk_level": "low | medium | high | critical",
  "actions": [
    {
      "type": "final_answer | call_capability | ask_user | retrieve_memory | reflect | create_subtask | handoff_to_agent",
      "capability": "可选；当 type=call_capability 时必填，必须来自 allowed_capabilities",
      "arguments": {},
      "message": "可选；只在 ask_user 或 final_answer 需要对用户说话时填写",
      "reason": "简短原因"
    }
  ],
  "max_steps": 1,
  "reason": "简短原因"
}

规则：
1. 优先选择最小够用的 action。
2. 简单查询或明确同步优先使用 simple_tool_call + call_capability。
3. 只有当 ContextPack 明确允许某个 capability 时，才能选择该 capability。
4. 当前第一版只执行 final_answer、ask_user、call_capability；create_subtask 和 handoff_to_agent 仅预留，不要主动使用。
5. 不要编造工具、不要编造数据源、不要直接声称已经执行工具。
""".strip()


class HybridExecutionPlanner:
    def __init__(self, llm_client: Any | None = None) -> None:
        self._llm_client = llm_client

    def plan(self, *, user_message: str, context_pack: dict[str, Any]) -> ExecutionPlan:
        if self._llm_client is None:
            return fallback_execution_plan("execution_planner_llm_unavailable")

        try:
            completion = self._llm_client.complete(
                messages=[
                    {"role": "system", "content": EXECUTION_PLANNER_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "user_message": user_message,
                                "context_pack": context_pack,
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                ]
            )
        except Exception:
            return fallback_execution_plan("execution_planner_llm_error")

        try:
            plan = ExecutionPlan.model_validate(json.loads(_extract_json_object(completion.content)))
        except Exception:
            return fallback_execution_plan("execution_planner_invalid_json")
        return _validate_plan_against_context_pack(plan, context_pack)


def _validate_plan_against_context_pack(plan: ExecutionPlan, context_pack: dict[str, Any]) -> ExecutionPlan:
    allowed_capabilities = {str(name) for name in context_pack.get("allowed_capabilities") or [] if str(name).strip()}
    action = plan.primary_action()
    if action is None or action.type != "call_capability":
        return plan
    if action.capability not in allowed_capabilities:
        return blocked_execution_plan(f"capability outside ContextPack: {action.capability}")
    return plan


def _extract_json_object(text: str) -> str:
    stripped = str(text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.IGNORECASE | re.DOTALL)
    if fenced is not None:
        return fenced.group(1)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Execution planner response did not contain a JSON object")
    return stripped[start : end + 1]
