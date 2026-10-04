from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from app.agent_runtime.loop_agent.schemas import LoopAgentAction, LoopAgentDecision


@dataclass(frozen=True)
class ToolLoopIntervention:
    detector: str
    action_key: str
    count: int
    reason: str

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "detector": self.detector,
            "action_key": self.action_key,
            "count": self.count,
            "reason": self.reason,
        }


class DeadLoopDetector:
    """Detects repeated tool calls before the loop spends another execution step."""

    def __init__(self, *, max_repeated_calls: int = 2) -> None:
        self.max_repeated_calls = max(1, int(max_repeated_calls))

    def action_key(self, decision: LoopAgentDecision) -> str:
        # The key intentionally ignores explanation text. A changed reason with
        # the same capability and same input is still the same tool action.
        return json.dumps(
            {
                "action": decision.action.value,
                "capability": decision.capability,
                "tool_input": decision.tool_input,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )

    def intervention_for(
        self,
        decision: LoopAgentDecision,
        previous_action_keys: list[str],
    ) -> ToolLoopIntervention | None:
        if decision.action != LoopAgentAction.CALL_TOOL:
            return None
        action_key = self.action_key(decision)
        previous_count = previous_action_keys.count(action_key)
        if previous_count < self.max_repeated_calls:
            return None
        return ToolLoopIntervention(
            detector="dead_loop_detector",
            action_key=action_key,
            count=previous_count + 1,
            reason="检测到重复工具调用：同一个能力和同一组参数已经连续/反复执行，当前轮次需要重规划或向用户确认。",
        )


__all__ = ["DeadLoopDetector", "ToolLoopIntervention"]
