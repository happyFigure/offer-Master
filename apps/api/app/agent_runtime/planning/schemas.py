from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.agent_runtime.understanding.schemas import RiskLevel


PlannerMode = Literal["direct_answer", "simple_tool_call", "bounded_react", "plan_execute", "blocked"]
PlannerActionType = Literal[
    "final_answer",
    "call_capability",
    "ask_user",
    "retrieve_memory",
    "reflect",
    "create_subtask",
    "handoff_to_agent",
]


class ExecutionPlannerAction(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: PlannerActionType = "final_answer"
    capability: Optional[str] = None
    arguments: Dict[str, Any] = Field(default_factory=dict)
    message: Optional[str] = None
    reason: Optional[str] = None

    @field_validator("capability", "message", "reason", mode="after")
    @classmethod
    def strip_optional_string(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        stripped = str(value).strip()
        return stripped or None

    @model_validator(mode="after")
    def validate_capability_action(self) -> "ExecutionPlannerAction":
        if self.type == "call_capability" and not self.capability:
            raise ValueError("call_capability action requires capability")
        return self

    def to_metadata_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class ExecutionPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mode: PlannerMode = "direct_answer"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_level: RiskLevel = "low"
    actions: List[ExecutionPlannerAction] = Field(default_factory=list)
    max_steps: int = Field(default=1, ge=0, le=5)
    reason: Optional[str] = None

    @field_validator("reason", mode="after")
    @classmethod
    def strip_reason(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        stripped = str(value).strip()
        return stripped or None

    def primary_action(self) -> ExecutionPlannerAction | None:
        return self.actions[0] if self.actions else None

    def to_metadata_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def fallback_execution_plan(reason: str) -> ExecutionPlan:
    return ExecutionPlan(mode="direct_answer", confidence=0.0, risk_level="low", actions=[], reason=reason)


def blocked_execution_plan(reason: str) -> ExecutionPlan:
    return ExecutionPlan(
        mode="blocked",
        confidence=1.0,
        risk_level="medium",
        actions=[ExecutionPlannerAction(type="final_answer", message=f"Planner action blocked: {reason}")],
        reason=reason,
    )
