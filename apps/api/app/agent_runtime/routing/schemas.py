from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RouteDecision:
    route: str
    capability: str | None = None
    executor_type: str = "chat"
    executor_name: str | None = None
    confidence: float = 1.0
    reason: str = ""
    allowed_capabilities: list[str] = field(default_factory=list)
    blocked_capabilities: list[str] = field(default_factory=list)
    requires_confirmation: bool = False
    max_steps: int = 1
    tool_input: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "capability": self.capability,
            "executor_type": self.executor_type,
            "executor_name": self.executor_name,
            "confidence": self.confidence,
            "reason": self.reason,
            "allowed_capabilities": list(self.allowed_capabilities),
            "blocked_capabilities": list(self.blocked_capabilities),
            "requires_confirmation": self.requires_confirmation,
            "max_steps": self.max_steps,
            "tool_input": dict(self.tool_input),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ResultEnvelope:
    status: str
    capability: str
    executor: str
    summary: str = ""
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    requires_user_action: bool = False
    risk_level: str = "low"
    raw_result: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    retryable: bool | None = None
    next_action: str | None = None
    business_refs: list[dict[str, Any]] = field(default_factory=list)
    source_type: str | None = None
    tool_call_log_id: str | None = None
    step_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "capability": self.capability,
            "executor": self.executor,
            "summary": self.summary,
            "artifacts": list(self.artifacts),
            "observations": list(self.observations),
            "requires_user_action": self.requires_user_action,
            "risk_level": self.risk_level,
            "raw_result": dict(self.raw_result),
            "error_code": self.error_code,
            "retryable": self.retryable,
            "next_action": self.next_action,
            "business_refs": [dict(item) for item in self.business_refs],
            "source_type": self.source_type,
            "tool_call_log_id": self.tool_call_log_id,
            "step_id": self.step_id,
        }
