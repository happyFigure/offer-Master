from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GoalState:
    """Structured target for the current turn; it is the grader, not the executor."""

    intent: str
    target_type: str | None = None
    expected_operation: str | None = None
    target: dict[str, Any] = field(default_factory=dict)
    success_criteria: tuple[str, ...] = ()
    answer_intent: str | None = None
    answer_policy: dict[str, Any] = field(default_factory=dict)
    source: str = "runtime"

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "target_type": self.target_type,
            "expected_operation": self.expected_operation,
            "target": dict(self.target),
            "success_criteria": list(self.success_criteria),
            "answer_intent": self.answer_intent,
            "answer_policy": dict(self.answer_policy),
            "source": self.source,
        }

    @classmethod
    def from_metadata_dict(cls, payload: dict[str, Any] | None) -> "GoalState | None":
        if not isinstance(payload, dict):
            return None
        return cls(
            intent=str(payload.get("intent") or ""),
            target_type=_optional_text(payload.get("target_type")),
            expected_operation=_optional_text(payload.get("expected_operation")),
            target=dict(payload.get("target") or {}) if isinstance(payload.get("target"), dict) else {},
            success_criteria=tuple(str(item) for item in payload.get("success_criteria") or []),
            answer_intent=_optional_text(payload.get("answer_intent")),
            answer_policy=dict(payload.get("answer_policy") or {}) if isinstance(payload.get("answer_policy"), dict) else {},
            source=str(payload.get("source") or "runtime"),
        )


@dataclass(frozen=True)
class GoalValidationResult:
    """Post-tool judgment that separates tool success from user-goal success."""

    completed: bool
    advanced: bool
    recoverable: bool
    reason: str
    next_action: str
    suggested_capability: str | None = None
    suggested_operation: str | None = None
    missing_information: tuple[str, ...] = ()

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "completed": self.completed,
            "advanced": self.advanced,
            "recoverable": self.recoverable,
            "reason": self.reason,
            "next_action": self.next_action,
            "suggested_capability": self.suggested_capability,
            "suggested_operation": self.suggested_operation,
            "missing_information": list(self.missing_information),
        }


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None
