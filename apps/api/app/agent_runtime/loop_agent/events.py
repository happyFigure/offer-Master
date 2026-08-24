from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import Enum
from typing import Any


class LoopAgentEventType(str, Enum):
    TASK_STARTED = "task_started"
    TURN_STARTED = "turn_started"
    MODEL_DECISION = "model_decision"
    TOOL_STARTED = "tool_started"
    TOOL_FINISHED = "tool_finished"
    TURN_FINISHED = "turn_finished"
    WAITING_USER = "waiting_user"
    TASK_FINISHED = "task_finished"

    @property
    def label(self) -> str:
        return _EVENT_LABELS[self]


@dataclass(frozen=True)
class LoopAgentEvent:
    event_type: LoopAgentEventType
    session_id: str | None = None
    task_id: str | None = None
    run_id: str | None = None
    turn_index: int | None = None
    step_index: int | None = None
    capability: str | None = None
    tool_call_id: str | None = None
    status: str | None = None
    summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "event_label": self.event_type.label,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "turn_index": self.turn_index,
            "step_index": self.step_index,
            "capability": self.capability,
            "tool_call_id": self.tool_call_id,
            "status": self.status,
            "summary": self.summary,
            "metadata": _json_safe(self.metadata),
            "created_at": self.created_at.isoformat(),
        }


_EVENT_LABELS = {
    LoopAgentEventType.TASK_STARTED: "任务开始",
    LoopAgentEventType.TURN_STARTED: "一轮开始",
    LoopAgentEventType.MODEL_DECISION: "模型决定调用能力",
    LoopAgentEventType.TOOL_STARTED: "工具开始执行",
    LoopAgentEventType.TOOL_FINISHED: "工具执行结束",
    LoopAgentEventType.TURN_FINISHED: "一轮结束",
    LoopAgentEventType.WAITING_USER: "等待用户",
    LoopAgentEventType.TASK_FINISHED: "任务结束",
}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, set):
        return sorted((_json_safe(item) for item in value), key=lambda item: str(item))
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_metadata_dict"):
        return _json_safe(value.to_metadata_dict())
    return str(value)


__all__ = ["LoopAgentEvent", "LoopAgentEventType"]
