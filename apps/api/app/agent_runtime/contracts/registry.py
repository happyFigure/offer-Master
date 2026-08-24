from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TypeAlias

from app.agent_runtime.contracts.base import ExecutionResultBase, TaskEnvelopeBase
from app.agent_runtime.tool_registry import APPLICATION_FIND_APPLY_ENTRY_TOOL


TaskEnvelopeClass: TypeAlias = type[TaskEnvelopeBase]
ExecutionResultClass: TypeAlias = type[ExecutionResultBase]


@dataclass(frozen=True)
class TaskContractEntry:
    capability: str
    task_type: str
    envelope_cls: TaskEnvelopeClass
    result_cls: ExecutionResultClass
    default_executor: str


class TaskEnvelopeRegistry:
    def __init__(self) -> None:
        self._by_capability: dict[str, TaskContractEntry] = {}
        self._by_task_type: dict[str, TaskContractEntry] = {}

    def register(
        self,
        *,
        capability: str,
        task_type: str,
        envelope_cls: TaskEnvelopeClass,
        result_cls: ExecutionResultClass,
        default_executor: str,
    ) -> TaskContractEntry:
        if capability in self._by_capability:
            raise ValueError(f"Task contract capability already registered: {capability}")
        if task_type in self._by_task_type:
            raise ValueError(f"Task contract task_type already registered: {task_type}")
        entry = TaskContractEntry(
            capability=capability,
            task_type=task_type,
            envelope_cls=envelope_cls,
            result_cls=result_cls,
            default_executor=default_executor,
        )
        self._by_capability[capability] = entry
        self._by_task_type[task_type] = entry
        return entry

    def get_by_capability(self, capability: str) -> TaskContractEntry:
        try:
            return self._by_capability[capability]
        except KeyError as exc:
            raise KeyError(f"No task contract registered for capability: {capability}") from exc

    def get_by_task_type(self, task_type: str) -> TaskContractEntry:
        try:
            return self._by_task_type[task_type]
        except KeyError as exc:
            raise KeyError(f"No task contract registered for task_type: {task_type}") from exc

    def list_entries(self) -> list[TaskContractEntry]:
        return sorted(self._by_capability.values(), key=lambda entry: entry.capability)


@lru_cache(maxsize=1)
def default_task_envelope_registry() -> TaskEnvelopeRegistry:
    from app.agent_runtime.contracts.tasks.browser_application import BrowserExecutionResult, BrowserTaskEnvelope, BrowserTaskType

    registry = TaskEnvelopeRegistry()
    registry.register(
        capability=APPLICATION_FIND_APPLY_ENTRY_TOOL,
        task_type=BrowserTaskType.PREPARE_APPLICATION.value,
        envelope_cls=BrowserTaskEnvelope,
        result_cls=BrowserExecutionResult,
        default_executor="codex_or_multica",
    )
    return registry
