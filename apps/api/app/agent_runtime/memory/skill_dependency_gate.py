from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agent_runtime.tool_registry import AgentToolRegistry
from app.domains.agent_memory.models import utc_now


@dataclass(frozen=True)
class SkillDependencyCheck:
    availability_state: str
    available_required_tools: tuple[str, ...]
    missing_required_tools: tuple[str, ...]
    missing_optional_tools: tuple[str, ...]

    def as_metadata(self) -> dict[str, Any]:
        return {
            "availability_state": self.availability_state,
            "tool_dependency_state": self.availability_state,
            "available_required_tools": list(self.available_required_tools),
            "missing_required_tools": list(self.missing_required_tools),
            "missing_optional_tools": list(self.missing_optional_tools),
            "tool_dependency_checked_at": utc_now().isoformat(),
        }


class SkillDependencyGate:
    """Evaluate whether a Skill's declared tools are available to the single Agent runtime."""

    def __init__(self, tool_registry: AgentToolRegistry) -> None:
        self._tool_names = frozenset(tool_registry.registered_tool_names())

    def evaluate_metadata(self, metadata: dict[str, Any] | None) -> SkillDependencyCheck:
        metadata = metadata or {}
        required_tools = _metadata_list(metadata.get("required_tools"))
        optional_tools = [
            tool
            for tool in _dedupe(
                [
                    *_metadata_list(metadata.get("allowed_tools")),
                    *_metadata_list(metadata.get("ask_tools")),
                ]
            )
            if tool not in required_tools
        ]

        available_required_tools = tuple(tool for tool in required_tools if self._has_tool(tool))
        missing_required_tools = tuple(tool for tool in required_tools if not self._has_tool(tool))
        missing_optional_tools = tuple(tool for tool in optional_tools if not self._has_tool(tool))

        if str(metadata.get("availability_state") or "") == "disabled":
            availability_state = "disabled"
        elif missing_required_tools:
            availability_state = "unavailable"
        elif not required_tools and optional_tools and missing_optional_tools:
            availability_state = "partial"
        else:
            availability_state = "available"

        return SkillDependencyCheck(
            availability_state=availability_state,
            available_required_tools=available_required_tools,
            missing_required_tools=missing_required_tools,
            missing_optional_tools=missing_optional_tools,
        )

    def decorate_metadata(self, metadata: dict[str, Any] | None) -> dict[str, Any]:
        current = dict(metadata or {})
        return {**current, **self.evaluate_metadata(current).as_metadata()}

    def _has_tool(self, tool_name: str) -> bool:
        normalized = tool_name.strip()
        if not normalized:
            return False
        return normalized in self._tool_names or f"mcp.{normalized}" in self._tool_names or (
            normalized.startswith("mcp.") and normalized.removeprefix("mcp.") in self._tool_names
        )


def _metadata_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(value).strip()] if str(value).strip() else []


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
