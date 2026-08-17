from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentToolPermissionDecision(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass(frozen=True)
class AgentToolPermissionResult:
    decision: AgentToolPermissionDecision
    reason: str
    skill_id: str | None = None
    skill_ids: tuple[str, ...] = field(default_factory=tuple)
    allowed_tools: tuple[str, ...] = field(default_factory=tuple)
    ask_tools: tuple[str, ...] = field(default_factory=tuple)
    disallowed_tools: tuple[str, ...] = field(default_factory=tuple)

    @property
    def error_details(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "skill_ids": list(self.skill_ids),
            "permission_decision": self.decision.value,
            "allowed_tools": list(self.allowed_tools),
            "ask_tools": list(self.ask_tools),
            "disallowed_tools": list(self.disallowed_tools),
        }


@dataclass(frozen=True)
class AgentToolPermissionPolicy:
    skill_id: str | None = None
    skill_ids: tuple[str, ...] = field(default_factory=tuple)
    allowed_tools: tuple[str, ...] = field(default_factory=tuple)
    ask_tools: tuple[str, ...] = field(default_factory=tuple)
    disallowed_tools: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_skill_metadata(cls, skill_id: str | None, metadata: dict[str, Any] | None) -> AgentToolPermissionPolicy:
        metadata = metadata or {}
        return cls(
            skill_id=skill_id,
            skill_ids=tuple([skill_id] if skill_id else []),
            allowed_tools=tuple(_metadata_list(metadata.get("allowed_tools"))),
            ask_tools=tuple(_metadata_list(metadata.get("ask_tools"))),
            disallowed_tools=tuple(_metadata_list(metadata.get("disallowed_tools"))),
        )

    @classmethod
    def from_loaded_skill_metadata(
        cls,
        loaded_skills: list[tuple[str, dict[str, Any] | None]],
    ) -> AgentToolPermissionPolicy:
        skill_ids: list[str] = []
        allowed_tools: list[str] = []
        ask_tools: list[str] = []
        disallowed_tools: list[str] = []
        for skill_id, metadata in loaded_skills:
            if skill_id:
                skill_ids.append(skill_id)
            metadata = metadata or {}
            allowed_tools.extend(_metadata_list(metadata.get("allowed_tools")))
            ask_tools.extend(_metadata_list(metadata.get("ask_tools")))
            disallowed_tools.extend(_metadata_list(metadata.get("disallowed_tools")))
        return cls(
            skill_id=skill_ids[0] if len(skill_ids) == 1 else None,
            skill_ids=tuple(_dedupe(skill_ids)),
            allowed_tools=tuple(_dedupe(allowed_tools)),
            ask_tools=tuple(_dedupe(ask_tools)),
            disallowed_tools=tuple(_dedupe(disallowed_tools)),
        )

    @classmethod
    def from_metadata_snapshot(cls, metadata: dict[str, Any] | None) -> AgentToolPermissionPolicy:
        metadata = metadata or {}
        skill_ids = _metadata_list(metadata.get("skill_ids"))
        skill_id = str(metadata.get("skill_id") or "").strip() or (skill_ids[0] if len(skill_ids) == 1 else None)
        return cls(
            skill_id=skill_id,
            skill_ids=tuple(skill_ids),
            allowed_tools=tuple(_metadata_list(metadata.get("allowed_tools"))),
            ask_tools=tuple(_metadata_list(metadata.get("ask_tools"))),
            disallowed_tools=tuple(_metadata_list(metadata.get("disallowed_tools"))),
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "policy_source": "loaded_skill_snapshot",
            "skill_id": self.skill_id,
            "skill_ids": list(self.skill_ids),
            "allowed_tools": list(self.allowed_tools),
            "ask_tools": list(self.ask_tools),
            "disallowed_tools": list(self.disallowed_tools),
        }

    def decide(self, tool_name: str, *, user_confirmed: bool = False) -> AgentToolPermissionResult:
        normalized_tool_name = tool_name.strip()
        if _contains_tool(self.disallowed_tools, normalized_tool_name):
            return self._result(
                AgentToolPermissionDecision.DENY,
                f"Tool {normalized_tool_name} is denied by the active Skill disallowed_tools.",
            )

        if _contains_tool(self.ask_tools, normalized_tool_name):
            if user_confirmed:
                return self._result(
                    AgentToolPermissionDecision.ALLOW,
                    f"Tool {normalized_tool_name} was confirmed by the user for the active Skill ask_tools.",
                )
            return self._result(
                AgentToolPermissionDecision.ASK,
                f"Tool {normalized_tool_name} requires user confirmation by the active Skill ask_tools.",
            )

        if self.allowed_tools and not _contains_tool(self.allowed_tools, normalized_tool_name):
            if user_confirmed:
                return self._result(
                    AgentToolPermissionDecision.ALLOW,
                    f"Tool {normalized_tool_name} was confirmed by the user although it is not declared in the active Skill allowed_tools.",
                )
            return self._result(
                AgentToolPermissionDecision.ASK,
                f"Tool {normalized_tool_name} is not declared in the active Skill allowed_tools.",
            )

        return self._result(AgentToolPermissionDecision.ALLOW, f"Tool {normalized_tool_name} is allowed by Skill policy.")

    def _result(self, decision: AgentToolPermissionDecision, reason: str) -> AgentToolPermissionResult:
        return AgentToolPermissionResult(
            decision=decision,
            reason=reason,
            skill_id=self.skill_id,
            skill_ids=self.skill_ids,
            allowed_tools=self.allowed_tools,
            ask_tools=self.ask_tools,
            disallowed_tools=self.disallowed_tools,
        )


def _metadata_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return _dedupe(str(item).strip() for item in value if str(item).strip())
    return _dedupe(item.strip() for item in str(value).split(",") if item.strip())


def _dedupe(items: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _contains_tool(tool_names: tuple[str, ...], tool_name: str) -> bool:
    return tool_name in tool_names
