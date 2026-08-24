from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agent_runtime.agent_as_tool import (
    DEFAULT_SUPPORTED_INTENTS_BY_CAPABILITY,
    AgentCapabilityDefinition,
    AgentCapabilityRegistry,
    create_default_agent_capability_registry,
)
from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry


_ALLOWED_INTENTS_BY_TOOL: dict[str, tuple[str, ...]] = dict(DEFAULT_SUPPORTED_INTENTS_BY_CAPABILITY)


@dataclass(frozen=True)
class CapabilityMetadata:
    name: str
    description: str
    risk_level: str
    input_summary: list[str]
    allowed_intents: tuple[str, ...]
    requires_confirmation: bool = False

    @classmethod
    def from_tool_definition(cls, definition: AgentToolDefinition) -> CapabilityMetadata:
        properties = definition.input_schema.get("properties") if isinstance(definition.input_schema, dict) else None
        return cls(
            name=definition.name,
            description=definition.description,
            risk_level=definition.risk_level.value,
            input_summary=sorted(str(key) for key in properties.keys()) if isinstance(properties, dict) else [],
            allowed_intents=_ALLOWED_INTENTS_BY_TOOL.get(definition.name, ()),
            requires_confirmation=definition.requires_confirmation,
        )

    @classmethod
    def from_agent_capability(cls, definition: AgentCapabilityDefinition) -> CapabilityMetadata:
        properties = definition.input_schema.get("properties") if isinstance(definition.input_schema, dict) else None
        return cls(
            name=definition.capability_id,
            description=definition.description,
            risk_level=definition.risk_level,
            input_summary=sorted(str(key) for key in properties.keys()) if isinstance(properties, dict) else [],
            allowed_intents=definition.supported_intents,
            requires_confirmation=definition.requires_confirmation,
        )

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "risk_level": self.risk_level,
            "input_summary": list(self.input_summary),
            "allowed_intents": list(self.allowed_intents),
            "requires_confirmation": self.requires_confirmation,
        }


class CapabilityCatalog:
    def __init__(self, capabilities: list[CapabilityMetadata]) -> None:
        self._capabilities = sorted(capabilities, key=lambda item: item.name)

    @classmethod
    def from_registry(cls, registry: AgentToolRegistry) -> CapabilityCatalog:
        agent_registry = create_default_agent_capability_registry(tool_registry=registry)
        return cls.from_agent_registry(agent_registry)

    @classmethod
    def from_agent_registry(cls, registry: AgentCapabilityRegistry) -> CapabilityCatalog:
        return cls([CapabilityMetadata.from_agent_capability(definition) for definition in registry.list_definitions()])

    def list_metadata(self) -> list[CapabilityMetadata]:
        return list(self._capabilities)

    def allowed_for_intent(self, intent: str) -> list[CapabilityMetadata]:
        return [capability for capability in self._capabilities if intent in capability.allowed_intents]

    def excluded_for_intent(self, intent: str) -> list[CapabilityMetadata]:
        allowed_names = {capability.name for capability in self.allowed_for_intent(intent)}
        return [capability for capability in self._capabilities if capability.name not in allowed_names]
