from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agent_runtime.agent_as_tool import (
    DEFAULT_SUPPORTED_INTENTS_BY_CAPABILITY,
    AgentCapabilityDefinition,
    AgentCapabilityRegistry,
    LEGACY_FILESYSTEM_CAPABILITIES,
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
    kind: str = "tool"
    always_available: bool = False
    requires_confirmation: bool = False
    allowed_source_types: tuple[str, ...] = ()
    semantic_profile: dict[str, Any] | None = None

    @classmethod
    def from_tool_definition(cls, definition: AgentToolDefinition) -> CapabilityMetadata:
        properties = definition.input_schema.get("properties") if isinstance(definition.input_schema, dict) else None
        return cls(
            name=definition.name,
            description=definition.description,
            risk_level=definition.risk_level.value,
            input_summary=sorted(str(key) for key in properties.keys()) if isinstance(properties, dict) else [],
            allowed_intents=_ALLOWED_INTENTS_BY_TOOL.get(definition.name, ()),
            kind="tool",
            requires_confirmation=definition.requires_confirmation,
            semantic_profile=_semantic_profile_payload(getattr(definition, "semantic_profile", None)),
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
            kind=definition.kind,
            always_available=definition.always_available,
            requires_confirmation=definition.requires_confirmation,
            allowed_source_types=tuple(sorted(definition.allowed_source_types)),
            semantic_profile=_semantic_profile_payload(getattr(definition, "semantic_profile", None)),
        )

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "risk_level": self.risk_level,
            "input_summary": list(self.input_summary),
            "allowed_intents": list(self.allowed_intents),
            "kind": self.kind,
            "always_available": self.always_available,
            "requires_confirmation": self.requires_confirmation,
            "allowed_source_types": list(self.allowed_source_types),
            "semantic_profile": self.semantic_profile,
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
        return cls(
            [
                CapabilityMetadata.from_agent_capability(definition)
                for definition in registry.list_definitions()
                if definition.capability_id not in LEGACY_FILESYSTEM_CAPABILITIES
            ]
        )

    def list_metadata(self) -> list[CapabilityMetadata]:
        return list(self._capabilities)

    def allowed_for_intent(self, intent: str) -> list[CapabilityMetadata]:
        return [
            capability
            for capability in self._capabilities
            if capability.always_available or intent in capability.allowed_intents
        ]

    def model_visible_for_source(self, source_type: str) -> list[CapabilityMetadata]:
        """Return the registered catalog the main Agent may reason over.

        Semantic intent is deliberately absent from this filter. The model
        needs the full capability directory to resolve multi-turn language;
        runtime validation remains the execution boundary.
        """

        normalized_source = str(source_type or "agent_chat").strip() or "agent_chat"
        return [
            capability
            for capability in self._capabilities
            if not capability.allowed_source_types or normalized_source in capability.allowed_source_types
        ]

    def excluded_for_intent(self, intent: str) -> list[CapabilityMetadata]:
        allowed_names = {capability.name for capability in self.allowed_for_intent(intent)}
        return [capability for capability in self._capabilities if capability.name not in allowed_names]


def _semantic_profile_payload(profile: Any | None) -> dict[str, Any] | None:
    if profile is None:
        return None
    to_metadata = getattr(profile, "to_metadata", None)
    if callable(to_metadata):
        return to_metadata()
    if isinstance(profile, dict):
        return dict(profile)
    return None
