from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.agent_runtime.routing.schemas import RouteDecision
from app.agent_runtime.tool_registry import AgentToolRegistry


_EXECUTABLE_ROUTES = {"external_agent", "local_tool", "local_workflow", "browser_executor"}


@dataclass(frozen=True)
class CapabilityRouteGuardResult:
    ok: bool
    reason: str = ""
    blocked: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "blocked": self.blocked,
            "details": dict(self.details),
        }


def validate_route_decision(
    decision: RouteDecision,
    *,
    context_pack: dict[str, Any],
    registry: AgentToolRegistry,
) -> CapabilityRouteGuardResult:
    allowed_capabilities = [str(name) for name in context_pack.get("allowed_capabilities") or [] if str(name).strip()]
    if decision.route not in _EXECUTABLE_ROUTES:
        return CapabilityRouteGuardResult(ok=True)

    capability = str(decision.capability or "").strip()
    if not capability:
        return CapabilityRouteGuardResult(
            ok=False,
            blocked=True,
            reason=f"Executable route {decision.route} did not provide a capability.",
            details={"route": decision.route, "allowed_capabilities": allowed_capabilities},
        )
    if capability not in allowed_capabilities:
        return CapabilityRouteGuardResult(
            ok=False,
            blocked=True,
            reason=f"Capability route requested a capability outside this turn's ContextPack: {capability}",
            details={"requested_capability": capability, "allowed_capabilities": allowed_capabilities},
        )
    if registry.get(capability) is None:
        return CapabilityRouteGuardResult(
            ok=False,
            blocked=True,
            reason=f"Capability route requested an unregistered capability: {capability}",
            details={"requested_capability": capability},
        )
    definition = registry.get(capability)
    validation_error = _validate_tool_input(definition.input_schema, decision.tool_input) if definition is not None else None
    if validation_error is not None:
        return CapabilityRouteGuardResult(
            ok=False,
            blocked=True,
            reason=validation_error,
            details={"requested_capability": capability, "tool_input": dict(decision.tool_input)},
        )
    return CapabilityRouteGuardResult(ok=True)


def _validate_tool_input(input_schema: dict[str, Any], tool_input: dict[str, Any]) -> str | None:
    required = input_schema.get("required") if isinstance(input_schema, dict) else None
    if isinstance(required, list):
        missing = [str(name) for name in required if str(name) not in tool_input]
        if missing:
            return f"Capability route tool input is missing required arguments: {', '.join(missing)}"

    properties = input_schema.get("properties") if isinstance(input_schema, dict) else None
    if input_schema.get("additionalProperties") is False and isinstance(properties, dict):
        extra = sorted(set(tool_input) - set(properties))
        if extra:
            return f"Capability route tool input included unsupported arguments: {', '.join(extra)}"

    if isinstance(properties, dict):
        for name, schema in properties.items():
            if name not in tool_input or not isinstance(schema, dict):
                continue
            if not _matches_json_schema_type(tool_input[name], schema.get("type")):
                return f"Capability route tool input argument {name} has invalid type."
    return None


def _matches_json_schema_type(value: Any, expected_type: Any) -> bool:
    if expected_type is None:
        return True
    if isinstance(expected_type, list):
        return any(_matches_json_schema_type(value, item) for item in expected_type)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "null":
        return value is None
    return True
