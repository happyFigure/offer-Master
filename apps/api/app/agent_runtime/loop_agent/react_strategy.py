from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DEFAULT_BOUNDED_REACT_MAX_STEPS = 5
DEFAULT_BOUNDED_REACT_CAPABILITIES = frozenset({"external.web_search"})
LOW_RISK_LEVELS = {"low"}


@dataclass(frozen=True)
class BoundedReActPolicy:
    enabled: bool
    allowed_capabilities: list[str] = field(default_factory=list)
    max_steps: int = 0
    disabled_reason: str | None = None

    @classmethod
    def from_context_pack(
        cls,
        context_pack: dict[str, Any],
        *,
        requested_max_steps: int,
        allowlist: frozenset[str] = DEFAULT_BOUNDED_REACT_CAPABILITIES,
    ) -> BoundedReActPolicy:
        risk_level = str(context_pack.get("risk_level") or "unknown")
        if risk_level not in LOW_RISK_LEVELS:
            return cls(enabled=False, disabled_reason=f"risk level is not allowed for bounded ReAct: {risk_level}")

        capabilities = [str(name) for name in context_pack.get("allowed_capabilities") or [] if str(name).strip()]
        safe_capabilities = [name for name in capabilities if name in allowlist]
        if not safe_capabilities:
            return cls(enabled=False, disabled_reason="no allowed low-risk capability is available for bounded ReAct")
        if len(safe_capabilities) != len(capabilities):
            blocked = sorted(set(capabilities) - set(safe_capabilities))
            return cls(enabled=False, disabled_reason=f"capabilities are not allowed for bounded ReAct: {', '.join(blocked)}")

        max_steps = min(DEFAULT_BOUNDED_REACT_MAX_STEPS, max(1, int(requested_max_steps or 1)))
        return cls(enabled=True, allowed_capabilities=safe_capabilities, max_steps=max_steps)

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": "bounded_react",
            "control": "runtime_limited",
            "allowed_capabilities": list(self.allowed_capabilities),
            "max_steps": self.max_steps,
            "disabled_reason": self.disabled_reason,
            "guards": [
                "max_steps",
                "capability_allowlist",
                "low_risk_only",
                "runtime_guard",
                "public_trace_only",
                "human_in_the_loop_on_risk",
            ],
        }
