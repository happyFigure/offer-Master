from __future__ import annotations

from typing import Any

from app.agent_runtime.sdk_agents.schemas import SdkAgentResultEnvelope, SdkAgentTaskEnvelope


SDK_AGENT_TELEMETRY_SCHEMA_VERSION = "offer_master.sdk_agent_telemetry.v1"


def build_sdk_agent_telemetry(
    envelope: SdkAgentTaskEnvelope,
    result: SdkAgentResultEnvelope,
) -> dict[str, Any]:
    sandbox_metadata = result.metadata.get("sandbox") if isinstance(result.metadata.get("sandbox"), dict) else None
    return {
        "schema_version": SDK_AGENT_TELEMETRY_SCHEMA_VERSION,
        "subagent_name": result.subagent_name,
        "capability_id": result.capability_id,
        "status": result.status,
        "allowed_tool_count": len(envelope.allowed_tools),
        "tool_call_count": result.trace_summary.tool_call_count,
        "retry_count": result.trace_summary.retry_count,
        "approval_required": result.status == "needs_approval",
        "operation_ref_count": len(result.operation_refs),
        "sandbox": _sandbox_telemetry(sandbox_metadata),
        "delegation": _delegation_telemetry(envelope.risk_policy),
    }


def with_sdk_agent_telemetry(
    envelope: SdkAgentTaskEnvelope,
    result: SdkAgentResultEnvelope,
) -> SdkAgentResultEnvelope:
    metadata = dict(result.metadata)
    # Keep observability compact: counts and public refs only, never raw tool traces or private sandbox paths.
    metadata["telemetry"] = build_sdk_agent_telemetry(envelope, result)
    return result.model_copy(update={"metadata": metadata})


def _sandbox_telemetry(sandbox_metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not sandbox_metadata:
        return {"enabled": False, "artifact_count": 0}
    artifacts = sandbox_metadata.get("artifacts") if isinstance(sandbox_metadata.get("artifacts"), list) else []
    payload: dict[str, Any] = {
        "enabled": True,
        "artifact_count": len(artifacts),
    }
    run_id = str(sandbox_metadata.get("run_id") or "").strip()
    if run_id:
        payload["run_id"] = run_id
    return payload


def _delegation_telemetry(risk_policy: dict[str, Any]) -> dict[str, Any]:
    delegation = risk_policy.get("delegation") if isinstance(risk_policy.get("delegation"), dict) else {}
    retained = delegation.get("runtime_retained_tools") if isinstance(delegation.get("runtime_retained_tools"), list) else []
    return {"runtime_retained_tool_count": len(retained)}


__all__ = [
    "SDK_AGENT_TELEMETRY_SCHEMA_VERSION",
    "build_sdk_agent_telemetry",
    "with_sdk_agent_telemetry",
]
