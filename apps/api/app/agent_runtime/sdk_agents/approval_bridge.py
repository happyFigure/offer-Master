from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from app.agent_runtime.agent_as_tool import StandardAgentResult
from app.domains.automation.schemas import ApprovalRequestCreate


SDK_AGENT_APPROVAL_METADATA_KEY = "sdk_agent_approval"
SDK_AGENT_APPROVAL_ERROR_CODE = "SDK_AGENT_TOOL_APPROVAL_REQUIRED"


def sdk_agent_approval_payload_from_result(
    result: StandardAgentResult,
    *,
    outer_capability: str,
    outer_tool_input: dict[str, Any],
    executor_id: str,
) -> dict[str, Any] | None:
    raw_result = _payload_to_dict(result.raw_result)
    approval_request = raw_result.get("approval_request")
    if not result.requires_user_action or not isinstance(approval_request, dict):
        return None

    requested_tool_name = str(approval_request.get("tool_name") or "").strip()
    if not requested_tool_name:
        return None

    tool_input = approval_request.get("tool_input")
    if not isinstance(tool_input, dict):
        tool_input = {}

    reason = str(approval_request.get("reason") or result.summary or "SDK agent requested user approval.").strip()
    user_message = str(approval_request.get("suggested_user_message") or reason).strip()
    risk_level = str(approval_request.get("risk_level") or "high").strip() or "high"

    guard_result = {
        "ok": False,
        "error_code": SDK_AGENT_APPROVAL_ERROR_CODE,
        "reason": reason,
        "user_message": user_message,
        "next_action": "wait_confirmation",
        "retryable": True,
        "error_details": {
            "executor_id": executor_id,
            "outer_capability": outer_capability,
            "requested_tool_name": requested_tool_name,
            "tool_input": dict(tool_input),
            "risk_level": risk_level,
            "approval_type": approval_request.get("approval_type") or "tool_call",
            "sdk_agent_task_id": raw_result.get("task_id"),
            "sdk_agent_capability_id": raw_result.get("capability_id"),
            "subagent_name": raw_result.get("subagent_name"),
            "raw_trace_ref": raw_result.get("raw_trace_ref"),
        },
        "cost": {},
        "artifacts": {},
    }

    return {
        "sdk_agent_approval": True,
        "executor_id": executor_id,
        "outer_capability": outer_capability,
        "outer_tool_input": dict(outer_tool_input),
        "requested_tool_name": requested_tool_name,
        "tool_input": dict(tool_input),
        "reason": reason,
        "user_message": user_message,
        "risk_level": risk_level,
        "approval_request": dict(approval_request),
        "sdk_agent_result": raw_result,
        "raw_trace_ref": raw_result.get("raw_trace_ref"),
        "guard_result": guard_result,
    }


def with_sdk_agent_approval_metadata(metadata: dict[str, Any], approval_payload: dict[str, Any]) -> dict[str, Any]:
    return {**dict(metadata), SDK_AGENT_APPROVAL_METADATA_KEY: dict(approval_payload)}


def sdk_agent_approval_payload_from_state_metadata(metadata: dict[str, Any]) -> dict[str, Any] | None:
    payload = metadata.get(SDK_AGENT_APPROVAL_METADATA_KEY) if isinstance(metadata, dict) else None
    return dict(payload) if isinstance(payload, dict) and payload.get("sdk_agent_approval") else None


def is_sdk_agent_approval_payload(payload: Any) -> bool:
    return isinstance(payload, dict) and bool(payload.get("sdk_agent_approval"))


def sdk_agent_run_state_from_approval_payload(payload: Any) -> Any | None:
    if not is_sdk_agent_approval_payload(payload):
        return None
    sdk_result = payload.get("sdk_agent_result")
    if not isinstance(sdk_result, dict):
        return None
    if sdk_result.get("run_state") is not None:
        return sdk_result.get("run_state")
    metadata = sdk_result.get("metadata")
    return metadata.get("run_state") if isinstance(metadata, dict) else None


def sdk_agent_has_resumable_run_state(payload: Any) -> bool:
    return sdk_agent_run_state_from_approval_payload(payload) is not None


def sdk_agent_approval_decision_metadata(
    approval_payload: dict[str, Any],
    *,
    approved: bool,
    decision_reason: str | None = None,
) -> dict[str, Any]:
    # Store a compact audit summary; raw SDK traces stay behind operation refs/logs.
    tool_input = approval_payload.get("tool_input")
    outer_tool_input = approval_payload.get("outer_tool_input")
    return {
        "sdk_agent_approval": True,
        "approved": bool(approved),
        "decision_reason": decision_reason,
        "requested_tool_name": str(approval_payload.get("requested_tool_name") or ""),
        "tool_input": dict(tool_input) if isinstance(tool_input, dict) else {},
        "outer_capability": str(approval_payload.get("outer_capability") or ""),
        "outer_tool_input": dict(outer_tool_input) if isinstance(outer_tool_input, dict) else {},
        "executor_id": str(approval_payload.get("executor_id") or ""),
        "risk_level": str(approval_payload.get("risk_level") or "high"),
        "raw_trace_ref": approval_payload.get("raw_trace_ref"),
    }


def with_sdk_agent_approval_decision_metadata(
    metadata: dict[str, Any],
    approval_payload: dict[str, Any],
    *,
    approved: bool,
    decision_reason: str | None = None,
) -> dict[str, Any]:
    cleaned_metadata = dict(metadata)
    cleaned_metadata.pop(SDK_AGENT_APPROVAL_METADATA_KEY, None)
    return {
        **cleaned_metadata,
        "sdk_agent_approval_decision": sdk_agent_approval_decision_metadata(
            approval_payload,
            approved=approved,
            decision_reason=decision_reason,
        ),
    }


def sdk_agent_approval_rejected_response(approval_payload: dict[str, Any], *, decision_reason: str | None = None) -> str:
    tool_name = str(approval_payload.get("requested_tool_name") or "the requested tool").strip()
    reason = str(decision_reason or "").strip()
    reason_text = f"拒绝原因：{reason}。" if reason else ""
    return f"用户已拒绝 SDK 子 Agent 申请的高风险工具：{tool_name}。{reason_text}系统没有执行该工具。"


def approval_request_create_from_waiting_payload(
    *,
    workflow_run_id: str,
    fallback_action_type: str,
    fallback_prompt: str,
    fallback_payload: dict[str, Any],
    state_metadata: dict[str, Any],
) -> ApprovalRequestCreate:
    sdk_payload = sdk_agent_approval_payload_from_state_metadata(state_metadata)
    if sdk_payload is None:
        return ApprovalRequestCreate(
            workflow_run_id=workflow_run_id,
            action_type=fallback_action_type,
            prompt=fallback_prompt,
            payload=fallback_payload,
        )

    return ApprovalRequestCreate(
        workflow_run_id=workflow_run_id,
        action_type=str(sdk_payload.get("requested_tool_name") or fallback_action_type),
        prompt=str(sdk_payload.get("user_message") or sdk_payload.get("reason") or fallback_prompt),
        payload={**sdk_payload, "outer_approval_payload": dict(fallback_payload)},
    )


def _payload_to_dict(raw_payload: Any) -> dict[str, Any]:
    if isinstance(raw_payload, dict):
        return dict(raw_payload)
    if is_dataclass(raw_payload):
        return asdict(raw_payload)
    model_dump = getattr(raw_payload, "model_dump", None)
    if callable(model_dump):
        return dict(model_dump(mode="json"))
    return {}


__all__ = [
    "SDK_AGENT_APPROVAL_ERROR_CODE",
    "SDK_AGENT_APPROVAL_METADATA_KEY",
    "approval_request_create_from_waiting_payload",
    "is_sdk_agent_approval_payload",
    "sdk_agent_has_resumable_run_state",
    "sdk_agent_approval_decision_metadata",
    "sdk_agent_approval_payload_from_result",
    "sdk_agent_approval_payload_from_state_metadata",
    "sdk_agent_approval_rejected_response",
    "sdk_agent_run_state_from_approval_payload",
    "with_sdk_agent_approval_decision_metadata",
    "with_sdk_agent_approval_metadata",
]
