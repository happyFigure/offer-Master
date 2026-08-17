from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol


TOOL_RESULT_MISSING = "TOOL_RESULT_MISSING"
ORPHAN_TOOL_RESULT = "ORPHAN_TOOL_RESULT"
MISSING_TOOL_RESULT_MESSAGE = "工具结果缺失，本轮不能判断工具成功。"


class AgentMessageLike(Protocol):
    id: str
    role: Any
    content_text: str | None
    visible_content_text: str | None
    runtime_content_text: str | None
    message_kind: Any
    token_estimate: int | None
    parent_message_id: str | None
    exclude_from_context: bool
    compacted_by_summary_id: str | None
    metadata_json: dict[str, Any] | None


@dataclass(frozen=True)
class HygieneMessage:
    id: str
    role: str
    content_text: str | None = None
    visible_content_text: str | None = None
    runtime_content_text: str | None = None
    message_kind: str | None = None
    token_estimate: int | None = None
    parent_message_id: str | None = None
    exclude_from_context: bool = False
    compacted_by_summary_id: str | None = None
    metadata_json: dict[str, Any] | None = None
    content_json: dict[str, Any] | None = None


@dataclass(frozen=True)
class HygieneResult:
    messages: list[AgentMessageLike]
    synthetic_error_ids: list[str]
    excluded_reasons: dict[str, str]


def repair_tool_result_pairing(messages: Sequence[AgentMessageLike]) -> HygieneResult:
    repaired: list[AgentMessageLike] = []
    synthetic_error_ids: list[str] = []
    excluded_reasons: dict[str, str] = {}
    pending_tool_call: AgentMessageLike | None = None

    for message in messages:
        role = _role_value(message)
        if pending_tool_call is not None:
            if role == "tool_result" and _is_result_for_call(pending_tool_call, message):
                repaired.append(message)
                pending_tool_call = None
                continue

            synthetic = build_synthetic_tool_error(pending_tool_call)
            repaired.append(synthetic)
            synthetic_error_ids.append(synthetic.id)
            pending_tool_call = None

        if role == "tool_call":
            repaired.append(message)
            pending_tool_call = message
        elif role == "tool_result":
            excluded_reasons[message.id] = ORPHAN_TOOL_RESULT
        else:
            repaired.append(message)

    if pending_tool_call is not None:
        synthetic = build_synthetic_tool_error(pending_tool_call)
        repaired.append(synthetic)
        synthetic_error_ids.append(synthetic.id)

    return HygieneResult(
        messages=repaired,
        synthetic_error_ids=synthetic_error_ids,
        excluded_reasons=excluded_reasons,
    )


def build_synthetic_tool_error(
    tool_call_message: AgentMessageLike,
    code: str = TOOL_RESULT_MISSING,
) -> HygieneMessage:
    content = f"{MISSING_TOOL_RESULT_MESSAGE} tool_call_id={tool_call_message.id}"
    return HygieneMessage(
        id=f"synthetic-error-{tool_call_message.id}",
        role="assistant",
        content_text=content,
        visible_content_text=content,
        runtime_content_text=None,
        message_kind="synthetic_error",
        token_estimate=24,
        parent_message_id=tool_call_message.id,
        metadata_json={
            "error_code": code,
            "tool_call_message_id": tool_call_message.id,
        },
    )


def filter_visible_transcript(messages: Sequence[AgentMessageLike]) -> list[HygieneMessage]:
    visible_messages: list[HygieneMessage] = []
    for message in messages:
        visible_text = _visible_text(message)
        if visible_text is None:
            continue
        visible_messages.append(
            HygieneMessage(
                id=message.id,
                role=_role_value(message),
                content_text=visible_text,
                visible_content_text=visible_text,
                runtime_content_text=None,
                message_kind=_message_kind_value(message),
                token_estimate=getattr(message, "token_estimate", None),
                parent_message_id=getattr(message, "parent_message_id", None),
                exclude_from_context=getattr(message, "exclude_from_context", False),
                compacted_by_summary_id=getattr(message, "compacted_by_summary_id", None),
                metadata_json=getattr(message, "metadata_json", None),
            )
        )
    return visible_messages


def _is_result_for_call(tool_call: AgentMessageLike, tool_result: AgentMessageLike) -> bool:
    parent_message_id = getattr(tool_result, "parent_message_id", None)
    return parent_message_id in {None, tool_call.id}


def _visible_text(message: AgentMessageLike) -> str | None:
    return getattr(message, "visible_content_text", None) or getattr(message, "content_text", None)


def _role_value(message: AgentMessageLike) -> str:
    role = getattr(message, "role", "")
    return str(getattr(role, "value", role))


def _message_kind_value(message: AgentMessageLike) -> str | None:
    message_kind = getattr(message, "message_kind", None)
    if message_kind is None:
        return None
    return str(getattr(message_kind, "value", message_kind))
