from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.agent_runtime.memory.token_budget import (
    DEFAULT_RESERVE_TOKENS,
    estimate_message_tokens,
    estimate_tokens,
    should_compact as should_compact_for_budget,
)


DEFAULT_KEEP_RECENT_TOKENS = 20000


class AgentMessageLike(Protocol):
    id: str
    role: str
    content_text: str | None
    visible_content_text: str | None
    runtime_content_text: str | None
    token_estimate: int | None
    parent_message_id: str | None


@dataclass(frozen=True)
class CompactionConfig:
    context_window: int
    reserve_tokens: int = DEFAULT_RESERVE_TOKENS
    keep_recent_tokens: int = DEFAULT_KEEP_RECENT_TOKENS


@dataclass(frozen=True)
class CompactionCut:
    messages_to_summarize: list[AgentMessageLike]
    recent_messages_to_keep: list[AgentMessageLike]
    first_kept_message_id: str | None
    kept_token_estimate: int


@dataclass(frozen=True)
class CompactionPlan:
    should_compact: bool
    context_tokens: int
    threshold_tokens: int
    previous_summary: str | None
    messages_to_summarize: list[AgentMessageLike]
    recent_messages_to_keep: list[AgentMessageLike]
    first_kept_message_id: str | None
    kept_token_estimate: int


def find_cut_point(
    messages: Sequence[AgentMessageLike],
    *,
    keep_recent_tokens: int = DEFAULT_KEEP_RECENT_TOKENS,
) -> CompactionCut:
    if not messages:
        return CompactionCut([], [], None, 0)

    running_tokens = 0
    cut_index = len(messages)
    for index in range(len(messages) - 1, -1, -1):
        message_tokens = estimate_message_tokens([messages[index]])
        if cut_index < len(messages) and running_tokens + message_tokens > keep_recent_tokens:
            break
        running_tokens += message_tokens
        cut_index = index

    while cut_index > 0 and _would_split_tool_pair(messages, cut_index):
        cut_index -= 1
        running_tokens += estimate_message_tokens([messages[cut_index]])

    kept = list(messages[cut_index:])
    return CompactionCut(
        messages_to_summarize=list(messages[:cut_index]),
        recent_messages_to_keep=kept,
        first_kept_message_id=kept[0].id if kept else None,
        kept_token_estimate=running_tokens,
    )


def prepare_compaction(
    messages: Sequence[AgentMessageLike],
    *,
    latest_summary: str | None,
    config: CompactionConfig,
) -> CompactionPlan:
    context_tokens = estimate_tokens(latest_summary) + estimate_message_tokens(messages)
    threshold_tokens = config.context_window - config.reserve_tokens
    cut = find_cut_point(messages, keep_recent_tokens=config.keep_recent_tokens)
    return CompactionPlan(
        should_compact=should_compact_for_budget(
            context_tokens=context_tokens,
            context_window=config.context_window,
            reserve_tokens=config.reserve_tokens,
        ),
        context_tokens=context_tokens,
        threshold_tokens=threshold_tokens,
        previous_summary=latest_summary,
        messages_to_summarize=cut.messages_to_summarize,
        recent_messages_to_keep=cut.recent_messages_to_keep,
        first_kept_message_id=cut.first_kept_message_id,
        kept_token_estimate=cut.kept_token_estimate,
    )


def build_summary_prompt(plan: CompactionPlan) -> str:
    previous_summary = plan.previous_summary or "None"
    transcript = "\n".join(_format_message_for_summary(message) for message in plan.messages_to_summarize)
    return f"""Summarize the older transcript for future Agent context.

Preserve evidence ids and concrete decisions. Do not invent facts. Output with these sections:

Goal:

Constraints & Preferences:

Progress:

Key Decisions:

Next Steps:

Critical Context:

Retrieval Hints:

Previous Summary:
{previous_summary}

Transcript To Summarize:
{transcript}
""".strip()


def build_deterministic_summary(plan: CompactionPlan) -> str:
    covered_ids = [message.id for message in plan.messages_to_summarize]
    first_id = covered_ids[0] if covered_ids else "None"
    last_id = covered_ids[-1] if covered_ids else "None"
    critical_context = "\n".join(f"- {_format_message_for_summary(message)}" for message in plan.messages_to_summarize)
    if not critical_context:
        critical_context = "- No older messages were selected for compaction."

    previous_summary_block = plan.previous_summary or "None"
    return f"""Goal:
- Preserve the older conversation context for future Agent turns.

Constraints & Preferences:
- Do not delete original transcript messages.
- Keep recent messages as raw context and use this summary for older history.

Progress:
- Compacted {len(covered_ids)} older messages into this deterministic summary.

Key Decisions:
- Covered message range: {first_id} -> {last_id}.
- First raw message kept after compaction: {plan.first_kept_message_id or 'None'}.

Next Steps:
- Build future context from latest summary plus recent uncompacted messages.

Critical Context:
{critical_context}

Retrieval Hints:
- covered_message_ids={','.join(covered_ids)}
- previous_summary={previous_summary_block}
""".strip()


def _would_split_tool_pair(messages: Sequence[AgentMessageLike], cut_index: int) -> bool:
    previous = messages[cut_index - 1]
    current = messages[cut_index]
    return _role_value(previous) == "tool_call" and _role_value(current) == "tool_result" and _is_result_for_call(
        tool_call=previous,
        tool_result=current,
    )


def _is_result_for_call(*, tool_call: AgentMessageLike, tool_result: AgentMessageLike) -> bool:
    parent_message_id = getattr(tool_result, "parent_message_id", None)
    return parent_message_id in {None, tool_call.id}


def _format_message_for_summary(message: AgentMessageLike) -> str:
    text = _message_text(message)
    return f"[{message.id}] {_role_value(message)}: {text}"


def _message_text(message: AgentMessageLike) -> str:
    return (
        getattr(message, "visible_content_text", None)
        or getattr(message, "content_text", None)
        or getattr(message, "runtime_content_text", None)
        or ""
    )


def _role_value(message: AgentMessageLike) -> str:
    role = getattr(message, "role", "")
    return str(getattr(role, "value", role))
