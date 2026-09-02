from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta

from app.agent_runtime.memory.compaction import (
    CompactionConfig,
    prepare_compaction,
)
from app.agent_runtime.memory.summary_provider import DeterministicSummaryProvider, SummaryProvider
from app.agent_runtime.memory.token_budget import estimate_tokens
from app.domains.conversations.models import (
    AgentContextSummary,
    AgentMessage,
    AgentMessageKind,
    AgentMessageProvenanceKind,
    AgentMessageRole,
    AgentMessageVisibilityScope,
    AgentSession,
    AgentSessionStatus,
    utc_now,
)
from app.domains.conversations.repository import ConversationRepository
from app.domains.conversations.schemas import AgentContextSummaryCreate, AgentMessageCreate


@dataclass(frozen=True)
class AgentCompactResult:
    summary: AgentContextSummary
    covered_message_count: int
    first_kept_message_id: str | None
    token_estimate_before: int
    token_estimate_after: int
    should_compact: bool


@dataclass(frozen=True)
class PreCompactionMemoryFlushCommand:
    session_id: str
    workflow_run_id: str | None
    agent_run_id: str | None
    target_scope: str
    message_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PreCompactionMemoryFlushResult:
    reviewed_message_count: int = 0
    reviewed_tool_call_count: int = 0
    created_candidate_ids: list[str] = field(default_factory=list)
    pending_candidate_ids: list[str] = field(default_factory=list)
    promoted_memory_ids: list[str] = field(default_factory=list)
    merged_memory_ids: list[str] = field(default_factory=list)
    skipped_reasons: list[str] = field(default_factory=list)

    def to_metadata(self) -> dict[str, object]:
        return {
            "reviewed_message_count": self.reviewed_message_count,
            "reviewed_tool_call_count": self.reviewed_tool_call_count,
            "created_candidate_ids": list(self.created_candidate_ids),
            "pending_candidate_ids": list(self.pending_candidate_ids),
            "promoted_memory_ids": list(self.promoted_memory_ids),
            "merged_memory_ids": list(self.merged_memory_ids),
            "skipped_reasons": list(self.skipped_reasons),
        }


def _summary_json_with_previous_summary(
    summary_json: dict | None,
    *,
    previous_summary_id: str | None,
) -> dict | None:
    if summary_json is None:
        return None
    payload = dict(summary_json)
    key_decisions = payload.get("Key Decisions")
    if isinstance(key_decisions, dict):
        payload["Key Decisions"] = {
            **key_decisions,
            "previous_summary_id": previous_summary_id,
        }
    return payload


class ConversationService:
    def __init__(
        self,
        repository: ConversationRepository,
        *,
        summary_provider: SummaryProvider | None = None,
        pre_compaction_memory_flush: Callable[[PreCompactionMemoryFlushCommand], PreCompactionMemoryFlushResult]
        | None = None,
    ) -> None:
        self._repository = repository
        self._summary_provider = summary_provider or DeterministicSummaryProvider()
        self._pre_compaction_memory_flush = pre_compaction_memory_flush

    def create_session(
        self,
        *,
        title: str | None = None,
        primary_intent: str | None = None,
        metadata_json: dict | None = None,
    ) -> AgentSession:
        return self._repository.add_session(
            AgentSession(
                title=title,
                status=AgentSessionStatus.ACTIVE,
                primary_intent=primary_intent,
                metadata_json=metadata_json,
            )
        )

    def list_sessions(self, *, limit: int = 50, offset: int = 0, include_archived: bool = False) -> list[AgentSession]:
        return self._repository.list_sessions(limit=limit, offset=offset, include_archived=include_archived)

    def get_session(self, session_id: str) -> AgentSession:
        return self._require_session(session_id)

    def update_session(
        self,
        session_id: str,
        *,
        title: str | None = None,
        primary_intent: str | None = None,
        metadata_json: dict | None = None,
    ) -> AgentSession:
        session = self._require_session(session_id)
        if title is not None:
            session.title = title.strip() or None
        if primary_intent is not None:
            session.primary_intent = primary_intent.strip() or None
        if metadata_json is not None:
            session.metadata_json = metadata_json
        session.updated_at = utc_now()
        self._repository.flush()
        return session

    def archive_session(self, session_id: str) -> AgentSession:
        session = self._require_session(session_id)
        session.status = AgentSessionStatus.ARCHIVED
        session.updated_at = utc_now()
        self._repository.flush()
        return session

    def append_message(self, session_id: str, draft: AgentMessageCreate) -> AgentMessage:
        session = self._require_session(session_id)
        created_at = self._next_message_created_at(session)
        message = self._repository.add_message(
            AgentMessage(
                session=session,
                role=draft.role,
                message_kind=draft.message_kind or self._default_message_kind(draft.role),
                agent_id=draft.agent_id,
                recipient_agent_id=draft.recipient_agent_id,
                visibility_scope=draft.visibility_scope or self._default_visibility_scope(draft.role),
                content_text=draft.content_text,
                content_json=draft.content_json,
                visible_content_text=draft.visible_content_text,
                runtime_content_text=draft.runtime_content_text,
                content_type=draft.content_type,
                provenance_kind=draft.provenance_kind or self._default_provenance_kind(draft.role),
                agent_run_id=draft.agent_run_id,
                workflow_run_id=draft.workflow_run_id,
                tool_call_log_id=draft.tool_call_log_id,
                parent_message_id=draft.parent_message_id,
                token_estimate=draft.token_estimate,
                exclude_from_context=draft.exclude_from_context,
                created_at=created_at,
                metadata_json=draft.metadata_json,
            )
        )
        session.message_count += 1
        session.last_message_at = created_at
        session.updated_at = utc_now()
        self._repository.flush()
        return message

    def list_messages(
        self,
        session_id: str,
        *,
        limit: int = 100,
        before_message_id: str | None = None,
    ) -> list[AgentMessage]:
        self._require_session(session_id)
        return self._repository.list_messages(
            session_id,
            limit=limit,
            before_message_id=before_message_id,
        )

    def get_latest_summary(self, session_id: str) -> AgentContextSummary | None:
        self._require_session(session_id)
        return self._repository.get_latest_summary(session_id)

    def create_context_summary(
        self,
        session_id: str,
        draft: AgentContextSummaryCreate,
    ) -> AgentContextSummary:
        session = self._require_session(session_id)
        previous_summary_id = draft.previous_summary_id
        if previous_summary_id is None:
            latest_summary = self._repository.get_latest_summary(session_id)
            previous_summary_id = latest_summary.id if latest_summary is not None else None

        summary = self._repository.add_summary(
            AgentContextSummary(
                session=session,
                summary_text=draft.summary_text,
                summary_json=draft.summary_json,
                covered_message_start_id=draft.covered_message_start_id,
                covered_message_end_id=draft.covered_message_end_id,
                first_kept_message_id=draft.first_kept_message_id,
                previous_summary_id=previous_summary_id,
                token_estimate=draft.token_estimate,
                created_by=draft.created_by,
                metadata_json=draft.metadata_json,
            )
        )
        session.last_context_summary_id = summary.id
        session.updated_at = utc_now()
        self._repository.flush()
        return summary

    def mark_messages_compacted(
        self,
        session_id: str,
        message_ids: list[str],
        summary_id: str,
    ) -> int:
        self._require_session(session_id)
        summary = self._repository.get_summary(summary_id)
        if summary is None or summary.session_id != session_id:
            raise ValueError(f"Agent context summary not found: {summary_id}")

        updated_count = 0
        for message_id in message_ids:
            message = self._repository.get_message(message_id)
            if message is None or message.session_id != session_id:
                continue
            message.compacted_by_summary_id = summary.id
            updated_count += 1
        self._repository.flush()
        return updated_count

    def compact_session(
        self,
        session_id: str,
        config: CompactionConfig,
        *,
        workflow_run_id: str | None = None,
        agent_run_id: str | None = None,
        target_scope: str = "agent_memory",
    ) -> AgentCompactResult:
        self._require_session(session_id)
        latest_summary = self._repository.get_latest_summary(session_id)
        candidate_messages = self._repository.list_uncompacted_messages(session_id)
        plan = prepare_compaction(
            candidate_messages,
            latest_summary=latest_summary.summary_text if latest_summary is not None else None,
            config=config,
        )
        if not plan.messages_to_summarize:
            raise ValueError(f"Agent session has no older messages to compact: {session_id}")

        pre_flush_metadata = self._run_pre_compaction_memory_flush(
            session_id=session_id,
            plan=plan,
            workflow_run_id=workflow_run_id,
            agent_run_id=agent_run_id,
            target_scope=target_scope,
        )
        provider_result = self._summary_provider.summarize(plan)
        summary_text = provider_result.summary_text
        token_estimate_after = estimate_tokens(summary_text) + plan.kept_token_estimate
        summary = self.create_context_summary(
            session_id,
            AgentContextSummaryCreate(
                summary_text=summary_text,
                summary_json=_summary_json_with_previous_summary(
                    provider_result.summary_json,
                    previous_summary_id=latest_summary.id if latest_summary is not None else None,
                ),
                covered_message_start_id=plan.messages_to_summarize[0].id,
                covered_message_end_id=plan.messages_to_summarize[-1].id,
                first_kept_message_id=plan.first_kept_message_id,
                previous_summary_id=latest_summary.id if latest_summary is not None else None,
                token_estimate=estimate_tokens(summary_text),
                created_by=provider_result.created_by,
                metadata_json={
                    **provider_result.metadata_json,
                    "pre_compaction_memory_flush": pre_flush_metadata,
                    "token_estimate_before": plan.context_tokens,
                    "token_estimate_after": token_estimate_after,
                    "threshold_tokens": plan.threshold_tokens,
                    "should_compact": plan.should_compact,
                    "keep_recent_tokens": config.keep_recent_tokens,
                },
            ),
        )
        self.mark_messages_compacted(
            session_id,
            [message.id for message in plan.messages_to_summarize],
            summary.id,
        )
        return AgentCompactResult(
            summary=summary,
            covered_message_count=len(plan.messages_to_summarize),
            first_kept_message_id=plan.first_kept_message_id,
            token_estimate_before=plan.context_tokens,
            token_estimate_after=token_estimate_after,
            should_compact=plan.should_compact,
        )

    def _run_pre_compaction_memory_flush(
        self,
        *,
        session_id: str,
        plan,
        workflow_run_id: str | None,
        agent_run_id: str | None,
        target_scope: str,
    ) -> dict[str, object]:
        if self._pre_compaction_memory_flush is None:
            return PreCompactionMemoryFlushResult(skipped_reasons=["pre_compaction_memory_flush_not_configured"]).to_metadata()
        try:
            result = self._pre_compaction_memory_flush(
                PreCompactionMemoryFlushCommand(
                    session_id=session_id,
                    workflow_run_id=workflow_run_id,
                    agent_run_id=agent_run_id,
                    target_scope=target_scope,
                    message_ids=[message.id for message in plan.messages_to_summarize],
                )
            )
            return result.to_metadata()
        except Exception as exc:
            return {
                **PreCompactionMemoryFlushResult(
                    skipped_reasons=["pre_compaction_memory_flush_failed"]
                ).to_metadata(),
                "error": str(exc)[:500] or exc.__class__.__name__,
            }

    def _require_session(self, session_id: str) -> AgentSession:
        session = self._repository.get_session(session_id)
        if session is None:
            raise ValueError(f"Agent session not found: {session_id}")
        return session

    @staticmethod
    def _next_message_created_at(session: AgentSession):
        current_time = utc_now()
        if session.last_message_at is not None and current_time <= session.last_message_at:
            return session.last_message_at + timedelta(microseconds=1)
        return current_time

    @staticmethod
    def _default_message_kind(role: AgentMessageRole) -> AgentMessageKind:
        return {
            AgentMessageRole.SYSTEM: AgentMessageKind.SYSTEM_TEXT,
            AgentMessageRole.USER: AgentMessageKind.USER_TEXT,
            AgentMessageRole.ASSISTANT: AgentMessageKind.ASSISTANT_TEXT,
            AgentMessageRole.TOOL_CALL: AgentMessageKind.TOOL_CALL,
            AgentMessageRole.TOOL_RESULT: AgentMessageKind.TOOL_RESULT,
        }[role]

    @staticmethod
    def _default_visibility_scope(role: AgentMessageRole) -> AgentMessageVisibilityScope:
        if role in {AgentMessageRole.TOOL_CALL, AgentMessageRole.TOOL_RESULT, AgentMessageRole.SYSTEM}:
            return AgentMessageVisibilityScope.RUNTIME_ONLY
        return AgentMessageVisibilityScope.USER_VISIBLE

    @staticmethod
    def _default_provenance_kind(role: AgentMessageRole) -> AgentMessageProvenanceKind:
        return {
            AgentMessageRole.SYSTEM: AgentMessageProvenanceKind.SYSTEM_GENERATED,
            AgentMessageRole.USER: AgentMessageProvenanceKind.USER_INPUT,
            AgentMessageRole.ASSISTANT: AgentMessageProvenanceKind.AGENT_GENERATED,
            AgentMessageRole.TOOL_CALL: AgentMessageProvenanceKind.TOOL_CALL,
            AgentMessageRole.TOOL_RESULT: AgentMessageProvenanceKind.TOOL_RESULT,
        }[role]
