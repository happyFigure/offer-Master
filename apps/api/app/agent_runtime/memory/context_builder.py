from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.agent_runtime.memory.compaction import DEFAULT_KEEP_RECENT_TOKENS
from app.agent_runtime.memory.skill_repository import AgentSkillRepository, SkillDocument
from app.agent_runtime.memory.token_budget import DEFAULT_RESERVE_TOKENS, estimate_message_tokens, estimate_tokens, should_compact
from app.agent_runtime.memory.transcript_hygiene import repair_tool_result_pairing
from app.agent_runtime.tool_permissions import AgentToolPermissionPolicy
from app.domains.agent_memory.models import AgentSkillStatus, AgentSkillUsageEvent
from app.domains.conversations.models import AgentContextSummary, AgentMessage, AgentMessageRole
from app.domains.conversations.service import ConversationService


@dataclass(frozen=True)
class ContextBuildConfig:
    context_window: int = 64000
    reserve_tokens: int = DEFAULT_RESERVE_TOKENS
    keep_recent_tokens: int = DEFAULT_KEEP_RECENT_TOKENS
    max_recent_messages: int = 50
    max_loaded_skills: int = 3
    max_skill_context_chars: int = 4000


@dataclass(frozen=True)
class BuiltContext:
    llm_messages: list[dict[str, Any]]
    context_metadata: dict[str, Any]
    loaded_session_history_ids: list[str]
    loaded_memory_ids: list[str] = field(default_factory=list)
    loaded_skill_ids: list[str] = field(default_factory=list)
    token_estimate: int = 0
    need_compaction: bool = False


class MemoryContextBuilder:
    def __init__(
        self,
        conversation_service: ConversationService,
        *,
        skill_repository: AgentSkillRepository | None = None,
    ) -> None:
        self._conversation_service = conversation_service
        self._skill_repository = skill_repository

    def build(
        self,
        session_id: str,
        *,
        new_user_message: str | None,
        config: ContextBuildConfig,
    ) -> BuiltContext:
        self._conversation_service.get_session(session_id)
        latest_summary = self._conversation_service.get_latest_summary(session_id)
        recent_messages = self._recent_context_messages(session_id, config.max_recent_messages)
        hygiene_result = repair_tool_result_pairing(recent_messages)
        context_messages = hygiene_result.messages
        skill_documents = self._load_relevant_skill_documents(new_user_message, config=config)

        llm_messages: list[dict[str, Any]] = []
        if latest_summary is not None:
            llm_messages.append(_summary_message(latest_summary))

        for document in skill_documents:
            llm_messages.append(_skill_message(document, max_chars=config.max_skill_context_chars))

        for message in context_messages:
            llm_message = _message_to_llm_message(message)
            if llm_message is not None:
                llm_messages.append(llm_message)

        if new_user_message:
            llm_messages.append(
                {
                    "role": "user",
                    "content": new_user_message,
                    "metadata": {"source": "new_user_message"},
                }
            )

        summary_tokens = latest_summary.token_estimate or estimate_tokens(latest_summary.summary_text) if latest_summary else 0
        skill_tokens = sum(estimate_tokens(_skill_context_content(document, config.max_skill_context_chars)) for document in skill_documents)
        token_estimate = summary_tokens + skill_tokens + estimate_message_tokens(context_messages) + estimate_tokens(new_user_message)
        threshold_tokens = config.context_window - config.reserve_tokens
        need_compaction = should_compact(
            context_tokens=token_estimate,
            context_window=config.context_window,
            reserve_tokens=config.reserve_tokens,
        )
        loaded_history_ids = [message.id for message in context_messages]
        loaded_skill_ids = [document.skill.id for document in skill_documents]
        skill_tool_permission_policy = AgentToolPermissionPolicy.from_loaded_skill_metadata(
            [(document.skill.id, document.skill.metadata_json) for document in skill_documents]
        ).to_metadata()

        return BuiltContext(
            llm_messages=llm_messages,
            context_metadata={
                "session_id": session_id,
                "summary_id": latest_summary.id if latest_summary is not None else None,
                "loaded_session_history_ids": loaded_history_ids,
                "loaded_memory_ids": [],
                "loaded_skill_ids": loaded_skill_ids,
                "skill_tool_permission_policy": skill_tool_permission_policy,
                "recent_message_count": len(context_messages),
                "hygiene_synthetic_error_ids": hygiene_result.synthetic_error_ids,
                "hygiene_excluded_reasons": hygiene_result.excluded_reasons,
                "new_user_message_included": bool(new_user_message),
                "token_estimate": token_estimate,
                "threshold_tokens": threshold_tokens,
                "need_compaction": need_compaction,
                "context_window": config.context_window,
                "reserve_tokens": config.reserve_tokens,
                "keep_recent_tokens": config.keep_recent_tokens,
                "max_recent_messages": config.max_recent_messages,
                "max_loaded_skills": config.max_loaded_skills,
                "max_skill_context_chars": config.max_skill_context_chars,
            },
            loaded_session_history_ids=loaded_history_ids,
            loaded_memory_ids=[],
            loaded_skill_ids=loaded_skill_ids,
            token_estimate=token_estimate,
            need_compaction=need_compaction,
        )

    def _load_relevant_skill_documents(
        self,
        new_user_message: str | None,
        *,
        config: ContextBuildConfig,
    ) -> list[SkillDocument]:
        if self._skill_repository is None or not new_user_message or config.max_loaded_skills <= 0:
            return []

        query = new_user_message.strip()
        if not query:
            return []

        scored_documents: list[tuple[int, int, SkillDocument]] = []
        for index, skill in enumerate(self._skill_repository.list_skills(status=AgentSkillStatus.ACTIVE, limit=100)):
            document = self._skill_repository.read_skill(skill.id)
            score = _skill_match_score(document, query)
            if score > 0:
                scored_documents.append((score, index, document))
        scored_documents.sort(key=lambda item: (-item[0], item[1]))
        matched_documents = [document for _, _, document in scored_documents[: config.max_loaded_skills]]
        for document in matched_documents:
            self._skill_repository.record_usage(document.skill.id, AgentSkillUsageEvent.USE)
        return matched_documents

    def _recent_context_messages(self, session_id: str, max_recent_messages: int) -> list[AgentMessage]:
        messages = self._conversation_service.list_messages(session_id, limit=500)
        context_messages = [
            message
            for message in messages
            if not message.exclude_from_context and message.compacted_by_summary_id is None
        ]
        if max_recent_messages <= 0:
            return context_messages
        return context_messages[-max_recent_messages:]


def _summary_message(summary: AgentContextSummary) -> dict[str, Any]:
    return {
        "role": "system",
        "content": f"Latest conversation summary:\n{summary.summary_text}",
        "metadata": {"summary_id": summary.id, "source": "agent_context_summary"},
    }


def _skill_message(document: SkillDocument, *, max_chars: int) -> dict[str, Any]:
    return {
        "role": "system",
        "content": _skill_context_content(document, max_chars),
        "metadata": {
            "source": "agent_skill",
            "skill_id": document.skill.id,
            "skill_name": document.skill.name,
            "version_hash": document.version_hash,
        },
    }


def _skill_context_content(document: SkillDocument, max_chars: int) -> str:
    content = document.content.strip()
    if max_chars > 0 and len(content) > max_chars:
        content = f"{content[:max_chars].rstrip()}\n...[skill context truncated]"
    return "\n".join(
        [
            "Relevant agent skill loaded for this run.",
            f"Skill: {document.skill.title} ({document.skill.name})",
            f"Category: {document.skill.category}",
            "Content:",
            content,
        ]
    )


def _skill_matches_query(document: SkillDocument, query: str) -> bool:
    return _skill_match_score(document, query) > 0


def _skill_match_score(document: SkillDocument, query: str) -> int:
    identity = "\n".join(
        [
            document.skill.name,
            document.skill.title,
            document.skill.category,
        ]
    ).lower()
    haystack = "\n".join(
        [
            document.skill.name,
            document.skill.title,
            document.skill.description,
            document.skill.category,
            document.content,
        ]
    ).lower()
    normalized_query = query.lower()
    if len(normalized_query) >= 8 and normalized_query in haystack:
        return 100

    score = _source_type_match_score(document, normalized_query)
    terms = _query_terms(normalized_query)
    for term in terms:
        if term in identity:
            score += 5
        elif term not in _GENERIC_SKILL_TERMS and term in haystack:
            score += 1
    return score


_GENERIC_SKILL_TERMS = {
    "2027",
    "2026",
    "java",
    "agent",
    "ai",
    "\u641c\u7d22",
    "\u83b7\u53d6",
    "\u62db\u8058",
    "\u5c97\u4f4d",
    "\u79cb\u62db",
    "\u6821\u62db",
}


def _source_type_match_score(document: SkillDocument, normalized_query: str) -> int:
    metadata = document.skill.metadata_json or {}
    source_types = set(_string_list(metadata.get("source_types")))
    if "xiaohongshu_note" in source_types and _query_mentions_xiaohongshu(normalized_query):
        return 50
    if source_types.intersection({"wechat_article", "wechat_account"}) and _query_mentions_wechat(normalized_query):
        return 50
    return 0


def _query_mentions_xiaohongshu(query: str) -> bool:
    return any(
        marker in query
        for marker in [
            "xiaohongshu",
            "xhslink",
            "\u5c0f\u7ea2\u4e66",
            "\u5c0f\u7d05\u66f8",
            "\u7ea2\u4e66",
            "\u7d05\u66f8",
        ]
    )


def _query_mentions_wechat(query: str) -> bool:
    return any(
        marker in query
        for marker in [
            "mp.weixin.qq.com",
            "weixin",
            "wechat",
            "\u5fae\u4fe1",
            "\u516c\u4f17\u53f7",
            "\u516c\u773e\u865f",
        ]
    )


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _query_terms(query: str) -> list[str]:
    import re

    return [term for term in re.findall(r"[a-z0-9_+-]+|[\u4e00-\u9fff]+", query) if len(term) >= 2]


def _message_to_llm_message(message: AgentMessage) -> dict[str, Any] | None:
    content = message.visible_content_text or message.content_text
    if not content:
        return None
    return {
        "role": _role_for_llm(message.role),
        "content": content,
        "metadata": {"message_id": message.id, "message_kind": str(getattr(message.message_kind, "value", message.message_kind))},
    }


def _role_for_llm(role: AgentMessageRole | str) -> str:
    role_value = str(getattr(role, "value", role))
    if role_value in {"user", "assistant", "system"}:
        return role_value
    return "assistant"
