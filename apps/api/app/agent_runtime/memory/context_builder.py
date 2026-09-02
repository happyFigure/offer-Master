from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from app.agent_runtime.memory.compaction import DEFAULT_KEEP_RECENT_TOKENS
from app.agent_runtime.memory.recall_policy import MemoryRecallResult, recall_relevant_memory_records
from app.agent_runtime.memory.skill_candidate_selector import SkillCandidate, select_skill_candidates
from app.agent_runtime.memory.skill_context_loader import SkillLoadRecord, load_selected_skill_context
from app.agent_runtime.memory.skill_resource_loader import SkillResourceLoadResult, load_skill_resource
from app.agent_runtime.memory.skill_repository import AgentSkillRepository, SkillDocument
from app.agent_runtime.memory.skill_section_parser import parse_skill_sections, select_skill_sections
from app.agent_runtime.memory.skill_summary_index import build_skill_summary_card
from app.agent_runtime.memory.token_budget import DEFAULT_RESERVE_TOKENS, estimate_message_tokens, estimate_tokens, should_compact
from app.agent_runtime.memory.transcript_hygiene import repair_tool_result_pairing
from app.agent_runtime.tool_permissions import AgentToolPermissionPolicy
from app.domains.agent_memory.models import AgentSkillStatus, AgentSkillUsageEvent
from app.domains.agent_memory.repository import AgentMemoryRepository
from app.domains.conversations.models import AgentContextSummary, AgentMessage, AgentMessageRole
from app.domains.conversations.service import ConversationService


@dataclass(frozen=True)
class ContextBuildConfig:
    context_window: int = 64000
    reserve_tokens: int = DEFAULT_RESERVE_TOKENS
    keep_recent_tokens: int = DEFAULT_KEEP_RECENT_TOKENS
    max_recent_messages: int = 50
    max_skill_candidates: int = 5
    max_loaded_skills: int = 3
    max_skill_context_chars: int = 4000
    max_skill_resources: int = 3
    max_skill_resource_chars: int = 4000
    max_loaded_memories: int = 3
    max_memory_context_chars: int = 2000


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
        memory_repository: AgentMemoryRepository | None = None,
        skill_repository: AgentSkillRepository | None = None,
    ) -> None:
        self._conversation_service = conversation_service
        self._memory_repository = memory_repository
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
        memory_recall_result = self._recall_relevant_memories(
            new_user_message,
            config=config,
        )
        skill_documents, skill_candidates, skill_load_records = self._load_relevant_skill_documents(
            new_user_message,
            config=config,
        )

        llm_messages: list[dict[str, Any]] = []
        if latest_summary is not None:
            llm_messages.append(_summary_message(latest_summary))

        if memory_recall_result.items:
            llm_messages.append(_memory_message(memory_recall_result))

        skill_load_record_by_id = {record.skill_id: record for record in skill_load_records}
        skill_resource_records = _load_referenced_skill_resources(
            skill_documents,
            skill_load_record_by_id,
            max_skill_context_chars=config.max_skill_context_chars,
            max_skill_resources=config.max_skill_resources,
            max_skill_resource_chars=config.max_skill_resource_chars,
        )
        for document in skill_documents:
            llm_messages.append(
                _skill_message(
                    document,
                    max_chars=config.max_skill_context_chars,
                    load_record=skill_load_record_by_id.get(document.skill.id),
                )
            )
        for document, resource_record in skill_resource_records:
            llm_messages.append(_skill_resource_message(document, resource_record))

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
        skill_tokens = sum(
            estimate_tokens(
                _skill_context_content(
                    document,
                    config.max_skill_context_chars,
                    selected_sections=skill_load_record_by_id.get(document.skill.id).selected_sections
                    if skill_load_record_by_id.get(document.skill.id) is not None
                    else (),
                )
            )
            for document in skill_documents
        )
        skill_resource_tokens = sum(resource_record.token_estimate for _, resource_record in skill_resource_records)
        memory_tokens = estimate_tokens(memory_recall_result.rendered_context)
        token_estimate = summary_tokens + memory_tokens + skill_tokens + skill_resource_tokens + estimate_message_tokens(context_messages) + estimate_tokens(new_user_message)
        threshold_tokens = config.context_window - config.reserve_tokens
        need_compaction = should_compact(
            context_tokens=token_estimate,
            context_window=config.context_window,
            reserve_tokens=config.reserve_tokens,
        )
        loaded_history_ids = [message.id for message in context_messages]
        loaded_memory_ids = [item.memory_id for item in memory_recall_result.items]
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
                "loaded_memory_ids": loaded_memory_ids,
                "memory_recall_trace": _memory_recall_trace_metadata(memory_recall_result),
                "loaded_skill_ids": loaded_skill_ids,
                "skill_candidate_selection": _skill_candidate_selection_metadata(skill_candidates),
                "skill_load_trace": _skill_load_trace_metadata(
                    skill_load_records,
                    candidate_count=len(skill_candidates),
                    max_loaded_skills=config.max_loaded_skills,
                ),
                "skill_resource_load_trace": _skill_resource_load_trace_metadata(skill_resource_records),
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
                "max_skill_candidates": config.max_skill_candidates,
                "max_loaded_skills": config.max_loaded_skills,
                "max_skill_context_chars": config.max_skill_context_chars,
                "max_skill_resources": config.max_skill_resources,
                "max_skill_resource_chars": config.max_skill_resource_chars,
                "max_loaded_memories": config.max_loaded_memories,
                "max_memory_context_chars": config.max_memory_context_chars,
            },
            loaded_session_history_ids=loaded_history_ids,
            loaded_memory_ids=loaded_memory_ids,
            loaded_skill_ids=loaded_skill_ids,
            token_estimate=token_estimate,
            need_compaction=need_compaction,
        )

    def _recall_relevant_memories(
        self,
        new_user_message: str | None,
        *,
        config: ContextBuildConfig,
    ) -> MemoryRecallResult:
        if self._memory_repository is None or not new_user_message or config.max_loaded_memories <= 0:
            return MemoryRecallResult(query=new_user_message or "", items=[], rendered_context="", truncated=False)
        memories = self._memory_repository.list_memories(limit=1000)
        return recall_relevant_memory_records(
            memories,
            query=new_user_message,
            limit=config.max_loaded_memories,
            max_chars=config.max_memory_context_chars,
        )

    def _load_relevant_skill_documents(
        self,
        new_user_message: str | None,
        *,
        config: ContextBuildConfig,
    ) -> tuple[list[SkillDocument], list[SkillCandidate], list[SkillLoadRecord]]:
        if self._skill_repository is None or not new_user_message or config.max_loaded_skills <= 0:
            return [], [], []

        query = new_user_message.strip()
        if not query:
            return [], [], []

        documents_by_skill_id: dict[str, SkillDocument] = {}
        cards = []
        for skill in self._skill_repository.list_skills(status=AgentSkillStatus.ACTIVE, limit=100):
            document = self._skill_repository.read_skill(skill.id)
            documents_by_skill_id[skill.id] = document
            cards.append(build_skill_summary_card(document))

        matched_candidates = select_skill_candidates(query, cards, limit=config.max_skill_candidates)
        section_headings = _skill_section_headings_for_query(query)
        load_result = load_selected_skill_context(
            matched_candidates,
            documents_by_skill_id,
            max_loaded_skills=config.max_loaded_skills,
            max_skill_context_chars=config.max_skill_context_chars,
            section_headings=section_headings,
        )
        matched_documents = list(load_result.loaded_documents)
        for document in matched_documents:
            self._skill_repository.record_usage(document.skill.id, AgentSkillUsageEvent.USE)
        return matched_documents, matched_candidates, list(load_result.load_records)

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


def _memory_message(result: MemoryRecallResult) -> dict[str, Any]:
    return {
        "role": "system",
        "content": "Relevant long-term memory for this turn:\n" + result.rendered_context,
        "metadata": {
            "source": "agent_memory",
            "loaded_memory_ids": [item.memory_id for item in result.items],
            "truncated": result.truncated,
        },
    }


def _memory_recall_trace_metadata(result: MemoryRecallResult) -> dict[str, Any]:
    return {
        "strategy": "deterministic_active_memory_recall",
        "query": result.query,
        "loaded_count": len(result.items),
        "truncated": result.truncated,
        "loaded_memories": [
            {
                "memory_id": item.memory_id,
                "scope": item.scope,
                "memory_type": item.memory_type,
                "importance": item.importance,
                "score": item.score,
            }
            for item in result.items
        ],
    }


def _skill_candidate_selection_metadata(candidates: list[SkillCandidate]) -> dict[str, Any]:
    return {
        "strategy": "skill_summary_card_keyword_rank",
        "candidates": [candidate.to_metadata() for candidate in candidates],
    }


def _skill_load_trace_metadata(
    records: list[SkillLoadRecord],
    *,
    candidate_count: int,
    max_loaded_skills: int,
) -> dict[str, Any]:
    return {
        "strategy": "load_selected_skill_bodies",
        "candidate_count": candidate_count,
        "loaded_count": len(records),
        "max_loaded_skills": max_loaded_skills,
        "loaded_skills": [record.to_metadata() for record in records],
    }


def _skill_resource_load_trace_metadata(
    records: list[tuple[SkillDocument, SkillResourceLoadResult]],
) -> dict[str, Any]:
    return {
        "strategy": "load_referenced_skill_resources",
        "loaded_count": len(records),
        "loaded_resources": [
            {
                "skill_id": document.skill.id,
                "skill_name": document.skill.name,
                **resource_record.to_metadata(),
            }
            for document, resource_record in records
        ],
    }


def _skill_message(document: SkillDocument, *, max_chars: int, load_record: SkillLoadRecord | None = None) -> dict[str, Any]:
    selected_sections = load_record.selected_sections if load_record is not None else ()
    return {
        "role": "system",
        "content": _skill_context_content(document, max_chars, selected_sections=selected_sections),
        "metadata": {
            "source": "agent_skill",
            "skill_id": document.skill.id,
            "skill_name": document.skill.name,
            "version_hash": document.version_hash,
            "load_layer": load_record.load_layer if load_record is not None else "body",
            "selected_sections": list(selected_sections),
        },
    }


def _skill_resource_message(document: SkillDocument, resource_record: SkillResourceLoadResult) -> dict[str, Any]:
    return {
        "role": "system",
        "content": "\n".join(
            [
                "Referenced agent skill resource loaded for this run.",
                f"Skill: {document.skill.title} ({document.skill.name})",
                f"Resource: {resource_record.resource_path}",
                "Content:",
                resource_record.content,
            ]
        ),
        "metadata": {
            "source": "agent_skill_resource",
            "skill_id": document.skill.id,
            "skill_name": document.skill.name,
            "resource_path": resource_record.resource_path,
            "load_layer": resource_record.load_layer,
            "truncated": resource_record.truncated,
        },
    }


def _skill_context_content(document: SkillDocument, max_chars: int, *, selected_sections: tuple[str, ...] = ()) -> str:
    content = _skill_loaded_content(document, max_chars, selected_sections=selected_sections)
    return "\n".join(
        [
            "Relevant agent skill loaded for this run.",
            f"Skill: {document.skill.title} ({document.skill.name})",
            f"Category: {document.skill.category}",
            "Content:",
            content,
        ]
    )


def _skill_loaded_content(document: SkillDocument, max_chars: int, *, selected_sections: tuple[str, ...] = ()) -> str:
    content = document.content.strip()
    if selected_sections:
        selection = select_skill_sections(parse_skill_sections(content), selected_sections, max_chars=max_chars)
        if selection.sections:
            content = selection.content
            if selection.truncated:
                content = f"{content.rstrip()}\n...[skill section context truncated]"
        elif max_chars > 0 and len(content) > max_chars:
            content = f"{content[:max_chars].rstrip()}\n...[skill context truncated]"
    elif max_chars > 0 and len(content) > max_chars:
        content = f"{content[:max_chars].rstrip()}\n...[skill context truncated]"
    return content


_SKILL_RESOURCE_PATH_PATTERN = re.compile(
    r"(?P<path>(?:references|scripts|assets|agents)[/\\][A-Za-z0-9._/@+=-]+)"
)


def _load_referenced_skill_resources(
    documents: list[SkillDocument],
    skill_load_record_by_id: dict[str, SkillLoadRecord],
    *,
    max_skill_context_chars: int,
    max_skill_resources: int,
    max_skill_resource_chars: int,
) -> list[tuple[SkillDocument, SkillResourceLoadResult]]:
    if max_skill_resources <= 0:
        return []

    loaded_resources: list[tuple[SkillDocument, SkillResourceLoadResult]] = []
    seen: set[tuple[str, str]] = set()
    for document in documents:
        record = skill_load_record_by_id.get(document.skill.id)
        loaded_content = _skill_loaded_content(
            document,
            max_skill_context_chars,
            selected_sections=record.selected_sections if record is not None else (),
        )
        for resource_path in _extract_skill_resource_paths(loaded_content):
            key = (document.skill.id, resource_path)
            if key in seen:
                continue
            seen.add(key)
            try:
                resource_record = load_skill_resource(
                    document,
                    resource_path,
                    max_chars=max_skill_resource_chars,
                )
            except (FileNotFoundError, PermissionError, ValueError):
                continue
            loaded_resources.append((document, resource_record))
            if len(loaded_resources) >= max_skill_resources:
                return loaded_resources
    return loaded_resources


def _extract_skill_resource_paths(content: str) -> list[str]:
    paths: list[str] = []
    for match in _SKILL_RESOURCE_PATH_PATTERN.finditer(content):
        path = match.group("path").replace("\\", "/").strip(".,;:，。；：）)]}")
        if path and path not in paths:
            paths.append(path)
    return paths


def _skill_section_headings_for_query(query: str) -> tuple[str, ...]:
    normalized = query.casefold()
    if any(marker in normalized for marker in ["输出格式", "输出", "格式", "返回格式", "结果格式"]):
        return ("输出", "错误处理")
    if any(marker in normalized for marker in ["权限", "确认", "风险", "边界", "能不能用工具"]):
        return ("工具边界", "用户确认点", "错误处理")
    return ()


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
    "application",
    "apply",
    "entry",
    "for",
    "from",
    "id",
    "job",
    "job_id",
    "lead_id",
    "open",
    "page",
    "the",
    "this",
    "to",
    "url",
    "with",
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
