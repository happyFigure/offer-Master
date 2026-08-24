from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any
from uuid import uuid4

from app.agent_runtime.reflection.schemas import (
    CapabilityResultEvaluationSpec,
    campus_recruiting_web_search_result_evaluation_spec,
    result_evaluation_spec_for_capability,
)
from app.agent_runtime.web_search_query import normalize_external_web_search_query
from app.mcp_gateway.tool_policy import MCPToolPolicy


class AgentToolRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class AgentToolCandidateProfile:
    categories: frozenset[str] = field(default_factory=frozenset)
    keywords: frozenset[str] = field(default_factory=frozenset)
    examples: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "categories", frozenset(str(item).strip() for item in self.categories if str(item).strip()))
        object.__setattr__(self, "keywords", frozenset(str(item).strip() for item in self.keywords if str(item).strip()))
        object.__setattr__(self, "examples", tuple(str(item).strip() for item in self.examples if str(item).strip()))


@dataclass(frozen=True)
class AgentToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    handler: Callable[..., Any] | None = None
    risk_level: AgentToolRiskLevel = AgentToolRiskLevel.LOW
    requires_confirmation: bool = False
    allowed_source_types: frozenset[str] = field(default_factory=frozenset)
    enabled: bool = True
    result_evaluation: CapabilityResultEvaluationSpec | None = None
    candidate_profile: AgentToolCandidateProfile | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Agent tool name is required")
        if not self.description.strip():
            raise ValueError(f"Agent tool description is required: {self.name}")
        object.__setattr__(self, "allowed_source_types", frozenset(self.allowed_source_types))
        if self.result_evaluation is None:
            object.__setattr__(self, "result_evaluation", result_evaluation_spec_for_capability(self.name))


class AgentToolRegistry:
    def __init__(self, definitions: Iterable[AgentToolDefinition] | None = None) -> None:
        self._definitions: dict[str, AgentToolDefinition] = {}
        for definition in definitions or ():
            self.register(definition)

    def register(self, definition: AgentToolDefinition) -> AgentToolDefinition:
        if definition.name in self._definitions:
            raise ValueError(f"Agent tool already registered: {definition.name}")
        self._definitions[definition.name] = definition
        return definition

    def register_many(self, definitions: Iterable[AgentToolDefinition]) -> None:
        for definition in definitions:
            self.register(definition)

    def get(self, name: str) -> AgentToolDefinition | None:
        definition = self._definitions.get(name)
        if definition is None or not definition.enabled:
            return None
        return definition

    def list_definitions(self) -> list[AgentToolDefinition]:
        return sorted(
            (definition for definition in self._definitions.values() if definition.enabled),
            key=lambda definition: definition.name,
        )

    def registered_tool_names(self) -> list[str]:
        return [definition.name for definition in self.list_definitions()]


APPLICATION_FIND_APPLY_ENTRY_TOOL = "applications.find_apply_entry"
EXTERNAL_WEB_SEARCH_TOOL = "external.web_search"
LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL = "local.company_database_overview"
LOCAL_JOB_SOURCE_OVERVIEW_TOOL = "local.job_source_overview"
OFFERIO_COMPANY_JOBS_TOOL = "offerio.sync_company_jobs"
OFFERIO_COMPANY_JOBS_SOURCE_NAME = "OfferIO 公司聚合岗位库"
OFFERIO_COMPANY_JOBS_LEGACY_SOURCE_NAMES = (OFFERIO_COMPANY_JOBS_SOURCE_NAME, "OfferIO company jobs")
OFFERIO_COMPANY_JOBS_ENTRY_URL = "https://offerio.work/api/recruitment/job-companies?jobType=校招&page=1&pageSize=50"


def create_default_agent_tool_registry(
    *,
    content_source_client: Any | None = None,
    offerio_provider_factory: Callable[[], Any] | None = None,
    external_task_dispatcher: Callable[[Any, str], dict[str, Any]] | None = None,
    external_web_search_executor: Callable[[str, int], dict[str, Any]] | None = None,
) -> AgentToolRegistry:
    registry = AgentToolRegistry()
    registry.register_many(create_application_agent_tool_definitions(external_task_dispatcher=external_task_dispatcher))
    registry.register_many(create_external_web_search_agent_tool_definitions(external_web_search_executor=external_web_search_executor))
    registry.register_many(create_local_company_database_agent_tool_definitions())
    registry.register_many(create_local_job_source_agent_tool_definitions(offerio_provider_factory=offerio_provider_factory))
    registry.register_many(_memory_tool_definitions())
    registry.register_many(create_job_source_agent_tool_definitions(offerio_provider_factory=offerio_provider_factory))
    registry.register_many(create_content_source_agent_tool_definitions(content_source_client))
    return registry


def create_application_agent_tool_definitions(
    *,
    external_task_dispatcher: Callable[[Any, str], dict[str, Any]] | None = None,
) -> list[AgentToolDefinition]:
    return [
        AgentToolDefinition(
            name=APPLICATION_FIND_APPLY_ENTRY_TOOL,
            description=(
                "Create a safe external-agent task to find and open a job application entry. "
                "The task must stop before final submission."
            ),
            input_schema={
                "type": "object",
                "required": ["job_id"],
                "properties": {
                    "task_id": {"type": ["string", "null"]},
                    "trace_id": {"type": ["string", "null"]},
                    "job_id": {"type": "string", "description": "Local Job or JobLead id."},
                    "company_name": {"type": ["string", "null"]},
                    "title": {"type": ["string", "null"]},
                    "source_url": {"type": ["string", "null"]},
                    "apply_url_candidate": {"type": ["string", "null"]},
                    "jd_summary": {"type": ["string", "null"]},
                    "profile_id": {"type": ["string", "null"], "default": "default"},
                    "resume_version_id": {"type": ["string", "null"], "default": "default"},
                },
                "additionalProperties": False,
            },
            output_schema={"type": "object", "required": ["tool_name", "ok", "result"]},
            handler=lambda session, **arguments: _queue_find_apply_entry_task(
                session,
                external_task_dispatcher=external_task_dispatcher,
                **arguments,
            ),
            risk_level=AgentToolRiskLevel.MEDIUM,
            allowed_source_types=frozenset({"agent_chat", "application", "job_discovery", "job_lead"}),
        )
    ]


def create_external_web_search_agent_tool_definitions(
    *,
    external_web_search_executor: Callable[[str, int], dict[str, Any]] | None = None,
) -> list[AgentToolDefinition]:
    return [
        AgentToolDefinition(
            name=EXTERNAL_WEB_SEARCH_TOOL,
            description="Search the public web through the configured external agent executor.",
            input_schema={
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                },
                "additionalProperties": False,
            },
            output_schema={"type": "object", "required": ["tool_name", "ok", "result"]},
            handler=lambda session, **arguments: _run_external_web_search(
                session,
                external_web_search_executor=external_web_search_executor,
                **arguments,
            ),
            risk_level=AgentToolRiskLevel.LOW,
            allowed_source_types=frozenset({"agent_chat", "web_search"}),
            result_evaluation=campus_recruiting_web_search_result_evaluation_spec(),
            candidate_profile=AgentToolCandidateProfile(
                categories=frozenset({"public_web_information", "realtime_public_information"}),
                keywords=frozenset(
                    {
                        "查一下",
                        "搜一下",
                        "搜索",
                        "今天",
                        "现在",
                        "最新",
                        "最近",
                        "比赛",
                        "新闻",
                        "官网",
                        "是什么",
                        "做什么",
                        "主要业务",
                    }
                ),
                examples=("给我查一下梅西今天的比赛", "Canonical Ltd. 是做什么的？主要业务是什么？"),
            ),
        )
    ]


def create_local_company_database_agent_tool_definitions() -> list[AgentToolDefinition]:
    return [
        AgentToolDefinition(
            name=LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL,
            description="Read a safe overview of local company, job, lead, and recruiting-signal counts.",
            input_schema={
                "type": "object",
                "properties": {
                    "sample_limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 10,
                        "description": "Maximum sample company names to include for each local source bucket.",
                    }
                },
                "additionalProperties": False,
            },
            output_schema={"type": "object", "required": ["tool_name", "ok", "result"]},
            handler=lambda session, **arguments: _local_company_database_overview(session, **arguments),
            risk_level=AgentToolRiskLevel.LOW,
            requires_confirmation=False,
            allowed_source_types=frozenset({"agent_chat", "job_discovery"}),
            candidate_profile=AgentToolCandidateProfile(
                categories=frozenset({"local_company_data", "local_database"}),
                keywords=frozenset({"数据库", "本地", "公司库", "企业库", "我的公司", "有哪些公司", "多少公司"}),
                examples=("数据库里有哪些公司，给我20个", "我的数据库里现在有多少企业？"),
            ),
        )
    ]


def create_local_job_source_agent_tool_definitions(
    *,
    offerio_provider_factory: Callable[[], Any] | None = None,
) -> list[AgentToolDefinition]:
    from app.domains.jobs.providers.offerio import OfferIORecruitmentProvider

    provider_factory = offerio_provider_factory or OfferIORecruitmentProvider
    return [
        AgentToolDefinition(
            name=LOCAL_JOB_SOURCE_OVERVIEW_TOOL,
            description="Read a safe overview of local job sources and the default external OfferIO job-board source totals.",
            input_schema={
                "type": "object",
                "properties": {
                    "sample_limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 10,
                    },
                    "include_external_job_board": {
                        "type": "boolean",
                        "default": True,
                    },
                },
                "additionalProperties": False,
            },
            output_schema={"type": "object", "required": ["tool_name", "ok", "result"]},
            handler=lambda session, **arguments: _local_job_source_overview(
                session,
                offerio_provider_factory=provider_factory,
                **arguments,
            ),
            risk_level=AgentToolRiskLevel.LOW,
            requires_confirmation=False,
            allowed_source_types=frozenset({"agent_chat", "job_discovery"}),
            candidate_profile=AgentToolCandidateProfile(
                categories=frozenset({"local_job_source_data", "local_database"}),
                keywords=frozenset({"岗位来源", "来源库", "岗位展览", "校招来源", "岗位线索", "开放岗位来源库"}),
                examples=("岗位来源库现在有多少条，给我20个", "岗位展览里有哪些来源？"),
            ),
        )
    ]


def create_job_source_agent_tool_definitions(
    *,
    offerio_provider_factory: Callable[[], Any] | None = None,
) -> list[AgentToolDefinition]:
    from app.domains.jobs.providers.offerio import OfferIORecruitmentProvider

    provider_factory = offerio_provider_factory or OfferIORecruitmentProvider
    return [
        AgentToolDefinition(
            name=OFFERIO_COMPANY_JOBS_TOOL,
            description="Sync OfferIO company aggregated campus recruiting jobs into local job leads.",
            input_schema={
                "type": "object",
                "properties": {
                    "source_id": {"type": ["string", "null"], "description": "Optional existing official_api JobSource id."},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5000,
                        "default": 1000,
                        "description": "Maximum total companies to sync across OfferIO pages.",
                    },
                },
            },
            output_schema={"type": "object", "required": ["tool_name", "ok", "result"]},
            handler=lambda session, **arguments: _sync_offerio_company_jobs(
                session,
                provider_factory=provider_factory,
                **arguments,
            ),
            risk_level=AgentToolRiskLevel.MEDIUM,
            allowed_source_types=frozenset({"agent_chat", "official_api", "job_discovery"}),
        )
    ]


def create_content_source_agent_tool_definitions(client: Any | None = None) -> list[AgentToolDefinition]:
    from app.mcp_gateway.content_source_client import ContentSourceMCPClient

    content_client = client or ContentSourceMCPClient()
    return [
        AgentToolDefinition(
            name="weixin-articles-mcp.read_article",
            description="Read one public WeChat official-account article URL and return extracted text/media blocks.",
            input_schema={
                "type": "object",
                "required": ["url"],
                "properties": {"url": {"type": "string", "description": "Public mp.weixin.qq.com article URL."}},
            },
            output_schema={"type": "object", "required": ["tool_name", "ok"]},
            handler=lambda _session, **arguments: content_client.read_weixin_article(url=str(arguments.get("url") or "")),
            allowed_source_types=frozenset({"agent_chat", "wechat_article", "wechat_account"}),
            candidate_profile=AgentToolCandidateProfile(
                categories=frozenset({"wechat_article_read", "content_source_read"}),
                keywords=frozenset({"微信公众号", "微信文章", "mp.weixin.qq.com", "公众号文章", "读取文章"}),
                examples=("请读取这个微信公众号招聘文章 https://mp.weixin.qq.com/s/example",),
            ),
        ),
        AgentToolDefinition(
            name="xiaohongshu-mcp.search_feeds",
            description="Search Xiaohongshu feeds for recruiting-related notes by keyword through MCP Gateway.",
            input_schema={
                "type": "object",
                "required": ["keyword"],
                "properties": {
                    "keyword": {"type": "string"},
                    "filters": {"type": ["object", "null"], "additionalProperties": True},
                },
            },
            output_schema={"type": "object", "required": ["tool_name", "ok"]},
            handler=lambda _session, **arguments: content_client.search_xiaohongshu_feeds(
                keyword=str(arguments.get("keyword") or ""),
                filters=arguments.get("filters") if isinstance(arguments.get("filters"), dict) else None,
            ),
            allowed_source_types=frozenset({"agent_chat", "xiaohongshu_note", "mcp_visible_page"}),
            candidate_profile=AgentToolCandidateProfile(
                categories=frozenset({"xiaohongshu_content_search", "content_source_search"}),
                keywords=frozenset({"小红书", "红书", "xhslink", "xiaohongshu", "搜索笔记", "搜笔记"}),
                examples=("请在小红书搜索 2027 秋招 Java 岗位",),
            ),
        ),
        AgentToolDefinition(
            name="xiaohongshu-mcp.get_feed_detail",
            description="Read one Xiaohongshu feed detail through MCP Gateway using feed_id and xsec_token.",
            input_schema={
                "type": "object",
                "required": ["feed_id", "xsec_token"],
                "properties": {
                    "feed_id": {"type": "string"},
                    "xsec_token": {"type": "string"},
                    "include_comments": {"type": "boolean"},
                    "comment_limit": {"type": "integer", "minimum": 0, "maximum": 100},
                },
                "additionalProperties": True,
            },
            output_schema={"type": "object", "required": ["tool_name", "ok"]},
            handler=lambda _session, **arguments: content_client.get_xiaohongshu_feed_detail(**arguments),
            allowed_source_types=frozenset({"agent_chat", "xiaohongshu_note", "mcp_visible_page"}),
            candidate_profile=AgentToolCandidateProfile(
                categories=frozenset({"xiaohongshu_content_detail", "content_source_read"}),
                keywords=frozenset({"小红书详情", "feed_id", "xsec_token", "笔记详情"}),
                examples=("小红书 feed_id=abc123 xsec_token=token456 读取详情",),
            ),
        ),
    ]


def create_mcp_agent_tool_definitions(client: Any, *, allowed_tool_names: Iterable[str]) -> list[AgentToolDefinition]:
    policy = MCPToolPolicy.from_allowlist(allowed_tool_names)
    definitions: list[AgentToolDefinition] = []
    for tool_name in policy.allowed_tool_names():
        definitions.append(
            AgentToolDefinition(
                name=f"mcp.{tool_name}",
                description=f"Call MCP Gateway tool: {tool_name}.",
                input_schema={"type": "object", "additionalProperties": True},
                output_schema={"type": "object", "required": ["tool_name", "ok"]},
                handler=_mcp_handler(client, tool_name),
                risk_level=_mcp_risk_level(policy, tool_name),
                requires_confirmation=policy.requires_confirmation(tool_name),
                allowed_source_types=frozenset({"agent_chat", "mcp_visible_page", "application"}),
            )
        )
    return definitions


def _memory_tool_definitions() -> list[AgentToolDefinition]:
    from app.agent_runtime.memory.memory_tools import memory_get, memory_search, sessions_history, sessions_search

    return [
        AgentToolDefinition(
            name="sessions_search",
            description="Search prior agent session transcript messages and context summaries.",
            input_schema={
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                },
            },
            output_schema={"type": "object", "required": ["corpus", "query", "items"]},
            handler=sessions_search,
            allowed_source_types=frozenset({"agent_chat", "agent_session", "history_recall"}),
        ),
        AgentToolDefinition(
            name="sessions_history",
            description="Read a bounded message window around a prior session message.",
            input_schema={
                "type": "object",
                "required": ["session_key"],
                "properties": {
                    "session_key": {"type": "string"},
                    "around_message_id": {"type": ["string", "null"]},
                    "window_before": {"type": "integer", "minimum": 0, "maximum": 50},
                    "window_after": {"type": "integer", "minimum": 0, "maximum": 50},
                },
            },
            output_schema={"type": "object", "required": ["session_id", "messages"]},
            handler=sessions_history,
            allowed_source_types=frozenset({"agent_chat", "agent_session", "history_recall"}),
        ),
        AgentToolDefinition(
            name="memory_search",
            description="Search long-term semantic memories and skill records only.",
            input_schema={
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string"},
                    "corpus": {"type": ["string", "null"]},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                },
            },
            output_schema={"type": "object", "required": ["corpus", "query", "items"]},
            handler=memory_search,
            allowed_source_types=frozenset({"agent_chat", "long_term_memory", "skill_recall"}),
        ),
        AgentToolDefinition(
            name="memory_get",
            description="Read one precise long-term memory or skill record by id.",
            input_schema={
                "type": "object",
                "required": ["memory_id"],
                "properties": {"memory_id": {"type": "string"}},
            },
            output_schema={"type": "object", "required": ["memory_id", "found"]},
            handler=memory_get,
            allowed_source_types=frozenset({"agent_chat", "long_term_memory", "skill_recall"}),
        ),
    ]


def _queue_find_apply_entry_task(
    session: Any,
    *,
    external_task_dispatcher: Callable[[Any, str], dict[str, Any]] | None = None,
    **arguments: Any,
) -> dict[str, Any]:
    from app.agent_runtime.external_tasks.repository import SqlAlchemyExternalAgentTaskRepository
    from app.agent_runtime.external_tasks.schemas import (
        ExternalTaskCandidateProfileRef,
        ExternalTaskJobContext,
        FindApplyEntryTaskEnvelope,
    )
    from app.agent_runtime.external_tasks.service import ExternalAgentTaskService

    job_context = _resolve_apply_entry_job_context(session, arguments)
    task_id = _non_empty_str(arguments.get("task_id")) or f"external-task-{uuid4()}"
    trace_id = _non_empty_str(arguments.get("trace_id")) or f"trace-{uuid4()}"
    envelope = FindApplyEntryTaskEnvelope(
        task_id=task_id,
        trace_id=trace_id,
        objective="Find and open the official application page for this job. Stop before final submit.",
        job=job_context,
        candidate_profile_ref=ExternalTaskCandidateProfileRef(
            profile_id=_non_empty_str(arguments.get("profile_id")) or "default",
            resume_version_id=_non_empty_str(arguments.get("resume_version_id")) or "default",
        ),
    )
    task = ExternalAgentTaskService(
        SqlAlchemyExternalAgentTaskRepository(session)
    ).create_find_apply_entry_task(envelope)
    result_payload = {
        "task_id": task.task_id,
        "task_type": _value(task.task_type),
        "status": _value(task.status),
        "trace_id": trace_id,
        "context_pack_hash": task.context_pack_hash,
        "task_envelope": task.input_payload,
        "next_action": "external_agent_dispatch",
    }
    if external_task_dispatcher is not None:
        dispatch_result = external_task_dispatcher(session, task.task_id)
        result_payload["dispatch"] = dispatch_result
        result_payload["next_action"] = (
            "external_agent_completed"
            if dispatch_result.get("ok") and dispatch_result.get("status") == "succeeded"
            else dispatch_result.get("next_action") or "external_agent_dispatch_failed"
        )
        result_payload["status"] = str(dispatch_result.get("status") or result_payload["status"])
    result_payload["result_envelope"] = _apply_entry_tool_result_envelope(result_payload)
    return {
        "tool_name": APPLICATION_FIND_APPLY_ENTRY_TOOL,
        "ok": True,
        "result": result_payload,
    }


def _apply_entry_tool_result_envelope(result_payload: dict[str, Any]) -> dict[str, Any] | None:
    dispatch = result_payload.get("dispatch") if isinstance(result_payload.get("dispatch"), dict) else {}
    if isinstance(dispatch.get("result_envelope"), dict):
        return dispatch["result_envelope"]

    from app.agent_runtime.routing.result_envelope import build_result_envelope

    envelope = build_result_envelope(
        capability=APPLICATION_FIND_APPLY_ENTRY_TOOL,
        status=str(result_payload.get("status") or "queued"),
        risk_level="medium",
        result_payload={
            "tool_name": APPLICATION_FIND_APPLY_ENTRY_TOOL,
            "ok": True,
            "result": result_payload,
        },
    )
    return envelope.to_dict() if envelope is not None else None


def _run_external_web_search(
    _session: Any,
    *,
    external_web_search_executor: Callable[[str, int], dict[str, Any]] | None = None,
    query: str,
    max_results: int | str | None = 5,
) -> dict[str, Any]:
    original_query = _required_str(query, "query")
    search_query = _normalize_external_web_search_query(original_query)
    result_limit = _bounded_int(max_results, default=5, minimum=1, maximum=10)
    if external_web_search_executor is None:
        return {
            "tool_name": EXTERNAL_WEB_SEARCH_TOOL,
            "ok": False,
            "error": "EXTERNAL_WEB_SEARCH_NOT_CONFIGURED",
            "result": {
                "query": search_query,
                "original_query": original_query,
                "max_results": result_limit,
                "message": "External web search executor is not configured.",
            },
        }
    try:
        search_result = external_web_search_executor(search_query, result_limit)
    except Exception as exc:
        return {
            "tool_name": EXTERNAL_WEB_SEARCH_TOOL,
            "ok": False,
            "error": str(exc),
            "result": {"query": search_query, "original_query": original_query, "max_results": result_limit},
        }
    return {
        "tool_name": EXTERNAL_WEB_SEARCH_TOOL,
        "ok": True,
        "result": {
            **dict(search_result),
            "query": search_query,
            "original_query": original_query,
            "max_results": result_limit,
        },
    }


def _normalize_external_web_search_query(query: str) -> str:
    return normalize_external_web_search_query(query)


def _resolve_apply_entry_job_context(session: Any, arguments: dict[str, Any]) -> Any:
    from app.agent_runtime.external_tasks.schemas import ExternalTaskJobContext

    job_id = _required_str(arguments.get("job_id"), "job_id")
    resolved = _load_apply_entry_job_context_from_db(session, job_id)
    company_name = _non_empty_str(arguments.get("company_name")) or resolved.get("company_name")
    title = _non_empty_str(arguments.get("title")) or resolved.get("title")
    source_url = _non_empty_str(arguments.get("source_url")) or resolved.get("source_url")
    apply_url_candidate = _non_empty_str(arguments.get("apply_url_candidate")) or resolved.get("apply_url_candidate")
    jd_summary = _non_empty_str(arguments.get("jd_summary")) or resolved.get("jd_summary")
    return ExternalTaskJobContext(
        job_id=job_id,
        company_name=_required_str(company_name, "company_name"),
        title=_required_str(title, "title"),
        source_url=source_url,
        apply_url_candidate=apply_url_candidate,
        jd_summary=jd_summary,
    )


def _load_apply_entry_job_context_from_db(session: Any, job_id: str) -> dict[str, Any]:
    from app.domains.jobs.models import Job, JobLead

    lead = session.get(JobLead, job_id)
    if lead is not None:
        return {
            "company_name": lead.company_name,
            "title": lead.title,
            "source_url": lead.source_url,
            "apply_url_candidate": lead.apply_url or lead.verified_url,
            "jd_summary": lead.jd_text,
        }
    job = session.get(Job, job_id)
    if job is not None:
        return {
            "company_name": job.company.name,
            "title": job.title,
            "source_url": job.source_url,
            "apply_url_candidate": None,
            "jd_summary": job.jd_text,
        }
    return {}


def _sync_offerio_company_jobs(
    session: Any,
    *,
    provider_factory: Callable[[], Any],
    source_id: str | None = None,
    limit: int | str | None = 1000,
) -> dict[str, Any]:
    from app.agent_runtime.workflows.job_discovery import OfficialApiSyncCommand, run_offerio_official_api_source_sync
    from app.domains.jobs.models import SourceSyncRunStatus
    from app.domains.jobs.repository import (
        ArticleCandidateRepository,
        JobLeadRepository,
        JobSourceRepository,
        RawJobLeadRepository,
        RecruitingSignalRepository,
        SourceSyncRunRepository,
    )
    from app.domains.jobs.service import JobLeadService

    total_limit = _bounded_int(limit, default=1000, minimum=1, maximum=5000)
    lead_service = JobLeadService(
        sources=JobSourceRepository(session),
        sync_runs=SourceSyncRunRepository(session),
        raw_leads=RawJobLeadRepository(session),
        leads=JobLeadRepository(session),
        article_candidates=ArticleCandidateRepository(session),
        recruiting_signals=RecruitingSignalRepository(session),
    )
    source = (
        lead_service.get_source(source_id)
        if source_id
        else _get_or_create_offerio_company_jobs_source(session, lead_service, _offerio_company_jobs_page_size(total_limit))
    )
    sync_result = run_offerio_official_api_source_sync(
        OfficialApiSyncCommand(source_id=source.id, limit=total_limit),
        lead_service=lead_service,
        provider=provider_factory(),
    )
    ok = _value(sync_result.sync_run.status) != SourceSyncRunStatus.FAILED.value
    error = sync_result.error or sync_result.sync_run.error
    return {
        "tool_name": OFFERIO_COMPANY_JOBS_TOOL,
        "ok": ok,
        "error": None if ok else error,
        "result": {
            "source_id": source.id,
            "source_name": source.name,
            "sync_run_id": sync_result.sync_run.id,
            "status": _value(sync_result.sync_run.status),
            "fetched_count": sync_result.fetched_count,
            "extracted_count": sync_result.extracted_count,
            "failed_count": sync_result.failed_count,
            "error": error,
            "raw_lead_ids": [capture.raw_lead.id for capture in sync_result.raw_captures],
            "lead_ids": [lead.id for lead in sync_result.leads],
            "lead_summaries": [
                {
                    "id": lead.id,
                    "company_name": lead.company_name,
                    "title": lead.title,
                    "job_direction": lead.job_direction,
                    "verification_status": _value(lead.verification_status),
                }
                for lead in sync_result.leads[:10]
            ],
        },
    }


def _local_company_database_overview(
    session: Any,
    *,
    sample_limit: int | str | None = 10,
) -> dict[str, Any]:
    if session is None:
        return {
            "tool_name": LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL,
            "ok": False,
            "error": "DATABASE_SESSION_UNAVAILABLE",
            "result": {"message": "Database session is unavailable."},
        }

    from sqlalchemy import func, select

    from app.domains.jobs.models import Company, Job, JobLead, RecruitingSignal

    limit = _bounded_int(sample_limit, default=10, minimum=1, maximum=50)
    company_count = int(session.scalar(select(func.count(Company.id))) or 0)
    job_count = int(session.scalar(select(func.count(Job.id))) or 0)
    job_lead_count = int(session.scalar(select(func.count(JobLead.id))) or 0)
    job_lead_company_count = int(session.scalar(select(func.count(func.distinct(JobLead.company_name)))) or 0)
    recruiting_signal_count = int(session.scalar(select(func.count(RecruitingSignal.id))) or 0)
    recruiting_signal_company_count = int(session.scalar(select(func.count(func.distinct(RecruitingSignal.company_name)))) or 0)

    return {
        "tool_name": LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL,
        "ok": True,
        "result": {
            "company_count": company_count,
            "job_count": job_count,
            "job_lead_count": job_lead_count,
            "job_lead_company_count": job_lead_company_count,
            "recruiting_signal_count": recruiting_signal_count,
            "recruiting_signal_company_count": recruiting_signal_company_count,
            "sample_companies": _sample_distinct_strings(session, select(Company.name).order_by(Company.name.asc()).limit(limit)),
            "sample_lead_companies": _sample_distinct_strings(
                session,
                select(JobLead.company_name).distinct().order_by(JobLead.company_name.asc()).limit(limit),
            ),
            "sample_signal_companies": _sample_distinct_strings(
                session,
                select(RecruitingSignal.company_name).distinct().order_by(RecruitingSignal.company_name.asc()).limit(limit),
            ),
            "company_rows": _local_company_overview_rows(session, Company, Job, JobLead, RecruitingSignal, limit),
        },
    }


def _local_company_overview_rows(session: Any, Company: Any, Job: Any, JobLead: Any, RecruitingSignal: Any, limit: int) -> list[dict[str, str]]:
    from sqlalchemy import func, select

    company_map: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    def ensure_company(name: Any, tier_rank: int) -> dict[str, Any] | None:
        company_name = _non_empty_str(name)
        if not company_name:
            return None
        key = company_name.casefold()
        if key not in company_map:
            company_map[key] = {
                "company_name": company_name,
                "tier_rank": tier_rank,
                "job_count": 0,
                "lead_count": 0,
                "signal_count": 0,
                "has_profile": False,
            }
            order.append(key)
        company_map[key]["tier_rank"] = min(int(company_map[key]["tier_rank"]), tier_rank)
        return company_map[key]

    formal_companies = session.execute(
        select(Company.name, func.count(Job.id))
        .outerjoin(Job, Job.company_id == Company.id)
        .group_by(Company.id, Company.name)
        .order_by(Company.name.asc())
        .limit(limit)
    ).all()
    for name, job_count in formal_companies:
        item = ensure_company(name, 0)
        if item is not None:
            item["has_profile"] = True
            item["job_count"] += int(job_count or 0)

    lead_companies = session.execute(
        select(JobLead.company_name, func.count(JobLead.id))
        .group_by(JobLead.company_name)
        .order_by(JobLead.company_name.asc())
        .limit(limit)
    ).all()
    for name, lead_count in lead_companies:
        item = ensure_company(name, 1)
        if item is not None:
            item["lead_count"] += int(lead_count or 0)

    signal_companies = session.execute(
        select(RecruitingSignal.company_name, func.count(RecruitingSignal.id))
        .group_by(RecruitingSignal.company_name)
        .order_by(RecruitingSignal.company_name.asc())
        .limit(limit)
    ).all()
    for name, signal_count in signal_companies:
        item = ensure_company(name, 2)
        if item is not None:
            item["signal_count"] += int(signal_count or 0)

    return [_company_overview_row(company_map[key]) for key in sorted(order, key=lambda item: (company_map[item]["tier_rank"], company_map[item]["company_name"]))][:limit]


def _company_overview_row(item: dict[str, Any]) -> dict[str, str]:
    tier = ["正式企业", "岗位线索企业", "校招来源企业"][int(item["tier_rank"])]
    known_info: list[str] = []
    quantities: list[str] = []
    if item["has_profile"]:
        known_info.append("企业档案")
    if item["job_count"]:
        known_info.append("正式岗位")
        quantities.append(f"{int(item['job_count'])} 条岗位")
    if item["lead_count"]:
        known_info.append("岗位线索")
        quantities.append(f"{int(item['lead_count'])} 条线索")
    if item["signal_count"]:
        known_info.append("校招来源")
        quantities.append(f"{int(item['signal_count'])} 条来源")

    if item["job_count"]:
        status = "可用于推荐"
    elif item["has_profile"]:
        status = "可补充岗位后用于推荐"
    elif item["lead_count"]:
        status = "待补全企业档案"
    else:
        status = "可继续验证"

    return {
        "tier": tier,
        "company_name": str(item["company_name"]),
        "known_info": "、".join(known_info) or "待补充",
        "quantity": "，".join(quantities) or "0 条岗位",
        "status": status,
    }


def _local_job_source_overview(
    session: Any,
    *,
    offerio_provider_factory: Callable[[], Any],
    sample_limit: int | str | None = 10,
    include_external_job_board: bool = True,
) -> dict[str, Any]:
    if session is None:
        return {
            "tool_name": LOCAL_JOB_SOURCE_OVERVIEW_TOOL,
            "ok": False,
            "error": "DATABASE_SESSION_UNAVAILABLE",
            "result": {"message": "Database session is unavailable."},
        }

    from sqlalchemy import func, select

    from app.domains.jobs.models import JobSource

    limit = _bounded_int(sample_limit, default=10, minimum=1, maximum=50)
    source_count = int(session.scalar(select(func.count(JobSource.id))) or 0)
    enabled_source_count = int(session.scalar(select(func.count(JobSource.id)).where(JobSource.enabled.is_(True))) or 0)
    disabled_source_count = int(session.scalar(select(func.count(JobSource.id)).where(JobSource.enabled.is_(False))) or 0)
    unsynced_source_count = int(session.scalar(select(func.count(JobSource.id)).where(JobSource.last_synced_at.is_(None))) or 0)
    sources = list(session.scalars(select(JobSource).order_by(JobSource.enabled.desc(), JobSource.name.asc()).limit(limit)).all())

    return {
        "tool_name": LOCAL_JOB_SOURCE_OVERVIEW_TOOL,
        "ok": True,
        "result": {
            "source_count": source_count,
            "enabled_source_count": enabled_source_count,
            "disabled_source_count": disabled_source_count,
            "unsynced_source_count": unsynced_source_count,
            "sources_by_type": _count_job_sources_by_attr(session, JobSource.source_type),
            "sources_by_fetch_mode": _count_job_sources_by_attr(session, JobSource.fetch_mode),
            "sample_sources": [_job_source_sample_payload(source) for source in sources],
            "external_job_board": _offerio_job_board_overview(offerio_provider_factory) if include_external_job_board else {"ok": False, "skipped": True},
        },
    }


def _count_job_sources_by_attr(session: Any, column: Any) -> dict[str, int]:
    from sqlalchemy import func, select

    rows = session.execute(select(column, func.count()).group_by(column).order_by(column)).all()
    return {str(_value(key) or "unknown"): int(count or 0) for key, count in rows}


def _job_source_sample_payload(source: Any) -> dict[str, Any]:
    return {
        "id": source.id,
        "name": source.name,
        "source_type": _value(source.source_type),
        "fetch_mode": _value(source.fetch_mode),
        "trust_level": _value(source.trust_level),
        "enabled": bool(source.enabled),
        "last_synced_at": source.last_synced_at.isoformat() if source.last_synced_at else None,
    }


def _offerio_job_board_overview(offerio_provider_factory: Callable[[], Any]) -> dict[str, Any]:
    try:
        provider = offerio_provider_factory()
        openings = provider.list_company_openings(page=1, page_size=1)
        companies = provider.list_companies(job_type="校招", page=1, page_size=1)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": True,
        "offerio_company_openings_total": int(getattr(openings, "total", 0) or 0),
        "offerio_company_jobs_total": int(getattr(companies, "total", 0) or 0),
    }


def _sample_distinct_strings(session: Any, query: Any) -> list[str]:
    result: list[str] = []
    for value in session.scalars(query).all():
        text = _non_empty_str(value)
        if text and text not in result:
            result.append(text)
    return result


def _get_or_create_offerio_company_jobs_source(session: Any, lead_service: Any, page_size: int) -> Any:
    from sqlalchemy import select

    from app.domains.jobs.models import JobSource, JobSourceFetchMode, JobSourceTrustLevel, JobSourceType
    from app.domains.jobs.schemas import JobSourceCreate

    for name in OFFERIO_COMPANY_JOBS_LEGACY_SOURCE_NAMES:
        source = session.scalar(select(JobSource).where(JobSource.name == name))
        if source is not None:
            return _normalize_offerio_company_jobs_source(source, page_size)

    source = session.scalar(
        select(JobSource)
        .where(
            JobSource.source_type == JobSourceType.OFFICIAL_API,
            JobSource.fetch_mode == JobSourceFetchMode.OFFICIAL_API,
            JobSource.entry_url.like("%/api/recruitment/job-companies%"),
        )
        .order_by(JobSource.enabled.desc(), JobSource.created_at.asc())
    )
    if source is not None:
        return _normalize_offerio_company_jobs_source(source, page_size)

    return lead_service.create_source(
        JobSourceCreate(
            name=OFFERIO_COMPANY_JOBS_SOURCE_NAME,
            source_type=JobSourceType.OFFICIAL_API,
            entry_url=_offerio_company_jobs_entry_url(page_size),
            trust_level=JobSourceTrustLevel.MEDIUM_HIGH,
            fetch_mode=JobSourceFetchMode.OFFICIAL_API,
            notes="Auto-created source for OfferIO company aggregated campus recruiting jobs.",
            raw_payload={"created_by": "agent_tool", "tool_name": OFFERIO_COMPANY_JOBS_TOOL},
        )
    )


def _normalize_offerio_company_jobs_source(source: Any, page_size: int) -> Any:
    from app.domains.jobs.models import JobSourceFetchMode, JobSourceTrustLevel, JobSourceType

    source.source_type = JobSourceType.OFFICIAL_API
    source.fetch_mode = JobSourceFetchMode.OFFICIAL_API
    source.trust_level = source.trust_level or JobSourceTrustLevel.MEDIUM_HIGH
    source.enabled = True
    if not source.entry_url or "/api/recruitment/job-companies" not in source.entry_url:
        source.entry_url = _offerio_company_jobs_entry_url(page_size)
    else:
        source.entry_url = _normalize_offerio_company_jobs_entry_url(source.entry_url)
    return source


def _offerio_company_jobs_entry_url(page_size: int) -> str:
    return OFFERIO_COMPANY_JOBS_ENTRY_URL.replace("pageSize=50", f"pageSize={page_size}")


def _normalize_offerio_company_jobs_entry_url(entry_url: str) -> str:
    if "pageSize=" in entry_url:
        return re.sub(r"([?&]pageSize=)\d+", r"\g<1>50", entry_url)
    separator = "&" if "?" in entry_url else "?"
    return f"{entry_url}{separator}pageSize=50"


def _offerio_company_jobs_page_size(total_limit: int) -> int:
    return 50


def _bounded_int(value: int | str | None, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _non_empty_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_str(value: Any, field_name: str) -> str:
    text = _non_empty_str(value)
    if text is None:
        raise ValueError(f"{field_name} is required")
    return text


def _value(value: Any) -> Any:
    return getattr(value, "value", value)


def _mcp_handler(client: Any, tool_name: str) -> Callable[..., Any]:
    def handler(_session: Any, **arguments: Any) -> Any:
        return client.call_tool(tool_name=tool_name, arguments=arguments)

    return handler


def _mcp_risk_level(policy: MCPToolPolicy, tool_name: str) -> AgentToolRiskLevel:
    if policy.requires_confirmation(tool_name):
        return AgentToolRiskLevel.HIGH
    if tool_name in {"open_page", "read_page"}:
        return AgentToolRiskLevel.LOW
    return AgentToolRiskLevel.MEDIUM
