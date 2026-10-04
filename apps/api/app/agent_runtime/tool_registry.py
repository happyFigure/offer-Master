from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
import logging
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import uuid4

from app.agent_runtime.reflection.schemas import (
    CapabilityResultEvaluationSpec,
    campus_recruiting_web_search_result_evaluation_spec,
    result_evaluation_spec_for_capability,
)
from app.agent_runtime.web_search_query import normalize_external_web_search_query
from app.mcp_gateway.tool_policy import MCPToolPolicy
from app.mcp_gateway.client import MCPToolCallResult
from app.mcp_gateway.dbx_readonly import (
    DBX_READONLY_TOOL_NAMES,
    dbx_tool_description,
    dbx_tool_input_schema,
    validate_dbx_tool_arguments,
)
from app.mcp_gateway.qq_mail_readonly import qq_mail_tool_description, qq_mail_tool_input_schema


logger = logging.getLogger(__name__)


class AgentToolRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class AgentToolCandidateProfile:
    categories: frozenset[str] = field(default_factory=frozenset)
    keywords: frozenset[str] = field(default_factory=frozenset)
    examples: tuple[str, ...] = ()
    # These hints make similar tools easier to disambiguate before the model emits a tool call.
    use_when: tuple[str, ...] = ()
    do_not_use_when: tuple[str, ...] = ()
    positive_examples: tuple[str, ...] = ()
    negative_examples: tuple[str, ...] = ()
    required_context_focus: frozenset[str] = field(default_factory=frozenset)
    disambiguation_notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "categories", frozenset(str(item).strip() for item in self.categories if str(item).strip()))
        object.__setattr__(self, "keywords", frozenset(str(item).strip() for item in self.keywords if str(item).strip()))
        object.__setattr__(self, "examples", tuple(str(item).strip() for item in self.examples if str(item).strip()))
        object.__setattr__(self, "use_when", _clean_tuple(self.use_when))
        object.__setattr__(self, "do_not_use_when", _clean_tuple(self.do_not_use_when))
        object.__setattr__(self, "positive_examples", _clean_tuple(self.positive_examples))
        object.__setattr__(self, "negative_examples", _clean_tuple(self.negative_examples))
        object.__setattr__(
            self,
            "required_context_focus",
            frozenset(str(item).strip() for item in self.required_context_focus if str(item).strip()),
        )
        object.__setattr__(self, "disambiguation_notes", _clean_tuple(self.disambiguation_notes))


def _clean_tuple(items: Iterable[Any]) -> tuple[str, ...]:
    return tuple(str(item).strip() for item in items if str(item).strip())


@dataclass(frozen=True)
class AgentToolSemanticProfile:
    """Runtime-readable meaning of one ordinary tool.

    candidate_profile is for selecting a tool before execution. This semantic
    profile is for deterministic runtime checks after execution: what operation
    did this tool represent, what target does it affect, and what result shape
    proves it advanced the user's goal.
    """

    intent: str
    target_type: str = "tool_result"
    modifies: tuple[str, ...] = ()
    does_not_modify: tuple[str, ...] = ()
    success_criteria: tuple[str, ...] = ("tool_result_ok",)
    failure_recovery: tuple[str, ...] = ("return_structured_error",)
    result_contract: dict[str, Any] = field(default_factory=dict)
    next_action_hints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "intent", str(self.intent or "").strip())
        object.__setattr__(self, "target_type", str(self.target_type or "tool_result").strip() or "tool_result")
        object.__setattr__(self, "modifies", _clean_tuple(self.modifies))
        object.__setattr__(self, "does_not_modify", _clean_tuple(self.does_not_modify))
        object.__setattr__(self, "success_criteria", _clean_tuple(self.success_criteria) or ("tool_result_ok",))
        object.__setattr__(self, "failure_recovery", _clean_tuple(self.failure_recovery) or ("return_structured_error",))
        object.__setattr__(self, "next_action_hints", _clean_tuple(self.next_action_hints))
        object.__setattr__(self, "result_contract", dict(self.result_contract or {}))

    def to_metadata(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "target_type": self.target_type,
            "modifies": list(self.modifies),
            "does_not_modify": list(self.does_not_modify),
            "success_criteria": list(self.success_criteria),
            "failure_recovery": list(self.failure_recovery),
            "result_contract": dict(self.result_contract),
            "next_action_hints": list(self.next_action_hints),
        }


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
    semantic_profile: AgentToolSemanticProfile | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Agent tool name is required")
        if not self.description.strip():
            raise ValueError(f"Agent tool description is required: {self.name}")
        object.__setattr__(self, "allowed_source_types", frozenset(self.allowed_source_types))
        if self.result_evaluation is None:
            object.__setattr__(self, "result_evaluation", result_evaluation_spec_for_capability(self.name))
        if self.semantic_profile is None:
            object.__setattr__(
                self,
                "semantic_profile",
                AgentToolSemanticProfile(intent=self.name.replace(".", "_"), target_type="tool_result"),
            )


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
APPLICATION_FIND_FOR_MAIL_TOOL = "applications.find_for_mail_notification"
APPLICATION_SYNC_MAIL_TOOL = "applications.sync_mail_application"
ARTIFACT_EXPORT_TOOL = "artifact.export"
EXTERNAL_WEB_SEARCH_TOOL = "external.web_search"
FILESYSTEM_LIST_DIR_TOOL = "filesystem.list_dir"
FILESYSTEM_PATH_EXISTS_TOOL = "filesystem.path_exists"
FILESYSTEM_PATH_STAT_TOOL = "filesystem.path_stat"
FILESYSTEM_READ_FILE_TOOL = "filesystem.read_file"
FILESYSTEM_WRITE_TEXT_TOOL = "filesystem.write_text"
FILESYSTEM_REPLACE_TEXT_TOOL = "filesystem.replace_text"
FILESYSTEM_COPY_FILE_TOOL = "filesystem.copy_file"
FILESYSTEM_MOVE_FILE_TOOL = "filesystem.move_file"
FILESYSTEM_DELETE_PATH_TOOL = "filesystem.delete_path"
FILESYSTEM_MAKE_DIR_TOOL = "filesystem.make_dir"
LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL = "local.company_database_overview"
LOCAL_JOB_SOURCE_OVERVIEW_TOOL = "local.job_source_overview"
DATABASE_COMPANY_SEARCH_TOOL = "database.company_search"
DATABASE_COMPANY_LIST_TOOL = "database.company_list"
DATABASE_COMPANY_PROFILE_TOOL = "database.company_profile"
DATABASE_JOB_SEARCH_TOOL = "database.job_search"
DATABASE_SOURCE_SEARCH_TOOL = "database.source_search"
DATABASE_COMPANY_UPDATE_TOOL = "database.company_update"
DATABASE_JOB_LEAD_DELETE_TOOL = "database.job_lead_delete"
OFFERIO_COMPANY_JOBS_TOOL = "offerio.sync_company_jobs"
OFFERIO_COMPANY_JOBS_SOURCE_NAME = "OfferIO 公司聚合岗位库"
OFFERIO_COMPANY_JOBS_LEGACY_SOURCE_NAMES = (OFFERIO_COMPANY_JOBS_SOURCE_NAME, "OfferIO company jobs")
OFFERIO_COMPANY_JOBS_ENTRY_URL = "https://offerio.work/api/recruitment/job-companies?jobType=校招&page=1&pageSize=50"
SKILL_LIST_TOOL = "skill_list"
SKILL_LIST_ACTIONS_TOOL = "skill_list_actions"
SKILL_READ_TOOL = "skill_read"
SKILL_CALL_TOOL = "skill_call"


def create_default_agent_tool_registry(
    *,
    content_source_client: Any | None = None,
    offerio_provider_factory: Callable[[], Any] | None = None,
    external_task_dispatcher: Callable[[Any, str], dict[str, Any]] | None = None,
    external_web_search_executor: Callable[[str, int], dict[str, Any]] | None = None,
) -> AgentToolRegistry:
    registry = AgentToolRegistry()
    registry.register_many(create_application_agent_tool_definitions(external_task_dispatcher=external_task_dispatcher))
    registry.register_many(create_artifact_agent_tool_definitions())
    registry.register_many(create_external_web_search_agent_tool_definitions(external_web_search_executor=external_web_search_executor))
    registry.register_many(create_database_agent_tool_definitions())
    registry.register_many(create_local_company_database_agent_tool_definitions())
    registry.register_many(create_local_job_source_agent_tool_definitions(offerio_provider_factory=offerio_provider_factory))
    registry.register_many(create_filesystem_agent_tool_definitions())
    registry.register_many(create_skill_lazy_agent_tool_definitions())
    registry.register_many(_memory_tool_definitions())
    registry.register_many(create_job_source_agent_tool_definitions(offerio_provider_factory=offerio_provider_factory))
    registry.register_many(create_content_source_agent_tool_definitions(content_source_client))
    return registry


def create_artifact_agent_tool_definitions(*, sandbox_base_dir: str | Path | None = None) -> list[AgentToolDefinition]:
    standard_output = {"type": "object", "required": ["tool_name", "ok", "result"]}
    return [
        AgentToolDefinition(
            name=ARTIFACT_EXPORT_TOOL,
            description=(
                "Export one SDK child-agent artifact from its sandbox output directory to a user-selected local path "
                "after explicit confirmation."
            ),
            input_schema={
                "type": "object",
                "required": ["artifact_uri", "destination"],
                "properties": {
                    "artifact_uri": {
                        "type": "string",
                        "description": "Sandbox artifact URI, e.g. artifact-sandbox://run-id/output/resume_tailored.md.",
                    },
                    "destination": {"type": "string", "description": "Local destination path selected by the user."},
                    "overwrite": {"type": "boolean", "default": False},
                },
                "additionalProperties": False,
            },
            output_schema=standard_output,
            handler=lambda session, **arguments: _export_artifact(
                session,
                sandbox_base_dir=sandbox_base_dir,
                **arguments,
            ),
            risk_level=AgentToolRiskLevel.HIGH,
            requires_confirmation=True,
            allowed_source_types=frozenset({"agent_chat", "filesystem"}),
            candidate_profile=AgentToolCandidateProfile(
                categories=frozenset({"artifact_export", "filesystem_write", "filesystem_operation"}),
                keywords=frozenset({"导出", "保存产物", "保存简历", "输出文件", "export artifact"}),
                examples=("把刚才生成的简历导出到 C:/Users/phoenix/Documents/resume.md",),
            ),
            semantic_profile=AgentToolSemanticProfile(
                intent="artifact_export",
                target_type="file",
                modifies=("destination_file",),
                success_criteria=("tool_result_ok", "destination_path_present"),
                failure_recovery=("ask user for destination path", "request overwrite confirmation when destination exists"),
                result_contract={
                    "required_fields": ["result.destination_path"],
                    "resource_effect_templates": [
                        {
                            "resource_type": "file",
                            "action": "created",
                            "operation": "artifact_export",
                            "source_path": "$result.result.source_uri",
                            "target_path": "$result.result.destination_path",
                            "focus_path": "$result.result.destination_path",
                            "focus_kind": "file",
                            "aliases": ["刚才导出的文件", "导出的简历", "保存出来的文件"],
                        }
                    ],
                },
                next_action_hints=("offer to open or rename the exported file",),
            ),
        )
    ]


def create_application_agent_tool_definitions(
    *,
    external_task_dispatcher: Callable[[Any, str], dict[str, Any]] | None = None,
) -> list[AgentToolDefinition]:
    return [
        AgentToolDefinition(
            name=APPLICATION_FIND_FOR_MAIL_TOOL,
            description=(
                "Read existing application records to match a structured QQ Mail recruitment notice. "
                "This tool is read-only and never changes jobs, applications, or events."
            ),
            input_schema={
                "type": "object",
                "required": [],
                "properties": {
                    "company_name": {"type": ["string", "null"]},
                    "job_title": {"type": ["string", "null"]},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 10},
                },
                "additionalProperties": False,
            },
            output_schema={"type": "object", "required": ["tool_name", "ok", "result"]},
            handler=_find_applications_for_mail,
            risk_level=AgentToolRiskLevel.LOW,
            allowed_source_types=frozenset({"agent_chat", "mail_sync", "qq_mail_agent"}),
            candidate_profile=AgentToolCandidateProfile(
                categories=frozenset({"application_progress", "mail_recruitment"}),
                keywords=frozenset({"匹配投递", "应聘记录", "邮件对应岗位", "投递面板"}),
                use_when=("邮件子 Agent 已提取公司或岗位，需要匹配现有投递记录",),
            ),
            semantic_profile=AgentToolSemanticProfile(
                intent="find_application_for_mail_notification",
                target_type="application_record",
                does_not_modify=("application", "application_event", "mailbox"),
                success_criteria=("match_candidates_returned",),
                next_action_hints=("choose_application_before_staging_mail_event",),
            ),
        ),
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
        ),
        AgentToolDefinition(
            name=APPLICATION_SYNC_MAIL_TOOL,
            description=(
                "Synchronize a structured QQ Mail recruiting event directly into the application board. "
                "A company name is required; job_title may be absent and will remain explicitly unspecified. "
                "Reuses an exact existing application when possible, otherwise creates a company, job, "
                "application, and mail event. Message-ID plus event type makes the operation idempotent."
            ),
            input_schema={
                "type": "object",
                "required": ["company_name", "event_type", "title", "confidence", "source_message_id"],
                "properties": {
                    "company_name": {"type": "string", "minLength": 1, "maxLength": 255},
                    "job_title": {"type": ["string", "null"], "maxLength": 255},
                    "event_type": {"type": "string", "minLength": 1, "maxLength": 64},
                    "title": {"type": "string", "minLength": 1, "maxLength": 255},
                    "body": {"type": ["string", "null"]},
                    "source_sent_at": {"type": ["string", "null"], "format": "date-time"},
                    "scheduled_at": {"type": ["string", "null"], "format": "date-time"},
                    "deadline_at": {"type": ["string", "null"], "format": "date-time"},
                    "deadline_offset_hours": {"type": ["integer", "null"], "minimum": 1, "maximum": 720},
                    "timezone": {"type": ["string", "null"], "maxLength": 64},
                    "join_url": {"type": ["string", "null"], "format": "uri", "maxLength": 2048},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "source_message_id": {"type": "string", "minLength": 1, "maxLength": 512},
                    "source_uid": {"type": ["string", "null"], "maxLength": 128},
                    "evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
                    "to_status": {"type": ["string", "null"]},
                    "model_name": {"type": ["string", "null"], "maxLength": 128},
                    "parser_version": {"type": ["string", "null"], "maxLength": 64},
                },
                "additionalProperties": False,
            },
            output_schema={"type": "object", "required": ["tool_name", "ok", "result"]},
            handler=_sync_mail_application,
            risk_level=AgentToolRiskLevel.MEDIUM,
            allowed_source_types=frozenset({"agent_chat", "mail_sync", "qq_mail_agent"}),
            candidate_profile=AgentToolCandidateProfile(
                categories=frozenset({"application_progress", "mail_recruitment"}),
                keywords=frozenset({"邮箱同步", "同步投递", "邮件面试", "邮件测评", "邮件笔试", "投递面板"}),
                use_when=("QQ Mail 子 Agent 已提取含公司名称的招聘事件，需要直接同步到投递面板",),
                do_not_use_when=("邮件中无法确定公司名称", "只有邮件列表头，没有读取正文和招聘事件证据"),
            ),
            semantic_profile=AgentToolSemanticProfile(
                intent="sync_mail_recruitment_to_application_board",
                target_type="application_record",
                modifies=("company", "job", "application", "application_event"),
                does_not_modify=("mailbox",),
                success_criteria=("application_visible_on_board", "mail_event_saved", "company_name_present"),
                failure_recovery=("do_not_sync_without_company_name", "retry_is_safe_with_same_message_id"),
            ),
        ),
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
                    "mode": {
                        "type": "string",
                        "enum": ["sources", "company_board_count"],
                        "default": "sources",
                        "description": "Use company_board_count for the Company Exhibition company total; sources is a local source inventory.",
                    },
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
                    "mode": {
                        "type": "string",
                        "enum": ["sources", "company_board_count"],
                        "default": "sources",
                        "description": "Use company_board_count for the Company Exhibition total; sources returns the local source inventory.",
                    },
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


def create_database_agent_tool_definitions() -> list[AgentToolDefinition]:
    standard_output = {"type": "object", "required": ["tool_name", "ok", "result"]}
    read_source_types = frozenset({"agent_chat", "job_discovery"})
    return [
        AgentToolDefinition(
            name=DATABASE_COMPANY_LIST_TOOL,
            description="List distinct companies found across local company profiles, formal jobs, job leads, and recruiting signals.",
            input_schema={
                "type": "object",
                "properties": {
                    "keyword": {"type": ["string", "null"]},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
                },
                "additionalProperties": False,
            },
            output_schema=standard_output,
            handler=lambda session, **arguments: _database_company_list(session, **arguments),
            risk_level=AgentToolRiskLevel.LOW,
            requires_confirmation=False,
            allowed_source_types=read_source_types,
            candidate_profile=AgentToolCandidateProfile(
                categories=frozenset({"local_company_list"}),
                keywords=frozenset({"公司列表", "企业列表", "有哪些公司", "列出公司", "数据库公司"}),
                examples=("数据库里有哪些公司，给我20个",),
            ),
            semantic_profile=AgentToolSemanticProfile(
                intent="list_local_companies",
                target_type="company_list",
                modifies=(),
                does_not_modify=("local_database", "user_files"),
                success_criteria=("tool_result_ok", "result_field_present:companies"),
                failure_recovery=("query local database with narrower keyword", "ask user for company filter"),
                result_contract={"required_fields": ["result.companies"]},
                next_action_hints=("summarize returned companies", "ask whether to filter or inspect company profiles"),
            ),
        ),
        AgentToolDefinition(
            name=DATABASE_COMPANY_SEARCH_TOOL,
            description="Search local company records, formal jobs, job leads, and recruiting signals by company name.",
            input_schema={
                "type": "object",
                "properties": {
                    "company_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 20,
                    },
                    "keyword": {"type": ["string", "null"]},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                },
                "additionalProperties": False,
            },
            output_schema=standard_output,
            handler=lambda session, **arguments: _database_company_search(session, **arguments),
            risk_level=AgentToolRiskLevel.LOW,
            requires_confirmation=False,
            allowed_source_types=read_source_types,
            candidate_profile=AgentToolCandidateProfile(
                categories=frozenset({"local_company_search"}),
                keywords=frozenset({"数据库", "公司", "企业", "查公司", "公司搜索", "公司列表"}),
                examples=("查数据库里腾讯和京东有没有记录",),
            ),
        ),
        AgentToolDefinition(
            name=DATABASE_COMPANY_PROFILE_TOOL,
            description="Read a joined local company profile with its formal jobs, job leads, and recruiting signals.",
            input_schema={
                "type": "object",
                "required": ["company_name"],
                "properties": {
                    "company_name": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                },
                "additionalProperties": False,
            },
            output_schema=standard_output,
            handler=lambda session, **arguments: _database_company_profile(session, **arguments),
            risk_level=AgentToolRiskLevel.LOW,
            requires_confirmation=False,
            allowed_source_types=read_source_types,
            candidate_profile=AgentToolCandidateProfile(
                categories=frozenset({"local_company_profile"}),
                keywords=frozenset({"公司详情", "企业详情", "公司信息", "企业档案", "关于公司"}),
                examples=("看一下数据库中关于京东这个公司的详细信息",),
            ),
        ),
        AgentToolDefinition(
            name=DATABASE_JOB_SEARCH_TOOL,
            description="Search formal local jobs using company, title, city, type, status, or keyword filters.",
            input_schema={
                "type": "object",
                "properties": {
                    "keyword": {"type": ["string", "null"]},
                    "company_name": {"type": ["string", "null"]},
                    "title": {"type": ["string", "null"]},
                    "city": {"type": ["string", "null"]},
                    "job_type": {"type": ["string", "null"]},
                    "status": {"type": ["string", "null"]},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
                },
                "additionalProperties": False,
            },
            output_schema=standard_output,
            handler=lambda session, **arguments: _database_job_search(session, **arguments),
            risk_level=AgentToolRiskLevel.LOW,
            requires_confirmation=False,
            allowed_source_types=read_source_types,
            candidate_profile=AgentToolCandidateProfile(
                categories=frozenset({"local_job_search"}),
                keywords=frozenset({"数据库岗位", "本地岗位", "岗位搜索", "岗位列表", "正式岗位"}),
                examples=("查数据库里腾讯的 Python 后端岗位",),
            ),
        ),
        AgentToolDefinition(
            name=DATABASE_SOURCE_SEARCH_TOOL,
            description="Search local job sources and return their type, trust, enabled state, and record counts.",
            input_schema={
                "type": "object",
                "properties": {
                    "keyword": {"type": ["string", "null"]},
                    "source_type": {"type": ["string", "null"]},
                    "enabled": {"type": ["boolean", "null"]},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
                },
                "additionalProperties": False,
            },
            output_schema=standard_output,
            handler=lambda session, **arguments: _database_source_search(session, **arguments),
            risk_level=AgentToolRiskLevel.LOW,
            requires_confirmation=False,
            allowed_source_types=read_source_types,
            candidate_profile=AgentToolCandidateProfile(
                categories=frozenset({"local_source_search"}),
                keywords=frozenset({"来源搜索", "来源详情", "岗位来源", "来源列表", "招聘来源"}),
                examples=("查一下本地岗位来源库里有哪些官方来源",),
            ),
        ),
        AgentToolDefinition(
            name=DATABASE_COMPANY_UPDATE_TOOL,
            description="Update a fixed set of local company profile fields after explicit user confirmation.",
            input_schema={
                "type": "object",
                "properties": {
                    "company_id": {"type": ["string", "null"]},
                    "company_name": {"type": ["string", "null"]},
                    "name": {"type": ["string", "null"]},
                    "website_url": {"type": ["string", "null"]},
                    "industry": {"type": ["string", "null"]},
                    "city": {"type": ["string", "null"]},
                    "country": {"type": ["string", "null"]},
                },
                "additionalProperties": False,
            },
            output_schema=standard_output,
            handler=lambda session, **arguments: _database_company_update(session, **arguments),
            risk_level=AgentToolRiskLevel.HIGH,
            requires_confirmation=True,
            allowed_source_types=frozenset({"agent_chat", "job_discovery"}),
        ),
        AgentToolDefinition(
            name=DATABASE_JOB_LEAD_DELETE_TOOL,
            description="Mark a local job lead invalid while preserving its audit record after explicit user confirmation.",
            input_schema={
                "type": "object",
                "required": ["lead_id"],
                "properties": {
                    "lead_id": {"type": "string"},
                    "reason": {"type": ["string", "null"]},
                },
                "additionalProperties": False,
            },
            output_schema=standard_output,
            handler=lambda session, **arguments: _database_job_lead_delete(session, **arguments),
            risk_level=AgentToolRiskLevel.HIGH,
            requires_confirmation=True,
            allowed_source_types=frozenset({"agent_chat", "job_discovery"}),
        ),
    ]


def create_filesystem_agent_tool_definitions(*, script_root: str | Path | None = None) -> list[AgentToolDefinition]:
    common_source_types = frozenset({"agent_chat", "filesystem"})
    standard_output = {"type": "object", "required": ["tool_name", "ok", "result"]}
    path_property = {"type": "string", "description": "Absolute local file or directory path."}
    return [
        AgentToolDefinition(
            name=FILESYSTEM_LIST_DIR_TOOL,
            description="List files and folders under a user-provided local directory path.",
            input_schema={
                "type": "object",
                "required": ["path"],
                "properties": {"path": path_property},
                "additionalProperties": False,
            },
            output_schema=standard_output,
            handler=_filesystem_script_handler(FILESYSTEM_LIST_DIR_TOOL, "list_dir.py", script_root=script_root),
            risk_level=AgentToolRiskLevel.LOW,
            requires_confirmation=False,
            allowed_source_types=common_source_types,
            candidate_profile=AgentToolCandidateProfile(
                categories=frozenset({"filesystem_list", "filesystem_operation"}),
                keywords=frozenset({"列目录", "查看目录", "文件夹", "有哪些文件", "list dir"}),
                examples=("列出 F:/pythonProject/OfferMaster 下面有哪些文件",),
            ),
        ),
        AgentToolDefinition(
            name=FILESYSTEM_PATH_EXISTS_TOOL,
            description="Check whether a user-provided local path exists.",
            input_schema={
                "type": "object",
                "required": ["path"],
                "properties": {"path": path_property},
                "additionalProperties": False,
            },
            output_schema=standard_output,
            handler=_filesystem_script_handler(FILESYSTEM_PATH_EXISTS_TOOL, "path_exists.py", script_root=script_root),
            risk_level=AgentToolRiskLevel.LOW,
            requires_confirmation=False,
            allowed_source_types=common_source_types,
            candidate_profile=AgentToolCandidateProfile(
                categories=frozenset({"filesystem_stat", "filesystem_operation"}),
                keywords=frozenset({"路径存在", "文件存在", "目录存在", "有没有这个文件"}),
                examples=("帮我确认这个本地简历文件是否存在",),
            ),
        ),
        AgentToolDefinition(
            name=FILESYSTEM_PATH_STAT_TOOL,
            description="Read size, type, and modified-time metadata for a user-provided local path.",
            input_schema={
                "type": "object",
                "required": ["path"],
                "properties": {"path": path_property},
                "additionalProperties": False,
            },
            output_schema=standard_output,
            handler=_filesystem_script_handler(FILESYSTEM_PATH_STAT_TOOL, "path_stat.py", script_root=script_root),
            risk_level=AgentToolRiskLevel.LOW,
            requires_confirmation=False,
            allowed_source_types=common_source_types,
            candidate_profile=AgentToolCandidateProfile(
                categories=frozenset({"filesystem_stat", "filesystem_operation"}),
                keywords=frozenset({"文件大小", "修改时间", "文件信息", "path stat"}),
                examples=("看一下这个 tex 文件大小和更新时间",),
            ),
        ),
        AgentToolDefinition(
            name=FILESYSTEM_READ_FILE_TOOL,
            description=(
                "Read a bounded slice of a user-provided local text file. Use this when the user asks to inspect "
                "file contents; do not use it only to answer a filename/path question or to rename/move a file."
            ),
            input_schema={
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": path_property,
                    "offset": {"type": "integer", "minimum": 0, "default": 0},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 200},
                    "encoding": {"type": "string", "default": "auto"},
                },
                "additionalProperties": False,
            },
            output_schema=standard_output,
            handler=_filesystem_script_handler(FILESYSTEM_READ_FILE_TOOL, "read_file.py", script_root=script_root),
            risk_level=AgentToolRiskLevel.LOW,
            requires_confirmation=False,
            allowed_source_types=common_source_types,
            candidate_profile=AgentToolCandidateProfile(
                categories=frozenset({"filesystem_read", "filesystem_operation"}),
                keywords=frozenset({"读取文件", "读文件", "打开文件", "查看文件", "read file", "tex", "简历文件"}),
                examples=("读取 C:/Users/phoenix/Documents/Obsidian Vault/简历/简历.tex",),
                use_when=(
                    "用户要查看、分析或摘取文件内容",
                    "用户问文件里面写了什么，或要求基于正文继续分析",
                ),
                do_not_use_when=(
                    "用户只是在问文件名、文件路径或扩展名",
                    "用户要移动、重命名、复制、删除或覆盖文件",
                ),
                positive_examples=(
                    "读一下这个 tex 简历文件内容",
                    "看一下这个文件里面主要写了什么",
                ),
                negative_examples=(
                    "这个文件叫什么名字",
                    "把这个文件名改成刘汉卿-后端开发-AI-Agent.tex",
                ),
                required_context_focus=frozenset({"content", "file"}),
                disambiguation_notes=(
                    "read_file 读取文件正文；如果用户关注的是文件名，不要为了看文件名而读取正文。",
                ),
            ),
            semantic_profile=AgentToolSemanticProfile(
                intent="read_file_content",
                target_type="file_content",
                modifies=(),
                does_not_modify=("file_path", "file_content"),
                success_criteria=("tool_result_ok", "result_field_present:result.content"),
                failure_recovery=("use active_file.path when path is omitted", "ask user for a precise file path"),
                result_contract={"required_fields": ["result.content"]},
                next_action_hints=("summarize bounded content", "continue with a narrower read if content is truncated"),
            ),
        ),
        AgentToolDefinition(
            name=FILESYSTEM_WRITE_TEXT_TOOL,
            description="Write text to a user-provided local file path; overwrites only when explicitly requested and confirmed.",
            input_schema={
                "type": "object",
                "required": ["path", "text"],
                "properties": {
                    "path": path_property,
                    "text": {"type": "string"},
                    "encoding": {"type": "string", "default": "utf-8"},
                    "overwrite": {"type": "boolean", "default": False},
                },
                "additionalProperties": False,
            },
            output_schema=standard_output,
            handler=_filesystem_script_handler(FILESYSTEM_WRITE_TEXT_TOOL, "write_text.py", script_root=script_root),
            risk_level=AgentToolRiskLevel.HIGH,
            requires_confirmation=True,
            allowed_source_types=common_source_types,
            candidate_profile=AgentToolCandidateProfile(
                categories=frozenset({"filesystem_write", "filesystem_operation"}),
                keywords=frozenset({"写文件", "保存文件", "修改文件", "覆盖文件", "write file", "改成"}),
                examples=("把修改后的简历写回 tex 文件",),
            ),
        ),
        AgentToolDefinition(
            name=FILESYSTEM_REPLACE_TEXT_TOOL,
            description=(
                "Replace exact text inside a user-provided local text file while preserving all other content. "
                "Use it for content edits only; do not use it to rename or move the file itself."
            ),
            input_schema={
                "type": "object",
                "required": ["path", "old_text", "new_text"],
                "properties": {
                    "path": path_property,
                    "old_text": {"type": "string", "description": "Exact text to replace."},
                    "new_text": {"type": "string", "description": "Replacement text."},
                    "encoding": {"type": "string", "default": "utf-8"},
                    "count": {"type": "integer", "minimum": 0, "default": 0, "description": "0 means replace all occurrences."},
                },
                "additionalProperties": False,
            },
            output_schema=standard_output,
            handler=_filesystem_script_handler(FILESYSTEM_REPLACE_TEXT_TOOL, "replace_text.py", script_root=script_root),
            risk_level=AgentToolRiskLevel.HIGH,
            requires_confirmation=True,
            allowed_source_types=common_source_types,
            candidate_profile=AgentToolCandidateProfile(
                categories=frozenset({"filesystem_replace", "filesystem_write", "filesystem_operation"}),
                keywords=frozenset({"替换文本", "换成", "换为", "改成", "改为", "replace text", "只改名字"}),
                examples=("把这个 tex 简历里的刘汉卿替换为王爷，其他不要动",),
                use_when=(
                    "用户要修改文件正文中的一段精确文本",
                    "用户明确说文件里、正文里、内容中、简历里要替换文字",
                ),
                do_not_use_when=(
                    "用户要修改文件名、文件路径、扩展名或移动位置",
                    "当前 active_file.last_focus 是 filename，并且用户只说把名字改成某个文件名",
                ),
                positive_examples=(
                    "把简历里的刘汉卿改成王爷",
                    "把这个 tex 文件中的 SpringBoot 替换为 Spring Boot",
                ),
                negative_examples=(
                    "把这个文件名改成刘汉卿-后端开发-AI-Agent.tex",
                    "把名字改成刘汉卿-后端开发-AI-Agent，上一轮关注的是文件名",
                ),
                required_context_focus=frozenset({"content"}),
                disambiguation_notes=(
                    "replace_text 修改文件内容，不修改文件名；出现文件名、文件名称、重命名时优先考虑 move_file。",
                ),
            ),
            semantic_profile=AgentToolSemanticProfile(
                intent="replace_file_text",
                target_type="file_content",
                modifies=("file_content",),
                does_not_modify=("file_path", "file_name"),
                success_criteria=("tool_result_ok", "replacement_count_positive"),
                failure_recovery=("read file content to locate exact old_text", "ask user for exact text to replace"),
                result_contract={"required_fields": ["result.stdout"]},
                next_action_hints=("verify replacement count", "report unchanged content when count is zero"),
            ),
        ),
        AgentToolDefinition(
            name=FILESYSTEM_COPY_FILE_TOOL,
             description=(
                 "Copy a user-provided local file or directory after confirmation. "
                 "When the user refers to a previous directory or delegates naming, "
                 "prefer operation_intent with destination.kind and destination.path; "
                 "runtime will normalize the concrete dst."
             ),
            input_schema={
                "type": "object",
                 "required": ["src"],
                "properties": {
                    "src": {"type": "string"},
                    "dst": {"type": "string"},
                    "overwrite": {"type": "boolean", "default": False},
                    "operation_intent": {
                        "type": "object",
                        "description": "Structured semantic candidate from the model; prefer this over guessing dst from the current sentence.",
                        "properties": {
                            "destination": {
                                "type": "object",
                                "properties": {
                                    "kind": {"type": "string", "enum": ["file", "directory"]},
                                    "path": {"type": "string"},
                                    "reference": {"type": "string"},
                                },
                                "required": ["kind", "path"],
                                "additionalProperties": False,
                            },
                            "name_intent": {
                                "type": "object",
                                "description": "Model-selected semantic filename; runtime normalizes it into a safe concrete path.",
                                "properties": {
                                    "mode": {"type": "string", "enum": ["model_proposed", "user_provided"]},
                                    "display_name": {"type": "string"},
                                    "filename": {"type": "string"},
                                    "filename_stem": {"type": "string"},
                                    "extension_policy": {"type": "string", "enum": ["preserve_source", "explicit"]},
                                    "source_basis": {"type": "string", "enum": ["file_content", "conversation", "user_input"]},
                                },
                                "additionalProperties": False,
                            },
                            "name_policy": {
                                "type": "string",
                                "enum": ["content_based"],
                            },
                            "avoid_conflict": {"type": "boolean", "default": True},
                        },
                        "required": ["destination"],
                        "additionalProperties": False,
                    },
                },
                "additionalProperties": False,
            },
            output_schema=standard_output,
            handler=_filesystem_script_handler(FILESYSTEM_COPY_FILE_TOOL, "copy_file.py", script_root=script_root),
            risk_level=AgentToolRiskLevel.HIGH,
            requires_confirmation=True,
            allowed_source_types=common_source_types,
            candidate_profile=AgentToolCandidateProfile(
                categories=frozenset({"filesystem_copy", "filesystem_operation"}),
                keywords=frozenset({"复制文件", "复制目录", "备份文件", "copy file"}),
                examples=("先复制一份简历作为备份",),
            ),
            semantic_profile=AgentToolSemanticProfile(
                intent="copy_file_or_directory",
                target_type="file_copy",
                modifies=("destination_path",),
                does_not_modify=("source_file", "source_directory"),
                success_criteria=("tool_result_ok", "dst_path_exists"),
                failure_recovery=("ask user for destination path", "retry with overwrite only after confirmation"),
                result_contract={"required_fields": ["result.stdout"]},
                next_action_hints=("confirm destination path", "do not claim completion if only source was read"),
            ),
        ),
        AgentToolDefinition(
            name=FILESYSTEM_MOVE_FILE_TOOL,
            description=(
                "Move or rename a user-provided local file or directory after confirmation. Use it when the target "
                "is the file path or filename; it does not edit text inside the file."
            ),
            input_schema={
                "type": "object",
                "required": ["src"],
                "properties": {
                    "src": {"type": "string"},
                    "dst": {"type": "string"},
                    "overwrite": {"type": "boolean", "default": False},
                    "operation_intent": {
                        "type": "object",
                        "description": "Structured semantic candidate from the model; runtime validates the concrete destination before approval.",
                        "properties": {
                            "destination": {
                                "type": "object",
                                "properties": {
                                    "kind": {"type": "string", "enum": ["file", "directory"]},
                                    "path": {"type": "string"},
                                    "reference": {"type": "string"},
                                },
                                "required": ["kind", "path"],
                                "additionalProperties": False,
                            },
                            "name_intent": {
                                "type": "object",
                                "description": "Model-selected semantic filename; runtime normalizes it into a safe concrete path.",
                                "properties": {
                                    "mode": {"type": "string", "enum": ["model_proposed", "user_provided"]},
                                    "display_name": {"type": "string"},
                                    "filename": {"type": "string"},
                                    "filename_stem": {"type": "string"},
                                    "extension_policy": {"type": "string", "enum": ["preserve_source", "explicit"]},
                                    "source_basis": {"type": "string", "enum": ["file_content", "conversation", "user_input"]},
                                },
                                "additionalProperties": False,
                            },
                            "name_policy": {
                                "type": "string",
                                "enum": ["content_based"],
                            },
                            "avoid_conflict": {"type": "boolean", "default": True},
                        },
                        "required": ["destination"],
                        "additionalProperties": False,
                    },
                },
                "additionalProperties": False,
            },
            output_schema=standard_output,
            handler=_filesystem_script_handler(FILESYSTEM_MOVE_FILE_TOOL, "move_file.py", script_root=script_root),
            risk_level=AgentToolRiskLevel.HIGH,
            requires_confirmation=True,
            allowed_source_types=common_source_types,
            candidate_profile=AgentToolCandidateProfile(
                categories=frozenset({"filesystem_move", "filesystem_operation"}),
                keywords=frozenset({"移动文件", "重命名文件", "移动目录", "move file", "rename"}),
                examples=("把这个简历文件重命名",),
                use_when=(
                    "用户要改变文件名、文件路径或移动位置",
                    "用户明确说文件名、文件名称、文件的名字、重命名文件或改名",
                    "active_file.last_focus 是 filename，并且用户说把名字改成新的文件名",
                ),
                do_not_use_when=(
                    "用户要修改文件正文里的姓名、标题或项目内容",
                    "用户明确说简历里、文件里、正文里或内容中要替换文字",
                ),
                positive_examples=(
                    "把这个文件名改成刘汉卿-后端开发-AI-Agent.tex",
                    "把刚才那个简历文件重命名为刘汉卿-后端开发-AI-Agent.tex",
                ),
                negative_examples=(
                    "把简历里的刘汉卿改成王爷",
                    "把文件内容里的项目名称改成 OfferMaster",
                ),
                required_context_focus=frozenset({"filename", "path", "file"}),
                disambiguation_notes=(
                    "move_file 改文件名或路径，不改正文；如果用户说简历里/正文里，应该用 replace_text。",
                    "rename 是 move_file 的同目录特例：src 为原路径，dst 为同目录下的新文件名。",
                ),
            ),
            semantic_profile=AgentToolSemanticProfile(
                intent="rename_or_move_file",
                target_type="file_path",
                modifies=("file_path", "file_name"),
                does_not_modify=("file_content",),
                success_criteria=("tool_result_ok", "dst_path_exists", "source_path_moved_or_renamed"),
                failure_recovery=("derive dst from active_file.path and requested filename", "ask user before overwrite"),
                result_contract={"required_fields": ["result.stdout"]},
                next_action_hints=("confirm old and new path", "do not use replace_text for filename changes"),
            ),
        ),
        AgentToolDefinition(
            name=FILESYSTEM_DELETE_PATH_TOOL,
            description="Delete a user-provided local file or directory after explicit confirmation.",
            input_schema={
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": path_property,
                    "recursive": {"type": "boolean", "default": False},
                    "force": {"type": "boolean", "default": False},
                },
                "additionalProperties": False,
            },
            output_schema=standard_output,
            handler=_filesystem_script_handler(FILESYSTEM_DELETE_PATH_TOOL, "delete_path.py", script_root=script_root),
            risk_level=AgentToolRiskLevel.HIGH,
            requires_confirmation=True,
            allowed_source_types=common_source_types,
            candidate_profile=AgentToolCandidateProfile(
                categories=frozenset({"filesystem_delete", "filesystem_operation"}),
                keywords=frozenset({"删除文件", "删除目录", "delete file", "remove file"}),
                examples=("删除这个临时文件",),
            ),
        ),
        AgentToolDefinition(
            name=FILESYSTEM_MAKE_DIR_TOOL,
            description="Create a user-provided local directory after confirmation.",
            input_schema={
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": path_property,
                    "exist_ok": {"type": "boolean", "default": True},
                },
                "additionalProperties": False,
            },
            output_schema=standard_output,
            handler=_filesystem_script_handler(FILESYSTEM_MAKE_DIR_TOOL, "make_dir.py", script_root=script_root),
            risk_level=AgentToolRiskLevel.HIGH,
            requires_confirmation=True,
            allowed_source_types=common_source_types,
            candidate_profile=AgentToolCandidateProfile(
                categories=frozenset({"filesystem_make_dir", "filesystem_operation"}),
                keywords=frozenset({"创建目录", "新建文件夹", "make dir", "mkdir"}),
                examples=("创建一个简历备份目录",),
            ),
        ),
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
        dbx_schema = dbx_tool_input_schema(tool_name)
        is_dbx_tool = dbx_schema is not None
        qq_mail_schema = qq_mail_tool_input_schema(tool_name)
        is_qq_mail_tool = qq_mail_schema is not None
        metadata = _mcp_tool_metadata(client, tool_name)
        definitions.append(
            AgentToolDefinition(
                name=f"mcp.{tool_name}",
                description=(
                    dbx_tool_description(tool_name)
                    if is_dbx_tool
                    else (qq_mail_tool_description(tool_name) if is_qq_mail_tool else str(metadata.get("description") or f"Call MCP Gateway tool: {tool_name}."))
                ),
                input_schema=qq_mail_schema or dbx_schema or dict(metadata.get("inputSchema") or metadata.get("input_schema") or {"type": "object", "additionalProperties": True}),
                output_schema={"type": "object", "required": ["tool_name", "ok"]},
                handler=_mcp_handler(client, tool_name),
                risk_level=_mcp_risk_level(policy, tool_name),
                requires_confirmation=policy.requires_confirmation(tool_name),
                allowed_source_types=frozenset({"agent_chat", "mcp_visible_page", "application"}),
            )
        )
    return definitions


def _mcp_tool_metadata(client: Any, tool_name: str) -> dict[str, Any]:
    getter = getattr(client, "tool_metadata", None)
    if not callable(getter):
        return {}
    try:
        metadata = getter(tool_name)
    except Exception:  # pragma: no cover - third-party gateway implementations vary.
        logger.warning("MCP tool metadata lookup failed", extra={"mcp_tool_name": tool_name})
        return {}
    return dict(metadata or {}) if isinstance(metadata, dict) else {}


def create_skill_lazy_agent_tool_definitions() -> list[AgentToolDefinition]:
    standard_output = {"type": "object", "required": ["tool_name", "ok", "result"]}
    return [
        AgentToolDefinition(
            name=SKILL_LIST_TOOL,
            description=(
                "List active Agent Skills as a compact capability index. Use this first when the model needs to "
                "discover which high-level Skill can handle the user's request without loading every SKILL.md body."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": ["string", "null"], "description": "Optional user task or keywords used to rank Skills."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
                    "offset": {"type": "integer", "minimum": 0, "default": 0},
                },
                "additionalProperties": False,
            },
            output_schema=standard_output,
            handler=lambda session, **arguments: _skill_list(session, **arguments),
            risk_level=AgentToolRiskLevel.LOW,
            requires_confirmation=False,
            allowed_source_types=frozenset({"agent_chat", "skill_recall", "long_term_memory"}),
            candidate_profile=AgentToolCandidateProfile(
                categories=frozenset({"skill_lazy_loading", "skill_discovery"}),
                keywords=frozenset({"skill", "技能", "能力", "有哪些能力", "找一下能力"}),
                examples=("先看一下有哪些 Skill 能处理公众号文章",),
            ),
        ),
        AgentToolDefinition(
            name=SKILL_LIST_ACTIONS_TOOL,
            description=(
                "List the machine-callable actions for one selected Skill, including action descriptions and input schemas. "
                "Use this after skill_list and before skill_call."
            ),
            input_schema={
                "type": "object",
                "required": ["skill"],
                "properties": {"skill": {"type": "string", "description": "Skill id, name, or title."}},
                "additionalProperties": False,
            },
            output_schema=standard_output,
            handler=lambda session, **arguments: _skill_list_actions(session, **arguments),
            risk_level=AgentToolRiskLevel.LOW,
            requires_confirmation=False,
            allowed_source_types=frozenset({"agent_chat", "skill_recall", "long_term_memory"}),
            candidate_profile=AgentToolCandidateProfile(
                categories=frozenset({"skill_lazy_loading", "skill_action_discovery"}),
                keywords=frozenset({"skill action", "动作", "参数", "action schema"}),
                examples=("列出 article-fetcher 这个 Skill 能调用哪些动作",),
            ),
        ),
        AgentToolDefinition(
            name=SKILL_READ_TOOL,
            description=(
                "Read one selected SKILL.md body when the compact index is not enough. This is the second progressive "
                "loading layer for detailed workflow instructions and references."
            ),
            input_schema={
                "type": "object",
                "required": ["skill"],
                "properties": {
                    "skill": {"type": "string", "description": "Skill id, name, or title."},
                    "include_frontmatter": {"type": "boolean", "default": False},
                    "max_chars": {"type": "integer", "minimum": 1, "maximum": 50000, "default": 16000},
                },
                "additionalProperties": False,
            },
            output_schema=standard_output,
            handler=lambda session, **arguments: _skill_read(session, **arguments),
            risk_level=AgentToolRiskLevel.LOW,
            requires_confirmation=False,
            allowed_source_types=frozenset({"agent_chat", "skill_recall", "long_term_memory"}),
            candidate_profile=AgentToolCandidateProfile(
                categories=frozenset({"skill_lazy_loading", "skill_read"}),
                keywords=frozenset({"读取skill", "skill全文", "skill.md", "技能说明"}),
                examples=("读取简历优化 Skill 的完整说明",),
            ),
        ),
        AgentToolDefinition(
            name=SKILL_CALL_TOOL,
            description=(
                "Execute one action from a selected Skill package script after confirmation. The runtime resolves the action "
                "from Skill metadata/action_map and runs only scripts inside that imported Skill package."
            ),
            input_schema={
                "type": "object",
                "required": ["skill"],
                "properties": {
                    "skill": {"type": "string", "description": "Skill id, name, or title."},
                    "action": {"type": ["string", "null"], "description": "Action name from skill_list_actions."},
                    "arguments": {"type": "object", "additionalProperties": True, "default": {}},
                    "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 120, "default": 30},
                },
                "additionalProperties": False,
            },
            output_schema=standard_output,
            handler=lambda session, **arguments: _skill_call(session, **arguments),
            risk_level=AgentToolRiskLevel.HIGH,
            requires_confirmation=True,
            allowed_source_types=frozenset({"agent_chat", "skill_recall", "long_term_memory"}),
            candidate_profile=AgentToolCandidateProfile(
                categories=frozenset({"skill_action_execution", "skill_lazy_loading"}),
                keywords=frozenset({"执行skill", "skill_call", "调用动作", "运行技能"}),
                examples=("用 article-fetcher 的 fetch_article 动作读取这篇文章",),
            ),
        ),
    ]


def _skill_repository(session: Any) -> Any:
    from app.agent_runtime.memory.skill_repository import AgentSkillRepository
    from app.domains.agent_memory.repository import AgentMemoryRepository

    return AgentSkillRepository(AgentMemoryRepository(session))


def _active_skill_documents(session: Any, *, limit: int = 500) -> list[Any]:
    if session is None:
        return []
    from app.domains.agent_memory.models import AgentSkillStatus

    repository = _skill_repository(session)
    documents: list[Any] = []
    for skill in repository.list_skills(status=AgentSkillStatus.ACTIVE, limit=limit):
        try:
            documents.append(repository.read_skill(skill.id))
        except (OSError, ValueError):
            continue
    return documents


def _skill_list(
    session: Any,
    *,
    query: str | None = None,
    limit: int | str | None = 20,
    offset: int | str | None = 0,
) -> dict[str, Any]:
    from app.agent_runtime.memory.skill_candidate_selector import select_skill_candidates
    from app.agent_runtime.memory.skill_summary_index import build_skill_summary_card

    cards = [build_skill_summary_card(document) for document in _active_skill_documents(session)]
    cleaned_query = _non_empty_str(query) or ""
    if cleaned_query:
        candidates = select_skill_candidates(cleaned_query, cards, limit=len(cards))
        ranked_cards = [candidate.card for candidate in candidates]
    else:
        ranked_cards = sorted(cards, key=lambda card: (not card.pinned, card.name))

    page_limit = _bounded_int(limit, default=20, minimum=1, maximum=100)
    page_offset = _bounded_int(offset, default=0, minimum=0, maximum=10_000)
    page = ranked_cards[page_offset : page_offset + page_limit]
    return {
        "tool_name": SKILL_LIST_TOOL,
        "ok": True,
        "error": None,
        "result": {
            "query": cleaned_query,
            "total": len(ranked_cards),
            "offset": page_offset,
            "limit": page_limit,
            "returned": len(page),
            "has_more": page_offset + page_limit < len(ranked_cards),
            "skills": [_skill_card_payload(card) for card in page],
        },
    }


def _skill_card_payload(card: Any) -> dict[str, Any]:
    metadata = card.to_metadata()
    return {
        "skill_id": metadata["skill_id"],
        "name": metadata["name"],
        "title": metadata["title"],
        "description": metadata["description"],
        "category": metadata["category"],
        "when_to_use": metadata["when_to_use"],
        "source_types": metadata["source_types"],
        "risk_level": metadata["risk_level"],
        "auto_load_enabled": metadata["auto_load_enabled"],
        "actions_count": metadata.get("actions_count", 0),
        "actions_preview": metadata.get("actions_preview", []),
        "summary_text": metadata["summary_text"],
    }


def _skill_list_actions(session: Any, *, skill: str) -> dict[str, Any]:
    from app.agent_runtime.memory.skill_action_catalog import action_details_from_metadata

    document = _resolve_skill_document(session, skill)
    if document is None:
        return _skill_tool_failure(SKILL_LIST_ACTIONS_TOOL, "SKILL_NOT_FOUND", message=f"Skill not found: {skill}")
    metadata = document.skill.metadata_json or {}
    actions = action_details_from_metadata(metadata)
    return {
        "tool_name": SKILL_LIST_ACTIONS_TOOL,
        "ok": True,
        "error": None,
        "result": {
            "skill_id": document.skill.id,
            "name": document.skill.name,
            "title": document.skill.title,
            "actions": actions,
            "action_map": dict(metadata.get("action_map") or {}),
            "action_schemas_file": metadata.get("action_schemas_file"),
        },
    }


def _skill_read(
    session: Any,
    *,
    skill: str,
    include_frontmatter: bool = False,
    max_chars: int | str | None = 16000,
) -> dict[str, Any]:
    document = _resolve_skill_document(session, skill)
    if document is None:
        return _skill_tool_failure(SKILL_READ_TOOL, "SKILL_NOT_FOUND", message=f"Skill not found: {skill}")
    content = document.content if include_frontmatter else _strip_frontmatter(document.content)
    bounded_chars = _bounded_int(max_chars, default=16000, minimum=1, maximum=50000)
    truncated = len(content) > bounded_chars
    return {
        "tool_name": SKILL_READ_TOOL,
        "ok": True,
        "error": None,
        "result": {
            "skill_id": document.skill.id,
            "name": document.skill.name,
            "title": document.skill.title,
            "version_hash": document.version_hash,
            "content": content[:bounded_chars],
            "content_chars": min(len(content), bounded_chars),
            "truncated": truncated,
        },
    }


def _skill_call(
    session: Any,
    *,
    skill: str,
    action: str | None = None,
    arguments: dict[str, Any] | None = None,
    timeout_seconds: int | str | None = 30,
) -> dict[str, Any]:
    from app.agent_runtime.memory.skill_action_catalog import (
        normalize_action_name,
        resolve_skill_action,
        resolve_skill_script_path,
        skill_action_command_arguments,
        validate_skill_action_arguments,
    )

    document = _resolve_skill_document(session, skill)
    if document is None:
        return _skill_tool_failure(SKILL_CALL_TOOL, "SKILL_NOT_FOUND", message=f"Skill not found: {skill}")
    metadata = document.skill.metadata_json or {}
    if bool(metadata.get("disable_model_invocation")):
        return _skill_tool_failure(SKILL_CALL_TOOL, "SKILL_MODEL_INVOCATION_DISABLED", message="This Skill disables model invocation.")
    if arguments is not None and not isinstance(arguments, dict):
        return _skill_tool_failure(SKILL_CALL_TOOL, "SKILL_ACTION_ARGUMENTS_INVALID", message="arguments must be an object.")

    action_detail = resolve_skill_action(metadata, action)
    if action_detail is None:
        return _skill_tool_failure(
            SKILL_CALL_TOOL,
            "SKILL_ACTION_NOT_FOUND",
            message="Skill action was not found or is ambiguous.",
            result={"available_actions": list(metadata.get("actions") or [])},
        )
    action_name = normalize_action_name(str(action_detail.get("action") or ""))
    policy_error = _skill_action_policy_error(metadata, action_name)
    if policy_error:
        return _skill_tool_failure(SKILL_CALL_TOOL, "SKILL_ACTION_NOT_EXECUTABLE", message=policy_error)

    cleaned_arguments = dict(arguments or {})
    validation_error = validate_skill_action_arguments(action_detail, cleaned_arguments)
    if validation_error:
        return _skill_tool_failure(SKILL_CALL_TOOL, "SKILL_ACTION_ARGUMENTS_INVALID", message=validation_error)

    skill_file = Path(str(document.skill.file_path or ""))
    script_file = resolve_skill_script_path(skill_file, action_detail)
    if script_file is None:
        return _skill_tool_failure(
            SKILL_CALL_TOOL,
            "SKILL_SCRIPT_NOT_FOUND",
            message="Skill action does not resolve to an executable script inside the Skill package.",
        )

    command_args = skill_action_command_arguments(cleaned_arguments)
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("PYTHONUTF8", "1")
    env["OFFERMASTER_SKILL_ROOT"] = str(skill_file.parent)
    env["OFFERMASTER_SKILL_NAME"] = document.skill.name
    env["OFFERMASTER_SKILL_ACTION"] = action_name
    timeout = _bounded_int(timeout_seconds, default=30, minimum=1, maximum=120)
    try:
        completed = subprocess.run(
            [sys.executable, str(script_file), *command_args],
            cwd=str(skill_file.parent),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
            shell=False,
        )
    except subprocess.TimeoutExpired:
        _record_skill_call_event(session, document.skill.id, success=False, action=action_name, error="timeout")
        return _skill_tool_failure(SKILL_CALL_TOOL, "SKILL_ACTION_TIMEOUT", message=f"Skill action timed out after {timeout}s.")
    except OSError as exc:
        _record_skill_call_event(session, document.skill.id, success=False, action=action_name, error=str(exc))
        return _skill_tool_failure(SKILL_CALL_TOOL, "SKILL_ACTION_EXECUTION_FAILED", message=str(exc))

    ok = completed.returncode == 0
    error = None if ok else (completed.stderr.strip() or f"Skill action exited with code {completed.returncode}")
    _record_skill_call_event(session, document.skill.id, success=ok, action=action_name, error=error)
    return {
        "tool_name": SKILL_CALL_TOOL,
        "ok": ok,
        "error": error,
        "result": {
            "skill_id": document.skill.id,
            "name": document.skill.name,
            "title": document.skill.title,
            "action": action_name,
            "arguments": cleaned_arguments,
            "stdout": completed.stdout or "",
            "stderr": completed.stderr or "",
            "return_code": completed.returncode,
            "script_path": str(script_file),
            "skill_root": str(skill_file.parent),
        },
    }


def _resolve_skill_document(session: Any, skill: str) -> Any | None:
    requested = str(skill or "").strip()
    if not requested or session is None:
        return None
    normalized = _normalize_skill_lookup_key(requested)
    for document in _active_skill_documents(session):
        if requested == document.skill.id:
            return document
        candidates = {
            _normalize_skill_lookup_key(document.skill.name),
            _normalize_skill_lookup_key(document.skill.title),
        }
        if normalized in candidates:
            return document
    return None


def _normalize_skill_lookup_key(value: str) -> str:
    lowered = str(value or "").strip().lower()
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", lowered).strip("-")


def _strip_frontmatter(content: str) -> str:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return content
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return "\n".join(lines[index + 1 :]).lstrip()
    return content


def _skill_action_policy_error(metadata: dict[str, Any], action_name: str) -> str | None:
    allowed = {str(item).strip().replace("-", "_") for item in metadata.get("execute_allowed_actions") or [] if str(item).strip()}
    denied = {str(item).strip().replace("-", "_") for item in metadata.get("execute_denied_actions") or [] if str(item).strip()}
    if action_name in denied:
        return f"Skill action {action_name} is denied by executeDeny."
    if allowed and action_name not in allowed:
        return f"Skill action {action_name} is not listed in executeAllow."
    return None


def _record_skill_call_event(session: Any, skill_id: str, *, success: bool, action: str, error: str | None = None) -> None:
    if session is None:
        return
    try:
        repository = _skill_repository(session)
        repository.record_runtime_event(
            skill_id,
            event="tool_succeeded" if success else "tool_failed",
            evidence={"tool_name": SKILL_CALL_TOOL, "action": action, "error": error},
        )
    except Exception:
        # Skill execution has already completed; telemetry failures must not
        # turn a successful user-visible action into a failed tool call.
        return


def _skill_tool_failure(
    tool_name: str,
    error: str,
    *,
    message: str,
    result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {"message": message, **(result or {})}
    return {"tool_name": tool_name, "ok": False, "error": error, "result": payload}


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


def _sync_mail_application(session: Any, **arguments: Any) -> dict[str, Any]:
    tool_name = APPLICATION_SYNC_MAIL_TOOL
    if session is None:
        return {
            "tool_name": tool_name,
            "ok": False,
            "error": "DATABASE_SESSION_REQUIRED",
            "result": {"message": "A database session is required to synchronize a mail event."},
        }
    from app.domains.applications.schemas import ApplicationMailSyncCreate
    from app.domains.applications.repository import ApplicationEventRepository, ApplicationRepository
    from app.domains.applications.service import ApplicationService
    from app.domains.jobs.repository import CompanyRepository, JobRepository
    from app.domains.jobs.service import JobService

    try:
        command = ApplicationMailSyncCreate.model_validate(arguments)
        result = ApplicationService(
            applications=ApplicationRepository(session),
            events=ApplicationEventRepository(session),
            jobs=JobService(CompanyRepository(session), JobRepository(session)),
        ).sync_mail_application(command)
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.exception("Failed to synchronize mail event to application board", extra={"tool_name": tool_name})
        return {
            "tool_name": tool_name,
            "ok": False,
            "error": type(exc).__name__,
            "result": {"message": str(exc)},
        }

    logger.info(
        "Mail sync tool completed",
        extra={
            "tool_name": tool_name,
            "application_id": result.application.id,
            "event_id": result.event.id,
            "status": _value(result.application.status),
            "created_application": result.created_application,
        },
    )
    return {
        "tool_name": tool_name,
        "ok": True,
        "result": {
            "application_id": result.application.id,
            "company_name": result.application.job.company.name,
            "job_title": result.application.job.title,
            "status": _value(result.application.status),
            "event_id": result.event.id,
            "event_type": result.event.event_type,
            "source_sent_at": (result.event.event_metadata or {}).get("source_sent_at"),
            "scheduled_at": result.event.scheduled_at.isoformat() if result.event.scheduled_at else None,
            "deadline_at": result.event.deadline_at.isoformat() if result.event.deadline_at else None,
            "deadline_offset_hours": (result.event.event_metadata or {}).get("deadline_offset_hours"),
            "timing_source": (result.event.event_metadata or {}).get("timing_source", "none"),
            "timing_note": (result.event.event_metadata or {}).get("timing_note"),
            "created_application": result.created_application,
            "next_action": "application_visible_on_board",
        },
    }


def _find_applications_for_mail(session: Any, **arguments: Any) -> dict[str, Any]:
    if session is None:
        return {
            "tool_name": APPLICATION_FIND_FOR_MAIL_TOOL,
            "ok": False,
            "error": "DATABASE_SESSION_REQUIRED",
            "result": {"message": "A database session is required for application matching."},
        }
    from app.domains.applications.repository import ApplicationRepository

    company_name = _non_empty_str(arguments.get("company_name"))
    job_title = _non_empty_str(arguments.get("job_title"))
    if not company_name and not job_title:
        return {
            "tool_name": APPLICATION_FIND_FOR_MAIL_TOOL,
            "ok": False,
            "error": "MATCH_QUERY_REQUIRED",
            "result": {"message": "company_name or job_title is required."},
        }
    limit = _bounded_int(arguments.get("limit"), default=10, minimum=1, maximum=20)
    applications = ApplicationRepository(session).list_for_mail_match(
        company_name=company_name,
        job_title=job_title,
        limit=limit,
    )
    return {
        "tool_name": APPLICATION_FIND_FOR_MAIL_TOOL,
        "ok": True,
        "result": {
            "match_count": len(applications),
            "matches": [
                {
                    "application_id": item.id,
                    "company_name": item.job.company.name,
                    "job_title": item.job.title,
                    "status": _value(item.status),
                    "job_id": item.job_id,
                }
                for item in applications
            ],
            "next_action": "choose_application_or_request_user_match" if applications else "request_user_match",
        },
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


def _database_company_search(
    session: Any,
    *,
    company_names: Any = None,
    keyword: Any = None,
    limit: int | str | None = 10,
) -> dict[str, Any]:
    if session is None:
        return _database_session_error(DATABASE_COMPANY_SEARCH_TOOL)

    from sqlalchemy import or_, select

    from app.domains.jobs.models import Company, Job, JobLead, RecruitingSignal

    result_limit = _bounded_int(limit, default=10, minimum=1, maximum=50)
    queries = _database_query_names(company_names, keyword)
    if not queries:
        return _database_input_error(
            DATABASE_COMPANY_SEARCH_TOOL,
            "company_names 或 keyword 至少提供一个非空查询条件。",
        )

    companies: list[dict[str, Any]] = []
    for query_name in queries[:20]:
        pattern = f"%{query_name}%"
        formal_companies = list(
            session.scalars(
                select(Company)
                .where(
                    or_(
                        Company.name.ilike(pattern),
                        Company.normalized_name.ilike(pattern),
                    )
                )
                .order_by(Company.name.asc())
                .limit(result_limit)
            ).all()
        )
        jobs = list(
            session.scalars(
                select(Job)
                .join(Job.company)
                .where(
                    or_(
                        Company.name.ilike(pattern),
                        Company.normalized_name.ilike(pattern),
                    )
                )
                .order_by(Job.updated_at.desc(), Job.created_at.desc())
                .limit(result_limit)
            ).all()
        )
        job_leads = list(
            session.scalars(
                select(JobLead)
                .where(JobLead.company_name.ilike(pattern))
                .order_by(JobLead.updated_at.desc(), JobLead.created_at.desc())
                .limit(result_limit)
            ).all()
        )
        recruiting_signals = list(
            session.scalars(
                select(RecruitingSignal)
                .where(
                    or_(
                        RecruitingSignal.company_name.ilike(pattern),
                        RecruitingSignal.normalized_company_name.ilike(pattern),
                    )
                )
                .order_by(RecruitingSignal.updated_at.desc(), RecruitingSignal.created_at.desc())
                .limit(result_limit)
            ).all()
        )

        evidence: list[dict[str, Any]] = []
        if formal_companies:
            evidence.append(
                {
                    "source": "正式企业表",
                    "count": len(formal_companies),
                    "record_ids": [company.id for company in formal_companies],
                }
            )
        if jobs:
            evidence.append(
                {
                    "source": "正式岗位表",
                    "count": len(jobs),
                    "record_ids": [job.id for job in jobs],
                }
            )
        if job_leads:
            evidence.append(
                {
                    "source": "岗位线索表",
                    "count": len(job_leads),
                    "record_ids": [lead.id for lead in job_leads],
                }
            )
        if recruiting_signals:
            evidence.append(
                {
                    "source": "招聘信号表",
                    "count": len(recruiting_signals),
                    "record_ids": [signal.id for signal in recruiting_signals],
                }
            )

        companies.append(
            {
                "query_name": query_name,
                "exists": bool(formal_companies or jobs or job_leads or recruiting_signals),
                "formal_company_count": len(formal_companies),
                "job_count": len(jobs),
                "job_lead_count": len(job_leads),
                "recruiting_signal_count": len(recruiting_signals),
                "formal_companies": [_database_company_summary(company) for company in formal_companies],
                "evidence": evidence,
            }
        )

    return {
        "tool_name": DATABASE_COMPANY_SEARCH_TOOL,
        "ok": True,
        "result": {
            "queries": queries,
            "companies": companies,
            "matched_count": sum(1 for company in companies if company["exists"]),
        },
    }


def _database_company_list(
    session: Any,
    *,
    keyword: Any = None,
    limit: int | str | None = 20,
) -> dict[str, Any]:
    if session is None:
        return _database_session_error(DATABASE_COMPANY_LIST_TOOL)

    from sqlalchemy import func, select

    from app.domains.jobs.models import Company, Job, JobLead, RecruitingSignal

    result_limit = _bounded_int(limit, default=20, minimum=1, maximum=100)
    keyword_text = _non_empty_str(keyword)
    keyword_casefold = keyword_text.casefold() if keyword_text else None
    company_map: dict[str, dict[str, Any]] = {}

    def ensure_company(
        name: Any,
        *,
        has_profile: bool = False,
        tier_rank: int = 2,
        sort_key: Any = None,
    ) -> dict[str, Any] | None:
        company_name = _non_empty_str(name)
        if not company_name:
            return None
        key = company_name.casefold()
        resolved_sort_key = _non_empty_str(sort_key) or company_name
        item = company_map.setdefault(
            key,
            {
                "company_name": company_name,
                "has_profile": False,
                "job_count": 0,
                "lead_count": 0,
                "signal_count": 0,
                "tier_rank": tier_rank,
                "sort_key": resolved_sort_key,
            },
        )
        item["has_profile"] = bool(item["has_profile"] or has_profile)
        if tier_rank < int(item["tier_rank"]):
            item["sort_key"] = resolved_sort_key
        item["tier_rank"] = min(int(item["tier_rank"]), tier_rank)
        if has_profile:
            item["company_name"] = company_name
            item["sort_key"] = resolved_sort_key
        return item

    for name, normalized_name, job_count in session.execute(
        select(Company.name, Company.normalized_name, func.count(Job.id))
        .outerjoin(Job, Job.company_id == Company.id)
        .group_by(Company.id, Company.name, Company.normalized_name)
    ).all():
        item = ensure_company(name, has_profile=True, tier_rank=0, sort_key=normalized_name)
        if item is not None:
            item["job_count"] += int(job_count or 0)

    for name, lead_count in session.execute(
        select(JobLead.company_name, func.count(JobLead.id)).group_by(JobLead.company_name)
    ).all():
        item = ensure_company(name, tier_rank=1)
        if item is not None:
            item["lead_count"] += int(lead_count or 0)

    for name, signal_count in session.execute(
        select(RecruitingSignal.company_name, func.count(RecruitingSignal.id)).group_by(RecruitingSignal.company_name)
    ).all():
        item = ensure_company(name, tier_rank=2)
        if item is not None:
            item["signal_count"] += int(signal_count or 0)

    companies = []
    for item in sorted(company_map.values(), key=lambda row: (int(row["tier_rank"]), str(row["sort_key"]).casefold())):
        if keyword_casefold and keyword_casefold not in str(item["company_name"]).casefold():
            continue
        companies.append(
            {
                "company_name": str(item["company_name"]),
                "has_profile": bool(item["has_profile"]),
                "job_count": int(item["job_count"]),
                "lead_count": int(item["lead_count"]),
                "signal_count": int(item["signal_count"]),
                "total_record_count": int(item["job_count"] + item["lead_count"] + item["signal_count"]),
            }
        )

    return {
        "tool_name": DATABASE_COMPANY_LIST_TOOL,
        "ok": True,
        "result": {
            "keyword": keyword_text,
            "total_count": len(companies),
            "count": min(result_limit, len(companies)),
            "companies": companies[:result_limit],
            "all_companies": companies,
        },
    }


def _database_company_profile(
    session: Any,
    *,
    company_name: str,
    limit: int | str | None = 10,
) -> dict[str, Any]:
    if session is None:
        return _database_session_error(DATABASE_COMPANY_PROFILE_TOOL)

    from sqlalchemy import or_, select

    from app.domains.jobs.models import Company, Job, JobLead, RecruitingSignal

    query_name = _non_empty_str(company_name)
    if not query_name:
        return _database_input_error(DATABASE_COMPANY_PROFILE_TOOL, "company_name 不能为空。")
    result_limit = _bounded_int(limit, default=10, minimum=1, maximum=50)
    pattern = f"%{query_name}%"
    formal_companies = list(
        session.scalars(
            select(Company)
            .where(
                or_(
                    Company.name.ilike(pattern),
                    Company.normalized_name.ilike(pattern),
                )
            )
            .order_by(Company.name.asc())
            .limit(result_limit)
        ).all()
    )
    jobs = list(
        session.scalars(
            select(Job)
            .join(Job.company)
            .where(
                or_(
                    Company.name.ilike(pattern),
                    Company.normalized_name.ilike(pattern),
                )
            )
            .order_by(Job.updated_at.desc(), Job.created_at.desc())
            .limit(result_limit)
        ).all()
    )
    job_leads = list(
        session.scalars(
            select(JobLead)
            .where(JobLead.company_name.ilike(pattern))
            .order_by(JobLead.updated_at.desc(), JobLead.created_at.desc())
            .limit(result_limit)
        ).all()
    )
    recruiting_signals = list(
        session.scalars(
            select(RecruitingSignal)
            .where(
                or_(
                    RecruitingSignal.company_name.ilike(pattern),
                    RecruitingSignal.normalized_company_name.ilike(pattern),
                )
            )
            .order_by(RecruitingSignal.updated_at.desc(), RecruitingSignal.created_at.desc())
            .limit(result_limit)
        ).all()
    )
    return {
        "tool_name": DATABASE_COMPANY_PROFILE_TOOL,
        "ok": True,
        "result": {
            "company_name": query_name,
            "exists": bool(formal_companies or jobs or job_leads or recruiting_signals),
            "formal_companies": [_database_company_summary(company) for company in formal_companies],
            "jobs": [_database_job_summary(job) for job in jobs],
            "job_leads": [_database_job_lead_summary(lead) for lead in job_leads],
            "recruiting_signals": [_database_recruiting_signal_summary(signal) for signal in recruiting_signals],
        },
    }


def _database_job_search(
    session: Any,
    *,
    keyword: Any = None,
    company_name: Any = None,
    title: Any = None,
    city: Any = None,
    job_type: Any = None,
    status: Any = None,
    limit: int | str | None = 20,
) -> dict[str, Any]:
    if session is None:
        return _database_session_error(DATABASE_JOB_SEARCH_TOOL)

    from sqlalchemy import or_, select

    from app.domains.jobs.models import Company, Job

    result_limit = _bounded_int(limit, default=20, minimum=1, maximum=100)
    statement = select(Job).join(Job.company).order_by(Job.updated_at.desc(), Job.created_at.desc())
    filters = []
    for field, value in (
        (Company.name, company_name),
        (Job.title, title),
        (Job.city, city),
        (Job.job_type, job_type),
    ):
        text = _non_empty_str(value)
        if text:
            filters.append(field.ilike(f"%{text}%"))
    keyword_text = _non_empty_str(keyword)
    if keyword_text:
        pattern = f"%{keyword_text}%"
        filters.append(
            or_(
                Company.name.ilike(pattern),
                Job.title.ilike(pattern),
                Job.jd_text.ilike(pattern),
                Job.job_type.ilike(pattern),
            )
        )
    status_text = _non_empty_str(status)
    if status_text:
        filters.append(Job.status == status_text)
    if filters:
        statement = statement.where(*filters)
    jobs = list(session.scalars(statement.limit(result_limit)).all())
    return {
        "tool_name": DATABASE_JOB_SEARCH_TOOL,
        "ok": True,
        "result": {
            "count": len(jobs),
            "jobs": [_database_job_summary(job) for job in jobs],
        },
    }


def _database_source_search(
    session: Any,
    *,
    keyword: Any = None,
    source_type: Any = None,
    enabled: bool | str | None = None,
    limit: int | str | None = 20,
) -> dict[str, Any]:
    if session is None:
        return _database_session_error(DATABASE_SOURCE_SEARCH_TOOL)

    from sqlalchemy import func, or_, select

    from app.domains.jobs.models import JobLead, JobSource, RecruitingSignal

    result_limit = _bounded_int(limit, default=20, minimum=1, maximum=100)
    statement = select(JobSource).order_by(JobSource.updated_at.desc(), JobSource.created_at.desc())
    filters = []
    keyword_text = _non_empty_str(keyword)
    if keyword_text:
        pattern = f"%{keyword_text}%"
        filters.append(or_(JobSource.name.ilike(pattern), JobSource.notes.ilike(pattern), JobSource.entry_url.ilike(pattern)))
    source_type_text = _non_empty_str(source_type)
    if source_type_text:
        filters.append(JobSource.source_type == source_type_text)
    enabled_value = _optional_bool(enabled)
    if enabled_value is not None:
        filters.append(JobSource.enabled.is_(enabled_value))
    if filters:
        statement = statement.where(*filters)
    sources = list(session.scalars(statement.limit(result_limit)).all())
    rows = []
    for source in sources:
        rows.append(
            {
                "id": source.id,
                "name": source.name,
                "source_type": _value(source.source_type),
                "entry_url": source.entry_url,
                "enabled": bool(source.enabled),
                "trust_level": _value(source.trust_level),
                "fetch_mode": _value(source.fetch_mode),
                "notes": source.notes,
                "job_lead_count": int(session.scalar(select(func.count(JobLead.id)).where(JobLead.source_id == source.id)) or 0),
                "recruiting_signal_count": int(
                    session.scalar(select(func.count(RecruitingSignal.id)).where(RecruitingSignal.source_id == source.id)) or 0
                ),
            }
        )
    return {
        "tool_name": DATABASE_SOURCE_SEARCH_TOOL,
        "ok": True,
        "result": {"count": len(rows), "sources": rows},
    }


def _database_company_update(
    session: Any,
    *,
    company_id: Any = None,
    company_name: Any = None,
    name: Any = None,
    website_url: Any = None,
    industry: Any = None,
    city: Any = None,
    country: Any = None,
) -> dict[str, Any]:
    if session is None:
        return _database_session_error(DATABASE_COMPANY_UPDATE_TOOL)

    from sqlalchemy import select

    from app.domains.jobs.models import Company

    company = session.get(Company, _non_empty_str(company_id)) if _non_empty_str(company_id) else None
    if company is None:
        lookup_name = _non_empty_str(company_name)
        if lookup_name:
            company = session.scalar(
                select(Company).where(
                    (Company.name == lookup_name) | (Company.normalized_name == _database_normalize_name(lookup_name))
                )
            )
    if company is None:
        return _database_input_error(DATABASE_COMPANY_UPDATE_TOOL, "找不到要更新的公司，请提供有效的 company_id 或 company_name。")

    updates: dict[str, str] = {}
    for field_name, value in (
        ("website_url", website_url),
        ("industry", industry),
        ("city", city),
        ("country", country),
    ):
        text = _optional_text(value)
        if text is not None:
            updates[field_name] = text

    new_name = _optional_text(name)
    if new_name is not None:
        normalized_name = _database_normalize_name(new_name)
        duplicate = session.scalar(
            select(Company).where(
                Company.normalized_name == normalized_name,
                Company.id != company.id,
            )
        )
        if duplicate is not None:
            return _database_input_error(DATABASE_COMPANY_UPDATE_TOOL, "更新后的公司名称已存在，未修改任何数据。")
        updates["name"] = new_name
        updates["normalized_name"] = normalized_name

    if not updates:
        return _database_input_error(DATABASE_COMPANY_UPDATE_TOOL, "至少提供一个要修改的公司字段。")
    for field_name, value in updates.items():
        setattr(company, field_name, value)
    session.flush()
    return {
        "tool_name": DATABASE_COMPANY_UPDATE_TOOL,
        "ok": True,
        "result": {
            "company": _database_company_summary(company),
            "updated_fields": [field_name for field_name in updates if field_name != "normalized_name"],
        },
    }


def _database_job_lead_delete(
    session: Any,
    *,
    lead_id: str,
    reason: Any = None,
) -> dict[str, Any]:
    if session is None:
        return _database_session_error(DATABASE_JOB_LEAD_DELETE_TOOL)

    from app.domains.jobs.models import JobLead, JobLeadStatus, utc_now

    resolved_lead_id = _non_empty_str(lead_id)
    if not resolved_lead_id:
        return _database_input_error(DATABASE_JOB_LEAD_DELETE_TOOL, "lead_id 不能为空。")
    lead = session.get(JobLead, resolved_lead_id)
    if lead is None:
        return _database_input_error(DATABASE_JOB_LEAD_DELETE_TOOL, "找不到要删除的岗位线索。")
    if _value(lead.verification_status) != JobLeadStatus.INVALID.value:
        reason_text = _optional_text(reason) or "用户确认将该岗位线索标记为无效。"
        audit_note = f"[database.job_lead_delete] {reason_text}"
        lead.verification_status = JobLeadStatus.INVALID
        lead.verification_notes = (
            f"{lead.verification_notes}\n{audit_note}".strip()
            if lead.verification_notes
            else audit_note
        )
        lead.updated_at = utc_now()
        session.flush()
        changed = True
    else:
        changed = False
    return {
        "tool_name": DATABASE_JOB_LEAD_DELETE_TOOL,
        "ok": True,
        "result": {
            "lead_id": lead.id,
            "company_name": lead.company_name,
            "title": lead.title,
            "deleted": True,
            "deletion_mode": "soft",
            "changed": changed,
            "verification_status": _value(lead.verification_status),
        },
    }


def _database_session_error(tool_name: str) -> dict[str, Any]:
    return {
        "tool_name": tool_name,
        "ok": False,
        "error": "DATABASE_SESSION_UNAVAILABLE",
        "result": {"message": "Database session is unavailable."},
    }


def _database_input_error(tool_name: str, message: str) -> dict[str, Any]:
    return {
        "tool_name": tool_name,
        "ok": False,
        "error": "DATABASE_INPUT_INVALID",
        "result": {"message": message},
    }


def _database_query_names(company_names: Any, keyword: Any) -> list[str]:
    values: list[Any] = []
    if isinstance(company_names, (list, tuple, set)):
        values.extend(company_names)
    elif company_names is not None:
        values.append(company_names)
    if not values:
        values.append(keyword)
    return list(dict.fromkeys(text for value in values if (text := _non_empty_str(value))))


def _database_normalize_name(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _database_company_summary(company: Any) -> dict[str, Any]:
    return {
        "id": company.id,
        "name": company.name,
        "normalized_name": company.normalized_name,
        "website_url": company.website_url,
        "industry": company.industry,
        "city": company.city,
        "country": company.country,
    }


def _database_job_summary(job: Any) -> dict[str, Any]:
    company = getattr(job, "company", None)
    return {
        "id": job.id,
        "company_id": job.company_id,
        "company_name": company.name if company is not None else None,
        "title": job.title,
        "city": job.city,
        "source": job.source,
        "source_job_id": job.source_job_id,
        "source_url": job.source_url,
        "job_type": job.job_type,
        "salary_text": job.salary_text,
        "skills": list(job.skills or []),
        "date_posted": _database_iso_value(job.date_posted),
        "status": _value(job.status),
    }


def _database_job_lead_summary(lead: Any) -> dict[str, Any]:
    source = getattr(lead, "source", None)
    return {
        "id": lead.id,
        "company_name": lead.company_name,
        "title": lead.title,
        "city": lead.city,
        "job_direction": lead.job_direction,
        "graduation_year": lead.graduation_year,
        "source_url": lead.source_url,
        "apply_url": lead.apply_url,
        "job_type": lead.job_type,
        "skills": list(lead.skills or []),
        "deadline": _database_iso_value(lead.deadline),
        "verification_status": _value(lead.verification_status),
        "trust_level": _value(lead.trust_level),
        "source_id": lead.source_id,
        "source_name": source.name if source is not None else None,
    }


def _database_recruiting_signal_summary(signal: Any) -> dict[str, Any]:
    source = getattr(signal, "source", None)
    return {
        "id": signal.id,
        "company_name": signal.company_name,
        "normalized_company_name": signal.normalized_company_name,
        "signal_type": _value(signal.signal_type),
        "graduation_year": signal.graduation_year,
        "source_url": signal.source_url,
        "original_source": signal.original_source,
        "confidence_score": _database_iso_value(signal.confidence_score),
        "trust_level": _value(signal.trust_level),
        "status": _value(signal.status),
        "source_id": signal.source_id,
        "source_name": source.name if source is not None else None,
    }


def _database_iso_value(value: Any) -> Any:
    return value.isoformat() if hasattr(value, "isoformat") else value


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "on"}:
        return True
    if lowered in {"false", "0", "no", "off"}:
        return False
    return None


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
    mode: str = "sources",
) -> dict[str, Any]:
    if str(mode or "sources").strip().lower() == "company_board_count":
        # Keep the conversational count identical to the Company Exhibition
        # page defaults. Legacy local company/job tables are intentionally not
        # consulted for this metric.
        company_board_filters = {
            "batch": "秋招",
            "target": "2027届",
        }
        try:
            page = offerio_provider_factory().list_company_openings(
                page=1,
                page_size=1,
                **company_board_filters,
            )
        except Exception as exc:
            logger.warning(
                "Company Exhibition total lookup failed",
                extra={"error_type": type(exc).__name__, **company_board_filters},
            )
            return {
                "tool_name": LOCAL_JOB_SOURCE_OVERVIEW_TOOL,
                "ok": False,
                "error": "COMPANY_BOARD_UNAVAILABLE",
                "result": {"message": "公司展览的开放岗位公司库暂时无法读取，请稍后重试。"},
            }
        total = getattr(page, "total", None)
        logger.info(
            "Company Exhibition total resolved",
            extra={"total": total, **company_board_filters},
        )
        if total is None:
            return {
                "tool_name": LOCAL_JOB_SOURCE_OVERVIEW_TOOL,
                "ok": False,
                "error": "COMPANY_BOARD_TOTAL_MISSING",
                "result": {"message": "公司展览数据源没有返回公司总数。"},
            }
        return {
            "tool_name": LOCAL_JOB_SOURCE_OVERVIEW_TOOL,
            "ok": True,
            "result": {
                "mode": "company_board_count",
                "company_board": {
                    "source": "offerio_company_openings",
                    "label": "公司展览 · 开放岗位公司库",
                    "company_count": int(total),
                },
            },
        }

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
        openings = provider.list_company_openings(
            page=1,
            page_size=1,
            batch="秋招",
            target="2027届",
        )
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


def _filesystem_script_handler(
    tool_name: str,
    script_name: str,
    *,
    script_root: str | Path | None = None,
) -> Callable[..., dict[str, Any]]:
    def handler(session: Any, **arguments: Any) -> dict[str, Any]:
        return _run_filesystem_skill_script(
            session,
            tool_name=tool_name,
            script_name=script_name,
            script_root=script_root,
            arguments=arguments,
        )

    return handler


def run_filesystem_skill_script(
    session: Any,
    *,
    tool_name: str,
    script_name: str,
    script_root: str | Path | None = None,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Shared entry point for filesystem Skill internals and legacy tool facades."""
    return _run_filesystem_skill_script(
        session,
        tool_name=tool_name,
        script_name=script_name,
        script_root=script_root,
        arguments=arguments,
    )


@dataclass(frozen=True)
class _ArtifactExportSource:
    run_id: str
    logical_path: str


class _ArtifactExportError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _export_artifact(
    _session: Any,
    *,
    artifact_uri: str,
    destination: str,
    overwrite: bool = False,
    sandbox_base_dir: str | Path | None = None,
) -> dict[str, Any]:
    try:
        source = _parse_artifact_sandbox_uri(artifact_uri)
        base_dir = _artifact_sandbox_base_dir(sandbox_base_dir)
        workspace_root = (base_dir / source.run_id).resolve(strict=False)
        if not _path_is_relative_to(workspace_root, base_dir):
            raise _ArtifactExportError("ARTIFACT_RUN_INVALID", "Sandbox run id resolves outside the sandbox base directory.")

        output_dir = (workspace_root / "output").resolve(strict=False)
        source_path = (workspace_root / source.logical_path).resolve(strict=False)
        # Only files under output/ are allowed to cross from sandbox storage into user storage.
        if not _path_is_relative_to(source_path, output_dir):
            raise _ArtifactExportError("ARTIFACT_SOURCE_NOT_EXPORTABLE", "Only sandbox output/ artifacts can be exported.")
        if not source_path.is_file():
            raise _ArtifactExportError("ARTIFACT_SOURCE_NOT_FOUND", "Sandbox artifact does not exist or is not a file.")

        destination_path = Path(_required_str(destination, "destination")).expanduser().resolve(strict=False)
        if destination_path.exists() and destination_path.is_dir():
            raise _ArtifactExportError("ARTIFACT_DESTINATION_IS_DIRECTORY", "Artifact export destination must be a file path.")
        existed_before = destination_path.exists()
        if existed_before and not overwrite:
            raise _ArtifactExportError("ARTIFACT_DESTINATION_EXISTS", "Destination already exists; set overwrite=true after confirmation.")

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
    except _ArtifactExportError as exc:
        return _artifact_tool_failure(exc.code, message=exc.message)
    except OSError as exc:
        return _artifact_tool_failure("ARTIFACT_EXPORT_FAILED", message=str(exc))
    except ValueError as exc:
        return _artifact_tool_failure("ARTIFACT_INPUT_INVALID", message=str(exc))

    return {
        "tool_name": ARTIFACT_EXPORT_TOOL,
        "ok": True,
        "error": None,
        "result": {
            "status": "exported",
            "source_uri": _artifact_sandbox_uri(source),
            "logical_path": source.logical_path,
            "destination_path": str(destination_path),
            "bytes_copied": destination_path.stat().st_size,
            "overwritten": existed_before,
        },
    }


def _parse_artifact_sandbox_uri(value: str) -> _ArtifactExportSource:
    raw = _required_str(value, "artifact_uri")
    parsed = urlparse(raw)
    if parsed.scheme != "artifact-sandbox" or not parsed.netloc or not parsed.path:
        raise _ArtifactExportError("ARTIFACT_URI_INVALID", "Artifact URI must look like artifact-sandbox://run-id/output/file.")
    run_id = unquote(parsed.netloc).strip()
    if not run_id or any(separator in run_id for separator in ("/", "\\")) or run_id in {".", ".."}:
        raise _ArtifactExportError("ARTIFACT_RUN_INVALID", "Artifact URI contains an invalid sandbox run id.")
    logical_path = _normalized_artifact_logical_path(unquote(parsed.path.lstrip("/")))
    return _ArtifactExportSource(run_id=run_id, logical_path=logical_path)


def _normalized_artifact_logical_path(value: str) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    if not raw or "\x00" in raw:
        raise _ArtifactExportError("ARTIFACT_SOURCE_INVALID", "Artifact logical path cannot be empty.")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise _ArtifactExportError("ARTIFACT_SOURCE_INVALID", "Artifact logical path must stay inside the sandbox.")
    logical_path = pure.as_posix()
    if not logical_path.startswith("output/"):
        raise _ArtifactExportError("ARTIFACT_SOURCE_NOT_EXPORTABLE", "Only sandbox output/ artifacts can be exported.")
    return logical_path


def _artifact_sandbox_base_dir(sandbox_base_dir: str | Path | None) -> Path:
    if sandbox_base_dir is not None:
        return Path(sandbox_base_dir).expanduser().resolve()
    from app.core.config import get_settings

    return (get_settings().exports_path.parent / "sdk-agent-runs").resolve()


def _artifact_sandbox_uri(source: _ArtifactExportSource) -> str:
    return f"artifact-sandbox://{source.run_id}/{source.logical_path}"


def _artifact_tool_failure(error: str, *, message: str) -> dict[str, Any]:
    return {
        "tool_name": ARTIFACT_EXPORT_TOOL,
        "ok": False,
        "error": error,
        "result": {"message": message},
    }


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _run_filesystem_skill_script(
    session: Any,
    *,
    tool_name: str,
    script_name: str,
    script_root: str | Path | None,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    root = _resolve_filesystem_skill_root(session, explicit_root=script_root)
    if root is None:
        return _filesystem_tool_failure(
            tool_name,
            "FILESYSTEM_SKILL_NOT_INSTALLED",
            result={"message": "filesystem Skill package is not installed."},
        )
    script_file = root / "scripts" / script_name
    if not script_file.is_file():
        return _filesystem_tool_failure(
            tool_name,
            "FILESYSTEM_SCRIPT_NOT_FOUND",
            result={"script_path": str(script_file)},
        )

    try:
        command_args = _filesystem_command_arguments(tool_name, arguments)
    except ValueError as exc:
        return _filesystem_tool_failure(tool_name, "FILESYSTEM_INPUT_INVALID", result={"message": str(exc)})

    env = dict(os.environ)
    env.setdefault("MY_AGENTS_DEFAULT_FILE_OUTPUT_DIR", str(_filesystem_default_output_root()))
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("PYTHONUTF8", "1")
    timeout_seconds = _bounded_int(arguments.get("timeout_seconds"), default=30, minimum=1, maximum=120)
    completed = subprocess.run(
        [sys.executable, str(script_file), *command_args],
        cwd=str(root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
        env=env,
        shell=False,
    )
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    error = _filesystem_error_message(stdout, stderr, completed.returncode)
    result = _filesystem_result_payload(
        tool_name,
        stdout=stdout,
        stderr=stderr,
        return_code=completed.returncode,
        root=root,
        script_file=script_file,
    )
    return {
        "tool_name": tool_name,
        "ok": error is None,
        "error": error,
        "result": result,
    }


def _filesystem_command_arguments(tool_name: str, arguments: dict[str, Any]) -> list[str]:
    if tool_name in {FILESYSTEM_LIST_DIR_TOOL, FILESYSTEM_PATH_EXISTS_TOOL, FILESYSTEM_PATH_STAT_TOOL}:
        return ["--path", _required_str(arguments.get("path"), "path")]
    if tool_name == FILESYSTEM_READ_FILE_TOOL:
        return [
            "--path",
            _required_str(arguments.get("path"), "path"),
            "--offset",
            str(_bounded_int(arguments.get("offset"), default=0, minimum=0, maximum=100_000)),
            "--limit",
            str(_bounded_int(arguments.get("limit"), default=200, minimum=1, maximum=500)),
            "--encoding",
            _non_empty_str(arguments.get("encoding")) or "auto",
        ]
    if tool_name == FILESYSTEM_WRITE_TEXT_TOOL:
        command = [
            "--path",
            _required_str(arguments.get("path"), "path"),
            "--text",
            _required_str(arguments.get("text"), "text"),
            "--encoding",
            _non_empty_str(arguments.get("encoding")) or "utf-8",
        ]
        if bool(arguments.get("overwrite")):
            command.append("--overwrite")
        return command
    if tool_name == FILESYSTEM_REPLACE_TEXT_TOOL:
        command = [
            "--path",
            _required_str(arguments.get("path"), "path"),
            "--old-text",
            _required_str(arguments.get("old_text"), "old_text"),
            "--new-text",
            _required_str(arguments.get("new_text"), "new_text"),
            "--encoding",
            _non_empty_str(arguments.get("encoding")) or "utf-8",
            "--count",
            str(_bounded_int(arguments.get("count"), default=0, minimum=0, maximum=100_000)),
        ]
        return command
    if tool_name in {FILESYSTEM_COPY_FILE_TOOL, FILESYSTEM_MOVE_FILE_TOOL}:
        source = _required_str(arguments.get("src"), "src")
        # The model must submit the concrete destination. Runtime validates its
        # boundary and permissions, but this legacy script adapter must not
        # invent a filename or revive the old operation-intent shortcut.
        destination = _required_str(arguments.get("dst"), "dst")
        command = ["--src", source, "--dst", destination]
        if bool(arguments.get("overwrite")):
            command.append("--overwrite")
        return command
    if tool_name == FILESYSTEM_DELETE_PATH_TOOL:
        command = ["--path", _required_str(arguments.get("path"), "path")]
        if bool(arguments.get("recursive")):
            command.append("--recursive")
        if bool(arguments.get("force")):
            command.append("--force")
        else:
            command.append("--force-file")
        return command
    if tool_name == FILESYSTEM_MAKE_DIR_TOOL:
        command = ["--path", _required_str(arguments.get("path"), "path")]
        if bool(arguments.get("exist_ok", True)):
            command.append("--exist-ok")
        return command
    raise ValueError(f"unsupported filesystem tool: {tool_name}")


def _resolve_filesystem_skill_root(session: Any, *, explicit_root: str | Path | None = None) -> Path | None:
    if explicit_root is not None:
        root = Path(explicit_root).expanduser().resolve()
        return root if (root / "scripts").is_dir() else None

    db_root = _filesystem_skill_root_from_db(session)
    if db_root is not None:
        return db_root

    return _filesystem_skill_root_from_files()


def _filesystem_skill_root_from_db(session: Any) -> Path | None:
    if session is None:
        return None
    try:
        from sqlalchemy import select

        from app.domains.agent_memory.models import AgentSkill, AgentSkillStatus

        skill = session.scalar(
            select(AgentSkill)
            .where(AgentSkill.name == "filesystem")
            .where(AgentSkill.status == AgentSkillStatus.ACTIVE)
            .order_by(AgentSkill.updated_at.desc(), AgentSkill.created_at.desc())
        )
    except Exception:
        return None
    if skill is None or not skill.file_path:
        return None
    root = Path(str(skill.file_path)).expanduser().resolve().parent
    return root if (root / "scripts").is_dir() else None


def _filesystem_skill_root_from_files() -> Path | None:
    root = Path(__file__).resolve().parents[4] / "docs" / "agent-skills"
    if not root.is_dir():
        return None
    for skill_file in root.glob("*/SKILL.md"):
        try:
            content = skill_file.read_text(encoding="utf-8")[:512]
        except OSError:
            continue
        if re.search(r"(?m)^name:\s*filesystem\s*$", content) and (skill_file.parent / "scripts").is_dir():
            return skill_file.parent.resolve()
    return None


def _filesystem_result_payload(
    tool_name: str,
    *,
    stdout: str,
    stderr: str,
    return_code: int,
    root: Path,
    script_file: Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "stdout": stdout,
        "stderr": stderr,
        "return_code": return_code,
        "script_path": str(script_file),
        "skill_root": str(root),
    }
    stripped = stdout.strip()
    if tool_name == FILESYSTEM_READ_FILE_TOOL:
        result["content"] = stdout
    elif tool_name == FILESYSTEM_LIST_DIR_TOOL:
        result["entries"] = _filesystem_list_entries(stdout)
    elif tool_name == FILESYSTEM_PATH_EXISTS_TOOL:
        result["exists"] = stripped.startswith("EXISTS:")
    elif tool_name == FILESYSTEM_PATH_STAT_TOOL:
        result["stat"] = _filesystem_stat_payload(stdout)
    return result


def _filesystem_error_message(stdout: str, stderr: str, return_code: int) -> str | None:
    for line in stdout.splitlines():
        if line.strip().startswith("ERROR:"):
            return line.strip()
    if return_code != 0:
        return stderr.strip() or f"filesystem script exited with code {return_code}"
    return None


def _filesystem_tool_failure(tool_name: str, error: str, *, result: dict[str, Any]) -> dict[str, Any]:
    return {"tool_name": tool_name, "ok": False, "error": error, "result": result}


def _filesystem_list_entries(stdout: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for line in stdout.splitlines():
        if "\t" not in line:
            continue
        kind, name = line.split("\t", 1)
        if kind in {"dir", "file"} and name:
            entries.append({"type": kind, "name": name})
    return entries


def _filesystem_stat_payload(stdout: str) -> dict[str, Any]:
    stat: dict[str, Any] = {}
    for line in stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized_key = key.strip().lower()
        cleaned_value = value.strip()
        stat[normalized_key] = int(cleaned_value) if normalized_key == "size" and cleaned_value.isdigit() else cleaned_value
    return stat


def _filesystem_default_output_root() -> Path:
    root = Path(__file__).resolve().parents[4] / "runtime" / "filesystem-tool-output"
    root.mkdir(parents=True, exist_ok=True)
    return root


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
        policy_error = validate_dbx_tool_arguments(tool_name, arguments)
        if policy_error is not None:
            logger.warning(
                "Blocked DBX MCP call by read-only policy",
                extra={"mcp_tool_name": tool_name, "policy_error": policy_error.error},
            )
            return policy_error
        return client.call_tool(tool_name=tool_name, arguments=arguments)

    return handler


def _mcp_risk_level(policy: MCPToolPolicy, tool_name: str) -> AgentToolRiskLevel:
    if policy.requires_confirmation(tool_name):
        return AgentToolRiskLevel.HIGH
    if _is_readonly_mcp_tool(tool_name):
        return AgentToolRiskLevel.LOW
    return AgentToolRiskLevel.MEDIUM


def _is_readonly_mcp_tool(tool_name: str) -> bool:
    normalized = str(tool_name or "").strip().lower().replace("-", "_")
    # DBX exposes a fixed read-only contract. Keep the SDK risk classification
    # aligned with that allowlist; execute_query remains safe because the DBX
    # gateway validates its SQL immediately before the MCP call.
    dbx_tool_name = normalized.removeprefix("mcp.")
    if dbx_tool_name in DBX_READONLY_TOOL_NAMES:
        return True
    if dbx_tool_name in {"qq_mail.list_messages", "qq_mail.get_message"}:
        return True
    if normalized in {"open_page", "read_page"}:
        return True
    action = normalized.rsplit(".", 1)[-1]
    # MCP tools from different servers use different namespaces, but read-only
    # verbs are stable enough to keep child-agent discovery tools usable without
    # exposing broad browser or cloud-file mutations.
    readonly_verbs = (
        "list",
        "read",
        "get",
        "search",
        "query",
        "status",
        "navigate",
        "select",
        "wait",
        "screenshot",
        "take_screenshot",
    )
    if action in readonly_verbs or action.startswith(tuple(f"{verb}_" for verb in readonly_verbs)):
        return True
    # Some MCP servers prefix every tool with their product name, e.g.
    # dbx_list_connections. Treat vendor_prefixed readonly verbs as low-risk
    # so harmless probes can be delegated to the SDK child agent.
    return re.match(r"^[a-z0-9]+_(list|read|get|search|query|status|navigate|select|wait|screenshot|take_screenshot)_", action) is not None
