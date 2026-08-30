from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
import os
from pathlib import Path
import re
import subprocess
import sys
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
DATABASE_COMPANY_PROFILE_TOOL = "database.company_profile"
DATABASE_JOB_SEARCH_TOOL = "database.job_search"
DATABASE_SOURCE_SEARCH_TOOL = "database.source_search"
DATABASE_COMPANY_UPDATE_TOOL = "database.company_update"
DATABASE_JOB_LEAD_DELETE_TOOL = "database.job_lead_delete"
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
    registry.register_many(create_database_agent_tool_definitions())
    registry.register_many(create_local_company_database_agent_tool_definitions())
    registry.register_many(create_local_job_source_agent_tool_definitions(offerio_provider_factory=offerio_provider_factory))
    registry.register_many(create_filesystem_agent_tool_definitions())
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


def create_database_agent_tool_definitions() -> list[AgentToolDefinition]:
    standard_output = {"type": "object", "required": ["tool_name", "ok", "result"]}
    read_source_types = frozenset({"agent_chat", "job_discovery"})
    return [
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
            description="Read a bounded slice of a user-provided local text file.",
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
            description="Replace exact text inside a user-provided local text file while preserving all other content.",
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
            ),
        ),
        AgentToolDefinition(
            name=FILESYSTEM_COPY_FILE_TOOL,
            description="Copy a user-provided local file or directory to another local path after confirmation.",
            input_schema={
                "type": "object",
                "required": ["src", "dst"],
                "properties": {
                    "src": {"type": "string"},
                    "dst": {"type": "string"},
                    "overwrite": {"type": "boolean", "default": False},
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
        ),
        AgentToolDefinition(
            name=FILESYSTEM_MOVE_FILE_TOOL,
            description="Move or rename a user-provided local file or directory after confirmation.",
            input_schema={
                "type": "object",
                "required": ["src", "dst"],
                "properties": {
                    "src": {"type": "string"},
                    "dst": {"type": "string"},
                    "overwrite": {"type": "boolean", "default": False},
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
        command = ["--src", _required_str(arguments.get("src"), "src"), "--dst", _required_str(arguments.get("dst"), "dst")]
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
        return client.call_tool(tool_name=tool_name, arguments=arguments)

    return handler


def _mcp_risk_level(policy: MCPToolPolicy, tool_name: str) -> AgentToolRiskLevel:
    if policy.requires_confirmation(tool_name):
        return AgentToolRiskLevel.HIGH
    if tool_name in {"open_page", "read_page"}:
        return AgentToolRiskLevel.LOW
    return AgentToolRiskLevel.MEDIUM
