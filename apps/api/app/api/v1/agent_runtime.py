from __future__ import annotations

from typing import Any
import logging

import httpx
from fastapi import APIRouter, Depends

from app.agent_runtime.agent_as_tool import (
    CLAUDE_SDK_AGENT_EXECUTOR_ID,
    FILESYSTEM_SKILL_EXECUTOR_ID,
    OPENAI_SDK_AGENT_EXECUTOR_ID,
    OPENAI_SDK_QQ_MAIL_AGENT_EXECUTOR_ID,
    TOOL_REGISTRY_EXECUTOR_ID,
    AgentCapabilityDefinition,
    create_default_agent_capability_registry,
)
from app.agent_runtime.external_tasks.configured import build_agent_runtime_executor_bundle
from app.agent_runtime.sdk_agents.delegation_policy import build_sdk_agent_delegation_policy
from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL, create_default_agent_tool_registry, create_mcp_agent_tool_definitions
from app.core.config import Settings, get_settings
from app.mcp_gateway.configured import configured_mcp_tool_names, configured_mcp_transport_label, create_configured_mcp_client


router = APIRouter(prefix="/api/v1/agent-runtime", tags=["agent-runtime"])
logger = logging.getLogger(__name__)


@router.get("/panel")
def get_agent_runtime_panel(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    tool_registry = create_default_agent_tool_registry()
    _register_configured_mcp_tools(tool_registry, settings=settings)
    # Keep the panel's runtime view aligned with the real agent graph: SDK
    # delegation decisions depend on the registered tools, especially the
    # coarse filesystem Skill route that is backed by concrete filesystem tools.
    agent_executors, capability_executor_ids = build_agent_runtime_executor_bundle(settings, tool_registry=tool_registry)
    capability_registry = create_default_agent_capability_registry(
        tool_registry=tool_registry,
        executor_id_by_capability=capability_executor_ids,
    )
    capabilities_by_id = {
        definition.capability_id: _serialize_capability(definition, settings=settings)
        for definition in capability_registry.list_definitions()
    }
    for agent in agent_executors.values():
        agent_capabilities = getattr(agent, "capabilities", None)
        if not callable(agent_capabilities):
            continue
        for definition in agent_capabilities():
            capabilities_by_id[definition.capability_id] = _serialize_capability(definition, settings=settings)
    if OPENAI_SDK_AGENT_EXECUTOR_ID in agent_executors:
        sdk_delegation = build_sdk_agent_delegation_policy(settings, tool_registry=tool_registry)
        for tool_name in sdk_delegation.internal_tool_names:
            capability = capabilities_by_id.get(tool_name)
            if tool_name.startswith("mcp.") and capability is not None:
                # Keep mailbox MCP tools grouped beneath their dedicated OpenAI child Agent.
                is_mail_tool = tool_name.startswith(("mcp.qq_mail.", "mcp.qq-mail.", "mcp.qqmail."))
                capability["executor_id"] = (
                    OPENAI_SDK_QQ_MAIL_AGENT_EXECUTOR_ID
                    if is_mail_tool and OPENAI_SDK_QQ_MAIL_AGENT_EXECUTOR_ID in agent_executors
                    else OPENAI_SDK_AGENT_EXECUTOR_ID
                )
    capabilities = [capabilities_by_id[key] for key in sorted(capabilities_by_id)]
    agents = _build_agent_members(
        capabilities,
        executor_ids=set(agent_executors.keys()),
        settings=settings,
    )
    low_risk_count = sum(1 for capability in capabilities if capability["risk_level"] == "low")
    confirmation_required_count = sum(1 for capability in capabilities if capability["requires_confirmation"])
    mcp_integrations = _mcp_integration_statuses(
        capabilities,
        settings=settings,
        client=_configured_mcp_client_for_panel(settings),
    )
    logger.info(
        "Agent runtime panel built from declarative MCP configuration",
        extra={
            "mcp_integration_count": len(mcp_integrations),
            "mcp_discovery_started": False,
        },
    )

    return {
        "main_agent": {
            "id": "offermaster-main-agent",
            "name": "OfferMaster 主 Agent",
            "role": "orchestrator",
            "status": "active",
            "description": "负责会话、提示词组织、工具选择、权限校验、结果汇总和最终回复。",
            "health": _static_health("healthy", "主 Agent 已运行", checked=False),
        },
        "summary": {
            "agent_count": len(agents),
            "capability_count": len(capabilities),
            "low_risk_count": low_risk_count,
            "confirmation_required_count": confirmation_required_count,
            "configured_web_search_provider": _web_search_provider(settings),
        },
        "mcp_integrations": mcp_integrations,
        "agents": agents,
        "capabilities": capabilities,
    }


def _register_configured_mcp_tools(tool_registry: Any, *, settings: Settings) -> None:
    # The panel is a read-only architecture view. Starting stdio MCP
    # processes here made a browser refresh depend on Chrome/DBX startup and
    # could exceed the frontend request timeout. The chat runtime performs
    # real discovery when a child agent is actually prepared to run.
    mcp_client = create_configured_mcp_client(settings, discover=False)
    if mcp_client is None:
        return
    # The panel should be a faithful read-only mirror of the real chat runtime.
    # Discovery runs through the same registry, and only discovered tools are
    # exposed when a real MCP server is unavailable.
    tool_registry.register_many(
        create_mcp_agent_tool_definitions(
            mcp_client,
            allowed_tool_names=configured_mcp_tool_names(mcp_client, fallback=settings.allowed_mcp_tools),
        )
    )


def _configured_mcp_client_for_panel(settings: Settings) -> Any | None:
    # Do not perform live MCP discovery in the synchronous panel endpoint.
    # Registry statuses distinguish configured tools from tools discovered by
    # an actual child-agent run.
    return create_configured_mcp_client(settings, discover=False)


def _mcp_integration_statuses(
    capabilities: list[dict[str, Any]],
    *,
    settings: Settings,
    client: Any | None = None,
) -> list[dict[str, Any]]:
    statuses = {
        str(item.get("id")): item
        for item in (client.statuses() if client is not None and hasattr(client, "statuses") else [])
    }
    integrations = [
        _mcp_integration_status(
            integration_id="chrome",
            name="Google Chrome MCP",
            enabled=bool(settings.sdk_agent_enable_chrome_mcp),
            prefixes=("mcp.chrome.", "mcp.google_chrome.", "mcp.google-chrome."),
            capabilities=capabilities,
            settings=settings,
            runtime_status=statuses.get("chrome"),
        ),
        _mcp_integration_status(
            integration_id="dbx",
            name="dbx MCP",
            enabled=bool(settings.sdk_agent_enable_dbx_mcp),
            prefixes=("mcp.dbx.", "mcp.dropbox."),
            capabilities=capabilities,
            settings=settings,
            runtime_status=statuses.get("dbx"),
        ),
        _mcp_integration_status(
            integration_id="qq_mail",
            name="QQ Mail MCP",
            enabled=bool(settings.sdk_agent_enable_qq_mail_mcp),
            prefixes=("mcp.qq_mail.", "mcp.qq-mail.", "mcp.qqmail."),
            capabilities=capabilities,
            settings=settings,
            runtime_status=statuses.get("qq_mail"),
        ),
    ]
    mail = next(item for item in integrations if item["id"] == "qq_mail")
    auth_code = getattr(settings, "qq_mail_auth_code", None)
    auth_code_configured = bool(
        auth_code.get_secret_value().strip() if hasattr(auth_code, "get_secret_value") else str(auth_code or "").strip()
    )
    if settings.sdk_agent_enable_qq_mail_mcp and not (
        str(getattr(settings, "qq_mail_username", "") or "").strip() and auth_code_configured
    ):
        mail.update(
            {
                "status": "credentials_missing",
                "label": "邮箱凭据未配置",
                "detail": "MCP 已启用，但本机运行环境尚未配置 QQ 邮箱账号或 IMAP 授权码。",
            }
        )
    return integrations


def _mcp_integration_status(
    *,
    integration_id: str,
    name: str,
    enabled: bool,
    prefixes: tuple[str, ...],
    capabilities: list[dict[str, Any]],
    settings: Settings,
    runtime_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    registered_tools = sorted(
        str(capability["id"])
        for capability in capabilities
        if any(str(capability.get("id") or "").startswith(prefix) for prefix in prefixes)
    )
    transport_label = configured_mcp_transport_label(settings, integration_id=integration_id)
    gateway_configured = transport_label is not None
    if not enabled:
        status = "disabled"
        label = "未启用"
        detail = "SDK 子 Agent 的这个 MCP 开关未开启。"
    elif runtime_status and runtime_status.get("status") == "unavailable":
        status = "unavailable"
        label = "MCP 不可用"
        detail = str(runtime_status.get("detail") or "MCP 服务没有返回可用工具。")
    elif runtime_status and runtime_status.get("status") == "configured":
        status = "configured"
        label = "已配置，待首次运行时发现"
        detail = "声明式配置已加载；首次由 SDK 子 Agent 实际运行时才启动 MCP 并发现工具。"
    elif registered_tools:
        status = "registered"
        label = "已注册到 SDK 子 Agent"
        if transport_label == "stdio_bridge":
            detail = "工具定义已进入 runtime registry，调用时会通过默认 stdio MCP bridge 拉起本地 MCP 服务器。"
        else:
            detail = "工具定义已进入 runtime registry，并会通过配置的 HTTP MCP Gateway 执行。"
    elif not gateway_configured:
        status = "not_configured"
        label = "MCP Gateway 未配置"
        detail = "已开启子 Agent MCP 开关，但 JOBPILOT_MCP_ENABLED 或 JOBPILOT_MCP_SERVER_URL 还没有配置好。"
    else:
        status = "not_registered"
        label = "网关已配置但工具未注册"
        detail = "检查 MCP allowlist、工具命名空间和网关暴露的工具名是否一致。"
    return {
        "id": integration_id,
        "name": name,
        "status": status,
        "label": label,
        "detail": detail,
        "enabled": enabled,
        "gateway_configured": gateway_configured,
        "transport": transport_label,
        "registered_tools": registered_tools,
        "configured_tools": list(runtime_status.get("configured_tools") or []) if runtime_status else [],
        "discovered_tools": list(runtime_status.get("discovered_tools") or []) if runtime_status else [],
        "policy": runtime_status.get("policy") if runtime_status else None,
    }


def _serialize_capability(definition: AgentCapabilityDefinition, *, settings: Settings) -> dict[str, Any]:
    candidate_profile = definition.candidate_profile
    semantic_profile = definition.semantic_profile
    return {
        "id": definition.capability_id,
        "name": _display_capability_name(definition),
        "description": definition.description,
        "kind": definition.kind,
        "executor_id": definition.executor_id,
        "risk_level": definition.risk_level,
        "requires_confirmation": definition.requires_confirmation,
        "allowed_source_types": sorted(definition.allowed_source_types),
        "supported_intents": list(definition.supported_intents),
        "input_fields": _schema_fields(definition.input_schema),
        "output_fields": _schema_fields(definition.output_schema),
        "candidate_categories": sorted(getattr(candidate_profile, "categories", []) or []),
        "candidate_keywords": sorted(getattr(candidate_profile, "keywords", []) or []),
        "candidate_examples": list(getattr(candidate_profile, "examples", ()) or ()),
        "candidate_use_when": list(getattr(candidate_profile, "use_when", ()) or ()),
        "candidate_do_not_use_when": list(getattr(candidate_profile, "do_not_use_when", ()) or ()),
        "candidate_positive_examples": list(getattr(candidate_profile, "positive_examples", ()) or ()),
        "candidate_negative_examples": list(getattr(candidate_profile, "negative_examples", ()) or ()),
        "candidate_required_context_focus": sorted(getattr(candidate_profile, "required_context_focus", []) or []),
        "candidate_disambiguation_notes": list(getattr(candidate_profile, "disambiguation_notes", ()) or ()),
        "semantic_profile": _semantic_profile_payload(semantic_profile),
        "provider": _capability_provider(definition, settings=settings),
        "status": "active",
    }


def _semantic_profile_payload(profile: Any | None) -> dict[str, Any] | None:
    if profile is None:
        return None
    to_metadata = getattr(profile, "to_metadata", None)
    if callable(to_metadata):
        return to_metadata()
    if isinstance(profile, dict):
        return dict(profile)
    return None


def _build_agent_members(
    capabilities: list[dict[str, Any]],
    *,
    executor_ids: set[str],
    settings: Settings,
) -> list[dict[str, Any]]:
    capabilities_by_executor: dict[str, list[dict[str, Any]]] = {}
    for capability in capabilities:
        capabilities_by_executor.setdefault(str(capability["executor_id"]), []).append(capability)

    agents = [
        {
            "id": TOOL_REGISTRY_EXECUTOR_ID,
            "name": "本地工具注册中心",
            "kind": "local_runtime",
            "status": "active",
            "role": "tool_registry",
            "description": "把本地数据库、网页搜索、岗位来源、记忆检索等工具登记给主 agent 调度。",
            "health": _static_health("healthy", "本地可用", checked=False),
            "capabilities": capabilities_by_executor.get(TOOL_REGISTRY_EXECUTOR_ID, []),
        }
    ]
    if FILESYSTEM_SKILL_EXECUTOR_ID in capabilities_by_executor:
        agents.append(
            {
                "id": FILESYSTEM_SKILL_EXECUTOR_ID,
                "name": "Filesystem Skill Executor",
                "kind": "local_runtime",
                "status": "active",
                "role": "skill_executor",
                "description": "执行 filesystem 高层 Skill，并在内部选择 read/move/replace 等脚本。",
                "health": _static_health("healthy", "本地可用", checked=False),
                "capabilities": capabilities_by_executor.get(FILESYSTEM_SKILL_EXECUTOR_ID, []),
            }
        )
    for executor_id in sorted(executor_ids):
        health = _agent_health_for_executor(executor_id, settings=settings)
        agents.append(
            {
                "id": executor_id,
                "name": _display_executor_name(executor_id),
                "kind": "external_agent",
                "status": "offline" if health["status"] == "unreachable" else "active",
                "role": "ability_agent",
                "description": (
                    "专门读取 QQ 招聘邮件并提取测评、笔试、面试和截止时间；仅使用只读邮箱工具。"
                    if executor_id == OPENAI_SDK_QQ_MAIL_AGENT_EXECUTOR_ID
                    else "实现统一 agent-as-tool 接口，并向主 agent 声明可执行能力。"
                ),
                "health": health,
                "capabilities": capabilities_by_executor.get(executor_id, []),
            }
        )

    registered_agent_ids = {str(agent["id"]) for agent in agents}
    if settings.claude_sdk_agent_base_url and CLAUDE_SDK_AGENT_EXECUTOR_ID not in registered_agent_ids:
        health = _claude_sdk_agent_health(settings)
        agents.append(
            {
                "id": CLAUDE_SDK_AGENT_EXECUTOR_ID,
                "name": "Claude SDK Agent",
                "kind": "external_agent",
                "status": "standby" if health["status"] == "healthy" else "offline",
                "role": "ability_agent",
                "description": "已配置 Claude SDK 子 agent，但当前网页搜索由其他 provider 接管，暂不接管能力。",
                "health": health,
                "capabilities": [],
            }
        )
    if OPENAI_SDK_AGENT_EXECUTOR_ID not in registered_agent_ids:
        health = _agent_health_for_executor(OPENAI_SDK_AGENT_EXECUTOR_ID, settings=settings)
        agents.append(
            {
                "id": OPENAI_SDK_AGENT_EXECUTOR_ID,
                "name": "OpenAI SDK Agent",
                "kind": "external_agent",
                "status": "active" if health["status"] == "healthy" else "offline",
                "role": "ability_agent",
                "description": "可接入 OpenAI SDK 子 agent，用于根据用户简历和目标 JD 生成简历修改结果。",
                "health": health,
                "capabilities": [],
            }
        )
    return agents


def _agent_health_for_executor(executor_id: str, *, settings: Settings) -> dict[str, Any]:
    if executor_id == CLAUDE_SDK_AGENT_EXECUTOR_ID:
        return _claude_sdk_agent_health(settings)
    if executor_id in {OPENAI_SDK_AGENT_EXECUTOR_ID, OPENAI_SDK_QQ_MAIL_AGENT_EXECUTOR_ID}:
        if not settings.openai_sdk_agent_enabled:
            return _static_health("not_configured", "未配置", checked=False)
        if not _openai_sdk_agent_api_key(settings):
            return _static_health("not_configured", "缺少 API Key", checked=False)
        return _static_health("healthy", "已配置", checked=False)
    return _static_health("healthy", "已注册", checked=False)


def _claude_sdk_agent_health(settings: Settings) -> dict[str, Any]:
    base_url = str(settings.claude_sdk_agent_base_url or "").strip().rstrip("/")
    if not base_url:
        return _static_health("not_configured", "未配置", checked=False)

    health_url = f"{base_url}/health"
    try:
        response = httpx.get(health_url, timeout=0.6)
        response.raise_for_status()
    except Exception as exc:  # pragma: no cover - exact transport exceptions vary by environment.
        return {
            "status": "unreachable",
            "label": "未启动或连接失败",
            "detail": exc.__class__.__name__,
            "checked": True,
            "url": health_url,
        }

    return {
        "status": "healthy",
        "label": "已连接",
        "detail": f"HTTP {response.status_code}",
        "checked": True,
        "url": health_url,
    }


def _static_health(status: str, label: str, *, checked: bool) -> dict[str, Any]:
    return {
        "status": status,
        "label": label,
        "detail": None,
        "checked": checked,
    }


def _openai_sdk_agent_api_key(settings: Settings) -> str:
    dedicated_key = (
        settings.openai_sdk_agent_api_key.get_secret_value().strip()
        if settings.openai_sdk_agent_api_key is not None
        else ""
    )
    if dedicated_key:
        return dedicated_key
    return settings.llm_api_key.get_secret_value().strip() if settings.llm_api_key is not None else ""


def _schema_fields(schema: dict[str, Any]) -> list[str]:
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(properties, dict):
        required = schema.get("required") if isinstance(schema, dict) else None
        return [str(item) for item in required] if isinstance(required, list) else []
    required_fields = [str(item) for item in schema.get("required", []) if str(item) in properties]
    optional_fields = [str(name) for name in properties if str(name) not in required_fields]
    return required_fields + optional_fields


def _display_capability_name(definition: AgentCapabilityDefinition) -> str:
    if definition.capability_id == EXTERNAL_WEB_SEARCH_TOOL:
        return "网页搜索"
    if definition.name != definition.capability_id:
        return definition.name
    return definition.capability_id.replace("_", " ").replace(".", " · ")


def _display_executor_name(executor_id: str) -> str:
    if executor_id == CLAUDE_SDK_AGENT_EXECUTOR_ID:
        return "Claude SDK Agent"
    if executor_id == OPENAI_SDK_QQ_MAIL_AGENT_EXECUTOR_ID:
        return "OpenAI SDK Agent（QQ 邮箱）"
    if executor_id == OPENAI_SDK_AGENT_EXECUTOR_ID:
        return "OpenAI SDK Agent"
    return executor_id.replace("-", " ").replace("_", " ").title()


def _capability_provider(definition: AgentCapabilityDefinition, *, settings: Settings) -> str:
    if definition.executor_id in {OPENAI_SDK_AGENT_EXECUTOR_ID, OPENAI_SDK_QQ_MAIL_AGENT_EXECUTOR_ID}:
        return "openai-sdk-agent"
    if definition.capability_id == EXTERNAL_WEB_SEARCH_TOOL:
        return _web_search_provider(settings)
    return "local"


def _web_search_provider(settings: Settings) -> str:
    provider = str(getattr(settings, "external_web_search_provider", "auto") or "auto").strip().lower()
    if provider == "bailian":
        return "bailian"
    if settings.claude_sdk_agent_base_url:
        return "claude-sdk-agent"
    return provider or "auto"
