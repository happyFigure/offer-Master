from __future__ import annotations

from typing import Any

import httpx
from fastapi import APIRouter, Depends

from app.agent_runtime.agent_as_tool import (
    CLAUDE_SDK_AGENT_EXECUTOR_ID,
    OPENAI_SDK_AGENT_EXECUTOR_ID,
    TOOL_REGISTRY_EXECUTOR_ID,
    AgentCapabilityDefinition,
    create_default_agent_capability_registry,
)
from app.agent_runtime.external_tasks.configured import build_agent_runtime_executor_bundle
from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL, create_default_agent_tool_registry
from app.core.config import Settings, get_settings


router = APIRouter(prefix="/api/v1/agent-runtime", tags=["agent-runtime"])


@router.get("/panel")
def get_agent_runtime_panel(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    tool_registry = create_default_agent_tool_registry()
    agent_executors, capability_executor_ids = build_agent_runtime_executor_bundle(settings)
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
    capabilities = [capabilities_by_id[key] for key in sorted(capabilities_by_id)]
    agents = _build_agent_members(
        capabilities,
        executor_ids=set(agent_executors.keys()),
        settings=settings,
    )
    low_risk_count = sum(1 for capability in capabilities if capability["risk_level"] == "low")
    confirmation_required_count = sum(1 for capability in capabilities if capability["requires_confirmation"])

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
        "agents": agents,
        "capabilities": capabilities,
    }


def _serialize_capability(definition: AgentCapabilityDefinition, *, settings: Settings) -> dict[str, Any]:
    candidate_profile = definition.candidate_profile
    return {
        "id": definition.capability_id,
        "name": _display_capability_name(definition),
        "description": definition.description,
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
        "provider": _capability_provider(definition, settings=settings),
        "status": "active",
    }


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
    for executor_id in sorted(executor_ids):
        health = _agent_health_for_executor(executor_id, settings=settings)
        agents.append(
            {
                "id": executor_id,
                "name": _display_executor_name(executor_id),
                "kind": "external_agent",
                "status": "offline" if health["status"] == "unreachable" else "active",
                "role": "ability_agent",
                "description": "实现统一 agent-as-tool 接口，并向主 agent 声明可执行能力。",
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
    if executor_id == OPENAI_SDK_AGENT_EXECUTOR_ID:
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
    if executor_id == OPENAI_SDK_AGENT_EXECUTOR_ID:
        return "OpenAI SDK Agent"
    return executor_id.replace("-", " ").replace("_", " ").title()


def _capability_provider(definition: AgentCapabilityDefinition, *, settings: Settings) -> str:
    if definition.executor_id == OPENAI_SDK_AGENT_EXECUTOR_ID:
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
