from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import logging
from typing import Any

from sqlalchemy.orm import Session

from app.agent_runtime.agent_as_tool import (
    AbilityAgent,
    AgentCapabilityDefinition,
    AgentRuntimeContext,
    DEFAULT_SUPPORTED_INTENTS_BY_CAPABILITY,
    FILESYSTEM_SKILL_CAPABILITY,
    OPENAI_SDK_AGENT_EXECUTOR_ID,
    OPENAI_SDK_QQ_MAIL_AGENT_EXECUTOR_ID,
    build_agent_runtime_bundle,
    filesystem_skill_capability_definition,
)
from app.agent_runtime.external_tasks.dispatcher import ExternalTaskDispatcher
from app.agent_runtime.external_tasks.executors import (
    BailianWebSearchExecutor,
    ClaudeSdkAgentExecutor,
    ClaudeSdkHttpExecutorAdapter,
    ClaudeSdkHttpExecutorConfig,
    OpenAISdkAgentConfig,
    OpenAISdkAgentExecutor,
    OpenAISdkResumeClientAdapter,
    _run_http_web_search,
)
from app.agent_runtime.external_tasks.repository import SqlAlchemyExternalAgentTaskRepository
from app.agent_runtime.sdk_agents.runner_adapter import (
    OpenAIAgentsSdkRunnerClient,
    OpenAISdkAgentRunnerAdapter,
    OpenAISdkAgentRunnerExecutor,
)
from app.agent_runtime.sdk_agents.delegation_policy import (
    DBX_READONLY_AGENT_CAPABILITY,
    GOOGLE_CHROME_AGENT_CAPABILITY,
    QQ_MAIL_READONLY_AGENT_CAPABILITY,
    SdkAgentDelegationPolicy,
    build_sdk_agent_delegation_policy,
)
from app.agent_runtime.sdk_agents.sandbox import SDK_AGENT_SANDBOX_MODE_TEMP_COPY, SdkAgentSandboxManager
from app.agent_runtime.sdk_agents.tool_gateway import SdkAgentToolGateway
from app.agent_runtime.tool_registry import (
    EXTERNAL_WEB_SEARCH_TOOL,
    AgentToolRegistry,
)
from app.core.config import OPENAI_SDK_AGENT_MODE_AGENTS_SDK, OPENAI_SDK_AGENT_MODE_CHAT_COMPLETIONS, Settings
from app.infrastructure.llm.client import build_llm_runtime_config


ExternalTaskDispatcherCallback = Callable[[Session, str], dict[str, Any]]
ExternalWebSearchCallback = Callable[[str, int], dict[str, Any]]

logger = logging.getLogger(__name__)


def build_external_task_dispatcher_callback(settings: Settings) -> ExternalTaskDispatcherCallback | None:
    if not settings.external_agent_auto_dispatch:
        return None
    base_url = str(settings.claude_sdk_agent_base_url or "").strip()
    if not base_url:
        return None

    config = _build_claude_sdk_http_executor_config(settings, base_url=base_url)

    def dispatch(session: Session, task_id: str) -> dict[str, Any]:
        dispatcher = ExternalTaskDispatcher(
            repository=SqlAlchemyExternalAgentTaskRepository(session),
            executor=ClaudeSdkHttpExecutorAdapter(config=config),
        )
        return dispatcher.dispatch(task_id).to_dict()

    return dispatch


def build_external_web_search_callback(settings: Settings) -> ExternalWebSearchCallback | None:
    if not settings.external_agent_auto_dispatch:
        return None
    provider = str(getattr(settings, "external_web_search_provider", "auto") or "auto").strip().lower()
    if provider == "bailian":
        adapter = BailianWebSearchExecutor(config=build_llm_runtime_config(settings))

        def search_with_bailian(query: str, max_results: int = 5) -> dict[str, Any]:
            return adapter.execute_web_search(query, max_results=max_results)

        return search_with_bailian

    base_url = str(settings.claude_sdk_agent_base_url or "").strip()
    if base_url:
        config = _build_claude_sdk_http_executor_config(settings, base_url=base_url)
        adapter = ClaudeSdkHttpExecutorAdapter(config=config)

        def search_with_claude_sdk_agent(query: str, max_results: int = 5) -> dict[str, Any]:
            return adapter.execute_web_search(query, max_results=max_results)

        return search_with_claude_sdk_agent

    def search(query: str, max_results: int = 5) -> dict[str, Any]:
        return _run_http_web_search(query, max_results=max_results)

    return search


def build_agent_runtime_executor_bundle(
    settings: Settings,
    *,
    tool_registry: AgentToolRegistry | None = None,
    session_provider: Callable[[AgentRuntimeContext], Any] | None = None,
) -> tuple[dict[str, AbilityAgent], dict[str, str]]:
    if not settings.external_agent_auto_dispatch:
        return {}, {}
    agents: list[Any] = []

    openai_api_key = _openai_sdk_agent_api_key(settings)
    if (
        settings.openai_sdk_agent_enabled
        and openai_api_key
        and settings.openai_sdk_agent_mode == OPENAI_SDK_AGENT_MODE_CHAT_COMPLETIONS
    ):
        agents.append(
            OpenAISdkAgentExecutor(
                OpenAISdkResumeClientAdapter(config=_build_openai_sdk_agent_config(settings))
            )
        )
    if (
        settings.openai_sdk_agent_enabled
        and openai_api_key
        and settings.openai_sdk_agent_mode == OPENAI_SDK_AGENT_MODE_AGENTS_SDK
    ):
        agent_config = _build_openai_sdk_agent_config(settings)
        delegation_policy = build_sdk_agent_delegation_policy(settings, tool_registry=tool_registry)
        sdk_tool_names = list(delegation_policy.internal_tool_names)
        logger.info(
            "Configured OpenAI SDK child-agent delegation policy",
            extra={
                "sdk_tool_count": len(sdk_tool_names),
                "sdk_exposed_capabilities": list(delegation_policy.exposed_capability_names),
                "sdk_runtime_retained_tools": list(delegation_policy.runtime_retained_tool_names),
            },
        )
        sdk_tool_gateway = (
            SdkAgentToolGateway(tool_registry, session_provider=session_provider)
            if tool_registry is not None and sdk_tool_names
            else None
        )
        mail_tool_names = tuple(delegation_policy.tools_by_capability.get(QQ_MAIL_READONLY_AGENT_CAPABILITY, ()))
        general_tool_names = [
            name for name in sdk_tool_names if not name.startswith(("mcp.qq_mail.", "mcp.qq-mail.", "mcp.qqmail."))
        ]
        general_delegation_policy = _delegation_without_qq_mail(delegation_policy)
        agents.append(
            OpenAISdkAgentRunnerExecutor(
                OpenAISdkAgentRunnerAdapter(
                    runner_client=OpenAIAgentsSdkRunnerClient(
                        model=agent_config.model,
                        api_key=agent_config.api_key,
                        base_url=agent_config.base_url,
                        timeout_seconds=agent_config.timeout_seconds,
                    ),
                    subagent_name="OfferMasterSdkAgent",
                    allowed_tools=general_tool_names,
                    risk_policy={
                        "mutation_tools": "approval_required",
                        "delegation": general_delegation_policy.to_metadata(),
                    },
                    max_turns=settings.sdk_agent_max_turns,
                    tool_gateway=sdk_tool_gateway,
                    sandbox_manager=_sdk_agent_sandbox_manager(settings),
                ),
                additional_capabilities=_sdk_agent_tool_capabilities(
                    tool_registry,
                    general_delegation_policy,
                ),
            )
        )
        if mail_tool_names:
            mail_delegation_policy = _qq_mail_delegation(delegation_policy, mail_tool_names)
            agents.append(
                OpenAISdkAgentRunnerExecutor(
                    OpenAISdkAgentRunnerAdapter(
                        runner_client=OpenAIAgentsSdkRunnerClient(
                            model=agent_config.model,
                            api_key=agent_config.api_key,
                            base_url=agent_config.base_url,
                            timeout_seconds=agent_config.timeout_seconds,
                        ),
                        subagent_name="QqMailReadOnlyAgent",
                        executor_id=OPENAI_SDK_QQ_MAIL_AGENT_EXECUTOR_ID,
                        allowed_tools=list(mail_tool_names),
                        risk_policy={
                            "mutation_tools": "approval_required",
                            "delegation": mail_delegation_policy.to_metadata(),
                        },
                        max_turns=settings.sdk_agent_max_turns,
                        tool_gateway=sdk_tool_gateway,
                    ),
                    executor_id=OPENAI_SDK_QQ_MAIL_AGENT_EXECUTOR_ID,
                    include_resume_tailoring=False,
                    additional_capabilities=_sdk_agent_tool_capabilities(
                        tool_registry,
                        mail_delegation_policy,
                        executor_id=OPENAI_SDK_QQ_MAIL_AGENT_EXECUTOR_ID,
                    ),
                )
            )

    provider = str(getattr(settings, "external_web_search_provider", "auto") or "auto").strip().lower()
    if provider != "bailian":
        base_url = str(settings.claude_sdk_agent_base_url or "").strip()
        if base_url:
            config = _build_claude_sdk_http_executor_config(settings, base_url=base_url)
            agents.append(ClaudeSdkAgentExecutor(ClaudeSdkHttpExecutorAdapter(config=config)))

    if not agents:
        return {}, {}

    bundle = build_agent_runtime_bundle(agents)
    return bundle.executors, bundle.capability_executor_ids


def _build_claude_sdk_http_executor_config(settings: Settings, *, base_url: str) -> ClaudeSdkHttpExecutorConfig:
    service_api_key = (
        settings.claude_sdk_agent_api_key.get_secret_value().strip()
        if settings.claude_sdk_agent_api_key is not None
        else None
    )
    provider_api_key = settings.llm_api_key.get_secret_value().strip() if settings.llm_api_key is not None else None
    return ClaudeSdkHttpExecutorConfig(
        base_url=base_url,
        model=settings.claude_sdk_agent_model,
        api_key=service_api_key,
        timeout_seconds=settings.claude_sdk_agent_timeout_seconds,
        provider_base_url=_anthropic_messages_base_url(settings.llm_base_url),
        provider_api_key=provider_api_key,
    )


def _build_openai_sdk_agent_config(settings: Settings) -> OpenAISdkAgentConfig:
    return OpenAISdkAgentConfig(
        base_url=str(settings.openai_sdk_agent_base_url or settings.llm_base_url or "").strip().rstrip("/") or None,
        model=str(settings.openai_sdk_agent_model or settings.llm_model or "").strip(),
        api_key=_openai_sdk_agent_api_key(settings) or None,
        timeout_seconds=settings.openai_sdk_agent_timeout_seconds,
    )


def _sdk_agent_sandbox_manager(settings: Settings) -> SdkAgentSandboxManager | None:
    if settings.sdk_agent_sandbox_mode != SDK_AGENT_SANDBOX_MODE_TEMP_COPY:
        return None
    return SdkAgentSandboxManager(base_dir=settings.exports_path.parent / "sdk-agent-runs")


def _sdk_agent_tool_capabilities(
    tool_registry: AgentToolRegistry | None,
    delegation_policy: SdkAgentDelegationPolicy,
    *,
    executor_id: str = OPENAI_SDK_AGENT_EXECUTOR_ID,
) -> tuple[AgentCapabilityDefinition, ...]:
    capabilities: list[AgentCapabilityDefinition] = []
    for tool_name in delegation_policy.exposed_capability_names:
        if tool_name == FILESYSTEM_SKILL_CAPABILITY:
            # Filesystem is exposed to the main runtime as one coarse Skill
            # capability, but the SDK child agent receives the detailed
            # skill-reading and filesystem tools through allowed_tools.
            capabilities.append(
                filesystem_skill_capability_definition(
                    executor_id=executor_id,
                )
            )
            continue
        if tool_name in {GOOGLE_CHROME_AGENT_CAPABILITY, DBX_READONLY_AGENT_CAPABILITY, QQ_MAIL_READONLY_AGENT_CAPABILITY}:
            is_chrome = tool_name == GOOGLE_CHROME_AGENT_CAPABILITY
            is_mail = tool_name == QQ_MAIL_READONLY_AGENT_CAPABILITY
            capabilities.append(
                AgentCapabilityDefinition(
                    capability_id=tool_name,
                    name=(
                        "Google Chrome 浏览器 Agent"
                        if is_chrome
                        else ("QQ 邮箱只读 Agent" if is_mail else "DBX 只读数据库 Agent")
                    ),
                    description=(
                        "接收主 Agent 委派的浏览器任务，只使用其对应的 Chrome MCP 工具；"
                        "点击、输入等会改变页面状态的操作必须先经 SDK 人工审批。"
                        if is_chrome
                        else (
                            "接收主 Agent 委派的 QQ 招聘邮件读取任务；只列出近期邮件或读取指定邮件，"
                            "使用 IMAPS 只读访问，不发送、删除、移动邮件，也不标记邮件为已读。"
                            if is_mail
                            else "接收主 Agent 委派的数据库只读查询任务；先发现连接，再查看数据库和表结构，"
                            "只使用 DBX 只读 MCP 工具，不执行写入操作。"
                        )
                    ),
                    executor_id=executor_id,
                    input_schema={
                        "type": "object",
                        "required": ["task"],
                        "properties": {
                            "task": {"type": "string", "minLength": 1, "description": "子 Agent 要完成的具体任务。"}
                        },
                        "additionalProperties": False,
                    },
                    output_schema={
                        "type": "object",
                        "required": ["status", "summary"],
                        "properties": {
                            "status": {"type": "string"},
                            "summary": {"type": "string"},
                            "evidence": {"type": "array"},
                            "diagnostics": {"type": "object"},
                        },
                    },
                    kind="agent",
                    risk_level="low",
                    always_available=True,
                    allowed_source_types=frozenset({"agent_chat"}),
                )
            )
            continue
        if tool_registry is None:
            continue
        definition = tool_registry.get(tool_name)
        if definition is None:
            continue
        risk_value = str(getattr(definition.risk_level, "value", definition.risk_level)).strip().lower()
        if risk_value != "low" or definition.requires_confirmation or definition.handler is None:
            continue
        capabilities.append(
            AgentCapabilityDefinition.from_tool_definition(
                definition,
                executor_id=executor_id,
                supported_intents=DEFAULT_SUPPORTED_INTENTS_BY_CAPABILITY.get(definition.name, ()),
            )
        )
    return tuple(capabilities)


def _delegation_without_qq_mail(policy: SdkAgentDelegationPolicy) -> SdkAgentDelegationPolicy:
    mail_prefixes = ("mcp.qq_mail.", "mcp.qq-mail.", "mcp.qqmail.")
    return replace(
        policy,
        internal_tool_names=tuple(name for name in policy.internal_tool_names if not name.startswith(mail_prefixes)),
        exposed_capability_names=tuple(
            name for name in policy.exposed_capability_names if name != QQ_MAIL_READONLY_AGENT_CAPABILITY
        ),
        reasons_by_tool={
            key: value for key, value in policy.reasons_by_tool.items() if not key.startswith(mail_prefixes)
        },
        tools_by_capability={
            key: value for key, value in policy.tools_by_capability.items() if key != QQ_MAIL_READONLY_AGENT_CAPABILITY
        },
        subagent_names_by_capability={
            key: value
            for key, value in policy.subagent_names_by_capability.items()
            if key != QQ_MAIL_READONLY_AGENT_CAPABILITY
        },
    )


def _qq_mail_delegation(
    policy: SdkAgentDelegationPolicy,
    mail_tool_names: tuple[str, ...],
) -> SdkAgentDelegationPolicy:
    return SdkAgentDelegationPolicy(
        internal_tool_names=mail_tool_names,
        exposed_capability_names=(QQ_MAIL_READONLY_AGENT_CAPABILITY,),
        runtime_retained_tool_names=(),
        reasons_by_tool={name: "qq_mail_readonly_agent" for name in mail_tool_names},
        tools_by_capability={QQ_MAIL_READONLY_AGENT_CAPABILITY: mail_tool_names},
        subagent_names_by_capability={
            QQ_MAIL_READONLY_AGENT_CAPABILITY: policy.subagent_names_by_capability.get(
                QQ_MAIL_READONLY_AGENT_CAPABILITY,
                "QqMailReadOnlyAgent",
            )
        },
    )


def _openai_sdk_agent_api_key(settings: Settings) -> str:
    dedicated_key = (
        settings.openai_sdk_agent_api_key.get_secret_value().strip()
        if settings.openai_sdk_agent_api_key is not None
        else ""
    )
    if dedicated_key:
        return dedicated_key
    return settings.llm_api_key.get_secret_value().strip() if settings.llm_api_key is not None else ""


def _anthropic_messages_base_url(base_url: str) -> str | None:
    normalized = str(base_url or "").strip().rstrip("/")
    if not normalized:
        return None
    suffixes = (
        "/compatible-mode/v1/chat/completions",
        "/compatible-mode/v1",
        "/v1/chat/completions",
    )
    lowered = normalized.lower()
    for suffix in suffixes:
        if lowered.endswith(suffix):
            return normalized[: -len(suffix)] + "/apps/anthropic"
    return normalized
