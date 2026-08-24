from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sqlalchemy.orm import Session

from app.agent_runtime.agent_as_tool import AbilityAgent, build_agent_runtime_bundle
from app.agent_runtime.external_tasks.dispatcher import ExternalTaskDispatcher
from app.agent_runtime.external_tasks.executors import (
    ClaudeSdkAgentExecutor,
    ClaudeSdkHttpExecutorAdapter,
    ClaudeSdkHttpExecutorConfig,
    _run_http_web_search,
)
from app.agent_runtime.external_tasks.repository import SqlAlchemyExternalAgentTaskRepository
from app.core.config import Settings


ExternalTaskDispatcherCallback = Callable[[Session, str], dict[str, Any]]
ExternalWebSearchCallback = Callable[[str, int], dict[str, Any]]


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


def build_agent_runtime_executor_bundle(settings: Settings) -> tuple[dict[str, AbilityAgent], dict[str, str]]:
    if not settings.external_agent_auto_dispatch:
        return {}, {}
    base_url = str(settings.claude_sdk_agent_base_url or "").strip()
    if not base_url:
        return {}, {}

    config = _build_claude_sdk_http_executor_config(settings, base_url=base_url)
    bundle = build_agent_runtime_bundle([ClaudeSdkAgentExecutor(ClaudeSdkHttpExecutorAdapter(config=config))])
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
