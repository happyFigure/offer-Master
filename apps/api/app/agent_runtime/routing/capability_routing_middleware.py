from __future__ import annotations

from typing import Any

from app.agent_runtime.routing.schemas import RouteDecision
from app.agent_runtime.tool_input import requested_sample_limit_from_text
from app.agent_runtime.tool_registry import (
    APPLICATION_FIND_APPLY_ENTRY_TOOL,
    EXTERNAL_WEB_SEARCH_TOOL,
    LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL,
    LOCAL_JOB_SOURCE_OVERVIEW_TOOL,
    OFFERIO_COMPANY_JOBS_TOOL,
)


class CapabilityRoutingMiddleware:
    """Deterministic Planner Gate for selecting the next execution route."""

    def decide(
        self,
        *,
        user_message: str,
        intent_frame: dict[str, Any],
        context_pack: dict[str, Any],
    ) -> RouteDecision:
        intent = _text(intent_frame.get("intent") or context_pack.get("intent") or "normal_chat")
        risk_level = _text(intent_frame.get("risk_level") or context_pack.get("risk_level") or "low")
        allowed_capabilities = _string_list(context_pack.get("allowed_capabilities"))
        excluded_capabilities = _string_list(context_pack.get("excluded_capabilities"))

        if risk_level in {"critical"}:
            return RouteDecision(
                route="block",
                executor_type="runtime_guard",
                confidence=1.0,
                reason=f"risk_level={risk_level} requires runtime block before execution",
                allowed_capabilities=allowed_capabilities,
                blocked_capabilities=excluded_capabilities,
                requires_confirmation=True,
            )

        if risk_level == "high":
            return RouteDecision(
                route="ask_user",
                executor_type="human_approval",
                confidence=1.0,
                reason="risk_level=high requires user confirmation before execution",
                allowed_capabilities=allowed_capabilities,
                blocked_capabilities=excluded_capabilities,
                requires_confirmation=True,
            )

        clarification = _clarification_decision(
            user_message=user_message,
            intent=intent,
            intent_frame=intent_frame,
            context_pack=context_pack,
            allowed_capabilities=allowed_capabilities,
            excluded_capabilities=excluded_capabilities,
        )
        if clarification is not None:
            return clarification

        if intent == "normal_chat" or not allowed_capabilities:
            return RouteDecision(
                route="chat_direct",
                executor_type="chat",
                confidence=1.0,
                reason=f"{intent} without executable routed capability",
                allowed_capabilities=allowed_capabilities,
                blocked_capabilities=excluded_capabilities,
            )

        if intent == "campus_recruiting_search" and EXTERNAL_WEB_SEARCH_TOOL in allowed_capabilities:
            return RouteDecision(
                route="external_agent",
                capability=EXTERNAL_WEB_SEARCH_TOOL,
                executor_type="external_agent",
                executor_name="claude_sdk_agent",
                confidence=0.95,
                reason="campus_recruiting_search requires fresh external recruiting information",
                allowed_capabilities=allowed_capabilities,
                blocked_capabilities=excluded_capabilities,
                tool_input=_campus_search_tool_input(user_message, intent_frame, context_pack),
                metadata={"intent": intent},
            )

        if intent == "offerio_company_jobs_sync" and OFFERIO_COMPANY_JOBS_TOOL in allowed_capabilities:
            sync_policy = context_pack.get("sync_policy") if isinstance(context_pack.get("sync_policy"), dict) else {}
            limit = int(sync_policy.get("default_limit") or 1000)
            return RouteDecision(
                route="local_workflow",
                capability=OFFERIO_COMPANY_JOBS_TOOL,
                executor_type="local_workflow",
                executor_name="offerio_company_jobs_sync",
                confidence=1.0,
                reason="offerio_company_jobs_sync is a deterministic local sync workflow",
                allowed_capabilities=allowed_capabilities,
                blocked_capabilities=excluded_capabilities,
                tool_input={"limit": limit},
                metadata={"sync_policy": dict(sync_policy)},
            )

        if intent == "local_company_database_overview" and LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL in allowed_capabilities:
            sync_policy = context_pack.get("sync_policy") if isinstance(context_pack.get("sync_policy"), dict) else {}
            sample_limit = requested_sample_limit_from_text(user_message, default=int(sync_policy.get("sample_limit") or 10))
            return RouteDecision(
                route="local_workflow",
                capability=LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL,
                executor_type="local_workflow",
                executor_name="company_database_overview",
                confidence=1.0,
                reason="local_company_database_overview is a read-only local database overview workflow",
                allowed_capabilities=allowed_capabilities,
                blocked_capabilities=excluded_capabilities,
                tool_input={"sample_limit": sample_limit},
                metadata={"read_only": True, "sync_policy": dict(sync_policy)},
            )

        if intent == "local_job_source_overview" and LOCAL_JOB_SOURCE_OVERVIEW_TOOL in allowed_capabilities:
            sync_policy = context_pack.get("sync_policy") if isinstance(context_pack.get("sync_policy"), dict) else {}
            sample_limit = requested_sample_limit_from_text(user_message, default=int(sync_policy.get("sample_limit") or 10))
            include_external = bool(sync_policy.get("include_external_job_board", True))
            return RouteDecision(
                route="local_workflow",
                capability=LOCAL_JOB_SOURCE_OVERVIEW_TOOL,
                executor_type="local_workflow",
                executor_name="job_source_overview",
                confidence=1.0,
                reason="local_job_source_overview is a read-only local job-source and job-board overview workflow",
                allowed_capabilities=allowed_capabilities,
                blocked_capabilities=excluded_capabilities,
                tool_input={"sample_limit": sample_limit, "include_external_job_board": include_external},
                metadata={"read_only": True, "sync_policy": dict(sync_policy)},
            )

        if intent == "application_entry_discovery" and APPLICATION_FIND_APPLY_ENTRY_TOOL in allowed_capabilities:
            job_id = _first_entity_value(intent_frame, context_pack, "job_ids")
            return RouteDecision(
                route="browser_executor",
                capability=APPLICATION_FIND_APPLY_ENTRY_TOOL,
                executor_type="browser_executor",
                executor_name="codex_or_multica",
                confidence=0.9,
                reason="application_entry_discovery requires a bounded browser/external executor task",
                allowed_capabilities=allowed_capabilities,
                blocked_capabilities=excluded_capabilities,
                requires_confirmation=True,
                tool_input={"job_id": job_id} if job_id else {},
                metadata={"stop_before_submit": True},
            )

        if intent in {"job_match_analysis", "resume_tailoring"}:
            return RouteDecision(
                route="execution_planner",
                executor_type="execution_planner",
                confidence=0.9,
                reason=f"{intent} is a multi-step reasoning task",
                allowed_capabilities=allowed_capabilities,
                blocked_capabilities=excluded_capabilities,
                max_steps=3,
            )

        return RouteDecision(
            route="chat_direct",
            executor_type="chat",
            confidence=0.6,
            reason=f"no deterministic execution route for intent={intent}",
            allowed_capabilities=allowed_capabilities,
            blocked_capabilities=excluded_capabilities,
        )


def _campus_search_tool_input(
    user_message: str,
    intent_frame: dict[str, Any],
    context_pack: dict[str, Any],
) -> dict[str, Any]:
    company_name = _first_entity_value(intent_frame, context_pack, "company_names")
    if company_name:
        query = f"{company_name} 校园招聘 官网"
    else:
        query = _text(user_message)
    return {"query": query, "max_results": 5}


def _clarification_decision(
    *,
    user_message: str,
    intent: str,
    intent_frame: dict[str, Any],
    context_pack: dict[str, Any],
    allowed_capabilities: list[str],
    excluded_capabilities: list[str],
) -> RouteDecision | None:
    if intent == "campus_recruiting_search":
        entity_message = _campus_entity_ambiguity_message(user_message, intent_frame, context_pack)
        if entity_message:
            return RouteDecision(
                route="ask_user",
                executor_type="clarification",
                confidence=_confidence(intent_frame, context_pack),
                reason=entity_message,
                allowed_capabilities=allowed_capabilities,
                blocked_capabilities=excluded_capabilities,
                requires_confirmation=True,
                metadata={
                    "clarification_required": True,
                    "clarification_kind": "entity_ambiguity",
                    "ask_user_message": entity_message,
                },
            )

    candidate_intents = _string_list(intent_frame.get("candidate_intents"))
    if intent != "normal_chat" and allowed_capabilities and len(candidate_intents) > 1 and _confidence(intent_frame, context_pack) < 0.7:
        message = "我需要先确认一下你的意图：你是想查询招聘/岗位信息，还是想做普通网页搜索？"
        return RouteDecision(
            route="ask_user",
            executor_type="clarification",
            confidence=_confidence(intent_frame, context_pack),
            reason=message,
            allowed_capabilities=allowed_capabilities,
            blocked_capabilities=excluded_capabilities,
            requires_confirmation=True,
            metadata={
                "clarification_required": True,
                "clarification_kind": "intent_ambiguity",
                "ask_user_message": message,
                "candidate_intents": candidate_intents,
            },
        )
    return None


def _campus_entity_ambiguity_message(
    user_message: str,
    intent_frame: dict[str, Any],
    context_pack: dict[str, Any],
) -> str | None:
    company_names = _entity_values(intent_frame, context_pack, "company_names")
    if "公牛" not in company_names:
        return None
    if _has_recruiting_cue(user_message, intent_frame, context_pack) and _confidence(intent_frame, context_pack) >= 0.75:
        return None
    return "你说的“公牛”是指公牛集团的招聘/校招信息，还是芝加哥公牛队相关信息？确认后我再继续搜索。"


def _entity_values(intent_frame: dict[str, Any], context_pack: dict[str, Any], field_name: str) -> list[str]:
    result: list[str] = []
    for source in (intent_frame.get("entities"), context_pack.get("entities")):
        if not isinstance(source, dict):
            continue
        values = source.get(field_name)
        if isinstance(values, list):
            for value in values:
                text = _text(value)
                if text and text not in result:
                    result.append(text)
            continue
        text = _text(values)
        if text and text not in result:
            result.append(text)
    return result


def _has_recruiting_cue(user_message: str, intent_frame: dict[str, Any], context_pack: dict[str, Any]) -> bool:
    text = " ".join([_text(user_message), *_entity_values(intent_frame, context_pack, "keywords")]).lower()
    return any(keyword in text for keyword in ["校招", "校园招聘", "秋招", "招聘", "岗位", "网申", "应届生", "campus"])


def _confidence(intent_frame: dict[str, Any], context_pack: dict[str, Any]) -> float:
    for value in (intent_frame.get("confidence"), context_pack.get("confidence")):
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _first_entity_value(intent_frame: dict[str, Any], context_pack: dict[str, Any], field_name: str) -> str | None:
    for source in (intent_frame.get("entities"), context_pack.get("entities")):
        if not isinstance(source, dict):
            continue
        values = source.get(field_name)
        if isinstance(values, list):
            for value in values:
                text = _text(value)
                if text:
                    return text
        text = _text(values)
        if text:
            return text
    return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for item in value if (text := _text(item))]


def _text(value: Any) -> str:
    return str(value or "").strip()
