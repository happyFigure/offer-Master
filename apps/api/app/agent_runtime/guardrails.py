from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.agent_runtime.tool_permissions import (
    AgentToolPermissionDecision,
    AgentToolPermissionPolicy,
    AgentToolPermissionResult,
)
from app.agent_runtime.tool_registry import (
    DATABASE_COMPANY_PROFILE_TOOL,
    DATABASE_COMPANY_SEARCH_TOOL,
    DATABASE_JOB_SEARCH_TOOL,
    DATABASE_SOURCE_SEARCH_TOOL,
    EXTERNAL_WEB_SEARCH_TOOL,
    LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL,
    LOCAL_JOB_SOURCE_OVERVIEW_TOOL,
    OFFERIO_COMPANY_JOBS_TOOL,
    AgentToolDefinition,
    AgentToolRegistry,
    AgentToolRiskLevel,
)


class AgentToolErrorCode(str, Enum):
    TOOL_NOT_REGISTERED = "TOOL_NOT_REGISTERED"
    TOOL_BUDGET_EXCEEDED = "TOOL_BUDGET_EXCEEDED"
    TOOL_SOURCE_TYPE_NOT_ALLOWED = "TOOL_SOURCE_TYPE_NOT_ALLOWED"
    TOOL_USER_CONFIRMATION_REQUIRED = "TOOL_USER_CONFIRMATION_REQUIRED"
    TOOL_SKILL_DENIED = "TOOL_SKILL_DENIED"
    TOOL_SKILL_CONFIRMATION_REQUIRED = "TOOL_SKILL_CONFIRMATION_REQUIRED"


class AgentToolNextAction(str, Enum):
    CONTINUE = "continue"
    STOP = "stop"
    SELECT_ALTERNATIVE_TOOL = "select_alternative_tool"
    REQUEST_USER_CONFIRMATION = "request_user_confirmation"


LOW_RISK_RUNTIME_CAPABILITY_TOOLS = frozenset(
    {
        EXTERNAL_WEB_SEARCH_TOOL,
        DATABASE_COMPANY_SEARCH_TOOL,
        DATABASE_COMPANY_PROFILE_TOOL,
        DATABASE_JOB_SEARCH_TOOL,
        DATABASE_SOURCE_SEARCH_TOOL,
        LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL,
        LOCAL_JOB_SOURCE_OVERVIEW_TOOL,
        OFFERIO_COMPANY_JOBS_TOOL,
        "resume.tailor",
    }
)


@dataclass(frozen=True)
class AgentToolPolicy:
    max_tool_calls: int = 10


@dataclass(frozen=True)
class AgentToolCallContext:
    stage: str
    tool_name: str
    source_type: str | None = None
    tool_call_count: int = 0
    user_confirmed: bool = False
    agent_run_id: str | None = None
    session_id: str | None = None


@dataclass(frozen=True)
class AgentToolGuardResult:
    ok: bool
    stage: str
    tool_name: str
    error_code: str | None = None
    reason: str | None = None
    user_message: str | None = None
    next_action: str = AgentToolNextAction.CONTINUE.value
    retryable: bool = False
    error_details: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)


class AgentToolRuntimeGuard:
    def __init__(self, policy: AgentToolPolicy | None = None) -> None:
        self._policy = policy or AgentToolPolicy()

    def pre_check(
        self,
        context: AgentToolCallContext,
        *,
        registry: AgentToolRegistry,
        skill_permission_policy: AgentToolPermissionPolicy | None = None,
    ) -> AgentToolGuardResult:
        definition = registry.get(context.tool_name)
        if definition is None:
            return self._blocked(
                context,
                AgentToolErrorCode.TOOL_NOT_REGISTERED,
                reason=f"Agent tool is not registered: {context.tool_name}",
                user_message=f"Tool {context.tool_name} is not registered for this agent.",
                next_action=AgentToolNextAction.STOP,
                retryable=False,
                error_details={"registered_tools": registry.registered_tool_names()},
            )

        if context.tool_call_count >= self._policy.max_tool_calls:
            return self._blocked(
                context,
                AgentToolErrorCode.TOOL_BUDGET_EXCEEDED,
                reason="Agent tool call budget exceeded",
                user_message="Agent has reached its tool call budget for this run.",
                next_action=AgentToolNextAction.STOP,
                retryable=False,
                error_details={
                    "budget_name": "max_tool_calls",
                    "limit": self._policy.max_tool_calls,
                    "actual": context.tool_call_count,
                },
                cost={"tool_calls": context.tool_call_count, "max_tool_calls": self._policy.max_tool_calls},
            )

        skill_permission_result = None
        skill_permission_decision = None
        if skill_permission_policy is not None:
            skill_permission_result = skill_permission_policy.decide(
                context.tool_name,
                user_confirmed=context.user_confirmed,
            )
            skill_permission_decision = skill_permission_result.decision.value
            if skill_permission_result.decision == AgentToolPermissionDecision.DENY:
                return self._blocked(
                    context,
                    AgentToolErrorCode.TOOL_SKILL_DENIED,
                    reason=skill_permission_result.reason,
                    user_message="This tool is denied by the active Skill policy.",
                    next_action=AgentToolNextAction.STOP,
                    retryable=False,
                    error_details=skill_permission_result.error_details,
                )
            if skill_permission_result.decision == AgentToolPermissionDecision.ASK:
                if self._allows_low_risk_runtime_capability(context, definition, skill_permission_result):
                    skill_permission_decision = "allow_low_risk_runtime_capability"
                else:
                    return self._blocked(
                        context,
                        AgentToolErrorCode.TOOL_SKILL_CONFIRMATION_REQUIRED,
                        reason=skill_permission_result.reason,
                        user_message="This tool requires user confirmation because it is outside the active Skill automatic permissions.",
                        next_action=AgentToolNextAction.REQUEST_USER_CONFIRMATION,
                        retryable=True,
                        error_details=skill_permission_result.error_details,
                    )

        source_type_result = self._check_source_type(context, definition)
        if source_type_result is not None:
            return source_type_result

        if definition.requires_confirmation and not context.user_confirmed:
            return self._blocked(
                context,
                AgentToolErrorCode.TOOL_USER_CONFIRMATION_REQUIRED,
                reason=f"Tool requires explicit user confirmation: {context.tool_name}",
                user_message="This tool requires explicit user confirmation before it can run.",
                next_action=AgentToolNextAction.REQUEST_USER_CONFIRMATION,
                retryable=True,
                error_details={
                    "risk_level": definition.risk_level.value,
                    "requires_confirmation": definition.requires_confirmation,
                },
            )

        return AgentToolGuardResult(
            ok=True,
            stage=context.stage,
            tool_name=context.tool_name,
            next_action=AgentToolNextAction.CONTINUE.value,
            cost={"tool_calls": context.tool_call_count, "max_tool_calls": self._policy.max_tool_calls},
            artifacts={
                "source_type": context.source_type,
                "risk_level": definition.risk_level.value,
                "requires_confirmation": definition.requires_confirmation,
                "skill_id": skill_permission_result.skill_id if skill_permission_result else None,
                "skill_permission_decision": skill_permission_decision,
            },
        )

    @staticmethod
    def _allows_low_risk_runtime_capability(
        context: AgentToolCallContext,
        definition: AgentToolDefinition,
        skill_permission_result: AgentToolPermissionResult,
    ) -> bool:
        return (
            context.tool_name in LOW_RISK_RUNTIME_CAPABILITY_TOOLS
            and definition.risk_level in {AgentToolRiskLevel.LOW, AgentToolRiskLevel.MEDIUM}
            and not definition.requires_confirmation
            and context.tool_name not in skill_permission_result.ask_tools
            and context.tool_name not in skill_permission_result.allowed_tools
        )

    def _check_source_type(
        self,
        context: AgentToolCallContext,
        definition: AgentToolDefinition,
    ) -> AgentToolGuardResult | None:
        if not definition.allowed_source_types:
            return None
        if context.source_type in definition.allowed_source_types:
            return None
        return self._blocked(
            context,
            AgentToolErrorCode.TOOL_SOURCE_TYPE_NOT_ALLOWED,
            reason=f"Tool {context.tool_name} is not allowed for source type {context.source_type}",
            user_message="This tool is not allowed for the current source type.",
            next_action=AgentToolNextAction.SELECT_ALTERNATIVE_TOOL,
            retryable=False,
            error_details={
                "source_type": context.source_type,
                "allowed_source_types": sorted(definition.allowed_source_types),
            },
        )

    @staticmethod
    def _blocked(
        context: AgentToolCallContext,
        error_code: AgentToolErrorCode,
        *,
        reason: str,
        user_message: str,
        next_action: AgentToolNextAction,
        retryable: bool,
        error_details: dict[str, Any] | None = None,
        cost: dict[str, Any] | None = None,
    ) -> AgentToolGuardResult:
        return AgentToolGuardResult(
            ok=False,
            stage=context.stage,
            tool_name=context.tool_name,
            error_code=error_code.value,
            reason=reason,
            user_message=user_message,
            next_action=next_action.value,
            retryable=retryable,
            error_details=error_details or {},
            cost=cost or {"tool_calls": context.tool_call_count},
            artifacts={"source_type": context.source_type},
        )
