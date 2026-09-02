from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from app.domains.jobs.models import DomainHealth, DomainHealthState, JobSourceType, utc_now
from app.domains.jobs.schemas import ToolErrorCode, ToolResult, ToolSuggestedNextAction


FETCH_TOOLS = {
    "HTTPArticleFetcher",
    "WeChatArticleFetcher",
    "PlaywrightFetcher",
    "Crawl4AIFetcher",
    "MCPVisiblePageFetcher",
    "OfficialAPIFetcher",
}
LLM_TOOLS = {"BailianJobLeadExtractor"}
DEFAULT_ALLOWED_TOOLS = FETCH_TOOLS | LLM_TOOLS


@dataclass(frozen=True)
class ToolPolicy:
    allowed_tools: frozenset[str] = frozenset(DEFAULT_ALLOWED_TOOLS)
    source_type_tools: dict[JobSourceType, frozenset[str]] = field(default_factory=dict)
    max_tool_calls: int = 15
    max_llm_calls: int = 3
    max_fetch_attempts_per_stage: int = 3
    max_runtime_seconds: int = 180
    max_mcp_requests: int = 1
    open_after_failure_count: int = 3
    circuit_cooldown_seconds: int = 30 * 60
    max_half_open_probe_count: int = 1

    def tools_for_source_type(self, source_type: JobSourceType) -> frozenset[str]:
        if self.source_type_tools:
            return self.source_type_tools.get(source_type, frozenset())
        return _default_source_type_tools(source_type)


@dataclass(frozen=True)
class ToolCallContext:
    stage: str
    tool_name: str
    source_type: JobSourceType
    domain: str
    run_id: str | None = None
    tool_call_count: int = 0
    llm_call_count: int = 0
    fetch_attempts_for_stage: int = 0
    mcp_request_count: int = 0
    run_started_at: datetime | None = None
    now: datetime | None = None
    user_confirmed: bool = False

    @property
    def current_time(self) -> datetime:
        return self.now or utc_now()


class ToolRuntimeGuard:
    def __init__(self, policy: ToolPolicy | None = None) -> None:
        self._policy = policy or ToolPolicy()

    def pre_check(
        self,
        context: ToolCallContext,
        *,
        domain_health: DomainHealth | None = None,
    ) -> ToolResult:
        now = context.current_time
        if context.tool_name not in self._policy.allowed_tools:
            return self._blocked(
                context,
                ToolErrorCode.TOOL_NOT_ALLOWED,
                f"Tool is not in URL import allowlist: {context.tool_name}",
                ToolSuggestedNextAction.STOP_TERMINAL_FAILURE,
                retryable=False,
                error_details={
                    "category": "tool_allowlist",
                    "tool_name": context.tool_name,
                    "domain": context.domain,
                    "source_type": context.source_type.value,
                    "allowed_tools": sorted(self._policy.allowed_tools),
                },
            )

        allowed_for_source = self._policy.tools_for_source_type(context.source_type)
        if context.tool_name not in allowed_for_source:
            next_action = (
                ToolSuggestedNextAction.REQUEST_USER_VISIBLE_PAGE
                if _requires_visible_page_source(context.source_type)
                else ToolSuggestedNextAction.RETRY_WITH_NEXT_FETCHER
            )
            return self._blocked(
                context,
                ToolErrorCode.SOURCE_TYPE_NOT_ALLOWED,
                f"Tool {context.tool_name} is not allowed for source type {context.source_type.value}",
                next_action,
                retryable=False,
                error_details={
                    "category": "source_type_policy",
                    "tool_name": context.tool_name,
                    "domain": context.domain,
                    "source_type": context.source_type.value,
                    "allowed_tools_for_source": sorted(allowed_for_source),
                },
            )

        if context.tool_name == "MCPVisiblePageFetcher" and not context.user_confirmed:
            return self._blocked(
                context,
                ToolErrorCode.MCP_USER_CONFIRMATION_REQUIRED,
                "MCP visible page access requires explicit user confirmation",
                ToolSuggestedNextAction.REQUEST_USER_VISIBLE_PAGE,
                retryable=True,
                error_details={
                    "category": "user_confirmation_required",
                    "tool_name": context.tool_name,
                    "domain": context.domain,
                    "source_type": context.source_type.value,
                },
            )

        budget_result = self._check_budgets(context, now)
        if budget_result is not None:
            return budget_result

        circuit_result = self._check_circuit(context, domain_health, now)
        if circuit_result is not None:
            return circuit_result

        return ToolResult(
            ok=True,
            stage=context.stage,
            tool_name=context.tool_name,
            suggested_next_action=ToolSuggestedNextAction.CONTINUE_WORKFLOW,
            cost={
                "tool_calls": context.tool_call_count,
                "llm_calls": context.llm_call_count,
                "fetch_attempts_for_stage": context.fetch_attempts_for_stage,
                "mcp_requests": context.mcp_request_count,
            },
            artifacts={"domain": context.domain, "source_type": context.source_type.value},
        )

    def post_record(
        self,
        result: ToolResult,
        domain_health: DomainHealth | None,
        *,
        now: datetime | None = None,
    ) -> DomainHealth | None:
        if domain_health is None:
            return None

        current_time = now or utc_now()
        if result.ok:
            domain_health.state = DomainHealthState.CLOSED
            domain_health.failure_count = 0
            domain_health.success_count = (domain_health.success_count or 0) + 1
            domain_health.last_error_code = None
            domain_health.last_error_message = None
            domain_health.opened_at = None
            domain_health.cooldown_until = None
            domain_health.half_open_probe_count = 0
            domain_health.updated_at = current_time
            return domain_health

        domain_health.failure_count = (domain_health.failure_count or 0) + 1
        domain_health.last_error_code = _enum_or_value(result.error_code)
        domain_health.last_error_message = result.error_message
        domain_health.updated_at = current_time
        if domain_health.failure_count >= self._policy.open_after_failure_count:
            domain_health.state = DomainHealthState.OPEN
            domain_health.opened_at = current_time
            domain_health.cooldown_until = current_time + timedelta(
                seconds=self._policy.circuit_cooldown_seconds
            )
            domain_health.half_open_probe_count = 0
        elif domain_health.state is None:
            domain_health.state = DomainHealthState.CLOSED
        return domain_health

    def _check_budgets(self, context: ToolCallContext, now: datetime) -> ToolResult | None:
        if context.tool_call_count >= self._policy.max_tool_calls:
            return self._blocked(
                context,
                ToolErrorCode.TOOL_BUDGET_EXCEEDED,
                "URL import tool call budget exceeded",
                ToolSuggestedNextAction.STOP_TERMINAL_FAILURE,
                retryable=False,
                error_details={
                    "category": "budget",
                    "budget_name": "max_tool_calls",
                    "limit": self._policy.max_tool_calls,
                    "actual": context.tool_call_count,
                },
            )
        if context.tool_name in LLM_TOOLS and context.llm_call_count >= self._policy.max_llm_calls:
            return self._blocked(
                context,
                ToolErrorCode.LLM_BUDGET_EXCEEDED,
                "URL import LLM call budget exceeded",
                ToolSuggestedNextAction.STOP_TERMINAL_FAILURE,
                retryable=False,
                error_details={
                    "category": "budget",
                    "budget_name": "max_llm_calls",
                    "limit": self._policy.max_llm_calls,
                    "actual": context.llm_call_count,
                },
            )
        if context.tool_name in FETCH_TOOLS and (
            context.fetch_attempts_for_stage >= self._policy.max_fetch_attempts_per_stage
        ):
            return self._blocked(
                context,
                ToolErrorCode.FETCH_ATTEMPTS_EXCEEDED,
                "URL import fetch attempts exceeded for current stage",
                ToolSuggestedNextAction.RETRY_WITH_NEXT_FETCHER,
                retryable=True,
                error_details={
                    "category": "budget",
                    "budget_name": "max_fetch_attempts_per_stage",
                    "limit": self._policy.max_fetch_attempts_per_stage,
                    "actual": context.fetch_attempts_for_stage,
                },
            )
        if context.run_started_at is not None:
            elapsed_seconds = (now - context.run_started_at).total_seconds()
            if elapsed_seconds > self._policy.max_runtime_seconds:
                return self._blocked(
                    context,
                    ToolErrorCode.TIME_BUDGET_EXCEEDED,
                    "URL import runtime budget exceeded",
                    ToolSuggestedNextAction.STOP_TERMINAL_FAILURE,
                    retryable=False,
                    error_details={
                        "category": "budget",
                        "budget_name": "max_runtime_seconds",
                        "limit": self._policy.max_runtime_seconds,
                        "actual": elapsed_seconds,
                    },
                )
        if (
            context.tool_name == "MCPVisiblePageFetcher"
            and context.mcp_request_count >= self._policy.max_mcp_requests
        ):
            return self._blocked(
                context,
                ToolErrorCode.TOOL_BUDGET_EXCEEDED,
                "URL import MCP request budget exceeded",
                ToolSuggestedNextAction.REQUEST_MANUAL_PASTE,
                retryable=True,
                error_details={
                    "category": "budget",
                    "budget_name": "max_mcp_requests",
                    "limit": self._policy.max_mcp_requests,
                    "actual": context.mcp_request_count,
                },
            )
        return None

    def _check_circuit(
        self,
        context: ToolCallContext,
        domain_health: DomainHealth | None,
        now: datetime,
    ) -> ToolResult | None:
        if domain_health is None:
            return None
        state = domain_health.state or DomainHealthState.UNKNOWN
        if state == DomainHealthState.OPEN:
            cooldown_until = domain_health.cooldown_until
            if cooldown_until is not None and cooldown_until > now:
                return self._blocked(
                    context,
                    ToolErrorCode.TOOL_CIRCUIT_OPEN,
                    "Domain/tool circuit is open",
                    ToolSuggestedNextAction.WAIT_FOR_COOLDOWN,
                    retryable=True,
                    error_details={
                        "category": "circuit",
                        "domain": context.domain,
                        "tool_name": context.tool_name,
                        "state": state.value,
                        "cooldown_until": cooldown_until.isoformat(),
                    },
                )
            domain_health.state = DomainHealthState.HALF_OPEN
            domain_health.half_open_probe_count = (domain_health.half_open_probe_count or 0) + 1
            domain_health.updated_at = now
            return None
        if state == DomainHealthState.HALF_OPEN and (
            (domain_health.half_open_probe_count or 0) >= self._policy.max_half_open_probe_count
        ):
            return self._blocked(
                context,
                ToolErrorCode.TOOL_CIRCUIT_OPEN,
                "Half-open circuit already used its probe budget",
                ToolSuggestedNextAction.WAIT_FOR_COOLDOWN,
                retryable=True,
                error_details={
                    "category": "circuit",
                    "domain": context.domain,
                    "tool_name": context.tool_name,
                    "state": state.value,
                    "half_open_probe_count": domain_health.half_open_probe_count or 0,
                    "max_half_open_probe_count": self._policy.max_half_open_probe_count,
                },
            )
        return None

    @staticmethod
    def _blocked(
        context: ToolCallContext,
        error_code: ToolErrorCode,
        error_message: str,
        next_action: ToolSuggestedNextAction,
        *,
        retryable: bool,
        error_details: dict[str, object] | None = None,
    ) -> ToolResult:
        return ToolResult(
            ok=False,
            stage=context.stage,
            tool_name=context.tool_name,
            error_code=error_code,
            error_message=error_message,
            retryable=retryable,
            suggested_next_action=next_action,
            error_details=error_details or {},
            cost={
                "tool_calls": context.tool_call_count,
                "llm_calls": context.llm_call_count,
                "fetch_attempts_for_stage": context.fetch_attempts_for_stage,
                "mcp_requests": context.mcp_request_count,
            },
            artifacts={"domain": context.domain, "source_type": context.source_type.value},
        )


def _default_source_type_tools(source_type: JobSourceType) -> frozenset[str]:
    mapping = {
        JobSourceType.PUBLIC_ARTICLE: frozenset(
            {"HTTPArticleFetcher", "PlaywrightFetcher", "Crawl4AIFetcher", "BailianJobLeadExtractor"}
        ),
        JobSourceType.UNIVERSITY_CAREER_SITE: frozenset(
            {"HTTPArticleFetcher", "PlaywrightFetcher", "Crawl4AIFetcher", "BailianJobLeadExtractor"}
        ),
        JobSourceType.OFFICIAL_CAREER_SITE: frozenset(
            {"HTTPArticleFetcher", "PlaywrightFetcher", "Crawl4AIFetcher", "BailianJobLeadExtractor"}
        ),
        JobSourceType.WECHAT_ARTICLE: frozenset(
            {"WeChatArticleFetcher", "PlaywrightFetcher", "Crawl4AIFetcher", "BailianJobLeadExtractor"}
        ),
        JobSourceType.XIAOHONGSHU_NOTE: frozenset(
            {"MCPVisiblePageFetcher", "BailianJobLeadExtractor"}
        ),
        JobSourceType.JOB_BOARD_VISIBLE_PAGE: frozenset(
            {"MCPVisiblePageFetcher", "BailianJobLeadExtractor"}
        ),
        JobSourceType.OFFICIAL_API: frozenset({"OfficialAPIFetcher", "BailianJobLeadExtractor"}),
        JobSourceType.MANUAL_CLIP: frozenset({"BailianJobLeadExtractor"}),
    }
    return mapping[source_type]


def _requires_visible_page_source(source_type: JobSourceType) -> bool:
    return source_type in {JobSourceType.XIAOHONGSHU_NOTE, JobSourceType.JOB_BOARD_VISIBLE_PAGE}


def _enum_or_value(value: object) -> str | None:
    if value is None:
        return None
    return getattr(value, "value", str(value))
