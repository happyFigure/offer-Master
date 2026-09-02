from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field, is_dataclass
import time
from typing import Any, Protocol

from app.agent_runtime.reflection.schemas import CapabilityResultEvaluationSpec, result_evaluation_spec_for_capability
from app.agent_runtime.tool_result_envelope import build_tool_result_envelope


TOOL_REGISTRY_EXECUTOR_ID = "agent_tool_registry"
CLAUDE_SDK_AGENT_EXECUTOR_ID = "claude-sdk-agent"
OPENAI_SDK_AGENT_EXECUTOR_ID = "openai-sdk-agent"


_SPECIALIZED_RESULT_ENVELOPE_CAPABILITIES = {
    "external.web_search",
    "applications.find_apply_entry",
}


DEFAULT_SUPPORTED_INTENTS_BY_CAPABILITY: dict[str, tuple[str, ...]] = {
    "external.web_search": ("campus_recruiting_search", "external_agent_task"),
    "local.company_database_overview": ("local_company_database_overview",),
    "database.company_list": ("local_company_database_list",),
    "local.job_source_overview": ("local_job_source_overview",),
    "offerio.sync_company_jobs": ("offerio_company_jobs_sync",),
    "applications.find_apply_entry": ("application_entry_discovery",),
    "resume.tailor": ("resume_tailoring",),
    "memory_search": ("memory_lookup",),
    "sessions_search": ("memory_lookup",),
    "sessions_history": ("memory_lookup",),
}


@dataclass(frozen=True)
class AgentCapabilityDefinition:
    capability_id: str
    name: str
    description: str
    executor_id: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    risk_level: str = "low"
    supported_intents: tuple[str, ...] = field(default_factory=tuple)
    requires_confirmation: bool = False
    allowed_source_types: frozenset[str] = field(default_factory=frozenset)
    result_evaluation: CapabilityResultEvaluationSpec | None = None
    candidate_profile: Any | None = None

    def __post_init__(self) -> None:
        if not self.capability_id.strip():
            raise ValueError("Agent capability id is required")
        if not self.name.strip():
            raise ValueError(f"Agent capability name is required: {self.capability_id}")
        if not self.description.strip():
            raise ValueError(f"Agent capability description is required: {self.capability_id}")
        if not self.executor_id.strip():
            raise ValueError(f"Agent capability executor id is required: {self.capability_id}")
        object.__setattr__(self, "supported_intents", tuple(str(item) for item in self.supported_intents))
        object.__setattr__(
            self,
            "allowed_source_types",
            frozenset(str(item) for item in self.allowed_source_types if str(item).strip()),
        )
        if self.result_evaluation is None:
            object.__setattr__(
                self,
                "result_evaluation",
                result_evaluation_spec_for_capability(
                    self.capability_id,
                    supported_intents=self.supported_intents,
                ),
            )

    @classmethod
    def from_tool_definition(
        cls,
        tool_definition: Any,
        *,
        executor_id: str,
        supported_intents: tuple[str, ...] = (),
    ) -> AgentCapabilityDefinition:
        risk_level = getattr(tool_definition, "risk_level", "low")
        risk_value = str(getattr(risk_level, "value", risk_level))
        return cls(
            capability_id=str(tool_definition.name),
            name=str(tool_definition.name),
            description=str(tool_definition.description),
            executor_id=executor_id,
            input_schema=dict(tool_definition.input_schema),
            output_schema=dict(tool_definition.output_schema),
            risk_level=risk_value,
            supported_intents=supported_intents,
            requires_confirmation=bool(getattr(tool_definition, "requires_confirmation", False)),
            allowed_source_types=frozenset(getattr(tool_definition, "allowed_source_types", frozenset())),
            result_evaluation=getattr(tool_definition, "result_evaluation", None),
            candidate_profile=getattr(tool_definition, "candidate_profile", None),
        )


class AgentCapabilityRegistry:
    def __init__(self, definitions: list[AgentCapabilityDefinition] | None = None) -> None:
        self._definitions: dict[str, AgentCapabilityDefinition] = {}
        for definition in definitions or []:
            self.register(definition)

    @classmethod
    def from_tool_registry(
        cls,
        tool_registry: Any,
        *,
        default_executor_id: str,
        executor_id_by_capability: dict[str, str] | None = None,
        supported_intents_by_capability: dict[str, tuple[str, ...]] | None = None,
    ) -> AgentCapabilityRegistry:
        executor_ids = executor_id_by_capability or {}
        supported_intents = supported_intents_by_capability or {}
        return cls(
            [
                AgentCapabilityDefinition.from_tool_definition(
                    definition,
                    executor_id=executor_ids.get(definition.name, default_executor_id),
                    supported_intents=supported_intents.get(definition.name, ()),
                )
                for definition in tool_registry.list_definitions()
            ]
        )

    def register(self, definition: AgentCapabilityDefinition) -> AgentCapabilityDefinition:
        if definition.capability_id in self._definitions:
            raise ValueError(f"Agent capability already registered: {definition.capability_id}")
        self._definitions[definition.capability_id] = definition
        return definition

    def get(self, capability_id: str) -> AgentCapabilityDefinition | None:
        return self._definitions.get(capability_id)

    def list_definitions(self) -> list[AgentCapabilityDefinition]:
        return sorted(self._definitions.values(), key=lambda definition: definition.capability_id)

    def allowed_for_intent(self, intent: str) -> list[AgentCapabilityDefinition]:
        return [definition for definition in self.list_definitions() if intent in definition.supported_intents]


@dataclass(frozen=True)
class AgentTask:
    capability_id: str
    goal: str
    input_payload: dict[str, Any] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    expected_output: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AgentRuntimeContext:
    session_id: str
    run_id: str
    task_id: str
    namespace: str | None = None
    permission_scope: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StandardAgentResult:
    status: str
    summary: str
    observation: str = ""
    evidence: list[dict[str, Any]] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    raw_result: dict[str, Any] = field(default_factory=dict)
    requires_user_action: bool = False


@dataclass(frozen=True)
class AgentRuntimeRetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 0.25
    max_delay_seconds: float = 2.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_attempts", max(1, min(int(self.max_attempts or 1), 5)))
        object.__setattr__(self, "base_delay_seconds", max(0.0, float(self.base_delay_seconds or 0.0)))
        object.__setattr__(self, "max_delay_seconds", max(0.0, float(self.max_delay_seconds or 0.0)))


class AbilityAgent(Protocol):
    def call(self, task: AgentTask, context: AgentRuntimeContext) -> StandardAgentResult:
        ...


class CapabilityDeclaringAgent(AbilityAgent, Protocol):
    executor_id: str

    def capabilities(self) -> list[AgentCapabilityDefinition]:
        ...


@dataclass(frozen=True)
class AgentRuntimeBundle:
    executors: dict[str, AbilityAgent]
    capability_registry: AgentCapabilityRegistry
    capability_executor_ids: dict[str, str]


class AgentRuntime:
    def __init__(
        self,
        *,
        registry: AgentCapabilityRegistry,
        executors: dict[str, AbilityAgent],
        retry_policy: AgentRuntimeRetryPolicy | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        self._registry = registry
        self._executors = dict(executors)
        self._retry_policy = retry_policy or AgentRuntimeRetryPolicy()
        self._sleeper = sleeper or time.sleep

    def call(self, task: AgentTask, context: AgentRuntimeContext) -> StandardAgentResult:
        definition = self._registry.get(task.capability_id)
        if definition is None:
            return StandardAgentResult(
                status="failed",
                summary=f"未注册的能力：{task.capability_id}",
                missing_information=[task.capability_id],
            )

        missing_inputs = _missing_required_inputs(definition.input_schema, task.input_payload)
        if missing_inputs:
            return StandardAgentResult(
                status="failed",
                summary=f"缺少必要输入：{', '.join(missing_inputs)}",
                missing_information=missing_inputs,
            )

        permission_result = _check_runtime_permission(definition, context)
        if permission_result is not None:
            return permission_result

        executor = self._executors.get(definition.executor_id)
        if executor is None:
            return StandardAgentResult(
                status="failed",
                summary=f"未注册的执行者：{definition.executor_id}",
                missing_information=[definition.executor_id],
            )

        scoped_context = _with_default_namespace(context, definition)
        result = _call_executor_with_transient_retries(
            executor,
            task,
            scoped_context,
            retry_policy=self._retry_policy,
            sleeper=self._sleeper,
        )
        return _standardize_agent_result(result, definition)


class ToolRegistryAgentExecutor:
    def __init__(
        self,
        tool_registry: Any,
        *,
        session_provider: Callable[[AgentRuntimeContext], Any] | None = None,
    ) -> None:
        self._tool_registry = tool_registry
        self._session_provider = session_provider or (lambda _context: None)

    def call(self, task: AgentTask, context: AgentRuntimeContext) -> StandardAgentResult:
        definition = self._tool_registry.get(task.capability_id)
        if definition is None:
            return StandardAgentResult(
                status="failed",
                summary=f"未注册的工具能力：{task.capability_id}",
                missing_information=[task.capability_id],
            )
        if definition.handler is None:
            return StandardAgentResult(
                status="failed",
                summary=f"能力没有绑定可执行处理器：{task.capability_id}",
                missing_information=[task.capability_id],
            )

        try:
            raw_payload = definition.handler(self._session_provider(context), **task.input_payload)
        except Exception as exc:
            return StandardAgentResult(
                status="failed",
                summary=f"{task.capability_id} 执行异常：{type(exc).__name__}: {exc}",
                raw_result={"error": str(exc), "error_type": type(exc).__name__},
            )
        return _standard_result_from_tool_payload(task.capability_id, raw_payload)


_TRANSIENT_ERROR_TYPES = {
    "APITimeoutError",
    "APIConnectionError",
    "APIStatusError",
    "RateLimitError",
    "TimeoutError",
    "ReadTimeout",
    "ConnectTimeout",
    "PoolTimeout",
    "TimeoutException",
    "ConnectError",
    "ReadError",
    "NetworkError",
    "ServiceUnavailableError",
}
_TRANSIENT_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
_TRANSIENT_ERROR_TOKENS = (
    "timeout",
    "timed out",
    "rate limit",
    "too many requests",
    "temporarily unavailable",
    "service unavailable",
    "connection reset",
    "connection aborted",
    "connection refused",
    "api connection",
    "429",
    " 500",
    " 502",
    " 503",
    " 504",
)


def _call_executor_with_transient_retries(
    executor: AbilityAgent,
    task: AgentTask,
    context: AgentRuntimeContext,
    *,
    retry_policy: AgentRuntimeRetryPolicy,
    sleeper: Callable[[float], None],
) -> StandardAgentResult:
    errors: list[dict[str, Any]] = []
    max_attempts = retry_policy.max_attempts
    for attempt in range(1, max_attempts + 1):
        try:
            result = executor.call(task, context)
        except Exception as exc:
            error = _transient_error_from_exception(exc, attempt=attempt)
            if error is None:
                raise
            errors.append(error)
            if attempt >= max_attempts:
                return _transient_retry_exhausted_result(task.capability_id, errors, max_attempts=max_attempts)
            _sleep_before_transient_retry(retry_policy, attempt=attempt, sleeper=sleeper)
            continue

        error = _transient_error_from_result(result, attempt=attempt)
        if error is None:
            if errors:
                return _with_transient_retry_metadata(
                    result,
                    errors,
                    attempts=attempt,
                    max_attempts=max_attempts,
                    recovered=True,
                )
            return result
        errors.append(error)
        if attempt >= max_attempts:
            return _with_transient_retry_metadata(
                result,
                errors,
                attempts=attempt,
                max_attempts=max_attempts,
                recovered=False,
            )
        _sleep_before_transient_retry(retry_policy, attempt=attempt, sleeper=sleeper)
    return _transient_retry_exhausted_result(task.capability_id, errors, max_attempts=max_attempts)


def _transient_error_from_exception(exc: Exception, *, attempt: int) -> dict[str, Any] | None:
    error_type = type(exc).__name__
    status_code = _status_code_from_exception(exc)
    message = str(exc)
    if not _is_transient_error(error_type=error_type, message=message, status_code=status_code, retryable=None):
        return None
    return _transient_error_record(
        attempt=attempt,
        error_type=error_type,
        message=message,
        status_code=status_code,
    )


def _transient_error_from_result(result: StandardAgentResult, *, attempt: int) -> dict[str, Any] | None:
    if result.status not in {"failed", "error"}:
        return None
    payload = _payload_to_dict(result.raw_result)
    retryable = _retryable_from_payload(payload)
    if retryable is False:
        return None
    error_type = _first_text(
        payload.get("error_type"),
        _nested(payload, "result", "error_type"),
        _nested(payload, "metadata", "error_type"),
    )
    message = _first_text(
        payload.get("error"),
        _nested(payload, "result", "error"),
        _nested(payload, "result", "message"),
        result.summary,
    )
    status_code = _status_code_from_payload(payload)
    if not _is_transient_error(
        error_type=error_type,
        message=message,
        status_code=status_code,
        retryable=retryable,
    ):
        return None
    return _transient_error_record(
        attempt=attempt,
        error_type=error_type or "TransientFailure",
        message=message,
        status_code=status_code,
    )


def _is_transient_error(
    *,
    error_type: str | None,
    message: str | None,
    status_code: int | None,
    retryable: bool | None,
) -> bool:
    if retryable is True:
        return True
    if status_code in _TRANSIENT_STATUS_CODES:
        return True
    normalized_type = str(error_type or "").strip()
    if normalized_type in _TRANSIENT_ERROR_TYPES:
        return True
    normalized_message = f" {str(message or '').lower()} "
    return any(token in normalized_message for token in _TRANSIENT_ERROR_TOKENS)


def _transient_error_record(
    *,
    attempt: int,
    error_type: str,
    message: str | None,
    status_code: int | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "attempt": attempt,
        "error_type": error_type,
        "message": str(message or ""),
    }
    if status_code is not None:
        record["status_code"] = status_code
    return record


def _with_transient_retry_metadata(
    result: StandardAgentResult,
    errors: list[dict[str, Any]],
    *,
    attempts: int,
    max_attempts: int,
    recovered: bool,
) -> StandardAgentResult:
    raw_result = _payload_to_dict(result.raw_result)
    raw_result["runtime_retry"] = {
        "type": "transient_error",
        "attempts": attempts,
        "max_attempts": max_attempts,
        "recovered": recovered,
        "errors": [dict(error) for error in errors],
    }
    return StandardAgentResult(
        status=result.status,
        summary=result.summary,
        observation=result.observation,
        evidence=list(result.evidence),
        missing_information=list(result.missing_information),
        next_actions=list(result.next_actions),
        raw_result=raw_result,
        requires_user_action=result.requires_user_action,
    )


def _transient_retry_exhausted_result(
    capability_id: str,
    errors: list[dict[str, Any]],
    *,
    max_attempts: int,
) -> StandardAgentResult:
    last_error = errors[-1] if errors else {}
    error_type = str(last_error.get("error_type") or "TransientFailure")
    message = str(last_error.get("message") or "temporary failure")
    return StandardAgentResult(
        status="failed",
        summary=f"{capability_id} 临时故障重试 {max_attempts} 次后仍失败：{error_type}: {message}",
        raw_result={
            "tool_name": capability_id,
            "ok": False,
            "error": message,
            "error_type": error_type,
            "runtime_retry": {
                "type": "transient_error",
                "attempts": max_attempts,
                "max_attempts": max_attempts,
                "recovered": False,
                "errors": [dict(error) for error in errors],
            },
        },
    )


def _sleep_before_transient_retry(
    retry_policy: AgentRuntimeRetryPolicy,
    *,
    attempt: int,
    sleeper: Callable[[float], None],
) -> None:
    if retry_policy.base_delay_seconds <= 0:
        return
    delay = retry_policy.base_delay_seconds * (2 ** max(0, attempt - 1))
    if retry_policy.max_delay_seconds > 0:
        delay = min(delay, retry_policy.max_delay_seconds)
    if delay > 0:
        sleeper(delay)


def _status_code_from_exception(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    return _to_int(getattr(response, "status_code", None))


def _status_code_from_payload(payload: dict[str, Any]) -> int | None:
    return _first_int(
        payload.get("status_code"),
        _nested(payload, "metadata", "status_code"),
        _nested(payload, "result", "status_code"),
        _nested(payload, "result", "metadata", "status_code"),
    )


def _retryable_from_payload(payload: dict[str, Any]) -> bool | None:
    for value in (
        payload.get("retryable"),
        _nested(payload, "metadata", "retryable"),
        _nested(payload, "result", "retryable"),
        _nested(payload, "result", "metadata", "retryable"),
    ):
        if isinstance(value, bool):
            return value
    return None


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _first_text(*values: Any) -> str | None:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return None


def _first_int(*values: Any) -> int | None:
    for value in values:
        parsed = _to_int(value)
        if parsed is not None:
            return parsed
    return None


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def create_default_agent_capability_registry(
    *,
    tool_registry: Any | None = None,
    default_executor_id: str = TOOL_REGISTRY_EXECUTOR_ID,
    executor_id_by_capability: dict[str, str] | None = None,
    supported_intents_by_capability: dict[str, tuple[str, ...]] | None = None,
) -> AgentCapabilityRegistry:
    if tool_registry is None:
        from app.agent_runtime.tool_registry import create_default_agent_tool_registry

        tool_registry = create_default_agent_tool_registry()
    supported_intents = {
        **DEFAULT_SUPPORTED_INTENTS_BY_CAPABILITY,
        **(supported_intents_by_capability or {}),
    }
    return AgentCapabilityRegistry.from_tool_registry(
        tool_registry,
        default_executor_id=default_executor_id,
        executor_id_by_capability=executor_id_by_capability,
        supported_intents_by_capability=supported_intents,
    )


def build_agent_runtime_bundle(
    agents: list[CapabilityDeclaringAgent],
    *,
    tool_registry: Any | None = None,
    legacy_executor_id: str = TOOL_REGISTRY_EXECUTOR_ID,
) -> AgentRuntimeBundle:
    definitions_by_id: dict[str, AgentCapabilityDefinition] = {}
    if tool_registry is not None:
        legacy_registry = create_default_agent_capability_registry(
            tool_registry=tool_registry,
            default_executor_id=legacy_executor_id,
        )
        definitions_by_id.update(
            {definition.capability_id: definition for definition in legacy_registry.list_definitions()}
        )

    executors: dict[str, AbilityAgent] = {}
    capability_executor_ids: dict[str, str] = {}
    for agent in agents:
        executor_id = str(getattr(agent, "executor_id", "") or "").strip()
        if not executor_id:
            raise ValueError("Agent executor id is required")
        if executor_id in executors:
            raise ValueError(f"Agent executor already registered: {executor_id}")
        executors[executor_id] = agent

        for capability in agent.capabilities():
            if capability.executor_id != executor_id:
                raise ValueError(
                    f"Agent capability executor mismatch: {capability.capability_id} "
                    f"declares {capability.executor_id}, expected {executor_id}"
                )
            definitions_by_id[capability.capability_id] = capability
            capability_executor_ids[capability.capability_id] = executor_id

    return AgentRuntimeBundle(
        executors=executors,
        capability_registry=AgentCapabilityRegistry(list(definitions_by_id.values())),
        capability_executor_ids=capability_executor_ids,
    )


def _missing_required_inputs(input_schema: dict[str, Any], input_payload: dict[str, Any]) -> list[str]:
    required = input_schema.get("required") if isinstance(input_schema, dict) else None
    if not isinstance(required, list):
        return []
    return [str(name) for name in required if str(name) not in input_payload]


def _with_default_namespace(context: AgentRuntimeContext, definition: AgentCapabilityDefinition) -> AgentRuntimeContext:
    if context.namespace:
        return context
    return AgentRuntimeContext(
        session_id=context.session_id,
        run_id=context.run_id,
        task_id=context.task_id,
        namespace=definition.executor_id,
        permission_scope=dict(context.permission_scope),
        metadata=dict(context.metadata),
    )


def _check_runtime_permission(
    definition: AgentCapabilityDefinition,
    context: AgentRuntimeContext,
) -> StandardAgentResult | None:
    source_type = _permission_text(context.permission_scope.get("source_type"))
    if definition.allowed_source_types and source_type not in definition.allowed_source_types:
        return _standardize_agent_result(
            StandardAgentResult(
                status="failed",
                summary=f"能力不允许当前来源调用：{definition.capability_id}",
                raw_result={
                    "permission": {
                        "error_code": "AGENT_CAPABILITY_SOURCE_TYPE_NOT_ALLOWED",
                        "source_type": source_type,
                        "allowed_source_types": sorted(definition.allowed_source_types),
                    }
                },
            ),
            definition,
        )

    user_confirmed = bool(context.permission_scope.get("user_confirmed"))
    if definition.requires_confirmation and not user_confirmed:
        return _standardize_agent_result(
            StandardAgentResult(
                status="waiting_user",
                summary=f"能力需要用户确认后才能执行：{definition.capability_id}",
                raw_result={
                    "permission": {
                        "error_code": "AGENT_CAPABILITY_USER_CONFIRMATION_REQUIRED",
                        "risk_level": definition.risk_level,
                        "requires_confirmation": definition.requires_confirmation,
                    }
                },
                requires_user_action=True,
            ),
            definition,
        )

    return None


def _permission_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _standardize_agent_result(
    result: StandardAgentResult,
    definition: AgentCapabilityDefinition,
) -> StandardAgentResult:
    raw_payload = _payload_to_dict(result.raw_result)
    original_payload = dict(raw_payload)
    raw_payload.setdefault("tool_name", definition.capability_id)
    raw_payload.setdefault("ok", result.status == "succeeded")
    raw_payload.setdefault("missing_information", list(result.missing_information))
    raw_payload.setdefault("next_actions", list(result.next_actions))
    raw_payload.setdefault(
        "result_envelope",
        _result_envelope_from_standard_result(
            result,
            definition=definition,
            raw_payload=original_payload,
        ),
    )
    return StandardAgentResult(
        status=result.status,
        summary=result.summary,
        observation=result.observation,
        evidence=list(result.evidence),
        missing_information=list(result.missing_information),
        next_actions=list(result.next_actions),
        raw_result=raw_payload,
        requires_user_action=result.requires_user_action,
    )


def _result_envelope_from_standard_result(
    result: StandardAgentResult,
    *,
    definition: AgentCapabilityDefinition,
    raw_payload: dict[str, Any],
) -> dict[str, Any]:
    specialized = _specialized_result_envelope(result, definition=definition, raw_payload=raw_payload)
    if specialized is not None:
        return specialized
    return build_tool_result_envelope(
        capability=definition.capability_id,
        status=result.status,
        executor=definition.executor_id,
        risk_level=definition.risk_level,
        result_payload=raw_payload,
        summary=result.summary,
        artifacts=[dict(item) for item in result.evidence if isinstance(item, dict)],
        observations=[result.observation] if result.observation else [],
        requires_user_action=result.requires_user_action,
        raw_result=raw_payload,
    ).to_dict()


def _specialized_result_envelope(
    result: StandardAgentResult,
    *,
    definition: AgentCapabilityDefinition,
    raw_payload: dict[str, Any],
) -> dict[str, Any] | None:
    if definition.capability_id not in _SPECIALIZED_RESULT_ENVELOPE_CAPABILITIES:
        return None
    try:
        from app.agent_runtime.routing.result_envelope import build_result_envelope

        envelope = build_result_envelope(
            capability=definition.capability_id,
            status=result.status,
            result_payload=raw_payload,
            risk_level=definition.risk_level,
        )
    except Exception:
        return None
    return envelope.to_dict() if envelope is not None else None


def _standard_result_from_tool_payload(capability_id: str, raw_payload: Any) -> StandardAgentResult:
    payload = _payload_to_dict(raw_payload)
    envelope = _find_result_envelope(payload)
    ok = bool(payload.get("ok", True))
    status = str(envelope.get("status") or ("succeeded" if ok else "failed")) if envelope else ("succeeded" if ok else "failed")
    return StandardAgentResult(
        status=status,
        summary=_summary_from_payload(capability_id, payload, envelope=envelope, ok=ok),
        observation=_observation_from_envelope(envelope),
        evidence=_evidence_from_envelope(envelope),
        raw_result=payload,
        requires_user_action=bool(envelope.get("requires_user_action", False)) if envelope else False,
    )


def _payload_to_dict(raw_payload: Any) -> dict[str, Any]:
    if isinstance(raw_payload, dict):
        return dict(raw_payload)
    if is_dataclass(raw_payload):
        return asdict(raw_payload)
    model_dump = getattr(raw_payload, "model_dump", None)
    if callable(model_dump):
        return dict(model_dump(mode="json"))
    return {"ok": True, "result": raw_payload}


def _find_result_envelope(payload: dict[str, Any]) -> dict[str, Any]:
    envelope = payload.get("result_envelope")
    if isinstance(envelope, dict):
        return envelope
    result = payload.get("result")
    if isinstance(result, dict) and isinstance(result.get("result_envelope"), dict):
        return result["result_envelope"]
    return {}


def _summary_from_payload(
    capability_id: str,
    payload: dict[str, Any],
    *,
    envelope: dict[str, Any],
    ok: bool,
) -> str:
    summary = str(envelope.get("summary") or "").strip() if envelope else ""
    if summary:
        return summary
    if ok:
        return f"{capability_id} 执行成功"
    error = str(payload.get("error") or "").strip()
    if error:
        return f"{capability_id} 执行失败：{error}"
    return f"{capability_id} 执行失败"


def _observation_from_envelope(envelope: dict[str, Any]) -> str:
    observations = envelope.get("observations") if envelope else None
    if not isinstance(observations, list):
        return ""
    return "\n".join(text for item in observations if (text := str(item or "").strip()))


def _evidence_from_envelope(envelope: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = envelope.get("artifacts") if envelope else None
    if not isinstance(artifacts, list):
        return []
    return [dict(item) for item in artifacts if isinstance(item, dict)]


__all__ = [
    "AbilityAgent",
    "AgentCapabilityDefinition",
    "AgentCapabilityRegistry",
    "AgentRuntime",
    "AgentRuntimeBundle",
    "AgentRuntimeContext",
    "AgentRuntimeRetryPolicy",
    "AgentTask",
    "CLAUDE_SDK_AGENT_EXECUTOR_ID",
    "CapabilityDeclaringAgent",
    "DEFAULT_SUPPORTED_INTENTS_BY_CAPABILITY",
    "OPENAI_SDK_AGENT_EXECUTOR_ID",
    "StandardAgentResult",
    "TOOL_REGISTRY_EXECUTOR_ID",
    "ToolRegistryAgentExecutor",
    "build_agent_runtime_bundle",
    "create_default_agent_capability_registry",
]
