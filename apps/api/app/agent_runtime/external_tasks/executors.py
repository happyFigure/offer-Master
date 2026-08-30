from __future__ import annotations

import base64
from dataclasses import dataclass
from html.parser import HTMLParser
import json
import re
from typing import Any, Protocol
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
from uuid import uuid4

import httpx
from pydantic import ValidationError

from app.agent_runtime.agent_as_tool import (
    AgentCapabilityDefinition,
    AgentRuntimeContext,
    AgentTask,
    CLAUDE_SDK_AGENT_EXECUTOR_ID,
    OPENAI_SDK_AGENT_EXECUTOR_ID,
    StandardAgentResult,
)
from app.agent_runtime.external_tasks.schemas import (
    ApplyEntryDiscoveryResult,
    FindApplyEntryTaskEnvelope,
)
from app.agent_runtime.reflection.schemas import campus_recruiting_web_search_result_evaluation_spec
from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL, AgentToolCandidateProfile
from app.infrastructure.llm.client import LLMRuntimeConfig


class ExternalExecutorError(RuntimeError):
    """Raised when an external agent executor cannot produce a valid result."""


class ExternalExecutor(Protocol):
    executor_name: str

    def execute_find_apply_entry(self, envelope: FindApplyEntryTaskEnvelope) -> ApplyEntryDiscoveryResult:
        ...

    def execute_web_search(self, query: str, *, max_results: int = 5) -> dict[str, Any]:
        ...


class ResumeTailoringExecutor(Protocol):
    executor_name: str

    def execute_resume_tailoring(
        self,
        *,
        resume_text: str,
        job_description: str,
        language: str | None = None,
        style: str | None = None,
        constraints: list[str] | None = None,
    ) -> dict[str, Any]:
        ...


class WebSearchFallback(Protocol):
    def __call__(self, query: str, *, max_results: int = 5) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class ClaudeSdkHttpExecutorConfig:
    base_url: str
    model: str = "MiniMax-M2.7"
    api_key: str | None = None
    timeout_seconds: float = 120.0
    provider_base_url: str | None = None
    provider_api_key: str | None = None
    provider_anthropic_version: str = "2023-06-01"

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/chat/completions"


class ClaudeSdkHttpExecutorAdapter:
    executor_name = "claude-sdk-agent"

    def __init__(
        self,
        *,
        config: ClaudeSdkHttpExecutorConfig,
        client: httpx.Client | None = None,
        web_search_fallback: WebSearchFallback | None = None,
    ) -> None:
        self._config = config
        self._client = client
        self._web_search_fallback = web_search_fallback or _run_http_web_search

    def execute_find_apply_entry(self, envelope: FindApplyEntryTaskEnvelope) -> ApplyEntryDiscoveryResult:
        payload = _with_provider_metadata(_build_chat_payload(envelope, model=self._config.model), self._config)
        headers = _headers(self._config.api_key)
        try:
            response_payload = self._post(payload, headers=headers)
            content = _assistant_content(response_payload)
            result_payload = _extract_json_object(content)
            return ApplyEntryDiscoveryResult.model_validate(result_payload)
        except ExternalExecutorError:
            raise
        except (httpx.HTTPError, json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise ExternalExecutorError(f"Claude SDK executor did not return a valid JSON result: {exc}") from exc

    def execute_web_search(self, query: str, *, max_results: int = 5) -> dict[str, Any]:
        search_query = str(query or "").strip()
        if not search_query:
            raise ExternalExecutorError("External web search query is required")
        payload = _with_provider_metadata(
            _build_web_search_payload(search_query, max_results=max_results, model=self._config.model),
            self._config,
        )
        headers = _headers(self._config.api_key)
        try:
            response_payload = self._post(payload, headers=headers)
            content = _assistant_content(response_payload)
            answer = content.strip()
            if _looks_like_tool_call_only_answer(answer):
                raise ExternalExecutorError("Claude SDK executor returned a tool call marker without final web search answer")
            sources = _extract_urls(content)
            return {
                "executor_name": self.executor_name,
                "query": search_query,
                "answer": answer,
                "sources": sources,
                "observations": [answer] if answer else [],
                "artifacts": _sources_to_artifacts(sources),
            }
        except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
            return self._run_web_search_fallback(search_query, max_results=max_results, cause=exc)
        except ExternalExecutorError as exc:
            return self._run_web_search_fallback(search_query, max_results=max_results, cause=exc)

    def _run_web_search_fallback(self, query: str, *, max_results: int, cause: Exception) -> dict[str, Any]:
        try:
            result = self._web_search_fallback(query, max_results=max_results)
        except Exception as fallback_exc:  # pragma: no cover - defensive wrapper for external network failures
            raise ExternalExecutorError(
                f"Claude SDK executor web search failed: {cause}; HTTP search fallback failed: {fallback_exc}"
            ) from fallback_exc
        return result

    def _post(self, payload: dict[str, Any], *, headers: dict[str, str]) -> dict[str, Any]:
        if self._client is not None:
            response = self._client.post(self._config.chat_completions_url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()

        with httpx.Client(timeout=self._config.timeout_seconds) as client:
            response = client.post(self._config.chat_completions_url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()


class BailianWebSearchExecutor:
    executor_name = "bailian-enable-search"

    def __init__(self, *, config: LLMRuntimeConfig, client: httpx.Client | None = None) -> None:
        self._config = config
        self._client = client

    def execute_web_search(self, query: str, *, max_results: int = 5) -> dict[str, Any]:
        search_query = str(query or "").strip()
        if not search_query:
            raise ExternalExecutorError("Bailian web search query is required")
        result_limit = max(1, min(int(max_results or 5), 10))
        response_payload = self._post_dashscope_generation(
            _build_bailian_dashscope_payload(search_query, max_results=result_limit, model=self._config.model)
        )
        answer = _bailian_dashscope_answer(response_payload).strip()
        if not answer:
            raise ExternalExecutorError("Bailian web search returned empty assistant content")
        search_results = _bailian_dashscope_search_results(response_payload, limit=result_limit)
        sources = [result["url"] for result in search_results if result.get("url")]
        return {
            "executor_name": self.executor_name,
            "query": search_query,
            "answer": answer,
            "sources": sources,
            "observations": [answer],
            "results": search_results,
            "artifacts": _search_results_to_artifacts(search_results),
            "raw_response": response_payload,
        }

    def _post_dashscope_generation(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = _headers(self._config.api_key)
        url = _dashscope_generation_url(self._config.base_url)
        if self._client is not None:
            response = self._client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()

        with httpx.Client(timeout=self._config.timeout_seconds) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()


@dataclass(frozen=True)
class OpenAISdkAgentConfig:
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    base_url: str | None = None
    timeout_seconds: float = 120.0


class OpenAISdkResumeClientAdapter:
    executor_name = OPENAI_SDK_AGENT_EXECUTOR_ID

    def __init__(self, *, config: OpenAISdkAgentConfig, client: Any | None = None) -> None:
        self._config = config
        self._client = client or _build_openai_sdk_client(config)

    def execute_resume_tailoring(
        self,
        *,
        resume_text: str,
        job_description: str,
        language: str | None = None,
        style: str | None = None,
        constraints: list[str] | None = None,
    ) -> dict[str, Any]:
        resume = _required_text(resume_text, "resume_text")
        jd = _required_text(job_description, "job_description")
        completion = self._client.chat.completions.create(
            model=self._config.model,
            messages=[
                {"role": "system", "content": _build_resume_tailoring_system_prompt()},
                {
                    "role": "user",
                    "content": _build_resume_tailoring_user_prompt(
                        resume_text=resume,
                        job_description=jd,
                        language=language,
                        style=style,
                        constraints=constraints,
                    ),
                },
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
            timeout=self._config.timeout_seconds,
        )
        content = _openai_assistant_content(completion)
        payload = _extract_json_object(content)
        revised_resume = _required_text(payload.get("revised_resume"), "revised_resume")
        change_summary = _text_list(payload.get("change_summary"))
        warnings = _text_list(payload.get("warnings"))
        return {
            "executor_name": self.executor_name,
            "revised_resume": revised_resume,
            "change_summary": change_summary or ["已根据目标 JD 改写简历表达。"],
            "warnings": warnings,
        }


class OpenAISdkAgentExecutor:
    executor_id = OPENAI_SDK_AGENT_EXECUTOR_ID

    def __init__(self, executor: ResumeTailoringExecutor) -> None:
        self._executor = executor

    def capabilities(self) -> list[AgentCapabilityDefinition]:
        return [
            AgentCapabilityDefinition(
                capability_id="resume.tailor",
                name="简历修改",
                description="根据用户简历和目标 JD 改写简历内容，只生成修改建议和新版文本，不直接覆盖文件。",
                executor_id=self.executor_id,
                input_schema={
                    "type": "object",
                    "required": ["resume_text", "job_description"],
                    "properties": {
                        "resume_text": {"type": "string", "description": "用户当前简历正文。"},
                        "job_description": {"type": "string", "description": "目标 JD 或岗位要求。"},
                        "language": {"type": ["string", "null"], "default": "zh-CN"},
                        "style": {"type": ["string", "null"], "description": "可选风格，例如简洁、STAR、校招投递版。"},
                        "constraints": {
                            "type": ["array", "null"],
                            "items": {"type": "string"},
                            "description": "必须遵守的改写约束。",
                        },
                    },
                    "additionalProperties": False,
                },
                output_schema={
                    "type": "object",
                    "required": ["tool_name", "ok", "result"],
                    "properties": {
                        "tool_name": {"type": "string"},
                        "ok": {"type": "boolean"},
                        "result": {
                            "type": "object",
                            "required": ["revised_resume", "change_summary"],
                            "properties": {
                                "revised_resume": {"type": "string"},
                                "change_summary": {"type": "array", "items": {"type": "string"}},
                                "warnings": {"type": "array", "items": {"type": "string"}},
                            },
                        },
                    },
                },
                risk_level="low",
                supported_intents=("resume_tailoring",),
                requires_confirmation=False,
                allowed_source_types=frozenset({"agent_chat"}),
                candidate_profile=AgentToolCandidateProfile(
                    categories=frozenset({"resume_tailoring", "content_processing"}),
                    keywords=frozenset({"简历", "优化简历", "修改简历", "改简历", "润色简历", "匹配 JD", "目标 JD"}),
                    examples=("根据这份简历和 Java 后端 JD 帮我改简历", "把我的简历改得更适合这个 Agent 开发岗位"),
                ),
            )
        ]

    def call(self, task: AgentTask, context: AgentRuntimeContext) -> StandardAgentResult:
        if task.capability_id != "resume.tailor":
            return StandardAgentResult(
                status="failed",
                summary=f"{self.executor_id} 暂不支持能力：{task.capability_id}",
                missing_information=[task.capability_id],
            )
        resume_text = str(task.input_payload.get("resume_text") or "").strip()
        if not resume_text:
            return _resume_tailoring_missing_input_result("resume_text", "缺少必要输入：resume_text")
        job_description = str(task.input_payload.get("job_description") or "").strip()
        if not job_description:
            return _resume_tailoring_missing_input_result("job_description", "缺少必要输入：job_description")
        try:
            tailoring_result = self._executor.execute_resume_tailoring(
                resume_text=resume_text,
                job_description=job_description,
                language=_optional_text(task.input_payload.get("language")),
                style=_optional_text(task.input_payload.get("style")),
                constraints=_text_list(task.input_payload.get("constraints")),
            )
        except Exception as exc:
            return StandardAgentResult(
                status="failed",
                summary=f"{self.executor_id} 执行简历修改失败：{type(exc).__name__}: {exc}",
                raw_result={
                    "tool_name": "resume.tailor",
                    "ok": False,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )
        result = dict(tailoring_result)
        result.setdefault("executor_name", getattr(self._executor, "executor_name", self.executor_id))
        revised_resume = str(result.get("revised_resume") or "").strip()
        if not revised_resume:
            return _resume_tailoring_missing_input_result("revised_resume", "OpenAI SDK agent 没有返回改写后的简历")
        result["change_summary"] = _text_list(result.get("change_summary")) or ["已根据目标 JD 改写简历表达。"]
        result["warnings"] = _text_list(result.get("warnings"))
        raw_payload = {"tool_name": "resume.tailor", "ok": True, "result": result}
        return StandardAgentResult(
            status="succeeded",
            summary="OpenAI SDK agent 已完成简历修改",
            observation=_resume_tailoring_observation(result),
            raw_result=raw_payload,
        )


class ClaudeSdkAgentExecutor:
    executor_id = CLAUDE_SDK_AGENT_EXECUTOR_ID

    def __init__(self, executor: ExternalExecutor) -> None:
        self._executor = executor

    def capabilities(self) -> list[AgentCapabilityDefinition]:
        return [
            AgentCapabilityDefinition(
                capability_id=EXTERNAL_WEB_SEARCH_TOOL,
                name="网页搜索",
                description="通过已配置的 Claude SDK agent 搜索公开网页。",
                executor_id=self.executor_id,
                input_schema={
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                    },
                    "additionalProperties": False,
                },
                output_schema={
                    "type": "object",
                    "required": ["tool_name", "ok", "result"],
                    "properties": {
                        "tool_name": {"type": "string"},
                        "ok": {"type": "boolean"},
                        "result": {"type": "object"},
                    },
                },
                risk_level="low",
                supported_intents=("campus_recruiting_search", "external_agent_task"),
                result_evaluation=campus_recruiting_web_search_result_evaluation_spec(),
            )
        ]

    def call(self, task: AgentTask, context: AgentRuntimeContext) -> StandardAgentResult:
        if task.capability_id != EXTERNAL_WEB_SEARCH_TOOL:
            return StandardAgentResult(
                status="failed",
                summary=f"{self.executor_id} 暂不支持能力：{task.capability_id}",
                missing_information=[task.capability_id],
            )
        query = str(task.input_payload.get("query") or "").strip()
        if not query:
            return StandardAgentResult(
                status="failed",
                summary="缺少必要输入：query",
                missing_information=["query"],
            )
        max_results = _bounded_web_search_limit(task.input_payload.get("max_results"))
        try:
            search_result = self._executor.execute_web_search(query, max_results=max_results)
        except Exception as exc:
            return StandardAgentResult(
                status="failed",
                summary=f"{self.executor_id} 执行网页搜索失败：{type(exc).__name__}: {exc}",
                raw_result={"tool_name": EXTERNAL_WEB_SEARCH_TOOL, "ok": False, "error": str(exc)},
            )
        result = dict(search_result)
        result.setdefault("executor_name", getattr(self._executor, "executor_name", self.executor_id))
        raw_payload = {"tool_name": EXTERNAL_WEB_SEARCH_TOOL, "ok": True, "result": result}
        return StandardAgentResult(
            status="succeeded",
            summary=_web_search_summary(result),
            observation=_web_search_observation(result),
            evidence=_web_search_evidence(result),
            raw_result=raw_payload,
        )


def _build_openai_sdk_client(config: OpenAISdkAgentConfig) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - depends on deployment packaging.
        raise ExternalExecutorError("OpenAI SDK is not installed. Install the openai package to enable openai-sdk-agent.") from exc
    kwargs: dict[str, Any] = {}
    api_key = str(config.api_key or "").strip()
    if api_key:
        kwargs["api_key"] = api_key
    base_url = str(config.base_url or "").strip().rstrip("/")
    if base_url:
        kwargs["base_url"] = base_url
    timeout = float(config.timeout_seconds or 0)
    if timeout > 0:
        kwargs["timeout"] = timeout
    return OpenAI(**kwargs)


def _build_resume_tailoring_system_prompt() -> str:
    return (
        "你是 OfferMaster 的简历修改子 agent。\n"
        "你的任务是根据用户提供的原始简历和目标 JD 改写简历。\n"
        "必须保留用户真实经历，不得编造公司、学历、项目、时间、奖项、指标或成果。\n"
        "如果目标 JD 信息不足，只能在已有事实基础上增强表达，并在 warnings 中说明限制。\n"
        "只返回一个 JSON 对象，不要输出 Markdown，不要解释内部推理。\n"
        "JSON 格式：{\"revised_resume\": \"改写后的完整简历\", \"change_summary\": [\"修改点\"], \"warnings\": [\"限制或风险\"]}"
    )


def _build_resume_tailoring_user_prompt(
    *,
    resume_text: str,
    job_description: str,
    language: str | None,
    style: str | None,
    constraints: list[str] | None,
) -> str:
    constraint_lines = "\n".join(f"- {item}" for item in constraints or [] if str(item).strip()) or "- 保留真实经历，不编造。"
    return (
        f"输出语言：{language or 'zh-CN'}\n"
        f"改写风格：{style or '清晰、具体、适合投递'}\n"
        "必须遵守的约束：\n"
        f"{constraint_lines}\n\n"
        "原始简历：\n"
        f"{resume_text}\n\n"
        "目标 JD：\n"
        f"{job_description}\n\n"
        "请输出 JSON。"
    )


def _openai_assistant_content(payload: Any) -> str:
    choices = payload.get("choices") if isinstance(payload, dict) else getattr(payload, "choices", None)
    if not isinstance(choices, list) or not choices:
        raise ExternalExecutorError("OpenAI SDK agent response did not contain choices")
    first_choice = choices[0]
    message = first_choice.get("message") if isinstance(first_choice, dict) else getattr(first_choice, "message", None)
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if isinstance(content, list):
        content = "".join(_openai_content_part_text(part) for part in content)
    if not isinstance(content, str) or not content.strip():
        raise ExternalExecutorError("OpenAI SDK agent response did not contain assistant content")
    return content


def _openai_content_part_text(part: Any) -> str:
    if isinstance(part, dict):
        return str(part.get("text") or part.get("content") or "")
    return str(getattr(part, "text", None) or getattr(part, "content", None) or "")


def _required_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ExternalExecutorError(f"{field_name} is required")
    return text


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list | tuple | set):
        return [text for item in value if (text := str(item or "").strip())]
    text = str(value or "").strip()
    return [text] if text else []


def _resume_tailoring_missing_input_result(field_name: str, summary: str) -> StandardAgentResult:
    error_code = f"{field_name.upper()}_REQUIRED"
    return StandardAgentResult(
        status="failed",
        summary=summary,
        missing_information=[field_name],
        raw_result={
            "tool_name": "resume.tailor",
            "ok": False,
            "error": error_code,
            "retryable": False,
        },
    )


def _resume_tailoring_observation(result: dict[str, Any]) -> str:
    revised_resume = str(result.get("revised_resume") or "").strip()
    change_summary = _text_list(result.get("change_summary"))
    warnings = _text_list(result.get("warnings"))
    lines: list[str] = []
    if revised_resume:
        lines.append(revised_resume)
    if change_summary:
        lines.append("修改摘要：" + "；".join(change_summary))
    if warnings:
        lines.append("注意事项：" + "；".join(warnings))
    return "\n".join(lines)


def _bounded_web_search_limit(value: Any) -> int:
    try:
        parsed = int(value) if value is not None else 5
    except (TypeError, ValueError):
        parsed = 5
    return max(1, min(parsed, 10))


def _web_search_summary(result: dict[str, Any]) -> str:
    answer = str(result.get("answer") or "").strip()
    if answer:
        return answer
    return "claude-sdk-agent 已完成网页搜索"


def _web_search_observation(result: dict[str, Any]) -> str:
    observations = result.get("observations")
    if not isinstance(observations, list):
        return ""
    return "\n".join(text for item in observations if (text := str(item or "").strip()))


def _web_search_evidence(result: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = result.get("artifacts")
    if isinstance(artifacts, list):
        return [dict(item) for item in artifacts if isinstance(item, dict)]
    sources = result.get("sources")
    if not isinstance(sources, list):
        return []
    evidence: list[dict[str, Any]] = []
    for source in sources:
        if isinstance(source, str):
            url = source.strip()
            title = "source"
        elif isinstance(source, dict):
            url = str(source.get("url") or "").strip()
            title = str(source.get("title") or "source").strip() or "source"
        else:
            continue
        if url:
            evidence.append({"type": "url", "title": title, "url": url})
    return evidence


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    key = str(api_key or "").strip()
    if key:
        headers["x-api-key"] = key
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _build_chat_payload(envelope: FindApplyEntryTaskEnvelope, *, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "stream": False,
        "session_id": envelope.task_id,
        "user": envelope.task_id,
        "messages": [
            {
                "role": "user",
                "content": _build_find_apply_entry_prompt(envelope),
            }
        ],
    }


def _build_web_search_payload(query: str, *, max_results: int, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "stream": False,
        "session_id": f"external-web-search-{uuid4().hex}",
        "user": "offermaster-web-search",
        "messages": [
            {
                "role": "user",
                "content": _build_web_search_prompt(query, max_results=max_results),
            }
        ],
    }


def _with_provider_metadata(payload: dict[str, Any], config: ClaudeSdkHttpExecutorConfig) -> dict[str, Any]:
    base_url = str(config.provider_base_url or "").strip()
    api_key = str(config.provider_api_key or "").strip()
    if not base_url and not api_key:
        return payload
    llm_config: dict[str, str] = {}
    if base_url:
        llm_config["base_url"] = base_url
    if api_key:
        llm_config["api_key"] = api_key
    anthropic_version = str(config.provider_anthropic_version or "").strip()
    if anthropic_version:
        llm_config["anthropic_version"] = anthropic_version
    return {
        **payload,
        "metadata": {
            "agentconfig": {
                "runtime_config": {
                    "llm": llm_config,
                }
            }
        },
    }


def _build_web_search_prompt(query: str, *, max_results: int) -> str:
    return (
        "You are an external web-search agent for OfferMaster.\n"
        "Use exactly one WebSearch call first. Use WebFetch only if the top result is ambiguous.\n"
        "Then answer directly in the user's language with 3-6 bullets or one short paragraph.\n"
        "Prefer recent sources for time-sensitive queries, and include concise source URLs. "
        "Source names alone are not enough; include URLs or say the evidence is insufficient.\n"
        "Do not continue researching after you have enough information for a concise answer.\n"
        "Do not expose reasoning or planning; return the final answer only.\n"
        f"Use no more than {max(1, int(max_results or 5))} relevant results.\n\n"
        f"User query: {query}"
    )


def _build_bailian_web_search_system_prompt(max_results: int) -> str:
    limit = max(1, min(int(max_results or 5), 10))
    return (
        "你是 OfferMaster 的联网搜索执行器。\n"
        "你必须基于联网搜索结果回答，不能只凭模型记忆回答。\n"
        "如果是实时问题，优先使用最新、权威、可核验来源。\n"
        "回答要简洁，使用用户的语言。\n"
        "必须尽量包含来源链接；如果搜索结果不足以支持结论，要明确说明证据不足。\n"
        f"最多整理 {limit} 条相关结果，不要展开无关信息。"
    )


def _build_bailian_dashscope_payload(query: str, *, max_results: int, model: str) -> dict[str, Any]:
    return {
        "model": str(model or "qwen-plus"),
        "input": {
            "messages": [
                {"role": "system", "content": _build_bailian_web_search_system_prompt(max_results)},
                {"role": "user", "content": query},
            ]
        },
        "parameters": {
            "result_format": "message",
            "enable_search": True,
            "search_options": {
                "forced_search": True,
                "enable_source": True,
                "enable_citation": True,
                "citation_format": "[ref_<number>]",
                "search_strategy": "max",
            },
        },
    }


def _dashscope_generation_url(base_url: str) -> str:
    normalized = str(base_url or "").strip().rstrip("/")
    if normalized.endswith("/api/v1/services/aigc/text-generation/generation"):
        return normalized
    parsed = urlparse(normalized or "https://dashscope.aliyuncs.com")
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc or "dashscope.aliyuncs.com"
    return f"{scheme}://{netloc}/api/v1/services/aigc/text-generation/generation"


def _bailian_dashscope_answer(response_payload: dict[str, Any]) -> str:
    output = response_payload.get("output") if isinstance(response_payload, dict) else None
    choices = output.get("choices") if isinstance(output, dict) else None
    if not isinstance(choices, list) or not choices:
        raise ExternalExecutorError("Bailian web search response did not contain output choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if content is None:
        raise ExternalExecutorError("Bailian web search response did not contain assistant content")
    return str(content)


def _bailian_dashscope_search_results(response_payload: dict[str, Any], *, limit: int) -> list[dict[str, str]]:
    output = response_payload.get("output") if isinstance(response_payload, dict) else None
    search_info = output.get("search_info") if isinstance(output, dict) else None
    raw_results = search_info.get("search_results") if isinstance(search_info, dict) else None
    if not isinstance(raw_results, list):
        return []
    results: list[dict[str, str]] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if not url:
            continue
        site_name = str(item.get("site_name") or "").strip()
        title = str(item.get("title") or "").strip() or url
        display_title = f"{site_name} - {title}" if site_name and site_name not in title else title
        results.append(
            {
                "title": display_title,
                "url": url,
                "snippet": str(item.get("snippet") or item.get("summary") or "").strip(),
            }
        )
        if len(results) >= max(1, min(int(limit or 5), 10)):
            break
    return results


def _search_results_to_artifacts(results: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {"type": "url", "title": result.get("title") or "source", "url": result["url"]}
        for result in results
        if result.get("url")
    ]


def _looks_like_tool_call_only_answer(content: str) -> bool:
    text = str(content or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if "(tool_call)" not in lowered and "<tool_call" not in lowered:
        return False
    has_url = bool(_extract_urls(text))
    has_web_search_name = "websearch" in lowered or "web_search" in lowered
    return has_web_search_name and not has_url


def _build_find_apply_entry_prompt(envelope: FindApplyEntryTaskEnvelope) -> str:
    task_json = json.dumps(envelope.model_dump(mode="json"), ensure_ascii=False, indent=2)
    return (
        "You are an external execution agent for OfferMaster.\n"
        "Use WebSearch/WebFetch or available browsing/search tools to find the official application entry for the job.\n"
        "For this Claude SDK phase, identify the official apply URL and evidence only; do not submit applications, "
        "create accounts, answer sensitive questions, or change resume files.\n"
        "Return only one JSON object matching ApplyEntryDiscoveryResultV1. Do not wrap it in explanations.\n"
        "If you find the official apply URL, use status='found_opened', set apply_url/final_browser_url, "
        "and include at least one evidence_artifacts item such as a web_search_result URL.\n"
        "If login/captcha/ambiguity blocks verification, use status='blocked' with blocked_reason.\n"
        "If the network/tool call fails, use status='failed'.\n\n"
        "Task envelope:\n"
        f"```json\n{task_json}\n```"
    )


def _run_http_web_search(query: str, *, max_results: int = 5) -> dict[str, Any]:
    search_query = str(query or "").strip()
    if not search_query:
        raise ExternalExecutorError("HTTP web search query is required")
    limit = max(1, min(int(max_results or 5), 10))
    search_errors: list[str] = []
    for searcher in (_run_bing_web_search, _run_duckduckgo_web_search):
        try:
            return searcher(search_query, max_results=limit)
        except Exception as exc:
            search_errors.append(f"{searcher.__name__}: {exc}")
    raise ExternalExecutorError("HTTP web search fallback failed: " + "; ".join(search_errors))


def _run_bing_web_search(query: str, *, max_results: int = 5) -> dict[str, Any]:
    limit = max(1, min(int(max_results or 5), 10))
    url = f"https://www.bing.com/search?q={quote_plus(query)}&setlang=zh-CN"
    headers = {
        "User-Agent": "Mozilla/5.0 OfferMaster/1.0 (+https://localhost)",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
    }
    with httpx.Client(timeout=20.0, follow_redirects=True, headers=headers) as client:
        response = client.get(url)
        response.raise_for_status()
    results = _parse_bing_results(response.text, limit=limit)
    if not results:
        raise ExternalExecutorError("Bing web search returned no results")
    return _web_search_result_payload(query, results, executor_name="http-web-search-fallback")


def _run_duckduckgo_web_search(query: str, *, max_results: int = 5) -> dict[str, Any]:
    limit = max(1, min(int(max_results or 5), 10))
    url = f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}"
    headers = {
        "User-Agent": "Mozilla/5.0 OfferMaster/1.0 (+https://localhost)",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
    }
    with httpx.Client(timeout=20.0, follow_redirects=True, headers=headers) as client:
        response = client.get(url)
        response.raise_for_status()
    results = _parse_duckduckgo_lite_results(response.text, limit=limit)
    if not results:
        raise ExternalExecutorError("DuckDuckGo web search returned no results")
    return _web_search_result_payload(query, results, executor_name="http-web-search-fallback")


def _web_search_result_payload(query: str, results: list[dict[str, str]], *, executor_name: str) -> dict[str, Any]:
    lines = ["联网搜索结果："]
    observations: list[str] = []
    artifacts: list[dict[str, str]] = []
    for result in results:
        title = str(result.get("title") or "搜索结果").strip()
        snippet = str(result.get("snippet") or "").strip()
        source_url = str(result.get("url") or "").strip()
        if snippet and source_url:
            observation = f"{title}：{snippet}（{source_url}）"
        elif source_url:
            observation = f"{title}：{source_url}"
        else:
            observation = title
        observations.append(observation)
        lines.append(f"- {observation}")
        if source_url:
            artifacts.append({"type": "url", "title": title, "url": source_url})
    return {
        "executor_name": executor_name,
        "query": query,
        "answer": "\n".join(lines),
        "results": results,
        "sources": [str(item.get("url") or "") for item in results if item.get("url")],
        "observations": observations,
        "artifacts": artifacts,
    }


def _parse_bing_results(html: str, *, limit: int) -> list[dict[str, str]]:
    parser = _BingSearchParser(limit=max(1, int(limit or 5)))
    parser.feed(html or "")
    return parser.results


def _parse_duckduckgo_lite_results(html: str, *, limit: int) -> list[dict[str, str]]:
    parser = _DuckDuckGoLiteParser(limit=max(1, int(limit or 5)))
    parser.feed(html or "")
    return parser.results


def _assistant_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ExternalExecutorError("Claude SDK executor response did not contain choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise ExternalExecutorError("Claude SDK executor response did not contain assistant content")
    return content


class _DuckDuckGoLiteParser(HTMLParser):
    def __init__(self, *, limit: int) -> None:
        super().__init__(convert_charrefs=True)
        self._limit = limit
        self._capture_title = False
        self._capture_snippet = False
        self._current_text: list[str] = []
        self._pending_href = ""
        self.results: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_map = {key.lower(): value or "" for key, value in attrs}
        class_name = attrs_map.get("class", "")
        if tag.lower() == "a" and "result-link" in class_name:
            self._capture_title = True
            self._pending_href = attrs_map.get("href", "")
            self._current_text = []
        elif tag.lower() in {"td", "div"} and "result-snippet" in class_name:
            self._capture_snippet = True
            self._current_text = []

    def handle_data(self, data: str) -> None:
        if self._capture_title or self._capture_snippet:
            self._current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        lower_tag = tag.lower()
        if lower_tag == "a" and self._capture_title:
            title = _collapse_spaces("".join(self._current_text))
            url = _normalize_duckduckgo_url(self._pending_href)
            if title and url and len(self.results) < self._limit:
                self.results.append({"title": title, "url": url, "snippet": ""})
            self._capture_title = False
            self._pending_href = ""
            self._current_text = []
            return
        if lower_tag in {"td", "div"} and self._capture_snippet:
            snippet = _collapse_spaces("".join(self._current_text))
            if snippet and self.results and not self.results[-1].get("snippet"):
                self.results[-1]["snippet"] = snippet
            self._capture_snippet = False
            self._current_text = []


class _BingSearchParser(HTMLParser):
    def __init__(self, *, limit: int) -> None:
        super().__init__(convert_charrefs=True)
        self._limit = limit
        self._in_result = False
        self._result_depth = 0
        self._in_h2 = False
        self._capture_title = False
        self._capture_snippet = False
        self._current_text: list[str] = []
        self._pending_href = ""
        self.results: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lower_tag = tag.lower()
        attrs_map = {key.lower(): value or "" for key, value in attrs}
        class_name = attrs_map.get("class", "")
        if lower_tag == "li" and "b_algo" in class_name:
            self._in_result = True
            self._result_depth = 1
            return
        if self._in_result:
            self._result_depth += 1
        if not self._in_result:
            return
        if lower_tag == "h2":
            self._in_h2 = True
            return
        if self._in_h2 and lower_tag == "a" and len(self.results) < self._limit:
            self._capture_title = True
            self._pending_href = attrs_map.get("href", "")
            self._current_text = []
            return
        if lower_tag == "p" and not self._capture_snippet and self.results and not self.results[-1].get("snippet"):
            self._capture_snippet = True
            self._current_text = []

    def handle_data(self, data: str) -> None:
        if self._capture_title or self._capture_snippet:
            self._current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        lower_tag = tag.lower()
        if lower_tag == "a" and self._capture_title:
            title = _collapse_spaces("".join(self._current_text))
            url = _normalize_bing_url(self._pending_href)
            if title and url and len(self.results) < self._limit:
                self.results.append({"title": title, "url": url, "snippet": ""})
            self._capture_title = False
            self._pending_href = ""
            self._current_text = []
        elif lower_tag == "p" and self._capture_snippet:
            snippet = _collapse_spaces("".join(self._current_text))
            if snippet and self.results and not self.results[-1].get("snippet"):
                self.results[-1]["snippet"] = snippet
            self._capture_snippet = False
            self._current_text = []
        elif lower_tag == "h2" and self._in_h2:
            self._in_h2 = False
        if self._in_result:
            self._result_depth -= 1
            if self._result_depth <= 0:
                self._in_result = False
                self._in_h2 = False
                self._capture_title = False
                self._capture_snippet = False
                self._current_text = []


def _normalize_bing_url(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    query = parse_qs(parsed.query)
    encoded = query.get("u", [""])[0]
    if encoded.startswith("a1"):
        decoded = _decode_bing_base64_url(encoded[2:])
        if decoded:
            return decoded
    return raw


def _decode_bing_base64_url(value: str) -> str:
    padded = value + "=" * (-len(value) % 4)
    try:
        return base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def _normalize_duckduckgo_url(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    query = parse_qs(parsed.query)
    redirected = query.get("uddg", [""])[0]
    if redirected:
        return unquote(redirected).strip()
    return raw


def _collapse_spaces(value: str) -> str:
    return " ".join(str(value or "").split())


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _strip_fenced_json(stripped)
    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ExternalExecutorError("Claude SDK executor did not return a valid JSON object")


def _strip_fenced_json(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text
    if lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_urls(text: str) -> list[dict[str, str]]:
    import re

    urls: list[dict[str, str]] = []
    for url in re.findall(r"https?://[^\s)）>\]]+", text):
        cleaned = url.rstrip("。.,，、;；")
        if cleaned and cleaned not in {item["url"] for item in urls}:
            urls.append({"url": cleaned})
    return urls[:10]


def _sources_to_artifacts(sources: list[dict[str, str]]) -> list[dict[str, str]]:
    artifacts: list[dict[str, str]] = []
    for source in sources:
        url = str(source.get("url") or "").strip()
        if not url:
            continue
        title = str(source.get("title") or "source").strip() or "source"
        artifacts.append({"type": "url", "title": title, "url": url})
    return artifacts
