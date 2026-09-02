from __future__ import annotations

from typing import Any

from app.agent_runtime.routing.schemas import ResultEnvelope
from app.agent_runtime.tool_result_envelope import build_tool_result_envelope
from app.agent_runtime.tool_registry import APPLICATION_FIND_APPLY_ENTRY_TOOL, EXTERNAL_WEB_SEARCH_TOOL


def build_result_envelope(
    *,
    capability: str,
    status: str,
    result_payload: dict[str, Any],
    risk_level: str = "low",
) -> ResultEnvelope | None:
    if capability == EXTERNAL_WEB_SEARCH_TOOL:
        return _web_search_result_envelope(status=status, capability=capability, result_payload=result_payload, risk_level=risk_level)
    if capability == APPLICATION_FIND_APPLY_ENTRY_TOOL:
        return _apply_entry_result_envelope(status=status, capability=capability, result_payload=result_payload, risk_level=risk_level)
    return build_tool_result_envelope(
        capability=capability,
        status=status,
        result_payload=result_payload,
        risk_level=risk_level,
    )


def build_apply_entry_task_result_envelope(
    *,
    result_payload: dict[str, Any],
    task_input_payload: dict[str, Any],
    executor_name: str | None,
    risk_level: str = "medium",
) -> ResultEnvelope:
    job = task_input_payload.get("job") if isinstance(task_input_payload.get("job"), dict) else {}
    company_name = str(result_payload.get("company_name") or job.get("company_name") or "目标公司")
    title = str(result_payload.get("job_title") or job.get("title") or "目标岗位")
    apply_url = str(result_payload.get("apply_url") or result_payload.get("final_browser_url") or "").strip()
    result_status = str(result_payload.get("status") or "unknown")
    executor = str(executor_name or result_payload.get("executor_name") or "external_agent")
    summary = f"{company_name} - {title} 申请入口任务状态：{result_status}"
    if apply_url:
        summary += f"，入口：{apply_url}"
    artifacts = _apply_entry_artifacts(result_payload, apply_url=apply_url)
    return ResultEnvelope(
        status=_task_envelope_status(result_status),
        capability=APPLICATION_FIND_APPLY_ENTRY_TOOL,
        executor=executor,
        summary=summary,
        artifacts=artifacts,
        observations=[summary],
        requires_user_action=True,
        risk_level=risk_level,
        raw_result=result_payload,
    )


def _web_search_result_envelope(
    *,
    status: str,
    capability: str,
    result_payload: dict[str, Any],
    risk_level: str,
) -> ResultEnvelope:
    nested = result_payload.get("result") if isinstance(result_payload.get("result"), dict) else {}
    executor = str(nested.get("executor_name") or "external_web_search")
    answer = str(nested.get("answer") or "").strip()
    sources = nested.get("sources") if isinstance(nested.get("sources"), list) else []
    observations = _normalize_observations(nested.get("observations")) or ([answer] if answer else [])
    artifacts = _normalize_artifacts(nested.get("artifacts"))
    if not artifacts:
        artifacts = [_source_to_artifact(source) for source in sources]
        artifacts = [artifact for artifact in artifacts if artifact]
    return ResultEnvelope(
        status=status,
        capability=capability,
        executor=executor,
        summary=answer,
        artifacts=artifacts,
        observations=observations,
        requires_user_action=False,
        risk_level=risk_level,
        raw_result=result_payload,
    )


def _normalize_observations(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    observations: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            observations.append(text)
    return observations


def _normalize_artifacts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    artifacts: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        artifact_type = str(item.get("type") or "").strip()
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        if not artifact_type or not title or not url:
            continue
        artifacts.append({"type": artifact_type, "title": title, "url": url})
    return artifacts


def _source_to_artifact(source: Any) -> dict[str, Any]:
    if isinstance(source, str):
        url = source.strip()
        title = "source"
    elif isinstance(source, dict):
        url = str(source.get("url") or "").strip()
        title = str(source.get("title") or "source").strip()
    else:
        return {}
    if not url:
        return {}
    return {"type": "url", "title": title, "url": url}


def _apply_entry_result_envelope(
    *,
    status: str,
    capability: str,
    result_payload: dict[str, Any],
    risk_level: str,
) -> ResultEnvelope:
    nested = result_payload.get("result") if isinstance(result_payload.get("result"), dict) else {}
    envelope = nested.get("task_envelope") if isinstance(nested.get("task_envelope"), dict) else {}
    job = envelope.get("job") if isinstance(envelope.get("job"), dict) else {}
    company_name = str(job.get("company_name") or "目标公司")
    title = str(job.get("title") or "目标岗位")
    dispatch = nested.get("dispatch") if isinstance(nested.get("dispatch"), dict) else {}
    executor = str(dispatch.get("executor_name") or "external_agent")
    apply_url = str(dispatch.get("apply_url") or dispatch.get("final_browser_url") or "").strip()
    result_status = str(dispatch.get("result_status") or dispatch.get("status") or nested.get("status") or status)
    summary = f"{company_name} - {title} 申请入口任务状态：{result_status}"
    if apply_url:
        summary += f"，入口：{apply_url}"
    artifacts = [{"type": "url", "title": "application_entry", "url": apply_url}] if apply_url else []
    return ResultEnvelope(
        status=status,
        capability=capability,
        executor=executor,
        summary=summary,
        artifacts=artifacts,
        observations=[summary],
        requires_user_action=True,
        risk_level=risk_level,
        raw_result=result_payload,
    )


def _apply_entry_artifacts(result_payload: dict[str, Any], *, apply_url: str) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    if apply_url:
        artifacts.append({"type": "url", "title": "application_entry", "url": apply_url})
    evidence = result_payload.get("evidence_artifacts")
    if isinstance(evidence, list):
        for item in evidence:
            if not isinstance(item, dict):
                continue
            path_or_uri = str(item.get("path_or_uri") or "").strip()
            if not path_or_uri:
                continue
            artifact_type = str(item.get("artifact_type") or "artifact").strip()
            artifacts.append({"type": artifact_type, "title": artifact_type, "url": path_or_uri})
    seen: set[tuple[str, str, str]] = set()
    unique: list[dict[str, Any]] = []
    for artifact in artifacts:
        key = (str(artifact.get("type") or ""), str(artifact.get("title") or ""), str(artifact.get("url") or ""))
        if key in seen:
            continue
        seen.add(key)
        unique.append(artifact)
    return unique


def _task_envelope_status(result_status: str) -> str:
    if result_status == "found_opened":
        return "succeeded"
    if result_status == "blocked":
        return "waiting_user"
    return "failed"
