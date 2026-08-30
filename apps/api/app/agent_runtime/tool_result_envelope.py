from __future__ import annotations

from typing import Any

from app.agent_runtime.routing.schemas import ResultEnvelope


DEFAULT_TOOL_RESULT_EXECUTOR = "agent_tool_registry"


def build_tool_result_envelope(
    *,
    capability: str,
    status: str,
    result_payload: dict[str, Any],
    risk_level: str = "low",
    executor: str = DEFAULT_TOOL_RESULT_EXECUTOR,
    summary: str | None = None,
    artifacts: list[dict[str, Any]] | None = None,
    observations: list[str] | None = None,
    requires_user_action: bool | None = None,
    raw_result: dict[str, Any] | None = None,
    source_type: str | None = None,
    tool_call_log_id: str | None = None,
    step_id: str | None = None,
) -> ResultEnvelope:
    payload = dict(result_payload)
    nested = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    normalized_status = _normalize_status(status, payload)
    next_action = _first_text(payload.get("next_action"), nested.get("next_action"))
    error_code = _first_text(payload.get("error_code"), nested.get("error_code"), _nested(payload, "metadata", "error_code"))
    retryable = _first_bool(payload.get("retryable"), nested.get("retryable"), _nested(payload, "metadata", "retryable"))
    resolved_observations = _observations(observations, payload, nested, normalized_status=normalized_status)
    resolved_summary = _summary(
        capability,
        normalized_status=normalized_status,
        summary=summary,
        payload=payload,
        nested=nested,
        observations=resolved_observations,
    )
    return ResultEnvelope(
        status=normalized_status,
        capability=capability,
        executor=executor,
        summary=resolved_summary,
        artifacts=_artifacts(artifacts, payload, nested),
        observations=resolved_observations,
        requires_user_action=_requires_user_action(
            explicit=requires_user_action,
            payload=payload,
            nested=nested,
            status=normalized_status,
            next_action=next_action,
        ),
        risk_level=risk_level,
        raw_result=dict(raw_result if raw_result is not None else payload),
        error_code=error_code,
        retryable=retryable,
        next_action=next_action,
        business_refs=_business_refs(payload, nested),
        source_type=source_type,
        tool_call_log_id=tool_call_log_id,
        step_id=step_id,
    )


def _normalize_status(status: str, payload: dict[str, Any]) -> str:
    if payload.get("ok") is False and status == "succeeded":
        return "failed"
    text = str(status or "").strip()
    return text or ("succeeded" if payload.get("ok", True) else "failed")


def _summary(
    capability: str,
    *,
    normalized_status: str,
    summary: str | None,
    payload: dict[str, Any],
    nested: dict[str, Any],
    observations: list[str],
) -> str:
    if normalized_status == "succeeded":
        explicit = _first_text(
            summary,
            payload.get("summary"),
            nested.get("summary"),
            payload.get("message"),
            nested.get("message"),
            payload.get("error"),
            nested.get("error"),
        )
    else:
        explicit = _first_text(
            summary,
            payload.get("summary"),
            nested.get("summary"),
            payload.get("error"),
            nested.get("error"),
            payload.get("message"),
            nested.get("message"),
        )
    if explicit:
        return explicit
    if observations:
        return observations[0]
    return f"{capability} 执行{'成功' if normalized_status == 'succeeded' else '失败'}"


def _observations(
    observations: list[str] | None,
    payload: dict[str, Any],
    nested: dict[str, Any],
    *,
    normalized_status: str,
) -> list[str]:
    result = _normalize_text_list(observations)
    if not result:
        result.extend(_normalize_text_list(payload.get("observations")))
        result.extend(_normalize_text_list(nested.get("observations")))
    if normalized_status != "succeeded":
        for value in (payload.get("error"), nested.get("error"), nested.get("message"), payload.get("message")):
            text = _optional_text(value)
            if text and text not in result:
                result.append(text)
    return result


def _artifacts(
    artifacts: list[dict[str, Any]] | None,
    payload: dict[str, Any],
    nested: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates: list[Any] = []
    candidates.extend(artifacts or [])
    candidates.extend(payload.get("artifacts") if isinstance(payload.get("artifacts"), list) else [])
    candidates.extend(nested.get("artifacts") if isinstance(nested.get("artifacts"), list) else [])
    if not candidates:
        candidates.extend(_source_to_artifact(source) for source in _source_list(nested.get("sources")))

    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in candidates:
        if not isinstance(item, dict):
            continue
        artifact_type = _optional_text(item.get("type") or item.get("artifact_type")) or "artifact"
        uri = _optional_text(item.get("url") or item.get("uri") or item.get("path_or_uri") or item.get("path"))
        title = _optional_text(item.get("title")) or artifact_type
        if not uri:
            continue
        artifact = {"type": artifact_type, "title": title, "url": uri}
        mime_type = _optional_text(item.get("mime_type"))
        if mime_type:
            artifact["mime_type"] = mime_type
        key = (artifact["type"], artifact["title"], artifact["url"])
        if key in seen:
            continue
        seen.add(key)
        normalized.append(artifact)
    return normalized


def _business_refs(payload: dict[str, Any], nested: dict[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for raw in (payload.get("business_refs"), nested.get("business_refs")):
        if not isinstance(raw, list):
            continue
        for item in raw:
            if isinstance(item, dict):
                refs.append(dict(item))

    for key, ref_type in (
        ("raw_job_lead_id", "raw_job_lead"),
        ("job_lead_id", "job_lead"),
        ("job_id", "job"),
        ("application_id", "application"),
        ("company_id", "company"),
        ("source_id", "job_source"),
    ):
        value = payload.get(key) or nested.get(key)
        if value:
            refs.append({"type": ref_type, "id": str(value)})

    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, Any]] = []
    for ref in refs:
        ref_type = _optional_text(ref.get("type"))
        ref_id = _optional_text(ref.get("id"))
        if not ref_type or not ref_id:
            continue
        key = (ref_type, ref_id)
        if key in seen:
            continue
        seen.add(key)
        unique.append(ref)
    return unique


def _requires_user_action(
    *,
    explicit: bool | None,
    payload: dict[str, Any],
    nested: dict[str, Any],
    status: str,
    next_action: str | None,
) -> bool:
    if explicit is not None:
        return explicit
    for value in (payload.get("requires_user_action"), nested.get("requires_user_action")):
        if isinstance(value, bool):
            return value
    if status == "waiting_user":
        return True
    action = str(next_action or "").lower()
    return action.startswith("request_") or "user" in action or "confirm" in action


def _source_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _source_to_artifact(source: Any) -> dict[str, Any]:
    if isinstance(source, str):
        url = source.strip()
        return {"type": "url", "title": "source", "url": url} if url else {}
    if not isinstance(source, dict):
        return {}
    url = _optional_text(source.get("url") or source.get("uri"))
    if not url:
        return {}
    return {"type": "url", "title": _optional_text(source.get("title")) or "source", "url": url}


def _normalize_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = _optional_text(item)
        if text:
            result.append(text)
    return result


def _first_text(*values: Any) -> str | None:
    for value in values:
        text = _optional_text(value)
        if text:
            return text
    return None


def _first_bool(*values: Any) -> bool | None:
    for value in values:
        if isinstance(value, bool):
            return value
    return None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


__all__ = ["build_tool_result_envelope"]
