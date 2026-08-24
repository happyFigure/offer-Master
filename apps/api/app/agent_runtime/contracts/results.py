from __future__ import annotations

from app.agent_runtime.contracts.tasks.browser_application import BrowserExecutionResult, BrowserTaskEnvelope
from app.agent_runtime.routing.schemas import ResultEnvelope


def browser_execution_result_to_result_envelope(
    *,
    task_envelope: BrowserTaskEnvelope,
    result: BrowserExecutionResult,
    executor_name: str,
) -> ResultEnvelope:
    status = str(result.status.value if hasattr(result.status, "value") else result.status)
    summary = result.summary or _default_browser_summary(task_envelope=task_envelope, status=status)
    return ResultEnvelope(
        status=status,
        capability=task_envelope.capability,
        executor=executor_name,
        summary=summary,
        artifacts=[artifact.model_dump(mode="json") for artifact in result.artifacts],
        observations=list(result.observations) or ([summary] if summary else []),
        requires_user_action=result.requires_user_action,
        risk_level=task_envelope.risk_level,
        raw_result={
            "task_envelope": task_envelope.model_dump(mode="json"),
            "browser_execution_result": result.model_dump(mode="json"),
        },
    )


def _default_browser_summary(*, task_envelope: BrowserTaskEnvelope, status: str) -> str:
    company_name = task_envelope.job.company_name
    title = task_envelope.job.title
    return f"{company_name} - {title} browser executor status: {status}"
