from __future__ import annotations

from dataclasses import dataclass

from app.agent_runtime.contracts.tasks.browser_application import BrowserExecutionResult, BrowserTaskEnvelope


@dataclass(frozen=True)
class SafetyGateResult:
    allowed: bool
    reason_code: str
    message: str


class BrowserSafetyGate:
    def validate_result(
        self,
        *,
        envelope: BrowserTaskEnvelope,
        result: BrowserExecutionResult,
    ) -> SafetyGateResult:
        executed_actions = set(result.executed_actions)
        if _claims_final_submit(envelope=envelope, result=result, executed_actions=executed_actions):
            return SafetyGateResult(
                allowed=False,
                reason_code="final_submit_forbidden",
                message="Browser executor result claims final submit, but this task must stop before final submit.",
            )
        if envelope.selected_resume_file_ref is None and "upload_user_selected_resume" in executed_actions:
            return SafetyGateResult(
                allowed=False,
                reason_code="resume_file_not_selected",
                message="Browser executor cannot upload a resume before the user selects a resume file.",
            )
        if result.blocked_reason in {"login_required", "captcha", "sensitive_question"} and not result.requires_user_action:
            return SafetyGateResult(
                allowed=False,
                reason_code="user_action_required",
                message="Blocked browser executor results must request user action for login, captcha, or sensitive questions.",
            )
        return SafetyGateResult(allowed=True, reason_code="ok", message="Browser executor result passed safety checks.")


def _claims_final_submit(
    *,
    envelope: BrowserTaskEnvelope,
    result: BrowserExecutionResult,
    executed_actions: set[str],
) -> bool:
    if not envelope.stop_before_submit:
        return False
    if result.submitted:
        return True
    status = str(result.status.value if hasattr(result.status, "value") else result.status)
    if status in {"submitted", "completed_submit"}:
        return True
    return bool({"submit_application", "final_submit"} & executed_actions)
