from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from app.agent_runtime.context.filename_intent import is_placeholder_filename


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FilesystemSemanticAdmissionResult:
    """Decision made after slot filling but before approval/execution."""

    allowed: bool
    reason: str = ""
    reason_code: str = ""
    next_action: str = "continue"
    corrected_arguments: dict[str, Any] | None = None
    confidence: float = 1.0

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "reason_code": self.reason_code,
            "next_action": self.next_action,
            "confidence": self.confidence,
            "corrected": bool(self.corrected_arguments),
        }


def validate_filesystem_semantic_admission(
    *,
    operation: str,
    arguments: dict[str, Any],
    user_message: str,
) -> FilesystemSemanticAdmissionResult:
    """Guard against mechanically valid but semantically unsafe file actions.

    Tool schemas can prove that `src` and `dst` exist as strings, but they cannot
    prove that `dst="一下.tex"` actually satisfies "把文件名改一下，改成没有用友版本".
    This layer is the OpenClaw-style beforeToolCall semantic checkpoint: it either
    repairs a high-confidence target or returns a recoverable result to the loop.
    """

    if operation != "rename_file":
        return FilesystemSemanticAdmissionResult(allowed=True)

    src = str(arguments.get("src") or arguments.get("path") or "").strip()
    dst = str(arguments.get("dst") or "").strip()
    if not dst:
        return FilesystemSemanticAdmissionResult(allowed=True)

    if is_placeholder_filename(dst):
        logger.warning(
            "Filesystem semantic admission rejected placeholder rename target: src=%s dst=%s",
            src,
            dst,
        )
        return FilesystemSemanticAdmissionResult(
            allowed=False,
            reason="目标文件名仍是对模型建议的指代，不是可执行的真实文件名。",
            reason_code="placeholder_filename_target",
            next_action="retry_operation",
            confidence=0.05,
        )

    # Runtime never repairs a model target from user prose. Once the target is
    # concrete and passes the placeholder guard, execution may continue through
    # path checks, approval, the filesystem script, and postcheck.
    return FilesystemSemanticAdmissionResult(allowed=True)


__all__ = ["FilesystemSemanticAdmissionResult", "validate_filesystem_semantic_admission"]
