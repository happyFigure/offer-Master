from __future__ import annotations

from typing import Any

_OPERATION_LABELS = {
    "read_file": "读取文件内容",
    "path_exists": "检查文件是否存在",
    "copy_file": "复制文件",
    "rename_file": "改名或重命名文件",
    "replace_text": "替换文件内容",
}


def build_filesystem_no_dead_end_outcome(
    *,
    operation: str,
    reason: str,
    candidates: tuple[dict[str, Any], ...] = (),
    suggested_operation: str = "",
    next_action: str = "ask_user",
    missing_information: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build a recoverable filesystem failure that never reads as success.

    This is the OpenClaw-style no-dead-end layer: even when a tool cannot run,
    the model and frontend receive a structured reason plus an explicit next step.
    """

    suggested_operations = [dict(candidate) for candidate in candidates]
    if not suggested_operation and suggested_operations:
        suggested_operation = str(suggested_operations[0].get("operation") or "")

    readable_next_action = _readable_next_action(next_action)
    summary = f"filesystem Skill 未完成：{reason}。下一步：{readable_next_action}"
    if next_action == "ask_user":
        # The summary is also rendered by the frontend. Do not leak catalog
        # keys such as copy_file; show the same plain-language clarification
        # that the model receives as its next-step observation.
        summary = f"{summary} {_ask_user_message(candidates)}"
    elif suggested_operation:
        summary = f"{summary}，建议动作是 {_operation_label(suggested_operation)}。"
    else:
        summary = f"{summary}。"

    return {
        "ok": False,
        "operation": operation or "unknown",
        "reason": reason,
        "summary": summary,
        "recoverable": next_action not in {"stop", "fatal"},
        "next_action": next_action,
        "suggested_operation": suggested_operation,
        "suggested_operations": suggested_operations,
        "missing_information": list(missing_information),
        "ask_user_message": _ask_user_message(candidates),
    }


def _readable_next_action(next_action: str) -> str:
    if next_action == "read_before_write":
        return "先读取文件内容，再让模型选择具体文件名"
    if next_action == "retry_operation":
        return "换用更匹配的文件动作继续执行"
    if next_action == "request_confirmation":
        return "请用户确认高风险文件操作"
    if next_action == "ask_user":
        return "向用户澄清要执行的具体文件动作"
    if next_action == "continue_model_loop":
        return "让模型补充结构化信息后继续"
    return "返回失败原因并等待下一步指令"


def _ask_user_message(candidates: tuple[dict[str, Any], ...]) -> str:
    if not candidates:
        return "我识别到了文件相关请求，但还不确定要读取、检查存在、复制、改名，还是替换内容。"

    labels = []
    for candidate in candidates[:5]:
        label = _operation_label(candidate.get("operation"))
        if label not in labels:
            labels.append(label)
    for fallback in ("读取文件内容", "检查文件是否存在", "复制文件", "改名或重命名文件", "替换文件内容"):
        if fallback not in labels:
            labels.append(fallback)
    names = "、".join(labels)
    return f"我识别到了你提供的是一个文件路径，但还不知道你想做什么。你是要{names}中的哪一种？请明确告诉我。"


def _operation_label(operation: str) -> str:
    return _OPERATION_LABELS.get(str(operation or "").strip(), "处理文件")


__all__ = ["build_filesystem_no_dead_end_outcome"]
