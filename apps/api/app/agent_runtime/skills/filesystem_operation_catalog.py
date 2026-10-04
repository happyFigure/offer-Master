from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agent_runtime.tool_registry import (
    FILESYSTEM_COPY_FILE_TOOL,
    FILESYSTEM_MOVE_FILE_TOOL,
    FILESYSTEM_PATH_EXISTS_TOOL,
    FILESYSTEM_READ_FILE_TOOL,
    FILESYSTEM_REPLACE_TEXT_TOOL,
)


# Keep this catalog import-light: routing, executor and goal validation all read
# it during startup, so importing agent_as_tool here would create a cycle.
FILESYSTEM_SKILL_CAPABILITY = "skill.filesystem"


@dataclass(frozen=True)
class FilesystemOperationSpec:
    """Single source of truth for one internal filesystem Skill operation."""

    operation: str
    legacy_tool_name: str
    script_name: str
    goal_kind: str
    target_type: str
    required_args: tuple[str, ...]
    input_fields: tuple[str, ...]
    success_criteria: tuple[str, ...]
    risk_level: str = "low"
    markers: tuple[str, ...] = ()
    summary: str = ""
    approval_reason: str = ""
    approval_summary_template: str = ""
    approval_fields: tuple[str, ...] = ()
    target_match_key: str = ""
    result_path_arg: str = ""
    target_match_mode: str = "exact"
    frame_reason: str = "file_operation_request"
    target_path_strategy: str = ""
    aliases: tuple[str, ...] = ()
    negative_markers: tuple[str, ...] = ()
    normalized_keywords: tuple[str, ...] = ()
    auto_execute_threshold: float = 0.7
    confirmation_threshold: float = 0.82

    @property
    def requires_confirmation(self) -> bool:
        return self.risk_level == "high"


# Internal filesystem scripts are deliberately catalog-driven. Adding another
# filesystem action should start by adding one spec here, then reusing the
# generic routing, approval, execution and goal-validation helpers downstream.
FILESYSTEM_OPERATION_CATALOG: dict[str, FilesystemOperationSpec] = {
    "copy_file": FilesystemOperationSpec(
        operation="copy_file",
        legacy_tool_name=FILESYSTEM_COPY_FILE_TOOL,
        script_name="copy_file.py",
        goal_kind="file_to_file",
        target_type="file_copy",
        required_args=("src", "dst"),
        input_fields=("src", "dst", "path", "overwrite", "operation_intent"),
        success_criteria=("operation_is_copy_file", "tool_result_ok", "target_path_matches"),
        risk_level="high",
        markers=("复制", "拷贝", "备份", "copy"),
        aliases=("复制一份", "拷贝一份", "另存为", "copy file", "backup file"),
        negative_markers=("复制文件内容", "复制内容", "刚才复制的文件", "复制的文件", "复制出来的文件"),
        normalized_keywords=("copy_file", "copy file", "backup file"),
        summary="复制本地文件，源文件保持不变。",
        approval_reason="复制本地文件会创建或覆盖文件，需要用户确认。",
        approval_summary_template="需要确认复制文件：{src} -> {dst}。",
        approval_fields=("src", "dst", "overwrite", "operation_intent"),
        target_match_key="target_path",
        result_path_arg="dst",
        target_match_mode="path",
        frame_reason="file_copy_request",
        target_path_strategy="copy_path",
    ),
    "path_exists": FilesystemOperationSpec(
        operation="path_exists",
        legacy_tool_name=FILESYSTEM_PATH_EXISTS_TOOL,
        script_name="path_exists.py",
        goal_kind="path_exists",
        target_type="file_path",
        required_args=("path",),
        input_fields=("path",),
        success_criteria=("operation_is_path_exists", "tool_result_ok"),
        markers=("是否存在", "是否存", "是否还在", "存不存在", "在不在", "有没有这个文件", "文件是否存在", "path exists", "exists"),
        aliases=("还在不", "还在吗", "有没", "有没有", "是否村", "是否寸", "是否还存", "文件在吗", "check file exists"),
        negative_markers=("内容", "全文", "读取内容", "显示内容"),
        normalized_keywords=("path_exists", "check file exists", "file exists"),
        summary="检查本地路径是否存在。",
        frame_reason="path_exists_request",
    ),
    "read_file": FilesystemOperationSpec(
        operation="read_file",
        legacy_tool_name=FILESYSTEM_READ_FILE_TOOL,
        script_name="read_file.py",
        goal_kind="read_content",
        target_type="file_content",
        required_args=("path",),
        input_fields=("path", "offset", "limit", "encoding"),
        success_criteria=("operation_is_read_file", "tool_result_ok"),
        markers=("读取", "读一下", "打开", "查看", "看一下", "read", "cat", "显示", "展示"),
        aliases=("文件内容", "这个文件的内容", "内容是什么", "全文", "read file content", "show file content"),
        negative_markers=("是否存在", "是否存", "是否村", "是否寸", "在不在", "还在不", "有没有"),
        normalized_keywords=("read_file", "read file", "read file content"),
        summary="读取本地文件内容。",
        frame_reason="explicit_read_request",
    ),
    "rename_file": FilesystemOperationSpec(
        operation="rename_file",
        legacy_tool_name=FILESYSTEM_MOVE_FILE_TOOL,
        script_name="move_file.py",
        goal_kind="file_to_file",
        target_type="file_name",
        required_args=("src", "dst"),
        input_fields=("src", "dst", "path", "overwrite", "operation_intent"),
        success_criteria=("operation_is_rename", "target_name_matches", "tool_result_ok"),
        risk_level="high",
        markers=("重命名", "改名", "命名为", "改成", "改为", "换成", "换为", "起名", "起名字", "取名", "rename"),
        aliases=("文件名改", "文件名称改", "名字改成", "名称改成", "给文件起名", "给它起名", "rename file"),
        negative_markers=("简历名字", "简历的名字", "姓名", "内容", "正文"),
        normalized_keywords=("rename_file", "rename file", "change filename"),
        summary="将文件重命名，不修改文件内容。",
        approval_reason="重命名本地文件属于高风险写操作，需要用户确认。",
        approval_summary_template="需要确认重命名文件：{src} -> {dst}。",
        approval_fields=("src", "dst", "overwrite", "operation_intent"),
        target_match_key="target_name",
        result_path_arg="dst",
        target_match_mode="filename",
        frame_reason="filename_rename_request",
        target_path_strategy="rename_same_directory",
    ),
    "replace_text": FilesystemOperationSpec(
        operation="replace_text",
        legacy_tool_name=FILESYSTEM_REPLACE_TEXT_TOOL,
        script_name="replace_text.py",
        goal_kind="replace_text",
        target_type="file_content",
        required_args=("path", "old_text", "new_text"),
        input_fields=("path", "old_text", "new_text", "encoding", "count"),
        success_criteria=("operation_is_replace_text", "replacement_count_positive", "tool_result_ok"),
        risk_level="high",
        markers=("替换", "换成", "换为", "改成", "改为", "修改", "只改", "写入", "保存"),
        aliases=("替换文本", "修改内容", "改正文", "简历名字", "简历的名字", "change file content"),
        negative_markers=("文件名", "文件名称", "重命名", "改名"),
        normalized_keywords=("replace_text", "replace text", "change file content"),
        summary="在文件内容中做精确文本替换。",
        approval_reason="替换本地文件内容属于高风险写操作，需要用户确认。",
        approval_summary_template="需要确认替换文件内容：{path} 中的 {old_text} -> {new_text}。",
        approval_fields=("path", "old_text", "new_text", "encoding", "count"),
        frame_reason="content_mutation_request",
    ),
}


def get_filesystem_operation_spec(operation: str | None) -> FilesystemOperationSpec | None:
    return FILESYSTEM_OPERATION_CATALOG.get(str(operation or "").strip())


def known_filesystem_operations() -> tuple[str, ...]:
    return tuple(sorted(FILESYSTEM_OPERATION_CATALOG))


def filesystem_operation_markers(operation: str) -> tuple[str, ...]:
    spec = get_filesystem_operation_spec(operation)
    return spec.markers if spec is not None else ()


def high_risk_filesystem_operations() -> frozenset[str]:
    return frozenset(spec.operation for spec in FILESYSTEM_OPERATION_CATALOG.values() if spec.requires_confirmation)


def build_filesystem_skill_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["user_task"],
        "properties": {
            "user_task": {"type": "string"},
            "context_metadata": {"type": "object", "additionalProperties": True},
            "path": {"type": "string"},
            "src": {"type": "string"},
            "dst": {"type": "string"},
            "overwrite": {"type": "boolean", "default": False},
            "operation_intent": {
                "type": "object",
                "description": "Structured semantic candidate from the model. Prefer this for multi-turn directory or naming references.",
                "properties": {
                    "destination": {
                        "type": "object",
                        "properties": {
                            "kind": {"type": "string", "enum": ["file", "directory"]},
                            "path": {"type": "string"},
                            "reference": {"type": "string"},
                        },
                        "required": ["kind", "path"],
                        "additionalProperties": False,
                    },
                    "name_intent": {
                        "type": "object",
                        "description": "Model-selected semantic filename; runtime normalizes it into a safe concrete path.",
                        "properties": {
                            "mode": {"type": "string", "enum": ["model_proposed", "user_provided"]},
                            "display_name": {"type": "string"},
                            "filename": {"type": "string"},
                            "filename_stem": {"type": "string"},
                            "extension_policy": {"type": "string", "enum": ["preserve_source", "explicit"]},
                            "source_basis": {"type": "string", "enum": ["file_content", "conversation", "user_input"]},
                        },
                        "additionalProperties": False,
                    },
                    "name_policy": {
                        "type": "string",
                        "enum": ["content_based"],
                    },
                    "avoid_conflict": {"type": "boolean", "default": True},
                },
                "required": ["destination"],
                "additionalProperties": False,
            },
            "operation": {"type": "string", "enum": list(known_filesystem_operations())},
            "offset": {"type": "integer", "minimum": 0, "default": 0},
            "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 200},
            "encoding": {"type": "string", "default": "auto"},
        },
        "additionalProperties": True,
    }


def build_operation_approval_payload(spec: FilesystemOperationSpec, arguments: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "capability": FILESYSTEM_SKILL_CAPABILITY,
        "operation": spec.operation,
        "summary": spec.summary,
        "risk_level": spec.risk_level,
        "requires_confirmation": spec.requires_confirmation,
    }
    for field in spec.approval_fields:
        value = arguments.get(field)
        if field == "overwrite":
            payload[field] = bool(value)
        elif field == "operation_intent":
            payload[field] = dict(value) if isinstance(value, dict) else {}
        else:
            payload[field] = str(value or "")
    return payload


def format_operation_approval_summary(spec: FilesystemOperationSpec, payload: dict[str, Any]) -> str:
    if not spec.approval_summary_template:
        return spec.summary
    return spec.approval_summary_template.format(**payload)


def format_operation_execution_summary(spec: FilesystemOperationSpec, raw_payload: dict[str, Any], *, ok: bool) -> str:
    if ok:
        return f"filesystem Skill 已完成：{spec.summary}"
    return str(raw_payload.get("error") or "filesystem Skill 执行失败。")


__all__ = [
    "FILESYSTEM_OPERATION_CATALOG",
    "FilesystemOperationSpec",
    "build_filesystem_skill_input_schema",
    "build_operation_approval_payload",
    "filesystem_operation_markers",
    "format_operation_approval_summary",
    "format_operation_execution_summary",
    "get_filesystem_operation_spec",
    "high_risk_filesystem_operations",
    "known_filesystem_operations",
]
