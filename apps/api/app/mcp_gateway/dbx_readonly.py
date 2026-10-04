from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from app.mcp_gateway.client import MCPToolCallResult


DBX_READONLY_TOOL_NAMES = (
    "dbx.dbx_list_connections",
    "dbx.dbx_list_databases",
    "dbx.dbx_list_tables",
    "dbx.dbx_get_schema_context",
    "dbx.dbx_describe_table",
    "dbx.dbx_execute_query",
)
DBX_READONLY_TOOL_SET = frozenset(DBX_READONLY_TOOL_NAMES)
DBX_EXECUTE_QUERY_TOOL = "dbx.dbx_execute_query"

_DBX_MUTATION_KEYWORDS = re.compile(
    r"\b(?:insert|update|delete|merge|replace|truncate|alter|drop|create|grant|revoke|call|load|set|use|commit|rollback)\b",
    re.IGNORECASE,
)
_DBX_READ_QUERY_START = re.compile(r"^(?:select|with|show|describe|desc|explain)\b", re.IGNORECASE)
_DBX_FORBIDDEN_READ_MODIFIERS = re.compile(r"\b(?:for\s+update|lock\s+(?:in\s+share\s+mode|table)|into\s+(?:out|dump)file)\b", re.IGNORECASE)


def is_dbx_tool_name(tool_name: str) -> bool:
    return str(tool_name or "").strip().lower().startswith("dbx.")


def is_allowed_dbx_readonly_tool(tool_name: str) -> bool:
    return str(tool_name or "").strip().lower() in DBX_READONLY_TOOL_SET


def filter_mcp_tool_names(tool_names: list[str] | tuple[str, ...]) -> list[str]:
    """Keep ordinary MCP names, but make the DBX namespace explicitly read-only."""

    filtered: list[str] = []
    for raw_name in tool_names:
        name = str(raw_name or "").strip()
        if not name:
            continue
        if is_dbx_tool_name(name) and not is_allowed_dbx_readonly_tool(name):
            continue
        if name not in filtered:
            filtered.append(name)
    return filtered


def dbx_tool_description(tool_name: str) -> str:
    descriptions = {
        "dbx.dbx_list_connections": "List configured DBX database connections. Read-only discovery; call this before assuming a connection name.",
        "dbx.dbx_list_databases": "List databases for a discovered DBX connection. Read-only metadata lookup.",
        "dbx.dbx_list_tables": "List tables in a discovered DBX database/schema. Read-only metadata lookup.",
        "dbx.dbx_get_schema_context": "Read compact schema context for DBX tables before writing a query. Read-only metadata lookup.",
        "dbx.dbx_describe_table": "Describe columns and indexes for one DBX table. Read-only metadata lookup.",
        "dbx.dbx_execute_query": "Execute one read-only SQL query through DBX. Use SELECT/WITH/SHOW/DESCRIBE/EXPLAIN only; database facts must come from the returned rows.",
    }
    return descriptions.get(tool_name, f"Call read-only DBX MCP tool: {tool_name}.")


def dbx_tool_input_schema(tool_name: str) -> dict[str, Any] | None:
    schemas: dict[str, dict[str, Any]] = {
        "dbx.dbx_list_connections": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        "dbx.dbx_list_databases": {
            "type": "object",
            "properties": {
                "connection_id": {"type": "string"},
                "connection_name": {"type": "string"},
            },
            "anyOf": [{"required": ["connection_id"]}, {"required": ["connection_name"]}],
            "additionalProperties": False,
        },
        "dbx.dbx_list_tables": {
            "type": "object",
            "properties": {
                "connection_id": {"type": "string"},
                "connection_name": {"type": "string"},
                "database": {"type": "string"},
                "schema": {"type": "string"},
            },
            "anyOf": [{"required": ["connection_id"]}, {"required": ["connection_name"]}],
            "additionalProperties": False,
        },
        "dbx.dbx_get_schema_context": {
            "type": "object",
            "properties": {
                "connection_id": {"type": "string"},
                "connection_name": {"type": "string"},
                "database": {"type": "string"},
                "schema": {"type": "string"},
                "tables": {"type": "array", "items": {"type": "string"}},
                "max_tables": {"type": "integer", "minimum": 1, "maximum": 500},
            },
            "anyOf": [{"required": ["connection_id"]}, {"required": ["connection_name"]}],
            "additionalProperties": False,
        },
        "dbx.dbx_describe_table": {
            "type": "object",
            "required": ["table"],
            "properties": {
                "table": {"type": "string"},
                "connection_id": {"type": "string"},
                "connection_name": {"type": "string"},
                "database": {"type": "string"},
                "schema": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "dbx.dbx_execute_query": {
            "type": "object",
            "required": ["sql"],
            "properties": {
                "sql": {"type": "string", "minLength": 1},
                "connection_id": {"type": "string"},
                "connection_name": {"type": "string"},
                "database": {"type": "string"},
                "session_id": {"type": "string"},
                "cell_char_offset": {"type": "integer", "minimum": 0},
                "cell_char_limit": {"type": "integer", "minimum": 1, "maximum": 200000},
                "max_rows": {"type": "integer", "minimum": 1, "maximum": 10000},
            },
            "additionalProperties": False,
        },
    }
    schema = schemas.get(str(tool_name or "").strip().lower())
    return deepcopy(schema) if schema is not None else None


def validate_dbx_tool_arguments(tool_name: str, arguments: dict[str, Any]) -> MCPToolCallResult | None:
    """Enforce the DBX read-only contract immediately before the MCP call."""

    normalized_name = str(tool_name or "").strip().lower()
    if not is_dbx_tool_name(normalized_name):
        return None
    if not is_allowed_dbx_readonly_tool(normalized_name):
        return _policy_error(normalized_name, "This DBX MCP tool is outside OfferMaster's read-only allowlist.")
    if normalized_name != DBX_EXECUTE_QUERY_TOOL:
        return None

    sql = arguments.get("sql")
    if not isinstance(sql, str) or not sql.strip():
        return _policy_error(normalized_name, "dbx_execute_query requires a non-empty SQL query.")
    normalized_sql = _strip_sql_comments_and_literals(sql).strip()
    if ";" in normalized_sql:
        return _policy_error(normalized_name, "Only one read-only SQL statement is allowed.")
    if not _DBX_READ_QUERY_START.match(normalized_sql):
        return _policy_error(normalized_name, "Only SELECT, WITH, SHOW, DESCRIBE, or EXPLAIN queries are allowed.")
    if _DBX_MUTATION_KEYWORDS.search(normalized_sql) or _DBX_FORBIDDEN_READ_MODIFIERS.search(normalized_sql):
        return _policy_error(normalized_name, "The SQL contains a write or locking operation and was blocked.")
    return None


def _policy_error(tool_name: str, message: str) -> MCPToolCallResult:
    return MCPToolCallResult(
        tool_name=tool_name,
        ok=False,
        error="DBX_READ_ONLY_POLICY",
        result={"message": message, "read_only": True},
    )


def _strip_sql_comments_and_literals(sql: str) -> str:
    """Remove comments and quoted content before checking SQL control keywords."""

    output: list[str] = []
    index = 0
    quote: str | None = None
    while index < len(sql):
        char = sql[index]
        next_char = sql[index + 1] if index + 1 < len(sql) else ""
        if quote:
            if char == "\\":
                index += 2
                continue
            if char == quote:
                if index + 1 < len(sql) and sql[index + 1] == quote:
                    index += 2
                    continue
                quote = None
            index += 1
            continue
        if char in ("'", '"', "`"):
            quote = char
            output.append(" ")
            index += 1
            continue
        if char == "-" and next_char == "-":
            index = _skip_until_line_end(sql, index + 2)
            output.append(" ")
            continue
        if char == "#":
            index = _skip_until_line_end(sql, index + 1)
            output.append(" ")
            continue
        if char == "/" and next_char == "*":
            end = sql.find("*/", index + 2)
            index = len(sql) if end < 0 else end + 2
            output.append(" ")
            continue
        output.append(char)
        index += 1
    return "".join(output)


def _skip_until_line_end(sql: str, index: int) -> int:
    newline = sql.find("\n", index)
    return len(sql) if newline < 0 else newline + 1


__all__ = [
    "DBX_EXECUTE_QUERY_TOOL",
    "DBX_READONLY_TOOL_NAMES",
    "dbx_tool_description",
    "dbx_tool_input_schema",
    "filter_mcp_tool_names",
    "is_allowed_dbx_readonly_tool",
    "is_dbx_tool_name",
    "validate_dbx_tool_arguments",
]
