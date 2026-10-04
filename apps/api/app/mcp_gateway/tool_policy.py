from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from app.mcp_gateway.dbx_readonly import filter_mcp_tool_names


CONFIRMATION_REQUIRED_TOOLS = frozenset(
    {
        "fill_form",
        "submit_application",
        "click_submit",
        "browser_submit",
        # Chrome DevTools MCP names these actions without a browser prefix.
        # They can change page state, so the runtime approval boundary applies
        # even when a model emits the call from a child-agent turn.
        "click",
        "fill",
        "type_text",
        "press_key",
    }
)


@dataclass(frozen=True)
class MCPToolPolicy:
    _allowed_tool_names: tuple[str, ...]

    @classmethod
    def from_allowlist(cls, tool_names: Iterable[str]) -> MCPToolPolicy:
        normalized = []
        seen = set()
        for raw_name in tool_names:
            tool_name = raw_name.strip()
            if not tool_name or tool_name in seen:
                continue
            seen.add(tool_name)
            normalized.append(tool_name)
        return cls(tuple(filter_mcp_tool_names(normalized)))

    def allowed_tool_names(self) -> list[str]:
        return list(self._allowed_tool_names)

    def is_allowed(self, tool_name: str) -> bool:
        return tool_name.strip() in self._allowed_tool_names

    def requires_confirmation(self, tool_name: str) -> bool:
        normalized = tool_name.strip().lower().replace("-", "_")
        action = normalized.rsplit(".", 1)[-1]
        return normalized in CONFIRMATION_REQUIRED_TOOLS or action in CONFIRMATION_REQUIRED_TOOLS
