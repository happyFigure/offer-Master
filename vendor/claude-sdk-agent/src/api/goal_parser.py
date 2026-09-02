from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GoalCommand:
    action: str
    raw: str
    condition: str = ""


def parse_goal_command(text: str) -> GoalCommand | None:
    raw = str(text or "")
    stripped = raw.strip()
    if not stripped.startswith("/goal"):
        return None
    if stripped == "/goal":
        return GoalCommand(action="query", raw=raw)
    remainder = stripped[len("/goal") :].strip()
    if not remainder:
        return GoalCommand(action="query", raw=raw)
    if remainder == "clear":
        return GoalCommand(action="clear", raw=raw)
    return GoalCommand(action="start", raw=raw, condition=remainder)
