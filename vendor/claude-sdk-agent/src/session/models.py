from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class SessionMapping:
    frontend_session_id: str
    claude_session_id: str
    updated_at: float
    model: str = ""
    workspace_cwd: str = ""
    workspace_add_dirs: list[str] = field(default_factory=list)
    workspace_source: str = ""
    workspace_configured: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class SessionCheckpoint:
    frontend_session_id: str
    claude_session_id: str
    checkpoint_id: str
    created_at: float
    prompt_excerpt: str = ""
    unavailable_reason: str = ""
    affected_files: list[str] = field(default_factory=list)
    rewound_at: float = 0.0
    rewound_checkpoint_id: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class SessionGoal:
    frontend_session_id: str
    objective: str
    status: str
    created_at: float
    updated_at: float
    active_run_id: str = ""
    last_run_id: str = ""
    last_summary: str = ""
    pause_reason: str = ""
    paused_at: float = 0.0
    tasks: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
