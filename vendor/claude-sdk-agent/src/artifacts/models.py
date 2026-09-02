from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


ARTIFACT_SCHEMA_VERSION = "agent.artifacts/v1"


@dataclass(frozen=True, slots=True)
class ArtifactRoot:
    path: Path
    role: str


def run_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    summary = record.get("summary")
    return {
        "sessionId": str(record.get("sessionId") or ""),
        "runId": str(record.get("runId") or ""),
        "runtime": str(record.get("runtime") or ""),
        "createdAt": float(record.get("createdAt") or 0.0),
        "completedAt": float(record.get("completedAt") or 0.0),
        "status": str(record.get("status") or ""),
        "summary": dict(summary) if isinstance(summary, Mapping) else {},
    }
