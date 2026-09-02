from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.agent_runtime.memory.skill_repository import SkillDocument
from app.agent_runtime.memory.token_budget import estimate_tokens


@dataclass(frozen=True)
class SkillResourceLoadResult:
    resource_path: str
    content: str
    load_layer: str
    content_chars: int
    token_estimate: int
    max_chars: int
    truncated: bool

    def to_metadata(self) -> dict[str, Any]:
        return {
            "resource_path": self.resource_path,
            "load_layer": self.load_layer,
            "content_chars": self.content_chars,
            "token_estimate": self.token_estimate,
            "max_chars": self.max_chars,
            "truncated": self.truncated,
        }


def load_skill_resource(
    skill_or_document: Any,
    resource_path: str,
    *,
    max_chars: int = 4000,
) -> SkillResourceLoadResult:
    skill = skill_or_document.skill if isinstance(skill_or_document, SkillDocument) else skill_or_document
    base_dir = _skill_base_dir(skill).resolve()
    target = _resolve_resource_path(base_dir, resource_path)
    if not target.is_file():
        raise FileNotFoundError(f"Skill resource not found: {resource_path}")

    raw_content = target.read_text(encoding="utf-8")
    truncated = max_chars > 0 and len(raw_content) > max_chars
    content = raw_content[:max_chars].rstrip() if truncated else raw_content
    return SkillResourceLoadResult(
        resource_path=_to_posix_relative_path(base_dir, target),
        content=content,
        load_layer="resource",
        content_chars=len(content),
        token_estimate=estimate_tokens(content),
        max_chars=max_chars,
        truncated=truncated,
    )


def _skill_base_dir(skill: Any) -> Path:
    file_path = getattr(skill, "file_path", None)
    if not file_path:
        raise ValueError("Skill has no markdown file path")
    skill_file = Path(file_path)
    if skill_file.name == "SKILL.md":
        return skill_file.parent
    return skill_file


def _resolve_resource_path(base_dir: Path, resource_path: str) -> Path:
    if not resource_path or not resource_path.strip():
        raise ValueError("Skill resource path is required")
    requested = Path(resource_path)
    if requested.is_absolute():
        raise PermissionError("Skill resource path must stay inside the skill package")
    target = (base_dir / requested).resolve()
    try:
        target.relative_to(base_dir)
    except ValueError as exc:
        raise PermissionError("Skill resource path must stay inside the skill package") from exc
    return target


def _to_posix_relative_path(base_dir: Path, target: Path) -> str:
    return target.relative_to(base_dir).as_posix()


__all__ = ["SkillResourceLoadResult", "load_skill_resource"]
