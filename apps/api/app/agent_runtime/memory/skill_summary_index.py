from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agent_runtime.memory.skill_repository import SkillDocument


@dataclass(frozen=True)
class SkillSummaryCard:
    skill_id: str
    name: str
    title: str
    description: str
    category: str
    when_to_use: str
    source_types: tuple[str, ...]
    allowed_tools: tuple[str, ...]
    ask_tools: tuple[str, ...]
    disallowed_tools: tuple[str, ...]
    risk_level: str
    protected: bool
    pinned: bool
    auto_load_enabled: bool
    description_quality_score: int
    version_hash: str
    summary_text: str

    def to_metadata(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "when_to_use": self.when_to_use,
            "source_types": list(self.source_types),
            "allowed_tools": list(self.allowed_tools),
            "ask_tools": list(self.ask_tools),
            "disallowed_tools": list(self.disallowed_tools),
            "risk_level": self.risk_level,
            "protected": self.protected,
            "pinned": self.pinned,
            "auto_load_enabled": self.auto_load_enabled,
            "description_quality_score": self.description_quality_score,
            "version_hash": self.version_hash,
            "summary_text": self.summary_text,
        }


def build_skill_summary_card(document: SkillDocument) -> SkillSummaryCard:
    skill = document.skill
    metadata = dict(skill.metadata_json or {})
    source_types = tuple(_metadata_list(metadata.get("source_types")))
    allowed_tools = tuple(_metadata_list(metadata.get("allowed_tools")))
    ask_tools = tuple(_metadata_list(metadata.get("ask_tools")))
    disallowed_tools = tuple(_metadata_list(metadata.get("disallowed_tools")))
    risk_level = _risk_level(
        metadata,
        allowed_tools=allowed_tools,
        ask_tools=ask_tools,
        disallowed_tools=disallowed_tools,
    )
    description_quality_score = _int_value(metadata.get("description_quality_score"))
    when_to_use = _extract_section(document.content, "何时使用") or str(metadata.get("when_to_use") or "").strip()
    auto_load_enabled = str(metadata.get("auto_trigger_state") or "").strip().lower() == "enabled"

    summary_text = _summary_text(
        title=skill.title,
        name=skill.name,
        description=skill.description,
        category=skill.category,
        when_to_use=when_to_use,
        source_types=source_types,
        allowed_tools=allowed_tools,
        ask_tools=ask_tools,
        disallowed_tools=disallowed_tools,
        risk_level=risk_level,
        auto_load_enabled=auto_load_enabled,
    )

    return SkillSummaryCard(
        skill_id=skill.id,
        name=skill.name,
        title=skill.title,
        description=skill.description,
        category=skill.category,
        when_to_use=when_to_use,
        source_types=source_types,
        allowed_tools=allowed_tools,
        ask_tools=ask_tools,
        disallowed_tools=disallowed_tools,
        risk_level=risk_level,
        protected=bool(skill.protected),
        pinned=bool(skill.pinned),
        auto_load_enabled=auto_load_enabled,
        description_quality_score=description_quality_score,
        version_hash=document.version_hash,
        summary_text=summary_text,
    )


def _summary_text(
    *,
    title: str,
    name: str,
    description: str,
    category: str,
    when_to_use: str,
    source_types: tuple[str, ...],
    allowed_tools: tuple[str, ...],
    ask_tools: tuple[str, ...],
    disallowed_tools: tuple[str, ...],
    risk_level: str,
    auto_load_enabled: bool,
) -> str:
    lines = [
        f"技能：{title}（{name}）",
        f"一句话说明：{description}",
        f"分类：{category}",
    ]
    if when_to_use:
        lines.append(f"适用场景：{when_to_use}")
    if source_types:
        lines.append(f"来源类型：{', '.join(source_types)}")
    tool_boundary_parts: list[str] = []
    if allowed_tools:
        tool_boundary_parts.append(f"自动允许：{', '.join(allowed_tools)}")
    if ask_tools:
        tool_boundary_parts.append(f"需确认：{', '.join(ask_tools)}")
    if disallowed_tools:
        tool_boundary_parts.append(f"禁止：{', '.join(disallowed_tools)}")
    if tool_boundary_parts:
        lines.append(f"工具边界：{'; '.join(tool_boundary_parts)}")
    lines.append(f"风险等级：{risk_level}")
    lines.append(f"自动加载：{'是' if auto_load_enabled else '否'}")
    return "\n".join(lines)


def _metadata_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _risk_level(
    metadata: dict[str, Any],
    *,
    allowed_tools: tuple[str, ...],
    ask_tools: tuple[str, ...],
    disallowed_tools: tuple[str, ...],
) -> str:
    explicit = (
        str(metadata.get("security_risk_level") or metadata.get("risk_level") or "")
        .strip()
        .lower()
    )
    if explicit:
        return explicit
    if disallowed_tools or ask_tools:
        return "medium"
    if allowed_tools:
        return "low"
    return "unknown"


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _extract_section(content: str, heading: str) -> str:
    marker = f"## {heading}"
    lines = content.splitlines()
    start_index: int | None = None
    for index, line in enumerate(lines):
        if line.strip() == marker:
            start_index = index + 1
            break
    if start_index is None:
        return ""

    section_lines: list[str] = []
    for line in lines[start_index:]:
        if line.startswith("## "):
            break
        section_lines.append(line)
    return "\n".join(section_lines).strip()


__all__ = ["SkillSummaryCard", "build_skill_summary_card"]
