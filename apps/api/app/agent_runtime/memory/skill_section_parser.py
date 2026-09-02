from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable

from app.agent_runtime.memory.token_budget import estimate_tokens


_SECTION_HEADING_PATTERN = re.compile(r"^##\s+(.+?)\s*$")


@dataclass(frozen=True)
class SkillSection:
    heading: str
    content: str
    start_line: int
    end_line: int
    token_estimate: int

    def to_metadata(self) -> dict[str, Any]:
        return {
            "heading": self.heading,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "token_estimate": self.token_estimate,
        }


@dataclass(frozen=True)
class SkillSectionSelection:
    sections: tuple[SkillSection, ...]
    content: str
    content_chars: int
    token_estimate: int
    max_chars: int
    truncated: bool

    def to_metadata(self) -> dict[str, Any]:
        return {
            "load_layer": "section",
            "section_count": len(self.sections),
            "sections": [section.to_metadata() for section in self.sections],
            "content_chars": self.content_chars,
            "token_estimate": self.token_estimate,
            "max_chars": self.max_chars,
            "truncated": self.truncated,
        }


def parse_skill_sections(content: str) -> list[SkillSection]:
    lines = content.splitlines()
    starts: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        match = _SECTION_HEADING_PATTERN.match(line.strip())
        if match is None:
            continue
        starts.append((index, _clean_heading(match.group(1))))

    sections: list[SkillSection] = []
    for position, (start_index, heading) in enumerate(starts):
        next_start = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        body_lines = lines[start_index + 1 : next_start]
        body = "\n".join(body_lines).strip()
        section_text = _format_section(heading, body)
        sections.append(
            SkillSection(
                heading=heading,
                content=body,
                start_line=start_index + 1,
                end_line=next_start,
                token_estimate=estimate_tokens(section_text),
            )
        )
    return sections


def select_skill_sections(
    sections: Iterable[SkillSection],
    headings: Iterable[str],
    *,
    max_chars: int,
) -> SkillSectionSelection:
    section_by_heading = {_normalize_heading(section.heading): section for section in sections}
    selected_sections: list[SkillSection] = []
    seen: set[str] = set()
    for heading in headings:
        key = _normalize_heading(heading)
        if not key or key in seen:
            continue
        section = section_by_heading.get(key)
        if section is None:
            continue
        selected_sections.append(section)
        seen.add(key)

    raw_content = "\n\n".join(_format_section(section.heading, section.content) for section in selected_sections).strip()
    truncated = max_chars > 0 and len(raw_content) > max_chars
    selected_content = raw_content[:max_chars].rstrip() if truncated else raw_content
    return SkillSectionSelection(
        sections=tuple(selected_sections),
        content=selected_content,
        content_chars=len(selected_content),
        token_estimate=estimate_tokens(selected_content),
        max_chars=max_chars,
        truncated=truncated,
    )


def _format_section(heading: str, content: str) -> str:
    body = content.strip()
    return f"## {heading}\n{body}".strip()


def _clean_heading(heading: str) -> str:
    return heading.strip().strip("#").strip()


def _normalize_heading(heading: str) -> str:
    return _clean_heading(heading).casefold()


__all__ = ["SkillSection", "SkillSectionSelection", "parse_skill_sections", "select_skill_sections"]
