from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from app.agent_runtime.memory.skill_candidate_selector import SkillCandidate
from app.agent_runtime.memory.skill_section_parser import parse_skill_sections, select_skill_sections
from app.agent_runtime.memory.skill_repository import SkillDocument
from app.agent_runtime.memory.token_budget import estimate_tokens


@dataclass(frozen=True)
class SkillLoadRecord:
    skill_id: str
    name: str
    title: str
    load_layer: str
    selected_by: str
    reason: str
    version_hash: str
    content_chars: int
    token_estimate: int
    max_chars: int
    truncated: bool
    selected_sections: tuple[str, ...] = ()

    def to_metadata(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "title": self.title,
            "load_layer": self.load_layer,
            "selected_by": self.selected_by,
            "reason": self.reason,
            "version_hash": self.version_hash,
            "content_chars": self.content_chars,
            "token_estimate": self.token_estimate,
            "max_chars": self.max_chars,
            "truncated": self.truncated,
            "selected_sections": list(self.selected_sections),
        }


@dataclass(frozen=True)
class SkillContextLoadResult:
    loaded_documents: tuple[SkillDocument, ...]
    load_records: tuple[SkillLoadRecord, ...]

    def to_metadata(self, *, candidate_count: int, max_loaded_skills: int) -> dict[str, Any]:
        return {
            "strategy": "load_selected_skill_bodies",
            "candidate_count": candidate_count,
            "loaded_count": len(self.loaded_documents),
            "max_loaded_skills": max_loaded_skills,
            "loaded_skills": [record.to_metadata() for record in self.load_records],
        }


def load_selected_skill_context(
    candidates: list[SkillCandidate],
    documents_by_skill_id: Mapping[str, SkillDocument],
    *,
    max_loaded_skills: int,
    max_skill_context_chars: int,
    section_headings: Iterable[str] | None = None,
) -> SkillContextLoadResult:
    if max_loaded_skills <= 0:
        return SkillContextLoadResult(loaded_documents=(), load_records=())

    loaded_documents: list[SkillDocument] = []
    load_records: list[SkillLoadRecord] = []
    for candidate in candidates[:max_loaded_skills]:
        document = documents_by_skill_id.get(candidate.card.skill_id)
        if document is None:
            continue
        loaded_documents.append(document)
        load_records.append(
            _load_record(
                document,
                candidate=candidate,
                max_skill_context_chars=max_skill_context_chars,
                section_headings=section_headings,
            )
        )

    return SkillContextLoadResult(loaded_documents=tuple(loaded_documents), load_records=tuple(load_records))


def _load_record(
    document: SkillDocument,
    *,
    candidate: SkillCandidate,
    max_skill_context_chars: int,
    section_headings: Iterable[str] | None,
) -> SkillLoadRecord:
    raw_content = document.content.strip()
    selected_section_names: tuple[str, ...] = ()
    loaded_content: str | None = None
    truncated = False
    if section_headings:
        section_selection = select_skill_sections(
            parse_skill_sections(raw_content),
            section_headings,
            max_chars=max_skill_context_chars,
        )
        if section_selection.sections:
            selected_section_names = tuple(section.heading for section in section_selection.sections)
            loaded_content = section_selection.content
            truncated = section_selection.truncated

    content_chars = len(raw_content)
    if loaded_content is None:
        truncated = max_skill_context_chars > 0 and content_chars > max_skill_context_chars
        loaded_content = raw_content[:max_skill_context_chars].rstrip() if truncated else raw_content
    return SkillLoadRecord(
        skill_id=document.skill.id,
        name=document.skill.name,
        title=document.skill.title,
        load_layer="section" if selected_section_names else "body",
        selected_by="skill_summary_candidate",
        reason=candidate.reason,
        version_hash=document.version_hash,
        content_chars=len(loaded_content),
        token_estimate=estimate_tokens(loaded_content),
        max_chars=max_skill_context_chars,
        truncated=truncated,
        selected_sections=selected_section_names,
    )


__all__ = ["SkillContextLoadResult", "SkillLoadRecord", "load_selected_skill_context"]
