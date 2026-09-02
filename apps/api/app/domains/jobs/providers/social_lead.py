from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any, Protocol

from pydantic import BaseModel, Field

from app.domains.jobs.models import JobSourceTrustLevel
from app.domains.jobs.schemas import JobLeadCreate


class ExtractedJobLead(BaseModel):
    company_name: str
    title: str
    city: str | None = None
    job_direction: str | None = None
    graduation_year: str | None = None
    source_url: str | None = None
    apply_url: str | None = None
    job_type: str | None = None
    salary_text: str | None = None
    jd_text: str | None = None
    skills: list[str] = Field(default_factory=list)
    deadline: date | None = None
    confidence_score: float | None = Field(default=None, ge=0, le=100)
    raw_payload: dict[str, Any] | None = None


class LeadExtractor(Protocol):
    def extract(
        self,
        raw_content: str,
        source_context: Mapping[str, Any],
    ) -> Sequence[ExtractedJobLead | Mapping[str, Any]]:
        ...


class SocialLeadImportProvider:
    name = "social_lead_import"

    def __init__(self, extractor: LeadExtractor) -> None:
        self._extractor = extractor

    def extract(
        self,
        source_id: str,
        raw_lead_id: str,
        raw_content: str,
        source_url: str | None,
        trust_level: JobSourceTrustLevel | str,
    ) -> list[JobLeadCreate]:
        normalized_trust_level = _normalize_trust_level(trust_level)
        source_context = {
            "source_id": source_id,
            "raw_lead_id": raw_lead_id,
            "source_url": source_url,
            "trust_level": normalized_trust_level.value,
        }
        extracted_items = self._extractor.extract(raw_content, source_context)

        drafts: list[JobLeadCreate] = []
        for item in extracted_items:
            extracted = ExtractedJobLead.model_validate(item)
            company_name = extracted.company_name.strip()
            title = extracted.title.strip()
            if not company_name or not title:
                continue

            drafts.append(
                JobLeadCreate(
                    source_id=source_id,
                    raw_lead_id=raw_lead_id,
                    company_name=company_name,
                    title=title,
                    city=_clean_text(extracted.city),
                    job_direction=_clean_text(extracted.job_direction),
                    graduation_year=_clean_text(extracted.graduation_year),
                    source_url=extracted.source_url or source_url,
                    apply_url=extracted.apply_url,
                    job_type=extracted.job_type,
                    salary_text=extracted.salary_text,
                    jd_text=extracted.jd_text,
                    skills=_normalize_skills(extracted.skills),
                    deadline=extracted.deadline,
                    confidence_score=extracted.confidence_score,
                    trust_level=normalized_trust_level,
                    raw_payload=extracted.raw_payload,
                )
            )
        return drafts


def _normalize_trust_level(trust_level: JobSourceTrustLevel | str) -> JobSourceTrustLevel:
    if isinstance(trust_level, JobSourceTrustLevel):
        return trust_level
    return JobSourceTrustLevel(str(trust_level))


def _clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(value.split())
    return cleaned or None


def _normalize_skills(skills: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for skill in skills:
        cleaned = _clean_text(skill)
        if cleaned is None:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(cleaned)
    return normalized
