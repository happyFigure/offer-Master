from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from app.domains.jobs.models import RecruitingSignalType
from app.domains.jobs.schemas import RecruitingSignalCreate


RECRUITING_KEYWORDS = ("招聘", "校招", "校园招聘", "秋招", "春招", "实习")
NAVIGATION_COLLECTION_TERMS = ("合集", "就业政策", "宣讲信息", "实习信息", "选调生", "公务员", "事业单位")
TITLE_PREFIX_RE = re.compile(r"^[\s\[\]【】()（）招聘校招秋招春招实习\-—_：:]+")
YEAR_RE = re.compile(r"(?P<year>20\d{2})\s*(届)?")


class RuleBasedRecruitingSignalProvider:
    name = "rule_based_recruiting_signal"

    def extract(
        self,
        *,
        source_id: str,
        raw_lead_id: str | None,
        raw_content: str,
        source_url: str | None,
        trust_level: object,
        source_context: Mapping[str, Any] | None = None,
    ) -> list[RecruitingSignalCreate]:
        context = source_context or {}
        candidates = [str(context.get("title") or ""), *raw_content.splitlines()]
        original_source = _extract_original_source(raw_content)
        seen: set[tuple[str, str | None, str]] = set()
        drafts: list[RecruitingSignalCreate] = []

        for line in candidates:
            parsed = _parse_signal_line(line)
            if parsed is None:
                continue
            company_name, graduation_year, signal_type = parsed
            key = (company_name, graduation_year, signal_type.value)
            if key in seen:
                continue
            seen.add(key)
            drafts.append(
                RecruitingSignalCreate(
                    source_id=source_id,
                    raw_lead_id=raw_lead_id,
                    company_name=company_name,
                    signal_type=signal_type,
                    graduation_year=graduation_year,
                    source_url=source_url,
                    original_source=original_source,
                    confidence_score=82.0 if graduation_year else 72.0,
                    trust_level=trust_level,
                    raw_payload={
                        "extraction_method": self.name,
                        "matched_line": line,
                    },
                )
            )
        return drafts


def _parse_signal_line(line: str) -> tuple[str, str | None, RecruitingSignalType] | None:
    cleaned = " ".join(line.strip().split())
    if not cleaned or not any(keyword in cleaned for keyword in RECRUITING_KEYWORDS):
        return None
    if _is_navigation_collection_line(cleaned):
        return None

    year_match = YEAR_RE.search(cleaned)
    if year_match is None:
        return None

    graduation_year = year_match.group("year")
    company_text = cleaned[: year_match.start()]
    company_name = _clean_company_fragment(company_text)
    if not company_name:
        return None

    signal_type = RecruitingSignalType.INTERNSHIP_OPEN if "实习" in cleaned and "校招" not in cleaned else RecruitingSignalType.CAMPUS_RECRUITMENT_OPEN
    return company_name, graduation_year, signal_type


def _is_navigation_collection_line(line: str) -> bool:
    if "合集" not in line:
        return False
    matched_terms = sum(1 for term in NAVIGATION_COLLECTION_TERMS if term in line)
    return matched_terms >= 3


def _clean_company_fragment(value: str) -> str | None:
    cleaned = TITLE_PREFIX_RE.sub("", value.strip())
    cleaned = re.split(r"[!！。；;，,：:\s]+", cleaned)[-1].strip()
    cleaned = cleaned.strip("[]【】()（）-—_· ")
    if not cleaned or cleaned in {"招聘", "校招", "秋招", "春招", "实习"}:
        return None
    if len(cleaned) > 40:
        return None
    return cleaned


def _extract_original_source(raw_content: str) -> str | None:
    for line in raw_content.splitlines():
        if "信息来源" not in line:
            continue
        _, _, tail = line.partition("：")
        if not tail:
            _, _, tail = line.partition(":")
        cleaned = tail.strip()
        return cleaned or None
    return None
