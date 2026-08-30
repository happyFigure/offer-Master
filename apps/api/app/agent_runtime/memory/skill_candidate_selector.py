from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import re

from app.agent_runtime.memory.skill_summary_index import SkillSummaryCard


@dataclass(frozen=True)
class SkillCandidate:
    card: SkillSummaryCard
    score: float
    matched_terms: tuple[str, ...]
    reason: str

    def to_metadata(self) -> dict[str, Any]:
        return {
            "skill_id": self.card.skill_id,
            "name": self.card.name,
            "title": self.card.title,
            "score": self.score,
            "matched_terms": list(self.matched_terms),
            "reason": self.reason,
            "auto_load_enabled": self.card.auto_load_enabled,
            "risk_level": self.card.risk_level,
        }


def select_skill_candidates(
    query: str,
    cards: Iterable[SkillSummaryCard],
    *,
    limit: int = 5,
) -> list[SkillCandidate]:
    if limit <= 0:
        return []

    query_terms = _query_terms(query)
    if not query_terms:
        return []

    scored: list[tuple[float, int, SkillCandidate]] = []
    for index, card in enumerate(cards):
        score, matched_terms = _score_card(query_terms, card)
        if score <= 0:
            continue
        candidate = SkillCandidate(
            card=card,
            score=round(score, 3),
            matched_terms=matched_terms,
            reason=_reason(matched_terms),
        )
        scored.append((score, index, candidate))

    scored.sort(
        key=lambda item: (
            -item[0],
            not item[2].card.auto_load_enabled,
            not item[2].card.pinned,
            item[1],
        )
    )
    return [candidate for _, _, candidate in scored[:limit]]


def _score_card(query_terms: tuple[str, ...], card: SkillSummaryCard) -> tuple[float, tuple[str, ...]]:
    weighted_fields = [
        (4.0, " ".join([card.name, card.title])),
        (3.0, card.when_to_use),
        (2.0, card.description),
        (5.0, " ".join(_source_type_aliases(card.source_types))),
        (1.0, " ".join([card.category, *card.source_types])),
        (0.4, " ".join([*card.allowed_tools, *card.ask_tools, *card.disallowed_tools])),
    ]

    score = 0.0
    matched_terms: list[str] = []
    for term in query_terms:
        term_score = 0.0
        for weight, field in weighted_fields:
            if term in _normalize(field):
                term_score = max(term_score, weight)
        if term_score <= 0:
            continue
        matched_terms.append(term)
        score += term_score * _term_weight(term)

    if card.pinned and score > 0:
        score += 0.2
    return score, tuple(_dedupe(matched_terms))


def _query_terms(query: str) -> tuple[str, ...]:
    normalized = _normalize(query)
    terms: list[str] = []

    for phrase in _IMPORTANT_PHRASES:
        if phrase in normalized:
            terms.append(phrase)

    terms.extend(
        token
        for token in re.findall(r"[a-z0-9_+.-]+", normalized)
        if len(token) >= 2 and token not in _STOP_WORDS
    )

    cjk_chunks = re.findall(r"[\u4e00-\u9fff]+", normalized)
    for chunk in cjk_chunks:
        if len(chunk) == 2:
            terms.append(chunk)
        elif len(chunk) > 2:
            terms.extend(chunk[index : index + 2] for index in range(len(chunk) - 1))

    return tuple(_dedupe(terms))


def _reason(matched_terms: tuple[str, ...]) -> str:
    preview = "、".join(matched_terms[:8])
    return f"命中用户问题里的关键词：{preview}"


def _term_weight(term: str) -> float:
    if term in _IMPORTANT_PHRASES:
        return 2.0
    if re.search(r"[\u4e00-\u9fff]", term):
        return 1.0
    return 0.8


def _normalize(text: str) -> str:
    return str(text or "").strip().lower()


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        cleaned = item.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def _source_type_aliases(source_types: tuple[str, ...]) -> tuple[str, ...]:
    aliases: list[str] = []
    for source_type in source_types:
        aliases.extend(_SOURCE_TYPE_ALIASES.get(source_type, ()))
    return tuple(_dedupe(aliases))


_IMPORTANT_PHRASES = (
    "公众号",
    "微信",
    "文章",
    "小红书",
    "简历",
    "岗位",
    "岗位线索",
    "公司",
    "校招",
    "秋招",
    "jd",
    "目标jd",
    "联网搜索",
    "网页搜索",
    "数据库",
)


_SOURCE_TYPE_ALIASES = {
    "xiaohongshu_note": ("小红书", "红书", "xiaohongshu", "xhs", "笔记"),
    "wechat_article": ("微信公众号", "公众号", "微信", "微信文章", "文章"),
    "wechat_account": ("微信公众号", "公众号", "微信"),
    "resume_text": ("简历", "履历", "resume"),
    "job_description": ("jd", "目标jd", "岗位描述", "职位描述"),
    "company": ("公司", "企业"),
    "job_lead": ("岗位线索", "岗位", "招聘"),
}

_STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "use",
    "using",
    "user",
}


__all__ = ["SkillCandidate", "select_skill_candidates"]
