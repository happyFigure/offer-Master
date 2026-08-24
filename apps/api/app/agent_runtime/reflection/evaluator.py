from __future__ import annotations

import re
from typing import Any

from app.agent_runtime.reflection.schemas import ReflectionDecision, ReflectionNextAction, ReflectionQuality


_CAMPUS_TERMS = ["校园招聘", "校招", "应届", "毕业生", "campus", "graduate", "students"]
_OFF_TARGET_TERMS = ["百度百科", "汉语", "汉字", "词典", "百科"]
_PUBLIC_OFF_TARGET_TERMS = [
    "百度百科",
    "百度知道",
    "汉语",
    "汉字",
    "词典",
    "百科",
    "utf-8",
    "编码转换",
    "与足球赛程无关",
    "与赛程无关",
    "与比赛无关",
    "向日葵远程控制",
    "360安全",
    "360官网",
    "软件管家",
    "安全卫士",
    "google docs editors help",
]
_FIXTURE_TERMS = ["比赛", "赛程", "日程", "fixtures", "fixture", "match", "schedule"]
_TIME_SENSITIVE_PUBLIC_TERMS = [
    *_FIXTURE_TERMS,
    "今天",
    "本周",
    "这周",
    "这个星期",
    "最近",
    "最新",
    "latest",
    "recent",
    "last match",
    "previous match",
    "next match",
]


class ReflectionEvaluator:
    def evaluate_web_search_result(
        self,
        *,
        tool_input: dict[str, Any],
        result_payload: dict[str, Any],
        expected_company_names: list[str],
    ) -> ReflectionDecision:
        result = result_payload.get("result") if isinstance(result_payload.get("result"), dict) else {}
        text = _web_search_text(result)
        expected_companies = [name.strip() for name in expected_company_names if name.strip()]
        if not expected_companies:
            return _evaluate_public_web_search_result(tool_input=tool_input, result_payload=result_payload, text=text)
        company_hit = any(name in text for name in expected_companies)
        campus_hit = _contains_any(text, _CAMPUS_TERMS)
        off_target = _contains_any(text, _OFF_TARGET_TERMS)
        ok = bool(result_payload.get("ok", True))

        if not ok:
            return ReflectionDecision(
                quality=ReflectionQuality.BAD,
                next_action=ReflectionNextAction.RETRY,
                confidence=0.2,
                reason="web search tool returned ok=false",
                suggested_input_patch={"query": _retry_query(tool_input, expected_companies)},
                metadata={"mode": "structured", "checks": {"tool_ok": False}},
            )

        if company_hit and campus_hit and not off_target:
            return ReflectionDecision(
                quality=ReflectionQuality.GOOD,
                next_action=ReflectionNextAction.CONTINUE,
                confidence=0.9,
                reason="web search result matches target company and campus recruiting intent",
                metadata={
                    "mode": "structured",
                    "checks": {"company_hit": True, "campus_hit": True, "off_target": False},
                },
            )

        if company_hit and not campus_hit and not off_target:
            return ReflectionDecision(
                quality=ReflectionQuality.UNKNOWN,
                next_action=ReflectionNextAction.RETRY,
                confidence=0.45,
                reason="web search result matches target company but campus recruiting intent is unclear",
                suggested_input_patch={"query": _retry_query(tool_input, expected_companies)},
                metadata={
                    "mode": "structured",
                    "checks": {
                        "company_hit": company_hit,
                        "campus_hit": campus_hit,
                        "off_target": off_target,
                    },
                },
            )

        reason_parts = []
        if not company_hit:
            reason_parts.append("result does not match target company")
        if not campus_hit:
            reason_parts.append("result does not match campus recruiting intent")
        if off_target:
            reason_parts.append("result appears off target")
        reason = "; ".join(reason_parts) or "web search result quality is insufficient"
        return ReflectionDecision(
            quality=ReflectionQuality.BAD,
            next_action=ReflectionNextAction.RETRY,
            confidence=0.3,
            reason=reason,
            suggested_input_patch={"query": _retry_query(tool_input, expected_companies)},
            metadata={
                "mode": "structured",
                "checks": {
                    "company_hit": company_hit,
                    "campus_hit": campus_hit,
                    "off_target": off_target,
                },
            },
        )


def _web_search_text(result: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("answer", "summary", "message"):
        value = result.get(key)
        if value:
            parts.append(str(value))
    sources = result.get("sources")
    if isinstance(sources, list):
        for source in sources:
            if not isinstance(source, dict):
                continue
            for key in ("title", "url", "snippet"):
                value = source.get(key)
                if value:
                    parts.append(str(value))
    return "\n".join(parts).lower()


def _contains_any(text: str, terms: list[str]) -> bool:
    normalized = text.lower()
    return any(term.lower() in normalized for term in terms)


def _evaluate_public_web_search_result(
    *,
    tool_input: dict[str, Any],
    result_payload: dict[str, Any],
    text: str,
) -> ReflectionDecision:
    ok = bool(result_payload.get("ok", True))
    if not ok:
        return ReflectionDecision(
            quality=ReflectionQuality.BAD,
            next_action=ReflectionNextAction.RETRY,
            confidence=0.2,
            reason="web search tool returned ok=false",
            suggested_input_patch={"query": _public_retry_query(tool_input)},
            metadata={"mode": "public_web", "checks": {"tool_ok": False}},
        )

    query = str(tool_input.get("query") or "")
    off_target = _contains_any(text, _PUBLIC_OFF_TARGET_TERMS)
    relevant = _public_result_matches_query(query, text)
    has_source_evidence = _web_search_has_traceable_source(result_payload)
    requires_source_evidence = _contains_any(query, _TIME_SENSITIVE_PUBLIC_TERMS)
    if relevant and not off_target and (has_source_evidence or not requires_source_evidence):
        return ReflectionDecision(
            quality=ReflectionQuality.GOOD,
            next_action=ReflectionNextAction.CONTINUE,
            confidence=0.8,
            reason="web search result matches the public information query",
            metadata={
                "mode": "public_web",
                "checks": {
                    "relevant": True,
                    "off_target": False,
                    "has_source_evidence": has_source_evidence,
                    "requires_source_evidence": requires_source_evidence,
                },
            },
        )

    reason_parts = []
    if not relevant:
        reason_parts.append("result does not match the public information query")
    if off_target:
        reason_parts.append("result appears off target")
    if relevant and requires_source_evidence and not has_source_evidence:
        reason_parts.append("result lacks traceable source evidence")
    return ReflectionDecision(
        quality=ReflectionQuality.BAD,
        next_action=ReflectionNextAction.RETRY,
        confidence=0.35,
        reason="; ".join(reason_parts) or "web search result quality is insufficient",
        suggested_input_patch={"query": _public_retry_query(tool_input)},
        metadata={
            "mode": "public_web",
            "checks": {
                "relevant": relevant,
                "off_target": off_target,
                "has_source_evidence": has_source_evidence,
                "requires_source_evidence": requires_source_evidence,
            },
        },
    )


def _web_search_has_traceable_source(result_payload: dict[str, Any]) -> bool:
    result = result_payload.get("result") if isinstance(result_payload.get("result"), dict) else {}
    sources = result.get("sources")
    if isinstance(sources, list):
        for source in sources:
            if isinstance(source, str) and _looks_like_url(source):
                return True
            if isinstance(source, dict) and _looks_like_url(str(source.get("url") or "")):
                return True
    artifacts = result.get("artifacts")
    if isinstance(artifacts, list):
        for artifact in artifacts:
            if isinstance(artifact, dict) and _looks_like_url(str(artifact.get("url") or "")):
                return True
    results = result.get("results")
    if isinstance(results, list):
        for item in results:
            if isinstance(item, dict) and _looks_like_url(str(item.get("url") or "")):
                return True
    return False


def _looks_like_url(value: str) -> bool:
    lowered = str(value or "").strip().lower()
    return lowered.startswith("http://") or lowered.startswith("https://")


def _public_result_matches_query(query: str, text: str) -> bool:
    query_lower = str(query or "").lower()
    text_lower = str(text or "").lower()
    if _contains_any(query_lower, _FIXTURE_TERMS):
        fixture_hit = _contains_any(text_lower, _FIXTURE_TERMS)
        if not fixture_hit:
            return False
        target_terms: list[str] = []
        if _contains_any(query_lower, ["cristiano ronaldo", "ronaldo", "c罗", "al nassr", "al-nassr", "portugal"]):
            target_terms.extend(["cristiano ronaldo", "ronaldo", "c罗", "al nassr", "al-nassr", "利雅得胜利", "portugal", "葡萄牙"])
        if _contains_any(query_lower, ["lionel messi", "messi", "梅西", "inter miami", "argentina"]):
            target_terms.extend(["lionel messi", "messi", "梅西", "inter miami", "迈阿密国际", "argentina", "阿根廷"])
        return _contains_any(text_lower, target_terms) if target_terms else fixture_hit
    query_terms = [term for term in re.split(r"\s+", query_lower) if len(term) >= 3]
    if not query_terms:
        return bool(text_lower.strip())
    hits = sum(1 for term in query_terms if term in text_lower)
    return hits >= 1


def _public_retry_query(tool_input: dict[str, Any]) -> str:
    original = re.sub(r"\s+", " ", str(tool_input.get("query") or "").strip())
    lowered = original.lower()
    asks_last_match = _contains_any(
        lowered,
        [
            "最近一次",
            "最近的一次",
            "最近一场",
            "最近的一场",
            "上一场",
            "上场比赛",
            "上一次比赛",
            "latest match",
            "latest result",
            "last match",
            "previous match",
        ],
    )
    if "cristiano ronaldo" in lowered or "c罗" in original or "c羅" in original:
        if asks_last_match:
            return "Cristiano Ronaldo last match result date ESPN Flashscore SofaScore Al Nassr Portugal"
        return "Cristiano Ronaldo next match fixtures ESPN Flashscore SofaScore Al Nassr Portugal"
    if "lionel messi" in lowered or "梅西" in original:
        if asks_last_match:
            return "Lionel Messi last match result date ESPN Flashscore SofaScore Inter Miami Argentina"
        return "Lionel Messi next match fixtures ESPN Flashscore SofaScore Inter Miami Argentina"
    return original or "latest public information official source"


def _retry_query(tool_input: dict[str, Any], expected_companies: list[str]) -> str:
    original = str(tool_input.get("query") or "").strip()
    company = expected_companies[0] if expected_companies else ""
    base = f"{company} 校园招聘 官网 2026".strip()
    if company and company not in base:
        base = f"{company} {base}"
    query = base or original or "校园招聘 官网 2026"
    query = re.sub(r"\s+", " ", query).strip()
    return query
