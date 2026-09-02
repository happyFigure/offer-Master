from __future__ import annotations

import re


_ISO_DATE_RE = re.compile(r"20\d{2}-\d{2}-\d{2}")
_FIXTURE_MARKERS = (
    "比赛",
    "赛程",
    "日程",
    "安排",
    "schedule",
    "fixtures",
    "fixture",
    "match",
    "matches",
)
_WEEK_MARKERS = ("本周", "这周", "这个星期", "这星期", "本星期", "这个礼拜", "本礼拜", "this week")
_TODAY_MARKERS = ("今天", "今晚", "today", "tonight")
_LAST_MATCH_MARKERS = (
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
)


def normalize_external_web_search_query(query: str) -> str:
    """Turn chatty user wording into search-engine-friendly query text."""

    normalized = str(query or "").strip()
    if not normalized:
        return normalized

    lowered = normalized.lower()
    mentions_ronaldo = _mentions_any(normalized, ("C罗", "c罗", "C羅", "c羅", "c 羅"))
    mentions_messi = "梅西" in normalized
    if _looks_like_fixture_query(normalized):
        if mentions_ronaldo:
            return _sports_fixture_query(
                query=normalized,
                player="Cristiano Ronaldo",
                local_alias="C罗",
                teams=("Al Nassr", "Portugal"),
            )
        if mentions_messi:
            return _sports_fixture_query(
                query=normalized,
                player="Lionel Messi",
                local_alias="梅西",
                teams=("Inter Miami", "Argentina"),
            )

    additions: list[str] = []
    if mentions_ronaldo:
        if "cristiano ronaldo" not in lowered:
            normalized = normalized.replace("C罗", "Cristiano Ronaldo C罗").replace("c罗", "Cristiano Ronaldo C罗")
            normalized = normalized.replace("C羅", "Cristiano Ronaldo C羅").replace("c羅", "Cristiano Ronaldo C羅")
        additions.extend(["Al Nassr", "Portugal", "football fixtures"])
    if mentions_messi and "lionel messi" not in lowered:
        normalized = normalized.replace("梅西", "Lionel Messi 梅西")
        additions.extend(["Inter Miami", "Argentina", "football fixtures"])

    return _append_missing(normalized, additions)


def _sports_fixture_query(*, query: str, player: str, local_alias: str, teams: tuple[str, ...]) -> str:
    date_hint = _extract_iso_date(query)
    if _mentions_any(query, _LAST_MATCH_MARKERS):
        return _dedupe_join([player, local_alias, *teams, "last match", "result date", "ESPN", "Flashscore", "SofaScore"])
    pieces = [player, local_alias, *teams, "football fixtures", "match schedule"]
    if _mentions_any(query, _WEEK_MARKERS):
        pieces.append("this week")
        if date_hint:
            pieces.extend(["week of", date_hint])
    elif _mentions_any(query, _TODAY_MARKERS):
        pieces.append("today")
        if date_hint:
            pieces.extend(["on", date_hint])
    elif date_hint:
        pieces.append(date_hint)
    return _dedupe_join(pieces)


def _looks_like_fixture_query(query: str) -> bool:
    lowered = query.lower()
    return any(marker in lowered for marker in _FIXTURE_MARKERS)


def _extract_iso_date(query: str) -> str | None:
    match = _ISO_DATE_RE.search(query)
    return match.group(0) if match else None


def _mentions_any(text: str, markers: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in markers)


def _append_missing(base: str, additions: list[str]) -> str:
    normalized = base
    existing_lower = normalized.lower()
    for item in additions:
        if item.lower() not in existing_lower:
            normalized = f"{normalized} {item}"
            existing_lower = normalized.lower()
    return normalized


def _dedupe_join(pieces: list[str]) -> str:
    output: list[str] = []
    seen: set[str] = set()
    for piece in pieces:
        value = str(piece or "").strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        output.append(value)
        seen.add(key)
    return " ".join(output)


__all__ = ["normalize_external_web_search_query"]
