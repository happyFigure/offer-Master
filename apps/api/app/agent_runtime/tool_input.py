from __future__ import annotations

import re


DEFAULT_SAMPLE_LIMIT = 10
MAX_SAMPLE_LIMIT = 50


def requested_sample_limit_from_text(text: str, *, default: int = DEFAULT_SAMPLE_LIMIT, maximum: int = MAX_SAMPLE_LIMIT) -> int:
    """Extract a user-requested list size such as "20个" or "三十家公司"."""

    message = str(text or "")
    for pattern in (
        r"(\d{1,3})\s*(?:个|家|条|行|家公司|个公司|家公司列表)",
        r"([零〇一二两三四五六七八九十百]{1,6})\s*(?:个|家|条|行|家公司|个公司|家公司列表)",
    ):
        match = re.search(pattern, message)
        if match is None:
            continue
        parsed = _parse_requested_count(match.group(1))
        if parsed is not None:
            return max(1, min(int(parsed), int(maximum)))
    return max(1, min(int(default), int(maximum)))


def _parse_requested_count(value: str) -> int | None:
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    return _parse_chinese_integer(text)


def _parse_chinese_integer(text: str) -> int | None:
    digits = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if text in digits:
        return digits[text]
    if text == "十":
        return 10
    if "百" in text:
        left, _, right = text.partition("百")
        hundreds = digits.get(left, 1 if left == "" else None)
        if hundreds is None:
            return None
        remainder = _parse_chinese_integer(right) if right else 0
        return hundreds * 100 + int(remainder or 0)
    if "十" in text:
        left, _, right = text.partition("十")
        tens = digits.get(left, 1 if left == "" else None)
        ones = digits.get(right, 0 if right == "" else None)
        if tens is None or ones is None:
            return None
        return tens * 10 + ones
    total = 0
    for char in text:
        if char not in digits:
            return None
        total = total * 10 + digits[char]
    return total
