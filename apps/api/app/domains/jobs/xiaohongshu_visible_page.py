from __future__ import annotations

import re
from dataclasses import dataclass


COMMENT_STOP_PREFIXES = (
    "共 ",
    "登录查看全部评论内容",
    "登录后评论",
    "发送",
    "取消",
)
NOISE_LINES = {
    "登录后推荐更懂你的笔记",
    "小红书",
    "或",
    "微信",
    "扫码",
    "小红书如何扫码",
    "手机号登录",
    "获取验证码",
    "登录",
    "关注",
    "创作中心",
    "业务合作",
    "发现",
    "RED",
    "直播",
    "发布",
    "通知",
}


@dataclass(frozen=True)
class ParsedXiaohongshuVisibleText:
    title: str | None
    text: str
    image_count: int
    image_parse_deferred: bool


def parse_xiaohongshu_visible_text(
    visible_text: str,
    *,
    page_title: str | None = None,
) -> ParsedXiaohongshuVisibleText:
    lines = [_clean_line(line) for line in visible_text.splitlines()]
    lines = [line for line in lines if line]
    image_count = _extract_image_count(lines)
    title = _normalize_page_title(page_title) or _guess_title(lines)
    note_lines = _extract_note_lines(lines, title)
    text = "\n".join(note_lines)
    return ParsedXiaohongshuVisibleText(
        title=title,
        text=text,
        image_count=image_count,
        image_parse_deferred=image_count > 0,
    )


def _extract_note_lines(lines: list[str], title: str | None) -> list[str]:
    start_index = 0
    if title:
        for index, line in enumerate(lines):
            if line == title:
                start_index = index
                break

    body: list[str] = []
    for line in lines[start_index:]:
        if _is_comment_or_footer_start(line):
            break
        if line in NOISE_LINES:
            continue
        if re.fullmatch(r"\d+/\d+", line):
            continue
        body.append(line)
    return body


def _normalize_page_title(page_title: str | None) -> str | None:
    if not page_title:
        return None
    title = page_title.strip()
    title = re.sub(r"\s*-\s*小红书\s*$", "", title)
    return title or None


def _guess_title(lines: list[str]) -> str | None:
    for line in lines:
        if any(term in line for term in ("秋招", "校招", "实习", "招聘")) and len(line) <= 80:
            return line
    return None


def _extract_image_count(lines: list[str]) -> int:
    for line in lines:
        match = re.fullmatch(r"\d+/(?P<count>\d+)", line)
        if match:
            return int(match.group("count"))
    return 0


def _is_comment_or_footer_start(line: str) -> bool:
    return any(line.startswith(prefix) for prefix in COMMENT_STOP_PREFIXES) or bool(
        re.fullmatch(r"\d{2}-\d{2}\s+\S+", line)
    )


def _clean_line(value: str) -> str:
    return " ".join(value.strip().split())
