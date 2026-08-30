from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.agent_runtime.tool_registry import (
    EXTERNAL_WEB_SEARCH_TOOL,
    LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL,
    LOCAL_JOB_SOURCE_OVERVIEW_TOOL,
)


@dataclass(frozen=True)
class ToolCandidateSelection:
    capabilities: tuple[str, ...] = ()
    signals: tuple[str, ...] = ()
    reasons: dict[str, str] = field(default_factory=dict)

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "capabilities": list(self.capabilities),
            "signals": list(self.signals),
            "reasons": dict(self.reasons),
        }


class ToolCandidateSelector:
    """Build a small candidate tool list from broad task signals and tool declarations."""

    def __init__(self, registry: Any) -> None:
        self._registry = registry

    def select(
        self,
        user_message: str,
        *,
        source_type: str = "agent_chat",
        auto_executable_only: bool = True,
    ) -> ToolCandidateSelection:
        signals = _detect_task_signals(user_message)
        if not signals:
            return ToolCandidateSelection()

        capabilities: list[str] = []
        reasons: dict[str, str] = {}
        signal_set = set(signals)
        for definition in self._registry.list_definitions():
            if not _tool_is_candidate_eligible(definition, source_type=source_type, auto_executable_only=auto_executable_only):
                continue
            categories = _candidate_categories(definition)
            if not categories:
                continue
            matched = sorted(signal_set.intersection(categories))
            if not matched:
                continue
            capability_id = _definition_id(definition)
            capabilities.append(capability_id)
            reasons[capability_id] = f"matched task signals: {', '.join(matched)}"

        return ToolCandidateSelection(capabilities=tuple(capabilities), signals=signals, reasons=reasons)


def _tool_is_candidate_eligible(
    definition: Any,
    *,
    source_type: str,
    auto_executable_only: bool,
) -> bool:
    if not bool(getattr(definition, "enabled", True)):
        return False
    allowed_source_types = frozenset(getattr(definition, "allowed_source_types", frozenset()) or frozenset())
    if allowed_source_types and source_type not in allowed_source_types:
        return False
    if not auto_executable_only:
        return True
    risk_level = str(getattr(definition.risk_level, "value", definition.risk_level))
    return risk_level == "low" and not definition.requires_confirmation


def _candidate_categories(definition: Any) -> frozenset[str]:
    candidate_profile = getattr(definition, "candidate_profile", None)
    if candidate_profile is not None:
        return frozenset(getattr(candidate_profile, "categories", frozenset()) or frozenset())
    capability_id = _definition_id(definition)
    if capability_id == EXTERNAL_WEB_SEARCH_TOOL:
        return frozenset({"public_web_information", "realtime_public_information"})
    if capability_id == LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL:
        return frozenset({"local_company_data", "local_database"})
    if capability_id == LOCAL_JOB_SOURCE_OVERVIEW_TOOL:
        return frozenset({"local_job_source_data", "local_database"})
    return frozenset()


def _definition_id(definition: Any) -> str:
    return str(getattr(definition, "capability_id", None) or getattr(definition, "name", ""))


def _detect_task_signals(user_message: str) -> tuple[str, ...]:
    text = user_message.strip()
    if not text:
        return ()

    signals: list[str] = []
    file_signals = _filesystem_task_signals(text)
    if file_signals:
        return file_signals
    if _looks_like_local_company_profile_request(text):
        signals.append("local_company_profile")
    elif _looks_like_local_job_search_request(text):
        signals.append("local_job_search")
    elif _looks_like_local_job_source_data_request(text):
        signals.append("local_job_source_data")
    elif _looks_like_local_company_data_request(text):
        signals.append("local_company_data")
    if signals:
        if _looks_like_external_enrichment_request(text):
            signals.append("public_web_information")
        return tuple(signals)
    if _looks_like_wechat_article_read_request(text):
        signals.append("wechat_article_read")
        signals.append("content_source_read")
        return tuple(signals)
    if _looks_like_xiaohongshu_detail_request(text):
        signals.append("xiaohongshu_content_detail")
        signals.append("content_source_read")
        return tuple(signals)
    if _looks_like_xiaohongshu_search_request(text):
        signals.append("xiaohongshu_content_search")
        signals.append("content_source_search")
        return tuple(signals)
    if _looks_like_resume_tailoring_request(text):
        signals.append("resume_tailoring")
        signals.append("content_processing")
        return tuple(signals)
    if _looks_like_realtime_public_information_request(text):
        signals.append("realtime_public_information")
    if _looks_like_public_web_information_request(text):
        signals.append("public_web_information")
    return tuple(signals)


def _filesystem_task_signals(text: str) -> tuple[str, ...]:
    if not _looks_like_filesystem_request(text):
        return ()

    signals: list[str] = []
    if any(marker in text for marker in ("删除", "删掉", "移除", "delete", "remove")):
        signals.append("filesystem_delete")
    if any(marker in text for marker in ("复制", "拷贝", "备份", "copy")):
        signals.append("filesystem_copy")
    if any(marker in text for marker in ("移动", "重命名", "改名", "move", "rename")):
        signals.append("filesystem_move")
    if any(marker in text for marker in ("创建目录", "新建目录", "新建文件夹", "mkdir", "make dir")):
        signals.append("filesystem_make_dir")
    if any(marker in text for marker in ("写入", "保存", "修改", "替换", "换成", "换为", "换了", "改成", "改为", "覆盖", "write")):
        if any(marker in text for marker in ("替换", "换成", "换为", "换了", "改成", "改为", "只改", "其他不要动", "其他的啥都不要动")):
            signals.append("filesystem_replace")
        signals.append("filesystem_write")
        if any(marker in text for marker in ("替换", "换成", "换为", "换了", "改成", "改为", "其他不要动", "其他的啥都不要动")):
            signals.append("filesystem_read")
    if any(marker in text for marker in ("列出", "查看目录", "目录下", "文件夹", "有哪些文件", "list")):
        signals.append("filesystem_list")
    if any(marker in text for marker in ("是否存在", "存不存在", "有没有这个文件", "文件大小", "修改时间", "stat", "exists")):
        signals.append("filesystem_stat")
    if any(marker in text for marker in ("读取", "读一下", "读文件", "读到", "能不能读", "能读", "打开", "查看", "看一下", "read")):
        signals.append("filesystem_read")

    if not signals:
        signals.append("filesystem_operation")
    return tuple(_dedupe(signals))


def _looks_like_filesystem_request(text: str) -> bool:
    lowered = text.lower()
    path_markers = (
        ":/",
        ":\\",
        "\\",
        "/",
        ".tex",
        ".md",
        ".txt",
        ".pdf",
        ".docx",
        ".json",
        ".csv",
        "本地文件",
        "文件路径",
        "目录",
        "文件夹",
    )
    action_markers = (
        "读取",
        "读一下",
        "读到",
        "能不能读",
        "能读",
        "打开",
        "查看",
        "列出",
        "修改",
        "替换",
        "换成",
        "换为",
        "换了",
        "改成",
        "改为",
        "写入",
        "保存",
        "复制",
        "移动",
        "删除",
        "重命名",
        "创建目录",
        "新建文件夹",
        "read",
        "write",
        "copy",
        "move",
        "delete",
        "rename",
        "mkdir",
    )
    return any(marker in lowered for marker in path_markers) and any(marker in lowered for marker in action_markers)


def _looks_like_local_job_source_data_request(text: str) -> bool:
    source_markers = ("岗位来源", "来源库", "岗位展览", "校招来源", "招聘来源", "岗位线索")
    query_markers = ("多少", "几个", "哪些", "列表", "看一下", "给我", "统计", "概览", "现在")
    return any(marker in text for marker in source_markers) and any(marker in text for marker in query_markers)


def _looks_like_local_company_data_request(text: str) -> bool:
    local_markers = ("数据库", "本地", "我的", "已有", "投递板", "岗位展览", "公司库", "企业库")
    company_markers = ("公司", "企业", "厂", "岗位线索", "校招来源")
    return any(marker in text for marker in local_markers) and any(marker in text for marker in company_markers)


def _looks_like_local_company_profile_request(text: str) -> bool:
    if not _looks_like_local_company_data_request(text):
        return False
    detail_markers = ("关于", "详情", "详细", "信息", "档案", "介绍", "主营业务", "主要业务")
    return any(marker in text for marker in detail_markers)


def _looks_like_local_job_search_request(text: str) -> bool:
    local_markers = ("数据库", "本地", "我的", "已有", "岗位库", "岗位展览")
    job_markers = ("岗位", "职位", "招聘", "jd", "Java", "Python")
    query_markers = ("查", "搜", "搜索", "看", "有哪些", "列出", "找")
    return (
        any(marker in text for marker in local_markers)
        and any(marker in text for marker in job_markers)
        and any(marker in text.lower() for marker in query_markers)
    )


def _looks_like_resume_tailoring_request(text: str) -> bool:
    resume_markers = ("简历", "resume", "履历")
    action_markers = ("优化", "修改", "改", "润色", "匹配", "更适合", "突出", "调整")
    return any(marker in text for marker in resume_markers) and any(marker in text for marker in action_markers)


def _looks_like_wechat_article_read_request(text: str) -> bool:
    return "mp.weixin.qq.com" in text or any(marker in text for marker in ("微信公众号", "微信文章", "公众号文章"))


def _looks_like_xiaohongshu_detail_request(text: str) -> bool:
    return "feed_id" in text and "xsec_token" in text


def _looks_like_xiaohongshu_search_request(text: str) -> bool:
    source_markers = ("小红书", "红书", "xiaohongshu", "xhslink")
    search_markers = ("搜索", "搜", "找", "查", "看看")
    return any(marker in text for marker in source_markers) and any(marker in text for marker in search_markers)


def _looks_like_realtime_public_information_request(text: str) -> bool:
    realtime_markers = (
        "今天",
        "现在",
        "最新",
        "最近",
        "刚刚",
        "明天",
        "本周",
        "这周",
        "这个星期",
        "这星期",
        "本星期",
        "这个礼拜",
        "本礼拜",
        "今晚",
        "目前",
    )
    public_objects = ("比赛", "赛程", "结果", "新闻", "股价", "价格", "天气", "官网", "招聘", "校招", "秋招", "开放")
    query_markers = ("查", "搜", "搜索", "看一下", "有没有", "有什么", "有啥", "多少", "哪里", "什么时候", "安排", "日程")
    return any(marker in text for marker in realtime_markers) and (
        any(marker in text for marker in public_objects) or any(marker in text for marker in query_markers)
    )


def _looks_like_public_web_information_request(text: str) -> bool:
    if _looks_like_local_company_data_request(text):
        return _looks_like_external_enrichment_request(text)
    public_markers = (
        "搜",
        "搜索",
        "查一下",
        "查询",
        "是什么",
        "做什么",
        "主要业务",
        "介绍一下",
        "了解一下",
        "官网",
        "公开信息",
        "新闻",
    )
    if not any(marker in text for marker in public_markers):
        return False
    return True


def _looks_like_external_enrichment_request(text: str) -> bool:
    enrichment_markers = (
        "补充公开",
        "公开资料",
        "公开信息",
        "外部资料",
        "外部信息",
        "联网补充",
        "主营业务",
        "主要业务",
        "官网信息",
        "招聘动态",
    )
    return any(marker in text for marker in enrichment_markers)


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


__all__ = ["ToolCandidateSelection", "ToolCandidateSelector"]
