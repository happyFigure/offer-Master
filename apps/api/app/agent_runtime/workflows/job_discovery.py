from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol, TypedDict
import warnings

from app.domains.jobs.models import (
    JobLead,
    JobSource,
    JobSourceFetchMode,
    JobSourceType,
    SourceSyncRun,
    SourceSyncRunStatus,
)
from app.domains.jobs.providers.recruiting_signal import RuleBasedRecruitingSignalProvider
from app.domains.jobs.schemas import JobLeadCreate, RawJobLeadCreate, SourceSyncRunCreate
from app.domains.jobs.schemas import ArticleCandidateCreate
from app.domains.jobs.service import JobLeadService, RawJobLeadCaptureResult


@dataclass(frozen=True)
class ManualSocialLeadImportCommand:
    source_id: str
    raw_content: str
    source_url: str | None = None
    content_type: str = "text/plain"
    sync_run_id: str | None = None


@dataclass(frozen=True)
class ManualSocialLeadImportResult:
    raw_capture: RawJobLeadCaptureResult
    leads: list[JobLead]


@dataclass(frozen=True)
class UniversityCareerSyncCommand:
    source_id: str
    limit: int = 20


@dataclass(frozen=True)
class UniversityCareerSyncResult:
    sync_run: SourceSyncRun
    raw_captures: list[RawJobLeadCaptureResult]
    leads: list[JobLead]
    fetched_count: int
    extracted_count: int
    failed_count: int
    error: str | None = None


@dataclass(frozen=True)
class OfficialApiSyncCommand:
    source_id: str
    limit: int = 50


@dataclass(frozen=True)
class WeChatAccountSyncCommand:
    source_id: str
    limit: int = 20


@dataclass(frozen=True)
class WeChatAccountArticleEntry:
    title: str
    url: str
    source_account: str | None = None
    published_at: datetime | None = None
    raw_payload: dict | None = None


@dataclass(frozen=True)
class WeChatAccountSyncResult:
    sync_run: SourceSyncRun
    article_candidates: list
    recruiting_signals: list
    fetched_count: int
    extracted_count: int
    failed_count: int
    error: str | None = None


@dataclass(frozen=True)
class DueJobSourceSyncCommand:
    limit_per_source: int = 20
    now: datetime | None = None


@dataclass(frozen=True)
class DueJobSourceSyncResult:
    sync_runs: list[SourceSyncRun]
    processed_source_ids: list[str]
    succeeded_count: int
    failed_count: int
    skipped_count: int


class SocialLeadProvider(Protocol):
    def extract(
        self,
        source_id: str,
        raw_lead_id: str,
        raw_content: str,
        source_url: str | None,
        trust_level: object,
    ) -> list:
        ...


class UniversityCareerEntryLike(Protocol):
    title: str
    source_url: str
    raw_content: str
    raw_payload: dict | None


class UniversityCareerContentProvider(Protocol):
    def fetch(self, entry_url: str, limit: int) -> list[UniversityCareerEntryLike]:
        ...


class OfferIOOfficialApiProvider(Protocol):
    def list_company_openings(self, **kwargs):
        ...

    def list_companies(self, **kwargs):
        ...


class WeChatAccountArticleProvider(Protocol):
    def discover(self, source: JobSource, limit: int) -> list[WeChatAccountArticleEntry]:
        ...


class ManualSocialLeadImportState(TypedDict, total=False):
    command: ManualSocialLeadImportCommand
    result: ManualSocialLeadImportResult


class DueJobSourceSyncState(TypedDict, total=False):
    command: DueJobSourceSyncCommand
    result: DueJobSourceSyncResult


def run_manual_social_lead_import(
    command: ManualSocialLeadImportCommand,
    *,
    lead_service: JobLeadService,
    provider: SocialLeadProvider,
) -> ManualSocialLeadImportResult:
    source = lead_service.get_source(command.source_id)
    raw_capture = lead_service.capture_raw_lead(
        RawJobLeadCreate(
            source_id=command.source_id,
            sync_run_id=command.sync_run_id,
            source_url=command.source_url or source.entry_url,
            raw_content=command.raw_content,
            content_type=command.content_type,
        )
    )
    drafts = provider.extract(
        source_id=command.source_id,
        raw_lead_id=raw_capture.raw_lead.id,
        raw_content=command.raw_content,
        source_url=command.source_url or source.entry_url,
        trust_level=source.trust_level,
    )
    leads = [lead_service.create_lead(draft) for draft in drafts]
    lead_service.mark_raw_lead_extracted(raw_capture.raw_lead)
    return ManualSocialLeadImportResult(raw_capture=raw_capture, leads=leads)


def run_university_career_source_sync(
    command: UniversityCareerSyncCommand,
    *,
    lead_service: JobLeadService,
    content_provider: UniversityCareerContentProvider,
    social_provider: SocialLeadProvider,
) -> UniversityCareerSyncResult:
    source = lead_service.get_source(command.source_id)
    if not source.entry_url:
        raise ValueError(f"Job source has no entry_url: {command.source_id}")

    sync_run = lead_service.start_sync_run(SourceSyncRunCreate(source_id=command.source_id))
    raw_captures: list[RawJobLeadCaptureResult] = []
    leads: list[JobLead] = []
    failed_count = 0

    try:
        entries = content_provider.fetch(source.entry_url, command.limit)
    except Exception as exc:
        error = _format_source_sync_error(exc)
        failed_run = lead_service.finish_sync_run(
            sync_run,
            status=SourceSyncRunStatus.FAILED,
            fetched_count=0,
            extracted_count=0,
            failed_count=1,
            error=error,
        )
        return UniversityCareerSyncResult(
            sync_run=failed_run,
            raw_captures=[],
            leads=[],
            fetched_count=0,
            extracted_count=0,
            failed_count=1,
            error=error,
        )

    for entry in entries:
        raw_capture = lead_service.capture_raw_lead(
            RawJobLeadCreate(
                source_id=command.source_id,
                sync_run_id=sync_run.id,
                source_url=entry.source_url,
                raw_content=entry.raw_content,
                content_type="text/plain",
                raw_payload=entry.raw_payload,
            )
        )
        raw_captures.append(raw_capture)
        try:
            drafts = social_provider.extract(
                source_id=command.source_id,
                raw_lead_id=raw_capture.raw_lead.id,
                raw_content=entry.raw_content,
                source_url=entry.source_url,
                trust_level=source.trust_level,
            )
            leads.extend(lead_service.create_lead(draft) for draft in drafts)
            lead_service.mark_raw_lead_extracted(raw_capture.raw_lead)
        except Exception:
            failed_count += 1

    status = SourceSyncRunStatus.SUCCEEDED if failed_count == 0 else SourceSyncRunStatus.PARTIAL
    lead_service.finish_sync_run(
        sync_run,
        status=status,
        fetched_count=len(entries),
        extracted_count=len(leads),
        failed_count=failed_count,
        error=sync_run.error,
    )
    return UniversityCareerSyncResult(
        sync_run=sync_run,
        raw_captures=raw_captures,
        leads=leads,
        fetched_count=len(entries),
        extracted_count=len(leads),
        failed_count=failed_count,
    )


def run_offerio_official_api_source_sync(
    command: OfficialApiSyncCommand,
    *,
    lead_service: JobLeadService,
    provider: OfferIOOfficialApiProvider,
) -> UniversityCareerSyncResult:
    source = lead_service.get_source(command.source_id)
    if not source.entry_url:
        raise ValueError(f"Job source has no entry_url: {command.source_id}")

    sync_run = lead_service.start_sync_run(SourceSyncRunCreate(source_id=command.source_id))
    raw_captures: list[RawJobLeadCaptureResult] = []
    leads: list[JobLead] = []

    try:
        if "/api/recruitment/companies" in source.entry_url:
            page = provider.list_company_openings(page=1, page_size=command.limit)
            items = list(page.items)
            for item in items:
                raw_capture = lead_service.capture_raw_lead(
                    RawJobLeadCreate(
                        source_id=command.source_id,
                        sync_run_id=sync_run.id,
                        source_url=getattr(item, "apply_link", None) or source.entry_url,
                        raw_content=_offerio_opening_raw_content(item),
                        content_type="application/json",
                        raw_payload=getattr(item, "raw_payload", None),
                    )
                )
                raw_captures.append(raw_capture)
                lead = lead_service.create_lead(_offerio_opening_lead_create(source, raw_capture.raw_lead.id, item))
                leads.append(lead)
                lead_service.mark_raw_lead_extracted(raw_capture.raw_lead)
        elif "/api/recruitment/job-companies" in source.entry_url:
            page = provider.list_companies(job_type="校招", page=1, page_size=command.limit)
            items = list(page.items)
            for item in items:
                raw_capture = lead_service.capture_raw_lead(
                    RawJobLeadCreate(
                        source_id=command.source_id,
                        sync_run_id=sync_run.id,
                        source_url=source.entry_url,
                        raw_content=_offerio_company_raw_content(item),
                        content_type="application/json",
                        raw_payload=getattr(item, "raw_payload", None),
                    )
                )
                raw_captures.append(raw_capture)
                lead = lead_service.create_lead(_offerio_company_lead_create(source, raw_capture.raw_lead.id, item))
                leads.append(lead)
                lead_service.mark_raw_lead_extracted(raw_capture.raw_lead)
        else:
            raise ValueError(f"Unsupported OfferIO official API source URL: {source.entry_url}")
    except Exception as exc:
        error = _format_source_sync_error(exc)
        failed_run = lead_service.finish_sync_run(
            sync_run,
            status=SourceSyncRunStatus.FAILED,
            fetched_count=0,
            extracted_count=0,
            failed_count=1,
            error=error,
        )
        return UniversityCareerSyncResult(
            sync_run=failed_run,
            raw_captures=raw_captures,
            leads=leads,
            fetched_count=0,
            extracted_count=len(leads),
            failed_count=1,
            error=error,
        )

    lead_service.finish_sync_run(
        sync_run,
        status=SourceSyncRunStatus.SUCCEEDED,
        fetched_count=len(raw_captures),
        extracted_count=len(leads),
        failed_count=0,
        error=None,
    )
    return UniversityCareerSyncResult(
        sync_run=sync_run,
        raw_captures=raw_captures,
        leads=leads,
        fetched_count=len(raw_captures),
        extracted_count=len(leads),
        failed_count=0,
    )


def run_wechat_account_source_sync(
    command: WeChatAccountSyncCommand,
    *,
    lead_service: JobLeadService,
    article_provider: WeChatAccountArticleProvider,
) -> WeChatAccountSyncResult:
    source = lead_service.get_source(command.source_id)
    if _enum_value(source.source_type) != JobSourceType.WECHAT_ACCOUNT.value:
        raise ValueError(f"Job source is not a WeChat account source: {command.source_id}")

    sync_run = lead_service.start_sync_run(SourceSyncRunCreate(source_id=command.source_id))
    try:
        entries = article_provider.discover(source, command.limit)
    except Exception as exc:
        error = _format_source_sync_error(exc)
        failed_run = lead_service.finish_sync_run(
            sync_run,
            status=SourceSyncRunStatus.FAILED,
            fetched_count=0,
            extracted_count=0,
            failed_count=1,
            error=error,
        )
        return WeChatAccountSyncResult(
            sync_run=failed_run,
            article_candidates=[],
            recruiting_signals=[],
            fetched_count=0,
            extracted_count=0,
            failed_count=1,
            error=error,
        )

    candidates = []
    recruiting_signals = []
    failed_count = 0
    signal_provider = RuleBasedRecruitingSignalProvider()
    for entry in entries:
        try:
            capture = lead_service.capture_article_candidate(
                ArticleCandidateCreate(
                    source_id=command.source_id,
                    sync_run_id=sync_run.id,
                    title=entry.title,
                    url=entry.url,
                    source_account=entry.source_account or source.name,
                    published_at=entry.published_at,
                    raw_payload=entry.raw_payload,
                )
            )
            candidates.append(capture.candidate)
            signal_drafts = signal_provider.extract(
                source_id=command.source_id,
                raw_lead_id=None,
                raw_content=f"信息来源：公众号：{entry.source_account or source.name}\n{entry.title}",
                source_url=entry.url,
                trust_level=source.trust_level,
                source_context={"title": entry.title},
            )
            for draft in signal_drafts:
                signal_capture = lead_service.create_recruiting_signal(
                    draft.model_copy(
                        update={
                            "article_candidate_id": capture.candidate.id,
                            "original_source": entry.source_account or source.name,
                        }
                    )
                )
                recruiting_signals.append(signal_capture.signal)
        except Exception:
            failed_count += 1

    status = SourceSyncRunStatus.SUCCEEDED if failed_count == 0 else SourceSyncRunStatus.PARTIAL
    lead_service.finish_sync_run(
        sync_run,
        status=status,
        fetched_count=len(entries),
        extracted_count=0,
        failed_count=failed_count,
        error=None,
    )
    return WeChatAccountSyncResult(
        sync_run=sync_run,
        article_candidates=candidates,
        recruiting_signals=recruiting_signals,
        fetched_count=len(entries),
        extracted_count=0,
        failed_count=failed_count,
    )


def run_due_job_source_syncs(
    command: DueJobSourceSyncCommand,
    *,
    lead_service: JobLeadService,
    university_provider: UniversityCareerContentProvider,
    social_provider: SocialLeadProvider,
) -> DueJobSourceSyncResult:
    enabled_sources = lead_service.list_enabled_sources()
    due_sources = lead_service.list_due_sources(command.now)
    skipped_count = len(enabled_sources) - len(due_sources)
    sync_runs: list[SourceSyncRun] = []
    processed_source_ids: list[str] = []
    succeeded_count = 0
    failed_count = 0

    for source in due_sources:
        processed_source_ids.append(source.id)
        if _supports_public_university_sync(source):
            try:
                result = run_university_career_source_sync(
                    UniversityCareerSyncCommand(
                        source_id=source.id,
                        limit=command.limit_per_source,
                    ),
                    lead_service=lead_service,
                    content_provider=university_provider,
                    social_provider=social_provider,
                )
            except Exception:
                failed_count += 1
                continue

            sync_runs.append(result.sync_run)
            if result.sync_run.status == SourceSyncRunStatus.FAILED:
                failed_count += 1
            else:
                succeeded_count += 1
            continue

        sync_runs.append(_record_unsupported_sync_run(lead_service, source))
        failed_count += 1

    return DueJobSourceSyncResult(
        sync_runs=sync_runs,
        processed_source_ids=processed_source_ids,
        succeeded_count=succeeded_count,
        failed_count=failed_count,
        skipped_count=skipped_count,
    )


def _supports_public_university_sync(source: JobSource) -> bool:
    return (
        _enum_value(source.source_type) == JobSourceType.UNIVERSITY_CAREER_SITE.value
        and _enum_value(source.fetch_mode) == JobSourceFetchMode.PUBLIC_HTML.value
    )


def _offerio_opening_raw_content(item) -> str:
    return "\n".join(
        value
        for value in [
            f"公司：{getattr(item, 'company_name', '')}",
            f"企业性质：{getattr(item, 'company_nature', '') or '未披露'}",
            f"行业：{getattr(item, 'industry', '') or '未披露'}",
            f"批次：{getattr(item, 'batch', '') or '未披露'}",
            f"届别：{getattr(item, 'target', '') or '未披露'}",
            f"地点：{getattr(item, 'location', '') or '未披露'}",
            f"岗位方向：{getattr(item, 'positions', '') or '未披露'}",
            f"截止：{getattr(item, 'deadline', '') or '未披露'}",
            f"投递链接：{getattr(item, 'apply_link', '') or '未披露'}",
            f"笔试：{getattr(item, 'has_written_test', '') or '未披露'}",
        ]
        if value
    )


def _offerio_company_raw_content(item) -> str:
    return "\n".join(
        value
        for value in [
            f"公司：{getattr(item, 'name', '')}",
            f"企业性质：{getattr(item, 'company_nature', '') or '未披露'}",
            f"行业：{getattr(item, 'industry', '') or '未披露'}",
            f"地点：{getattr(item, 'locations', '') or '未披露'}",
            f"岗位数：{getattr(item, 'job_count', 0)}",
            f"更新时间：{getattr(item, 'updated_at', '') or '未披露'}",
        ]
        if value
    )


def _offerio_opening_lead_create(source: JobSource, raw_lead_id: str, item) -> JobLeadCreate:
    positions = getattr(item, "positions", None)
    company_name = getattr(item, "company_name")
    return JobLeadCreate(
        source_id=source.id,
        raw_lead_id=raw_lead_id,
        company_name=_limit_text(company_name, 255) or company_name,
        title=_limit_text(positions or f"{company_name} 校招开放", 255) or f"{company_name} 校招开放",
        city=_limit_text(getattr(item, "location", None), 128),
        job_direction=_limit_text(positions, 128),
        graduation_year=_limit_text(getattr(item, "target", None), 32),
        source_url=source.entry_url,
        apply_url=getattr(item, "apply_link", None),
        job_type=_limit_text(getattr(item, "batch", None), 128),
        jd_text=_offerio_opening_raw_content(item),
        skills=[_limit_text(positions, 255)] if positions else [],
        deadline=_parse_offerio_deadline(getattr(item, "deadline", None)),
        confidence_score=82.0,
        trust_level=source.trust_level,
        raw_payload=getattr(item, "raw_payload", None),
    )


def _offerio_company_lead_create(source: JobSource, raw_lead_id: str, item) -> JobLeadCreate:
    job_count = getattr(item, "job_count", 0)
    name = getattr(item, "name")
    return JobLeadCreate(
        source_id=source.id,
        raw_lead_id=raw_lead_id,
        company_name=_limit_text(name, 255) or name,
        title=_limit_text(f"{name} 校招岗位聚合（{job_count} 个）", 255) or f"{name} 校招岗位聚合",
        city=_limit_text(getattr(item, "locations", None), 128),
        job_direction=_limit_text(getattr(item, "industry", None), 128),
        graduation_year=None,
        source_url=source.entry_url,
        apply_url=None,
        job_type="校招",
        jd_text=_offerio_company_raw_content(item),
        skills=[_limit_text(getattr(item, "industry"), 255)] if getattr(item, "industry", None) else [],
        confidence_score=68.0,
        trust_level=source.trust_level,
        raw_payload=getattr(item, "raw_payload", None),
    )


def _parse_offerio_deadline(value: str | None) -> date | None:
    if not value:
        return None
    normalized = value.strip().replace("/", "-")
    try:
        return date.fromisoformat(normalized[:10])
    except ValueError:
        return None


def _limit_text(value: object, limit: int) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _record_unsupported_sync_run(
    lead_service: JobLeadService,
    source: JobSource,
) -> SourceSyncRun:
    sync_run = lead_service.start_sync_run(
        SourceSyncRunCreate(
            source_id=source.id,
            run_metadata={"reason": "unsupported_automated_sync"},
        )
    )
    return lead_service.finish_sync_run(
        sync_run,
        status=SourceSyncRunStatus.FAILED,
        fetched_count=0,
        extracted_count=0,
        failed_count=1,
        error=(
            "No automated sync provider for "
            f"source_type={_enum_value(source.source_type)}, "
            f"fetch_mode={_enum_value(source.fetch_mode)}"
        ),
    )


def _format_source_sync_error(exc: Exception) -> str:
    message = str(exc)
    lowered = message.lower()
    if "mp.weixin.qq.com" in lowered and ("captcha" in lowered or "appmsgcaptcha" in lowered or "302" in lowered):
        return (
            "微信公众号链接被重定向到验证码页，后台公开 HTML 同步无法读取正文。"
            "请把来源类型改为“微信公众号”，然后到“岗位线索”的“粘贴链接并解析”入口处理；"
            "如果仍被限制，请使用手动粘贴正文兜底。"
        )
    if "captcha" in lowered or "access restricted" in lowered:
        return "来源页面访问受限或需要验证码，请改用用户可见页面/MCP 边界或手动粘贴正文。"
    if "timeout" in lowered or "timed out" in lowered:
        return "来源页面请求超时，请稍后重试或检查入口 URL 是否可访问。"
    return message[:500]


def _enum_value(value: object) -> object:
    return getattr(value, "value", value)


def build_manual_social_lead_import_graph(
    *,
    lead_service: JobLeadService,
    provider: SocialLeadProvider,
):
    END, START, StateGraph = _load_langgraph_graph()

    def import_social_leads_node(
        state: ManualSocialLeadImportState,
    ) -> ManualSocialLeadImportState:
        return {
            "result": run_manual_social_lead_import(
                state["command"],
                lead_service=lead_service,
                provider=provider,
            )
        }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = StateGraph(ManualSocialLeadImportState)
        graph.add_node("import_social_leads", import_social_leads_node)
        graph.add_edge(START, "import_social_leads")
        graph.add_edge("import_social_leads", END)
        return graph.compile()


def build_due_job_source_sync_graph(
    *,
    lead_service: JobLeadService,
    university_provider: UniversityCareerContentProvider,
    social_provider: SocialLeadProvider,
):
    END, START, StateGraph = _load_langgraph_graph()

    def sync_due_sources_node(state: DueJobSourceSyncState) -> DueJobSourceSyncState:
        return {
            "result": run_due_job_source_syncs(
                state["command"],
                lead_service=lead_service,
                university_provider=university_provider,
                social_provider=social_provider,
            )
        }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = StateGraph(DueJobSourceSyncState)
        graph.add_node("sync_due_sources", sync_due_sources_node)
        graph.add_edge(START, "sync_due_sources")
        graph.add_edge("sync_due_sources", END)
        return graph.compile()


def _load_langgraph_graph():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from langgraph.graph import END, START, StateGraph

    return END, START, StateGraph
