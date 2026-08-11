from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
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
from app.domains.jobs.schemas import RawJobLeadCreate, SourceSyncRunCreate
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
        lead_service.finish_sync_run(
            sync_run,
            status=SourceSyncRunStatus.FAILED,
            fetched_count=0,
            extracted_count=0,
            failed_count=1,
            error=str(exc),
        )
        raise

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
    )
    return UniversityCareerSyncResult(
        sync_run=sync_run,
        raw_captures=raw_captures,
        leads=leads,
        fetched_count=len(entries),
        extracted_count=len(leads),
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
