from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypedDict

from app.domains.jobs.models import JobLead, SourceSyncRun, SourceSyncRunStatus
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


def build_manual_social_lead_import_graph(
    *,
    lead_service: JobLeadService,
    provider: SocialLeadProvider,
):
    from langgraph.graph import END, START, StateGraph

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

    graph = StateGraph(ManualSocialLeadImportState)
    graph.add_node("import_social_leads", import_social_leads_node)
    graph.add_edge(START, "import_social_leads")
    graph.add_edge("import_social_leads", END)
    return graph.compile()
