from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypedDict

from app.domains.jobs.models import JobLead
from app.domains.jobs.schemas import RawJobLeadCreate
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
