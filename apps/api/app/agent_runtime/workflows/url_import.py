from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypedDict
import time
import warnings

from sqlalchemy.orm import Session

from app.domains.automation.models import ToolCallStatus, WorkflowRun, WorkflowRunStatus, utc_now
from app.domains.automation.repository import (
    ApprovalRequestRepository,
    ToolCallLogRepository,
    WorkflowCheckpointRepository,
    WorkflowRunRepository,
)
from app.domains.automation.schemas import ToolCallLogCreate, WorkflowCheckpointCreate, WorkflowRunCreate
from app.domains.automation.service import AutomationService
from app.domains.jobs.content_fetcher import HTTPArticleFetcher
from app.domains.jobs.js_render_fetcher import Crawl4AIFetcher, PlaywrightFetcher
from app.domains.jobs.models import (
    JobLead,
    JobLeadStatus,
    JobSourceFetchMode,
    JobSourceTrustLevel,
    JobSourceType,
    RawJobLead,
    RecruitingSignal,
    UrlImportRun,
    UrlImportRunStatus,
)
from app.domains.jobs.providers.recruiting_signal import RuleBasedRecruitingSignalProvider
from app.domains.jobs.repository import (
    ArticleCandidateRepository,
    DomainHealthRepository,
    JobLeadRepository,
    JobSourceRepository,
    RawJobLeadRepository,
    RecruitingSignalRepository,
    SourceSyncRunRepository,
    UrlImportRunRepository,
)
from app.domains.jobs.schemas import (
    JobLeadCreate,
    JobSourceCreate,
    RawJobLeadCreate,
    ToolErrorCode,
    ToolResult,
    ToolSuggestedNextAction,
)
from app.domains.jobs.service import JobLeadService, RawJobLeadCaptureResult
from app.domains.jobs.tool_error_details import exception_error_details
from app.domains.jobs.tool_guard import ToolCallContext
from app.domains.jobs.url_import import UrlImportAnalysis, analyze_import_url
from app.domains.jobs.wechat_fetcher import WeChatArticleFetcher
from app.domains.jobs.xiaohongshu_visible_page import parse_xiaohongshu_visible_text


@dataclass(frozen=True)
class UrlImportCommand:
    url: str
    source_id: str | None = None
    source_hint: JobSourceType | None = None
    trust_level: JobSourceTrustLevel | None = None
    force_refresh: bool = False


@dataclass(frozen=True)
class UrlImportWorkflowResult:
    workflow_run: WorkflowRun
    url_import_run: UrlImportRun
    raw_capture: RawJobLeadCaptureResult | None = None
    leads: list[JobLead] | None = None
    recruiting_signals: list[RecruitingSignal] | None = None


class SocialLeadProvider(Protocol):
    def extract(
        self,
        source_id: str,
        raw_lead_id: str,
        raw_content: str,
        source_url: str | None,
        trust_level: JobSourceTrustLevel,
    ) -> list[JobLeadCreate]: ...


class Fetcher(Protocol):
    tool_name: str

    def fetch(self, url: str, context: ToolCallContext, *, domain_health=None) -> ToolResult: ...


class RecruitingSignalProvider(Protocol):
    def extract(
        self,
        *,
        source_id: str,
        raw_lead_id: str | None,
        raw_content: str,
        source_url: str | None,
        trust_level: object,
        source_context: dict[str, Any] | None = None,
    ) -> list: ...


@dataclass(frozen=True)
class UrlImportWorkflowDependencies:
    workflow_runs: WorkflowRunRepository
    automation_service: AutomationService
    lead_service: JobLeadService
    url_import_runs: UrlImportRunRepository
    raw_leads: RawJobLeadRepository
    domain_health: DomainHealthRepository
    social_provider: SocialLeadProvider
    recruiting_signal_provider: RecruitingSignalProvider
    fetchers: dict[str, Fetcher]


class UrlImportWorkflowState(TypedDict, total=False):
    command: UrlImportCommand
    result: UrlImportWorkflowResult


def build_url_import_dependencies(
    session: Session,
    *,
    fetchers: dict[str, Fetcher] | None = None,
    social_provider: SocialLeadProvider,
    recruiting_signal_provider: RecruitingSignalProvider | None = None,
) -> UrlImportWorkflowDependencies:
    workflow_runs = WorkflowRunRepository(session)
    checkpoints = WorkflowCheckpointRepository(session)
    tool_call_logs = ToolCallLogRepository(session)
    approvals = ApprovalRequestRepository(session)
    raw_leads = RawJobLeadRepository(session)
    return UrlImportWorkflowDependencies(
        workflow_runs=workflow_runs,
        automation_service=AutomationService(
            workflow_runs=workflow_runs,
            checkpoints=checkpoints,
            tool_call_logs=tool_call_logs,
            approvals=approvals,
        ),
        lead_service=JobLeadService(
            sources=JobSourceRepository(session),
            sync_runs=SourceSyncRunRepository(session),
            raw_leads=raw_leads,
            leads=JobLeadRepository(session),
            article_candidates=ArticleCandidateRepository(session),
            recruiting_signals=RecruitingSignalRepository(session),
        ),
        url_import_runs=UrlImportRunRepository(session),
        raw_leads=raw_leads,
        domain_health=DomainHealthRepository(session),
        social_provider=social_provider,
        recruiting_signal_provider=recruiting_signal_provider or RuleBasedRecruitingSignalProvider(),
        fetchers=fetchers or _default_fetchers(),
    )


def run_url_import_workflow(
    command: UrlImportCommand,
    *,
    dependencies: UrlImportWorkflowDependencies,
) -> UrlImportWorkflowResult:
    created = create_url_import_run(command, dependencies=dependencies)
    if created.url_import_run.status != UrlImportRunStatus.RUNNING:
        return created
    return continue_url_import_workflow(created.url_import_run.id, dependencies=dependencies)


def create_url_import_run(
    command: UrlImportCommand,
    *,
    dependencies: UrlImportWorkflowDependencies,
) -> UrlImportWorkflowResult:
    workflow = dependencies.automation_service.start_workflow(
        WorkflowRunCreate(
            workflow_type="url_import",
            current_step="normalize_url",
            user_goal=f"Import recruiting leads from URL: {command.url}",
        )
    )
    analysis = analyze_import_url(command.url, source_hint=command.source_hint)
    source_id = command.source_id or _ensure_import_source(command, analysis, dependencies).id
    existing = (
        None
        if command.force_refresh
        else dependencies.url_import_runs.get_by_normalized_url_hash(analysis.normalized_url_hash)
    )
    url_run = dependencies.url_import_runs.add(
        UrlImportRun(
            workflow_run_id=workflow.id,
            source_id=source_id,
            input_url=command.url,
            normalized_url=analysis.normalized_url,
            normalized_url_hash=analysis.normalized_url_hash,
            source_type=analysis.source_type,
            domain=analysis.domain,
            fetch_layer=analysis.fetch_layer,
            status=UrlImportRunStatus.RUNNING,
            current_stage="normalize_url",
            run_metadata={
                "fetch_mode": analysis.fetch_mode.value,
                "requires_mcp_visible_page": analysis.requires_mcp_visible_page,
                "requires_user_confirmation": analysis.requires_user_confirmation,
            },
        )
    )
    _checkpoint(dependencies, workflow, url_run, "normalize_url", _state_for_checkpoint(command, analysis, url_run))

    if existing is not None:
        url_run.status = UrlImportRunStatus.DUPLICATE
        url_run.current_stage = "duplicate_url"
        url_run.duplicate_of_run_id = existing.id
        url_run.next_action = ToolSuggestedNextAction.SKIP_DUPLICATE.value
        url_run.finished_at = utc_now()
        workflow.status = WorkflowRunStatus.COMPLETED
        workflow.current_step = "duplicate_url"
        workflow.completed_at = url_run.finished_at
        _checkpoint(dependencies, workflow, url_run, "duplicate_url", _state_for_checkpoint(command, analysis, url_run))
        dependencies.url_import_runs.flush()
        return UrlImportWorkflowResult(workflow_run=workflow, url_import_run=url_run, leads=[])

    url_run.current_stage = "classify_source"
    workflow.current_step = "classify_source"
    _checkpoint(dependencies, workflow, url_run, "classify_source", _state_for_checkpoint(command, analysis, url_run))
    if analysis.requires_mcp_visible_page:
        _mark_waiting_user(
            workflow,
            url_run,
            ToolErrorCode.REQUIRES_MCP_VISIBLE_PAGE.value,
            "URL source requires user-visible page access",
            ToolSuggestedNextAction.REQUEST_USER_VISIBLE_PAGE.value,
        )
        _checkpoint(dependencies, workflow, url_run, "request_user_visible_page", _state_for_checkpoint(command, analysis, url_run))
        dependencies.url_import_runs.flush()
        return UrlImportWorkflowResult(workflow_run=workflow, url_import_run=url_run, leads=[])

    url_run.current_stage = "queued"
    workflow.current_step = "queued"
    _checkpoint(dependencies, workflow, url_run, "queued", _state_for_checkpoint(command, analysis, url_run))
    dependencies.url_import_runs.flush()
    return UrlImportWorkflowResult(workflow_run=workflow, url_import_run=url_run, leads=[])


def continue_url_import_workflow(
    url_import_run_id: str,
    *,
    dependencies: UrlImportWorkflowDependencies,
) -> UrlImportWorkflowResult:
    url_run = dependencies.url_import_runs.get(url_import_run_id)
    if url_run is None:
        raise ValueError(f"URL import run not found: {url_import_run_id}")
    workflow = dependencies.workflow_runs.get(url_run.workflow_run_id)
    if workflow is None:
        raise ValueError(f"Workflow run not found: {url_run.workflow_run_id}")
    if url_run.status != UrlImportRunStatus.RUNNING:
        return UrlImportWorkflowResult(workflow_run=workflow, url_import_run=url_run, leads=[])

    command = UrlImportCommand(
        url=url_run.input_url,
        source_id=url_run.source_id,
        source_hint=_source_hint_from_url_run(url_run),
    )
    analysis = analyze_import_url(url_run.input_url, source_hint=command.source_hint)

    if url_run.raw_job_lead_id is not None:
        raw_lead = dependencies.raw_leads.get(url_run.raw_job_lead_id)
        if raw_lead is None:
            raise ValueError(f"Raw job lead not found: {url_run.raw_job_lead_id}")
        extraction = _extract_job_leads_from_raw(
            command=command,
            analysis=analysis,
            workflow=workflow,
            url_run=url_run,
            raw_lead=raw_lead,
            dependencies=dependencies,
        )
        return UrlImportWorkflowResult(
            workflow_run=workflow,
            url_import_run=url_run,
            raw_capture=None,
            leads=extraction.leads if extraction.ok else [],
            recruiting_signals=extraction.recruiting_signals,
        )

    fetch_result = _run_fetch_stage(command, analysis, workflow, url_run, dependencies)
    if not fetch_result.ok:
        _apply_failed_tool_result(workflow, url_run, fetch_result)
        _checkpoint(dependencies, workflow, url_run, url_run.current_stage, _state_for_checkpoint(command, analysis, url_run, tool_result=fetch_result))
        dependencies.url_import_runs.flush()
        return UrlImportWorkflowResult(workflow_run=workflow, url_import_run=url_run, leads=[])

    raw_capture = _save_raw_lead(command, analysis, url_run, fetch_result, dependencies)
    _checkpoint(
        dependencies,
        workflow,
        url_run,
        "save_raw_job_lead",
        _state_for_checkpoint(command, analysis, url_run, raw_lead_id=raw_capture.raw_lead.id, tool_result=fetch_result),
    )
    extraction = _extract_job_leads_from_raw(
        command=command,
        analysis=analysis,
        workflow=workflow,
        url_run=url_run,
        raw_lead=raw_capture.raw_lead,
        dependencies=dependencies,
    )
    if extraction.ok:
        return UrlImportWorkflowResult(
            workflow_run=workflow,
            url_import_run=url_run,
            raw_capture=raw_capture,
            leads=extraction.leads,
            recruiting_signals=extraction.recruiting_signals,
        )
    return UrlImportWorkflowResult(
        workflow_run=workflow,
        url_import_run=url_run,
        raw_capture=raw_capture,
        leads=[],
        recruiting_signals=extraction.recruiting_signals,
    )


def continue_url_import_with_visible_page_content(
    url_import_run_id: str,
    *,
    visible_text: str,
    title: str | None,
    final_url: str | None,
    dependencies: UrlImportWorkflowDependencies,
) -> UrlImportWorkflowResult:
    url_run = dependencies.url_import_runs.get(url_import_run_id)
    if url_run is None:
        raise ValueError(f"URL import run not found: {url_import_run_id}")
    workflow = dependencies.workflow_runs.get(url_run.workflow_run_id)
    if workflow is None:
        raise ValueError(f"Workflow run not found: {url_run.workflow_run_id}")
    if url_run.status != UrlImportRunStatus.WAITING_USER:
        raise ValueError("URL import run is not waiting for visible page content")

    command = UrlImportCommand(
        url=url_run.input_url,
        source_id=url_run.source_id,
        source_hint=_source_hint_from_url_run(url_run),
    )
    analysis = analyze_import_url(url_run.input_url, source_hint=command.source_hint)
    raw_capture = _save_visible_page_raw_lead(
        command=command,
        analysis=analysis,
        workflow=workflow,
        url_run=url_run,
        visible_text=visible_text,
        title=title,
        final_url=final_url,
        dependencies=dependencies,
    )
    _checkpoint(
        dependencies,
        workflow,
        url_run,
        "save_visible_page_raw_job_lead",
        _state_for_checkpoint(command, analysis, url_run, raw_lead_id=raw_capture.raw_lead.id),
    )
    extraction = _extract_job_leads_from_raw(
        command=command,
        analysis=analysis,
        workflow=workflow,
        url_run=url_run,
        raw_lead=raw_capture.raw_lead,
        dependencies=dependencies,
    )
    return UrlImportWorkflowResult(
        workflow_run=workflow,
        url_import_run=url_run,
        raw_capture=raw_capture,
        leads=extraction.leads if extraction.ok else [],
        recruiting_signals=extraction.recruiting_signals,
    )


def resume_url_import_workflow(
    url_import_run_id: str,
    *,
    dependencies: UrlImportWorkflowDependencies,
) -> UrlImportWorkflowResult:
    url_run = dependencies.url_import_runs.get(url_import_run_id)
    if url_run is None:
        raise ValueError(f"URL import run not found: {url_import_run_id}")
    if url_run.raw_job_lead_id is None:
        raise ValueError("URL import run has no raw lead checkpoint to resume from")
    raw_lead = dependencies.raw_leads.get(url_run.raw_job_lead_id)
    if raw_lead is None:
        raise ValueError(f"Raw job lead not found: {url_run.raw_job_lead_id}")
    workflow = dependencies.workflow_runs.get(url_run.workflow_run_id)
    if workflow is None:
        raise ValueError(f"Workflow run not found: {url_run.workflow_run_id}")

    workflow.status = WorkflowRunStatus.RUNNING
    workflow.current_step = "resume_extract_job_leads"
    workflow.error = None
    url_run.status = UrlImportRunStatus.RUNNING
    url_run.current_stage = "resume_extract_job_leads"
    url_run.error_code = None
    url_run.error_message = None
    url_run.next_action = None
    analysis = analyze_import_url(url_run.input_url)
    _checkpoint(
        dependencies,
        workflow,
        url_run,
        "resume_from_raw_job_lead",
        {
            "url_import_run_id": url_run.id,
            "workflow_run_id": workflow.id,
            "raw_job_lead_id": raw_lead.id,
            "current_stage": url_run.current_stage,
        },
    )
    extraction = _extract_job_leads_from_raw(
        command=UrlImportCommand(url=url_run.input_url, source_id=url_run.source_id),
        analysis=analysis,
        workflow=workflow,
        url_run=url_run,
        raw_lead=raw_lead,
        dependencies=dependencies,
    )
    return UrlImportWorkflowResult(
        workflow_run=workflow,
        url_import_run=url_run,
        raw_capture=None,
        leads=extraction.leads if extraction.ok else [],
        recruiting_signals=extraction.recruiting_signals,
    )


def build_url_import_graph(*, dependencies: UrlImportWorkflowDependencies):
    END, START, StateGraph = _load_langgraph_graph()

    def import_url_node(state: UrlImportWorkflowState) -> UrlImportWorkflowState:
        return {
            "result": run_url_import_workflow(
                state["command"],
                dependencies=dependencies,
            )
        }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = StateGraph(UrlImportWorkflowState)
        graph.add_node("import_url", import_url_node)
        graph.add_edge(START, "import_url")
        graph.add_edge("import_url", END)
        return graph.compile()


@dataclass(frozen=True)
class _ExtractionResult:
    ok: bool
    leads: list[JobLead]
    recruiting_signals: list[RecruitingSignal]


def _run_fetch_stage(
    command: UrlImportCommand,
    analysis: UrlImportAnalysis,
    workflow: WorkflowRun,
    url_run: UrlImportRun,
    dependencies: UrlImportWorkflowDependencies,
) -> ToolResult:
    fetcher = _select_fetcher(analysis, dependencies)
    stage = _stage_for_fetcher(fetcher.tool_name)
    url_run.current_stage = stage
    workflow.current_step = stage
    context = ToolCallContext(
        stage=stage,
        tool_name=fetcher.tool_name,
        source_type=analysis.source_type,
        domain=analysis.domain,
        run_id=url_run.id,
        tool_call_count=url_run.tool_call_count,
        llm_call_count=url_run.llm_call_count,
        fetch_attempts_for_stage=url_run.attempt_count,
    )
    domain_health = dependencies.domain_health.get_or_create(analysis.domain, fetcher.tool_name)
    started = time.perf_counter()
    result = fetcher.fetch(analysis.normalized_url, context, domain_health=domain_health)
    duration_ms = int((time.perf_counter() - started) * 1000)
    _sync_run_counters_from_tool_result(url_run, result, fetch_increment=1)
    _record_tool_result(dependencies, workflow, fetcher.tool_name, "fetch", result, duration_ms)
    return result


def _save_raw_lead(
    command: UrlImportCommand,
    analysis: UrlImportAnalysis,
    url_run: UrlImportRun,
    fetch_result: ToolResult,
    dependencies: UrlImportWorkflowDependencies,
) -> RawJobLeadCaptureResult:
    text = str(fetch_result.artifacts.get("text") or "")
    raw_capture = dependencies.lead_service.capture_raw_lead(
        RawJobLeadCreate(
            source_id=url_run.source_id,
            source_url=str(fetch_result.artifacts.get("final_url") or analysis.normalized_url),
            raw_content=text,
            extracted_text=text,
            content_type="text/plain",
            raw_payload={
                "url_import_run_id": url_run.id,
                "input_url": command.url,
                "normalized_url": analysis.normalized_url,
                "source_type": analysis.source_type.value,
                "fetch_layer": analysis.fetch_layer,
                "title": fetch_result.artifacts.get("title"),
                "candidate_links": fetch_result.artifacts.get("candidate_links", []),
                "extraction_method": fetch_result.artifacts.get("extraction_method"),
            },
        )
    )
    url_run.raw_job_lead_id = raw_capture.raw_lead.id
    return raw_capture


def _save_visible_page_raw_lead(
    *,
    command: UrlImportCommand,
    analysis: UrlImportAnalysis,
    workflow: WorkflowRun,
    url_run: UrlImportRun,
    visible_text: str,
    title: str | None,
    final_url: str | None,
    dependencies: UrlImportWorkflowDependencies,
) -> RawJobLeadCaptureResult:
    parsed_title = title
    text = "\n".join(line.strip() for line in visible_text.splitlines() if line.strip())
    raw_payload: dict[str, Any] = {
        "url_import_run_id": url_run.id,
        "input_url": command.url,
        "normalized_url": analysis.normalized_url,
        "source_type": analysis.source_type.value,
        "fetch_layer": "mcp_visible_page",
        "final_url": final_url or analysis.normalized_url,
        "title": title,
        "extraction_method": "visible_page_text",
    }

    if analysis.source_type == JobSourceType.XIAOHONGSHU_NOTE:
        parsed = parse_xiaohongshu_visible_text(visible_text, page_title=title)
        parsed_title = parsed.title or title
        text = parsed.text
        raw_payload.update(
            {
                "title": parsed_title,
                "extraction_method": "xiaohongshu_visible_text",
                "image_count": parsed.image_count,
                "image_parse_deferred": parsed.image_parse_deferred,
            }
        )

    url_run.status = UrlImportRunStatus.RUNNING
    url_run.current_stage = "save_raw_job_lead"
    url_run.error_code = None
    url_run.error_message = None
    url_run.next_action = None
    url_run.tool_call_count += 1
    workflow.status = WorkflowRunStatus.RUNNING
    workflow.current_step = "save_raw_job_lead"
    workflow.error = None
    raw_capture = dependencies.lead_service.capture_raw_lead(
        RawJobLeadCreate(
            source_id=url_run.source_id,
            source_url=final_url or analysis.normalized_url,
            raw_content=text,
            extracted_text=text,
            content_type="text/plain",
            raw_payload=raw_payload,
        )
    )
    url_run.raw_job_lead_id = raw_capture.raw_lead.id
    return raw_capture


def _extract_job_leads_from_raw(
    *,
    command: UrlImportCommand,
    analysis: UrlImportAnalysis,
    workflow: WorkflowRun,
    url_run: UrlImportRun,
    raw_lead: RawJobLead,
    dependencies: UrlImportWorkflowDependencies,
) -> _ExtractionResult:
    url_run.current_stage = "extract_job_leads"
    workflow.current_step = "extract_job_leads"
    url_run.llm_call_count += 1
    url_run.tool_call_count += 1
    started = time.perf_counter()
    try:
        source = dependencies.lead_service.get_source(url_run.source_id)
        drafts = dependencies.social_provider.extract(
            source_id=url_run.source_id,
            raw_lead_id=raw_lead.id,
            raw_content=raw_lead.extracted_text or raw_lead.raw_content,
            source_url=raw_lead.source_url or analysis.normalized_url,
            trust_level=source.trust_level,
        )
        leads = [dependencies.lead_service.create_lead(_normalize_lead_status(draft)) for draft in drafts]
        if leads:
            dependencies.lead_service.mark_raw_lead_extracted(raw_lead)
        duration_ms = int((time.perf_counter() - started) * 1000)
        dependencies.automation_service.record_tool_call(
            ToolCallLogCreate(
                workflow_run_id=workflow.id,
                tool_name="BailianJobLeadExtractor",
                tool_group="llm",
                status=ToolCallStatus.SUCCEEDED,
                input_payload={"raw_job_lead_id": raw_lead.id, "source_url": raw_lead.source_url},
                output_payload={"extracted_count": len(leads), "lead_ids": [lead.id for lead in leads]},
                duration_ms=duration_ms,
            )
        )
    except Exception as exc:
        duration_ms = int((time.perf_counter() - started) * 1000)
        details = exception_error_details(
            category="llm_extraction_exception",
            exc=exc,
            url=raw_lead.source_url or command.url,
        )
        dependencies.automation_service.record_tool_call(
            ToolCallLogCreate(
                workflow_run_id=workflow.id,
                tool_name="BailianJobLeadExtractor",
                tool_group="llm",
                status=ToolCallStatus.FAILED,
                input_payload={"raw_job_lead_id": raw_lead.id, "source_url": raw_lead.source_url},
                output_payload={
                    "error_code": ToolErrorCode.LLM_EXTRACTION_FAILED.value,
                    "error_message": str(exc) or "LLM extraction failed",
                    "error_details": details,
                },
                error=str(exc) or "LLM extraction failed",
                duration_ms=duration_ms,
            )
        )
        url_run.status = UrlImportRunStatus.PARTIAL
        url_run.error_code = ToolErrorCode.LLM_EXTRACTION_FAILED.value
        url_run.error_message = str(exc) or "LLM extraction failed"
        url_run.next_action = ToolSuggestedNextAction.RETRY_SAME_STAGE.value
        url_run.run_metadata = _merge_metadata(url_run.run_metadata, {"error_details": details})
        workflow.status = WorkflowRunStatus.FAILED_RECOVERABLE
        workflow.error = url_run.error_message
        _checkpoint(
            dependencies,
            workflow,
            url_run,
            "extract_job_leads_failed",
            _state_for_checkpoint(command, analysis, url_run, raw_lead_id=raw_lead.id),
        )
        dependencies.url_import_runs.flush()
        return _ExtractionResult(ok=False, leads=[], recruiting_signals=[])

    recruiting_signals: list[RecruitingSignal] = []
    if not leads:
        recruiting_signals = _extract_recruiting_signals_from_raw(
            command=command,
            analysis=analysis,
            workflow=workflow,
            raw_lead=raw_lead,
            source_trust_level=source.trust_level,
            dependencies=dependencies,
        )
        if recruiting_signals:
            dependencies.lead_service.mark_raw_lead_extracted(raw_lead)
            url_run.status = UrlImportRunStatus.PARTIAL
            url_run.current_stage = "extract_recruiting_signals"
            url_run.extracted_count = 0
            url_run.error_code = None
            url_run.error_message = None
            url_run.next_action = ToolSuggestedNextAction.ENRICH_RECRUITING_SIGNAL.value
            url_run.run_metadata = _merge_metadata(
                url_run.run_metadata,
                {
                    "recruiting_signal_count": len(recruiting_signals),
                    "recruiting_signal_ids": [signal.id for signal in recruiting_signals],
                },
            )
            workflow.status = WorkflowRunStatus.COMPLETED
            workflow.current_step = "extract_recruiting_signals"
            workflow.error = None
            _checkpoint(
                dependencies,
                workflow,
                url_run,
                "save_recruiting_signals",
                _state_for_checkpoint(command, analysis, url_run, raw_lead_id=raw_lead.id),
            )
            dependencies.url_import_runs.flush()
            return _ExtractionResult(ok=False, leads=[], recruiting_signals=recruiting_signals)

    url_run.status = UrlImportRunStatus.SUCCEEDED if leads else UrlImportRunStatus.PARTIAL
    url_run.current_stage = "completed" if leads else "extract_job_leads"
    url_run.extracted_count = len(leads)
    url_run.error_code = None
    url_run.error_message = None
    url_run.next_action = ToolSuggestedNextAction.CONTINUE_WORKFLOW.value if leads else ToolSuggestedNextAction.REQUEST_MANUAL_PASTE.value
    url_run.finished_at = utc_now() if leads else None
    workflow.status = WorkflowRunStatus.COMPLETED if leads else WorkflowRunStatus.FAILED_RECOVERABLE
    workflow.current_step = "completed" if leads else "extract_job_leads"
    workflow.error = None if leads else "No job leads extracted"
    workflow.completed_at = url_run.finished_at if leads else None
    _checkpoint(
        dependencies,
        workflow,
        url_run,
        "save_job_leads" if leads else "extract_job_leads_empty",
        _state_for_checkpoint(command, analysis, url_run, raw_lead_id=raw_lead.id),
    )
    dependencies.url_import_runs.flush()
    return _ExtractionResult(ok=bool(leads), leads=leads, recruiting_signals=[])


def _extract_recruiting_signals_from_raw(
    *,
    command: UrlImportCommand,
    analysis: UrlImportAnalysis,
    workflow: WorkflowRun,
    raw_lead: RawJobLead,
    source_trust_level: JobSourceTrustLevel,
    dependencies: UrlImportWorkflowDependencies,
) -> list[RecruitingSignal]:
    started = time.perf_counter()
    raw_payload = raw_lead.raw_payload or {}
    drafts = dependencies.recruiting_signal_provider.extract(
        source_id=raw_lead.source_id,
        raw_lead_id=raw_lead.id,
        raw_content=raw_lead.extracted_text or raw_lead.raw_content,
        source_url=raw_lead.source_url or analysis.normalized_url,
        trust_level=source_trust_level,
        source_context={
            "input_url": command.url,
            "title": raw_payload.get("title"),
            "source_type": analysis.source_type.value,
            "candidate_links": raw_payload.get("candidate_links", []),
        },
    )
    signals = [dependencies.lead_service.create_recruiting_signal(draft).signal for draft in drafts]
    duration_ms = int((time.perf_counter() - started) * 1000)
    dependencies.automation_service.record_tool_call(
        ToolCallLogCreate(
            workflow_run_id=workflow.id,
            tool_name="RuleBasedRecruitingSignalExtractor",
            tool_group="program",
            status=ToolCallStatus.SUCCEEDED,
            input_payload={"raw_job_lead_id": raw_lead.id, "source_url": raw_lead.source_url},
            output_payload={"extracted_count": len(signals), "signal_ids": [signal.id for signal in signals]},
            duration_ms=duration_ms,
        )
    )
    return signals


def _ensure_import_source(
    command: UrlImportCommand,
    analysis: UrlImportAnalysis,
    dependencies: UrlImportWorkflowDependencies,
):
    return dependencies.lead_service.create_source(
        JobSourceCreate(
            name=f"URL Import {analysis.domain} {analysis.source_type.value}",
            source_type=analysis.source_type,
            entry_url=analysis.normalized_url,
            trust_level=command.trust_level or JobSourceTrustLevel.MEDIUM,
            fetch_mode=_fetch_mode_for_source_type(analysis.source_type),
            notes="Auto-created source for URL Import Pipeline",
            raw_payload={"created_by": "url_import_workflow"},
        )
    )


def _fetch_mode_for_source_type(source_type: JobSourceType) -> JobSourceFetchMode:
    if source_type in {JobSourceType.XIAOHONGSHU_NOTE, JobSourceType.JOB_BOARD_VISIBLE_PAGE}:
        return JobSourceFetchMode.MCP_VISIBLE_PAGE
    if source_type == JobSourceType.OFFICIAL_API:
        return JobSourceFetchMode.OFFICIAL_API
    if source_type == JobSourceType.MANUAL_CLIP:
        return JobSourceFetchMode.MANUAL_CLIP
    return JobSourceFetchMode.PUBLIC_HTML


def _select_fetcher(analysis: UrlImportAnalysis, dependencies: UrlImportWorkflowDependencies) -> Fetcher:
    if analysis.source_type == JobSourceType.WECHAT_ARTICLE:
        return _required_fetcher(dependencies, "WeChatArticleFetcher")
    if analysis.source_type in {
        JobSourceType.PUBLIC_ARTICLE,
        JobSourceType.UNIVERSITY_CAREER_SITE,
        JobSourceType.OFFICIAL_CAREER_SITE,
    }:
        return _required_fetcher(dependencies, "HTTPArticleFetcher")
    raise ValueError(f"No background fetcher for source type: {analysis.source_type.value}")


def _source_hint_from_url_run(url_run: UrlImportRun) -> JobSourceType | None:
    if url_run.source_type is None:
        return None
    if isinstance(url_run.source_type, JobSourceType):
        return url_run.source_type
    return JobSourceType(str(url_run.source_type))


def _required_fetcher(dependencies: UrlImportWorkflowDependencies, tool_name: str) -> Fetcher:
    fetcher = dependencies.fetchers.get(tool_name)
    if fetcher is None:
        raise ValueError(f"Fetcher not configured: {tool_name}")
    return fetcher


def _stage_for_fetcher(tool_name: str) -> str:
    return {
        "HTTPArticleFetcher": "http_article_fetch",
        "WeChatArticleFetcher": "wechat_article_fetch",
        "PlaywrightFetcher": "js_render_fetch",
        "Crawl4AIFetcher": "crawl4ai_extract",
    }.get(tool_name, "fetch_content")


def _sync_run_counters_from_tool_result(
    url_run: UrlImportRun,
    result: ToolResult,
    *,
    fetch_increment: int = 0,
) -> None:
    cost = result.cost or {}
    url_run.tool_call_count = int(cost.get("tool_calls", url_run.tool_call_count + 1))
    url_run.llm_call_count = int(cost.get("llm_calls", url_run.llm_call_count))
    url_run.attempt_count += fetch_increment


def _record_tool_result(
    dependencies: UrlImportWorkflowDependencies,
    workflow: WorkflowRun,
    tool_name: str,
    tool_group: str,
    result: ToolResult,
    duration_ms: int,
) -> None:
    status = ToolCallStatus.SUCCEEDED if result.ok else ToolCallStatus.FAILED
    if not result.ok and result.error_code in {
        ToolErrorCode.SOURCE_TYPE_NOT_ALLOWED,
        ToolErrorCode.TOOL_NOT_ALLOWED,
        ToolErrorCode.MCP_USER_CONFIRMATION_REQUIRED,
        ToolErrorCode.TOOL_CIRCUIT_OPEN,
    }:
        status = ToolCallStatus.BLOCKED
    dependencies.automation_service.record_tool_call(
        ToolCallLogCreate(
            workflow_run_id=workflow.id,
            tool_name=tool_name,
            tool_group=tool_group,
            status=status,
            input_payload={"stage": result.stage},
            output_payload=result.model_dump(mode="json"),
            error=result.error_message,
            duration_ms=duration_ms,
        )
    )


def _apply_failed_tool_result(workflow: WorkflowRun, url_run: UrlImportRun, result: ToolResult) -> None:
    url_run.error_code = _value(result.error_code)
    url_run.error_message = result.error_message
    url_run.next_action = _value(result.suggested_next_action)
    url_run.current_stage = result.stage
    url_run.run_metadata = _merge_metadata(url_run.run_metadata, {"error_details": result.error_details})
    if result.suggested_next_action == ToolSuggestedNextAction.REQUEST_USER_VISIBLE_PAGE:
        url_run.status = UrlImportRunStatus.WAITING_USER
        workflow.status = WorkflowRunStatus.WAITING_USER
    else:
        url_run.status = UrlImportRunStatus.FAILED_RECOVERABLE if result.retryable else UrlImportRunStatus.FAILED_TERMINAL
        workflow.status = WorkflowRunStatus.FAILED_RECOVERABLE if result.retryable else WorkflowRunStatus.FAILED
    workflow.current_step = url_run.current_stage
    workflow.error = result.error_message


def _mark_waiting_user(
    workflow: WorkflowRun,
    url_run: UrlImportRun,
    error_code: str,
    error_message: str,
    next_action: str,
) -> None:
    url_run.status = UrlImportRunStatus.WAITING_USER
    url_run.error_code = error_code
    url_run.error_message = error_message
    url_run.next_action = next_action
    workflow.status = WorkflowRunStatus.WAITING_USER
    workflow.error = error_message


def _checkpoint(
    dependencies: UrlImportWorkflowDependencies,
    workflow: WorkflowRun,
    url_run: UrlImportRun,
    checkpoint_key: str,
    state: dict[str, Any],
) -> None:
    dependencies.automation_service.save_checkpoint(
        WorkflowCheckpointCreate(
            workflow_run_id=workflow.id,
            checkpoint_key=checkpoint_key,
            state=state,
        )
    )


def _state_for_checkpoint(
    command: UrlImportCommand,
    analysis: UrlImportAnalysis,
    url_run: UrlImportRun,
    *,
    raw_lead_id: str | None = None,
    tool_result: ToolResult | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "input_url": command.url,
        "url_import_run_id": url_run.id,
        "workflow_run_id": url_run.workflow_run_id,
        "normalized_url": analysis.normalized_url,
        "normalized_url_hash": analysis.normalized_url_hash,
        "source_type": analysis.source_type.value,
        "domain": analysis.domain,
        "current_stage": url_run.current_stage,
        "status": _value(url_run.status),
        "tool_call_count": url_run.tool_call_count,
        "llm_call_count": url_run.llm_call_count,
    }
    if raw_lead_id is not None:
        state["raw_job_lead_id"] = raw_lead_id
    if tool_result is not None:
        state["tool_result"] = tool_result.model_dump(mode="json")
    return state


def _normalize_lead_status(draft: JobLeadCreate) -> JobLeadCreate:
    if draft.confidence_score is not None and draft.confidence_score < 70:
        return draft.model_copy(update={"verification_status": JobLeadStatus.PENDING_REVIEW})
    if not draft.apply_url:
        return draft.model_copy(update={"verification_status": JobLeadStatus.PENDING_REVIEW})
    return draft


def _merge_metadata(current: dict[str, Any] | None, values: dict[str, Any]) -> dict[str, Any]:
    merged = dict(current or {})
    merged.update(values)
    return merged


def _value(value: object) -> str | None:
    if value is None:
        return None
    return getattr(value, "value", str(value))


def _default_fetchers() -> dict[str, Fetcher]:
    return {
        "HTTPArticleFetcher": HTTPArticleFetcher(),
        "WeChatArticleFetcher": WeChatArticleFetcher(),
        "PlaywrightFetcher": PlaywrightFetcher(),
        "Crawl4AIFetcher": Crawl4AIFetcher(),
    }


def _load_langgraph_graph():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from langgraph.graph import END, START, StateGraph

    return END, START, StateGraph
