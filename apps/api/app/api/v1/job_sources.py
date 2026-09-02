from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.agent_runtime.workflows.url_import import (
    Fetcher,
    UrlImportCommand,
    build_url_import_dependencies,
    continue_url_import_workflow,
    continue_url_import_with_visible_page_content,
    create_url_import_run,
)
from app.agent_runtime.workflows.job_discovery import (
    OfficialApiSyncCommand,
    ManualSocialLeadImportCommand,
    UniversityCareerSyncCommand,
    WeChatAccountSyncCommand,
    run_manual_social_lead_import,
    run_offerio_official_api_source_sync,
    run_university_career_source_sync,
    run_wechat_account_source_sync,
)
from app.db.session import SessionLocal, get_db_session
from app.domains.automation.models import WorkflowRunStatus
from app.domains.jobs.models import (
    DomainHealthState,
    JobLeadStatus,
    JobSourceTrustLevel,
    JobSourceType,
    RawJobLead,
    UrlImportRun,
    UrlImportRunStatus,
)
from app.domains.jobs.providers.social_lead import SocialLeadImportProvider
from app.domains.jobs.providers.university_career import UniversityCareerProvider
from app.domains.jobs.providers.offerio import OfferIORecruitmentProvider
from app.domains.jobs.repository import (
    ArticleCandidateRepository,
    CompanyRepository,
    DomainHealthRepository,
    JobLeadRepository,
    JobRepository,
    JobSourceRepository,
    RawJobLeadRepository,
    RecruitingSignalRepository,
    SourceSyncRunRepository,
    UrlImportRunRepository,
)
from app.domains.jobs.schemas import (
    ArticleCandidateListResponse,
    ArticleCandidateRead,
    CompanySummaryRead,
    DomainHealthListResponse,
    DomainHealthRead,
    ImportUrlAcceptedResponse,
    ImportUrlRequest,
    JobLeadConversionResponse,
    JobLeadCreate,
    JobLeadExtractionRequest,
    JobLeadExtractionResponse,
    JobLeadListResponse,
    JobLeadRead,
    JobLeadVerification,
    JobSourceCreate,
    JobSourceListResponse,
    JobSourceRead,
    JobSourceSyncRequest,
    JobSourceSyncResponse,
    JobSourceUpdate,
    JobSummaryRead,
    RawJobLeadCaptureResponse,
    RawJobLeadCreate,
    RawJobLeadRead,
    RecruitingSignalListResponse,
    RecruitingSignalRead,
    ToolSuggestedNextAction,
    UrlImportRunRead,
    VisiblePageContentRequest,
)
from app.domains.jobs.providers.wechat_account_search import WeChatAccountSearchProvider
from app.domains.jobs.service import JobLeadService, JobService
from app.domains.jobs.verification import LeadVerifier, LeadVerificationService
from app.infrastructure.jobs.lead_verifier import HTTPLeadVerifier
from app.infrastructure.llm.job_lead_extractor import LLMJobLeadExtractor


source_router = APIRouter(prefix="/api/v1/job-sources", tags=["job-sources"])
lead_router = APIRouter(prefix="/api/v1/job-leads", tags=["job-leads"])
tool_health_router = APIRouter(prefix="/api/v1/tool-health", tags=["tool-health"])
article_candidate_router = APIRouter(prefix="/api/v1/article-candidates", tags=["article-candidates"])
recruiting_signal_router = APIRouter(prefix="/api/v1/recruiting-signals", tags=["recruiting-signals"])


def get_social_lead_provider() -> SocialLeadImportProvider:
    return SocialLeadImportProvider(extractor=LLMJobLeadExtractor())


def get_university_career_provider() -> UniversityCareerProvider:
    return UniversityCareerProvider()


def get_wechat_account_article_provider() -> WeChatAccountSearchProvider:
    return WeChatAccountSearchProvider()


def get_offerio_provider() -> OfferIORecruitmentProvider:
    return OfferIORecruitmentProvider()


def get_lead_verifier() -> LeadVerifier:
    return HTTPLeadVerifier()


def get_url_import_fetchers() -> dict[str, Fetcher] | None:
    return None


def get_url_import_session_factory() -> Callable[[], Session]:
    return SessionLocal


@source_router.post("", response_model=JobSourceRead, status_code=status.HTTP_201_CREATED)
def create_job_source(
    request: JobSourceCreate,
    session: Session = Depends(get_db_session),
) -> JobSourceRead:
    service = _lead_service(session)
    source = service.create_source(request)
    session.commit()
    return JobSourceRead.model_validate(source)


@source_router.get("", response_model=JobSourceListResponse)
def list_job_sources(session: Session = Depends(get_db_session)) -> JobSourceListResponse:
    sources = JobSourceRepository(session).list_enabled()
    return JobSourceListResponse(items=[JobSourceRead.model_validate(source) for source in sources])


@source_router.patch("/{source_id}", response_model=JobSourceRead)
def update_job_source(
    source_id: str,
    request: JobSourceUpdate,
    session: Session = Depends(get_db_session),
) -> JobSourceRead:
    try:
        source = _lead_service(session).update_source(source_id, request)
    except ValueError as exc:
        message = str(exc)
        status_code = 409 if "already exists" in message else 404
        raise HTTPException(status_code=status_code, detail=message) from exc
    session.commit()
    return JobSourceRead.model_validate(source)


@source_router.delete("/{source_id}", response_model=JobSourceRead)
def disable_job_source(
    source_id: str,
    session: Session = Depends(get_db_session),
) -> JobSourceRead:
    try:
        source = _lead_service(session).disable_source(source_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    session.commit()
    return JobSourceRead.model_validate(source)


@source_router.post("/{source_id}/sync", response_model=JobSourceSyncResponse)
def sync_job_source(
    source_id: str,
    request: JobSourceSyncRequest,
    session: Session = Depends(get_db_session),
    university_provider: UniversityCareerProvider = Depends(get_university_career_provider),
    social_provider: SocialLeadImportProvider = Depends(get_social_lead_provider),
    wechat_account_provider: WeChatAccountSearchProvider = Depends(get_wechat_account_article_provider),
    offerio_provider: OfferIORecruitmentProvider = Depends(get_offerio_provider),
) -> JobSourceSyncResponse:
    service = _lead_service(session)
    try:
        source = service.get_source(source_id)
        if _enum_value(source.source_type) == JobSourceType.WECHAT_ACCOUNT.value:
            account_result = run_wechat_account_source_sync(
                WeChatAccountSyncCommand(source_id=source_id, limit=request.limit),
                lead_service=service,
                article_provider=wechat_account_provider,
            )
            session.commit()
            return JobSourceSyncResponse(
                sync_run_id=account_result.sync_run.id,
                status=account_result.sync_run.status,
                fetched_count=account_result.fetched_count,
                extracted_count=account_result.extracted_count,
                failed_count=account_result.failed_count,
                error=account_result.error or account_result.sync_run.error,
                raw_leads=[],
                leads=[],
                article_candidates=[ArticleCandidateRead.model_validate(item) for item in account_result.article_candidates],
                recruiting_signals=[RecruitingSignalRead.model_validate(item) for item in account_result.recruiting_signals],
            )

        if _enum_value(source.source_type) == JobSourceType.OFFICIAL_API.value:
            official_result = run_offerio_official_api_source_sync(
                OfficialApiSyncCommand(source_id=source_id, limit=request.limit),
                lead_service=service,
                provider=offerio_provider,
            )
            session.commit()
            return JobSourceSyncResponse(
                sync_run_id=official_result.sync_run.id,
                status=official_result.sync_run.status,
                fetched_count=official_result.fetched_count,
                extracted_count=official_result.extracted_count,
                failed_count=official_result.failed_count,
                error=official_result.error or official_result.sync_run.error,
                raw_leads=[RawJobLeadRead.model_validate(item.raw_lead) for item in official_result.raw_captures],
                leads=[JobLeadRead.model_validate(lead) for lead in official_result.leads],
                article_candidates=[],
                recruiting_signals=[],
            )

        result = run_university_career_source_sync(
            UniversityCareerSyncCommand(source_id=source_id, limit=request.limit),
            lead_service=service,
            content_provider=university_provider,
            social_provider=social_provider,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Job source sync failed") from exc

    session.commit()
    return JobSourceSyncResponse(
        sync_run_id=result.sync_run.id,
        status=result.sync_run.status,
        fetched_count=result.fetched_count,
        extracted_count=result.extracted_count,
        failed_count=result.failed_count,
        error=result.error or result.sync_run.error,
        raw_leads=[RawJobLeadRead.model_validate(item.raw_lead) for item in result.raw_captures],
        leads=[JobLeadRead.model_validate(lead) for lead in result.leads],
        article_candidates=[],
        recruiting_signals=[],
    )


@lead_router.post("/raw", response_model=RawJobLeadCaptureResponse, status_code=status.HTTP_201_CREATED)
def capture_raw_job_lead(
    request: RawJobLeadCreate,
    session: Session = Depends(get_db_session),
) -> RawJobLeadCaptureResponse:
    try:
        result = _lead_service(session).capture_raw_lead(request)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    session.commit()
    return RawJobLeadCaptureResponse(
        raw_lead=result.raw_lead,
        created=result.created,
    )


@lead_router.post("", response_model=JobLeadRead, status_code=status.HTTP_201_CREATED)
def create_job_lead(
    request: JobLeadCreate,
    session: Session = Depends(get_db_session),
) -> JobLeadRead:
    try:
        lead = _lead_service(session).create_lead(request)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    session.commit()
    return JobLeadRead.model_validate(lead)


@lead_router.post("/extract", response_model=JobLeadExtractionResponse, status_code=status.HTTP_201_CREATED)
def extract_job_leads(
    request: JobLeadExtractionRequest,
    session: Session = Depends(get_db_session),
    provider: SocialLeadImportProvider = Depends(get_social_lead_provider),
) -> JobLeadExtractionResponse:
    try:
        result = run_manual_social_lead_import(
            ManualSocialLeadImportCommand(
                source_id=request.source_id,
                sync_run_id=request.sync_run_id,
                source_url=request.source_url,
                raw_content=request.raw_content,
                content_type=request.content_type,
            ),
            lead_service=_lead_service(session),
            provider=provider,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Job lead extraction failed") from exc

    session.commit()
    return JobLeadExtractionResponse(
        raw_lead=RawJobLeadRead.model_validate(result.raw_capture.raw_lead),
        raw_created=result.raw_capture.created,
        extracted_count=len(result.leads),
        leads=[JobLeadRead.model_validate(lead) for lead in result.leads],
    )


@lead_router.post("/import-url", response_model=ImportUrlAcceptedResponse, status_code=status.HTTP_202_ACCEPTED)
def import_job_leads_from_url(
    request: ImportUrlRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_db_session),
    provider: SocialLeadImportProvider = Depends(get_social_lead_provider),
    fetchers: dict[str, Fetcher] | None = Depends(get_url_import_fetchers),
    session_factory: Callable[[], Session] = Depends(get_url_import_session_factory),
) -> ImportUrlAcceptedResponse:
    command = UrlImportCommand(
        url=request.url,
        source_id=request.source_id,
        source_hint=request.source_hint,
        trust_level=request.trust_level,
        force_refresh=request.force_refresh,
    )
    try:
        result = create_url_import_run(
            command,
            dependencies=build_url_import_dependencies(
                session,
                fetchers=fetchers,
                social_provider=provider,
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    url_run = result.url_import_run
    domain_health_state = _domain_health_state(session, url_run.domain)
    session.commit()
    if url_run.status == UrlImportRunStatus.RUNNING:
        background_tasks.add_task(
            _run_url_import_background,
            url_run.id,
            provider,
            fetchers,
            session_factory,
        )
    return ImportUrlAcceptedResponse(
        run_id=url_run.id,
        status=url_run.status,
        current_stage=url_run.current_stage,
        domain_health_state=domain_health_state,
        message=_url_import_message(url_run),
    )


@lead_router.get("/import-runs/{run_id}", response_model=UrlImportRunRead)
def get_url_import_run(
    run_id: str,
    session: Session = Depends(get_db_session),
) -> UrlImportRunRead:
    url_run = UrlImportRunRepository(session).get(run_id)
    if url_run is None:
        raise HTTPException(status_code=404, detail=f"URL import run not found: {run_id}")
    return _url_import_run_read(session, url_run)


@lead_router.post("/import-runs/{run_id}/visible-page-content", response_model=UrlImportRunRead)
def submit_visible_page_content_for_import_run(
    run_id: str,
    request: VisiblePageContentRequest,
    session: Session = Depends(get_db_session),
    provider: SocialLeadImportProvider = Depends(get_social_lead_provider),
    fetchers: dict[str, Fetcher] | None = Depends(get_url_import_fetchers),
) -> UrlImportRunRead:
    try:
        result = continue_url_import_with_visible_page_content(
            run_id,
            visible_text=request.visible_text,
            title=request.title,
            final_url=request.final_url,
            dependencies=build_url_import_dependencies(
                session,
                fetchers=fetchers,
                social_provider=provider,
            ),
        )
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if "not found" in message else 400
        raise HTTPException(status_code=status_code, detail=message) from exc
    session.commit()
    return _url_import_run_read(session, result.url_import_run)


@tool_health_router.get("/domains", response_model=DomainHealthListResponse)
def list_domain_health(
    session: Session = Depends(get_db_session),
) -> DomainHealthListResponse:
    items = DomainHealthRepository(session).list_all()
    return DomainHealthListResponse(items=[DomainHealthRead.model_validate(item) for item in items])


@tool_health_router.get("/domains/{domain}", response_model=DomainHealthListResponse)
def list_domain_health_by_domain(
    domain: str,
    session: Session = Depends(get_db_session),
) -> DomainHealthListResponse:
    items = DomainHealthRepository(session).list_by_domain(domain)
    return DomainHealthListResponse(items=[DomainHealthRead.model_validate(item) for item in items])


@lead_router.get("", response_model=JobLeadListResponse)
def list_job_leads(
    verification_status: JobLeadStatus | None = None,
    source_id: str | None = None,
    source_type: JobSourceType | None = None,
    trust_level: JobSourceTrustLevel | None = None,
    company: str | None = None,
    job_direction: str | None = None,
    graduation_year: str | None = None,
    keyword: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    session: Session = Depends(get_db_session),
) -> JobLeadListResponse:
    leads = _lead_service(session).list_leads(
        source_id=source_id,
        source_type=source_type,
        trust_level=trust_level,
        verification_status=verification_status,
        company=company,
        job_direction=job_direction,
        graduation_year=graduation_year,
        keyword=keyword,
        limit=limit,
    )
    return JobLeadListResponse(items=[JobLeadRead.model_validate(lead) for lead in leads])


@article_candidate_router.get("", response_model=ArticleCandidateListResponse)
def list_article_candidates(
    source_id: str | None = None,
    status: str | None = None,
    limit: int = Query(default=100, ge=1, le=200),
    session: Session = Depends(get_db_session),
) -> ArticleCandidateListResponse:
    candidates = _lead_service(session).list_article_candidates(
        source_id=source_id,
        status=status,
        limit=limit,
    )
    return ArticleCandidateListResponse(items=[ArticleCandidateRead.model_validate(item) for item in candidates])


@recruiting_signal_router.get("", response_model=RecruitingSignalListResponse)
def list_recruiting_signals(
    source_id: str | None = None,
    status: str | None = None,
    company: str | None = None,
    graduation_year: str | None = None,
    limit: int = Query(default=100, ge=1, le=200),
    session: Session = Depends(get_db_session),
) -> RecruitingSignalListResponse:
    signals = _lead_service(session).list_recruiting_signals(
        source_id=source_id,
        status=status,
        company=company,
        graduation_year=graduation_year,
        limit=limit,
    )
    return RecruitingSignalListResponse(items=[RecruitingSignalRead.model_validate(item) for item in signals])


@lead_router.post("/{lead_id}/verify", response_model=JobLeadRead)
def verify_job_lead(
    lead_id: str,
    request: JobLeadVerification,
    session: Session = Depends(get_db_session),
) -> JobLeadRead:
    try:
        lead = _lead_service(session).verify_lead(lead_id, request)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    session.commit()
    return JobLeadRead.model_validate(lead)


@lead_router.post("/{lead_id}/convert", response_model=JobLeadConversionResponse)
def convert_job_lead(
    lead_id: str,
    session: Session = Depends(get_db_session),
) -> JobLeadConversionResponse:
    lead_service = _lead_service(session)
    job_service = JobService(
        companies=CompanyRepository(session),
        jobs=JobRepository(session),
    )
    try:
        result = lead_service.convert_verified_lead_to_job(lead_id, job_service)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    session.commit()
    return JobLeadConversionResponse(
        lead=JobLeadRead.model_validate(result.lead),
        job=_job_summary(result.job),
        created=result.created,
    )


@lead_router.post("/{lead_id}/verify-and-convert", response_model=JobLeadConversionResponse)
def verify_and_convert_job_lead(
    lead_id: str,
    session: Session = Depends(get_db_session),
    verifier: LeadVerifier = Depends(get_lead_verifier),
) -> JobLeadConversionResponse:
    lead_service = _lead_service(session)
    job_service = JobService(
        companies=CompanyRepository(session),
        jobs=JobRepository(session),
    )
    try:
        result = LeadVerificationService(
            lead_service=lead_service,
            job_service=job_service,
            verifier=verifier,
        ).verify_and_convert(lead_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    session.commit()
    return JobLeadConversionResponse(
        lead=JobLeadRead.model_validate(result.lead),
        job=_job_summary(result.job),
        created=result.created,
    )


def _run_url_import_background(
    run_id: str,
    provider: SocialLeadImportProvider,
    fetchers: dict[str, Fetcher] | None,
    session_factory: Callable[[], Session],
) -> None:
    with session_factory() as session:
        try:
            continue_url_import_workflow(
                run_id,
                dependencies=build_url_import_dependencies(
                    session,
                    fetchers=fetchers,
                    social_provider=provider,
                ),
            )
            session.commit()
        except Exception as exc:
            session.rollback()
            _mark_url_import_background_failed(session, run_id, exc)
            session.commit()


def _mark_url_import_background_failed(session: Session, run_id: str, exc: Exception) -> None:
    url_run = UrlImportRunRepository(session).get(run_id)
    if url_run is None:
        return
    url_run.status = UrlImportRunStatus.FAILED_RECOVERABLE
    url_run.error_code = "BACKGROUND_WORKFLOW_FAILED"
    url_run.error_message = str(exc) or "URL import background workflow failed"
    url_run.next_action = ToolSuggestedNextAction.RETRY_SAME_STAGE.value
    workflow = None
    if url_run.workflow_run_id:
        from app.domains.automation.repository import WorkflowRunRepository

        workflow = WorkflowRunRepository(session).get(url_run.workflow_run_id)
    if workflow is not None:
        workflow.status = WorkflowRunStatus.FAILED_RECOVERABLE
        workflow.error = url_run.error_message


def _domain_health_state(session: Session, domain: str | None) -> DomainHealthState:
    if not domain:
        return DomainHealthState.UNKNOWN
    states = [item.state for item in DomainHealthRepository(session).list_by_domain(domain)]
    if DomainHealthState.OPEN in states:
        return DomainHealthState.OPEN
    if DomainHealthState.HALF_OPEN in states:
        return DomainHealthState.HALF_OPEN
    if DomainHealthState.CLOSED in states:
        return DomainHealthState.CLOSED
    return DomainHealthState.UNKNOWN


def _url_import_message(url_run: UrlImportRun) -> str:
    if url_run.status == UrlImportRunStatus.RUNNING:
        return "已创建 URL 导入任务，正在后台解析"
    if url_run.status == UrlImportRunStatus.WAITING_USER:
        return "该链接需要用户可见页面确认后继续解析"
    if url_run.status == UrlImportRunStatus.DUPLICATE:
        return "该 URL 已存在导入记录，已跳过重复抓取"
    if url_run.status == UrlImportRunStatus.SUCCEEDED:
        return "URL 导入已完成"
    return "URL 导入任务已创建"


def _url_import_run_read(session: Session, url_run: UrlImportRun) -> UrlImportRunRead:
    run_read = UrlImportRunRead.model_validate(url_run)
    if not url_run.raw_job_lead_id:
        return run_read

    raw_lead = session.get(RawJobLead, url_run.raw_job_lead_id)
    if raw_lead is None:
        return run_read

    raw_payload = raw_lead.raw_payload or {}
    return run_read.model_copy(
        update={
            "raw_content_preview": raw_lead.raw_content[:500],
            "raw_extraction_method": raw_payload.get("extraction_method"),
            "raw_image_count": raw_payload.get("image_count"),
            "raw_image_parse_deferred": raw_payload.get("image_parse_deferred"),
        }
    )


def _lead_service(session: Session) -> JobLeadService:
    return JobLeadService(
        sources=JobSourceRepository(session),
        sync_runs=SourceSyncRunRepository(session),
        raw_leads=RawJobLeadRepository(session),
        leads=JobLeadRepository(session),
        article_candidates=ArticleCandidateRepository(session),
        recruiting_signals=RecruitingSignalRepository(session),
    )


def _job_summary(job) -> JobSummaryRead:
    return JobSummaryRead(
        id=job.id,
        title=job.title,
        company=CompanySummaryRead(id=job.company.id, name=job.company.name),
        city=job.city,
        source=job.source,
        source_job_id=job.source_job_id,
        source_url=job.source_url,
        job_type=job.job_type,
        skills=job.skills,
        status=job.status,
    )


def _enum_value(value: object) -> object:
    return getattr(value, "value", value)
