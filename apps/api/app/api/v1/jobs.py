from __future__ import annotations

from collections.abc import Mapping

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.session import get_db_session
from app.domains.jobs.providers.base import JobProvider, JobSearchQuery
from app.domains.jobs.providers.jobicy import JobicyProvider
from app.domains.jobs.providers.offerio import OfferIOCompany, OfferIOCompanyOpening, OfferIOJob, OfferIOPage, OfferIORecruitmentProvider
from app.domains.jobs.providers.tencent_campus import TencentCampusProvider
from app.domains.jobs.repository import CompanyRepository, JobRepository
from app.domains.jobs.schemas import (
    CompanySummaryRead,
    OfferIOCompanyOpeningListResponse,
    OfferIOCompanyOpeningRead,
    OfferIOCompanyListResponse,
    OfferIOCompanyRead,
    OfferIOJobListResponse,
    OfferIOJobRead,
    JobSummaryRead,
    JobSyncRequest,
    JobSyncResponse,
)
from app.domains.jobs.service import JobService


router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


def get_job_providers() -> Mapping[str, JobProvider]:
    return {
        "tencent_campus": TencentCampusProvider(),
        "jobicy": JobicyProvider(),
    }


def get_offerio_provider() -> OfferIORecruitmentProvider:
    return OfferIORecruitmentProvider()


@router.get("/offerio/companies", response_model=OfferIOCompanyListResponse)
def list_offerio_companies(
    job_type: str = Query(default="校招"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    keyword: str | None = Query(default=None),
    industry: str | None = Query(default=None),
    provider: OfferIORecruitmentProvider = Depends(get_offerio_provider),
) -> OfferIOCompanyListResponse:
    try:
        result = provider.list_companies(
            job_type=job_type,
            page=page,
            page_size=page_size,
            keyword=keyword,
            industry=industry,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OfferIO companies fetch failed: {exc}") from exc
    return _offerio_company_page(result)


@router.get("/offerio/company-openings", response_model=OfferIOCompanyOpeningListResponse)
def list_offerio_company_openings(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=100),
    keyword: str | None = Query(default=None),
    industry: str | None = Query(default=None),
    batch: str | None = Query(default=None),
    target: str | None = Query(default=None),
    company_nature: str | None = Query(default=None),
    provider: OfferIORecruitmentProvider = Depends(get_offerio_provider),
) -> OfferIOCompanyOpeningListResponse:
    try:
        result = provider.list_company_openings(
            page=page,
            page_size=page_size,
            keyword=keyword,
            industry=industry,
            batch=batch,
            target=target,
            company_nature=company_nature,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OfferIO company openings fetch failed: {exc}") from exc
    return _offerio_company_opening_page(result)


@router.get("/offerio/jobs", response_model=OfferIOJobListResponse)
def list_offerio_jobs(
    job_type: str = Query(default="校招"),
    company: str | None = Query(default=None),
    keyword: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    provider: OfferIORecruitmentProvider = Depends(get_offerio_provider),
) -> OfferIOJobListResponse:
    try:
        result = provider.list_jobs(
            job_type=job_type,
            company=company,
            keyword=keyword,
            page=page,
            page_size=page_size,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OfferIO jobs fetch failed: {exc}") from exc
    return _offerio_job_page(result)


@router.post("/sync", response_model=JobSyncResponse)
def sync_jobs(
    request: JobSyncRequest,
    session: Session = Depends(get_db_session),
    providers: Mapping[str, JobProvider] = Depends(get_job_providers),
) -> JobSyncResponse:
    unsupported_sources = [source for source in request.sources if source not in providers]
    if unsupported_sources:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported job providers: {', '.join(unsupported_sources)}",
        )

    service = JobService(
        companies=CompanyRepository(session),
        jobs=JobRepository(session),
    )
    query = JobSearchQuery(
        keyword=request.keyword,
        city=request.city,
        remote_type=request.remote_type,
        job_type=request.job_type,
        sources=request.sources,
        limit=request.limit,
    )

    imported = 0
    duplicates = 0
    failed = 0
    synced_jobs: list[JobSummaryRead] = []

    for source in request.sources:
        provider = providers[source]
        try:
            raw_jobs = provider.search(query)
        except Exception:
            failed += 1
            continue

        for raw_job in raw_jobs:
            try:
                result = service.import_job(provider.normalize(raw_job))
            except Exception:
                failed += 1
                continue

            if result.created:
                imported += 1
            else:
                duplicates += 1
            synced_jobs.append(_job_summary(result.job))

    session.commit()

    return JobSyncResponse(
        requested_sources=request.sources,
        imported=imported,
        duplicates=duplicates,
        failed=failed,
        jobs=synced_jobs,
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


def _offerio_company_page(page: OfferIOPage) -> OfferIOCompanyListResponse:
    return OfferIOCompanyListResponse(
        items=[_offerio_company_item(item) for item in page.items],
        page=page.page,
        page_size=page.page_size,
        total=page.total,
        total_pages=page.total_pages,
    )


def _offerio_company_item(item: OfferIOCompany) -> OfferIOCompanyRead:
    return OfferIOCompanyRead(
        name=item.name,
        company_nature=item.company_nature,
        industry=item.industry,
        locations=item.locations,
        job_count=item.job_count,
        updated_at=item.updated_at,
        raw_payload=item.raw_payload,
    )


def _offerio_company_opening_page(page: OfferIOPage) -> OfferIOCompanyOpeningListResponse:
    return OfferIOCompanyOpeningListResponse(
        items=[_offerio_company_opening_item(item) for item in page.items],
        page=page.page,
        page_size=page.page_size,
        total=page.total,
        total_pages=page.total_pages,
    )


def _offerio_company_opening_item(item: OfferIOCompanyOpening) -> OfferIOCompanyOpeningRead:
    return OfferIOCompanyOpeningRead(
        id=item.id,
        company_name=item.company_name,
        company_nature=item.company_nature,
        industry=item.industry,
        batch=item.batch,
        target=item.target,
        location=item.location,
        positions=item.positions,
        update_date=item.update_date,
        deadline=item.deadline,
        apply_link=item.apply_link,
        has_written_test=item.has_written_test,
        raw_payload=item.raw_payload,
    )


def _offerio_job_page(page: OfferIOPage) -> OfferIOJobListResponse:
    return OfferIOJobListResponse(
        items=[_offerio_job_item(item) for item in page.items],
        page=page.page,
        page_size=page.page_size,
        total=page.total,
        total_pages=page.total_pages,
    )


def _offerio_job_item(item: OfferIOJob) -> OfferIOJobRead:
    return OfferIOJobRead(
        id=item.id,
        title=item.title,
        company=item.company,
        location=item.location,
        category=item.category,
        job_type=item.job_type,
        publish_date=item.publish_date,
        salary=item.salary,
        deadline=item.deadline,
        department=item.department,
        apply_link=item.apply_link,
        source=item.source,
        responsibilities=item.responsibilities or [],
        requirements=item.requirements or [],
        raw_payload=item.raw_payload,
    )
