from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from app.domains.jobs.events import (
    JobImported,
    JobLeadCaptured,
    JobLeadConverted,
    JobLeadCreated,
    JobLeadVerified,
)
from app.domains.jobs.models import (
    Company,
    Job,
    JobLead,
    JobLeadStatus,
    JobSource,
    RawJobLead,
    RawJobLeadStatus,
    SourceSyncRun,
    SourceSyncRunStatus,
    utc_now,
)
from app.domains.jobs.repository import (
    CompanyRepository,
    JobLeadRepository,
    JobRepository,
    JobSourceRepository,
    RawJobLeadRepository,
    SourceSyncRunRepository,
)
from app.domains.jobs.schemas import (
    JobImportDraft,
    JobLeadCreate,
    JobLeadVerification,
    JobSourceCreate,
    RawJobLeadCreate,
    SourceSyncRunCreate,
)


@dataclass(frozen=True)
class JobImportResult:
    company: Company
    job: Job
    created: bool
    event: JobImported


@dataclass(frozen=True)
class RawJobLeadCaptureResult:
    raw_lead: RawJobLead
    created: bool
    event: JobLeadCaptured


@dataclass(frozen=True)
class JobLeadCreateResult:
    lead: JobLead
    created: bool
    event: JobLeadCreated


@dataclass(frozen=True)
class JobLeadConversionResult:
    lead: JobLead
    job: Job
    created: bool
    event: JobLeadConverted


class JobService:
    def __init__(self, companies: CompanyRepository, jobs: JobRepository) -> None:
        self._companies = companies
        self._jobs = jobs

    def import_job(self, draft: JobImportDraft) -> JobImportResult:
        existing = self._jobs.get_by_source_identity(draft.source, draft.source_job_id)
        if existing is not None:
            return JobImportResult(
                company=existing.company,
                job=existing,
                created=False,
                event=JobImported(
                    job_id=existing.id,
                    company_id=existing.company_id,
                    source=existing.source,
                    source_job_id=existing.source_job_id,
                    occurred_at=utc_now(),
                ),
            )

        normalized_name = self.normalize_company_name(draft.company_name)
        company = self._companies.get_by_normalized_name(normalized_name)
        if company is None:
            company = self._companies.add(
                Company(
                    name=draft.company_name,
                    normalized_name=normalized_name,
                    website_url=draft.company_website_url,
                    industry=draft.company_industry,
                    city=draft.company_city,
                    country=draft.company_country,
                    raw_payload=draft.raw_payload,
                )
            )

        job = self._jobs.add(
            Job(
                company=company,
                title=draft.title,
                city=draft.city,
                source=draft.source,
                source_job_id=draft.source_job_id,
                source_url=draft.source_url,
                job_type=draft.job_type,
                salary_text=draft.salary_text,
                jd_text=draft.jd_text,
                skills=draft.skills,
                date_posted=draft.date_posted,
                match_score=draft.match_score,
                status=draft.status,
                raw_payload=draft.raw_payload,
            )
        )

        return JobImportResult(
            company=company,
            job=job,
            created=True,
            event=JobImported(
                job_id=job.id,
                company_id=company.id,
                source=job.source,
                source_job_id=job.source_job_id,
                occurred_at=utc_now(),
            ),
        )

    @staticmethod
    def normalize_company_name(name: str) -> str:
        return " ".join(name.strip().lower().split())


class JobLeadService:
    def __init__(
        self,
        sources: JobSourceRepository,
        sync_runs: SourceSyncRunRepository,
        raw_leads: RawJobLeadRepository,
        leads: JobLeadRepository,
    ) -> None:
        self._sources = sources
        self._sync_runs = sync_runs
        self._raw_leads = raw_leads
        self._leads = leads

    def create_source(self, draft: JobSourceCreate) -> JobSource:
        existing = self._sources.get_by_name(draft.name)
        if existing is not None:
            return existing

        return self._sources.add(
            JobSource(
                name=draft.name,
                source_type=draft.source_type,
                entry_url=draft.entry_url,
                enabled=draft.enabled,
                sync_interval_hours=draft.sync_interval_hours,
                trust_level=draft.trust_level,
                fetch_mode=draft.fetch_mode,
                notes=draft.notes,
                raw_payload=draft.raw_payload,
            )
        )

    def get_source(self, source_id: str) -> JobSource:
        return self._require_source(source_id)

    def get_lead(self, lead_id: str) -> JobLead:
        return self._require_lead(lead_id)

    def start_sync_run(self, draft: SourceSyncRunCreate) -> SourceSyncRun:
        self._require_source(draft.source_id)
        return self._sync_runs.add(
            SourceSyncRun(
                source_id=draft.source_id,
                status=draft.status,
                fetched_count=draft.fetched_count,
                extracted_count=draft.extracted_count,
                failed_count=draft.failed_count,
                error=draft.error,
                run_metadata=draft.run_metadata,
            )
        )

    def finish_sync_run(
        self,
        sync_run: SourceSyncRun,
        *,
        status: SourceSyncRunStatus,
        fetched_count: int,
        extracted_count: int,
        failed_count: int,
        error: str | None = None,
    ) -> SourceSyncRun:
        sync_run.status = status
        sync_run.fetched_count = fetched_count
        sync_run.extracted_count = extracted_count
        sync_run.failed_count = failed_count
        sync_run.error = error
        sync_run.finished_at = utc_now()
        sync_run.source.last_synced_at = sync_run.finished_at
        return sync_run

    def capture_raw_lead(self, draft: RawJobLeadCreate) -> RawJobLeadCaptureResult:
        self._require_source(draft.source_id)
        content_hash = self._content_hash(draft.raw_content)
        existing = self._raw_leads.get_by_source_content_hash(draft.source_id, content_hash)
        if existing is not None:
            return RawJobLeadCaptureResult(
                raw_lead=existing,
                created=False,
                event=JobLeadCaptured(
                    raw_lead_id=existing.id,
                    source_id=existing.source_id,
                    content_hash=existing.content_hash,
                    occurred_at=utc_now(),
                ),
            )

        raw_lead = self._raw_leads.add(
            RawJobLead(
                source_id=draft.source_id,
                sync_run_id=draft.sync_run_id,
                source_url=draft.source_url,
                content_hash=content_hash,
                content_type=draft.content_type,
                raw_content=draft.raw_content,
                extracted_text=draft.extracted_text,
                status=draft.status,
                raw_payload=draft.raw_payload,
            )
        )
        return RawJobLeadCaptureResult(
            raw_lead=raw_lead,
            created=True,
            event=JobLeadCaptured(
                raw_lead_id=raw_lead.id,
                source_id=raw_lead.source_id,
                content_hash=raw_lead.content_hash,
                occurred_at=utc_now(),
            ),
        )

    def create_lead(self, draft: JobLeadCreate) -> JobLead:
        source = self._require_source(draft.source_id)
        lead_hash = self._lead_hash(draft)
        existing = self._leads.get_by_source_lead_hash(draft.source_id, lead_hash)
        if existing is not None:
            return existing

        return self._leads.add(
            JobLead(
                source_id=draft.source_id,
                raw_lead_id=draft.raw_lead_id,
                lead_hash=lead_hash,
                company_name=draft.company_name,
                title=draft.title,
                city=draft.city,
                job_direction=draft.job_direction,
                graduation_year=draft.graduation_year,
                source_url=draft.source_url,
                apply_url=draft.apply_url,
                job_type=draft.job_type,
                salary_text=draft.salary_text,
                jd_text=draft.jd_text,
                skills=draft.skills,
                deadline=draft.deadline,
                confidence_score=draft.confidence_score,
                trust_level=draft.trust_level or source.trust_level,
                verification_status=draft.verification_status,
                raw_payload=draft.raw_payload,
            )
        )

    def verify_lead(self, lead_id: str, verification: JobLeadVerification) -> JobLead:
        lead = self._require_lead(lead_id)
        lead.verification_status = verification.verification_status
        lead.verified_url = verification.verified_url
        lead.verification_notes = verification.verification_notes
        lead.verified_at = utc_now()
        return lead

    def mark_raw_lead_extracted(self, raw_lead: RawJobLead) -> RawJobLead:
        raw_lead.status = RawJobLeadStatus.EXTRACTED
        return raw_lead

    def convert_verified_lead_to_job(
        self,
        lead_id: str,
        job_service: JobService,
    ) -> JobLeadConversionResult:
        lead = self._require_lead(lead_id)
        if lead.verification_status == JobLeadStatus.CONVERTED and lead.converted_job is not None:
            return self._conversion_result(lead, lead.converted_job, created=False)
        if lead.verification_status != JobLeadStatus.VERIFIED:
            raise ValueError("Only verified job leads can be converted to formal jobs")

        result = job_service.import_job(
            JobImportDraft(
                company_name=lead.company_name,
                title=lead.title,
                city=lead.city,
                source="job_lead",
                source_job_id=lead.id,
                source_url=lead.verified_url or lead.apply_url or lead.source_url,
                job_type=lead.job_type,
                salary_text=lead.salary_text,
                jd_text=lead.jd_text,
                skills=lead.skills,
                raw_payload={
                    "job_lead_id": lead.id,
                    "source_id": lead.source_id,
                    "raw_lead_id": lead.raw_lead_id,
                    "source_type": lead.source.source_type,
                    "source_url": lead.source_url,
                    "apply_url": lead.apply_url,
                },
            )
        )
        lead.converted_job_id = result.job.id
        lead.verification_status = JobLeadStatus.CONVERTED
        lead.converted_at = utc_now()
        return self._conversion_result(lead, result.job, created=result.created)

    def _require_source(self, source_id: str) -> JobSource:
        source = self._sources.get(source_id)
        if source is None:
            raise ValueError(f"Job source not found: {source_id}")
        return source

    def _require_lead(self, lead_id: str) -> JobLead:
        lead = self._leads.get(lead_id)
        if lead is None:
            raise ValueError(f"Job lead not found: {lead_id}")
        return lead

    @staticmethod
    def _content_hash(raw_content: str) -> str:
        normalized = " ".join(raw_content.split())
        return sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _lead_hash(draft: JobLeadCreate) -> str:
        values = [
            draft.company_name,
            draft.title,
            draft.city or "",
            draft.source_url or "",
            draft.apply_url or "",
        ]
        normalized = "|".join(" ".join(value.lower().split()) for value in values)
        return sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _conversion_result(lead: JobLead, job: Job, created: bool) -> JobLeadConversionResult:
        return JobLeadConversionResult(
            lead=lead,
            job=job,
            created=created,
            event=JobLeadConverted(
                lead_id=lead.id,
                job_id=job.id,
                source_id=lead.source_id,
                occurred_at=utc_now(),
            ),
        )
