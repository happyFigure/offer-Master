from __future__ import annotations

from dataclasses import dataclass

from app.domains.jobs.events import JobImported
from app.domains.jobs.models import Company, Job, utc_now
from app.domains.jobs.repository import CompanyRepository, JobRepository
from app.domains.jobs.schemas import JobImportDraft


@dataclass(frozen=True)
class JobImportResult:
    company: Company
    job: Job
    created: bool
    event: JobImported


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
