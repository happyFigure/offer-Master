from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domains.jobs.models import (
    Company,
    Job,
    JobLead,
    JobLeadStatus,
    JobSource,
    RawJobLead,
    SourceSyncRun,
    JobStatus,
)


class CompanyRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, company_id: str) -> Company | None:
        return self._session.get(Company, company_id)

    def get_by_normalized_name(self, normalized_name: str) -> Company | None:
        return self._session.scalar(
            select(Company).where(Company.normalized_name == normalized_name)
        )

    def add(self, company: Company) -> Company:
        self._session.add(company)
        self._session.flush()
        return company


class JobRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, job_id: str) -> Job | None:
        return self._session.get(Job, job_id)

    def get_by_source_identity(self, source: str, source_job_id: str) -> Job | None:
        return self._session.scalar(
            select(Job).where(
                Job.source == source,
                Job.source_job_id == source_job_id,
            )
        )

    def list_by_status(self, status: JobStatus | None = None) -> list[Job]:
        statement = select(Job).order_by(Job.created_at.desc())
        if status is not None:
            statement = statement.where(Job.status == status)
        return list(self._session.scalars(statement).all())

    def add(self, job: Job) -> Job:
        self._session.add(job)
        self._session.flush()
        return job


class JobSourceRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, source_id: str) -> JobSource | None:
        return self._session.get(JobSource, source_id)

    def get_by_name(self, name: str) -> JobSource | None:
        return self._session.scalar(select(JobSource).where(JobSource.name == name))

    def list_enabled(self) -> list[JobSource]:
        statement = select(JobSource).where(JobSource.enabled.is_(True)).order_by(JobSource.name)
        return list(self._session.scalars(statement).all())

    def add(self, source: JobSource) -> JobSource:
        self._session.add(source)
        self._session.flush()
        return source


class SourceSyncRunRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, run_id: str) -> SourceSyncRun | None:
        return self._session.get(SourceSyncRun, run_id)

    def add(self, run: SourceSyncRun) -> SourceSyncRun:
        self._session.add(run)
        self._session.flush()
        return run


class RawJobLeadRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, raw_lead_id: str) -> RawJobLead | None:
        return self._session.get(RawJobLead, raw_lead_id)

    def get_by_source_content_hash(self, source_id: str, content_hash: str) -> RawJobLead | None:
        return self._session.scalar(
            select(RawJobLead).where(
                RawJobLead.source_id == source_id,
                RawJobLead.content_hash == content_hash,
            )
        )

    def add(self, raw_lead: RawJobLead) -> RawJobLead:
        self._session.add(raw_lead)
        self._session.flush()
        return raw_lead


class JobLeadRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, lead_id: str) -> JobLead | None:
        return self._session.get(JobLead, lead_id)

    def get_by_source_lead_hash(self, source_id: str, lead_hash: str) -> JobLead | None:
        return self._session.scalar(
            select(JobLead).where(
                JobLead.source_id == source_id,
                JobLead.lead_hash == lead_hash,
            )
        )

    def list_by_status(self, status: JobLeadStatus | None = None) -> list[JobLead]:
        statement = select(JobLead).order_by(JobLead.created_at.desc())
        if status is not None:
            statement = statement.where(JobLead.verification_status == status)
        return list(self._session.scalars(statement).all())

    def add(self, lead: JobLead) -> JobLead:
        self._session.add(lead)
        self._session.flush()
        return lead
