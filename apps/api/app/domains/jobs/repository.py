from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domains.jobs.models import Company, Job, JobStatus


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
