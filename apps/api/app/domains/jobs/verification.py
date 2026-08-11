from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.domains.jobs.models import JobLead, JobLeadStatus
from app.domains.jobs.schemas import JobLeadVerification
from app.domains.jobs.service import JobLeadConversionResult, JobLeadService, JobService


@dataclass(frozen=True)
class LeadVerificationCheck:
    is_open: bool
    verified_url: str | None = None
    notes: str | None = None


class LeadVerifier(Protocol):
    def verify(self, lead: JobLead) -> LeadVerificationCheck:
        ...


class LeadVerificationService:
    def __init__(
        self,
        *,
        lead_service: JobLeadService,
        job_service: JobService,
        verifier: LeadVerifier,
    ) -> None:
        self._lead_service = lead_service
        self._job_service = job_service
        self._verifier = verifier

    def verify_and_convert(self, lead_id: str) -> JobLeadConversionResult:
        lead = self._lead_service.get_lead(lead_id)
        check = self._verifier.verify(lead)
        if not check.is_open:
            self._lead_service.verify_lead(
                lead_id,
                JobLeadVerification(
                    verification_status=JobLeadStatus.EXPIRED,
                    verified_url=check.verified_url,
                    verification_notes=check.notes or "Verification failed before conversion",
                ),
            )
            raise ValueError("Job lead is not open after lazy verification")

        self._lead_service.verify_lead(
            lead_id,
            JobLeadVerification(
                verification_status=JobLeadStatus.VERIFIED,
                verified_url=check.verified_url,
                verification_notes=check.notes,
            ),
        )
        return self._lead_service.convert_verified_lead_to_job(lead_id, self._job_service)
