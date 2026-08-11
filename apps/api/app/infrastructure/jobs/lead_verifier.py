from __future__ import annotations

import httpx

from app.domains.jobs.models import JobLead
from app.domains.jobs.verification import LeadVerificationCheck


class HTTPLeadVerifier:
    def __init__(self, client: httpx.Client | None = None, timeout_seconds: float = 10.0) -> None:
        self._client = client or httpx.Client(follow_redirects=True)
        self._timeout_seconds = timeout_seconds

    def verify(self, lead: JobLead) -> LeadVerificationCheck:
        url = lead.apply_url or lead.verified_url or lead.source_url
        if not url:
            return LeadVerificationCheck(is_open=False, notes="No URL is available for verification")

        try:
            response = self._client.get(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=self._timeout_seconds,
            )
        except httpx.HTTPError as exc:
            return LeadVerificationCheck(is_open=False, verified_url=url, notes=str(exc))

        if response.status_code < 400:
            return LeadVerificationCheck(
                is_open=True,
                verified_url=str(response.url),
                notes=f"HTTP {response.status_code}",
            )
        return LeadVerificationCheck(
            is_open=False,
            verified_url=str(response.url),
            notes=f"HTTP {response.status_code}",
        )
