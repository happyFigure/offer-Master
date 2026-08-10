from __future__ import annotations

import re
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from html import unescape
from typing import Any
from urllib.parse import urlparse

import httpx

from app.domains.jobs.providers.base import JobSearchQuery, ProviderStatus, RawJob
from app.domains.jobs.schemas import JobImportDraft


class JobicyProvider:
    name = "jobicy"
    source_type = "public_api"
    api_url = "https://jobicy.com/api/v2/remote-jobs"

    def __init__(self, client: Any | None = None, timeout_seconds: int = 15) -> None:
        self._client = client or httpx.Client()
        self._timeout_seconds = timeout_seconds

    def health_check(self) -> ProviderStatus:
        try:
            self.search(JobSearchQuery(limit=1))
        except Exception as exc:  # pragma: no cover - defensive boundary for runtime checks
            return ProviderStatus(name=self.name, available=False, message=str(exc))
        return ProviderStatus(name=self.name, available=True)

    def search(self, query: JobSearchQuery) -> list[RawJob]:
        params: dict[str, Any] = {"count": self._normalize_limit(query.limit)}
        if query.keyword:
            params["tag"] = query.keyword
        if query.city:
            params["geo"] = query.city

        response = self._client.get(
            self.api_url,
            params=params,
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        jobs = payload.get("jobs", []) if isinstance(payload, dict) else []
        return [RawJob(source=self.name, payload=job) for job in jobs if isinstance(job, dict)]

    def normalize(self, raw_job: RawJob) -> JobImportDraft:
        payload = raw_job.payload
        title = self._required_text(payload, "jobTitle")
        company_name = self._required_text(payload, "companyName")
        source_url = self._optional_text(payload.get("url"))
        source_job_id = self._source_job_id(payload, source_url)
        if not source_job_id:
            raise ValueError("Jobicy payload is missing id and url identity")

        return JobImportDraft(
            company_name=company_name,
            title=title,
            city=self._optional_text(payload.get("jobGeo")),
            source=self.name,
            source_job_id=source_job_id,
            source_url=source_url,
            job_type=self._optional_text(payload.get("jobType")),
            jd_text=self._strip_html(self._optional_text(payload.get("jobDescription"))),
            skills=self._skills(payload),
            date_posted=self._parse_date(payload.get("pubDate")),
            raw_payload=payload,
        )

    @staticmethod
    def _normalize_limit(limit: int) -> int:
        return min(max(limit, 1), 100)

    @staticmethod
    def _required_text(payload: dict[str, Any], key: str) -> str:
        value = JobicyProvider._optional_text(payload.get(key))
        if not value:
            raise ValueError(f"Jobicy payload is missing required field: {key}")
        return value

    @staticmethod
    def _optional_text(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _source_job_id(payload: dict[str, Any], source_url: str | None) -> str | None:
        payload_id = JobicyProvider._optional_text(payload.get("id"))
        if payload_id:
            return payload_id
        if not source_url:
            return None
        path = urlparse(source_url).path.rstrip("/")
        return path.rsplit("/", 1)[-1] or None

    @staticmethod
    def _strip_html(value: str | None) -> str | None:
        if value is None:
            return None
        without_tags = re.sub(r"<[^>]+>", " ", value)
        normalized = " ".join(unescape(without_tags).split())
        return normalized or None

    @staticmethod
    def _skills(payload: dict[str, Any]) -> list[str]:
        values: list[str] = []
        for key in ("jobIndustry", "jobLevel"):
            raw_value = payload.get(key)
            if isinstance(raw_value, list):
                values.extend(str(item).strip() for item in raw_value if str(item).strip())
            elif raw_value:
                values.append(str(raw_value).strip())
        return list(dict.fromkeys(values))

    @staticmethod
    def _parse_date(value: Any) -> date | None:
        text = JobicyProvider._optional_text(value)
        if text is None:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
        except ValueError:
            try:
                return parsedate_to_datetime(text).date()
            except (TypeError, ValueError):
                return None
