from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any

import httpx

from app.domains.jobs.providers.base import JobSearchQuery, ProviderStatus, RawJob
from app.domains.jobs.schemas import JobImportDraft


OFFERIO_BASE_URL = "https://offerio.work"


@dataclass(frozen=True)
class OfferIOCompany:
    name: str
    company_nature: str | None = None
    industry: str | None = None
    locations: str | None = None
    job_count: int = 0
    updated_at: str | None = None
    raw_payload: dict[str, Any] | None = None


@dataclass(frozen=True)
class OfferIOCompanyOpening:
    id: str
    company_name: str
    company_nature: str | None = None
    industry: str | None = None
    batch: str | None = None
    target: str | None = None
    location: str | None = None
    positions: str | None = None
    update_date: str | None = None
    deadline: str | None = None
    apply_link: str | None = None
    has_written_test: str | None = None
    raw_payload: dict[str, Any] | None = None


@dataclass(frozen=True)
class OfferIOJob:
    id: str
    title: str
    company: str
    location: str | None = None
    category: str | None = None
    job_type: str | None = None
    publish_date: str | None = None
    salary: str | None = None
    deadline: str | None = None
    department: str | None = None
    apply_link: str | None = None
    source: str | None = None
    responsibilities: list[str] | None = None
    requirements: list[str] | None = None
    raw_payload: dict[str, Any] | None = None


@dataclass(frozen=True)
class OfferIOPage:
    items: list[Any]
    page: int
    page_size: int
    total: int
    total_pages: int


class OfferIORecruitmentProvider:
    name = "offerio"
    source_type = "temporary_recruitment_api"

    def __init__(self, client: Any | None = None, base_url: str = OFFERIO_BASE_URL) -> None:
        self._client = client or httpx.Client(timeout=12.0, follow_redirects=True)
        self._base_url = base_url.rstrip("/")

    def health_check(self) -> ProviderStatus:
        try:
            self.list_companies(page=1, page_size=1)
        except Exception as exc:  # pragma: no cover - defensive boundary
            return ProviderStatus(name=self.name, available=False, message=str(exc))
        return ProviderStatus(name=self.name, available=True)

    def list_companies(
        self,
        *,
        job_type: str = "校招",
        page: int = 1,
        page_size: int = 20,
        keyword: str | None = None,
        industry: str | None = None,
    ) -> OfferIOPage:
        payload = self._get_json(
            "/api/recruitment/job-companies",
            {
                "page": page,
                "pageSize": page_size,
                "jobType": job_type,
                "keyword": keyword,
                "industry": industry,
            },
        )
        companies = [_normalize_company(item) for item in _list_from(payload, "companies")]
        if keyword:
            normalized_keyword = keyword.strip().lower()
            companies = [item for item in companies if normalized_keyword in item.name.lower()]
        if industry:
            normalized_industry = industry.strip().lower()
            companies = [item for item in companies if normalized_industry in (item.industry or "").lower()]
        return OfferIOPage(
            items=companies,
            page=_int_payload(payload, "page", page),
            page_size=_int_payload(payload, "pageSize", page_size),
            total=_int_payload(payload, "total", len(companies)),
            total_pages=_int_payload(payload, "totalPages", 1),
        )

    def list_company_openings(
        self,
        *,
        page: int = 1,
        page_size: int = 50,
        keyword: str | None = None,
        industry: str | None = None,
        batch: str | None = None,
        target: str | None = None,
        company_nature: str | None = None,
    ) -> OfferIOPage:
        payload = self._get_json(
            "/api/recruitment/companies",
            {
                "page": page,
                "pageSize": page_size,
                "keyword": keyword,
                "industry": industry,
                "batch": batch,
                "target": target,
                "companyNature": company_nature,
            },
        )
        openings = [_normalize_company_opening(item) for item in _list_from(payload, "companies")]
        return OfferIOPage(
            items=openings,
            page=_int_payload(payload, "page", page),
            page_size=_int_payload(payload, "pageSize", page_size),
            total=_int_payload(payload, "total", len(openings)),
            total_pages=_int_payload(payload, "totalPages", 1),
        )

    def list_jobs(
        self,
        *,
        job_type: str = "校招",
        company: str | None = None,
        keyword: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> OfferIOPage:
        payload = self._get_json(
            "/api/recruitment/jobs",
            {
                "page": page,
                "pageSize": page_size,
                "jobType": job_type,
                "company": company,
                "keyword": keyword,
            },
        )
        jobs = [_normalize_job(item) for item in _list_from(payload, "jobs")]
        return OfferIOPage(
            items=jobs,
            page=_int_payload(payload, "page", page),
            page_size=_int_payload(payload, "pageSize", page_size),
            total=_int_payload(payload, "total", len(jobs)),
            total_pages=_int_payload(payload, "totalPages", 1),
        )

    def search(self, query: JobSearchQuery) -> list[RawJob]:
        page = self.list_jobs(
            job_type=query.job_type if query.job_type != "any" else "校招",
            company=query.keyword,
            keyword=query.keyword,
            page=1,
            page_size=query.limit,
        )
        return [RawJob(source=self.name, payload=item.raw_payload or {}) for item in page.items]

    def normalize(self, raw_job: RawJob) -> JobImportDraft:
        item = _normalize_job(raw_job.payload)
        return JobImportDraft(
            company_name=item.company,
            company_industry=_text(raw_job.payload, "industry"),
            title=item.title,
            city=item.location,
            source=self.name,
            source_job_id=item.id,
            source_url=item.apply_link,
            job_type=item.job_type,
            salary_text=item.salary,
            jd_text=_build_jd_text(item),
            skills=_extract_skills(raw_job.payload, item.category),
            date_posted=_parse_date(item.publish_date),
            raw_payload=raw_job.payload,
        )

    def _get_json(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        clean_params = {key: value for key, value in params.items() if value not in (None, "")}
        response = self._client.get(
            f"{self._base_url}{path}",
            params=clean_params,
            headers=_browser_headers(),
        )
        response.raise_for_status()
        return _decode_json_response(response)


def _browser_headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Referer": "https://offerio.work/recruitment",
        "Origin": "https://offerio.work",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }


def _decode_json_response(response: Any) -> dict[str, Any]:
    if hasattr(response, "content"):
        text = bytes(response.content).decode("utf-8", errors="replace")
        return json.loads(text)
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("OfferIO response must be a JSON object")
    return data


def _normalize_company(payload: dict[str, Any]) -> OfferIOCompany:
    return OfferIOCompany(
        name=_required_text(payload, "company", "name", "companyName"),
        company_nature=_text(payload, "companyNature", "nature", "enterpriseNature", "property"),
        industry=_text(payload, "industry", "industryName"),
        locations=_text(payload, "location", "locations", "workLocation", "workplace", "city"),
        job_count=_int_text(payload, "jobCount", "job_count", "jobNum", "positionCount"),
        updated_at=_text(payload, "updateTime", "updatedAt", "updated_at", "date"),
        raw_payload=payload,
    )


def _normalize_company_opening(payload: dict[str, Any]) -> OfferIOCompanyOpening:
    return OfferIOCompanyOpening(
        id=_required_text(payload, "id"),
        company_name=_required_text(payload, "companyName", "company", "name"),
        company_nature=_text(payload, "companyNature", "nature", "enterpriseNature", "property"),
        industry=_text(payload, "industry", "industryName"),
        batch=_text(payload, "batch"),
        target=_text(payload, "target", "graduationYear"),
        location=_text(payload, "location", "locations", "workLocation", "workplace", "city"),
        positions=_text(payload, "positions", "position", "jobName", "title"),
        update_date=_text(payload, "updateDate", "updateTime", "updatedAt", "date"),
        deadline=_text(payload, "deadline", "endDate"),
        apply_link=_text(payload, "applyLink", "applyUrl", "sourceUrl", "url"),
        has_written_test=_text(payload, "hasWrittenTest", "writtenTest"),
        raw_payload=payload,
    )


def _normalize_job(payload: dict[str, Any]) -> OfferIOJob:
    return OfferIOJob(
        id=_required_text(payload, "id", "jobId", "sourceJobId"),
        title=_required_text(payload, "title", "jobName", "name"),
        company=_required_text(payload, "company", "companyName"),
        location=_text(payload, "location", "workLocation", "city"),
        category=_text(payload, "category", "jobCategory"),
        job_type=_text(payload, "internType", "jobType", "type"),
        publish_date=_text(payload, "publishDate", "date", "createdAt"),
        salary=_text(payload, "salary", "salaryText"),
        deadline=_text(payload, "deadline", "endDate"),
        department=_text(payload, "department"),
        apply_link=_text(payload, "applyLink", "applyUrl", "sourceUrl", "url"),
        source=_text(payload, "source"),
        responsibilities=_text_list(payload.get("responsibilities")),
        requirements=_text_list(payload.get("requirements")),
        raw_payload=payload,
    )


def _build_jd_text(item: OfferIOJob) -> str:
    sections: list[str] = []
    if item.department:
        sections.append(f"Department: {item.department}")
    if item.responsibilities:
        sections.append("Responsibilities:\n" + "\n".join(f"- {value}" for value in item.responsibilities))
    if item.requirements:
        sections.append("Requirements:\n" + "\n".join(f"- {value}" for value in item.requirements))
    return "\n\n".join(sections)


def _extract_skills(payload: dict[str, Any], category: str | None) -> list[str]:
    values = _text_list(payload.get("skills"))
    if category and category not in values:
        values.append(category)
    return values


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _text(payload: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            return "、".join(str(item).strip() for item in value if str(item).strip()) or None
        cleaned = str(value).strip()
        if cleaned:
            return cleaned
    return None


def _required_text(payload: dict[str, Any], *keys: str) -> str:
    value = _text(payload, *keys)
    if not value:
        raise ValueError(f"OfferIO item missing required field: {'/'.join(keys)}")
    return value


def _int_text(payload: dict[str, Any], *keys: str) -> int:
    value = _text(payload, *keys)
    if not value:
        return 0
    try:
        return int(float(value))
    except ValueError:
        return 0


def _int_payload(payload: dict[str, Any], key: str, fallback: int) -> int:
    try:
        return int(payload.get(key, fallback))
    except (TypeError, ValueError):
        return fallback


def _list_from(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    values = payload.get(key, [])
    if not isinstance(values, list):
        return []
    return [item for item in values if isinstance(item, dict)]


def _text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []
