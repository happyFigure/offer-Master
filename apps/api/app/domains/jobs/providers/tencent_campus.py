from __future__ import annotations

from html import unescape
from typing import Any

import httpx

from app.domains.jobs.providers.base import JobSearchQuery, ProviderStatus, RawJob
from app.domains.jobs.schemas import JobImportDraft


class TencentCampusProvider:
    name = "tencent_campus"
    source_type = "official_campus_api"
    search_url = "https://join.qq.com/api/v1/position/searchPosition"
    detail_url = "https://join.qq.com/api/v1/jobDetails/getJobDetailsByPostId"

    _headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Referer": "https://join.qq.com/post.html",
        "User-Agent": "Mozilla/5.0",
    }
    _campus_terms = ("校招", "校园", "应届", "实习")
    _target_terms = (
        "java",
        "后端",
        "后台",
        "服务端",
        "开发",
        "研发",
        "算法",
        "agent",
        "ai",
        "大模型",
        "llm",
        "rag",
    )
    _excluded_title_terms = ("产品经理", "运营", "销售", "市场", "设计")
    _skill_terms = (
        ("Java", "java"),
        ("Go", "go"),
        ("C/C++", "c/c++"),
        ("C++", "c++"),
        ("Python", "python"),
        ("MySQL", "mysql"),
        ("SQL", "sql"),
        ("AI", "ai"),
        ("Agent", "agent"),
        ("大模型", "大模型"),
        ("LLM", "llm"),
        ("RAG", "rag"),
    )

    def __init__(self, client: Any | None = None, timeout_seconds: int = 15) -> None:
        self._client = client or httpx.Client()
        self._timeout_seconds = timeout_seconds

    def health_check(self) -> ProviderStatus:
        try:
            self.search(JobSearchQuery(keyword="后端", limit=1, job_type="campus"))
        except Exception as exc:  # pragma: no cover - defensive boundary for runtime checks
            return ProviderStatus(name=self.name, available=False, message=str(exc))
        return ProviderStatus(name=self.name, available=True)

    def search(self, query: JobSearchQuery) -> list[RawJob]:
        keyword = query.keyword or "后端"
        body = {
            "keyword": keyword,
            "pageIndex": 1,
            "pageSize": self._normalize_limit(query.limit),
            "lang": "zh-cn",
        }
        response = self._client.post(
            self.search_url,
            json=body,
            headers=self._headers,
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        positions = self._positions(payload)

        raw_jobs: list[RawJob] = []
        for summary in positions:
            if not self._is_target_summary(summary, query):
                continue
            detail = self._fetch_detail(summary)
            if detail is None:
                continue
            raw_jobs.append(
                RawJob(source=self.name, payload={"summary": summary, "detail": detail})
            )
            if len(raw_jobs) >= query.limit:
                break
        return raw_jobs

    def normalize(self, raw_job: RawJob) -> JobImportDraft:
        payload = raw_job.payload
        summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
        detail = payload.get("detail", {}) if isinstance(payload, dict) else {}
        if not isinstance(summary, dict) or not isinstance(detail, dict):
            raise ValueError("Tencent campus payload must contain summary and detail")

        post_id = self._required_text(summary.get("postId") or detail.get("postId"), "postId")
        title = self._required_text(detail.get("title") or summary.get("positionTitle"), "title")
        job_type = self._optional_text(
            detail.get("recruitLabelName")
            or summary.get("recruitLabelName")
            or summary.get("projectName")
        )
        city = self._city(detail, summary)
        jd_text = self._jd_text(detail)

        return JobImportDraft(
            company_name="腾讯",
            company_country="中国",
            company_city=city,
            title=title,
            city=city,
            source=self.name,
            source_job_id=post_id,
            source_url=self._source_url(summary, detail, post_id),
            job_type=job_type,
            jd_text=jd_text,
            skills=self._skills(" ".join([title, jd_text or "", str(summary)])),
            raw_payload=payload,
        )

    def _fetch_detail(self, summary: dict[str, Any]) -> dict[str, Any] | None:
        post_id = self._optional_text(summary.get("postId"))
        if post_id is None:
            return None
        response = self._client.get(
            self.detail_url,
            params={"postId": post_id, "lang": "zh-cn"},
            headers=self._headers,
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or payload.get("status") != 0:
            return None
        detail = payload.get("data")
        return detail if isinstance(detail, dict) else None

    @staticmethod
    def _positions(payload: Any) -> list[dict[str, Any]]:
        if not isinstance(payload, dict) or payload.get("status") != 0:
            return []
        data = payload.get("data")
        if not isinstance(data, dict):
            return []
        positions = data.get("positionList", [])
        return [position for position in positions if isinstance(position, dict)]

    def _is_target_summary(self, summary: dict[str, Any], query: JobSearchQuery) -> bool:
        text = " ".join(
            str(summary.get(key) or "")
            for key in ("positionTitle", "projectName", "recruitLabelName", "workCities", "bgs")
        )
        lowered = text.lower()
        title = str(summary.get("positionTitle") or "")
        if not any(term in text for term in self._campus_terms):
            return False
        if any(term in title for term in self._excluded_title_terms):
            return False
        if query.city and query.city not in text:
            return False
        return any(term in lowered or term in text for term in self._target_terms)

    @staticmethod
    def _normalize_limit(limit: int) -> int:
        return min(max(limit, 1), 100)

    @classmethod
    def _required_text(cls, value: Any, field_name: str) -> str:
        text = cls._optional_text(value)
        if not text:
            raise ValueError(f"Tencent campus payload is missing required field: {field_name}")
        return text

    @staticmethod
    def _optional_text(value: Any) -> str | None:
        if value is None:
            return None
        text = " ".join(unescape(str(value)).split())
        return text or None

    @classmethod
    def _city(cls, detail: dict[str, Any], summary: dict[str, Any]) -> str | None:
        detail_cities = detail.get("workCityList")
        if isinstance(detail_cities, list):
            values = [cls._optional_text(city) for city in detail_cities]
            cities = [city for city in values if city]
            if cities:
                return ", ".join(cities)
        return cls._optional_text(summary.get("workCities"))

    @classmethod
    def _jd_text(cls, detail: dict[str, Any]) -> str | None:
        sections = [
            ("岗位描述", detail.get("desc")),
            ("岗位要求", detail.get("request")),
            ("加分项", detail.get("graduateBonus") or detail.get("internBonus")),
        ]
        parts = []
        for label, value in sections:
            text = cls._optional_text(value)
            if text:
                parts.append(f"{label}: {text}")
        return "\n".join(parts) or None

    @staticmethod
    def _source_url(summary: dict[str, Any], detail: dict[str, Any], post_id: str) -> str:
        project_id = summary.get("projectId") or detail.get("projectId") or detail.get("tid")
        detail_id = detail.get("id") or summary.get("position") or summary.get("id")
        if project_id and detail_id:
            return f"https://join.qq.com/post_detail.html?pid={project_id}&id={detail_id}&postId={post_id}"
        return f"https://join.qq.com/post_detail.html?postId={post_id}"

    @classmethod
    def _skills(cls, text: str) -> list[str]:
        lowered = text.lower()
        skills: list[str] = []
        for label, needle in cls._skill_terms:
            if label == "C++" and "C/C++" in skills:
                continue
            if needle in lowered and label not in skills:
                skills.append(label)
        return skills
