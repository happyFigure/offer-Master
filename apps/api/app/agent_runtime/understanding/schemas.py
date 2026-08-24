from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


AgentIntent = Literal[
    "normal_chat",
    "memory_lookup",
    "campus_recruiting_search",
    "local_company_database_overview",
    "local_job_source_overview",
    "offerio_company_jobs_sync",
    "application_entry_discovery",
    "job_match_analysis",
    "resume_tailoring",
    "external_agent_task",
]

RiskLevel = Literal["low", "medium", "high", "critical"]


class EntityFrame(BaseModel):
    model_config = ConfigDict(extra="ignore")

    company_names: List[str] = Field(default_factory=list)
    job_titles: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)
    source_names: List[str] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)
    job_ids: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    time_range: Optional[str] = None

    @field_validator("company_names", "job_titles", "locations", "source_names", "urls", "job_ids", "keywords", mode="after")
    @classmethod
    def strip_string_items(cls, value: List[str]) -> List[str]:
        cleaned: List[str] = []
        for item in value:
            text = str(item).strip()
            if text and text not in cleaned:
                cleaned.append(text)
        return cleaned

    @field_validator("time_range", mode="after")
    @classmethod
    def strip_time_range(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        stripped = str(value).strip()
        return stripped or None


class IntentFrame(BaseModel):
    model_config = ConfigDict(extra="ignore")

    intent: AgentIntent = "normal_chat"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_external_info: bool = False
    risk_level: RiskLevel = "low"
    entities: EntityFrame = Field(default_factory=EntityFrame)
    candidate_intents: List[str] = Field(default_factory=list)
    reason: Optional[str] = None

    @field_validator("candidate_intents", mode="after")
    @classmethod
    def strip_candidate_intents(cls, value: List[str]) -> List[str]:
        cleaned: List[str] = []
        for item in value:
            text = str(item).strip()
            if text and text not in cleaned:
                cleaned.append(text)
        return cleaned


def fallback_intent_frame(reason: str) -> IntentFrame:
    return IntentFrame(
        intent="normal_chat",
        confidence=0.0,
        needs_external_info=False,
        risk_level="low",
        reason=reason,
    )
