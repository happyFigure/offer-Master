from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.agent_runtime.context.capability_catalog import CapabilityCatalog
from app.agent_runtime.tool_registry import DATABASE_COMPANY_LIST_TOOL, LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL, LOCAL_JOB_SOURCE_OVERVIEW_TOOL, OFFERIO_COMPANY_JOBS_TOOL
from app.agent_runtime.understanding.schemas import IntentFrame


@dataclass(frozen=True)
class ContextPack:
    intent: str
    confidence: float
    needs_external_info: bool
    risk_level: str
    entities: dict[str, Any]
    allowed_capabilities: list[str]
    excluded_capabilities: list[str]
    capability_metadata: list[dict[str, Any]]
    loaded_capabilities: list[str] = field(default_factory=list)
    memory_policy: str = "default"
    notes: list[str] = field(default_factory=list)
    sync_policy: dict[str, Any] = field(default_factory=dict)
    filesystem_operation: str | None = None
    operation_intent: dict[str, Any] = field(default_factory=dict)
    required_capability: str | None = None
    missing_required_capability: bool = False
    capability_selection_mode: str = "model_driven"

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "needs_external_info": self.needs_external_info,
            "risk_level": self.risk_level,
            "entities": self.entities,
            "allowed_capabilities": list(self.allowed_capabilities),
            "excluded_capabilities": list(self.excluded_capabilities),
            "capability_metadata": list(self.capability_metadata),
            "loaded_capabilities": list(self.loaded_capabilities),
            "memory_policy": self.memory_policy,
            "notes": list(self.notes),
            "sync_policy": dict(self.sync_policy),
            "filesystem_operation": self.filesystem_operation,
            "operation_intent": dict(self.operation_intent),
            "required_capability": self.required_capability,
            "missing_required_capability": self.missing_required_capability,
            "capability_selection_mode": self.capability_selection_mode,
        }


class ContextPackBuilder:
    def __init__(self, capability_catalog: CapabilityCatalog) -> None:
        self._capability_catalog = capability_catalog

    def build(self, frame: IntentFrame, *, source_type: str = "agent_chat") -> ContextPack:
        # The main model receives the source-eligible catalog. Intent is
        # retained as observation metadata and must not remove tools before
        # the model has seen the conversation history.
        source_allowed = self._capability_catalog.model_visible_for_source(source_type)
        required_capability = str(frame.required_capability or "").strip() or None
        if required_capability is None:
            allowed = source_allowed
            missing_required_capability = False
        else:
            allowed = [capability for capability in source_allowed if capability.name == required_capability]
            # An explicit source is a hard user constraint. Do not replace an
            # unavailable capability with a similar local tool.
            missing_required_capability = not allowed
        excluded = [capability for capability in source_allowed if capability not in allowed]
        memory_policy = _memory_policy_for_intent(frame.intent)
        sync_policy = _sync_policy_for_intent(frame.intent, [capability.name for capability in allowed])
        return ContextPack(
            intent=frame.intent,
            confidence=frame.confidence,
            needs_external_info=frame.needs_external_info,
            risk_level=frame.risk_level,
            entities=frame.entities.model_dump(mode="json"),
            allowed_capabilities=[capability.name for capability in allowed],
            excluded_capabilities=[capability.name for capability in excluded],
            capability_metadata=[capability.to_metadata_dict() for capability in allowed],
            loaded_capabilities=[],
            memory_policy=memory_policy,
            notes=_notes_for_intent(frame.intent),
            sync_policy=sync_policy,
            filesystem_operation=frame.filesystem_operation,
            operation_intent=dict(frame.operation_intent),
            required_capability=required_capability,
            missing_required_capability=missing_required_capability,
            capability_selection_mode="model_driven",
        )


def _memory_policy_for_intent(intent: str) -> str:
    if intent in {
        "campus_recruiting_search",
        "local_company_database_overview",
        "local_company_database_list",
        "local_job_source_overview",
        "company_board_overview",
        "offerio_company_jobs_sync",
        "application_entry_discovery",
        "filesystem_operation",
    }:
        return "do_not_load_resume_full_text"
    if intent in {"job_match_analysis", "resume_tailoring"}:
        return "load_resume_summary_only"
    if intent == "memory_lookup":
        return "load_relevant_memory"
    return "default"


def _notes_for_intent(intent: str) -> list[str]:
    if intent == "campus_recruiting_search":
        return ["public_web_search_context_only", "do_not_open_application_page", "do_not_load_resume_full_text"]
    if intent == "offerio_company_jobs_sync":
        return ["offerio_company_jobs_source_available", "keep_page_size_50", "do_not_load_resume_full_text"]
    if intent == "local_company_database_overview":
        return ["read_only_local_company_database", "do_not_modify_database", "do_not_load_resume_full_text"]
    if intent == "local_company_database_list":
        return ["read_only_local_company_list", "do_not_modify_database", "do_not_load_resume_full_text"]
    if intent == "local_job_source_overview":
        return ["read_only_local_job_sources", "include_offerio_job_board_totals", "do_not_modify_database"]
    if intent == "company_board_overview":
        return ["company_count_means_company_exhibition", "read_only_external_company_board", "do_not_fallback_to_legacy_company_tables"]
    if intent == "application_entry_discovery":
        return ["stop_before_final_submission", "do_not_upload_unselected_resume"]
    if intent == "filesystem_operation":
        return ["use_active_file_context", "do_not_guess_missing_paths", "confirm_high_risk_file_writes"]
    return []


def _sync_policy_for_intent(intent: str, allowed_capabilities: list[str]) -> dict[str, Any]:
    if intent == "offerio_company_jobs_sync" and OFFERIO_COMPANY_JOBS_TOOL in allowed_capabilities:
        return {"page_size": 50, "allow_multiple_pages": True, "default_limit": 1000}
    if intent == "local_company_database_overview" and LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL in allowed_capabilities:
        return {"read_only": True, "sample_limit": 10}
    if intent == "local_company_database_list" and DATABASE_COMPANY_LIST_TOOL in allowed_capabilities:
        return {"read_only": True, "default_limit": 20}
    if intent == "local_job_source_overview" and LOCAL_JOB_SOURCE_OVERVIEW_TOOL in allowed_capabilities:
        return {"read_only": True, "sample_limit": 10, "include_external_job_board": True}
    if intent == "company_board_overview" and LOCAL_JOB_SOURCE_OVERVIEW_TOOL in allowed_capabilities:
        return {"read_only": True, "mode": "company_board_count", "company_board": "offerio_company_openings"}
    return {}
