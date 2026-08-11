from __future__ import annotations

import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any


def _ensure_api_import_path() -> None:
    api_path = Path(__file__).resolve().parents[3] / "api"
    api_path_text = str(api_path)
    if api_path_text not in sys.path:
        sys.path.insert(0, api_path_text)


def run_scheduled_sync_once(
    *,
    session_factory: Callable[[], Any] | None = None,
    university_provider: Any | None = None,
    social_provider: Any | None = None,
    now: datetime | None = None,
    limit_per_source: int = 20,
) -> dict[str, int]:
    _ensure_api_import_path()

    from app.agent_runtime.workflows.job_discovery import (
        DueJobSourceSyncCommand,
        run_due_job_source_syncs,
    )
    from app.db.session import SessionLocal
    from app.domains.jobs.providers.social_lead import SocialLeadImportProvider
    from app.domains.jobs.providers.university_career import UniversityCareerProvider
    from app.domains.jobs.repository import (
        JobLeadRepository,
        JobSourceRepository,
        RawJobLeadRepository,
        SourceSyncRunRepository,
    )
    from app.domains.jobs.service import JobLeadService
    from app.infrastructure.llm.job_lead_extractor import LLMJobLeadExtractor

    runtime_session_factory = session_factory or SessionLocal
    runtime_university_provider = university_provider or UniversityCareerProvider()
    runtime_social_provider = social_provider or SocialLeadImportProvider(
        extractor=LLMJobLeadExtractor()
    )

    with runtime_session_factory() as session:
        lead_service = JobLeadService(
            sources=JobSourceRepository(session),
            sync_runs=SourceSyncRunRepository(session),
            raw_leads=RawJobLeadRepository(session),
            leads=JobLeadRepository(session),
        )
        result = run_due_job_source_syncs(
            DueJobSourceSyncCommand(limit_per_source=limit_per_source, now=now),
            lead_service=lead_service,
            university_provider=runtime_university_provider,
            social_provider=runtime_social_provider,
        )
        session.commit()

    return {
        "processed": len(result.processed_source_ids),
        "succeeded": result.succeeded_count,
        "failed": result.failed_count,
        "skipped": result.skipped_count,
    }
