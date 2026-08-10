from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class WorkflowStarted:
    workflow_run_id: str
    workflow_type: str
    occurred_at: datetime

    event_type: str = "WorkflowStarted"


@dataclass(frozen=True)
class AutomationWaitingForUser:
    workflow_run_id: str
    approval_request_id: str
    action_type: str
    occurred_at: datetime

    event_type: str = "AutomationWaitingForUser"


@dataclass(frozen=True)
class WorkflowCheckpointSaved:
    workflow_run_id: str
    checkpoint_key: str
    occurred_at: datetime

    event_type: str = "WorkflowCheckpointSaved"
