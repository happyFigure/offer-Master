from __future__ import annotations

from app.agent_runtime.contracts.tasks.browser_application import BrowserTaskEnvelope
from app.agent_runtime.external_tasks.schemas import FindApplyEntryTaskEnvelope


class HandoffPayloadBuilder:
    def build_browser_application_task(
        self,
        *,
        find_apply_entry_envelope: FindApplyEntryTaskEnvelope,
    ) -> BrowserTaskEnvelope:
        return BrowserTaskEnvelope.from_find_apply_entry_task(find_apply_entry_envelope)
