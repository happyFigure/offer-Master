from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentState:
    session_id: str
    workflow_run_id: str
    agent_run_id: str
    user_message: str
    current_step: str
    latest_summary_id: str | None = None
    loaded_session_history_ids: list[str] = field(default_factory=list)
    loaded_memory_ids: list[str] = field(default_factory=list)
    loaded_skill_ids: list[str] = field(default_factory=list)
    tool_call_ids: list[str] = field(default_factory=list)
    requested_tool_name: str | None = None
    source_type: str = "agent_chat"
    need_compaction: bool = False
    token_estimate: int = 0
    guard_result: dict[str, Any] | None = None
    approval_request_id: str | None = None
    final_response: str | None = None
    response_mode: str = "deterministic_stub"
    llm_messages: list[dict[str, Any]] = field(default_factory=list)
    context_metadata: dict[str, Any] = field(default_factory=dict)

    def with_updates(self, **changes: Any) -> AgentState:
        data = self.to_checkpoint_state()
        data.update(changes)
        return AgentState.from_checkpoint_state(data)

    def to_checkpoint_state(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "workflow_run_id": self.workflow_run_id,
            "agent_run_id": self.agent_run_id,
            "user_message": self.user_message,
            "current_step": self.current_step,
            "latest_summary_id": self.latest_summary_id,
            "loaded_session_history_ids": list(self.loaded_session_history_ids),
            "loaded_memory_ids": list(self.loaded_memory_ids),
            "loaded_skill_ids": list(self.loaded_skill_ids),
            "tool_call_ids": list(self.tool_call_ids),
            "requested_tool_name": self.requested_tool_name,
            "source_type": self.source_type,
            "need_compaction": self.need_compaction,
            "token_estimate": self.token_estimate,
            "guard_result": self.guard_result,
            "approval_request_id": self.approval_request_id,
            "final_response": self.final_response,
            "response_mode": self.response_mode,
            "llm_messages": list(self.llm_messages),
            "context_metadata": dict(self.context_metadata),
        }

    @classmethod
    def from_checkpoint_state(cls, state: dict[str, Any]) -> AgentState:
        return cls(
            session_id=str(state["session_id"]),
            workflow_run_id=str(state["workflow_run_id"]),
            agent_run_id=str(state["agent_run_id"]),
            user_message=str(state.get("user_message") or ""),
            current_step=str(state.get("current_step") or "build_context"),
            latest_summary_id=state.get("latest_summary_id"),
            loaded_session_history_ids=list(state.get("loaded_session_history_ids") or []),
            loaded_memory_ids=list(state.get("loaded_memory_ids") or []),
            loaded_skill_ids=list(state.get("loaded_skill_ids") or []),
            tool_call_ids=list(state.get("tool_call_ids") or []),
            requested_tool_name=state.get("requested_tool_name"),
            source_type=str(state.get("source_type") or "agent_chat"),
            need_compaction=bool(state.get("need_compaction") or False),
            token_estimate=int(state.get("token_estimate") or 0),
            guard_result=state.get("guard_result"),
            approval_request_id=state.get("approval_request_id"),
            final_response=state.get("final_response"),
            response_mode=str(state.get("response_mode") or "deterministic_stub"),
            llm_messages=list(state.get("llm_messages") or []),
            context_metadata=dict(state.get("context_metadata") or {}),
        )
