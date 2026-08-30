export type AgentSessionStatus = "active" | "archived" | "closed";

export type AgentMessageRole = "system" | "user" | "assistant" | "tool_call" | "tool_result";

export type AgentMessageKind =
  | "system_text"
  | "user_text"
  | "assistant_text"
  | "tool_call"
  | "tool_result"
  | "summary_notice"
  | "synthetic_error";

export type AgentMessageVisibilityScope = "user_visible" | "runtime_only" | "internal";

export interface AgentSession {
  id: string;
  title: string | null;
  status: AgentSessionStatus;
  primary_intent: string | null;
  current_agent_run_id: string | null;
  last_context_summary_id: string | null;
  message_count: number;
  created_at: string;
  updated_at: string;
  last_message_at: string | null;
  metadata_json: Record<string, unknown> | null;
}

export interface AgentMessage {
  id: string;
  session_id: string;
  role: AgentMessageRole;
  message_kind: AgentMessageKind;
  agent_id: string | null;
  recipient_agent_id: string | null;
  visibility_scope: AgentMessageVisibilityScope;
  content_text: string | null;
  content_json: Record<string, unknown> | null;
  visible_content_text: string | null;
  content_type: string;
  provenance_kind: string | null;
  agent_run_id: string | null;
  workflow_run_id: string | null;
  tool_call_log_id: string | null;
  parent_message_id: string | null;
  token_estimate: number | null;
  exclude_from_context: boolean;
  compacted_by_summary_id: string | null;
  created_at: string;
  metadata_json: Record<string, unknown> | null;
}

export interface AgentContextSummary {
  id: string;
  session_id: string;
  summary_text: string;
  summary_json: Record<string, unknown> | null;
  covered_message_start_id: string | null;
  covered_message_end_id: string | null;
  first_kept_message_id: string | null;
  previous_summary_id: string | null;
  token_estimate: number | null;
  created_at: string;
  created_by: string | null;
  metadata_json: Record<string, unknown> | null;
}

export interface AgentContextMetadata {
  summary_id?: string | null;
  loaded_session_history_ids?: string[];
  loaded_memory_ids?: string[];
  loaded_skill_ids?: string[];
  token_estimate?: number;
  context_window?: number;
  reserve_tokens?: number;
  keep_recent_tokens?: number;
  need_compaction?: boolean;
  auto_compacted?: boolean;
  auto_compacted_summary_id?: string | null;
  auto_compacted_message_count?: number;
  auto_compaction_error?: string;
  hygiene_synthetic_error_ids?: string[];
  hygiene_excluded_reasons?: string[];
  [key: string]: unknown;
}

export type AgentApprovalStatus = "pending" | "approved" | "rejected" | "expired" | "canceled";

export interface AgentApprovalRequest {
  id: string;
  workflow_run_id: string;
  application_id: string | null;
  action_type: string;
  status: AgentApprovalStatus;
  prompt: string;
  payload: Record<string, unknown> | null;
  decision: string | null;
  decided_at: string | null;
  created_at: string;
  expires_at: string | null;
}

export interface AgentApprovalRequiredPayload {
  approval: AgentApprovalRequest;
  approval_request_id: string;
  workflow_run_id: string;
  tool_name: string;
  reason: string | null;
  user_message: string | null;
  permission_decision: string | null;
  skill_ids: string[];
  context_metadata: AgentContextMetadata;
}

export interface AgentApprovalDecisionInput {
  decision_reason?: string | null;
}

export interface AgentApprovalDecisionResponse {
  approval: AgentApprovalRequest;
  assistant_message: AgentMessage;
  context_metadata: AgentContextMetadata;
}

export interface AgentStreamOuterSessionEvent {
  event_type: string;
  event_label: string;
  session_id: string;
  task_id?: string | null;
  run_id?: string | null;
  status?: string | null;
  summary?: string | null;
  is_resume?: boolean;
  turn_index?: number;
  user_goal?: string | null;
  requires_user_action?: boolean;
  waiting_message?: string | null;
  final_answer?: string | null;
  outer_session?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface AgentStreamToolEvent {
  event_type: string;
  event_label: string;
  session_id: string;
  workflow_run_id?: string | null;
  agent_run_id?: string | null;
  step_index?: number;
  tool_name?: string | null;
  capability?: string | null;
  tool_call_id?: string | null;
  status?: string | null;
  summary?: string | null;
  tool_input_keys?: string[];
  candidate_capabilities?: string[];
  input_preview?: Record<string, unknown> | null;
  result_summary?: Record<string, unknown> | null;
  evidence?: Array<Record<string, unknown>> | null;
  metadata?: Record<string, unknown> | null;
  reflection?: Record<string, unknown> | null;
  suggested_input_patch?: Record<string, unknown> | null;
  [key: string]: unknown;
}

export interface AgentSessionCreateInput {
  title?: string | null;
  primary_intent?: string | null;
  metadata_json?: Record<string, unknown> | null;
}

export interface AgentSessionUpdateInput {
  title?: string | null;
  primary_intent?: string | null;
  metadata_json?: Record<string, unknown> | null;
}

export interface AgentUserMessageInput {
  content_text: string;
  requested_tool_name?: string | null;
  source_type?: string;
  user_confirmed?: boolean;
  tool_input?: Record<string, unknown> | null;
  metadata_json?: Record<string, unknown> | null;
}

export interface AgentChatTurnResponse {
  user_message: AgentMessage;
  assistant_message: AgentMessage;
}

export interface AgentTaskPlanStage {
  step_id: string;
  stage_id: string;
  sequence_index: number;
  title: string;
  objective: string;
  business_action?: string | null;
  allowed_capabilities: string[];
  tool_strategy: Record<string, unknown>;
  ranking_policy: string[];
  capability: string;
  status: string;
  execution_status?: string | null;
  waiting_message?: string | null;
  final_answer_preview?: string | null;
  depends_on: string[];
  received_context?: Record<string, unknown> | null;
  handoff_payload?: Record<string, unknown> | null;
}

export interface AgentTaskPlan {
  task_id: string;
  user_goal: string | null;
  current_stage_id: string | null;
  stages: AgentTaskPlanStage[];
}

export interface AgentCompactInput {
  context_window?: number;
  reserve_tokens?: number;
  keep_recent_tokens?: number;
}

export interface AgentCompactResult {
  summary: AgentContextSummary;
  covered_message_count: number;
  first_kept_message_id: string | null;
  token_estimate_before: number;
  token_estimate_after: number;
  should_compact: boolean;
}
