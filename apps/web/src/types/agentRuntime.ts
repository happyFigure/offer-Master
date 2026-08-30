export type AgentRuntimeMemberStatus = "active" | "standby" | "offline" | "disabled";

export type AgentRuntimeMemberKind = "local_runtime" | "external_agent";

export interface AgentRuntimeHealth {
  status: "healthy" | "unreachable" | "not_configured" | string;
  label: string;
  detail: string | null;
  checked: boolean;
  url?: string;
}

export interface AgentRuntimeMainAgent {
  id: string;
  name: string;
  role: string;
  status: AgentRuntimeMemberStatus;
  description: string;
  health?: AgentRuntimeHealth;
}

export interface AgentRuntimeSummary {
  agent_count: number;
  capability_count: number;
  low_risk_count: number;
  confirmation_required_count: number;
  configured_web_search_provider: string;
}

export interface AgentRuntimeCapability {
  id: string;
  name: string;
  description: string;
  executor_id: string;
  risk_level: "low" | "medium" | "high" | string;
  requires_confirmation: boolean;
  allowed_source_types: string[];
  supported_intents: string[];
  input_fields: string[];
  output_fields: string[];
  candidate_categories: string[];
  candidate_keywords: string[];
  candidate_examples: string[];
  provider: string;
  status: "active" | "standby" | "disabled" | string;
}

export interface AgentRuntimeMember {
  id: string;
  name: string;
  kind: AgentRuntimeMemberKind;
  status: AgentRuntimeMemberStatus;
  role: string;
  description: string;
  health?: AgentRuntimeHealth;
  capabilities: AgentRuntimeCapability[];
}

export interface AgentRuntimePanel {
  main_agent: AgentRuntimeMainAgent;
  summary: AgentRuntimeSummary;
  agents: AgentRuntimeMember[];
  capabilities: AgentRuntimeCapability[];
}
