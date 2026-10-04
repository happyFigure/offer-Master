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
  kind: "tool" | "skill" | "agent" | string;
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
  candidate_use_when: string[];
  candidate_do_not_use_when: string[];
  candidate_positive_examples: string[];
  candidate_negative_examples: string[];
  candidate_required_context_focus: string[];
  candidate_disambiguation_notes: string[];
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

export interface AgentRuntimeMcpIntegration {
  id: string;
  name: string;
  status: "configured" | "registered" | "unavailable" | "credentials_missing" | "not_registered" | "not_configured" | "disabled" | string;
  label: string;
  detail: string;
  enabled: boolean;
  gateway_configured: boolean;
  transport?: string | null;
  registered_tools: string[];
  configured_tools: string[];
  discovered_tools: string[];
}

export interface AgentRuntimePanel {
  main_agent: AgentRuntimeMainAgent;
  summary: AgentRuntimeSummary;
  mcp_integrations: AgentRuntimeMcpIntegration[];
  agents: AgentRuntimeMember[];
  capabilities: AgentRuntimeCapability[];
}
