export type AgentSkillStatus = "active" | "archived";

export type AgentSkillStorageType = "markdown_file";

export type AgentSkillAvailabilityState = "available" | "partial" | "unavailable" | "disabled";

export type AgentSkillDependencyState = AgentSkillAvailabilityState;

export type AgentSkillSecurityRiskLevel = "low" | "medium" | "high";

export type AgentSkillAutoTriggerState = "enabled" | "manual_only" | "disabled";

export interface AgentSkillResources {
  scripts?: string[];
  references?: string[];
  assets?: string[];
  agents?: string[];
}

export interface AgentSkillMetadata {
  import_source_path?: string;
  source_types?: string[];
  required_tools?: string[];
  allowed_tools?: string[];
  ask_tools?: string[];
  disallowed_tools?: string[];
  compatibility?: string[];
  resources?: AgentSkillResources;
  version_hash?: string;
  description_quality_score?: number;
  auto_trigger_state?: AgentSkillAutoTriggerState;
  security_risk_level?: AgentSkillSecurityRiskLevel;
  import_warnings?: string[];
  blocking_errors?: string[];
  permission_notice?: string;
  availability_state?: AgentSkillAvailabilityState;
  tool_dependency_state?: AgentSkillDependencyState;
  available_required_tools?: string[];
  missing_required_tools?: string[];
  missing_optional_tools?: string[];
  tool_dependency_checked_at?: string;
  [key: string]: unknown;
}

export interface AgentSkill {
  id: string;
  name: string;
  title: string;
  description: string;
  category: string;
  storage_type: AgentSkillStorageType;
  file_path: string | null;
  status: AgentSkillStatus;
  protected: boolean;
  pinned: boolean;
  created_by: string;
  created_at: string;
  updated_at: string;
  metadata_json: AgentSkillMetadata | null;
}

export interface AgentSkillDocument {
  skill: AgentSkill;
  content: string;
  version_hash: string;
}

export interface AgentSkillImportInput {
  source_path: string;
  category?: string;
  protected?: boolean;
  pinned?: boolean;
  created_by?: string;
  metadata_json?: Record<string, unknown> | null;
}

export interface AgentSkillUsage {
  id: string;
  skill_id: string;
  use_count: number;
  view_count: number;
  patch_count: number;
  success_count: number;
  failure_count: number;
  last_used_at: string | null;
  last_viewed_at: string | null;
  last_patched_at: string | null;
  last_success_at: string | null;
  last_failure_at: string | null;
  state: AgentSkillStatus;
  archived_at: string | null;
  metadata_json: Record<string, unknown> | null;
}
