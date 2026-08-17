export type JobSourceType =
  | "manual_clip"
  | "xiaohongshu_note"
  | "wechat_article"
  | "wechat_account"
  | "university_career_site"
  | "official_career_site"
  | "job_board_visible_page"
  | "official_api";

export type JobSourceFetchMode = "manual_clip" | "public_html" | "mcp_visible_page" | "official_api";

export type JobSourceTrustLevel = "high" | "medium_high" | "medium" | "low_medium" | "low";

export type JobLeadStatus = "unverified" | "pending_review" | "verified" | "converted" | "expired" | "invalid";

export type SourceSyncRunStatus = "running" | "succeeded" | "failed" | "partial";

export type UrlImportRunStatus = "running" | "waiting_user" | "succeeded" | "partial" | "failed_recoverable" | "failed_terminal" | "duplicate";

export type DomainHealthState = "unknown" | "closed" | "open" | "half_open";

export type ToolSuggestedNextAction =
  | "continue_workflow"
  | "retry_same_stage"
  | "retry_with_next_fetcher"
  | "wait_for_cooldown"
  | "request_user_visible_page"
  | "request_manual_paste"
  | "skip_duplicate"
  | "stop_terminal_failure"
  | "enrich_recruiting_signal";

export type ArticleCandidateStatus = "pending" | "parsed" | "skipped" | "needs_visible_page";

export type RecruitingSignalStatus = "needs_job_enrichment" | "job_found" | "no_matching_job" | "expired";

export type RecruitingSignalType = "campus_recruitment_open" | "internship_open" | "info_summary";

export interface UrlImportInput {
  url: string;
  source_id?: string | null;
  source_hint?: JobSourceType | null;
  trust_level?: JobSourceTrustLevel | null;
  force_refresh?: boolean;
}

export interface VisiblePageContentInput {
  visible_text: string;
  title?: string | null;
  final_url?: string | null;
}

export interface ImportUrlAcceptedResponse {
  run_id: string;
  status: UrlImportRunStatus;
  current_stage: string;
  domain_health_state: DomainHealthState;
  message: string;
}

export interface ImportUrlRun {
  id: string;
  workflow_run_id: string;
  source_id: string | null;
  input_url: string;
  normalized_url: string | null;
  normalized_url_hash: string | null;
  source_type: JobSourceType | null;
  domain: string | null;
  fetch_layer: string | null;
  status: UrlImportRunStatus;
  current_stage: string;
  attempt_count: number;
  tool_call_count: number;
  llm_call_count: number;
  error_code: string | null;
  error_message: string | null;
  next_action: ToolSuggestedNextAction | string | null;
  raw_job_lead_id: string | null;
  raw_content_preview: string | null;
  raw_extraction_method: string | null;
  raw_image_count: number | null;
  raw_image_parse_deferred: boolean | null;
  extracted_count: number;
  duplicate_of_run_id: string | null;
  run_metadata: Record<string, unknown> | null;
  started_at: string;
  updated_at: string;
  finished_at: string | null;
}

export interface DomainHealth {
  id: string;
  domain: string;
  tool_name: string;
  state: DomainHealthState;
  failure_count: number;
  success_count: number;
  last_error_code: string | null;
  last_error_message: string | null;
  opened_at: string | null;
  cooldown_until: string | null;
  half_open_probe_count: number;
  created_at: string;
  updated_at: string;
}

export interface ArticleCandidate {
  id: string;
  source_id: string;
  sync_run_id: string | null;
  title: string;
  url: string;
  url_hash: string;
  source_account: string | null;
  published_at: string | null;
  status: ArticleCandidateStatus;
  raw_payload: Record<string, unknown> | null;
  created_at: string;
  updated_at: string;
}

export interface RecruitingSignal {
  id: string;
  source_id: string;
  raw_lead_id: string | null;
  article_candidate_id: string | null;
  signal_hash: string;
  company_name: string;
  normalized_company_name: string;
  signal_type: RecruitingSignalType;
  graduation_year: string | null;
  source_url: string | null;
  original_source: string | null;
  confidence_score: number | null;
  trust_level: JobSourceTrustLevel;
  status: RecruitingSignalStatus;
  raw_payload: Record<string, unknown> | null;
  created_at: string;
  updated_at: string;
}

export interface JobSource {
  id: string;
  name: string;
  source_type: JobSourceType;
  entry_url: string | null;
  enabled: boolean;
  sync_interval_hours: number;
  trust_level: JobSourceTrustLevel;
  fetch_mode: JobSourceFetchMode;
  notes: string | null;
  last_synced_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface JobSourceCreateInput {
  name: string;
  source_type: JobSourceType;
  entry_url?: string | null;
  enabled?: boolean;
  sync_interval_hours: number;
  trust_level: JobSourceTrustLevel;
  fetch_mode: JobSourceFetchMode;
  notes?: string | null;
}

export type JobSourceUpdateInput = Partial<JobSourceCreateInput>;

export interface JobLead {
  id: string;
  source_id: string;
  raw_lead_id: string | null;
  converted_job_id: string | null;
  company_name: string;
  title: string;
  city: string | null;
  job_direction: string | null;
  graduation_year: string | null;
  source_url: string | null;
  apply_url: string | null;
  verified_url: string | null;
  job_type: string | null;
  salary_text: string | null;
  jd_text: string | null;
  skills: string[];
  deadline: string | null;
  confidence_score: number | null;
  trust_level: JobSourceTrustLevel;
  verification_status: JobLeadStatus;
  verification_notes: string | null;
  verified_at: string | null;
  converted_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface JobLeadFilters {
  source_id?: string;
  source_type?: JobSourceType;
  trust_level?: JobSourceTrustLevel;
  verification_status?: JobLeadStatus;
  company?: string;
  job_direction?: string;
  graduation_year?: string;
  keyword?: string;
  limit?: number;
}

export interface JobLeadExtractionInput {
  source_id: string;
  raw_content: string;
  source_url?: string | null;
  content_type?: string;
  sync_run_id?: string | null;
}

export interface RawJobLead {
  id: string;
  source_id: string;
  sync_run_id: string | null;
  source_url: string | null;
  content_hash: string;
  content_type: string;
  status: string;
  created_at: string;
  updated_at: string;
}

export interface JobLeadExtractionResponse {
  raw_created: boolean;
  extracted_count: number;
  leads: JobLead[];
}

export interface JobSourceSyncResponse {
  sync_run_id: string;
  status: SourceSyncRunStatus;
  fetched_count: number;
  extracted_count: number;
  failed_count: number;
  error: string | null;
  raw_leads: RawJobLead[];
  leads: JobLead[];
  article_candidates: ArticleCandidate[];
  recruiting_signals: RecruitingSignal[];
}

export interface JobSummary {
  id: string;
  title: string;
  company: {
    id: string;
    name: string;
  };
  city: string | null;
  source: string;
  source_job_id: string;
  source_url: string | null;
  job_type: string | null;
  skills: string[];
  status: string;
}

export interface JobLeadConversionResponse {
  lead: JobLead;
  job: JobSummary;
  created: boolean;
}

export interface OfferIOCompany {
  name: string;
  company_nature: string | null;
  industry: string | null;
  locations: string | null;
  job_count: number;
  updated_at: string | null;
  raw_payload: Record<string, unknown> | null;
}

export interface OfferIOCompanyOpening {
  id: string;
  company_name: string;
  company_nature: string | null;
  industry: string | null;
  batch: string | null;
  target: string | null;
  location: string | null;
  positions: string | null;
  update_date: string | null;
  deadline: string | null;
  apply_link: string | null;
  has_written_test: string | null;
  raw_payload: Record<string, unknown> | null;
}

export interface OfferIOJob {
  id: string;
  title: string;
  company: string;
  location: string | null;
  category: string | null;
  job_type: string | null;
  publish_date: string | null;
  salary: string | null;
  deadline: string | null;
  department: string | null;
  apply_link: string | null;
  source: string | null;
  responsibilities: string[];
  requirements: string[];
  raw_payload: Record<string, unknown> | null;
}

export interface OfferIOPage<T> {
  items: T[];
  page: number;
  page_size: number;
  total: number;
  total_pages: number;
}

export interface JobImportDraft {
  company_name: string;
  company_website_url?: string | null;
  company_industry?: string | null;
  company_city?: string | null;
  company_country?: string | null;
  title: string;
  city?: string | null;
  source: string;
  source_job_id: string;
  source_url?: string | null;
  job_type?: string | null;
  salary_text?: string | null;
  jd_text?: string | null;
  skills?: string[];
  date_posted?: string | null;
  match_score?: number | null;
  status?: string;
  raw_payload?: Record<string, unknown> | null;
}

export type ApplicationStatus =
  | "evaluating"
  | "preparing"
  | "applied"
  | "assessment"
  | "written_test"
  | "interview_1"
  | "interview_2"
  | "hr_interview"
  | "offer"
  | "rejected"
  | "withdrawn";

export interface ApplicationBoardItem {
  id: string;
  job_id: string;
  status: ApplicationStatus;
  priority: string;
  channel: string | null;
  applied_at: string | null;
  next_follow_up_at: string | null;
  notes: string | null;
  created_at: string;
  updated_at: string;
  job: JobSummary;
}

export interface ApplicationCreateFromJobInput {
  job: JobImportDraft;
  status?: ApplicationStatus;
  priority?: string;
  channel?: string | null;
  notes?: string | null;
}

export interface ApplicationUpdateInput {
  status?: ApplicationStatus;
  priority?: string;
  channel?: string | null;
  notes?: string | null;
  actor?: string;
  source?: string;
}
