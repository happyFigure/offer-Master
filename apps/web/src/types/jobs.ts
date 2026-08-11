export type JobSourceType =
  | "manual_clip"
  | "xiaohongshu_note"
  | "wechat_article"
  | "university_career_site"
  | "official_career_site"
  | "job_board_visible_page"
  | "official_api";

export type JobSourceFetchMode = "manual_clip" | "public_html" | "mcp_visible_page" | "official_api";

export type JobSourceTrustLevel = "high" | "medium_high" | "medium" | "low_medium" | "low";

export type JobLeadStatus = "unverified" | "pending_review" | "verified" | "converted" | "expired" | "invalid";

export type SourceSyncRunStatus = "running" | "succeeded" | "failed" | "partial";

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
  leads: JobLead[];
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
