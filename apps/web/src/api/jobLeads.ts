import { apiRequest, toQueryString } from "./client";
import type {
  JobLead,
  JobLeadConversionResponse,
  JobLeadExtractionInput,
  JobLeadExtractionResponse,
  JobLeadFilters,
  JobLeadStatus,
} from "../types/jobs";

interface JobLeadListResponse {
  items: JobLead[];
}

export async function listJobLeads(filters: JobLeadFilters = {}): Promise<JobLead[]> {
  const query = toQueryString({
    ...filters,
    limit: filters.limit ?? 80,
  });
  const response = await apiRequest<JobLeadListResponse>(`/api/v1/job-leads${query}`);
  return response.items;
}

export async function extractJobLeads(input: JobLeadExtractionInput): Promise<JobLeadExtractionResponse> {
  return apiRequest<JobLeadExtractionResponse>("/api/v1/job-leads/extract", {
    method: "POST",
    body: JSON.stringify({
      ...input,
      content_type: input.content_type ?? "text/plain",
    }),
  });
}

export async function verifyJobLead(leadId: string, status: JobLeadStatus): Promise<JobLead> {
  return apiRequest<JobLead>(`/api/v1/job-leads/${leadId}/verify`, {
    method: "POST",
    body: JSON.stringify({ verification_status: status }),
  });
}

export async function verifyAndConvertJobLead(leadId: string): Promise<JobLeadConversionResponse> {
  return apiRequest<JobLeadConversionResponse>(`/api/v1/job-leads/${leadId}/verify-and-convert`, {
    method: "POST",
  });
}
