import { apiRequest } from "./client";
import type { DomainHealth, ImportUrlAcceptedResponse, ImportUrlRun, UrlImportInput, VisiblePageContentInput } from "../types/jobs";

interface DomainHealthListResponse {
  items: DomainHealth[];
}

export async function importJobLeadsFromUrl(input: UrlImportInput): Promise<ImportUrlAcceptedResponse> {
  return apiRequest<ImportUrlAcceptedResponse>("/api/v1/job-leads/import-url", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export async function pollUrlImportRun(runId: string): Promise<ImportUrlRun> {
  return apiRequest<ImportUrlRun>(`/api/v1/job-leads/import-runs/${runId}`);
}

export async function submitVisiblePageContent(runId: string, input: VisiblePageContentInput): Promise<ImportUrlRun> {
  return apiRequest<ImportUrlRun>(`/api/v1/job-leads/import-runs/${encodeURIComponent(runId)}/visible-page-content`, {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export async function listDomainHealth(): Promise<DomainHealth[]> {
  const response = await apiRequest<DomainHealthListResponse>("/api/v1/tool-health/domains");
  return response.items;
}

export async function listDomainHealthByDomain(domain: string): Promise<DomainHealth[]> {
  const response = await apiRequest<DomainHealthListResponse>(`/api/v1/tool-health/domains/${encodeURIComponent(domain)}`);
  return response.items;
}
