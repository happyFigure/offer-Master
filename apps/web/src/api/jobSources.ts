import { apiRequest } from "./client";
import type { JobSource, JobSourceCreateInput, JobSourceSyncResponse } from "../types/jobs";

interface JobSourceListResponse {
  items: JobSource[];
}

export async function listJobSources(): Promise<JobSource[]> {
  const response = await apiRequest<JobSourceListResponse>("/api/v1/job-sources");
  return response.items;
}

export async function createJobSource(input: JobSourceCreateInput): Promise<JobSource> {
  return apiRequest<JobSource>("/api/v1/job-sources", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export async function syncJobSource(sourceId: string, limit = 20): Promise<JobSourceSyncResponse> {
  return apiRequest<JobSourceSyncResponse>(`/api/v1/job-sources/${sourceId}/sync`, {
    method: "POST",
    body: JSON.stringify({ limit }),
  });
}
