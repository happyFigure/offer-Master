import { apiRequest, toQueryString } from "./client";
import type { ApplicationBoardItem, ApplicationCreateFromJobInput, ApplicationUpdateInput } from "../types/jobs";


interface ApplicationListResponse {
  items: ApplicationBoardItem[];
}

export async function listApplications(limit = 120): Promise<ApplicationBoardItem[]> {
  const query = toQueryString({ limit });
  const response = await apiRequest<ApplicationListResponse>(`/api/v1/applications${query}`);
  return response.items;
}

export async function createApplicationFromJob(input: ApplicationCreateFromJobInput): Promise<ApplicationBoardItem> {
  return apiRequest<ApplicationBoardItem>("/api/v1/applications/from-job", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export async function updateApplication(applicationId: string, input: ApplicationUpdateInput): Promise<ApplicationBoardItem> {
  return apiRequest<ApplicationBoardItem>(`/api/v1/applications/${applicationId}`, {
    method: "PATCH",
    body: JSON.stringify(input),
  });
}
