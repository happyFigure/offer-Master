import { apiRequest, toQueryString } from "./client";
import type { ApplicationBoardItem, ApplicationCreateFromJobInput, ApplicationNotification, ApplicationStatus, ApplicationUpdateInput } from "../types/jobs";


interface ApplicationListResponse {
  items: ApplicationBoardItem[];
}

interface ApplicationNotificationListResponse {
  items: ApplicationNotification[];
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

export async function listApplicationNotifications(limit = 100): Promise<ApplicationNotification[]> {
  const query = toQueryString({ limit });
  const response = await apiRequest<ApplicationNotificationListResponse>(`/api/v1/applications/notification-candidates${query}`);
  return response.items;
}

export async function confirmApplicationNotification(eventId: string, to_status?: ApplicationStatus): Promise<ApplicationNotification> {
  return apiRequest<ApplicationNotification>(`/api/v1/applications/notification-candidates/${eventId}/confirm`, {
    method: "POST",
    body: JSON.stringify({ to_status: to_status ?? null }),
  });
}

export async function rejectApplicationNotification(eventId: string): Promise<ApplicationNotification> {
  return apiRequest<ApplicationNotification>(`/api/v1/applications/notification-candidates/${eventId}/reject`, {
    method: "POST",
  });
}
