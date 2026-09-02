import { API_BASE_URL, ApiError, apiRequest, toQueryString } from "./client";
import type {
  AgentApprovalDecisionInput,
  AgentApprovalDecisionResponse,
  AgentApprovalRequiredPayload,
  AgentChatTurnResponse,
  AgentMessage,
  AgentSession,
  AgentSessionCreateInput,
  AgentSessionUpdateInput,
  AgentStreamOuterSessionEvent,
  AgentStreamToolEvent,
  AgentTaskPlan,
  AgentUserMessageInput,
} from "../types/agent";

interface AgentSessionListResponse {
  items: AgentSession[];
}

interface AgentMessageListResponse {
  items: AgentMessage[];
}

interface AgentStreamHandlers {
  onUserMessage?: (message: AgentMessage) => void;
  onToken?: (content: string) => void;
  onApprovalRequired?: (payload: AgentApprovalRequiredPayload) => void;
  onOuterSessionEvent?: (payload: AgentStreamOuterSessionEvent) => void;
  onToolEvent?: (payload: AgentStreamToolEvent) => void;
  onDone?: (assistantMessage: AgentMessage) => void;
  onError?: (message: string) => void;
}

export async function createAgentSession(input: AgentSessionCreateInput): Promise<AgentSession> {
  return apiRequest<AgentSession>("/api/v1/agent/sessions", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export async function listAgentSessions(limit = 20, offset = 0): Promise<AgentSession[]> {
  const query = toQueryString({ limit, offset });
  const response = await apiRequest<AgentSessionListResponse>(`/api/v1/agent/sessions${query}`);
  return response.items;
}

export async function updateAgentSession(sessionId: string, input: AgentSessionUpdateInput): Promise<AgentSession> {
  return apiRequest<AgentSession>(`/api/v1/agent/sessions/${sessionId}`, {
    method: "PATCH",
    body: JSON.stringify(input),
  });
}

export async function deleteAgentSession(sessionId: string): Promise<void> {
  await apiRequest<void>(`/api/v1/agent/sessions/${sessionId}`, {
    method: "DELETE",
  });
}

export async function getAgentMessages(sessionId: string, limit = 100): Promise<AgentMessage[]> {
  const query = toQueryString({ limit });
  const response = await apiRequest<AgentMessageListResponse>(`/api/v1/agent/sessions/${sessionId}/messages${query}`);
  return response.items;
}

export async function sendAgentMessage(sessionId: string, input: AgentUserMessageInput): Promise<AgentChatTurnResponse> {
  return apiRequest<AgentChatTurnResponse>(`/api/v1/agent/sessions/${sessionId}/messages`, {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export async function getAgentTaskPlan(taskId: string): Promise<AgentTaskPlan> {
  return apiRequest<AgentTaskPlan>(`/api/v1/agent/tasks/${taskId}/plan`);
}

export async function approveAgentApproval(approvalRequestId: string, input: AgentApprovalDecisionInput = {}): Promise<AgentApprovalDecisionResponse> {
  return apiRequest<AgentApprovalDecisionResponse>(`/api/v1/agent/approvals/${approvalRequestId}/approve`, {
    method: "POST",
    body: JSON.stringify(input),
    timeoutMs: 120_000,
  });
}

export async function rejectAgentApproval(approvalRequestId: string, input: AgentApprovalDecisionInput = {}): Promise<AgentApprovalDecisionResponse> {
  return apiRequest<AgentApprovalDecisionResponse>(`/api/v1/agent/approvals/${approvalRequestId}/reject`, {
    method: "POST",
    body: JSON.stringify(input),
    timeoutMs: 120_000,
  });
}

export async function streamAgentMessage(sessionId: string, input: AgentUserMessageInput, handlers: AgentStreamHandlers = {}): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/v1/agent/sessions/${sessionId}/messages/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });

  if (!response.ok || !response.body) {
    throw new ApiError(`Agent stream request failed with HTTP ${response.status}`, response.status);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";
    parts.forEach((part) => dispatchAgentStreamEvent(part, handlers));
  }

  buffer += decoder.decode();
  if (buffer.trim()) {
    dispatchAgentStreamEvent(buffer, handlers);
  }
}

function dispatchAgentStreamEvent(rawEvent: string, handlers: AgentStreamHandlers): void {
  const lines = rawEvent.split("\n");
  const eventName = lines
    .find((line) => line.startsWith("event:"))
    ?.replace("event:", "")
    .trim();
  const dataLine = lines.find((line) => line.startsWith("data:"));
  if (!eventName || !dataLine) {
    return;
  }

  const data = JSON.parse(dataLine.replace("data:", "").trim()) as Record<string, unknown>;
  if (eventName === "user_message" && isAgentMessage(data.message)) {
    handlers.onUserMessage?.(data.message);
  }
  if (eventName === "token" && typeof data.content === "string") {
    handlers.onToken?.(data.content);
  }
  if (eventName === "approval_required" && isAgentApprovalRequiredPayload(data)) {
    handlers.onApprovalRequired?.(data);
  }
  if (eventName === "outer_session_event" && isAgentStreamOuterSessionEvent(data)) {
    handlers.onOuterSessionEvent?.(data);
  }
  if (eventName === "tool_event" && isAgentStreamToolEvent(data)) {
    handlers.onToolEvent?.(data);
  }
  if (eventName === "done" && isAgentMessage(data.assistant_message)) {
    handlers.onDone?.(data.assistant_message);
  }
  if (eventName === "error") {
    const message = typeof data.message === "string" ? data.message : "Agent stream failed";
    handlers.onError?.(message);
    throw new ApiError(message, 500, data);
  }
}

function isAgentMessage(value: unknown): value is AgentMessage {
  return typeof value === "object" && value !== null && "id" in value && "role" in value && "content_text" in value;
}

function isAgentApprovalRequiredPayload(value: unknown): value is AgentApprovalRequiredPayload {
  return typeof value === "object" && value !== null && "approval" in value && "tool_name" in value && "context_metadata" in value;
}

function isAgentStreamOuterSessionEvent(value: unknown): value is AgentStreamOuterSessionEvent {
  return typeof value === "object" && value !== null && "event_type" in value && "event_label" in value && "session_id" in value;
}

function isAgentStreamToolEvent(value: unknown): value is AgentStreamToolEvent {
  return (
    typeof value === "object" &&
    value !== null &&
    "event_type" in value &&
    "event_label" in value &&
    "session_id" in value &&
    ("tool_name" in value || "capability" in value)
  );
}
