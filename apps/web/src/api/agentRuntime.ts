import { apiRequest } from "./client";
import type { AgentRuntimePanel } from "../types/agentRuntime";


export async function getAgentRuntimePanel(): Promise<AgentRuntimePanel> {
  return apiRequest<AgentRuntimePanel>("/api/v1/agent-runtime/panel");
}
