import { apiRequest, toQueryString } from "./client";
import type { AgentSkill, AgentSkillDocument, AgentSkillImportInput } from "../types/agentSkills";

interface AgentSkillListResponse {
  items: AgentSkill[];
}

const AGENT_SKILLS_PATH = "/api/v1/agent-skills";

export async function listAgentSkills(status?: AgentSkill["status"], limit = 100): Promise<AgentSkill[]> {
  const query = toQueryString({ status, limit });
  const response = await apiRequest<AgentSkillListResponse>(`${AGENT_SKILLS_PATH}${query}`);
  return response.items;
}

export async function getAgentSkill(skillId: string): Promise<AgentSkillDocument> {
  return apiRequest<AgentSkillDocument>(`${AGENT_SKILLS_PATH}/${skillId}`);
}

export async function importAgentSkill(input: AgentSkillImportInput): Promise<AgentSkill> {
  return apiRequest<AgentSkill>("/api/v1/agent-skills/import", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export async function pinAgentSkill(skillId: string): Promise<AgentSkill> {
  return apiRequest<AgentSkill>(`${AGENT_SKILLS_PATH}/${skillId}/pin`, {
    method: "POST",
  });
}

export async function archiveAgentSkill(skillId: string): Promise<AgentSkill> {
  return apiRequest<AgentSkill>(`${AGENT_SKILLS_PATH}/${skillId}/archive`, {
    method: "POST",
  });
}
