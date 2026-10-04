export type RuntimeDelegationStatus = "running" | "succeeded" | "failed" | "waiting_user" | "unknown";

export interface RuntimeDelegationEvent {
  id: string;
  eventType: string;
  toolCallId?: string | null;
  delegationId?: string | null;
  toolName?: string | null;
  parentCapability?: string | null;
  executorId?: string | null;
  agentName?: string | null;
  status?: string | null;
}

export interface RuntimeDelegationSummary {
  id: string;
  capability: string;
  name: string;
  status: RuntimeDelegationStatus;
  mcpCallCount: number;
  lastEvent: RuntimeDelegationEvent;
}

export function appendRuntimeEvent<T>(events: T[], event: T): T[] {
  return [...events, event];
}

export function summarizeRuntimeDelegations(events: RuntimeDelegationEvent[]): RuntimeDelegationSummary[] {
  type MutableSummary = RuntimeDelegationSummary & { mcpCallKeys: Set<string> };
  const runs = new Map<string, MutableSummary>();

  for (const event of events) {
    const isLifecycleEvent = event.eventType === "subagent_started" || event.eventType === "subagent_finished";
    const isMcpEvent = event.eventType === "subagent_tool_started" || event.eventType === "subagent_tool_finished";
    if (!isLifecycleEvent && !isMcpEvent) {
      continue;
    }

    const runId = event.delegationId || `legacy:${event.parentCapability || event.toolName || "subagent"}:${event.agentName || "unknown"}`;
    const capability = event.parentCapability || (event.toolName?.startsWith("agent.") ? event.toolName : null) || event.toolName || "子 Agent";
    const current = runs.get(runId);
    const run: MutableSummary = current ?? {
      id: runId,
      capability,
      name: event.agentName || fallbackSubAgentName(capability, event.executorId),
      status: "unknown",
      mcpCallCount: 0,
      lastEvent: event,
      mcpCallKeys: new Set<string>(),
    };
    run.capability = run.capability || capability;
    if (run.name === "能力子 Agent" || run.name === "子 Agent") {
      run.name = event.agentName || fallbackSubAgentName(capability, event.executorId);
    }
    run.lastEvent = event;

    if (isMcpEvent && event.eventType === "subagent_tool_started") {
      const mcpCallKey = event.toolCallId || event.id;
      if (!run.mcpCallKeys.has(mcpCallKey)) {
        run.mcpCallKeys.add(mcpCallKey);
        run.mcpCallCount += 1;
      }
    }
    if (isLifecycleEvent) {
      run.status = lifecycleStatus(event.status, event.eventType);
    }
    runs.set(runId, run);
  }

  return Array.from(runs.values()).map(({ mcpCallKeys: _mcpCallKeys, ...run }) => run);
}

function lifecycleStatus(status: string | null | undefined, eventType: string): RuntimeDelegationStatus {
  if (eventType === "subagent_started") {
    return "running";
  }
  if (status === "waiting_user" || status === "needs_approval") {
    return "waiting_user";
  }
  if (status === "failed" || status === "error") {
    return "failed";
  }
  if (status === "succeeded" || status === "success") {
    return "succeeded";
  }
  return "unknown";
}

function fallbackSubAgentName(capability: string, executorId?: string | null): string {
  if (capability === "agent.google_chrome") {
    return "Google Chrome Agent";
  }
  if (capability === "agent.dbx_readonly") {
    return "DBX 只读 Agent";
  }
  if (executorId?.includes("openai")) {
    return "OpenAI SDK Agent";
  }
  if (executorId?.includes("claude")) {
    return "Claude SDK Agent";
  }
  return "能力子 Agent";
}
