import assert from "node:assert/strict";
import test from "node:test";
import { appendRuntimeEvent, summarizeRuntimeDelegations } from "../src/app/runtimeDelegations.ts";

function event(eventType: string, delegationId: string, overrides: Record<string, unknown> = {}) {
  return {
    id: `${eventType}-${delegationId}-${overrides.toolCallId || "event"}`,
    eventType,
    delegationId,
    toolName: "agent.dbx_readonly",
    parentCapability: "agent.dbx_readonly",
    agentName: "DbxReadOnlyAgent",
    status: eventType === "subagent_started" ? "running" : "succeeded",
    ...overrides,
  };
}

test("a delegated agent counts once and its MCP start/finish counts as one inner call", () => {
  const runs = summarizeRuntimeDelegations([
    event("subagent_started", "delegation:1"),
    event("subagent_tool_started", "delegation:1", { toolName: "mcp.dbx.dbx_list_connections", toolCallId: "mcp-1" }),
    event("subagent_tool_finished", "delegation:1", { toolName: "mcp.dbx.dbx_list_connections", toolCallId: "mcp-1" }),
    event("subagent_finished", "delegation:1"),
  ]);

  assert.deepEqual(runs.map(({ id, name, status, mcpCallCount }) => ({ id, name, status, mcpCallCount })), [
    { id: "delegation:1", name: "DbxReadOnlyAgent", status: "succeeded", mcpCallCount: 1 },
  ]);
});

test("delegations remain counted after many ordinary timeline events", () => {
  const ordinaryEvents = Array.from({ length: 40 }, (_, index) => ({ id: `reasoning-${index}`, eventType: "reasoning_summary" }));
  const runs = summarizeRuntimeDelegations([
    event("subagent_started", "delegation:1"),
    event("subagent_finished", "delegation:1", { status: "waiting_user" }),
    ...ordinaryEvents,
    event("subagent_started", "delegation:2", { toolName: "agent.google_chrome", parentCapability: "agent.google_chrome", agentName: "GoogleChromeAgent" }),
    event("subagent_finished", "delegation:2", { toolName: "agent.google_chrome", parentCapability: "agent.google_chrome", agentName: "GoogleChromeAgent", status: "failed" }),
  ]);

  assert.deepEqual(runs.map(({ name, status }) => ({ name, status })), [
    { name: "DbxReadOnlyAgent", status: "waiting_user" },
    { name: "GoogleChromeAgent", status: "failed" },
  ]);
});

test("the current turn retains early events after the visible timeline grows", () => {
  const allEvents = [event("subagent_started", "delegation:early")];
  for (let index = 0; index < 40; index += 1) {
    allEvents.push({ id: `reasoning-${index}`, eventType: "reasoning_summary" });
  }
  allEvents.push(event("subagent_finished", "delegation:early"));

  const retained = allEvents.reduce((current, item) => appendRuntimeEvent(current, item), [] as typeof allEvents);

  assert.equal(retained.length, 42);
  assert.deepEqual(summarizeRuntimeDelegations(retained).map(({ id }) => id), ["delegation:early"]);
});

test("ordinary local tools do not imply a child agent was called", () => {
  assert.deepEqual(summarizeRuntimeDelegations([
    { id: "tool-1", eventType: "tool_started", toolName: "database.company_list", executorId: "agent_tool_registry" },
    { id: "tool-2", eventType: "tool_finished", toolName: "database.company_list", executorId: "agent_tool_registry" },
  ]), []);
});
