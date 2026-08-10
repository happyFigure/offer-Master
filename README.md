# JobPilot Agent

JobPilot is a local-first interview and job application assistant built as a DDD + LangGraph-first modular monolith.

The current implementation stage is the SDD phase 0 scaffold:

- `apps/api`: FastAPI backend and domain boundaries.
- `apps/web`: React, Vite, and TypeScript frontend shell.
- `apps/worker`: local background worker shell.
- `packages`: shared schemas and prompts.
- `infra`, `data`, `tests`, and `docs`: local runtime support, fixtures, verification, and runbooks.

High-risk browser automation must go through `apps/api/app/mcp_gateway`; workflows that pause or recover must go through `apps/api/app/agent_runtime`.

