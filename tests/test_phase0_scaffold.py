import sys
import unittest
from asyncio import run
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Phase0ScaffoldTest(unittest.TestCase):
    def test_sdd_required_scaffold_directories_exist(self):
        expected_directories = [
            "apps/api/app/api/v1",
            "apps/api/app/core",
            "apps/api/app/db",
            "apps/api/app/domains/jobs/providers",
            "apps/api/app/domains/applications",
            "apps/api/app/domains/knowledge",
            "apps/api/app/domains/interviews",
            "apps/api/app/domains/automation",
            "apps/api/app/domains/conversations",
            "apps/api/app/agent_runtime/workflows",
            "apps/api/app/rag",
            "apps/api/app/mcp_gateway",
            "apps/api/app/infrastructure/llm",
            "apps/api/app/infrastructure/vector_store",
            "apps/api/app/infrastructure/speech",
            "apps/api/app/infrastructure/scheduler",
            "apps/api/app/infrastructure/observability",
            "apps/web/src/app",
            "apps/web/src/api",
            "apps/web/src/components/layout",
            "apps/web/src/components/chat",
            "apps/web/src/components/jobs",
            "apps/web/src/components/applications",
            "apps/web/src/components/knowledge",
            "apps/web/src/components/dashboard",
            "apps/web/src/components/common",
            "apps/web/src/pages",
            "apps/web/src/stores",
            "apps/web/src/types/generated",
            "apps/web/src/styles",
            "apps/worker/app/jobs",
            "apps/worker/app/runtime",
            "packages/schemas",
            "packages/prompts",
            "infra/docker",
            "infra/migrations",
            "infra/scripts",
            "data/mock",
            "data/imports",
            "data/uploads",
            "data/exports",
            "data/vector_store",
            "tests/e2e",
            "tests/fixtures",
            "docs/architecture",
            "docs/runbooks",
        ]

        missing = [
            directory
            for directory in expected_directories
            if not (PROJECT_ROOT / directory).is_dir()
        ]

        self.assertEqual([], missing)

    def test_phase0_entrypoint_and_boundary_files_exist(self):
        expected_files = [
            "README.md",
            ".env.example",
            ".gitignore",
            "requirements-dev.txt",
            "apps/api/pyproject.toml",
            "apps/api/app/main.py",
            "apps/api/app/agent_runtime/checkpoints.py",
            "apps/api/app/agent_runtime/human_approval.py",
            "apps/api/app/agent_runtime/guardrails.py",
            "apps/api/app/mcp_gateway/tool_policy.py",
            "apps/web/package.json",
            "apps/web/index.html",
            "apps/web/src/main.tsx",
            "apps/worker/pyproject.toml",
            "apps/worker/app/main.py",
            "packages/schemas/README.md",
            "packages/prompts/README.md",
            "docs/architecture/README.md",
            "docs/runbooks/README.md",
        ]

        missing = [
            file_path
            for file_path in expected_files
            if not (PROJECT_ROOT / file_path).is_file()
        ]

        self.assertEqual([], missing)

    def test_backend_health_endpoint_reports_api_runtime(self):
        sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))

        from app.main import create_app
        from httpx import ASGITransport, AsyncClient

        async def get_health_payload():
            transport = ASGITransport(app=create_app())
            async with AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.get("/health")

        response = run(get_health_payload())

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            {
                "service": "jobpilot-api",
                "status": "ok",
                "architecture": "ddd-langgraph-modular-monolith",
            },
            response.json(),
        )


if __name__ == "__main__":
    unittest.main()
