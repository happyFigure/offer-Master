import sys
import unittest
from asyncio import run
from pathlib import Path
import subprocess

from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class FakeJobicyProvider:
    name = "jobicy"
    source_type = "public_api"

    def __init__(self):
        self.queries = []

    def health_check(self):
        raise NotImplementedError

    def search(self, query):
        from app.domains.jobs.providers.base import RawJob

        self.queries.append(query)
        return [
            RawJob(
                source="jobicy",
                payload={
                    "id": "150364",
                    "title": "Software Engineer - Python - Container Images",
                    "company": "Canonical Ltd.",
                },
            )
        ]

    def normalize(self, raw_job):
        from app.domains.jobs.schemas import JobImportDraft

        return JobImportDraft(
            company_name=raw_job.payload["company"],
            title=raw_job.payload["title"],
            city="Remote",
            source="jobicy",
            source_job_id=raw_job.payload["id"],
            source_url="https://jobicy.com/jobs/150364-software-engineer-python-container-images",
            job_type="full-time",
            jd_text="Build and maintain Ubuntu container images.",
            skills=["Python", "Containers"],
            raw_payload=raw_job.payload,
        )


class JobsSyncApiTest(unittest.TestCase):
    def setUp(self):
        from app.db.base import Base
        from app.domains.applications import models as application_models  # noqa: F401
        from app.domains.automation import models as automation_models  # noqa: F401
        from app.domains.jobs import models as job_models  # noqa: F401

        self.engine = create_engine(
            "sqlite+pysqlite://",
            connect_args={"check_same_thread": False},
            future=True,
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)
        self.provider = FakeJobicyProvider()

    def tearDown(self):
        self.engine.dispose()

    def test_sync_jobs_imports_provider_results_and_reports_duplicates(self):
        from app.api.v1.jobs import get_job_providers
        from app.db.session import get_db_session
        from app.domains.jobs.models import Job
        from app.main import create_app

        app = create_app()

        def override_session():
            with self.Session() as session:
                yield session

        app.dependency_overrides[get_db_session] = override_session
        app.dependency_overrides[get_job_providers] = lambda: {"jobicy": self.provider}

        async def run_syncs():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                first = await client.post(
                    "/api/v1/jobs/sync",
                    json={"keyword": "python", "sources": ["jobicy"], "limit": 5},
                )
                second = await client.post(
                    "/api/v1/jobs/sync",
                    json={"keyword": "python", "sources": ["jobicy"], "limit": 5},
                )
                return first, second

        first_response, second_response = run(run_syncs())

        with self.Session() as session:
            stored_jobs = session.scalars(select(Job)).all()

        self.assertEqual(200, first_response.status_code)
        self.assertEqual(200, second_response.status_code)
        self.assertEqual(1, first_response.json()["imported"])
        self.assertEqual(0, first_response.json()["duplicates"])
        self.assertEqual(0, first_response.json()["failed"])
        self.assertEqual(0, second_response.json()["imported"])
        self.assertEqual(1, second_response.json()["duplicates"])
        self.assertEqual(1, len(stored_jobs))
        self.assertEqual("Software Engineer - Python - Container Images", stored_jobs[0].title)
        self.assertEqual("Canonical Ltd.", stored_jobs[0].company.name)
        self.assertEqual("python", self.provider.queries[0].keyword)
        self.assertEqual(5, self.provider.queries[0].limit)

    def test_sync_jobs_rejects_unknown_provider(self):
        from app.api.v1.jobs import get_job_providers
        from app.db.session import get_db_session
        from app.main import create_app

        app = create_app()

        def override_session():
            with self.Session() as session:
                yield session

        app.dependency_overrides[get_db_session] = override_session
        app.dependency_overrides[get_job_providers] = lambda: {"jobicy": self.provider}

        async def run_sync():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.post(
                    "/api/v1/jobs/sync",
                    json={"keyword": "python", "sources": ["unknown"], "limit": 5},
                )

        response = run(run_sync())

        self.assertEqual(400, response.status_code)
        self.assertIn("Unsupported job providers", response.json()["detail"])

    def test_job_repository_works_in_fresh_runtime_without_manual_model_imports(self):
        script = """
from app.db.session import SessionLocal
from app.domains.jobs.repository import JobRepository
with SessionLocal() as session:
    JobRepository(session).get_by_source_identity('jobicy', 'missing-id')
print('ok')
"""

        completed = subprocess.run(
            [str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"), "-c", script],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertEqual("ok", completed.stdout.strip(), completed.stderr)

    def test_job_model_query_works_in_fresh_runtime_without_manual_model_imports(self):
        script = """
from sqlalchemy import select
from app.db.session import SessionLocal
from app.domains.jobs.models import Job
with SessionLocal() as session:
    session.scalars(select(Job).limit(1)).all()
print('ok')
"""

        completed = subprocess.run(
            [str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"), "-c", script],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertEqual("ok", completed.stdout.strip(), completed.stderr)


if __name__ == "__main__":
    unittest.main()
