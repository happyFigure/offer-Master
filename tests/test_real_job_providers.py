import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeHttpClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, *, params, timeout):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return FakeResponse(self.payload)


class RealJobProvidersTest(unittest.TestCase):
    def test_jobicy_search_requests_public_api_with_keyword_and_limit(self):
        from app.domains.jobs.providers.base import JobSearchQuery
        from app.domains.jobs.providers.jobicy import JobicyProvider

        fake_client = FakeHttpClient(
            {
                "jobs": [
                    {
                        "id": 150364,
                        "jobTitle": "Software Engineer - Python - Container Images",
                        "companyName": "Canonical Ltd.",
                    }
                ]
            }
        )
        provider = JobicyProvider(client=fake_client, timeout_seconds=7)

        raw_jobs = provider.search(JobSearchQuery(keyword="python", limit=2))

        self.assertEqual(1, len(raw_jobs))
        self.assertEqual("jobicy", raw_jobs[0].source)
        self.assertEqual(
            {
                "url": "https://jobicy.com/api/v2/remote-jobs",
                "params": {"count": 2, "tag": "python"},
                "timeout": 7,
            },
            fake_client.calls[0],
        )

    def test_jobicy_normalize_maps_payload_to_import_draft(self):
        from app.domains.jobs.providers.base import RawJob
        from app.domains.jobs.providers.jobicy import JobicyProvider

        provider = JobicyProvider()
        raw_job = RawJob(
            source="jobicy",
            payload={
                "id": 150364,
                "jobTitle": "Software Engineer - Python - Container Images",
                "companyName": "Canonical Ltd.",
                "jobGeo": "APAC, EMEA, LATAM, Canada, USA",
                "url": "https://jobicy.com/jobs/150364-software-engineer-python-container-images",
                "jobType": "full-time",
                "jobDescription": "<p>Build and maintain Ubuntu container images.</p>",
                "pubDate": "2026-08-10T09:00:00+00:00",
                "jobIndustry": ["Engineering", "DevOps"],
                "jobLevel": "Senior",
            },
        )

        draft = provider.normalize(raw_job)

        self.assertEqual("Canonical Ltd.", draft.company_name)
        self.assertEqual("Software Engineer - Python - Container Images", draft.title)
        self.assertEqual("APAC, EMEA, LATAM, Canada, USA", draft.city)
        self.assertEqual("jobicy", draft.source)
        self.assertEqual("150364", draft.source_job_id)
        self.assertEqual(
            "https://jobicy.com/jobs/150364-software-engineer-python-container-images",
            draft.source_url,
        )
        self.assertEqual("full-time", draft.job_type)
        self.assertEqual("Build and maintain Ubuntu container images.", draft.jd_text)
        self.assertEqual(["Engineering", "DevOps", "Senior"], draft.skills)
        self.assertEqual("2026-08-10", draft.date_posted.isoformat())
        self.assertEqual(raw_job.payload, draft.raw_payload)

    def test_jobicy_normalize_rejects_payload_without_required_identity(self):
        from app.domains.jobs.providers.base import RawJob
        from app.domains.jobs.providers.jobicy import JobicyProvider

        provider = JobicyProvider()

        with self.assertRaises(ValueError):
            provider.normalize(RawJob(source="jobicy", payload={"companyName": "No Title Inc"}))


if __name__ == "__main__":
    unittest.main()
