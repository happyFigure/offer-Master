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


class FakeTencentClient:
    def __init__(self, search_payload, detail_payloads):
        self.search_payload = search_payload
        self.detail_payloads = detail_payloads
        self.calls = []

    def post(self, url, *, json, headers, timeout):
        self.calls.append(
            {"method": "post", "url": url, "json": json, "headers": headers, "timeout": timeout}
        )
        return FakeResponse(self.search_payload)

    def get(self, url, *, params, headers, timeout):
        self.calls.append(
            {"method": "get", "url": url, "params": params, "headers": headers, "timeout": timeout}
        )
        return FakeResponse(self.detail_payloads[params["postId"]])


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

    def test_tencent_campus_search_enriches_only_campus_tech_jobs(self):
        from app.domains.jobs.providers.base import JobSearchQuery
        from app.domains.jobs.providers.tencent_campus import TencentCampusProvider

        fake_client = FakeTencentClient(
            search_payload={
                "status": 0,
                "data": {
                    "positionList": [
                        {
                            "postId": "1200791473415778304",
                            "id": 21417,
                            "position": 101,
                            "positionTitle": "后台开发",
                            "projectId": 2,
                            "projectName": "应届实习",
                            "recruitLabelName": "应届实习",
                            "workCities": "深圳总部 北京 上海",
                            "bgs": "CSIG WXG",
                        },
                        {
                            "postId": "2077347119940939776",
                            "positionTitle": "QQ-Agent产品经理",
                            "projectName": "应届实习",
                            "recruitLabelName": "应届实习",
                        },
                        {
                            "postId": "2006202335759585280",
                            "positionTitle": "微信基础-后端开发工程师-Agent",
                            "projectName": "社会招聘",
                            "recruitLabelName": "社会招聘",
                        },
                    ]
                },
            },
            detail_payloads={
                "1200791473415778304": {
                    "status": 0,
                    "data": {
                        "postId": "1200791473415778304",
                        "id": 101,
                        "title": "后台开发",
                        "desc": "负责关键服务与基础设施，参与 AI 能力在后端的集成与落地。",
                        "request": "熟练掌握 C/C++/Java/Go 等其中一门开发语言，了解 MySQL。",
                        "workCityList": ["深圳总部", "北京", "上海"],
                        "projectName": "应届实习",
                        "recruitLabelName": "应届实习",
                    },
                }
            },
        )
        provider = TencentCampusProvider(client=fake_client, timeout_seconds=6)

        raw_jobs = provider.search(JobSearchQuery(keyword="后端", limit=3, job_type="campus"))

        self.assertEqual(1, len(raw_jobs))
        self.assertEqual("tencent_campus", raw_jobs[0].source)
        self.assertEqual("后台开发", raw_jobs[0].payload["summary"]["positionTitle"])
        self.assertEqual("后台开发", raw_jobs[0].payload["detail"]["title"])
        self.assertEqual("post", fake_client.calls[0]["method"])
        self.assertEqual("后端", fake_client.calls[0]["json"]["keyword"])
        self.assertEqual(3, fake_client.calls[0]["json"]["pageSize"])
        self.assertEqual("get", fake_client.calls[1]["method"])
        self.assertEqual("1200791473415778304", fake_client.calls[1]["params"]["postId"])

    def test_tencent_campus_normalize_maps_campus_detail_to_import_draft(self):
        from app.domains.jobs.providers.base import RawJob
        from app.domains.jobs.providers.tencent_campus import TencentCampusProvider

        provider = TencentCampusProvider()
        raw_job = RawJob(
            source="tencent_campus",
            payload={
                "summary": {
                    "postId": "1200791473415778304",
                    "id": 21417,
                    "position": 101,
                    "positionTitle": "后台开发",
                    "projectId": 2,
                    "projectName": "应届实习",
                    "recruitLabelName": "应届实习",
                    "bgs": "CSIG WXG",
                    "workCities": "深圳总部 北京 上海 广州 成都 杭州",
                },
                "detail": {
                    "postId": "1200791473415778304",
                    "id": 101,
                    "title": "后台开发",
                    "desc": "负责实现和优化产品功能，参与 AI 能力在后端的集成与落地。",
                    "request": "熟练掌握 C/C++/Java/Go；了解 MySQL、SQL、大模型 API 调用。",
                    "workCityList": ["深圳总部", "北京", "上海", "广州", "成都", "杭州"],
                    "graduateBonus": "有利用 AI 工具完成实际项目的经验。",
                    "recruitLabelName": "应届实习",
                },
            },
        )

        draft = provider.normalize(raw_job)

        self.assertEqual("腾讯", draft.company_name)
        self.assertEqual("后台开发", draft.title)
        self.assertEqual("深圳总部, 北京, 上海, 广州, 成都, 杭州", draft.city)
        self.assertEqual("tencent_campus", draft.source)
        self.assertEqual("1200791473415778304", draft.source_job_id)
        self.assertIn("join.qq.com", draft.source_url)
        self.assertEqual("应届实习", draft.job_type)
        self.assertIn("AI 能力在后端的集成", draft.jd_text)
        self.assertIn("Java", draft.skills)
        self.assertIn("Go", draft.skills)
        self.assertIn("大模型", draft.skills)
        self.assertIn("MySQL", draft.skills)


if __name__ == "__main__":
    unittest.main()
