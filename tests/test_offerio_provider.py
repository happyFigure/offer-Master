import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.content = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeHttpClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return self.responses.pop(0)


class OfferIOProviderTest(unittest.TestCase):
    def test_list_companies_uses_browser_headers_and_normalizes_response(self):
        from app.domains.jobs.providers.offerio import OfferIORecruitmentProvider

        client = FakeHttpClient(
            [
                FakeResponse(
                    {
                        "companies": [
                            {
                                "company": "腾讯",
                                "companyNature": "民企",
                                "industry": "互联网/游戏/软件",
                                "location": "深圳总部、上海、北京",
                                "jobCount": 98,
                                "updateTime": "2026-07-29",
                            }
                        ],
                        "page": 1,
                        "pageSize": 20,
                        "total": 76,
                        "totalPages": 4,
                    }
                )
            ]
        )
        provider = OfferIORecruitmentProvider(client=client)

        result = provider.list_companies(job_type="校招", page=1, page_size=20)

        self.assertEqual(76, result.total)
        self.assertEqual("腾讯", result.items[0].name)
        self.assertEqual("民企", result.items[0].company_nature)
        self.assertEqual("互联网/游戏/软件", result.items[0].industry)
        self.assertEqual("深圳总部、上海、北京", result.items[0].locations)
        self.assertEqual(98, result.items[0].job_count)
        self.assertIn("Mozilla/5.0", client.calls[0]["headers"]["User-Agent"])
        self.assertEqual("https://offerio.work/recruitment", client.calls[0]["headers"]["Referer"])

    def test_list_jobs_and_normalize_to_import_draft(self):
        from app.domains.jobs.providers.base import JobSearchQuery, RawJob
        from app.domains.jobs.providers.offerio import OfferIORecruitmentProvider

        payload = {
            "jobs": [
                {
                    "id": "tencent_1150161895786041345",
                    "title": "硬件开发-芯片设计方向",
                    "company": "腾讯",
                    "location": "深圳总部",
                    "category": "技术",
                    "internType": "校招",
                    "publishDate": "2026-07-29",
                    "salary": "面议",
                    "department": "TEG",
                    "responsibilities": ["负责芯片设计相关开发"],
                    "requirements": ["熟悉计算机体系结构"],
                    "applyLink": "https://join.qq.com/post_detail.html?postid=1150161895786041345",
                    "source": "tencent",
                }
            ],
            "page": 1,
            "pageSize": 5,
            "total": 98,
            "totalPages": 10,
        }
        client = FakeHttpClient([FakeResponse(payload), FakeResponse(payload)])
        provider = OfferIORecruitmentProvider(client=client)

        result = provider.list_jobs(job_type="校招", company="腾讯", page=1, page_size=5)
        raw_jobs = provider.search(JobSearchQuery(keyword="腾讯", job_type="校招", limit=5))
        draft = provider.normalize(RawJob(source="offerio", payload=payload["jobs"][0]))

        self.assertEqual(98, result.total)
        self.assertEqual("硬件开发-芯片设计方向", result.items[0].title)
        self.assertEqual("腾讯", raw_jobs[0].payload["company"])
        self.assertEqual("腾讯", draft.company_name)
        self.assertEqual("offerio", draft.source)
        self.assertEqual("tencent_1150161895786041345", draft.source_job_id)
        self.assertIn("负责芯片设计相关开发", draft.jd_text)
        self.assertIn("熟悉计算机体系结构", draft.jd_text)
        self.assertEqual("https://join.qq.com/post_detail.html?postid=1150161895786041345", draft.source_url)

    def test_list_company_openings_uses_companies_endpoint_and_preserves_pagination(self):
        from app.domains.jobs.providers.offerio import OfferIORecruitmentProvider

        client = FakeHttpClient(
            [
                FakeResponse(
                    {
                        "companies": [
                            {
                                "id": "opening-001",
                                "companyName": "Tencent",
                                "companyNature": "private",
                                "industry": "internet/software",
                                "batch": "autumn",
                                "target": "2027",
                                "location": "Shenzhen, Shanghai",
                                "positions": "Java backend, AI agent platform",
                                "updateDate": "2026/08/17",
                                "deadline": "until filled",
                                "applyLink": "https://join.qq.com/",
                                "hasWrittenTest": "written test required",
                            }
                        ],
                        "page": 2,
                        "pageSize": 50,
                        "total": 3506,
                        "totalPages": 71,
                    }
                )
            ]
        )
        provider = OfferIORecruitmentProvider(client=client)

        result = provider.list_company_openings(page=2, page_size=50, keyword="Java", industry="internet")

        self.assertEqual(3506, result.total)
        self.assertEqual(50, result.page_size)
        self.assertEqual(71, result.total_pages)
        self.assertEqual("Tencent", result.items[0].company_name)
        self.assertEqual("Java backend, AI agent platform", result.items[0].positions)
        self.assertEqual("https://join.qq.com/", result.items[0].apply_link)
        self.assertTrue(client.calls[0]["url"].endswith("/api/recruitment/companies"))
        self.assertEqual(50, client.calls[0]["params"]["pageSize"])
        self.assertEqual("Java", client.calls[0]["params"]["keyword"])


if __name__ == "__main__":
    unittest.main()
