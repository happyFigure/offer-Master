import sys
import unittest
from pathlib import Path

import httpx
import base64
import zlib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class UniversityCareerProviderTest(unittest.TestCase):
    def test_fetch_extracts_recruiting_entries_from_public_html(self):
        from app.domains.jobs.providers.university_career import UniversityCareerProvider

        def handler(request):
            if str(request.url) == "https://career.example.edu/jobs":
                return httpx.Response(
                    200,
                    text=(
                        "<html><body>"
                        '<a href="/job/101">Li Auto 2027 campus recruiting backend engineer</a>'
                        '<a href="/news/notice">Library opening notice</a>'
                        "</body></html>"
                    ),
                )
            if str(request.url) == "https://career.example.edu/job/101":
                return httpx.Response(
                    200,
                    text=(
                        "<html><body><h1>Li Auto 2027 campus recruiting</h1>"
                        "<p>Backend development engineer, Beijing, Java, Spring.</p>"
                        "<script>ignore me</script></body></html>"
                    ),
                )
            return httpx.Response(404)

        provider = UniversityCareerProvider(client=httpx.Client(transport=httpx.MockTransport(handler)))

        entries = provider.fetch("https://career.example.edu/jobs", limit=5)

        self.assertEqual(1, len(entries))
        self.assertEqual("Li Auto 2027 campus recruiting backend engineer", entries[0].title)
        self.assertEqual("https://career.example.edu/job/101", entries[0].source_url)
        self.assertIn("Backend development engineer", entries[0].raw_content)
        self.assertNotIn("ignore me", entries[0].raw_content)

    def test_fetch_ignores_navigation_category_links(self):
        from app.domains.jobs.providers.university_career import UniversityCareerProvider

        def handler(request):
            if str(request.url) == "https://career.example.edu/jobs":
                return httpx.Response(
                    200,
                    text=(
                        "<html><body>"
                        '<a href="/campus">Recruiting announcements</a>'
                        '<a href="/jobfair">Job fairs</a>'
                        '<a href="/job/view/id/101">Li Auto 2027 campus recruiting backend engineer</a>'
                        "</body></html>"
                    ),
                )
            if str(request.url) == "https://career.example.edu/job/view/id/101":
                return httpx.Response(200, text="<p>Backend development engineer, Java.</p>")
            return httpx.Response(404)

        provider = UniversityCareerProvider(client=httpx.Client(transport=httpx.MockTransport(handler)))

        entries = provider.fetch("https://career.example.edu/jobs", limit=5)

        self.assertEqual(1, len(entries))
        self.assertEqual("Li Auto 2027 campus recruiting backend engineer", entries[0].title)
        self.assertEqual("https://career.example.edu/job/view/id/101", entries[0].source_url)

    def test_fetch_decodes_jysd_compressed_runtime_list(self):
        from app.domains.jobs.providers.university_career import UniversityCareerProvider

        injected_html = (
            '<ul class="infoList">'
            '<li class="span7"><a href="/campus/view/id/868376">圣邦微电子2027届校园招聘</a></li>'
            '<li class="span4">2026-08-12 11:38:06</li>'
            '</ul>'
            '<ul class="infoList">'
            '<li class="span7"><a href="/campus/view/id/868369">小米集团2027届全球校园招聘</a></li>'
            '<li class="span4">2026-08-11 10:10:51</li>'
            '</ul>'
        )
        payload = "x" * 95 + base64.b64encode(("y" * 66 + injected_html).encode()).decode()
        compressed = base64.b64encode(zlib.compress(payload.encode())).decode()
        list_html = (
            "<html><body>"
            '<section id="content9061"></section>'
            f'<script>$("#content9061").replaceWith(Base64.decode(unzip("{compressed}").substr(95)).substr(66));</script>'
            "</body></html>"
        )

        def handler(request):
            if str(request.url) == "https://myjob.example.edu/campus":
                return httpx.Response(200, text=list_html)
            if str(request.url) == "https://myjob.example.edu/campus/view/id/868376":
                return httpx.Response(200, text="<p>模拟芯片岗位，Java 平台和嵌入式方向。</p>")
            if str(request.url) == "https://myjob.example.edu/campus/view/id/868369":
                return httpx.Response(200, text="<p>软件开发、后端、AI 应用方向。</p>")
            return httpx.Response(404)

        provider = UniversityCareerProvider(client=httpx.Client(transport=httpx.MockTransport(handler)))

        entries = provider.fetch("https://myjob.example.edu/campus", limit=5)

        self.assertEqual(2, len(entries))
        self.assertEqual("圣邦微电子2027届校园招聘", entries[0].title)
        self.assertEqual("https://myjob.example.edu/campus/view/id/868376", entries[0].source_url)
        self.assertEqual("2026-08-12 11:38:06", entries[0].raw_payload["published_at_text"])
        self.assertEqual("小米集团2027届全球校园招聘", entries[1].title)


if __name__ == "__main__":
    unittest.main()
