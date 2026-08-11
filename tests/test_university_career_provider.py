import sys
import unittest
from pathlib import Path

import httpx


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


if __name__ == "__main__":
    unittest.main()
