import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class WebSearchQueryTest(unittest.TestCase):
    def test_ronaldo_recent_match_query_keeps_last_match_direction(self) -> None:
        from app.agent_runtime.web_search_query import normalize_external_web_search_query

        query = normalize_external_web_search_query("c罗最近的一次比赛是什么时候的")

        self.assertIn("Cristiano Ronaldo", query)
        self.assertIn("Al Nassr", query)
        self.assertIn("last match", query)
        self.assertIn("result date", query)
        self.assertNotIn("next match", query)
        self.assertNotIn("match schedule", query)


if __name__ == "__main__":
    unittest.main()
