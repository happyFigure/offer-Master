import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class BoundedReActPolicyTest(unittest.TestCase):
    def test_policy_enables_low_risk_external_search_with_step_budget(self) -> None:
        from app.agent_runtime.loop_agent.react_strategy import BoundedReActPolicy

        policy = BoundedReActPolicy.from_context_pack(
            {
                "risk_level": "low",
                "allowed_capabilities": ["external.web_search"],
                "intent_frame": {"intent": "campus_recruiting_search"},
            },
            requested_max_steps=8,
        )

        self.assertTrue(policy.enabled)
        self.assertEqual(5, policy.max_steps)
        self.assertEqual(["external.web_search"], policy.allowed_capabilities)
        metadata = policy.to_metadata_dict()
        self.assertEqual("bounded_react", metadata["mode"])
        self.assertEqual("runtime_limited", metadata["control"])
        self.assertIn("max_steps", metadata["guards"])

    def test_policy_blocks_high_risk_or_non_allowlisted_capabilities(self) -> None:
        from app.agent_runtime.loop_agent.react_strategy import BoundedReActPolicy

        high_risk = BoundedReActPolicy.from_context_pack(
            {
                "risk_level": "high",
                "allowed_capabilities": ["external.web_search"],
                "intent_frame": {"intent": "campus_recruiting_search"},
            },
            requested_max_steps=3,
        )
        browser_task = BoundedReActPolicy.from_context_pack(
            {
                "risk_level": "low",
                "allowed_capabilities": ["applications.find_apply_entry"],
                "intent_frame": {"intent": "application_entry_discovery"},
            },
            requested_max_steps=3,
        )

        self.assertFalse(high_risk.enabled)
        self.assertIn("risk", high_risk.disabled_reason)
        self.assertFalse(browser_task.enabled)
        self.assertIn("allowed", browser_task.disabled_reason)


if __name__ == "__main__":
    unittest.main()
