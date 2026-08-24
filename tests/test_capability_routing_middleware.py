import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class CapabilityRoutingMiddlewareTest(unittest.TestCase):
    def _context_pack(self, *, intent: str, company_names=None, risk_level: str = "low") -> dict[str, object]:
        from app.agent_runtime.context.capability_catalog import CapabilityCatalog
        from app.agent_runtime.context.context_pack import ContextPackBuilder
        from app.agent_runtime.tool_registry import create_default_agent_tool_registry
        from app.agent_runtime.understanding.schemas import EntityFrame, IntentFrame

        frame = IntentFrame(
            intent=intent,
            confidence=0.95,
            needs_external_info=intent in {"campus_recruiting_search", "external_agent_task"},
            risk_level=risk_level,
            entities=EntityFrame(company_names=list(company_names or []), keywords=["校园招聘"], time_range="latest"),
        )
        return ContextPackBuilder(CapabilityCatalog.from_registry(create_default_agent_tool_registry())).build(frame).to_metadata_dict()

    def test_routes_normal_chat_directly_without_planner_or_tools(self) -> None:
        from app.agent_runtime.routing.capability_routing_middleware import CapabilityRoutingMiddleware

        decision = CapabilityRoutingMiddleware().decide(
            user_message="Planner Gate 是什么？",
            intent_frame={"intent": "normal_chat", "confidence": 0.95, "risk_level": "low"},
            context_pack=self._context_pack(intent="normal_chat"),
        )

        self.assertEqual("chat_direct", decision.route)
        self.assertIsNone(decision.capability)
        self.assertEqual("chat", decision.executor_type)
        self.assertFalse(decision.requires_confirmation)
        self.assertIn("normal_chat", decision.reason)

    def test_routes_campus_recruiting_search_to_external_agent_capability(self) -> None:
        from app.agent_runtime.routing.capability_routing_middleware import CapabilityRoutingMiddleware
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL

        decision = CapabilityRoutingMiddleware().decide(
            user_message="你给我搜一下中科曙光的校园招聘信息",
            intent_frame={
                "intent": "campus_recruiting_search",
                "confidence": 0.95,
                "risk_level": "low",
                "entities": {"company_names": ["中科曙光"], "keywords": ["校园招聘"], "time_range": "latest"},
            },
            context_pack=self._context_pack(intent="campus_recruiting_search", company_names=["中科曙光"]),
        )

        self.assertEqual("external_agent", decision.route)
        self.assertEqual(EXTERNAL_WEB_SEARCH_TOOL, decision.capability)
        self.assertEqual("external_agent", decision.executor_type)
        self.assertEqual("claude_sdk_agent", decision.executor_name)
        self.assertEqual({"query": "中科曙光 校园招聘 官网", "max_results": 5}, decision.tool_input)

    def test_asks_user_when_search_intent_has_ambiguous_entity(self) -> None:
        from app.agent_runtime.routing.capability_routing_middleware import CapabilityRoutingMiddleware

        decision = CapabilityRoutingMiddleware().decide(
            user_message="帮我搜一下公牛",
            intent_frame={
                "intent": "campus_recruiting_search",
                "confidence": 0.62,
                "risk_level": "low",
                "candidate_intents": ["campus_recruiting_search", "external_agent_task"],
                "entities": {"company_names": ["公牛"], "keywords": []},
            },
            context_pack=self._context_pack(intent="campus_recruiting_search", company_names=["公牛"]),
        )

        self.assertEqual("ask_user", decision.route)
        self.assertEqual("clarification", decision.executor_type)
        self.assertTrue(decision.requires_confirmation)
        self.assertIn("公牛集团", decision.reason)
        self.assertIn("芝加哥公牛队", decision.reason)
        self.assertEqual("entity_ambiguity", decision.metadata["clarification_kind"])

    def test_routes_offerio_sync_to_local_workflow_with_page_size_policy(self) -> None:
        from app.agent_runtime.routing.capability_routing_middleware import CapabilityRoutingMiddleware
        from app.agent_runtime.tool_registry import OFFERIO_COMPANY_JOBS_TOOL

        context_pack = self._context_pack(intent="offerio_company_jobs_sync")

        decision = CapabilityRoutingMiddleware().decide(
            user_message="请从 OfferIO 公司聚合岗位库更新一下岗位",
            intent_frame={"intent": "offerio_company_jobs_sync", "confidence": 1.0, "risk_level": "medium"},
            context_pack=context_pack,
        )

        self.assertEqual("local_workflow", decision.route)
        self.assertEqual(OFFERIO_COMPANY_JOBS_TOOL, decision.capability)
        self.assertEqual("local_workflow", decision.executor_type)
        self.assertEqual({"limit": 1000}, decision.tool_input)
        self.assertIn(OFFERIO_COMPANY_JOBS_TOOL, decision.allowed_capabilities)

    def test_routes_local_company_database_question_to_readonly_workflow(self) -> None:
        from app.agent_runtime.routing.capability_routing_middleware import CapabilityRoutingMiddleware
        from app.agent_runtime.tool_registry import LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL

        context_pack = self._context_pack(intent="local_company_database_overview")

        decision = CapabilityRoutingMiddleware().decide(
            user_message="我的数据库里现在有多少企业？",
            intent_frame={"intent": "local_company_database_overview", "confidence": 1.0, "risk_level": "low"},
            context_pack=context_pack,
        )

        self.assertEqual("local_workflow", decision.route)
        self.assertEqual(LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL, decision.capability)
        self.assertEqual("local_workflow", decision.executor_type)
        self.assertEqual("company_database_overview", decision.executor_name)
        self.assertEqual({"sample_limit": 10}, decision.tool_input)
        self.assertFalse(decision.requires_confirmation)
        self.assertEqual(True, decision.metadata["read_only"])

    def test_routes_local_company_database_question_with_requested_count_to_readonly_workflow(self) -> None:
        from app.agent_runtime.routing.capability_routing_middleware import CapabilityRoutingMiddleware
        from app.agent_runtime.tool_registry import LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL

        context_pack = self._context_pack(intent="local_company_database_overview")

        decision = CapabilityRoutingMiddleware().decide(
            user_message="给我看一下有哪些公司，给我37个就行",
            intent_frame={"intent": "local_company_database_overview", "confidence": 1.0, "risk_level": "low"},
            context_pack=context_pack,
        )

        self.assertEqual("local_workflow", decision.route)
        self.assertEqual(LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL, decision.capability)
        self.assertEqual({"sample_limit": 37}, decision.tool_input)

    def test_routes_job_source_count_question_to_readonly_job_source_workflow(self) -> None:
        from app.agent_runtime.routing.capability_routing_middleware import CapabilityRoutingMiddleware
        from app.agent_runtime.tool_registry import LOCAL_JOB_SOURCE_OVERVIEW_TOOL

        context_pack = self._context_pack(intent="local_job_source_overview")

        decision = CapabilityRoutingMiddleware().decide(
            user_message="有多少岗位来源？",
            intent_frame={"intent": "local_job_source_overview", "confidence": 1.0, "risk_level": "low"},
            context_pack=context_pack,
        )

        self.assertEqual("local_workflow", decision.route)
        self.assertEqual(LOCAL_JOB_SOURCE_OVERVIEW_TOOL, decision.capability)
        self.assertEqual("local_workflow", decision.executor_type)
        self.assertEqual("job_source_overview", decision.executor_name)
        self.assertEqual({"sample_limit": 10, "include_external_job_board": True}, decision.tool_input)
        self.assertFalse(decision.requires_confirmation)
        self.assertEqual(True, decision.metadata["read_only"])


if __name__ == "__main__":
    unittest.main()
