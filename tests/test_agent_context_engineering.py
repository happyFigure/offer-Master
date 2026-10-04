import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class FakeIntentLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.messages = []

    def complete(self, *, messages):
        from app.infrastructure.llm.chat_client import LLMChatCompletion

        self.messages.append(messages)
        return LLMChatCompletion(content=self.content)


class AgentContextEngineeringTest(unittest.TestCase):
    def test_context_pack_exposes_registered_agent_chat_capabilities_to_model(self) -> None:
        """The model receives the capability catalog; intent is telemetry only."""
        from app.agent_runtime.context.capability_catalog import CapabilityCatalog
        from app.agent_runtime.context.context_pack import ContextPackBuilder
        from app.agent_runtime.tool_registry import (
            EXTERNAL_WEB_SEARCH_TOOL,
            LOCAL_JOB_SOURCE_OVERVIEW_TOOL,
            create_default_agent_tool_registry,
        )
        from app.agent_runtime.understanding.schemas import fallback_intent_frame

        context_pack = ContextPackBuilder(
            CapabilityCatalog.from_registry(create_default_agent_tool_registry())
        ).build(fallback_intent_frame("model_driven_context"))

        self.assertEqual("model_driven", context_pack.capability_selection_mode)
        self.assertIn(EXTERNAL_WEB_SEARCH_TOOL, context_pack.allowed_capabilities)
        self.assertIn(LOCAL_JOB_SOURCE_OVERVIEW_TOOL, context_pack.allowed_capabilities)
        self.assertNotIn(EXTERNAL_WEB_SEARCH_TOOL, context_pack.excluded_capabilities)

    def test_deterministic_matcher_recognizes_only_explicit_offerio_sync_command(self) -> None:
        from app.agent_runtime.context.context_pack import ContextPackBuilder
        from app.agent_runtime.context.capability_catalog import CapabilityCatalog
        from app.agent_runtime.tool_registry import OFFERIO_COMPANY_JOBS_TOOL, create_default_agent_tool_registry
        from app.agent_runtime.understanding.intent_detector import HybridIntentDetector

        detector = HybridIntentDetector(llm_client=None)
        frame = detector.detect("请从 OfferIO 公司聚合岗位库更新一下岗位")
        context_pack = ContextPackBuilder(CapabilityCatalog.from_registry(create_default_agent_tool_registry())).build(frame)

        self.assertEqual("offerio_company_jobs_sync", frame.intent)
        self.assertEqual(1.0, frame.confidence)
        self.assertEqual(["OfferIO 公司聚合岗位库"], frame.entities.source_names)
        self.assertIn(OFFERIO_COMPANY_JOBS_TOOL, context_pack.allowed_capabilities)
        self.assertIn("external.web_search", context_pack.allowed_capabilities)
        self.assertEqual("model_driven", context_pack.capability_selection_mode)
        self.assertEqual("do_not_load_resume_full_text", context_pack.memory_policy)

    def test_deterministic_matcher_allows_reading_local_company_database_overview(self) -> None:
        from app.agent_runtime.context.capability_catalog import CapabilityCatalog
        from app.agent_runtime.context.context_pack import ContextPackBuilder
        from app.agent_runtime.tool_registry import LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL, create_default_agent_tool_registry
        from app.agent_runtime.understanding.intent_detector import HybridIntentDetector

        detector = HybridIntentDetector(llm_client=None)
        frame = detector.detect("我的数据库里现在有多少企业？")
        context_pack = ContextPackBuilder(CapabilityCatalog.from_registry(create_default_agent_tool_registry())).build(frame)

        self.assertEqual("local_company_database_overview", frame.intent)
        self.assertFalse(frame.needs_external_info)
        self.assertEqual("low", frame.risk_level)
        self.assertIn(LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL, context_pack.allowed_capabilities)
        self.assertIn("read_only_local_company_database", context_pack.notes)

    def test_deterministic_matcher_routes_path_existence_question_to_filesystem_skill(self) -> None:
        from app.agent_runtime.context.capability_catalog import CapabilityCatalog
        from app.agent_runtime.context.context_pack import ContextPackBuilder
        from app.agent_runtime.tool_registry import create_default_agent_tool_registry
        from app.agent_runtime.understanding.intent_detector import HybridIntentDetector

        detector = HybridIntentDetector(llm_client=None)
        frame = detector.detect(
            "C:/Users/phoenix/Documents/Obsidian Vault/简历/刘汉卿-后端开发-AI-Agent平台简历.tex 你看下这个文件是否存在"
        )
        context_pack = ContextPackBuilder(CapabilityCatalog.from_registry(create_default_agent_tool_registry())).build(frame)

        self.assertEqual("filesystem_operation", frame.intent)
        self.assertIn("skill.filesystem", context_pack.allowed_capabilities)
        self.assertNotIn("skill.filesystem", context_pack.excluded_capabilities)
        self.assertNotIn("filesystem.path_exists", context_pack.allowed_capabilities)

    def test_deterministic_matcher_routes_generic_company_list_to_local_database_tool(self) -> None:
        from app.agent_runtime.context.capability_catalog import CapabilityCatalog
        from app.agent_runtime.context.context_pack import ContextPackBuilder
        from app.agent_runtime.tool_registry import DATABASE_COMPANY_LIST_TOOL, EXTERNAL_WEB_SEARCH_TOOL, create_default_agent_tool_registry
        from app.agent_runtime.understanding.intent_detector import HybridIntentDetector

        detector = HybridIntentDetector(llm_client=None)
        frame = detector.detect("给我看一下有哪些公司，给我37个就行")
        context_pack = ContextPackBuilder(CapabilityCatalog.from_registry(create_default_agent_tool_registry())).build(frame)

        self.assertEqual("local_company_database_list", frame.intent)
        self.assertFalse(frame.needs_external_info)
        self.assertIn(DATABASE_COMPANY_LIST_TOOL, context_pack.allowed_capabilities)
        self.assertIn(EXTERNAL_WEB_SEARCH_TOOL, context_pack.allowed_capabilities)

    def test_deterministic_matcher_routes_job_source_count_to_job_source_overview(self) -> None:
        from app.agent_runtime.context.capability_catalog import CapabilityCatalog
        from app.agent_runtime.context.context_pack import ContextPackBuilder
        from app.agent_runtime.tool_registry import LOCAL_JOB_SOURCE_OVERVIEW_TOOL, create_default_agent_tool_registry
        from app.agent_runtime.understanding.intent_detector import HybridIntentDetector

        detector = HybridIntentDetector(llm_client=None)
        frame = detector.detect("有多少岗位来源？")
        company_frame = detector.detect("公司展览里现在有多少公司？")
        context_pack = ContextPackBuilder(CapabilityCatalog.from_registry(create_default_agent_tool_registry())).build(frame)

        self.assertEqual("local_job_source_overview", frame.intent)
        self.assertEqual("company_board_overview", company_frame.intent)
        self.assertFalse(frame.needs_external_info)
        self.assertEqual("low", frame.risk_level)
        self.assertIn(LOCAL_JOB_SOURCE_OVERVIEW_TOOL, context_pack.allowed_capabilities)
        self.assertIn("read_only_local_job_sources", context_pack.notes)

    def test_company_count_defaults_to_company_board_count(self) -> None:
        from app.agent_runtime.context.capability_catalog import CapabilityCatalog
        from app.agent_runtime.context.context_pack import ContextPackBuilder
        from app.agent_runtime.tool_registry import LOCAL_JOB_SOURCE_OVERVIEW_TOOL, create_default_agent_tool_registry
        from app.agent_runtime.understanding.intent_detector import HybridIntentDetector

        frame = HybridIntentDetector(llm_client=None).detect("当前数据库中的公司数有多少个")
        context_pack = ContextPackBuilder(CapabilityCatalog.from_registry(create_default_agent_tool_registry())).build(frame)

        self.assertEqual("company_board_overview", frame.intent)
        self.assertIn(LOCAL_JOB_SOURCE_OVERVIEW_TOOL, context_pack.allowed_capabilities)

    def test_explicit_dbx_only_constraint_survives_deterministic_company_count_match(self) -> None:
        from app.agent_runtime.understanding.intent_detector import HybridIntentDetector

        llm = FakeIntentLLM(
            json.dumps(
                {
                    "intent": "company_board_overview",
                    "confidence": 1.0,
                    "needs_external_info": True,
                    "risk_level": "low",
                    "required_capability": "agent.dbx_readonly",
                    "reason": "用户明确要求只通过 DBX 只读子 Agent 查询",
                },
                ensure_ascii=False,
            )
        )

        frame = HybridIntentDetector(llm_client=llm).detect(
            "请只通过已注册的 DBX 只读子 Agent 查询公司展览当前开放岗位公司库的公司数"
        )

        self.assertEqual("company_board_overview", frame.intent)
        self.assertEqual("agent.dbx_readonly", frame.required_capability)
        self.assertEqual(1, len(llm.messages))

    def test_explicit_local_database_qualifier_keeps_local_database_overview(self) -> None:
        from app.agent_runtime.understanding.intent_detector import HybridIntentDetector

        frame = HybridIntentDetector(llm_client=None).detect("我的本地数据库正式企业表有多少条记录")

        self.assertEqual("local_company_database_overview", frame.intent)

    def test_llm_intent_detector_extracts_full_chinese_company_name_without_regex_cutting(self) -> None:
        from app.agent_runtime.understanding.intent_detector import HybridIntentDetector

        llm = FakeIntentLLM(
            """
            ```json
            {
              "intent": "campus_recruiting_search",
              "confidence": 0.94,
              "needs_external_info": true,
              "risk_level": "low",
              "entities": {
                "company_names": ["中科曙光"],
                "keywords": ["校园招聘"],
                "time_range": "latest"
              }
            }
            ```
            """
        )

        frame = HybridIntentDetector(llm_client=llm).detect("你给我搜一下中科曙光的校园招聘信息")

        self.assertEqual("campus_recruiting_search", frame.intent)
        self.assertEqual(["中科曙光"], frame.entities.company_names)
        self.assertNotEqual(["中"], frame.entities.company_names)
        self.assertTrue(frame.needs_external_info)
        self.assertEqual("low", frame.risk_level)
        self.assertIn("只能输出 JSON", llm.messages[0][0]["content"])

    def test_context_pack_filters_capabilities_without_running_tools(self) -> None:
        from app.agent_runtime.context.context_pack import ContextPackBuilder
        from app.agent_runtime.context.capability_catalog import CapabilityCatalog
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL, create_default_agent_tool_registry
        from app.agent_runtime.understanding.schemas import EntityFrame, IntentFrame

        frame = IntentFrame(
            intent="campus_recruiting_search",
            confidence=0.93,
            needs_external_info=True,
            risk_level="low",
            entities=EntityFrame(company_names=["中科曙光"], keywords=["校园招聘"], time_range="latest"),
        )
        context_pack = ContextPackBuilder(CapabilityCatalog.from_registry(create_default_agent_tool_registry())).build(frame)

        self.assertEqual("campus_recruiting_search", context_pack.intent)
        self.assertIn(EXTERNAL_WEB_SEARCH_TOOL, context_pack.allowed_capabilities)
        self.assertIn("offerio.sync_company_jobs", context_pack.allowed_capabilities)
        self.assertNotIn("applications.find_apply_entry", context_pack.excluded_capabilities)
        self.assertEqual("do_not_load_resume_full_text", context_pack.memory_policy)
        self.assertEqual([], context_pack.loaded_capabilities)

    def test_context_pack_filters_capabilities_from_agent_capability_registry(self) -> None:
        from app.agent_runtime.agent_as_tool import create_default_agent_capability_registry
        from app.agent_runtime.context.capability_catalog import CapabilityCatalog
        from app.agent_runtime.context.context_pack import ContextPackBuilder
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL, OFFERIO_COMPANY_JOBS_TOOL, AgentToolDefinition, AgentToolRegistry
        from app.agent_runtime.understanding.schemas import EntityFrame, IntentFrame

        tool_registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name=EXTERNAL_WEB_SEARCH_TOOL,
                    description="搜索公开网页招聘信息。",
                    input_schema={"type": "object", "required": ["query"], "properties": {"query": {"type": "string"}}},
                    output_schema={"type": "object"},
                ),
                AgentToolDefinition(
                    name=OFFERIO_COMPANY_JOBS_TOOL,
                    description="同步 OfferIO 公司聚合岗位库。",
                    input_schema={"type": "object", "properties": {"limit": {"type": "integer"}}},
                    output_schema={"type": "object"},
                ),
            ]
        )
        capability_registry = create_default_agent_capability_registry(tool_registry=tool_registry)
        frame = IntentFrame(
            intent="campus_recruiting_search",
            confidence=0.93,
            needs_external_info=True,
            risk_level="low",
            entities=EntityFrame(company_names=["腾讯"], keywords=["校园招聘"], time_range="latest"),
        )

        context_pack = ContextPackBuilder(CapabilityCatalog.from_agent_registry(capability_registry)).build(frame)

        self.assertIn(EXTERNAL_WEB_SEARCH_TOOL, context_pack.allowed_capabilities)
        self.assertIn(OFFERIO_COMPANY_JOBS_TOOL, context_pack.allowed_capabilities)
        self.assertIn("skill.filesystem", context_pack.allowed_capabilities)
        self.assertEqual(["query"], context_pack.capability_metadata[0]["input_summary"])

    def test_invalid_llm_json_falls_back_to_normal_chat(self) -> None:
        from app.agent_runtime.understanding.intent_detector import HybridIntentDetector

        frame = HybridIntentDetector(llm_client=FakeIntentLLM("not json at all")).detect("随便聊聊")

        self.assertEqual("normal_chat", frame.intent)
        self.assertEqual(0.0, frame.confidence)
        self.assertFalse(frame.needs_external_info)
        self.assertEqual("fallback_invalid_llm_json", frame.reason)

    def test_contextual_web_search_follow_up_uses_recent_user_goal(self) -> None:
        from app.agent_runtime.understanding.intent_detector import HybridIntentDetector

        llm = FakeIntentLLM(
            json.dumps(
                {
                    "intent": "campus_recruiting_search",
                    "confidence": 0.96,
                    "needs_external_info": True,
                    "risk_level": "low",
                    "entities": {"company_names": ["东方希望"], "keywords": ["网页搜索"]},
                    "candidate_intents": ["campus_recruiting_search"],
                    "reason": "当前句省略了主题，需沿用最近一条网页搜索东方希望的用户目标。",
                },
                ensure_ascii=False,
            )
        )

        frame = HybridIntentDetector(llm_client=llm).detect(
            "我让你去网页搜索啊",
            recent_user_context="你去给我网络搜索一下东方希望",
        )

        self.assertEqual("campus_recruiting_search", frame.intent)
        self.assertEqual(["东方希望"], frame.entities.company_names)
        self.assertTrue(frame.needs_external_info)
        intent_user_message = llm.messages[0][1]["content"]
        self.assertIn("最近用户上下文", intent_user_message)
        self.assertIn("东方希望", intent_user_message)


if __name__ == "__main__":
    unittest.main()
