import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class CapabilityResultEvaluatorTest(unittest.TestCase):
    def test_result_evaluation_is_bound_to_capability_not_executor(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentCapabilityDefinition
        from app.agent_runtime.reflection.capability_evaluator import (
            CapabilityResultEvaluationRequest,
            CapabilityResultEvaluator,
        )
        from app.agent_runtime.reflection.schemas import campus_recruiting_web_search_result_evaluation_spec
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL

        web_search = AgentCapabilityDefinition(
            capability_id=EXTERNAL_WEB_SEARCH_TOOL,
            name="网页搜索",
            description="通过 Claude SDK agent 搜索公开网页。",
            executor_id="claude-sdk-agent",
            input_schema={"type": "object", "required": ["query"]},
            output_schema={"type": "object"},
            result_evaluation=campus_recruiting_web_search_result_evaluation_spec(),
        )
        resume_tailoring = AgentCapabilityDefinition(
            capability_id="resume.tailoring",
            name="简历润色",
            description="通过同一个 Claude SDK agent 润色简历。",
            executor_id="claude-sdk-agent",
            input_schema={"type": "object", "required": ["resume", "job_description"]},
            output_schema={"type": "object"},
        )

        payload = {
            "ok": True,
            "result": {
                "answer": "中（汉语汉字）_百度百科",
                "sources": [{"title": "中（汉语汉字）_百度百科", "url": "https://baike.baidu.com/item/中"}],
            },
        }
        evaluator = CapabilityResultEvaluator()

        web_decision = evaluator.evaluate(
            CapabilityResultEvaluationRequest(
                capability=web_search,
                tool_input={"query": "中科曙光 招聘"},
                result_payload=payload,
                expected_entities={"company_names": ["中科曙光"]},
            )
        )
        resume_decision = evaluator.evaluate(
            CapabilityResultEvaluationRequest(
                capability=resume_tailoring,
                tool_input={"resume": "...", "job_description": "..."},
                result_payload=payload,
                expected_entities={"company_names": ["中科曙光"]},
            )
        )

        self.assertIsNotNone(web_decision)
        self.assertEqual("bad", web_decision.quality.value)
        self.assertEqual("retry", web_decision.next_action.value)
        self.assertIsNone(resume_decision)

    def test_tool_and_agent_capabilities_can_declare_result_evaluation_specs(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentCapabilityDefinition, CLAUDE_SDK_AGENT_EXECUTOR_ID
        from app.agent_runtime.external_tasks.executors import ClaudeSdkAgentExecutor
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL, create_external_web_search_agent_tool_definitions

        tool_definition = create_external_web_search_agent_tool_definitions()[0]
        self.assertEqual("campus_recruiting_web_search", tool_definition.result_evaluation.evaluator_id)

        capability = AgentCapabilityDefinition.from_tool_definition(
            tool_definition,
            executor_id=CLAUDE_SDK_AGENT_EXECUTOR_ID,
        )
        self.assertEqual("campus_recruiting_web_search", capability.result_evaluation.evaluator_id)

        class FakeExternalExecutor:
            executor_name = "claude-sdk-agent"

            def execute_find_apply_entry(self, envelope):
                raise AssertionError("not used")

            def execute_web_search(self, query: str, *, max_results: int = 5):
                raise AssertionError("not used")

        claude_capability = ClaudeSdkAgentExecutor(FakeExternalExecutor()).capabilities()[0]
        self.assertEqual(EXTERNAL_WEB_SEARCH_TOOL, claude_capability.capability_id)
        self.assertEqual("campus_recruiting_web_search", claude_capability.result_evaluation.evaluator_id)

    def test_business_capabilities_auto_attach_declared_acceptance_standards(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentCapabilityDefinition
        from app.agent_runtime.tool_registry import (
            APPLICATION_FIND_APPLY_ENTRY_TOOL,
            OFFERIO_COMPANY_JOBS_TOOL,
            AgentToolDefinition,
        )

        resume_capability = AgentCapabilityDefinition(
            capability_id="resume.tailor",
            name="简历优化",
            description="根据目标岗位优化简历表达。",
            executor_id="openai-sdk-agent",
            input_schema={"type": "object", "required": ["resume_text", "job_description"]},
            output_schema={"type": "object", "required": ["revised_resume"]},
            supported_intents=("resume_tailoring",),
        )
        apply_entry_tool = AgentToolDefinition(
            name=APPLICATION_FIND_APPLY_ENTRY_TOOL,
            description="寻找岗位申请入口。",
            input_schema={"type": "object", "required": ["job_id"]},
            output_schema={"type": "object"},
        )
        offerio_tool = AgentToolDefinition(
            name=OFFERIO_COMPANY_JOBS_TOOL,
            description="同步 OfferIO 公司聚合岗位库。",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )

        self.assertEqual("resume_tailoring", resume_capability.result_evaluation.evaluator_id)
        self.assertEqual("application_entry_discovery", apply_entry_tool.result_evaluation.evaluator_id)
        self.assertEqual("offerio_company_jobs_sync", offerio_tool.result_evaluation.evaluator_id)

    def test_agent_explicit_acceptance_standard_overrides_default_business_standard(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentCapabilityDefinition
        from app.agent_runtime.reflection.schemas import CapabilityResultEvaluationSpec

        custom_spec = CapabilityResultEvaluationSpec(
            evaluator_id="custom_resume_review",
            good_result_criteria=("必须输出英文简历版本。",),
            bad_result_criteria=("只输出中文版本。",),
        )

        capability = AgentCapabilityDefinition(
            capability_id="resume.tailor",
            name="简历优化",
            description="根据目标岗位优化简历表达。",
            executor_id="openai-sdk-agent",
            input_schema={"type": "object", "required": ["resume_text", "job_description"]},
            output_schema={"type": "object", "required": ["revised_resume"]},
            result_evaluation=custom_spec,
        )

        self.assertEqual("custom_resume_review", capability.result_evaluation.evaluator_id)

    def test_clear_rule_result_does_not_call_model_evaluator(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentCapabilityDefinition
        from app.agent_runtime.reflection.capability_evaluator import (
            CapabilityResultEvaluationRequest,
            CapabilityResultEvaluator,
        )
        from app.agent_runtime.reflection.schemas import campus_recruiting_web_search_result_evaluation_spec
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL

        class FakeLLM:
            def complete(self, *, messages):
                raise AssertionError("model evaluator should not be called for clear rule results")

        capability = AgentCapabilityDefinition(
            capability_id=EXTERNAL_WEB_SEARCH_TOOL,
            name="网页搜索",
            description="通过 Claude SDK agent 搜索公开网页。",
            executor_id="claude-sdk-agent",
            input_schema={"type": "object", "required": ["query"]},
            output_schema={"type": "object"},
            result_evaluation=campus_recruiting_web_search_result_evaluation_spec(),
        )

        decision = CapabilityResultEvaluator(llm_client=FakeLLM()).evaluate(
            CapabilityResultEvaluationRequest(
                capability=capability,
                tool_input={"query": "腾讯 校园招聘 官网"},
                result_payload={
                    "ok": True,
                    "result": {
                        "answer": "腾讯校园招聘官网 join.qq.com",
                        "sources": [{"title": "腾讯校园招聘官网", "url": "https://join.qq.com"}],
                    },
                },
                expected_entities={"company_names": ["腾讯"]},
            )
        )

        self.assertEqual("good", decision.quality.value)
        self.assertEqual("continue", decision.next_action.value)
        self.assertEqual("rules", decision.metadata["capability_result_evaluation"]["decision_source"])

    def test_uncertain_rule_result_uses_model_with_capability_acceptance_standard(self) -> None:
        import json

        from app.agent_runtime.agent_as_tool import AgentCapabilityDefinition
        from app.agent_runtime.reflection.capability_evaluator import (
            CapabilityResultEvaluationRequest,
            CapabilityResultEvaluator,
        )
        from app.agent_runtime.reflection.schemas import campus_recruiting_web_search_result_evaluation_spec
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL

        class FakeLLM:
            def __init__(self) -> None:
                self.calls = []

            def complete(self, *, messages):
                self.calls.append(messages)
                prompt = "\n".join(str(message.get("content") or "") for message in messages)
                self.prompt = prompt
                return type(
                    "Completion",
                    (),
                    {
                        "content": json.dumps(
                            {
                                "quality": "partial",
                                "next_action": "retry",
                                "confidence": 0.66,
                                "reason": "公司匹配，但没有明确校招入口。",
                                "suggested_input_patch": {"query": "中科曙光 校园招聘 官网 2026"},
                            },
                            ensure_ascii=False,
                        )
                    },
                )()

        capability = AgentCapabilityDefinition(
            capability_id=EXTERNAL_WEB_SEARCH_TOOL,
            name="网页搜索",
            description="通过 Claude SDK agent 搜索公开网页。",
            executor_id="claude-sdk-agent",
            input_schema={"type": "object", "required": ["query"]},
            output_schema={"type": "object"},
            result_evaluation=campus_recruiting_web_search_result_evaluation_spec(),
        )
        llm = FakeLLM()

        decision = CapabilityResultEvaluator(llm_client=llm).evaluate(
            CapabilityResultEvaluationRequest(
                capability=capability,
                task_goal="搜一下中科曙光校园招聘官网",
                tool_input={"query": "中科曙光 人才发展"},
                result_payload={
                    "ok": True,
                    "result": {
                        "answer": "中科曙光人才发展中心，介绍公司人才理念和培养计划。",
                        "sources": [{"title": "中科曙光人才发展", "url": "https://example.com/talent"}],
                    },
                },
                expected_entities={"company_names": ["中科曙光"]},
            )
        )

        self.assertEqual(1, len(llm.calls))
        self.assertIn("结果明确匹配目标公司", llm.prompt)
        self.assertIn("结果是百科、词典、汉字解释", llm.prompt)
        self.assertIn("查中科曙光校招官网", llm.prompt)
        self.assertEqual("partial", decision.quality.value)
        self.assertEqual("retry", decision.next_action.value)
        self.assertEqual({"query": "中科曙光 校园招聘 官网 2026"}, decision.suggested_input_patch)
        self.assertEqual("model", decision.metadata["capability_result_evaluation"]["decision_source"])

    def test_resume_tailoring_uses_its_own_acceptance_standard_for_model_evaluation(self) -> None:
        import json

        from app.agent_runtime.agent_as_tool import AgentCapabilityDefinition
        from app.agent_runtime.reflection.capability_evaluator import (
            CapabilityResultEvaluationRequest,
            CapabilityResultEvaluator,
        )

        class FakeLLM:
            def __init__(self) -> None:
                self.prompt = ""

            def complete(self, *, messages):
                self.prompt = "\n".join(str(message.get("content") or "") for message in messages)
                return type(
                    "Completion",
                    (),
                    {
                        "content": json.dumps(
                            {
                                "quality": "bad",
                                "next_action": "retry",
                                "confidence": 0.82,
                                "reason": "只给建议，没有输出改写后的简历。",
                                "suggested_input_patch": {
                                    "instruction": "请保留真实经历，并直接输出 revised_resume。"
                                },
                            },
                            ensure_ascii=False,
                        )
                    },
                )()

        capability = AgentCapabilityDefinition(
            capability_id="resume.tailor",
            name="简历优化",
            description="根据目标岗位优化简历表达。",
            executor_id="openai-sdk-agent",
            input_schema={"type": "object", "required": ["resume_text", "job_description"]},
            output_schema={"type": "object", "required": ["revised_resume"]},
            supported_intents=("resume_tailoring",),
        )
        llm = FakeLLM()

        decision = CapabilityResultEvaluator(llm_client=llm).evaluate(
            CapabilityResultEvaluationRequest(
                capability=capability,
                task_goal="根据 Java 后端 JD 优化简历",
                tool_input={"resume_text": "原简历", "job_description": "Java 后端岗位"},
                result_payload={"ok": True, "result": {"advice": "建议突出 Spring Boot 项目"}},
            )
        )

        self.assertIn("保留用户原始经历", llm.prompt)
        self.assertIn("直接输出改写后的简历", llm.prompt)
        self.assertNotIn("校园招聘", llm.prompt)
        self.assertEqual("bad", decision.quality.value)
        self.assertEqual("retry", decision.next_action.value)
        self.assertEqual(
            {"instruction": "请保留真实经历，并直接输出 revised_resume。"},
            decision.suggested_input_patch,
        )
        self.assertEqual("resume_tailoring", decision.metadata["capability_result_evaluation"]["evaluator_id"])
        self.assertEqual("model", decision.metadata["capability_result_evaluation"]["decision_source"])


if __name__ == "__main__":
    unittest.main()
