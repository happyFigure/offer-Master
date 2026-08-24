import sys
import threading
import unittest
from asyncio import run
from pathlib import Path
from unittest.mock import patch

from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class AgentApiTest(unittest.TestCase):
    def setUp(self):
        from app.db.base import Base
        import app.agent_runtime.durable_state.models  # noqa: F401
        from app.agent_runtime.external_tasks import models as external_task_models  # noqa: F401
        from app.domains.agent_memory import models as agent_memory_models  # noqa: F401
        from app.domains.automation import models as automation_models  # noqa: F401
        from app.domains.conversations import models as conversation_models  # noqa: F401

        self.engine = create_engine(
            "sqlite+pysqlite://",
            connect_args={"check_same_thread": False},
            future=True,
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)
        self._patchers = []

    def tearDown(self):
        for patcher in reversed(self._patchers):
            patcher.stop()
        self.engine.dispose()

    def _app(self, *, llm_client=None, intent_detector=None):
        from app.api.v1 import agent as agent_api
        from app.agent_runtime.understanding.intent_detector import HybridIntentDetector
        from app.db.session import get_db_session
        from app.main import create_app

        llm_patcher = patch.object(agent_api, "_build_agent_llm_client", return_value=llm_client)
        llm_patcher.start()
        self._patchers.append(llm_patcher)

        intent_patcher = patch.object(
            agent_api,
            "_build_agent_intent_detector",
            return_value=intent_detector or HybridIntentDetector(llm_client=None),
        )
        intent_patcher.start()
        self._patchers.append(intent_patcher)

        app = create_app()

        def override_session():
            with self.Session() as session:
                yield session

        app.dependency_overrides[get_db_session] = override_session
        return app

    def test_create_and_list_agent_sessions(self):
        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                created = await client.post(
                    "/api/v1/agent/sessions",
                    json={"title": "秋招规划", "primary_intent": "job_search"},
                )
                if created.status_code != 201:
                    return created, None, None
                listed = await client.get("/api/v1/agent/sessions")
                fetched = await client.get(f"/api/v1/agent/sessions/{created.json()['id']}")
                return created, listed, fetched

        created_response, listed_response, fetched_response = run(call_api())

        self.assertEqual(201, created_response.status_code)
        self.assertIsNotNone(listed_response)
        self.assertIsNotNone(fetched_response)
        self.assertEqual("秋招规划", created_response.json()["title"])
        self.assertEqual("active", created_response.json()["status"])
        self.assertEqual("job_search", created_response.json()["primary_intent"])
        self.assertEqual(0, created_response.json()["message_count"])
        self.assertEqual(200, listed_response.status_code)
        self.assertEqual(created_response.json()["id"], listed_response.json()["items"][0]["id"])
        self.assertEqual(200, fetched_response.status_code)
        self.assertEqual(created_response.json()["id"], fetched_response.json()["id"])

    def test_agent_graph_dependencies_include_durable_state_service(self):
        from app.api.v1.agent import _agent_graph_dependencies, _conversation_service
        from app.agent_runtime.durable_state.service import DurableStateService

        with self.Session() as session:
            conversation_service = _conversation_service(session)
            dependencies = _agent_graph_dependencies(session, conversation_service)

        self.assertIsInstance(dependencies.durable_state_service, DurableStateService)

    def test_resume_agent_task_endpoint_creates_retry_step(self):
        from app.agent_runtime.durable_state.repository import SqlAlchemyDurableStateRepository
        from app.agent_runtime.durable_state.service import DurableStateService

        with self.Session() as session:
            service = DurableStateService(SqlAlchemyDurableStateRepository(session))
            service.create_task(
                task_id="task-api-resume",
                root_workflow_run_id="workflow-api-resume",
                conversation_session_id="session-api-resume",
                task_type="campus_search",
                capability="external.web_search",
            )
            step = service.add_step(
                task_id="task-api-resume",
                step_id="step-api-resume-1",
                sequence_index=1,
                step_type="external_agent",
                executor_type="external_agent",
                executor_name="claude_sdk_agent",
                capability="external.web_search",
                input_payload={"query": "腾讯 校园招聘 官网"},
            )
            step.retry_count = 1
            service.mark_step_failed("step-api-resume-1", output_payload={"error": "timeout"})
            session.commit()

        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.post("/api/v1/agent/tasks/task-api-resume/resume")

        response = run(call_api())

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual("retry_failed_step", payload["action"])
        self.assertEqual("task-api-resume", payload["task_id"])
        self.assertEqual("step-api-resume-1", payload["source_step_id"])
        self.assertIsNotNone(payload["resume_step_id"])
        self.assertFalse(payload["requires_user_action"])

    def test_rename_and_delete_agent_session_hides_it_from_default_list(self):
        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                created = await client.post(
                    "/api/v1/agent/sessions",
                    json={"title": "旧会话标题", "primary_intent": "agent_chat"},
                )
                session_id = created.json()["id"]
                renamed = await client.patch(
                    f"/api/v1/agent/sessions/{session_id}",
                    json={"title": "新的会话标题"},
                )
                delete_response = await client.delete(f"/api/v1/agent/sessions/{session_id}")
                default_list = await client.get("/api/v1/agent/sessions")
                archived_list = await client.get("/api/v1/agent/sessions?include_archived=true")
                return renamed, delete_response, default_list, archived_list

        renamed_response, delete_response, default_list_response, archived_list_response = run(call_api())

        self.assertEqual(200, renamed_response.status_code)
        self.assertEqual("新的会话标题", renamed_response.json()["title"])
        self.assertEqual(204, delete_response.status_code)
        self.assertEqual([], default_list_response.json()["items"])
        self.assertEqual("archived", archived_list_response.json()["items"][0]["status"])
        self.assertEqual("新的会话标题", archived_list_response.json()["items"][0]["title"])

    def test_post_agent_message_stores_user_and_deterministic_assistant_reply(self):
        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                session_response = await client.post(
                    "/api/v1/agent/sessions",
                    json={"title": "AI 对话", "primary_intent": "agent_chat"},
                )
                if session_response.status_code != 201:
                    return session_response, None, None
                session_id = session_response.json()["id"]
                posted = await client.post(
                    f"/api/v1/agent/sessions/{session_id}/messages",
                    json={
                        "content_text": "帮我记住：我主要找 Java 后端岗位",
                        "runtime_content_text": "runtime-only prompt should not leak",
                    },
                )
                messages = await client.get(f"/api/v1/agent/sessions/{session_id}/messages")
                fetched_session = await client.get(f"/api/v1/agent/sessions/{session_id}")
                return posted, messages, fetched_session

        posted_response, messages_response, fetched_session_response = run(call_api())

        self.assertEqual(201, posted_response.status_code)
        self.assertIsNotNone(messages_response)
        self.assertIsNotNone(fetched_session_response)
        posted_payload = posted_response.json()
        self.assertEqual("帮我记住：我主要找 Java 后端岗位", posted_payload["user_message"]["content_text"])
        self.assertEqual("assistant", posted_payload["assistant_message"]["role"])
        self.assertIn("已经记录", posted_payload["assistant_message"]["content_text"])
        self.assertNotIn("runtime_content_text", posted_payload["user_message"])
        self.assertEqual(200, messages_response.status_code)
        messages = messages_response.json()["items"]
        self.assertEqual(["user", "assistant"], [message["role"] for message in messages])
        self.assertEqual("帮我记住：我主要找 Java 后端岗位", messages[0]["content_text"])
        self.assertIn("已经记录", messages[1]["content_text"])
        self.assertEqual(2, fetched_session_response.json()["message_count"])

    def test_post_agent_message_to_missing_session_returns_404(self):
        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.post(
                    "/api/v1/agent/sessions/missing-session/messages",
                    json={"content_text": "hello"},
                )

        response = run(call_api())

        self.assertEqual(404, response.status_code)
        self.assertIn("Agent session not found", response.json()["detail"])

    def test_post_agent_message_attaches_built_context_metadata_to_assistant_message(self):
        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                session_response = await client.post(
                    "/api/v1/agent/sessions",
                    json={"title": "上下文集成", "primary_intent": "agent_chat"},
                )
                session_id = session_response.json()["id"]
                return await client.post(
                    f"/api/v1/agent/sessions/{session_id}/messages",
                    json={"content_text": "这一轮应该先构建上下文"},
                )

        response = run(call_api())

        self.assertEqual(201, response.status_code)
        metadata = response.json()["assistant_message"]["metadata_json"]
        self.assertEqual("deterministic_stub", metadata["response_mode"])
        self.assertTrue(metadata["context_metadata"]["new_user_message_included"])
        self.assertEqual([], metadata["context_metadata"]["loaded_memory_ids"])
        self.assertEqual([], metadata["context_metadata"]["loaded_skill_ids"])
        self.assertTrue(metadata["context_metadata"]["agent_run_id"].startswith("agent-run-"))
        self.assertEqual("final_response", metadata["context_metadata"]["current_step"])
        self.assertIsNotNone(metadata["context_metadata"]["workflow_run_id"])

    def test_post_agent_message_returns_llm_assistant_reply_when_model_is_configured(self):
        from app.infrastructure.llm.chat_client import LLMChatCompletion

        class FakeLLMClient:
            def complete(self, *, messages):
                return LLMChatCompletion(content="模型回复：我会先确认你的 Java 秋招目标。")

        app = self._app(llm_client=FakeLLMClient())

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                session_response = await client.post(
                    "/api/v1/agent/sessions",
                    json={"title": "AI 对话", "primary_intent": "agent_chat"},
                )
                session_id = session_response.json()["id"]
                return await client.post(
                    f"/api/v1/agent/sessions/{session_id}/messages",
                    json={"content_text": "我想找 Java 后端秋招"},
                )

        response = run(call_api())

        self.assertEqual(201, response.status_code)
        assistant = response.json()["assistant_message"]
        self.assertEqual("模型回复：我会先确认你的 Java 秋招目标。", assistant["content_text"])
        self.assertEqual("llm", assistant["metadata_json"]["response_mode"])

    def test_post_agent_message_resumes_waiting_outer_session_task(self):
        from app.agent_runtime.durable_state.models import AgentStepState, AgentTaskState
        from app.agent_runtime.durable_state.schemas import AgentStepStatus, AgentTaskStatus
        from app.agent_runtime.graph_factory import AgentWorkflowResult
        from app.agent_runtime.state import AgentState

        workflow_results = []

        def fake_workflow(command, *, dependencies):
            index = len(workflow_results) + 1
            if index == 1:
                state = AgentState(
                    session_id=command.session_id,
                    workflow_run_id="workflow-outer-1",
                    agent_run_id="agent-run-outer-1",
                    user_message=command.user_message,
                    current_step="final_response",
                    final_response="请补充你的简历文本。",
                    response_mode="clarification_ask_user",
                    context_metadata={"outer_test_step": 1},
                )
            else:
                state = AgentState(
                    session_id=command.session_id,
                    workflow_run_id="workflow-outer-2",
                    agent_run_id="agent-run-outer-2",
                    user_message=command.user_message,
                    current_step="final_response",
                    final_response="已根据你的简历继续优化。",
                    response_mode="llm",
                    context_metadata={"outer_test_step": 2},
                )
            result = AgentWorkflowResult(workflow_run_id=state.workflow_run_id, state=state)
            workflow_results.append(result)
            return result

        with patch("app.api.v1.agent.run_agent_workflow", side_effect=fake_workflow):
            app = self._app()

            async def call_api():
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                    session_response = await client.post(
                        "/api/v1/agent/sessions",
                        json={"title": "outer session", "primary_intent": "agent_chat"},
                    )
                    session_id = session_response.json()["id"]
                    first = await client.post(
                        f"/api/v1/agent/sessions/{session_id}/messages",
                        json={"content_text": "帮我根据 Java 后端 JD 优化简历"},
                    )
                    after_first = await client.get(f"/api/v1/agent/sessions/{session_id}")
                    second = await client.post(
                        f"/api/v1/agent/sessions/{session_id}/messages",
                        json={"content_text": "这是我的简历文本：..."},
                    )
                    after_second = await client.get(f"/api/v1/agent/sessions/{session_id}")
                    return first, after_first, second, after_second

            first_response, after_first_response, second_response, after_second_response = run(call_api())

        self.assertEqual(201, first_response.status_code)
        self.assertEqual(201, second_response.status_code)
        first_outer = first_response.json()["assistant_message"]["metadata_json"]["context_metadata"]["outer_session_loop"]
        persisted_first_outer = after_first_response.json()["metadata_json"]["outer_session_loop"]
        second_outer = second_response.json()["assistant_message"]["metadata_json"]["context_metadata"]["outer_session_loop"]
        persisted_second_outer = after_second_response.json()["metadata_json"]["outer_session_loop"]

        self.assertEqual("waiting_user", first_outer["status"])
        self.assertEqual("请补充你的简历文本。", first_outer["waiting_message"])
        self.assertEqual(first_outer["active_task_id"], persisted_first_outer["active_task_id"])
        self.assertEqual("finished", second_outer["status"])
        self.assertEqual(first_outer["active_task_id"], second_outer["active_task_id"])
        self.assertEqual(2, second_outer["run_count"])
        self.assertEqual(["这是我的简历文本：..."], persisted_second_outer["user_followups"])
        self.assertEqual("已根据你的简历继续优化。", second_response.json()["assistant_message"]["content_text"])

        with self.Session() as db_session:
            durable_task = db_session.get(AgentTaskState, first_outer["active_task_id"])
            durable_steps = list(
                db_session.scalars(
                    select(AgentStepState)
                    .where(AgentStepState.task_id == first_outer["active_task_id"])
                    .order_by(AgentStepState.sequence_index)
                ).all()
            )

        self.assertIsNotNone(durable_task)
        self.assertEqual(AgentTaskStatus.SUCCEEDED, durable_task.status)
        self.assertEqual("outer_session_task", durable_task.task_type)
        self.assertEqual("agent.outer_session", durable_task.capability)
        self.assertEqual(["workflow-outer-1", "workflow-outer-2"], durable_task.output_payload["workflow_run_ids"])
        self.assertEqual(2, len(durable_steps))
        self.assertEqual([AgentStepStatus.WAITING_USER, AgentStepStatus.SUCCEEDED], [step.status for step in durable_steps])
        self.assertEqual(["workflow-outer-1", "workflow-outer-2"], [step.output_payload["workflow_run_id"] for step in durable_steps])

    def test_post_offerio_sync_request_routes_through_middleware_local_workflow(self):
        from app.infrastructure.llm.chat_client import LLMChatCompletion

        class FakeLLMClient:
            def complete(self, *, messages):
                return LLMChatCompletion(content="普通模型回复：需要 planner 决定是否同步 OfferIO。")

        with patch("app.agent_runtime.tool_registry._sync_offerio_company_jobs") as sync_mock:
            sync_mock.return_value = {
                "tool_name": "offerio.sync_company_jobs",
                "ok": True,
                "result": {
                    "source_name": "OfferIO 公司聚合岗位库",
                    "status": "succeeded",
                    "fetched_count": 50,
                    "extracted_count": 50,
                    "failed_count": 0,
                },
            }
            app = self._app(llm_client=FakeLLMClient())

            async def call_api():
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                    session_response = await client.post(
                        "/api/v1/agent/sessions",
                        json={"title": "OfferIO sync", "primary_intent": "agent_chat"},
                    )
                    session_id = session_response.json()["id"]
                    return await client.post(
                        f"/api/v1/agent/sessions/{session_id}/messages",
                        json={"content_text": "请从 OfferIO 公司聚合岗位库更新一下岗位"},
                    )

            response = run(call_api())

        self.assertEqual(201, response.status_code)
        assistant = response.json()["assistant_message"]
        self.assertIn("已从 OfferIO 公司聚合岗位库同步岗位", assistant["content_text"])
        self.assertEqual("tool_result_summary", assistant["metadata_json"]["response_mode"])
        self.assertEqual("local_workflow", assistant["metadata_json"]["context_metadata"]["capability_routing"]["route"])
        self.assertEqual({"limit": 1000}, assistant["metadata_json"]["context_metadata"]["capability_routing"]["tool_input"])
        self.assertEqual(1, sync_mock.call_count)

    def test_post_local_company_database_question_uses_readonly_overview_tool(self):
        from app.domains.jobs.models import (
            Company,
            JobLead,
            JobLeadStatus,
            JobSource,
            JobSourceFetchMode,
            JobSourceTrustLevel,
            JobSourceType,
            RecruitingSignal,
            RecruitingSignalType,
        )
        from app.infrastructure.llm.chat_client import LLMChatCompletion

        class FakeLLMClient:
            def complete(self, *, messages):
                return LLMChatCompletion(content="不应该走普通模型回复。")

        with self.Session() as session:
            source = JobSource(
                name="Local company source",
                source_type=JobSourceType.OFFICIAL_API,
                entry_url="https://example.com/jobs",
                trust_level=JobSourceTrustLevel.MEDIUM_HIGH,
                fetch_mode=JobSourceFetchMode.OFFICIAL_API,
            )
            session.add_all(
                [
                    source,
                    Company(name="Tencent", normalized_name="tencent"),
                    JobLead(
                        source=source,
                        lead_hash="lead-api-company-overview-1",
                        company_name="ByteDance",
                        title="ByteDance 校招岗位聚合",
                        skills=[],
                        trust_level=JobSourceTrustLevel.MEDIUM_HIGH,
                        verification_status=JobLeadStatus.VERIFIED,
                    ),
                    RecruitingSignal(
                        source=source,
                        signal_hash="signal-api-company-overview-1",
                        company_name="Meituan",
                        normalized_company_name="meituan",
                        signal_type=RecruitingSignalType.CAMPUS_RECRUITMENT_OPEN,
                        trust_level=JobSourceTrustLevel.MEDIUM_HIGH,
                    ),
                ]
            )
            session.commit()

        app = self._app(llm_client=FakeLLMClient())

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                session_response = await client.post(
                    "/api/v1/agent/sessions",
                    json={"title": "company database overview", "primary_intent": "agent_chat"},
                )
                session_id = session_response.json()["id"]
                return await client.post(
                    f"/api/v1/agent/sessions/{session_id}/messages",
                    json={"content_text": "我的数据库里现在有多少企业？"},
                )

        response = run(call_api())

        self.assertEqual(201, response.status_code)
        assistant = response.json()["assistant_message"]
        self.assertIn("可以看", assistant["content_text"])
        self.assertIn("正式企业表 1 家", assistant["content_text"])
        self.assertIn("岗位线索 1 条，去重企业 1 家", assistant["content_text"])
        self.assertIn("公司展览", assistant["content_text"])
        self.assertIn("不是同一个统计口径", assistant["content_text"])
        self.assertIn("文章/社媒信号暂不计入公司数", assistant["content_text"])
        self.assertNotIn("无法", assistant["content_text"])
        metadata = assistant["metadata_json"]["context_metadata"]
        self.assertEqual("local_company_database_overview", metadata["intent_frame"]["intent"])
        self.assertEqual("local_workflow", metadata["capability_routing"]["route"])
        self.assertEqual("local.company_database_overview", metadata["capability_routing"]["capability"])

    def test_post_job_source_count_question_uses_job_source_overview_tool(self):
        from app.infrastructure.llm.chat_client import LLMChatCompletion

        class FakeLLMClient:
            def complete(self, *, messages):
                return LLMChatCompletion(content="不应该走普通模型回复。")

        with patch("app.agent_runtime.tool_registry._local_job_source_overview") as overview_mock:
            overview_mock.return_value = {
                "tool_name": "local.job_source_overview",
                "ok": True,
                "result": {
                    "source_count": 3,
                    "enabled_source_count": 2,
                    "disabled_source_count": 1,
                    "unsynced_source_count": 1,
                    "sources_by_type": {"official_api": 1, "wechat_account": 2},
                    "sources_by_fetch_mode": {"official_api": 1, "mcp_visible_page": 2},
                    "sample_sources": [
                        {"name": "OfferIO 公司聚合岗位库", "source_type": "official_api", "enabled": True},
                        {"name": "OfferIO 开放岗位来源库", "source_type": "official_api", "enabled": True},
                    ],
                    "external_job_board": {
                        "ok": True,
                        "offerio_company_openings_total": 1247,
                        "offerio_company_jobs_total": 987,
                    },
                },
            }
            app = self._app(llm_client=FakeLLMClient())

            async def call_api():
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                    session_response = await client.post(
                        "/api/v1/agent/sessions",
                        json={"title": "job source count", "primary_intent": "agent_chat"},
                    )
                    session_id = session_response.json()["id"]
                    return await client.post(
                        f"/api/v1/agent/sessions/{session_id}/messages",
                        json={"content_text": "有多少岗位来源？"},
                    )

            response = run(call_api())

        self.assertEqual(201, response.status_code)
        assistant = response.json()["assistant_message"]
        self.assertIn("本地登记的岗位信息源共有 3 个", assistant["content_text"])
        self.assertIn("开放岗位公司库 1247 个", assistant["content_text"])
        self.assertIn("公司聚合岗位库 987 家", assistant["content_text"])
        self.assertIn("OfferIO 开放岗位公司库", assistant["content_text"])
        self.assertNotIn("OfferIO 开放岗位来源库", assistant["content_text"])
        self.assertIn("公司展览", assistant["content_text"])
        self.assertNotIn("岗位展览", assistant["content_text"])
        self.assertNotIn("正式企业表", assistant["content_text"])
        metadata = assistant["metadata_json"]["context_metadata"]
        self.assertEqual("local_job_source_overview", metadata["intent_frame"]["intent"])
        self.assertEqual("local_workflow", metadata["capability_routing"]["route"])
        self.assertEqual("local.job_source_overview", metadata["capability_routing"]["capability"])
        self.assertEqual(1, overview_mock.call_count)

    def test_post_find_apply_entry_request_does_not_auto_call_tool_without_planner(self):
        from app.domains.jobs.models import (
            JobLead,
            JobLeadStatus,
            JobSource,
            JobSourceFetchMode,
            JobSourceTrustLevel,
            JobSourceType,
        )
        from app.infrastructure.llm.chat_client import LLMChatCompletion

        class FakeLLMClient:
            def complete(self, *, messages):
                return LLMChatCompletion(content="I cannot access the company's application page.")

        with self.Session() as session:
            source = JobSource(
                name="Campus leads API",
                source_type=JobSourceType.OFFICIAL_API,
                entry_url="https://example.com/jobs",
                trust_level=JobSourceTrustLevel.MEDIUM_HIGH,
                fetch_mode=JobSourceFetchMode.OFFICIAL_API,
            )
            session.add(
                JobLead(
                    id="lead-api-apply-1",
                    source=source,
                    lead_hash="lead-api-apply-1",
                    company_name="Tencent",
                    title="Backend Engineer Intern",
                    source_url="https://careers.tencent.com/job/1",
                    apply_url="https://careers.tencent.com/apply/1",
                    skills=[],
                    trust_level=JobSourceTrustLevel.MEDIUM_HIGH,
                    verification_status=JobLeadStatus.VERIFIED,
                )
            )
            session.commit()

        app = self._app(llm_client=FakeLLMClient())

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                session_response = await client.post(
                    "/api/v1/agent/sessions",
                    json={"title": "Apply entry", "primary_intent": "agent_chat"},
                )
                session_id = session_response.json()["id"]
                return await client.post(
                    f"/api/v1/agent/sessions/{session_id}/messages",
                    json={"content_text": "open application entry for job_id=lead-api-apply-1"},
                )

        response = run(call_api())

        self.assertEqual(201, response.status_code)
        assistant = response.json()["assistant_message"]
        self.assertEqual("I cannot access the company's application page.", assistant["content_text"])
        self.assertEqual("llm", assistant["metadata_json"]["response_mode"])

    def test_stream_agent_message_returns_sse_tokens_and_persists_turn(self):
        class FakeStreamingLLMClient:
            def stream_complete(self, *, messages):
                self.messages = messages
                yield "你"
                yield "好"

        app = self._app(llm_client=FakeStreamingLLMClient())

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                session_response = await client.post(
                    "/api/v1/agent/sessions",
                    json={"title": "AI 瀵硅瘽", "primary_intent": "agent_chat"},
                )
                session_id = session_response.json()["id"]
                stream_response = await client.post(
                    f"/api/v1/agent/sessions/{session_id}/messages/stream",
                    json={"content_text": "你好"},
                )
                messages_response = await client.get(f"/api/v1/agent/sessions/{session_id}/messages")
                session_after_response = await client.get(f"/api/v1/agent/sessions/{session_id}")
                return stream_response, messages_response, session_after_response

        stream_response, messages_response, session_after_response = run(call_api())

        self.assertEqual(200, stream_response.status_code)
        self.assertIn("text/event-stream", stream_response.headers["content-type"])
        stream_text = stream_response.text
        self.assertIn("event: token", stream_text)
        self.assertIn('"content":"你"', stream_text)
        self.assertIn('"content":"好"', stream_text)
        self.assertIn("event: done", stream_text)
        outer_events = _sse_payloads(stream_text, "outer_session_event")
        self.assertEqual(["task_started", "task_finished"], [event["event_type"] for event in outer_events])
        self.assertEqual("running", outer_events[0]["status"])
        self.assertEqual("finished", outer_events[-1]["status"])

        done_payload = _sse_payload(stream_text, "done")
        done_outer = done_payload["context_metadata"]["outer_session_loop"]
        self.assertEqual("finished", done_outer["status"])
        self.assertEqual(1, done_outer["run_count"])
        self.assertEqual("finished", session_after_response.json()["metadata_json"]["outer_session_loop"]["status"])

        messages = messages_response.json()["items"]
        self.assertEqual(["user", "assistant"], [message["role"] for message in messages])
        self.assertEqual("你好", messages[0]["content_text"])
        self.assertEqual("你好", messages[1]["content_text"])
        self.assertEqual("llm_stream", messages[1]["metadata_json"]["response_mode"])

    def test_stream_agent_message_emits_waiting_outer_session_event_for_clarification(self):
        from app.agent_runtime.graph_factory import AgentPreparedResponse, AgentWorkflowResult
        from app.agent_runtime.state import AgentState

        def fake_prepare(command, *, dependencies, on_workflow_started=None):
            state = AgentState(
                session_id=command.session_id,
                workflow_run_id="workflow-stream-waiting-1",
                agent_run_id="agent-run-stream-waiting-1",
                user_message=command.user_message,
                current_step="final_response",
                final_response="请补充你的简历文本。",
                response_mode="clarification_ask_user",
                context_metadata={"stream_outer_test": "waiting"},
            )
            return AgentPreparedResponse(workflow_run_id=state.workflow_run_id, workflow=None, state=state)

        def fake_finalize(state, *, final_response, response_mode, dependencies):
            final_state = state.with_updates(final_response=final_response, response_mode=response_mode)
            return AgentWorkflowResult(workflow_run_id=final_state.workflow_run_id, state=final_state)

        with patch("app.api.v1.agent.prepare_agent_workflow_response", side_effect=fake_prepare), patch(
            "app.api.v1.agent.finalize_agent_workflow_response",
            side_effect=fake_finalize,
        ):
            app = self._app()

            async def call_api():
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                    session_response = await client.post(
                        "/api/v1/agent/sessions",
                        json={"title": "stream waiting", "primary_intent": "agent_chat"},
                    )
                    session_id = session_response.json()["id"]
                    stream_response = await client.post(
                        f"/api/v1/agent/sessions/{session_id}/messages/stream",
                        json={"content_text": "帮我优化简历"},
                    )
                    session_after_response = await client.get(f"/api/v1/agent/sessions/{session_id}")
                    return stream_response, session_after_response

            stream_response, session_after_response = run(call_api())

        self.assertEqual(200, stream_response.status_code)
        stream_text = stream_response.text
        outer_events = _sse_payloads(stream_text, "outer_session_event")
        self.assertEqual(["task_started", "waiting_user"], [event["event_type"] for event in outer_events])
        self.assertEqual("请补充你的简历文本。", outer_events[-1]["waiting_message"])

        done_payload = _sse_payload(stream_text, "done")
        done_outer = done_payload["context_metadata"]["outer_session_loop"]
        self.assertEqual("waiting_user", done_outer["status"])
        self.assertEqual("请补充你的简历文本。", done_outer["waiting_message"])
        self.assertEqual("waiting_user", session_after_response.json()["metadata_json"]["outer_session_loop"]["status"])

    def test_stream_agent_message_sanitizes_internal_tool_protocol_before_tokens(self):
        from app.agent_runtime.graph_factory import AgentPreparedResponse, AgentWorkflowResult
        from app.agent_runtime.state import AgentState

        def fake_prepare(command, *, dependencies, on_workflow_started=None):
            state = AgentState(
                session_id=command.session_id,
                workflow_run_id="workflow-stream-sanitize-1",
                agent_run_id="agent-run-stream-sanitize-1",
                user_message=command.user_message,
                current_step="final_response",
                final_response='**OfferMaster AI**\nTool call: external.web_search{"query":"C罗 本周 比赛日程"}',
                response_mode="llm_tool_choice_loop",
                context_metadata={"stream_sanitize_test": True},
            )
            return AgentPreparedResponse(workflow_run_id=state.workflow_run_id, workflow=None, state=state)

        def fake_finalize(state, *, final_response, response_mode, dependencies):
            final_state = state.with_updates(final_response=final_response, response_mode=response_mode)
            return AgentWorkflowResult(workflow_run_id=final_state.workflow_run_id, state=final_state)

        with patch("app.api.v1.agent.prepare_agent_workflow_response", side_effect=fake_prepare), patch(
            "app.api.v1.agent.finalize_agent_workflow_response",
            side_effect=fake_finalize,
        ):
            app = self._app()

            async def call_api():
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                    session_response = await client.post(
                        "/api/v1/agent/sessions",
                        json={"title": "stream sanitize", "primary_intent": "agent_chat"},
                    )
                    session_id = session_response.json()["id"]
                    stream_response = await client.post(
                        f"/api/v1/agent/sessions/{session_id}/messages/stream",
                        json={"content_text": "c罗这个星期有什么比赛吗"},
                    )
                    messages_response = await client.get(f"/api/v1/agent/sessions/{session_id}/messages")
                    return stream_response, messages_response

            stream_response, messages_response = run(call_api())

        self.assertEqual(200, stream_response.status_code)
        self.assertNotIn("Tool call:", stream_response.text)
        self.assertNotIn("external.web_search", stream_response.text)
        self.assertIn("最终回答需要重新整理", stream_response.text)
        assistant = [message for message in messages_response.json()["items"] if message["role"] == "assistant"][0]
        self.assertNotIn("Tool call:", assistant["content_text"])
        self.assertNotIn("external.web_search", assistant["content_text"])

    def test_stream_agent_message_emits_tool_events_from_loop_trace(self):
        from app.agent_runtime.graph_factory import AgentPreparedResponse, AgentWorkflowResult
        from app.agent_runtime.state import AgentState

        def fake_prepare(command, *, dependencies, on_workflow_started=None):
            state = AgentState(
                session_id=command.session_id,
                workflow_run_id="workflow-stream-tool-trace-1",
                agent_run_id="agent-run-stream-tool-trace-1",
                user_message=command.user_message,
                current_step="final_response",
                final_response="已找到中科曙光校招官网：https://jobs.example.com/sugon",
                response_mode="llm_tool_loop",
                context_metadata={
                    "loop_agent": {
                        "enabled": True,
                        "reflection_retry_count": 1,
                        "trace": [
                            {
                                "iteration": 1,
                                "action": "call_tool",
                                "capability": "external.web_search",
                                "observation_status": "succeeded",
                                "observation_summary": "搜索结果是百科页面，没有校招入口。",
                                "tool_call_id": "tool-call-bad-1",
                                "metadata": {
                                    "tool_input_keys": ["max_results", "query"],
                                    "reflection": {
                                        "quality": "bad",
                                        "next_action": "retry",
                                        "confidence": 0.9,
                                        "reason": "结果没有命中校招官网。",
                                        "suggested_input_patch": {"query": "中科曙光 校园招聘 官网 2026"},
                                    },
                                },
                            },
                            {
                                "iteration": 2,
                                "action": "call_tool",
                                "capability": "external.web_search",
                                "observation_status": "succeeded",
                                "observation_summary": "找到中科曙光校园招聘官网。",
                                "tool_call_id": "tool-call-good-2",
                                "metadata": {
                                    "tool_input_keys": ["max_results", "query"],
                                    "reflection": {
                                        "quality": "good",
                                        "next_action": "continue",
                                        "confidence": 0.95,
                                        "reason": "结果命中校招官网。",
                                    },
                                },
                            },
                        ],
                    }
                },
            )
            return AgentPreparedResponse(workflow_run_id=state.workflow_run_id, workflow=None, state=state)

        def fake_finalize(state, *, final_response, response_mode, dependencies):
            final_state = state.with_updates(final_response=final_response, response_mode=response_mode)
            return AgentWorkflowResult(workflow_run_id=final_state.workflow_run_id, state=final_state)

        with patch("app.api.v1.agent.prepare_agent_workflow_response", side_effect=fake_prepare), patch(
            "app.api.v1.agent.finalize_agent_workflow_response",
            side_effect=fake_finalize,
        ):
            app = self._app()

            async def call_api():
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                    session_response = await client.post(
                        "/api/v1/agent/sessions",
                        json={"title": "stream tool trace", "primary_intent": "agent_chat"},
                    )
                    session_id = session_response.json()["id"]
                    return await client.post(
                        f"/api/v1/agent/sessions/{session_id}/messages/stream",
                        json={"content_text": "搜一下中科曙光校招官网"},
                    )

            stream_response = run(call_api())

        self.assertEqual(200, stream_response.status_code)
        tool_events = _sse_payloads(stream_response.text, "tool_event")
        self.assertEqual(
            ["tool_started", "tool_finished", "tool_reflection_retry", "tool_started", "tool_finished"],
            [event["event_type"] for event in tool_events],
        )
        self.assertEqual("external.web_search", tool_events[0]["tool_name"])
        self.assertEqual("tool-call-bad-1", tool_events[1]["tool_call_id"])
        self.assertEqual("搜索结果是百科页面，没有校招入口。", tool_events[1]["summary"])
        self.assertEqual("retry", tool_events[2]["reflection"]["next_action"])
        self.assertEqual("中科曙光 校园招聘 官网 2026", tool_events[2]["suggested_input_patch"]["query"])
        self.assertEqual("tool-call-good-2", tool_events[-1]["tool_call_id"])

    def test_stream_offerio_sync_request_routes_through_middleware_local_workflow(self):
        class FakeStreamingLLMClient:
            def stream_complete(self, *, messages):
                yield "普通流式回复："
                yield "需要 planner 决定是否同步。"

        with patch("app.agent_runtime.tool_registry._sync_offerio_company_jobs") as sync_mock:
            sync_mock.return_value = {
                "tool_name": "offerio.sync_company_jobs",
                "ok": True,
                "result": {
                    "source_name": "OfferIO 公司聚合岗位库",
                    "status": "succeeded",
                    "fetched_count": 50,
                    "extracted_count": 50,
                    "failed_count": 0,
                },
            }
            app = self._app(llm_client=FakeStreamingLLMClient())

            async def call_api():
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                    session_response = await client.post(
                        "/api/v1/agent/sessions",
                        json={"title": "OfferIO sync", "primary_intent": "agent_chat"},
                    )
                    session_id = session_response.json()["id"]
                    stream_response = await client.post(
                        f"/api/v1/agent/sessions/{session_id}/messages/stream",
                        json={"content_text": "请从 OfferIO 公司聚合岗位库更新一下岗位"},
                    )
                    messages_response = await client.get(f"/api/v1/agent/sessions/{session_id}/messages")
                    return stream_response, messages_response

            stream_response, messages_response = run(call_api())

        self.assertEqual(200, stream_response.status_code)
        self.assertIn("已从 OfferIO 公司聚合岗位库同步岗位", stream_response.text)
        self.assertEqual(1, sync_mock.call_count)
        tool_events = _sse_payloads(stream_response.text, "tool_event")
        self.assertEqual(["tool_started", "tool_finished"], [event["event_type"] for event in tool_events])
        self.assertEqual("offerio.sync_company_jobs", tool_events[0]["tool_name"])
        self.assertEqual("succeeded", tool_events[1]["status"])

        messages = messages_response.json()["items"]
        self.assertEqual(["assistant", "tool_call", "tool_result", "user"], sorted(message["role"] for message in messages))
        self.assertIn("已从 OfferIO 公司聚合岗位库同步岗位", messages[-1]["content_text"])
        self.assertEqual("tool_result_summary", messages[-1]["metadata_json"]["response_mode"])

    def test_stream_agent_message_emits_tool_started_before_tool_finishes(self):
        import json
        from queue import Empty, Queue

        from app.agent_runtime.understanding.intent_detector import HybridIntentDetector
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry
        from app.domains.conversations.repository import ConversationRepository
        from app.domains.conversations.schemas import AgentUserMessageRequest
        from app.domains.conversations.service import ConversationService
        from app.api.v1 import agent as agent_api

        tool_entered = threading.Event()
        allow_tool_finish = threading.Event()

        def slow_tool(_session, *, query: str, limit: int = 10):
            tool_entered.set()
            allow_tool_finish.wait(timeout=2)
            return {"tool_name": "memory_search", "ok": True, "result": {"items": [], "query": query, "limit": limit}}

        def fake_registry(*args, **kwargs):
            return AgentToolRegistry(
                [
                    AgentToolDefinition(
                        name="memory_search",
                        description="Search memory.",
                        input_schema={"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}},
                        output_schema={"type": "object"},
                        handler=slow_tool,
                        allowed_source_types=frozenset({"agent_chat"}),
                    )
                ]
            )

        with self.Session() as session:
            service = ConversationService(ConversationRepository(session))
            created = service.create_session(title="realtime tool", primary_intent="agent_chat")
            session.commit()
            session_id = created.id
            db_bind = session.get_bind()

        event_queue = Queue()
        with patch("app.api.v1.agent.create_default_agent_tool_registry", side_effect=fake_registry), patch.object(
            agent_api,
            "_build_agent_llm_client",
            return_value=None,
        ), patch.object(
            agent_api,
            "_build_agent_intent_detector",
            return_value=HybridIntentDetector(llm_client=None),
        ):
            worker = threading.Thread(
                target=agent_api._run_agent_message_stream_worker,
                kwargs={
                    "session_id": session_id,
                    "request": AgentUserMessageRequest(
                        content_text="查一下记忆",
                        requested_tool_name="memory_search",
                        tool_input={"query": "秋招", "limit": 1},
                    ),
                    "db_bind": db_bind,
                    "event_queue": event_queue,
                },
                daemon=True,
            )
            worker.start()

            payload = None
            for _ in range(20):
                item = event_queue.get(timeout=0.2)
                if item is agent_api.STREAM_QUEUE_DONE:
                    break
                if "event: tool_event" not in str(item):
                    continue
                data_line = next(line for line in str(item).splitlines() if line.startswith("data:"))
                candidate = json.loads(data_line.removeprefix("data:").strip())
                if candidate.get("event_type") == "tool_started":
                    payload = candidate
                    break
            try:
                self.assertIsNotNone(payload)
                self.assertEqual("memory_search", payload["tool_name"])
                self.assertTrue(tool_entered.is_set())
                self.assertFalse(allow_tool_finish.is_set())
            finally:
                allow_tool_finish.set()
                worker.join(timeout=3)

        if worker.is_alive():
            raise AssertionError("stream worker did not finish after releasing slow tool")

    def test_stream_external_search_result_uses_main_llm_synthesis(self):
        from app.agent_runtime.understanding.intent_detector import HybridIntentDetector
        from app.infrastructure.llm.chat_client import LLMChatCompletion

        class FakeIntentLLM:
            def complete(self, *, messages):
                return LLMChatCompletion(
                    content=(
                        '{"intent":"campus_recruiting_search","confidence":0.95,'
                        '"needs_external_info":true,"risk_level":"low",'
                        '"entities":{"company_names":["公牛集团"],"keywords":["校园招聘"],"time_range":"latest"}}'
                    )
                )

        test_case = self

        class FakeStreamingLLMClient:
            def __init__(self):
                self.calls = []

            def stream_complete(self, *, messages):
                self.calls.append(messages)
                combined = "\n".join(str(message.get("content") or "") for message in messages)
                test_case.assertIn("芝加哥公牛队", combined)
                test_case.assertIn("公牛集团校园招聘", combined)
                test_case.assertIn("不要向用户展示无关结果", combined)
                test_case.assertIn("不要解释过滤过程", combined)
                yield "公牛集团校招入口："
                yield "https://campus.gongniu.cn/"

        def fake_external_search_executor(query, max_results):
            return {
                "executor_name": "claude-sdk-agent",
                "answer": (
                    "联网搜索结果：\n"
                    "- 公牛集团校园招聘：https://campus.gongniu.cn/\n"
                    "- 芝加哥公牛队_百度百科：NBA 球队资料：https://baike.baidu.com/item/bulls"
                ),
                "sources": [
                    {"title": "公牛集团校园招聘", "url": "https://campus.gongniu.cn/"},
                    {"title": "芝加哥公牛队_百度百科", "url": "https://baike.baidu.com/item/bulls"},
                ],
            }

        class FakeClaudeSdkAgent:
            def call(self, task, context):
                from app.agent_runtime.agent_as_tool import StandardAgentResult
                from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL

                search_result = fake_external_search_executor(task.input_payload["query"], task.input_payload.get("max_results", 5))
                return StandardAgentResult(
                    status="succeeded",
                    summary=search_result["answer"],
                    raw_result={"tool_name": EXTERNAL_WEB_SEARCH_TOOL, "ok": True, "result": search_result},
                )

        fake_llm = FakeStreamingLLMClient()
        with patch("app.api.v1.agent.build_external_web_search_callback", return_value=fake_external_search_executor), patch(
            "app.api.v1.agent.build_agent_runtime_executor_bundle",
            return_value=(
                {"claude-sdk-agent": FakeClaudeSdkAgent()},
                {"external.web_search": "claude-sdk-agent"},
            ),
        ):
            app = self._app(
                llm_client=fake_llm,
                intent_detector=HybridIntentDetector(llm_client=FakeIntentLLM()),
            )

            async def call_api():
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                    session_response = await client.post(
                        "/api/v1/agent/sessions",
                        json={"title": "external search synthesis", "primary_intent": "agent_chat"},
                    )
                    session_id = session_response.json()["id"]
                    stream_response = await client.post(
                        f"/api/v1/agent/sessions/{session_id}/messages/stream",
                        json={"content_text": "你搜索一下公牛的校园招聘"},
                    )
                    messages_response = await client.get(f"/api/v1/agent/sessions/{session_id}/messages")
                    return stream_response, messages_response

            stream_response, messages_response = run(call_api())

        self.assertEqual(200, stream_response.status_code)
        stream_text = stream_response.text
        self.assertIn('"content":"公牛集团校招入口："', stream_text)
        self.assertIn('"content":"https://campus.gongniu.cn/"', stream_text)
        self.assertNotIn('"content":"联网搜索结果：', stream_text)
        self.assertNotIn("NBA", stream_text)
        self.assertNotIn("芝加哥", stream_text)
        self.assertNotIn("过滤", stream_text)
        self.assertEqual(1, len(fake_llm.calls))

        messages = messages_response.json()["items"]
        self.assertIn("公牛集团校招入口：https://campus.gongniu.cn/", messages[-1]["content_text"])
        self.assertNotIn("NBA", messages[-1]["content_text"])
        self.assertNotIn("芝加哥", messages[-1]["content_text"])
        self.assertNotIn("过滤", messages[-1]["content_text"])
        self.assertEqual("llm_stream_tool_result_summary", messages[-1]["metadata_json"]["response_mode"])

    def test_xiaohongshu_search_uses_configured_rest_adapter_from_agent_chat(self):
        from app.core.config import get_settings
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        query = "\u8bf7\u5728\u5c0f\u7ea2\u4e66\u641c\u7d22 2027 \u79cb\u62db Java \u5c97\u4f4d"

        class FakeToolChoiceLLM:
            def complete(self, *, messages, tools=None, tool_choice=None):
                combined = "\n".join(str(message.get("content") or "") for message in messages)
                if tools and "Tencent 2027" not in combined:
                    self_tool_names = [tool["function"]["name"] for tool in tools]
                    self_tool_names.sort()
                    assert self_tool_names == ["xiaohongshu_mcp_search_feeds"]
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-xhs-api-search",
                                name="xiaohongshu_mcp_search_feeds",
                                arguments={"keyword": query},
                            )
                        ],
                    )
                return LLMChatCompletion(content="小红书搜索完成，找到 Tencent 2027 相关内容。")

        class FakeResponse:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return {"success": True, "data": {"items": [{"title": "Tencent 2027"}]}, "message": "ok"}

        rest_calls = []

        def fake_post(url, *, json, headers, timeout):
            rest_calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
            return FakeResponse()

        with patch.dict(
            "os.environ",
            {"JOBPILOT_XIAOHONGSHU_MCP_BASE_URL": "http://127.0.0.1:18060/"},
            clear=False,
        ), patch("app.mcp_gateway.content_source_client.httpx.post", side_effect=fake_post):
            get_settings.cache_clear()
            try:
                app = self._app(llm_client=FakeToolChoiceLLM())

                async def call_api():
                    transport = ASGITransport(app=app)
                    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                        session_response = await client.post(
                            "/api/v1/agent/sessions",
                            json={"title": "xhs search", "primary_intent": "agent_chat"},
                        )
                        session_id = session_response.json()["id"]
                        posted = await client.post(
                            f"/api/v1/agent/sessions/{session_id}/messages",
                            json={"content_text": query},
                        )
                        messages = await client.get(f"/api/v1/agent/sessions/{session_id}/messages")
                        return posted, messages

                posted_response, messages_response = run(call_api())
            finally:
                get_settings.cache_clear()

        self.assertEqual(201, posted_response.status_code)
        self.assertEqual(
            [
                {
                    "url": "http://127.0.0.1:18060/api/v1/feeds/search",
                    "json": {"keyword": query, "filters": None},
                    "headers": {},
                    "timeout": 30.0,
                }
            ],
            rest_calls,
        )
        messages = messages_response.json()["items"]
        self.assertEqual(["assistant", "tool_call", "tool_result", "user"], sorted(message["role"] for message in messages))
        tool_result = next(message for message in messages if message["role"] == "tool_result")
        self.assertTrue(tool_result["content_json"]["result"]["ok"])
        self.assertEqual("xiaohongshu_rest", tool_result["content_json"]["result"]["metadata"]["adapter"])

    def test_stream_agent_message_emits_approval_required_for_skill_ask_tool(self):
        self._create_approval_memory_skill()
        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                session_response = await client.post(
                    "/api/v1/agent/sessions",
                    json={"title": "approval stream", "primary_intent": "agent_chat"},
                )
                session_id = session_response.json()["id"]
                stream_response = await client.post(
                    f"/api/v1/agent/sessions/{session_id}/messages/stream",
                    json={
                        "content_text": "approval memory",
                        "requested_tool_name": "memory_search",
                        "source_type": "agent_chat",
                    },
                )
                messages_response = await client.get(f"/api/v1/agent/sessions/{session_id}/messages")
                return stream_response, messages_response

        stream_response, messages_response = run(call_api())

        self.assertEqual(200, stream_response.status_code)
        self.assertIn("event: user_message", stream_response.text)
        self.assertIn("event: approval_required", stream_response.text)
        self.assertIn('"tool_name":"memory_search"', stream_response.text)
        self.assertIn('"permission_decision":"ask"', stream_response.text)
        self.assertNotIn("event: error", stream_response.text)

        messages = messages_response.json()["items"]
        self.assertEqual(["user"], [message["role"] for message in messages])

    def test_stream_local_company_overview_runs_without_approval_outside_skill_allowlist(self):
        self._create_company_allowlist_skill_without_local_tool()
        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                session_response = await client.post(
                    "/api/v1/agent/sessions",
                    json={"title": "local company overview", "primary_intent": "agent_chat"},
                )
                session_id = session_response.json()["id"]
                stream_response = await client.post(
                    f"/api/v1/agent/sessions/{session_id}/messages/stream",
                    json={
                        "content_text": "company database overview",
                        "requested_tool_name": "local.company_database_overview",
                        "source_type": "agent_chat",
                    },
                )
                messages_response = await client.get(f"/api/v1/agent/sessions/{session_id}/messages")
                return stream_response, messages_response

        stream_response, messages_response = run(call_api())

        self.assertEqual(200, stream_response.status_code)
        self.assertIn("event: user_message", stream_response.text)
        self.assertIn("event: done", stream_response.text)
        self.assertNotIn("event: approval_required", stream_response.text)
        self.assertNotIn("outside the active Skill automatic permissions", stream_response.text)

        roles = [message["role"] for message in messages_response.json()["items"]]
        self.assertEqual(["user", "tool_call", "tool_result", "assistant"], roles)

    def test_approve_agent_approval_executes_waiting_tool_and_persists_assistant_reply(self):
        self._create_approval_memory_skill()
        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                session_response = await client.post(
                    "/api/v1/agent/sessions",
                    json={"title": "approval approve", "primary_intent": "agent_chat"},
                )
                session_id = session_response.json()["id"]
                stream_response = await client.post(
                    f"/api/v1/agent/sessions/{session_id}/messages/stream",
                    json={
                        "content_text": "approval memory",
                        "requested_tool_name": "memory_search",
                        "source_type": "agent_chat",
                    },
                )
                approval_payload = _sse_payload(stream_response.text, "approval_required")
                approve_response = await client.post(
                    f"/api/v1/agent/approvals/{approval_payload['approval']['id']}/approve",
                    json={"decision_reason": "user approved skill ask tool"},
                )
                messages_response = await client.get(f"/api/v1/agent/sessions/{session_id}/messages")
                session_after_response = await client.get(f"/api/v1/agent/sessions/{session_id}")
                return approve_response, messages_response, session_after_response, approval_payload

        approve_response, messages_response, session_after_response, approval_payload = run(call_api())

        self.assertEqual(200, approve_response.status_code)
        payload = approve_response.json()
        self.assertEqual("approved", payload["approval"]["status"])
        self.assertEqual("assistant", payload["assistant_message"]["role"])
        self.assertEqual("final_response", payload["context_metadata"]["current_step"])
        self.assertEqual("finished", payload["context_metadata"]["outer_session_loop"]["status"])
        self.assertEqual(
            approval_payload["context_metadata"]["outer_session_loop"]["active_task_id"],
            payload["context_metadata"]["outer_session_loop"]["active_task_id"],
        )
        self.assertEqual("finished", session_after_response.json()["metadata_json"]["outer_session_loop"]["status"])

        roles = [message["role"] for message in messages_response.json()["items"]]
        self.assertEqual(["user", "tool_call", "tool_result", "assistant"], roles)

        with self.Session() as db_session:
            from app.agent_runtime.durable_state.models import AgentTaskState
            from app.agent_runtime.durable_state.schemas import AgentTaskStatus

            outer_task_id = payload["context_metadata"]["outer_session_loop"]["active_task_id"]
            durable_task = db_session.get(AgentTaskState, outer_task_id)

        self.assertIsNotNone(durable_task)
        self.assertEqual(AgentTaskStatus.SUCCEEDED, durable_task.status)

    def _create_approval_memory_skill(self) -> None:
        from app.agent_runtime.memory.skill_repository import AgentSkillRepository
        from app.domains.agent_memory.repository import AgentMemoryRepository
        from app.domains.agent_memory.schemas import AgentSkillCreate

        with self.Session() as session:
            AgentSkillRepository(AgentMemoryRepository(session)).create_skill(
                AgentSkillCreate(
                    name="approval-memory-search",
                    title="Approval Memory Search",
                    description="Use this skill when the user asks for approval memory context.",
                    category="agent_guardrail",
                    metadata_json={"allowed_tools": ["memory_search"], "ask_tools": ["memory_search"]},
                    sections={"workflow": "Ask before memory_search for approval memory."},
                )
            )
            session.commit()

    def _create_company_allowlist_skill_without_local_tool(self) -> None:
        from app.agent_runtime.memory.skill_repository import AgentSkillRepository
        from app.domains.agent_memory.repository import AgentMemoryRepository
        from app.domains.agent_memory.schemas import AgentSkillCreate

        with self.Session() as session:
            AgentSkillRepository(AgentMemoryRepository(session)).create_skill(
                AgentSkillCreate(
                    name="company-overview-allowlist",
                    title="Company Overview Allowlist",
                    description="Use this skill when the user asks for company database overview.",
                    category="agent_guardrail",
                    metadata_json={"allowed_tools": ["memory_search"]},
                    sections={"workflow": "Company overview can use memory context, but local read-only overview is a runtime capability."},
                )
            )
            session.commit()


def _sse_payloads(stream_text: str, event_name: str) -> list[dict]:
    import json

    payloads = []
    for raw_event in stream_text.split("\n\n"):
        if f"event: {event_name}" not in raw_event:
            continue
        for line in raw_event.splitlines():
            if line.startswith("data: "):
                payloads.append(json.loads(line.removeprefix("data: ")))
            elif line.startswith("data:"):
                payloads.append(json.loads(line.removeprefix("data:")))
    return payloads


def _sse_payload(stream_text: str, event_name: str) -> dict:
    payloads = _sse_payloads(stream_text, event_name)
    if payloads:
        return payloads[0]
    raise AssertionError(f"SSE event not found: {event_name}")


if __name__ == "__main__":
    unittest.main()
