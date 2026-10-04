import sys
import shutil
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


def _model_selected_api_tool(messages, tools=None, tool_choice=None):
    """Model-driven API fixture: select a registered alias, then let runtime summarize."""

    from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

    if not tools or any(message.get("role") == "tool" for message in messages):
        return LLMChatCompletion(content="")
    joined = "\n".join(str(message.get("content") or "") for message in messages)
    aliases = {
        tool["function"]["name"]: tool["function"]["name"]
        for tool in tools
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    }

    def choose(prefix: str):
        return next((name for name in aliases if name.startswith(prefix)), None)

    if "OfferIO" in joined:
        tool_name = choose("offerio_sync_company_jobs")
        arguments = {"limit": 1000} if tool_name else {}
    elif "岗位来源" in joined or "岗位信息源" in joined:
        tool_name = choose("local_job_source_overview")
        arguments = {"mode": "sources"} if tool_name else {}
    elif "当前数据库中的公司数" in joined or "公司数有多少" in joined:
        tool_name = choose("local_job_source_overview")
        arguments = {"mode": "company_board_count"} if tool_name else {}
    elif "数据库" in joined and "关于" in joined:
        tool_name = choose("local_company_database_overview") or choose("database_company_search")
        arguments = {"sample_limit": 10} if tool_name and tool_name.startswith("local_") else {"company_names": ["京东"]}
    elif "数据库" in joined or "企业" in joined or "公司" in joined:
        tool_name = choose("local_company_database_overview")
        arguments = {"sample_limit": 10} if tool_name else {}
    else:
        return LLMChatCompletion(content="")

    if not tool_name:
        return LLMChatCompletion(content="")
    return LLMChatCompletion(
        content="",
        tool_calls=[LLMToolCall(id="api-model-tool-call", name=tool_name, arguments=arguments)],
    )


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

    def _skill_root(self, test_name: str) -> Path:
        root = PROJECT_ROOT / ".tmp-test-artifacts" / "agent-api" / test_name
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        return root

    def _write_copy_script(self, root: Path) -> None:
        scripts_dir = root / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        (scripts_dir / "copy_file.py").write_text(
            "import argparse\n"
            "import shutil\n"
            "from pathlib import Path\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--src', required=True)\n"
            "parser.add_argument('--dst', required=True)\n"
            "parser.add_argument('--overwrite', action='store_true')\n"
            "args = parser.parse_args()\n"
            "src = Path(args.src)\n"
            "dst = Path(args.dst)\n"
            "if dst.exists() and not args.overwrite:\n"
            "    raise SystemExit(1)\n"
            "dst.parent.mkdir(parents=True, exist_ok=True)\n"
            "shutil.copy2(src, dst)\n"
            "print(f'COPIED:{src}->{dst}')\n",
            encoding="utf-8",
        )

    def _patch_filesystem_skill_executor(self, skill_root: Path) -> None:
        from app.agent_runtime.agent_as_tool import FILESYSTEM_SKILL_EXECUTOR_ID
        from app.agent_runtime.skills.filesystem_executor import FilesystemSkillExecutor
        from app.api.v1 import agent as agent_api

        original_dependencies_builder = agent_api._agent_graph_dependencies

        def build_dependencies(db_session, conversation_service):
            dependencies = original_dependencies_builder(db_session, conversation_service)
            return dependencies.with_agent_runtime(
                executors={
                    FILESYSTEM_SKILL_EXECUTOR_ID: FilesystemSkillExecutor(
                        script_root=skill_root,
                        session_provider=lambda: db_session,
                    )
                }
            )

        patcher = patch.object(agent_api, "_agent_graph_dependencies", side_effect=build_dependencies)
        patcher.start()
        self._patchers.append(patcher)

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

    def test_runtime_result_summary_preserves_filesystem_trace_for_sse_events(self):
        from app.api.v1.agent import _runtime_result_summary, _runtime_result_summary_text

        metadata = {
            "result_observation": {
                "filesystem_trace": {
                    "operation": "copy_file",
                    "precheck": {
                        "source_path": "A.tex",
                        "source_exists_before": True,
                        "target_path": "B.tex",
                        "target_exists_before": False,
                    },
                    "script": {"internal_script": "copy_file.py", "ok": True, "return_code": 0},
                    "postcheck": {
                        "completed": True,
                        "source_path": "A.tex",
                        "target_path": "B.tex",
                        "source_exists_after": True,
                        "target_exists_after": True,
                        "reason": "file_to_file_postcheck_passed",
                    },
                },
                "goal_validation": {"completed": True, "next_action": "finish"},
            }
        }

        summary = _runtime_result_summary(metadata)

        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual("copy_file", summary["filesystem_trace"]["operation"])
        self.assertTrue(summary["filesystem_trace"]["postcheck"]["completed"])
        self.assertEqual(True, summary["goal_completed"])
        self.assertIn("已执行并复核", _runtime_result_summary_text(summary))

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
        self.assertEqual("running", payload["task"]["status"])
        self.assertEqual(payload["resume_step_id"], payload["task"]["current_step_id"])
        self.assertEqual(["failed", "pending"], [step["status"] for step in payload["task"]["steps"]])
        self.assertEqual("step-api-resume-1", payload["task"]["steps"][1]["parent_step_id"])

    def test_recover_agent_session_latest_task_without_client_knowing_task_id(self):
        from app.agent_runtime.durable_state.repository import SqlAlchemyDurableStateRepository
        from app.agent_runtime.durable_state.schemas import AgentStepStatus
        from app.agent_runtime.durable_state.service import DurableStateService

        app = self._app()

        async def create_session():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.post(
                    "/api/v1/agent/sessions",
                    json={"title": "恢复任务", "primary_intent": "agent_chat"},
                )
                return response.json()["id"]

        session_id = run(create_session())

        with self.Session() as session:
            service = DurableStateService(SqlAlchemyDurableStateRepository(session))
            service.create_task(
                task_id="outer-job-api-recover",
                root_workflow_run_id="workflow-api-recover",
                conversation_session_id=session_id,
                task_type="outer_session_task",
                capability="agent.outer_session",
                user_goal="帮我找 20 家上海 Java 后端秋招公司",
            )
            for index, stage in enumerate(
                [
                    ("clarify_goal", "明确目标和约束", "agent.stage.clarify_goal", AgentStepStatus.SUCCEEDED),
                    ("collect_candidates", "收集本地候选信息", "agent.stage.collect_candidates", AgentStepStatus.SUCCEEDED),
                    ("enrich_external_info", "补充外部公开信息", "agent.stage.enrich_external_info", AgentStepStatus.FAILED),
                    ("analyze_rank", "分析匹配和排序", "agent.stage.analyze_rank", AgentStepStatus.PENDING),
                    ("finalize_answer", "整理最终输出", "agent.stage.finalize_answer", AgentStepStatus.PENDING),
                ],
                start=1,
            ):
                service.add_step(
                    task_id="outer-job-api-recover",
                    step_id=f"outer-job-api-recover:stage-{index}",
                    sequence_index=index * 100,
                    step_type="workflow_plan_stage",
                    status=stage[3],
                    executor_type="planner",
                    executor_name="offermaster_stage_planner",
                    capability=stage[2],
                    input_payload={"stage_id": stage[0], "stage_index": index, "title": stage[1]},
                    output_payload={"execution_status": stage[3].value},
                )
            step = service.add_step(
                task_id="outer-job-api-recover",
                step_id="outer-job-api-recover:tool-1",
                sequence_index=1101,
                step_type="workflow_tool_call",
                executor_type="tool_registry",
                executor_name="agent_tool_registry",
                capability="external.web_search",
                input_payload={"query": "上海 Java 后端 秋招 公司"},
            )
            step.retry_count = 1
            service.mark_step_failed("outer-job-api-recover:tool-1", output_payload={"error": "temporary timeout"})
            session.commit()

        async def recover_session():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.post(f"/api/v1/agent/sessions/{session_id}/tasks/recover")

        response = run(recover_session())

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual("retry_failed_step", payload["action"])
        self.assertEqual("outer-job-api-recover", payload["task_id"])
        self.assertEqual("outer-job-api-recover", payload["task"]["id"])
        self.assertEqual("running", payload["task"]["status"])
        self.assertEqual(payload["resume_step_id"], payload["task"]["current_step_id"])
        execution_steps = [step for step in payload["task"]["steps"] if step["step_type"] != "workflow_plan_stage"]
        self.assertEqual(["workflow_tool_call", "workflow_tool_call"], [step["step_type"] for step in execution_steps])
        self.assertEqual("external.web_search", execution_steps[1]["capability"])
        self.assertEqual({"query": "上海 Java 后端 秋招 公司"}, payload["payload"])
        self.assertEqual("enrich_external_info", payload["resume_stage_id"])
        self.assertEqual("补充外部公开信息", payload["resume_stage_title"])
        self.assertEqual("outer-job-api-recover:stage-3", payload["resume_stage_step_id"])
        self.assertEqual("failed", payload["resume_stage_status"])

    def test_recover_and_run_agent_session_latest_task_passes_stage_context_to_workflow(self):
        from app.agent_runtime.durable_state.repository import SqlAlchemyDurableStateRepository
        from app.agent_runtime.durable_state.schemas import AgentStepStatus
        from app.agent_runtime.durable_state.service import DurableStateService
        from app.agent_runtime.graph_factory import AgentWorkflowResult
        from app.agent_runtime.state import AgentState

        captured_user_messages = []

        def fake_run_agent_workflow(command, *, dependencies):
            captured_user_messages.append(command.user_message)
            state = AgentState(
                session_id=command.session_id,
                workflow_run_id="workflow-stage-recover-run",
                agent_run_id="agent-run-stage-recover-run",
                user_message=command.user_message,
                current_step="final_response",
                final_response="已从补充外部公开信息阶段继续。",
                response_mode="stage_recover_test",
            )
            return AgentWorkflowResult(workflow_run_id=state.workflow_run_id, state=state)

        with patch("app.api.v1.agent.run_agent_workflow", side_effect=fake_run_agent_workflow):
            app = self._app()

            async def create_session():
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                    response = await client.post(
                        "/api/v1/agent/sessions",
                        json={"title": "阶段恢复执行", "primary_intent": "agent_chat"},
                    )
                    return response.json()["id"]

            session_id = run(create_session())

            with self.Session() as session:
                service = DurableStateService(SqlAlchemyDurableStateRepository(session))
                service.create_task(
                    task_id="outer-job-api-stage-recover-run",
                    root_workflow_run_id="workflow-api-stage-recover-run",
                    conversation_session_id=session_id,
                    task_type="outer_session_task",
                    capability="agent.outer_session",
                    user_goal="帮我找 20 家适合我的秋招公司",
                )
                service.add_step(
                    task_id="outer-job-api-stage-recover-run",
                    step_id="outer-job-api-stage-recover-run:stage-3",
                    sequence_index=300,
                    step_type="workflow_plan_stage",
                    status=AgentStepStatus.FAILED,
                    executor_type="planner",
                    executor_name="offermaster_stage_planner",
                    capability="agent.stage.enrich_external_info",
                    input_payload={"stage_id": "enrich_external_info", "stage_index": 3, "title": "补充外部公开信息"},
                    output_payload={"execution_status": "failed"},
                )
                step = service.add_step(
                    task_id="outer-job-api-stage-recover-run",
                    step_id="outer-job-api-stage-recover-run:tool-1",
                    sequence_index=1101,
                    step_type="workflow_tool_call",
                    executor_type="tool_registry",
                    executor_name="agent_tool_registry",
                    capability="external.web_search",
                    input_payload={"query": "上海 秋招 公司 公开信息"},
                )
                step.retry_count = 1
                service.mark_step_failed("outer-job-api-stage-recover-run:tool-1", output_payload={"error": "timeout"})
                session.commit()

            async def recover_and_run_session():
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                    return await client.post(f"/api/v1/agent/sessions/{session_id}/tasks/recover/run")

            response = run(recover_and_run_session())

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertTrue(payload["executed"])
        self.assertEqual("enrich_external_info", payload["resume"]["resume_stage_id"])
        self.assertEqual("补充外部公开信息", payload["resume"]["resume_stage_title"])
        self.assertEqual(
            [
                "帮我找 20 家适合我的秋招公司\n\n阶段级恢复上下文：\n- 恢复阶段：补充外部公开信息\n- 阶段标识：enrich_external_info"
            ],
            captured_user_messages,
        )

    def test_recover_and_run_failed_stage_reenters_stage_loop_instead_of_direct_tool_retry(self):
        from app.agent_runtime.durable_state.repository import SqlAlchemyDurableStateRepository
        from app.agent_runtime.durable_state.schemas import AgentStepStatus
        from app.agent_runtime.durable_state.service import DurableStateService
        from app.agent_runtime.graph_factory import AgentWorkflowResult
        from app.agent_runtime.state import AgentState

        captured_commands = []

        def fake_run_agent_workflow(command, *, dependencies):
            captured_commands.append(command)
            state = AgentState(
                session_id=command.session_id,
                workflow_run_id="workflow-stage-loop-recover",
                agent_run_id="agent-run-stage-loop-recover",
                user_message=command.user_message,
                current_step="final_response",
                final_response="已从失败阶段重新进入阶段 loop 并继续。",
                response_mode="stage_loop_recover_test",
            )
            return AgentWorkflowResult(workflow_run_id=state.workflow_run_id, state=state)

        with patch("app.api.v1.agent.run_agent_workflow", side_effect=fake_run_agent_workflow):
            app = self._app()

            async def create_session():
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                    response = await client.post(
                        "/api/v1/agent/sessions",
                        json={"title": "失败阶段恢复", "primary_intent": "agent_chat"},
                    )
                    return response.json()["id"]

            session_id = run(create_session())

            with self.Session() as session:
                service = DurableStateService(SqlAlchemyDurableStateRepository(session))
                service.create_task(
                    task_id="outer-job-api-stage-loop-recover",
                    root_workflow_run_id="workflow-api-stage-loop-recover",
                    conversation_session_id=session_id,
                    task_type="outer_session_task",
                    capability="agent.outer_session",
                    user_goal="帮我找 20 家适合我的秋招公司，并补充公开资料",
                )
                for index, stage in enumerate(
                    [
                        ("clarify_goal", "明确目标和约束", "agent.stage.clarify_goal", AgentStepStatus.SUCCEEDED),
                        ("collect_candidates", "收集本地候选信息", "agent.stage.collect_candidates", AgentStepStatus.SUCCEEDED),
                        ("enrich_external_info", "补充外部公开信息", "agent.stage.enrich_external_info", AgentStepStatus.FAILED),
                        ("analyze_rank", "分析匹配和排序", "agent.stage.analyze_rank", AgentStepStatus.PENDING),
                        ("finalize_answer", "整理最终输出", "agent.stage.finalize_answer", AgentStepStatus.PENDING),
                    ],
                    start=1,
                ):
                    service.add_step(
                        task_id="outer-job-api-stage-loop-recover",
                        step_id=f"outer-job-api-stage-loop-recover:stage-{index}",
                        sequence_index=index * 100,
                        step_type="workflow_plan_stage",
                        status=stage[3],
                        executor_type="planner",
                        executor_name="offermaster_stage_planner",
                        capability=stage[2],
                        input_payload={"stage_id": stage[0], "stage_index": index, "title": stage[1]},
                        output_payload={"execution_status": stage[3].value},
                    )
                step = service.add_step(
                    task_id="outer-job-api-stage-loop-recover",
                    step_id="outer-job-api-stage-loop-recover:tool-1",
                    sequence_index=1101,
                    step_type="workflow_tool_call",
                    executor_type="tool_registry",
                    executor_name="agent_tool_registry",
                    capability="external.web_search",
                    input_payload={"query": "秋招 公司 公开资料"},
                )
                step.retry_count = 1
                service.mark_step_failed("outer-job-api-stage-loop-recover:tool-1", output_payload={"error": "timeout"})
                session.commit()

            async def recover_and_run_session():
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                    return await client.post(f"/api/v1/agent/sessions/{session_id}/tasks/recover/run")

            response = run(recover_and_run_session())

        self.assertEqual(200, response.status_code)
        self.assertEqual(1, len(captured_commands))
        self.assertIsNone(captured_commands[0].requested_tool_name)
        self.assertEqual({}, captured_commands[0].tool_input)
        self.assertIn("阶段标识：enrich_external_info", captured_commands[0].user_message)

    def test_recover_and_run_agent_session_latest_task_executes_retry_payload(self):
        from app.agent_runtime.durable_state.repository import SqlAlchemyDurableStateRepository
        from app.agent_runtime.durable_state.service import DurableStateService
        from app.domains.jobs.models import Company

        app = self._app()

        async def create_session():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.post(
                    "/api/v1/agent/sessions",
                    json={"title": "恢复并执行", "primary_intent": "agent_chat"},
                )
                return response.json()["id"]

        session_id = run(create_session())

        with self.Session() as session:
            session.add(Company(name="Tencent", normalized_name="tencent"))
            service = DurableStateService(SqlAlchemyDurableStateRepository(session))
            service.create_task(
                task_id="outer-job-api-recover-run",
                root_workflow_run_id="workflow-api-recover-run",
                conversation_session_id=session_id,
                task_type="outer_session_task",
                capability="agent.outer_session",
                user_goal="我的数据库里现在有多少企业？",
            )
            step = service.add_step(
                task_id="outer-job-api-recover-run",
                step_id="outer-job-api-recover-run:tool-1",
                sequence_index=1101,
                step_type="workflow_tool_call",
                executor_type="tool_registry",
                executor_name="agent_tool_registry",
                capability="local.company_database_overview",
                input_payload={"sample_limit": 20},
            )
            step.retry_count = 1
            service.mark_step_failed("outer-job-api-recover-run:tool-1", output_payload={"error": "temporary timeout"})
            session.commit()

        async def recover_and_run_session():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.post(f"/api/v1/agent/sessions/{session_id}/tasks/recover/run")
                messages = await client.get(f"/api/v1/agent/sessions/{session_id}/messages")
                return response, messages

        response, messages_response = run(recover_and_run_session())

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertTrue(payload["executed"])
        self.assertEqual("retry_failed_step", payload["resume"]["action"])
        self.assertIn("正式企业表 1 家", payload["assistant_message"]["content_text"])
        self.assertEqual("succeeded", payload["task"]["status"])
        self.assertEqual("local.company_database_overview", payload["task"]["steps"][-2]["capability"])
        self.assertEqual("workflow_final_response", payload["task"]["steps"][-1]["step_type"])
        resume_step = next(step for step in payload["task"]["steps"] if step["id"] == payload["resume"]["resume_step_id"])
        self.assertEqual("succeeded", resume_step["status"])
        self.assertEqual(payload["assistant_message"]["workflow_run_id"], resume_step["output_payload"]["recovery_workflow_run_id"])
        assistant_messages = [message for message in messages_response.json()["items"] if message["role"] == "assistant"]
        self.assertEqual(1, len(assistant_messages))
        self.assertIn("正式企业表 1 家", assistant_messages[0]["content_text"])

    def test_enqueue_agent_session_followup_for_latest_task_updates_outer_queue(self):
        from app.agent_runtime.durable_state.repository import SqlAlchemyDurableStateRepository
        from app.agent_runtime.durable_state.schemas import AgentTaskStatus
        from app.agent_runtime.durable_state.service import DurableStateService

        app = self._app()

        async def create_session():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.post(
                    "/api/v1/agent/sessions",
                    json={"title": "运行中插话", "primary_intent": "agent_chat"},
                )
                return response.json()["id"]

        session_id = run(create_session())

        with self.Session() as session:
            service = DurableStateService(SqlAlchemyDurableStateRepository(session))
            task = service.create_task(
                task_id="outer-job-api-followup",
                root_workflow_run_id="workflow-api-followup",
                conversation_session_id=session_id,
                task_type="outer_session_task",
                capability="agent.outer_session",
                user_goal="帮我找 20 家秋招公司",
            )
            task.status = AgentTaskStatus.RUNNING
            session.commit()

        async def enqueue_followup():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                queued = await client.post(
                    f"/api/v1/agent/sessions/{session_id}/tasks/followups",
                    json={"content_text": "只看上海公司"},
                )
                session_after = await client.get(f"/api/v1/agent/sessions/{session_id}")
                return queued, session_after

        response, session_response = run(enqueue_followup())

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual("outer-job-api-followup", payload["task_id"])
        self.assertEqual(1, payload["queued_count"])
        self.assertEqual(["只看上海公司"], payload["user_followups"])
        self.assertEqual("outer-job-api-followup", payload["task"]["id"])

        outer_metadata = session_response.json()["metadata_json"]["outer_session_loop"]
        self.assertEqual("outer-job-api-followup", outer_metadata["active_task_id"])
        self.assertEqual("running", outer_metadata["status"])
        self.assertEqual(["只看上海公司"], outer_metadata["user_followups"])

    def test_recover_and_run_agent_session_consumes_queued_followups_in_user_message(self):
        from app.agent_runtime.durable_state.repository import SqlAlchemyDurableStateRepository
        from app.agent_runtime.durable_state.service import DurableStateService
        from app.agent_runtime.graph_factory import AgentWorkflowResult
        from app.agent_runtime.state import AgentState

        captured_user_messages = []

        def fake_run_agent_workflow(command, *, dependencies):
            captured_user_messages.append(command.user_message)
            state = AgentState(
                session_id=command.session_id,
                workflow_run_id="workflow-recovered-with-followup",
                agent_run_id="agent-run-recovered-with-followup",
                user_message=command.user_message,
                current_step="final_response",
                final_response="已按补充要求继续执行。",
                response_mode="recovered_test",
            )
            return AgentWorkflowResult(workflow_run_id=state.workflow_run_id, state=state)

        with patch("app.api.v1.agent.run_agent_workflow", side_effect=fake_run_agent_workflow):
            app = self._app()

            async def create_session():
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                    response = await client.post(
                        "/api/v1/agent/sessions",
                        json={"title": "恢复时带插话", "primary_intent": "agent_chat"},
                    )
                    return response.json()["id"]

            session_id = run(create_session())

            with self.Session() as session:
                service = DurableStateService(SqlAlchemyDurableStateRepository(session))
                service.create_task(
                    task_id="outer-job-api-recover-followup",
                    root_workflow_run_id="workflow-api-recover-followup",
                    conversation_session_id=session_id,
                    task_type="outer_session_task",
                    capability="agent.outer_session",
                    user_goal="帮我找 20 家秋招公司",
                )
                step = service.add_step(
                    task_id="outer-job-api-recover-followup",
                    step_id="outer-job-api-recover-followup:tool-1",
                    sequence_index=1101,
                    step_type="workflow_tool_call",
                    executor_type="tool_registry",
                    executor_name="agent_tool_registry",
                    capability="external.web_search",
                    input_payload={"query": "秋招 公司"},
                )
                step.retry_count = 1
                service.mark_step_failed("outer-job-api-recover-followup:tool-1", output_payload={"error": "timeout"})
                session.commit()

            async def enqueue_and_recover():
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                    await client.post(
                        f"/api/v1/agent/sessions/{session_id}/tasks/followups",
                        json={"content_text": "只看上海公司"},
                    )
                    recovered = await client.post(f"/api/v1/agent/sessions/{session_id}/tasks/recover/run")
                    session_after = await client.get(f"/api/v1/agent/sessions/{session_id}")
                    return recovered, session_after

            response, session_response = run(enqueue_and_recover())

        self.assertEqual(200, response.status_code)
        self.assertTrue(response.json()["executed"])
        self.assertEqual(
            [
                "帮我找 20 家秋招公司\n\n运行中用户补充要求：\n- 只看上海公司\n\n阶段级恢复上下文：\n- 恢复阶段：补充外部公开信息\n- 阶段标识：enrich_external_info"
            ],
            captured_user_messages,
        )
        self.assertEqual("已按补充要求继续执行。", response.json()["assistant_message"]["content_text"])
        self.assertEqual(0, response.json()["task"]["output_payload"]["pending_followup_count"])
        self.assertEqual(["只看上海公司"], response.json()["task"]["output_payload"]["last_consumed_followups"])

        outer_metadata = session_response.json()["metadata_json"]["outer_session_loop"]
        self.assertEqual([], outer_metadata["user_followups"])
        self.assertEqual(["只看上海公司"], outer_metadata["metadata"]["last_consumed_followups"])

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




    def test_agent_session_tasks_endpoint_lists_persisted_outer_task(self):
        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                session_response = await client.post(
                    "/api/v1/agent/sessions",
                    json={"title": "任务单", "primary_intent": "agent_chat"},
                )
                session_id = session_response.json()["id"]
                message_response = await client.post(
                    f"/api/v1/agent/sessions/{session_id}/messages",
                    json={"content_text": "帮我找 20 家上海 Java 后端秋招公司"},
                )
                tasks_response = await client.get(f"/api/v1/agent/sessions/{session_id}/tasks")
                return message_response, tasks_response

        message_response, tasks_response = run(call_api())

        self.assertEqual(201, message_response.status_code)
        self.assertEqual(200, tasks_response.status_code)
        payload = tasks_response.json()
        self.assertEqual(1, len(payload["items"]))
        task = payload["items"][0]
        outer = message_response.json()["assistant_message"]["metadata_json"]["context_metadata"]["outer_session_loop"]
        self.assertEqual(outer["active_task_id"], task["id"])
        self.assertEqual("outer_session_task", task["task_type"])
        self.assertEqual("agent.outer_session", task["capability"])
        self.assertEqual("succeeded", task["status"])
        self.assertEqual("帮我找 20 家上海 Java 后端秋招公司", task["user_goal"])
        self.assertEqual([message_response.json()["assistant_message"]["workflow_run_id"]], task["output_payload"]["workflow_run_ids"])

    def test_agent_task_detail_endpoint_returns_steps_for_task_sheet(self):
        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                session_response = await client.post(
                    "/api/v1/agent/sessions",
                    json={"title": "任务详情", "primary_intent": "agent_chat"},
                )
                session_id = session_response.json()["id"]
                message_response = await client.post(
                    f"/api/v1/agent/sessions/{session_id}/messages",
                    json={"content_text": "帮我查腾讯校招"},
                )
                task_id = message_response.json()["assistant_message"]["metadata_json"]["context_metadata"]["outer_session_loop"]["active_task_id"]
                detail_response = await client.get(f"/api/v1/agent/tasks/{task_id}")
                return task_id, detail_response

        task_id, detail_response = run(call_api())

        self.assertEqual(200, detail_response.status_code)
        payload = detail_response.json()
        self.assertEqual(task_id, payload["id"])
        self.assertEqual("succeeded", payload["status"])
        execution_steps = [step for step in payload["steps"] if step["step_type"] != "workflow_plan_stage"]
        self.assertEqual(
            ["outer_session_turn", "workflow_context", "workflow_final_response"],
            [step["step_type"] for step in execution_steps],
        )
        self.assertEqual("succeeded", execution_steps[0]["status"])
        self.assertEqual("agent.outer_session", execution_steps[0]["capability"])
        self.assertEqual("agent.context_builder", execution_steps[1]["capability"])
        self.assertEqual("agent.final_response", execution_steps[2]["capability"])

    def test_agent_task_detail_includes_default_plan_stage_steps(self):
        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                session_response = await client.post(
                    "/api/v1/agent/sessions",
                    json={"title": "多阶段规划", "primary_intent": "agent_chat"},
                )
                session_id = session_response.json()["id"]
                message_response = await client.post(
                    f"/api/v1/agent/sessions/{session_id}/messages",
                    json={"content_text": "帮我找 20 家适合我的秋招公司，并告诉我哪些最值得投"},
                )
                task_id = message_response.json()["assistant_message"]["metadata_json"]["context_metadata"]["outer_session_loop"]["active_task_id"]
                detail_response = await client.get(f"/api/v1/agent/tasks/{task_id}")
                return detail_response

        detail_response = run(call_api())

        self.assertEqual(200, detail_response.status_code)
        payload = detail_response.json()
        stage_steps = [step for step in payload["steps"] if step["step_type"] == "workflow_plan_stage"]
        self.assertEqual(
            [
                "agent.stage.clarify_goal",
                "agent.stage.collect_candidates",
                "agent.stage.enrich_external_info",
                "agent.stage.analyze_rank",
                "agent.stage.finalize_answer",
            ],
            [step["capability"] for step in stage_steps],
        )
        self.assertEqual(["succeeded", "skipped", "skipped", "skipped", "succeeded"], [step["status"] for step in stage_steps])
        self.assertEqual("明确目标和约束", stage_steps[0]["input_payload"]["title"])
        self.assertEqual("整理最终输出", stage_steps[-1]["input_payload"]["title"])

    def test_agent_task_plan_endpoint_returns_readable_stage_plan(self):
        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                session_response = await client.post(
                    "/api/v1/agent/sessions",
                    json={"title": "计划查询", "primary_intent": "agent_chat"},
                )
                session_id = session_response.json()["id"]
                message_response = await client.post(
                    f"/api/v1/agent/sessions/{session_id}/messages",
                    json={"content_text": "帮我找 20 家适合我的秋招公司，并告诉我哪些最值得投"},
                )
                task_id = message_response.json()["assistant_message"]["metadata_json"]["context_metadata"]["outer_session_loop"]["active_task_id"]
                plan_response = await client.get(f"/api/v1/agent/tasks/{task_id}/plan")
                return task_id, plan_response

        task_id, plan_response = run(call_api())

        self.assertEqual(200, plan_response.status_code)
        payload = plan_response.json()
        self.assertEqual(task_id, payload["task_id"])
        self.assertEqual("帮我找 20 家适合我的秋招公司，并告诉我哪些最值得投", payload["user_goal"])
        self.assertEqual("agent.stage.finalize_answer", payload["current_stage_id"])
        self.assertEqual(5, len(payload["stages"]))
        self.assertEqual("明确目标和约束", payload["stages"][0]["title"])
        self.assertEqual("agent.stage.collect_candidates", payload["stages"][1]["capability"])
        self.assertEqual("succeeded", payload["stages"][0]["status"])
        self.assertTrue(payload["stages"][0]["step_id"].endswith(":stage-1"))

    def test_tool_choice_loop_receives_outer_stage_context_before_execution(self):
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        class StageAwareLLMClient:
            def __init__(self) -> None:
                self.seen_prompt = ""

            def complete(self, *, messages, tools=None, tool_choice=None):
                self.seen_prompt = "\n".join(str(message.get("content") or "") for message in messages)
                self_ref.assertIn("当前任务阶段：明确目标和约束", self.seen_prompt)
                self_ref.assertIn("阶段标识：clarify_goal", self.seen_prompt)
                self_ref.assertIn("阶段目标：确认用户要完成什么、有什么范围限制、最终需要什么输出。", self.seen_prompt)
                return LLMChatCompletion(content="我已理解当前阶段。")

        self_ref = self
        llm_client = StageAwareLLMClient()
        app = self._app(llm_client=llm_client)

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                session_response = await client.post(
                    "/api/v1/agent/sessions",
                    json={"title": "阶段上下文", "primary_intent": "agent_chat"},
                )
                session_id = session_response.json()["id"]
                message_response = await client.post(
                    f"/api/v1/agent/sessions/{session_id}/messages",
                    json={"content_text": "Canonical Ltd. 是做什么的？主要业务是什么？"},
                )
                task_id = message_response.json()["assistant_message"]["metadata_json"]["context_metadata"]["outer_session_loop"]["active_task_id"]
                plan_response = await client.get(f"/api/v1/agent/tasks/{task_id}/plan")
                return message_response, plan_response

        message_response, plan_response = run(call_api())

        self.assertEqual(201, message_response.status_code)
        self.assertEqual(200, plan_response.status_code)
        self.assertIn("当前任务阶段：明确目标和约束", llm_client.seen_prompt)

    def test_agent_task_plan_endpoint_exposes_stage_business_actions_and_tool_strategies(self):
        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                session_response = await client.post(
                    "/api/v1/agent/sessions",
                    json={"title": "阶段工具策略", "primary_intent": "agent_chat"},
                )
                session_id = session_response.json()["id"]
                message_response = await client.post(
                    f"/api/v1/agent/sessions/{session_id}/messages",
                    json={"content_text": "帮我分析数据库里的公司机会，并补充公开资料"},
                )
                task_id = message_response.json()["assistant_message"]["metadata_json"]["context_metadata"]["outer_session_loop"]["active_task_id"]
                return await client.get(f"/api/v1/agent/tasks/{task_id}/plan")

        plan_response = run(call_api())

        self.assertEqual(200, plan_response.status_code)
        stages = plan_response.json()["stages"]
        collect_stage = stages[1]
        enrich_stage = stages[2]
        analyze_stage = stages[3]
        self.assertIn("本地", collect_stage["business_action"])
        self.assertEqual(
            ["database.company_list", "local.company_database_overview", "local.job_source_overview"],
            collect_stage["allowed_capabilities"],
        )
        self.assertEqual(["external.web_search"], enrich_stage["allowed_capabilities"][:1])
        self.assertEqual("none", analyze_stage["tool_strategy"]["mode"])
        self.assertIn("技术匹配度", analyze_stage["ranking_policy"][0])

    def test_stage_pipeline_runs_local_then_external_then_analysis_before_final_answer(self):
        from app.agent_runtime.understanding.schemas import IntentFrame
        from app.domains.jobs.models import Company
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        class FakeNormalIntentDetector:
            def detect(self, _message):
                return IntentFrame(intent="normal_chat", confidence=0.0)

        test_case = self

        class FakeStagePipelineLLM:
            def __init__(self) -> None:
                self.tool_names_by_call: list[list[str]] = []
                self.prompts: list[str] = []

            def complete(self, *, messages, tools=None, tool_choice=None):
                tool_names = [tool["function"]["name"] for tool in tools or []]
                joined_prompt = "\n".join(str(message.get("content") or "") for message in messages)
                self.tool_names_by_call.append(tool_names)
                self.prompts.append(joined_prompt)
                if len(self.tool_names_by_call) == 1:
                    test_case.assertIn("local_company_database_overview", tool_names)
                    test_case.assertIn("external_web_search", tool_names)
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-local-candidates",
                                name="local_company_database_overview",
                                arguments={"sample_limit": 2},
                            )
                        ],
                    )
                if len(self.tool_names_by_call) == 2:
                    test_case.assertIn("external_web_search", tool_names)
                    test_case.assertNotIn("local_company_database_overview", tool_names)
                    test_case.assertIn("阶段标识：enrich_external_info", joined_prompt)
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-public-info",
                                name="external_web_search",
                                arguments={"query": "Canonical Ltd. 主要业务 校招", "max_results": 3},
                            )
                        ],
                    )
                test_case.assertEqual([], tool_names)
                test_case.assertIn("阶段标识：analyze_rank", joined_prompt)
                test_case.assertIn("技术匹配度", joined_prompt)
                return LLMChatCompletion(content="推荐排序：1. Canonical Ltd.，理由是技术匹配且公开资料完整。")

        def fake_external_web_search(query: str, max_results: int):
            return {
                "answer": "Canonical Ltd. 是 Ubuntu 背后的公司，提供企业 Linux、云和安全支持。",
                "sources": [{"title": "Canonical", "url": "https://canonical.com"}],
            }

        fake_llm = FakeStagePipelineLLM()
        search_patcher = patch("app.api.v1.agent.build_external_web_search_callback", return_value=fake_external_web_search)
        search_patcher.start()
        self._patchers.append(search_patcher)
        app = self._app(llm_client=fake_llm, intent_detector=FakeNormalIntentDetector())

        with self.Session() as session:
            session.add(Company(name="Canonical Ltd.", normalized_name="canonical ltd"))
            session.commit()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                session_response = await client.post(
                    "/api/v1/agent/sessions",
                    json={"title": "阶段完整链", "primary_intent": "agent_chat"},
                )
                session_id = session_response.json()["id"]
                message_response = await client.post(
                    f"/api/v1/agent/sessions/{session_id}/messages",
                    json={"content_text": "帮我分析数据库里的公司机会，并补充公开资料，最后给我推荐排序"},
                )
                task_id = message_response.json()["assistant_message"]["metadata_json"]["context_metadata"]["outer_session_loop"]["active_task_id"]
                plan_response = await client.get(f"/api/v1/agent/tasks/{task_id}/plan")
                return message_response, plan_response

        message_response, plan_response = run(call_api())

        self.assertEqual(201, message_response.status_code)
        self.assertIn("推荐排序", message_response.json()["assistant_message"]["content_text"])
        first_stage_tools = set(fake_llm.tool_names_by_call[0])
        self.assertTrue({"external_web_search", "local_company_database_overview"}.issubset(first_stage_tools))
        self.assertIn(
            "model_driven_capability_catalog",
            message_response.json()["assistant_message"]["metadata_json"]["context_metadata"]["tool_candidate_selection"]["signals"],
        )
        # The model-driven loop receives the complete capability catalog for
        # the current source/stage. It is no longer reduced to one keyword-
        # selected tool before the model makes its decision.
        self.assertIn("external_web_search", fake_llm.tool_names_by_call[1])
        self.assertNotIn("local_company_database_overview", fake_llm.tool_names_by_call[1])
        self.assertEqual([], fake_llm.tool_names_by_call[2])
        stages = {stage["stage_id"]: stage for stage in plan_response.json()["stages"]}
        self.assertEqual("succeeded", stages["collect_candidates"]["status"])
        self.assertEqual("succeeded", stages["enrich_external_info"]["status"])
        self.assertEqual("succeeded", stages["analyze_rank"]["status"])
        self.assertEqual("succeeded", stages["finalize_answer"]["status"])

    def test_tool_choice_loop_advances_to_next_stage_from_outer_stage_plan(self):
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        class StageAdvancingLLMClient:
            def __init__(self) -> None:
                self.prompts: list[str] = []

            def complete(self, *, messages, tools=None, tool_choice=None):
                joined_prompt = "\n".join(str(message.get("content") or "") for message in messages)
                self.prompts.append(joined_prompt)
                if len(self.prompts) == 1:
                    test_case.assertIn("当前任务阶段：明确目标和约束", joined_prompt)
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-canonical-search",
                                name="external_web_search",
                                arguments={"query": "Canonical Ltd. 主要业务", "max_results": 3},
                            )
                        ],
                    )
                test_case.assertIn("当前任务阶段：分析匹配和排序", messages[-1]["content"])
                test_case.assertIn("阶段标识：analyze_rank", messages[-1]["content"])
                return LLMChatCompletion(content="已根据阶段计划进入分析匹配。")

        test_case = self
        llm_client = StageAdvancingLLMClient()
        def fake_external_web_search(query: str, max_results: int):
            return {"answer": "Canonical 是 Ubuntu 背后的公司。", "sources": [{"title": "Canonical", "url": "https://canonical.com"}]}

        search_patcher = patch("app.api.v1.agent.build_external_web_search_callback", return_value=fake_external_web_search)
        search_patcher.start()
        self._patchers.append(search_patcher)
        app = self._app(llm_client=llm_client)

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                session_response = await client.post(
                    "/api/v1/agent/sessions",
                    json={"title": "阶段自动推进", "primary_intent": "agent_chat"},
                )
                session_id = session_response.json()["id"]
                message_response = await client.post(
                    f"/api/v1/agent/sessions/{session_id}/messages",
                    json={"content_text": "Canonical Ltd. 是做什么的？主要业务是什么？"},
                )
                task_id = message_response.json()["assistant_message"]["metadata_json"]["context_metadata"]["outer_session_loop"]["active_task_id"]
                plan_response = await client.get(f"/api/v1/agent/tasks/{task_id}/plan")
                return message_response, plan_response

        message_response, plan_response = run(call_api())

        self.assertEqual(201, message_response.status_code)
        self.assertEqual(200, plan_response.status_code)
        self.assertEqual(2, len(llm_client.prompts))
        tool_choice_loop = message_response.json()["assistant_message"]["metadata_json"]["context_metadata"]["tool_choice_loop"]
        self.assertEqual(["clarify_goal", "collect_candidates", "enrich_external_info", "analyze_rank"], tool_choice_loop["metadata"]["stage_context_history"])
        self.assertEqual("analyze_rank", tool_choice_loop["metadata"]["stage_context"]["stage_id"])

    def test_agent_task_plan_endpoint_tracks_finished_stage_execution(self):
        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                session_response = await client.post(
                    "/api/v1/agent/sessions",
                    json={"title": "阶段执行", "primary_intent": "agent_chat"},
                )
                session_id = session_response.json()["id"]
                message_response = await client.post(
                    f"/api/v1/agent/sessions/{session_id}/messages",
                    json={"content_text": "帮我找 20 家适合我的秋招公司，并告诉我哪些最值得投"},
                )
                task_id = message_response.json()["assistant_message"]["metadata_json"]["context_metadata"]["outer_session_loop"]["active_task_id"]
                return await client.get(f"/api/v1/agent/tasks/{task_id}/plan")

        plan_response = run(call_api())

        self.assertEqual(200, plan_response.status_code)
        payload = plan_response.json()
        self.assertEqual("agent.stage.finalize_answer", payload["current_stage_id"])
        self.assertEqual(
            ["succeeded", "skipped", "skipped", "skipped", "succeeded"],
            [stage["status"] for stage in payload["stages"]],
        )
        self.assertEqual("skipped_not_needed_in_mvp", payload["stages"][1]["execution_status"])
        self.assertEqual("finished", payload["stages"][-1]["execution_status"])

    def test_agent_task_plan_endpoint_passes_stage_handoff_context_between_stages(self):
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
            def complete(self, *, messages, tools=None, tool_choice=None):
                return _model_selected_api_tool(messages, tools=tools, tool_choice=tool_choice)

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
                        lead_hash="lead-plan-handoff-1",
                        company_name="ByteDance",
                        title="ByteDance 校招岗位聚合",
                        skills=[],
                        trust_level=JobSourceTrustLevel.MEDIUM_HIGH,
                        verification_status=JobLeadStatus.VERIFIED,
                    ),
                    RecruitingSignal(
                        source=source,
                        signal_hash="signal-plan-handoff-1",
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
                    json={"title": "阶段产物", "primary_intent": "agent_chat"},
                )
                session_id = session_response.json()["id"]
                message_response = await client.post(
                    f"/api/v1/agent/sessions/{session_id}/messages",
                    json={"content_text": "我的数据库里现在有多少企业？"},
                )
                task_id = message_response.json()["assistant_message"]["metadata_json"]["context_metadata"]["outer_session_loop"]["active_task_id"]
                return await client.get(f"/api/v1/agent/tasks/{task_id}/plan")

        plan_response = run(call_api())

        self.assertEqual(200, plan_response.status_code)
        payload = plan_response.json()
        collect_stage = payload["stages"][1]
        enrich_stage = payload["stages"][2]
        self.assertEqual("collect_candidates", collect_stage["handoff_payload"]["source_stage_id"])
        self.assertEqual(["local.company_database_overview"], collect_stage["handoff_payload"]["tool_names"])
        self.assertIn("正式企业表 1 家", collect_stage["handoff_payload"]["summary"])
        self.assertEqual(["collect_candidates"], enrich_stage["received_context"]["upstream_stage_ids"])
        self.assertIn("正式企业表 1 家", enrich_stage["received_context"]["summary"])
        self.assertEqual(collect_stage["step_id"], enrich_stage["received_context"]["from_step_ids"][0])

    def test_agent_task_plan_endpoint_tracks_waiting_user_stage_execution(self):
        from app.agent_runtime.graph_factory import AgentWorkflowResult
        from app.agent_runtime.state import AgentState

        def fake_workflow(command, *, dependencies):
            state = AgentState(
                session_id=command.session_id,
                workflow_run_id="workflow-plan-waiting-1",
                agent_run_id="agent-run-plan-waiting-1",
                user_message=command.user_message,
                current_step="final_response",
                final_response="请补充你的简历文本。",
                response_mode="clarification_ask_user",
            )
            return AgentWorkflowResult(workflow_run_id=state.workflow_run_id, state=state)

        with patch("app.api.v1.agent.run_agent_workflow", side_effect=fake_workflow):
            app = self._app()

            async def call_api():
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                    session_response = await client.post(
                        "/api/v1/agent/sessions",
                        json={"title": "阶段等待", "primary_intent": "agent_chat"},
                    )
                    session_id = session_response.json()["id"]
                    message_response = await client.post(
                        f"/api/v1/agent/sessions/{session_id}/messages",
                        json={"content_text": "帮我优化简历"},
                    )
                    task_id = message_response.json()["assistant_message"]["metadata_json"]["context_metadata"]["outer_session_loop"]["active_task_id"]
                    return await client.get(f"/api/v1/agent/tasks/{task_id}/plan")

            plan_response = run(call_api())

        self.assertEqual(200, plan_response.status_code)
        payload = plan_response.json()
        self.assertEqual("agent.stage.clarify_goal", payload["current_stage_id"])
        self.assertEqual(
            ["waiting_user", "pending", "pending", "pending", "pending"],
            [stage["status"] for stage in payload["stages"]],
        )
        self.assertEqual("请补充你的简历文本。", payload["stages"][0]["waiting_message"])

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
        execution_steps = [step for step in durable_steps if step.step_type != "workflow_plan_stage"]
        self.assertEqual(
            [
                "outer_session_turn",
                "workflow_context",
                "workflow_waiting_user",
                "outer_session_turn",
                "workflow_context",
                "workflow_final_response",
            ],
            [step.step_type for step in execution_steps],
        )
        self.assertEqual(
            [
                AgentStepStatus.WAITING_USER,
                AgentStepStatus.SUCCEEDED,
                AgentStepStatus.WAITING_USER,
                AgentStepStatus.SUCCEEDED,
                AgentStepStatus.SUCCEEDED,
                AgentStepStatus.SUCCEEDED,
            ],
            [step.status for step in execution_steps],
        )
        turn_steps = [step for step in durable_steps if step.step_type == "outer_session_turn"]
        final_steps = [step for step in durable_steps if step.step_type in {"workflow_waiting_user", "workflow_final_response"}]
        self.assertEqual(["workflow-outer-1", "workflow-outer-2"], [step.output_payload["workflow_run_id"] for step in turn_steps])
        self.assertEqual(["workflow-outer-1", "workflow-outer-2"], [step.output_payload["workflow_run_id"] for step in final_steps])

    def test_post_offerio_sync_request_routes_through_middleware_local_workflow(self):
        from app.infrastructure.llm.chat_client import LLMChatCompletion

        class FakeLLMClient:
            def complete(self, *, messages, tools=None, tool_choice=None):
                return _model_selected_api_tool(messages, tools=tools, tool_choice=tool_choice)

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
        self.assertNotIn("capability_routing", assistant["metadata_json"]["context_metadata"])
        self.assertIn("tool_candidate_selection", assistant["metadata_json"]["context_metadata"])
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
            def complete(self, *, messages, tools=None, tool_choice=None):
                return _model_selected_api_tool(messages, tools=tools, tool_choice=tool_choice)

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
                message_response = await client.post(
                    f"/api/v1/agent/sessions/{session_id}/messages",
                    json={"content_text": "我的数据库里现在有多少企业？"},
                )
                task_id = message_response.json()["assistant_message"]["metadata_json"]["context_metadata"]["outer_session_loop"]["active_task_id"]
                detail_response = await client.get(f"/api/v1/agent/tasks/{task_id}")
                return message_response, detail_response

        response, detail_response = run(call_api())

        self.assertEqual(201, response.status_code)
        self.assertEqual(200, detail_response.status_code)
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
        self.assertNotIn("capability_routing", metadata)
        self.assertIn("tool_candidate_selection", metadata)
        steps = detail_response.json()["steps"]
        self.assertIn("outer_session_turn", [step["step_type"] for step in steps])
        self.assertIn("workflow_context", [step["step_type"] for step in steps])
        self.assertIn("workflow_tool_call", [step["step_type"] for step in steps])
        self.assertIn("workflow_final_response", [step["step_type"] for step in steps])
        tool_step = next(step for step in steps if step["step_type"] == "workflow_tool_call")
        self.assertEqual("local.company_database_overview", tool_step["capability"])
        self.assertEqual("succeeded", tool_step["status"])
        self.assertEqual("local.company_database_overview", tool_step["output_payload"]["tool_name"])
        final_step = next(step for step in steps if step["step_type"] == "workflow_final_response")
        self.assertEqual("tool_result_summary", final_step["output_payload"]["response_mode"])
        self.assertIn("正式企业表 1 家", final_step["output_payload"]["final_answer_preview"])

    def test_post_specific_company_database_question_does_not_answer_with_full_overview(self):
        from app.domains.jobs.models import Company
        from app.infrastructure.llm.chat_client import LLMChatCompletion

        class FakeLLMClient:
            def complete(self, *, messages, tools=None, tool_choice=None):
                return _model_selected_api_tool(messages, tools=tools, tool_choice=tool_choice)

        with self.Session() as session:
            session.add_all(
                [
                    Company(name="Canonical Ltd.", normalized_name="canonical ltd"),
                    Company(name="腾讯", normalized_name="tencent"),
                ]
            )
            session.commit()

        app = self._app(llm_client=FakeLLMClient())

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                session_response = await client.post(
                    "/api/v1/agent/sessions",
                    json={"title": "specific company lookup", "primary_intent": "agent_chat"},
                )
                session_id = session_response.json()["id"]
                message_response = await client.post(
                    f"/api/v1/agent/sessions/{session_id}/messages",
                    json={"content_text": "你给我看一下数据库中关于京东这个公司的信息有什么"},
                )
                task_id = message_response.json()["assistant_message"]["metadata_json"]["context_metadata"]["outer_session_loop"]["active_task_id"]
                detail_response = await client.get(f"/api/v1/agent/tasks/{task_id}")
                return message_response, detail_response

        response, detail_response = run(call_api())

        self.assertEqual(201, response.status_code)
        self.assertEqual(200, detail_response.status_code)
        assistant = response.json()["assistant_message"]
        self.assertIn("京东", assistant["content_text"])
        self.assertIn("不能把全库概览当成答案", assistant["content_text"])
        self.assertNotIn("我先按公司档次列出来", assistant["content_text"])
        self.assertNotIn("Canonical Ltd.", assistant["content_text"])
        self.assertNotIn("腾讯", assistant["content_text"])
        self.assertEqual("tool_result_summary_insufficient", assistant["metadata_json"]["response_mode"])
        final_step = next(step for step in detail_response.json()["steps"] if step["step_type"] == "workflow_final_response")
        self.assertEqual("tool_result_summary_insufficient", final_step["output_payload"]["response_mode"])

    def test_post_specific_company_database_question_falls_back_to_local_company_sources(self):
        from app.domains.jobs.models import (
            Company,
            JobLead,
            JobLeadStatus,
            JobSource,
            JobSourceFetchMode,
            JobSourceTrustLevel,
            JobSourceType,
            RecruitingSignal,
            RecruitingSignalStatus,
            RecruitingSignalType,
        )
        from app.infrastructure.llm.chat_client import LLMChatCompletion

        class FakeLLMClient:
            def complete(self, *, messages, tools=None, tool_choice=None):
                return _model_selected_api_tool(messages, tools=tools, tool_choice=tool_choice)

        with self.Session() as session:
            source = JobSource(
                name="OfferIO 公司聚合岗位库",
                source_type=JobSourceType.OFFICIAL_API,
                entry_url="https://example.com/offerio",
                trust_level=JobSourceTrustLevel.MEDIUM_HIGH,
                fetch_mode=JobSourceFetchMode.OFFICIAL_API,
            )
            session.add_all(
                [
                    source,
                    Company(name="Canonical Ltd.", normalized_name="canonical ltd"),
                    Company(name="腾讯", normalized_name="tencent"),
                    JobLead(
                        source=source,
                        lead_hash="lead-api-jd-fallback-1",
                        company_name="京东",
                        title="京东 校招岗位聚合（0 个）",
                        city="北京、深圳、广州",
                        job_direction="互联网/游戏/软件",
                        graduation_year="2027",
                        source_url="https://example.com/offerio/jd",
                        job_type="校招",
                        skills=[],
                        trust_level=JobSourceTrustLevel.MEDIUM_HIGH,
                        verification_status=JobLeadStatus.UNVERIFIED,
                    ),
                    JobLead(
                        source=source,
                        lead_hash="lead-api-boe-fallback-1",
                        company_name="京东方",
                        title="京东方 校招岗位聚合（0 个）",
                        city="北京、合肥",
                        job_direction="电子/通信/硬件",
                        graduation_year="2027",
                        source_url="https://example.com/offerio/boe",
                        job_type="校招",
                        skills=[],
                        trust_level=JobSourceTrustLevel.MEDIUM_HIGH,
                        verification_status=JobLeadStatus.UNVERIFIED,
                    ),
                    RecruitingSignal(
                        source=source,
                        signal_hash="signal-api-jd-fallback-1",
                        company_name="京东",
                        normalized_company_name="京东",
                        signal_type=RecruitingSignalType.CAMPUS_RECRUITMENT_OPEN,
                        graduation_year="2027",
                        source_url="https://mp.weixin.qq.com/s/jd-campus",
                        original_source="大连海事就业",
                        confidence_score=82,
                        trust_level=JobSourceTrustLevel.HIGH,
                        status=RecruitingSignalStatus.NEEDS_JOB_ENRICHMENT,
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
                    json={"title": "specific company local fallback", "primary_intent": "agent_chat"},
                )
                session_id = session_response.json()["id"]
                message_response = await client.post(
                    f"/api/v1/agent/sessions/{session_id}/messages",
                    json={"content_text": "你给我看一下数据库中关于京东这个公司的信息有什么"},
                )
                task_id = message_response.json()["assistant_message"]["metadata_json"]["context_metadata"]["outer_session_loop"]["active_task_id"]
                detail_response = await client.get(f"/api/v1/agent/tasks/{task_id}")
                return message_response, detail_response

        response, detail_response = run(call_api())

        self.assertEqual(201, response.status_code)
        self.assertEqual(200, detail_response.status_code)
        assistant = response.json()["assistant_message"]
        self.assertIn("京东", assistant["content_text"])
        self.assertIn("岗位线索 1 条", assistant["content_text"])
        self.assertIn("京东 校招岗位聚合", assistant["content_text"])
        self.assertIn("校招来源 1 条", assistant["content_text"])
        self.assertIn("大连海事就业", assistant["content_text"])
        self.assertIn("正式企业档案：未找到", assistant["content_text"])
        self.assertNotIn("我先按公司档次列出来", assistant["content_text"])
        self.assertNotIn("Canonical Ltd.", assistant["content_text"])
        self.assertNotIn("腾讯", assistant["content_text"])
        self.assertNotIn("京东方", assistant["content_text"])
        self.assertEqual("tool_result_summary_fallback", assistant["metadata_json"]["response_mode"])
        final_step = next(step for step in detail_response.json()["steps"] if step["step_type"] == "workflow_final_response")
        self.assertEqual("tool_result_summary_fallback", final_step["output_payload"]["response_mode"])

    def test_stream_specific_company_database_question_uses_local_fallback(self):
        from app.domains.jobs.models import (
            JobLead,
            JobLeadStatus,
            JobSource,
            JobSourceFetchMode,
            JobSourceTrustLevel,
            JobSourceType,
            RecruitingSignal,
            RecruitingSignalStatus,
            RecruitingSignalType,
        )
        from app.infrastructure.llm.chat_client import LLMChatCompletion

        class FakeLLMClient:
            def complete(self, *, messages, tools=None, tool_choice=None):
                return _model_selected_api_tool(messages, tools=tools, tool_choice=tool_choice)

        with self.Session() as session:
            source = JobSource(
                name="OfferIO 公司聚合岗位库",
                source_type=JobSourceType.OFFICIAL_API,
                entry_url="https://example.com/offerio",
                trust_level=JobSourceTrustLevel.MEDIUM_HIGH,
                fetch_mode=JobSourceFetchMode.OFFICIAL_API,
            )
            session.add_all(
                [
                    source,
                    JobLead(
                        source=source,
                        lead_hash="lead-api-jd-stream-fallback-1",
                        company_name="京东",
                        title="京东 校招岗位聚合（0 个）",
                        city="北京、深圳、广州",
                        job_direction="互联网/游戏/软件",
                        graduation_year="2027",
                        source_url="https://example.com/offerio/jd",
                        job_type="校招",
                        skills=[],
                        trust_level=JobSourceTrustLevel.MEDIUM_HIGH,
                        verification_status=JobLeadStatus.UNVERIFIED,
                    ),
                    RecruitingSignal(
                        source=source,
                        signal_hash="signal-api-jd-stream-fallback-1",
                        company_name="京东",
                        normalized_company_name="京东",
                        signal_type=RecruitingSignalType.CAMPUS_RECRUITMENT_OPEN,
                        graduation_year="2027",
                        source_url="https://mp.weixin.qq.com/s/jd-campus",
                        original_source="大连海事就业",
                        confidence_score=82,
                        trust_level=JobSourceTrustLevel.HIGH,
                        status=RecruitingSignalStatus.NEEDS_JOB_ENRICHMENT,
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
                    json={"title": "specific company stream fallback", "primary_intent": "agent_chat"},
                )
                session_id = session_response.json()["id"]
                stream_response = await client.post(
                    f"/api/v1/agent/sessions/{session_id}/messages/stream",
                    json={"content_text": "你给我看一下数据库中关于京东这个公司的信息有什么"},
                )
                messages_response = await client.get(f"/api/v1/agent/sessions/{session_id}/messages")
                return stream_response, messages_response

        stream_response, messages_response = run(call_api())

        self.assertEqual(200, stream_response.status_code)
        self.assertIn("岗位线索 1 条", stream_response.text)
        self.assertIn("校招来源 1 条", stream_response.text)
        self.assertIn("大连海事就业", stream_response.text)
        self.assertNotIn("我先按公司档次列出来", stream_response.text)
        assistant_messages = [item for item in messages_response.json()["items"] if item["role"] == "assistant"]
        self.assertEqual("tool_result_summary_fallback", assistant_messages[-1]["metadata_json"]["response_mode"])

    def test_post_job_source_count_question_uses_job_source_overview_tool(self):
        from app.infrastructure.llm.chat_client import LLMChatCompletion

        class FakeLLMClient:
            def complete(self, *, messages, tools=None, tool_choice=None):
                return _model_selected_api_tool(messages, tools=tools, tool_choice=tool_choice)

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
        self.assertNotIn("capability_routing", metadata)
        self.assertIn("tool_candidate_selection", metadata)
        self.assertEqual(1, overview_mock.call_count)

    def test_post_generic_company_count_question_uses_company_exhibition_total_only(self):
        from app.infrastructure.llm.chat_client import LLMChatCompletion

        class FakeLLMClient:
            def complete(self, *, messages, tools=None, tool_choice=None):
                return _model_selected_api_tool(messages, tools=tools, tool_choice=tool_choice)

        with patch("app.agent_runtime.tool_registry._local_job_source_overview") as overview_mock:
            overview_mock.return_value = {
                "tool_name": "local.job_source_overview",
                "ok": True,
                "result": {
                    "company_board": {
                        "ok": True,
                        "source": "offerio_company_openings",
                        "label": "公司展览 · 开放岗位公司库",
                        "company_count": 1066,
                    },
                    "company_count": 1066,
                },
            }
            app = self._app(llm_client=FakeLLMClient())

            async def call_api():
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                    session_response = await client.post(
                        "/api/v1/agent/sessions",
                        json={"title": "company exhibition count", "primary_intent": "agent_chat"},
                    )
                    session_id = session_response.json()["id"]
                    return await client.post(
                        f"/api/v1/agent/sessions/{session_id}/messages",
                        json={"content_text": "当前数据库中的公司数有多少个"},
                    )

            response = run(call_api())

        self.assertEqual(201, response.status_code)
        assistant = response.json()["assistant_message"]
        answer = assistant["content_text"]
        self.assertIn("公司展览当前的公司数是 1066 家", answer)
        self.assertNotIn("正式企业表", answer)
        self.assertNotIn("岗位线索", answer)
        self.assertNotIn("招聘信号", answer)
        metadata = assistant["metadata_json"]["context_metadata"]
        self.assertEqual("company_board_overview", metadata["intent_frame"]["intent"])
        self.assertEqual("company_board_count", metadata["context_pack"]["sync_policy"]["mode"])
        self.assertNotIn("capability_routing", metadata)
        self.assertIn("tool_candidate_selection", metadata)
        overview_mock.assert_called_once()
        self.assertEqual("company_board_count", overview_mock.call_args.kwargs["mode"])

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
                tasks_response = await client.get(f"/api/v1/agent/sessions/{session_id}/tasks")
                return stream_response, messages_response, session_after_response, tasks_response

        stream_response, messages_response, session_after_response, tasks_response = run(call_api())

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

        self.assertEqual(200, tasks_response.status_code)
        task_items = tasks_response.json()["items"]
        self.assertEqual(1, len(task_items))
        self.assertEqual(done_outer["active_task_id"], task_items[0]["id"])
        self.assertEqual("succeeded", task_items[0]["status"])

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

    def test_stream_agent_message_persists_terminal_error_and_done_event(self):
        def fake_prepare(*_args, **_kwargs):
            raise RuntimeError("simulated stream failure")

        with patch("app.api.v1.agent.prepare_agent_workflow_response", side_effect=fake_prepare):
            app = self._app()

            async def call_api():
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                    session_response = await client.post(
                        "/api/v1/agent/sessions",
                        json={"title": "stream terminal error", "primary_intent": "agent_chat"},
                    )
                    session_id = session_response.json()["id"]
                    stream_response = await client.post(
                        f"/api/v1/agent/sessions/{session_id}/messages/stream",
                        json={"content_text": "请执行一个会失败的操作"},
                    )
                    messages_response = await client.get(f"/api/v1/agent/sessions/{session_id}/messages")
                    return stream_response, messages_response

            stream_response, messages_response = run(call_api())

        self.assertEqual(200, stream_response.status_code)
        error_payload = _sse_payload(stream_response.text, "error")
        done_payload = _sse_payload(stream_response.text, "done")
        self.assertEqual("AGENT_STREAM_EXCEPTION", error_payload["error_code"])
        self.assertTrue(error_payload["terminal"])
        self.assertEqual("failed", done_payload["terminal_status"])
        self.assertEqual("stream_error", done_payload["assistant_message"]["metadata_json"]["response_mode"])
        assistant_messages = [message for message in messages_response.json()["items"] if message["role"] == "assistant"]
        self.assertEqual(1, len(assistant_messages))
        self.assertIn("simulated stream failure", assistant_messages[0]["content_text"])







    def test_stream_agent_message_splits_prepared_final_response_into_multiple_token_events(self):
        from app.agent_runtime.graph_factory import AgentPreparedResponse
        from app.agent_runtime.state import AgentState
        from app.domains.automation.schemas import WorkflowRunCreate

        final_response = (
            "我已经读取到文件内容，下面按原文展示第一部分。"
            "这段内容用于模拟后端已经准备好的最终回答，但前端仍然需要看到逐步输出。"
            "如果整段一次性发给前端，用户就会感觉不像流式输出。"
        )

        def fake_prepare(command, *, dependencies, on_workflow_started=None):
            workflow = dependencies.automation_service.start_workflow(
                WorkflowRunCreate(
                    workflow_type="agent_chat",
                    current_step="final_response",
                    user_goal=command.user_message,
                )
            )
            state = AgentState(
                session_id=command.session_id,
                workflow_run_id=workflow.id,
                agent_run_id="agent-run-stream-prepared-final-1",
                user_message=command.user_message,
                current_step="final_response",
                final_response=final_response,
                response_mode="llm_tool_choice_loop",
                context_metadata={"stream_prepared_final_test": True},
            )
            return AgentPreparedResponse(workflow_run_id=workflow.id, workflow=workflow, state=state)

        with patch("app.api.v1.agent.prepare_agent_workflow_response", side_effect=fake_prepare):
            app = self._app()

            async def call_api():
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                    session_response = await client.post(
                        "/api/v1/agent/sessions",
                        json={"title": "stream prepared final", "primary_intent": "agent_chat"},
                    )
                    session_id = session_response.json()["id"]
                    return await client.post(
                        f"/api/v1/agent/sessions/{session_id}/messages/stream",
                        json={"content_text": "读取文件内容"},
                    )

            stream_response = run(call_api())

        token_payloads = _sse_payloads(stream_response.text, "token")
        self.assertEqual(200, stream_response.status_code)
        self.assertGreater(len(token_payloads), 1)
        self.assertEqual(final_response, "".join(str(payload.get("content") or "") for payload in token_payloads))



    def test_stream_agent_message_suppresses_false_tool_execution_claim_without_tool_logs(self):
        class FakeStreamingFalseClaimLLM:
            def stream_complete(self, *, messages):
                yield "我已经调用联网搜索，查到了 Canonical 的主要业务。"

        app = self._app(llm_client=FakeStreamingFalseClaimLLM())

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                session_response = await client.post(
                    "/api/v1/agent/sessions",
                    json={"title": "stream false tool claim", "primary_intent": "agent_chat"},
                )
                session_id = session_response.json()["id"]
                stream_response = await client.post(
                    f"/api/v1/agent/sessions/{session_id}/messages/stream",
                    json={"content_text": "帮我写一句求职备注"},
                )
                messages_response = await client.get(f"/api/v1/agent/sessions/{session_id}/messages")
                return stream_response, messages_response

        stream_response, messages_response = run(call_api())

        self.assertEqual(200, stream_response.status_code)
        token_text = "".join(str(payload.get("content") or "") for payload in _sse_payloads(stream_response.text, "token"))
        self.assertNotIn("我已经调用联网搜索", token_text)
        self.assertIn("本轮没有真实工具执行记录", token_text)
        assistant = [message for message in messages_response.json()["items"] if message["role"] == "assistant"][0]
        self.assertNotIn("我已经调用联网搜索", assistant["content_text"])
        self.assertIn("本轮没有真实工具执行记录", assistant["content_text"])

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
                                "decision_reason": "用户要找校招官网，需要先用公开网页搜索核对入口。",
                                "observation_summary": "搜索结果是百科页面，没有校招入口。",
                                "tool_call_id": "tool-call-bad-1",
                                "metadata": {
                                    "tool_input": {"query": "中科曙光 校招 官网", "max_results": 5},
                                    "tool_input_keys": ["max_results", "query"],
                                    "result_observation": {
                                        "result_count": 4,
                                        "source_count": 1,
                                        "source_domains": ["baike.example.com"],
                                        "evidence": [],
                                    },
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
                                "decision_reason": "第一次结果没有命中校招入口，需要用更明确的关键词再次搜索。",
                                "observation_summary": "找到中科曙光校园招聘官网。",
                                "tool_call_id": "tool-call-good-2",
                                "metadata": {
                                    "tool_input": {"query": "中科曙光 校园招聘 官网 2026", "max_results": 5},
                                    "tool_input_keys": ["max_results", "query"],
                                    "result_observation": {
                                        "result_count": 6,
                                        "source_count": 2,
                                        "source_domains": ["sugon.com", "jobs.example.com"],
                                        "evidence": [
                                            {"title": "中科曙光校园招聘", "url": "https://jobs.example.com/sugon"}
                                        ],
                                    },
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
            [
                "reasoning_summary",
                "tool_input_preview",
                "tool_started",
                "tool_finished",
                "tool_result_summary",
                "reflection_evaluation",
                "tool_reflection_retry",
                "reasoning_summary",
                "tool_input_preview",
                "tool_started",
                "tool_finished",
                "tool_result_summary",
                "reflection_evaluation",
                "evidence_selected",
            ],
            [event["event_type"] for event in tool_events],
        )
        self.assertEqual("external.web_search", tool_events[0]["tool_name"])
        self.assertIn("公开网页搜索", tool_events[0]["summary"])
        self.assertEqual("中科曙光 校招 官网", tool_events[1]["input_preview"]["query"])
        self.assertEqual("tool-call-bad-1", tool_events[3]["tool_call_id"])
        self.assertEqual("搜索结果是百科页面，没有校招入口。", tool_events[3]["summary"])
        self.assertEqual(4, tool_events[4]["result_summary"]["result_count"])
        self.assertEqual(["baike.example.com"], tool_events[4]["result_summary"]["source_domains"])
        self.assertEqual("retry", tool_events[5]["reflection"]["next_action"])
        self.assertEqual("中科曙光 校园招聘 官网 2026", tool_events[6]["suggested_input_patch"]["query"])
        self.assertEqual("tool-call-good-2", tool_events[-1]["tool_call_id"])
        self.assertEqual("中科曙光校园招聘", tool_events[-1]["evidence"][0]["title"])

    def test_stream_offerio_sync_request_routes_through_middleware_local_workflow(self):
        class FakeStreamingLLMClient:
            def complete(self, *, messages, tools=None, tool_choice=None):
                from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

                if tools and not any(message.get("role") == "tool" for message in messages):
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="stream-offerio-sync",
                                name="offerio_sync_company_jobs",
                                arguments={"limit": 1000},
                            )
                        ],
                    )
                return LLMChatCompletion(content="")

            def stream_complete(self, *, messages):
                yield ""

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
        self.assertEqual(
            [
                "candidate_capabilities",
                "task_started",
                "turn_started",
                "model_decision",
                "reasoning_summary",
                "tool_input_preview",
                "tool_started",
                "tool_finished",
                "tool_result_summary",
                "turn_finished",
                "turn_started",
                "model_decision",
                "task_finished",
            ],
            [event["event_type"] for event in tool_events],
        )
        candidate_event = tool_events[0]
        self.assertEqual("agent_loop", candidate_event["tool_name"])
        model_decision = next(event for event in tool_events if event["event_type"] == "model_decision")
        self.assertEqual("offerio.sync_company_jobs", model_decision["tool_name"])
        tool_started = next(event for event in tool_events if event["event_type"] == "tool_started")
        self.assertEqual("offerio.sync_company_jobs", tool_started["tool_name"])
        tool_finished = next(event for event in tool_events if event["event_type"] == "tool_finished")
        self.assertEqual("succeeded", tool_finished["status"])

        messages = messages_response.json()["items"]
        self.assertEqual(["assistant", "tool_call", "tool_result", "user"], sorted(message["role"] for message in messages))
        self.assertIn("已从 OfferIO 公司聚合岗位库同步岗位", messages[-1]["content_text"])
        self.assertEqual("tool_result_summary", messages[-1]["metadata_json"]["response_mode"])

    def test_stream_agent_message_emits_tool_started_before_tool_finishes(self):
        import json
        from queue import Empty, Queue
        from time import monotonic

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
            allow_tool_finish.wait(timeout=5)
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
            deadline = monotonic() + 3
            while monotonic() < deadline:
                try:
                    item = event_queue.get(timeout=0.2)
                except Empty:
                    continue
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
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

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
                self.complete_calls = []

            def complete(self, *, messages, tools=None, tool_choice=None):
                self.complete_calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
                if tools and not any(message.get("role") == "tool" for message in messages):
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="stream-gongniu-search",
                                name="external_web_search",
                                arguments={"query": "公牛集团 校园招聘 官网", "max_results": 5},
                            )
                        ],
                    )
                return LLMChatCompletion(content="")

            def stream_complete(self, *, messages):
                self.calls.append(messages)
                combined = "\n".join(str(message.get("content") or "") for message in messages)
                test_case.assertIn("芝加哥公牛队", combined)
                test_case.assertIn("公牛集团校园招聘", combined)
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
        token_text = "".join(payload.get("content", "") for payload in _sse_payloads(stream_text, "token"))
        self.assertIn("公牛集团校招入口：", token_text)
        self.assertIn("https://campus.gongniu.cn/", token_text)
        self.assertNotIn("联网搜索结果：", token_text)
        self.assertNotIn("NBA", token_text)
        self.assertNotIn("芝加哥", token_text)
        self.assertNotIn("过滤", token_text)
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
                    assert "xiaohongshu_mcp_search_feeds" in self_tool_names
                    assert not any(name.startswith("mcp_") for name in self_tool_names)
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

    def test_approve_stale_agent_approval_returns_409_and_keeps_approval_pending(self):
        self._create_approval_memory_skill()
        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                session_response = await client.post(
                    "/api/v1/agent/sessions",
                    json={"title": "stale approval", "primary_intent": "agent_chat"},
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
                approval_id = approval_payload["approval"]["id"]
                workflow_id = approval_payload["workflow_run_id"]

                with self.Session() as db_session:
                    from app.agent_runtime.checkpoints import AgentCheckpointStore
                    from app.domains.automation.models import ApprovalRequest, ApprovalRequestStatus, WorkflowRun, WorkflowRunStatus, utc_now
                    from app.domains.automation.repository import (
                        ApprovalRequestRepository,
                        ToolCallLogRepository,
                        WorkflowCheckpointRepository,
                        WorkflowRunRepository,
                    )
                    from app.domains.automation.service import AutomationService

                    automation_service = AutomationService(
                        workflow_runs=WorkflowRunRepository(db_session),
                        checkpoints=WorkflowCheckpointRepository(db_session),
                        tool_call_logs=ToolCallLogRepository(db_session),
                        approvals=ApprovalRequestRepository(db_session),
                    )
                    checkpoint_store = AgentCheckpointStore(session=db_session, automation_service=automation_service)
                    workflow = db_session.get(WorkflowRun, workflow_id)
                    approval = db_session.get(ApprovalRequest, approval_id)
                    assert workflow is not None
                    assert approval is not None
                    latest = checkpoint_store.load_latest(workflow_id)
                    workflow.status = WorkflowRunStatus.COMPLETED
                    workflow.current_step = "final_response"
                    workflow.completed_at = utc_now()
                    checkpoint_store.save(
                        workflow_run_id=workflow_id,
                        checkpoint_key="final_response",
                        state=latest.state.with_updates(current_step="final_response", final_response="already finished"),
                    )
                    db_session.commit()

                approve_response = await client.post(
                    f"/api/v1/agent/approvals/{approval_id}/approve",
                    json={"decision_reason": "clicked stale button"},
                )

                with self.Session() as db_session:
                    from app.domains.automation.models import ApprovalRequest

                    approval_after = db_session.get(ApprovalRequest, approval_id)
                    approval_status = approval_after.status if approval_after is not None else None
                return approve_response, approval_status

        approve_response, approval_status = run(call_api())

        self.assertEqual(409, approve_response.status_code)
        self.assertEqual("STALE_APPROVAL_REQUEST", approve_response.json()["detail"]["error_code"])
        self.assertEqual("pending", approval_status)

    def test_chat_confirmation_message_resumes_pending_approval(self):
        self._create_approval_memory_skill()
        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                session_response = await client.post(
                    "/api/v1/agent/sessions",
                    json={"title": "chat confirmation recovery", "primary_intent": "agent_chat"},
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
                confirmation_response = await client.post(
                    f"/api/v1/agent/sessions/{session_id}/messages",
                    json={"content_text": "是"},
                )
                messages_response = await client.get(f"/api/v1/agent/sessions/{session_id}/messages")
                return approval_payload, confirmation_response, messages_response

        approval_payload, confirmation_response, messages_response = run(call_api())

        self.assertEqual(201, confirmation_response.status_code)
        confirmation_body = confirmation_response.json()
        self.assertEqual("assistant", confirmation_body["assistant_message"]["role"])
        self.assertEqual(approval_payload["workflow_run_id"], confirmation_body["assistant_message"]["workflow_run_id"])
        from app.domains.automation.models import ApprovalRequest

        with self.Session() as db_session:
            approval = db_session.get(ApprovalRequest, approval_payload["approval_request_id"])

        self.assertIsNotNone(approval)
        self.assertEqual("approved", str(approval.status.value if hasattr(approval.status, "value") else approval.status))
        self.assertNotIn("缺少", confirmation_body["assistant_message"]["content_text"] or "")
        roles = [message["role"] for message in messages_response.json()["items"]]
        self.assertIn("user", roles)
        self.assertIn("assistant", roles)

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
