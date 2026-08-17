import sys
import unittest
import shutil
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class AgentRuntimeGraphTest(unittest.TestCase):
    def setUp(self) -> None:
        from app.db.base import Base
        import app.domains.agent_memory.models  # noqa: F401
        import app.domains.automation.models  # noqa: F401
        import app.domains.conversations.models  # noqa: F401

        self.skill_root = PROJECT_ROOT / ".tmp-test-artifacts" / "agent-runtime-graph" / self._testMethodName
        shutil.rmtree(self.skill_root, ignore_errors=True)
        self.skill_root.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

    def tearDown(self) -> None:
        self.engine.dispose()
        shutil.rmtree(self.skill_root, ignore_errors=True)

    def _dependencies(self, session, *, skill_repository=None, llm_client=None):
        from app.agent_runtime.checkpoints import AgentCheckpointStore
        from app.agent_runtime.guardrails import AgentToolPolicy, AgentToolRuntimeGuard
        from app.agent_runtime.graph_factory import AgentGraphDependencies
        from app.agent_runtime.tool_registry import create_default_agent_tool_registry
        from app.domains.automation.repository import (
            ApprovalRequestRepository,
            ToolCallLogRepository,
            WorkflowCheckpointRepository,
            WorkflowRunRepository,
        )
        from app.domains.automation.service import AutomationService
        from app.domains.conversations.repository import ConversationRepository
        from app.domains.conversations.service import ConversationService

        automation_service = AutomationService(
            workflow_runs=WorkflowRunRepository(session),
            checkpoints=WorkflowCheckpointRepository(session),
            tool_call_logs=ToolCallLogRepository(session),
            approvals=ApprovalRequestRepository(session),
        )
        return AgentGraphDependencies(
            automation_service=automation_service,
            checkpoint_store=AgentCheckpointStore(session=session, automation_service=automation_service),
            conversation_service=ConversationService(ConversationRepository(session)),
            registry=create_default_agent_tool_registry(),
            guard=AgentToolRuntimeGuard(policy=AgentToolPolicy(max_tool_calls=10)),
            skill_repository=skill_repository,
            db_session=session,
            llm_client=llm_client,
        )

    def _skill_repository(self, session):
        from app.agent_runtime.memory.skill_repository import AgentSkillRepository
        from app.domains.agent_memory.repository import AgentMemoryRepository

        return AgentSkillRepository(AgentMemoryRepository(session), skill_root=self.skill_root)

    def _session_id(self, session) -> str:
        from app.domains.conversations.repository import ConversationRepository
        from app.domains.conversations.service import ConversationService

        conversation = ConversationService(ConversationRepository(session)).create_session(
            title="Agent runtime",
            primary_intent="agent_chat",
        )
        return conversation.id

    def test_agent_run_creates_workflow_run_and_checkpoints_context_state(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.domains.automation.models import WorkflowCheckpoint, WorkflowRun, WorkflowRunStatus

        with self.Session() as session:
            session_id = self._session_id(session)
            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="帮我回忆之前的岗位线索策略"),
                dependencies=self._dependencies(session),
            )
            session.commit()

            workflow = session.get(WorkflowRun, result.workflow_run_id)
            checkpoints = session.scalars(
                select(WorkflowCheckpoint).where(WorkflowCheckpoint.workflow_run_id == result.workflow_run_id)
            ).all()

        self.assertIsNotNone(workflow)
        self.assertEqual("agent_chat", workflow.workflow_type)
        self.assertEqual(WorkflowRunStatus.COMPLETED, workflow.status)
        self.assertEqual("final_response", workflow.current_step)
        self.assertEqual(session_id, result.state.session_id)
        self.assertEqual(result.workflow_run_id, result.state.workflow_run_id)
        self.assertTrue(result.state.agent_run_id.startswith("agent-run-"))
        self.assertEqual("final_response", result.state.current_step)
        self.assertIn("build_context", {checkpoint.checkpoint_key for checkpoint in checkpoints})
        self.assertIn("plan_or_reply", {checkpoint.checkpoint_key for checkpoint in checkpoints})
        self.assertIn("final_response", {checkpoint.checkpoint_key for checkpoint in checkpoints})
        self.assertEqual([], result.state.loaded_skill_ids)
        self.assertEqual([], result.state.tool_call_ids)

    def test_agent_run_checkpoints_loaded_skill_ids_from_runtime_context(self) -> None:
        from app.agent_runtime.checkpoints import AgentCheckpointStore
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.domains.agent_memory.schemas import AgentSkillCreate

        with self.Session() as session:
            session_id = self._session_id(session)
            skill_repository = self._skill_repository(session)
            skill = skill_repository.create_skill(
                AgentSkillCreate(
                    name="java-job-discovery",
                    title="Java Job Discovery",
                    description="Use this skill when the user asks for Java campus recruiting leads.",
                    category="job_discovery",
                )
            )
            dependencies = self._dependencies(session, skill_repository=skill_repository)

            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="Java"),
                dependencies=dependencies,
            )
            session.commit()

            latest = AgentCheckpointStore(session=session, automation_service=dependencies.automation_service).load_latest(
                result.workflow_run_id
            )

        self.assertEqual([skill.id], result.state.loaded_skill_ids)
        self.assertEqual([skill.id], latest.state.loaded_skill_ids)

    def test_high_risk_tool_stops_at_waiting_user_confirmation_node(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry, AgentToolRiskLevel
        from app.domains.automation.models import ApprovalRequest, WorkflowCheckpoint, WorkflowRun, WorkflowRunStatus

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(session)
            registry = AgentToolRegistry(dependencies.registry.list_definitions())
            registry.register(
                AgentToolDefinition(
                    name="submit_application",
                    description="Submit a real job application.",
                    input_schema={"type": "object"},
                    output_schema={"type": "object"},
                    risk_level=AgentToolRiskLevel.HIGH,
                    requires_confirmation=True,
                    allowed_source_types=frozenset({"application"}),
                )
            )
            dependencies = dependencies.with_registry(registry)

            result = run_agent_workflow(
                AgentRunCommand(
                    session_id=session_id,
                    user_message="帮我投递这个岗位",
                    requested_tool_name="submit_application",
                    source_type="application",
                ),
                dependencies=dependencies,
            )
            session.commit()

            workflow = session.get(WorkflowRun, result.workflow_run_id)
            checkpoints = session.scalars(
                select(WorkflowCheckpoint).where(WorkflowCheckpoint.workflow_run_id == result.workflow_run_id)
            ).all()
            approval = session.scalars(select(ApprovalRequest)).one()

        self.assertEqual(WorkflowRunStatus.WAITING_USER, workflow.status)
        self.assertEqual("wait_confirmation", workflow.current_step)
        self.assertEqual("wait_confirmation", result.state.current_step)
        self.assertEqual(approval.id, workflow.approval_request_id)
        self.assertEqual("submit_application", approval.action_type)
        self.assertIn("wait_confirmation", {checkpoint.checkpoint_key for checkpoint in checkpoints})
        self.assertNotIn("final_response", {checkpoint.checkpoint_key for checkpoint in checkpoints})

    def test_skill_permission_snapshot_denies_tool_before_execution(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.domains.agent_memory.schemas import AgentSkillCreate
        from app.domains.automation.models import ToolCallLog

        with self.Session() as session:
            session_id = self._session_id(session)
            skill_repository = self._skill_repository(session)
            skill = skill_repository.create_skill(
                AgentSkillCreate(
                    name="java-deny-sessions-search",
                    title="Java Deny Sessions Search",
                    description="Use this skill when the user asks for Java context.",
                    category="agent_guardrail",
                    metadata_json={"disallowed_tools": ["sessions_search"]},
                    sections={"workflow": "Java context must not use sessions_search."},
                )
            )
            dependencies = self._dependencies(session, skill_repository=skill_repository)

            result = run_agent_workflow(
                AgentRunCommand(
                    session_id=session_id,
                    user_message="Java",
                    requested_tool_name="sessions_search",
                    source_type="agent_chat",
                ),
                dependencies=dependencies,
            )
            session.commit()

            tool_logs = list(session.scalars(select(ToolCallLog)).all())

        self.assertEqual([skill.id], result.state.loaded_skill_ids)
        self.assertEqual("TOOL_SKILL_DENIED", result.state.guard_result["error_code"])
        self.assertEqual("stop", result.state.guard_result["next_action"])
        self.assertEqual([skill.id], result.state.guard_result["error_details"]["skill_ids"])
        self.assertEqual([], tool_logs)

    def test_skill_permission_snapshot_requires_confirmation_for_ask_tool(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.domains.agent_memory.schemas import AgentSkillCreate
        from app.domains.automation.models import ApprovalRequest, ToolCallLog, WorkflowRun, WorkflowRunStatus

        with self.Session() as session:
            session_id = self._session_id(session)
            skill_repository = self._skill_repository(session)
            skill = skill_repository.create_skill(
                AgentSkillCreate(
                    name="java-ask-memory-search",
                    title="Java Ask Memory Search",
                    description="Use this skill when the user asks for Java memory context.",
                    category="agent_guardrail",
                    metadata_json={"allowed_tools": ["sessions_search"], "ask_tools": ["memory_search"]},
                    sections={"workflow": "Java memory context may ask before memory_search."},
                )
            )
            skill_id = skill.id
            dependencies = self._dependencies(session, skill_repository=skill_repository)

            result = run_agent_workflow(
                AgentRunCommand(
                    session_id=session_id,
                    user_message="Java memory",
                    requested_tool_name="memory_search",
                    source_type="agent_chat",
                ),
                dependencies=dependencies,
            )
            session.commit()

            workflow = session.get(WorkflowRun, result.workflow_run_id)
            approval = session.scalars(select(ApprovalRequest)).one()
            tool_logs = list(session.scalars(select(ToolCallLog)).all())

        self.assertIn(skill_id, result.state.loaded_skill_ids)
        self.assertEqual(WorkflowRunStatus.WAITING_USER, workflow.status)
        self.assertEqual("wait_confirmation", result.state.current_step)
        self.assertEqual("TOOL_SKILL_CONFIRMATION_REQUIRED", result.state.guard_result["error_code"])
        self.assertEqual("request_user_confirmation", result.state.guard_result["next_action"])
        self.assertEqual("ask", approval.payload["guard_result"]["error_details"]["permission_decision"])
        self.assertEqual([], tool_logs)

    def test_approved_skill_tool_call_records_usage_evidence_for_learning(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, continue_agent_workflow_after_approval, run_agent_workflow
        from app.domains.agent_memory.schemas import AgentSkillCreate
        from app.domains.automation.models import ApprovalRequest
        from app.domains.conversations.models import AgentMessageRole
        from app.domains.conversations.schemas import AgentMessageCreate

        with self.Session() as session:
            session_id = self._session_id(session)
            skill_repository = self._skill_repository(session)
            skill = skill_repository.create_skill(
                AgentSkillCreate(
                    name="java-approval-evidence",
                    title="Java Approval Evidence",
                    description="Use this skill when the user asks for Java approval evidence.",
                    category="agent_guardrail",
                    metadata_json={"allowed_tools": ["memory_search"], "ask_tools": ["memory_search"]},
                    sections={"workflow": "Ask before memory_search and retain usage evidence."},
                )
            )
            dependencies = self._dependencies(session, skill_repository=skill_repository)
            dependencies.conversation_service.append_message(
                session_id,
                AgentMessageCreate(
                    role=AgentMessageRole.USER,
                    content_text="previous Java approval evidence",
                    visible_content_text="previous Java approval evidence",
                    token_estimate=8,
                ),
            )

            first = run_agent_workflow(
                AgentRunCommand(
                    session_id=session_id,
                    user_message="Java approval evidence",
                    requested_tool_name="memory_search",
                    source_type="agent_chat",
                ),
                dependencies=dependencies,
            )
            session.commit()
            approval = session.scalars(select(ApprovalRequest)).one()
            approval_id = approval.id
            continued = continue_agent_workflow_after_approval(
                approval_id,
                approved=True,
                decision_reason="approved for learning evidence",
                dependencies=dependencies,
            )
            usage = skill_repository.get_usage(skill.id)
            session.commit()

        events = usage.metadata_json["runtime_events"]
        self.assertEqual("wait_confirmation", first.state.current_step)
        self.assertEqual("final_response", continued.state.current_step)
        self.assertEqual(1, usage.success_count)
        self.assertEqual(0, usage.failure_count)
        self.assertEqual(["approval_requested", "approval_approved", "tool_succeeded"], [event["event"] for event in events])
        self.assertEqual(approval_id, events[1]["approval_request_id"])
        self.assertEqual(continued.state.tool_call_ids[0], events[2]["tool_call_log_id"])
        self.assertEqual("memory_search", events[2]["tool_name"])

    def test_rejected_skill_tool_confirmation_records_refusal_without_failure_count(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, continue_agent_workflow_after_approval, run_agent_workflow
        from app.domains.agent_memory.schemas import AgentSkillCreate
        from app.domains.automation.models import ApprovalRequest
        from app.domains.conversations.models import AgentMessageRole
        from app.domains.conversations.schemas import AgentMessageCreate

        with self.Session() as session:
            session_id = self._session_id(session)
            skill_repository = self._skill_repository(session)
            skill = skill_repository.create_skill(
                AgentSkillCreate(
                    name="java-rejection-evidence",
                    title="Java Rejection Evidence",
                    description="Use this skill when the user asks for Java rejection evidence.",
                    category="agent_guardrail",
                    metadata_json={"allowed_tools": ["memory_search"], "ask_tools": ["memory_search"]},
                    sections={"workflow": "Ask before memory_search and retain refusal evidence."},
                )
            )
            dependencies = self._dependencies(session, skill_repository=skill_repository)
            dependencies.conversation_service.append_message(
                session_id,
                AgentMessageCreate(
                    role=AgentMessageRole.USER,
                    content_text="previous Java rejection evidence",
                    visible_content_text="previous Java rejection evidence",
                    token_estimate=8,
                ),
            )

            first = run_agent_workflow(
                AgentRunCommand(
                    session_id=session_id,
                    user_message="Java rejection evidence",
                    requested_tool_name="memory_search",
                    source_type="agent_chat",
                ),
                dependencies=dependencies,
            )
            session.commit()
            approval = session.scalars(select(ApprovalRequest)).one()
            approval_id = approval.id
            continued = continue_agent_workflow_after_approval(
                approval_id,
                approved=False,
                decision_reason="not this time",
                dependencies=dependencies,
            )
            usage = skill_repository.get_usage(skill.id)
            session.commit()

        events = usage.metadata_json["runtime_events"]
        self.assertEqual("wait_confirmation", first.state.current_step)
        self.assertEqual("approval_rejected", continued.state.current_step)
        self.assertEqual(0, usage.success_count)
        self.assertEqual(0, usage.failure_count)
        self.assertEqual(["approval_requested", "approval_rejected"], [event["event"] for event in events])
        self.assertEqual(approval_id, events[1]["approval_request_id"])
        self.assertEqual("not this time", events[1]["decision_reason"])

    def test_resume_uses_latest_checkpoint_without_duplicate_tool_execution(self) -> None:
        from app.agent_runtime.checkpoints import AgentCheckpointStore
        from app.agent_runtime.graph_factory import AgentRunCommand, resume_agent_workflow, run_agent_workflow
        from app.domains.automation.models import ToolCallLog, WorkflowCheckpoint

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(session)
            first = run_agent_workflow(
                AgentRunCommand(
                    session_id=session_id,
                    user_message="查一下之前怎么处理公众号失败",
                    requested_tool_name="sessions_search",
                    source_type="agent_chat",
                ),
                dependencies=dependencies,
            )
            session.commit()

            checkpoint_count_before = len(session.scalars(select(WorkflowCheckpoint)).all())
            resumed = resume_agent_workflow(first.workflow_run_id, dependencies=self._dependencies(session))
            checkpoint_count_after = len(session.scalars(select(WorkflowCheckpoint)).all())
            tool_log_count = len(session.scalars(select(ToolCallLog)).all())
            latest = AgentCheckpointStore(session=session, automation_service=dependencies.automation_service).load_latest(
                first.workflow_run_id
            )

        self.assertEqual(first.state.tool_call_ids, resumed.state.tool_call_ids)
        self.assertEqual(1, len(first.state.tool_call_ids))
        self.assertEqual(1, tool_log_count)
        self.assertEqual("final_response", resumed.state.current_step)
        self.assertEqual("final_response", latest.checkpoint_key)
        self.assertEqual(checkpoint_count_before, checkpoint_count_after)

    def test_registered_memory_tool_executes_handler_and_writes_tool_pair_to_transcript(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.domains.automation.models import ToolCallLog, ToolCallStatus
        from app.domains.conversations.models import AgentMessageRole
        from app.domains.conversations.schemas import AgentMessageCreate

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(session)
            dependencies.conversation_service.append_message(
                session_id,
                AgentMessageCreate(
                    role=AgentMessageRole.USER,
                    content_text="previous Java anchor",
                    visible_content_text="previous Java anchor",
                    token_estimate=8,
                ),
            )

            result = run_agent_workflow(
                AgentRunCommand(
                    session_id=session_id,
                    user_message="find prior Java context",
                    requested_tool_name="sessions_search",
                    source_type="agent_chat",
                    tool_input={"query": "previous Java anchor", "limit": 5},
                ),
                dependencies=dependencies,
            )
            session.commit()

            tool_log = session.scalars(select(ToolCallLog)).one()
            messages = dependencies.conversation_service.list_messages(session_id, limit=20)

        self.assertEqual(ToolCallStatus.SUCCEEDED, tool_log.status)
        self.assertEqual("handler", tool_log.output_payload["execution"])
        self.assertEqual("sessions", tool_log.output_payload["result"]["corpus"])
        self.assertIn("previous Java anchor", tool_log.output_payload["result"]["items"][0]["excerpt"])
        self.assertEqual([tool_log.id], result.state.tool_call_ids)

        tool_messages = [message for message in messages if message.tool_call_log_id == tool_log.id]
        self.assertEqual([AgentMessageRole.TOOL_CALL, AgentMessageRole.TOOL_RESULT], [message.role for message in tool_messages])
        self.assertEqual("sessions_search", tool_messages[0].content_json["tool_name"])
        self.assertEqual("succeeded", tool_messages[1].content_json["status"])
        self.assertEqual(tool_messages[0].id, tool_messages[1].parent_message_id)

    def test_mcp_gateway_tool_definition_executes_through_agent_tool_runtime(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.tool_registry import AgentToolRegistry, create_mcp_agent_tool_definitions
        from app.domains.automation.models import ToolCallLog, ToolCallStatus
        from app.mcp_gateway.client import MCPToolCallResult

        class FakeMCPClient:
            def __init__(self) -> None:
                self.calls = []

            def call_tool(self, *, tool_name: str, arguments: dict) -> MCPToolCallResult:
                self.calls.append({"tool_name": tool_name, "arguments": arguments})
                return MCPToolCallResult(
                    tool_name=tool_name,
                    ok=True,
                    result={"title": "Example", "url": arguments["url"]},
                    error=None,
                )

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(session)
            fake_client = FakeMCPClient()
            registry = AgentToolRegistry(dependencies.registry.list_definitions())
            registry.register_many(create_mcp_agent_tool_definitions(fake_client, allowed_tool_names=["open_page"]))

            result = run_agent_workflow(
                AgentRunCommand(
                    session_id=session_id,
                    user_message="open page",
                    requested_tool_name="mcp.open_page",
                    source_type="agent_chat",
                    tool_input={"url": "https://example.com"},
                ),
                dependencies=dependencies.with_registry(registry),
            )
            session.commit()

            tool_log = session.scalars(select(ToolCallLog)).one()

        self.assertEqual([{"tool_name": "open_page", "arguments": {"url": "https://example.com"}}], fake_client.calls)
        self.assertEqual(ToolCallStatus.SUCCEEDED, tool_log.status)
        self.assertEqual("mcp.open_page", tool_log.tool_name)
        self.assertEqual("handler", tool_log.output_payload["execution"])
        self.assertEqual("Example", tool_log.output_payload["result"]["result"]["title"])
        self.assertEqual([tool_log.id], result.state.tool_call_ids)

    def test_agent_auto_selects_weixin_article_tool_from_public_article_url(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry
        from app.domains.agent_memory.schemas import AgentSkillCreate
        from app.domains.automation.models import ToolCallLog, ToolCallStatus
        from app.mcp_gateway.client import MCPToolCallResult

        calls = []

        def fake_weixin_reader(_session, *, url: str) -> MCPToolCallResult:
            calls.append({"url": url})
            return MCPToolCallResult(
                tool_name="weixin-articles-mcp.read_article",
                ok=True,
                result={"title": "Tencent 2027", "content_text": "腾讯 2027 校园招聘"},
            )

        with self.Session() as session:
            session_id = self._session_id(session)
            skill_repository = self._skill_repository(session)
            skill = skill_repository.create_skill(
                AgentSkillCreate(
                    name="wechat-article-content-fetch",
                    title="WeChat Article Content Fetch",
                    description="Use this skill when the user provides a weixin mp.weixin.qq.com recruiting article URL.",
                    category="content_source",
                    metadata_json={"allowed_tools": ["weixin-articles-mcp.read_article"], "source_types": ["wechat_article"]},
                    sections={"workflow": "Read public WeChat recruiting articles."},
                )
            )
            skill_id = skill.id
            dependencies = self._dependencies(session, skill_repository=skill_repository)
            registry = AgentToolRegistry(
                definition
                for definition in dependencies.registry.list_definitions()
                if definition.name != "weixin-articles-mcp.read_article"
            )
            registry.register(
                AgentToolDefinition(
                    name="weixin-articles-mcp.read_article",
                    description="Read one public WeChat article.",
                    input_schema={"type": "object", "required": ["url"]},
                    output_schema={"type": "object", "required": ["tool_name", "ok"]},
                    handler=fake_weixin_reader,
                    allowed_source_types=frozenset({"agent_chat", "wechat_article"}),
                )
            )

            result = run_agent_workflow(
                AgentRunCommand(
                    session_id=session_id,
                    user_message="请读取这个 weixin 招聘文章 https://mp.weixin.qq.com/s/example 并提取秋招信息",
                ),
                dependencies=dependencies.with_registry(registry),
            )
            session.commit()

            tool_log = session.scalars(select(ToolCallLog)).one()

        self.assertIn(skill_id, result.state.loaded_skill_ids)
        self.assertEqual("weixin-articles-mcp.read_article", result.state.requested_tool_name)
        self.assertEqual("wechat_article", result.state.source_type)
        self.assertEqual([{"url": "https://mp.weixin.qq.com/s/example"}], calls)
        self.assertEqual(ToolCallStatus.SUCCEEDED, tool_log.status)
        self.assertEqual("weixin-articles-mcp.read_article", tool_log.tool_name)
        self.assertEqual({"url": "https://mp.weixin.qq.com/s/example"}, tool_log.input_payload)

    def test_agent_auto_selects_xiaohongshu_search_tool_from_keyword_request(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry
        from app.domains.agent_memory.schemas import AgentSkillCreate
        from app.domains.automation.models import ToolCallLog, ToolCallStatus
        from app.mcp_gateway.client import MCPToolCallResult

        calls = []

        def fake_search(_session, *, keyword: str, filters=None) -> MCPToolCallResult:
            calls.append({"keyword": keyword, "filters": filters})
            return MCPToolCallResult(
                tool_name="xiaohongshu-mcp.search_feeds",
                ok=True,
                result={"items": [{"title": "2027 秋招 Java"}]},
            )

        with self.Session() as session:
            session_id = self._session_id(session)
            skill_repository = self._skill_repository(session)
            skill_repository.create_skill(
                AgentSkillCreate(
                    name="xiaohongshu-content-fetch",
                    title="Xiaohongshu Content Fetch",
                    description="Use this skill when the user asks to search Xiaohongshu recruiting notes.",
                    category="content_source",
                    metadata_json={"allowed_tools": ["xiaohongshu-mcp.search_feeds"], "source_types": ["xiaohongshu_note"]},
                    sections={"workflow": "Search Xiaohongshu recruiting notes."},
                )
            )
            dependencies = self._dependencies(session, skill_repository=skill_repository)
            registry = AgentToolRegistry(
                definition
                for definition in dependencies.registry.list_definitions()
                if definition.name != "xiaohongshu-mcp.search_feeds"
            )
            registry.register(
                AgentToolDefinition(
                    name="xiaohongshu-mcp.search_feeds",
                    description="Search Xiaohongshu feeds.",
                    input_schema={"type": "object", "required": ["keyword"]},
                    output_schema={"type": "object", "required": ["tool_name", "ok"]},
                    handler=fake_search,
                    allowed_source_types=frozenset({"agent_chat", "xiaohongshu_note"}),
                )
            )

            message = "\u8bf7\u5728\u5c0f\u7ea2\u4e66\u641c\u7d22 2027 \u79cb\u62db Java \u5c97\u4f4d"
            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message=message),
                dependencies=dependencies.with_registry(registry),
            )
            session.commit()

            tool_log = session.scalars(select(ToolCallLog)).one()

        self.assertEqual("xiaohongshu-mcp.search_feeds", result.state.requested_tool_name)
        self.assertEqual("xiaohongshu_note", result.state.source_type)
        self.assertEqual([{"keyword": message, "filters": None}], calls)
        self.assertEqual(ToolCallStatus.SUCCEEDED, tool_log.status)
        self.assertEqual("xiaohongshu-mcp.search_feeds", tool_log.tool_name)

    def test_agent_records_failed_status_when_tool_result_ok_false(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry
        from app.domains.agent_memory.schemas import AgentSkillCreate
        from app.domains.automation.models import ToolCallLog, ToolCallStatus
        from app.domains.conversations.models import AgentMessageRole
        from app.domains.conversations.repository import ConversationRepository
        from app.mcp_gateway.client import MCPToolCallResult

        def fake_search(_session, *, keyword: str, filters=None) -> MCPToolCallResult:
            return MCPToolCallResult(
                tool_name="xiaohongshu-mcp.search_feeds",
                ok=False,
                error="XIAOHONGSHU_MCP_NOT_CONFIGURED",
                result={"message": "Xiaohongshu service is not configured."},
            )

        with self.Session() as session:
            session_id = self._session_id(session)
            skill_repository = self._skill_repository(session)
            skill_repository.create_skill(
                AgentSkillCreate(
                    name="xiaohongshu-content-fetch",
                    title="Xiaohongshu Content Fetch",
                    description="Use this skill when the user asks to search Xiaohongshu recruiting notes.",
                    category="content_source",
                    metadata_json={"allowed_tools": ["xiaohongshu-mcp.search_feeds"], "source_types": ["xiaohongshu_note"]},
                    sections={"workflow": "Search Xiaohongshu recruiting notes."},
                )
            )
            dependencies = self._dependencies(session, skill_repository=skill_repository)
            registry = AgentToolRegistry(
                definition
                for definition in dependencies.registry.list_definitions()
                if definition.name != "xiaohongshu-mcp.search_feeds"
            )
            registry.register(
                AgentToolDefinition(
                    name="xiaohongshu-mcp.search_feeds",
                    description="Search Xiaohongshu feeds.",
                    input_schema={"type": "object", "required": ["keyword"]},
                    output_schema={"type": "object", "required": ["tool_name", "ok"]},
                    handler=fake_search,
                    allowed_source_types=frozenset({"agent_chat", "xiaohongshu_note"}),
                )
            )

            result = run_agent_workflow(
                AgentRunCommand(
                    session_id=session_id,
                    user_message="\u8bf7\u5728\u5c0f\u7ea2\u4e66\u641c\u7d22 2027 \u79cb\u62db Java \u5c97\u4f4d",
                ),
                dependencies=dependencies.with_registry(registry),
            )
            session.commit()

            tool_log = session.scalars(select(ToolCallLog)).one()
            tool_result_message = next(
                message
                for message in ConversationRepository(session).list_messages(session_id, limit=20)
                if message.role == AgentMessageRole.TOOL_RESULT
            )

        self.assertEqual("xiaohongshu-mcp.search_feeds", result.state.requested_tool_name)
        self.assertEqual(ToolCallStatus.FAILED, tool_log.status)
        self.assertEqual("XIAOHONGSHU_MCP_NOT_CONFIGURED", tool_log.error)
        self.assertEqual("failed", tool_result_message.content_json["status"])
        self.assertEqual("XIAOHONGSHU_MCP_NOT_CONFIGURED", tool_result_message.content_json["error"])

    def test_agent_auto_selects_xiaohongshu_detail_tool_from_feed_id_and_xsec_token(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry
        from app.domains.agent_memory.schemas import AgentSkillCreate
        from app.domains.automation.models import ToolCallLog, ToolCallStatus
        from app.mcp_gateway.client import MCPToolCallResult

        calls = []

        def fake_detail(_session, **arguments) -> MCPToolCallResult:
            calls.append(arguments)
            return MCPToolCallResult(
                tool_name="xiaohongshu-mcp.get_feed_detail",
                ok=True,
                result={"feed_id": arguments["feed_id"], "text": "招聘信息"},
            )

        with self.Session() as session:
            session_id = self._session_id(session)
            skill_repository = self._skill_repository(session)
            skill_repository.create_skill(
                AgentSkillCreate(
                    name="xiaohongshu-detail-fetch",
                    title="Xiaohongshu Detail Fetch",
                    description="Use this skill when the user provides Xiaohongshu feed_id and xsec_token.",
                    category="content_source",
                    metadata_json={"allowed_tools": ["xiaohongshu-mcp.get_feed_detail"], "source_types": ["xiaohongshu_note"]},
                    sections={"workflow": "Read Xiaohongshu note detail."},
                )
            )
            dependencies = self._dependencies(session, skill_repository=skill_repository)
            registry = AgentToolRegistry(
                definition
                for definition in dependencies.registry.list_definitions()
                if definition.name != "xiaohongshu-mcp.get_feed_detail"
            )
            registry.register(
                AgentToolDefinition(
                    name="xiaohongshu-mcp.get_feed_detail",
                    description="Read Xiaohongshu feed detail.",
                    input_schema={"type": "object", "required": ["feed_id", "xsec_token"]},
                    output_schema={"type": "object", "required": ["tool_name", "ok"]},
                    handler=fake_detail,
                    allowed_source_types=frozenset({"agent_chat", "xiaohongshu_note"}),
                )
            )

            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="小红书 feed_id=abc123 xsec_token=token456 读取详情"),
                dependencies=dependencies.with_registry(registry),
            )
            session.commit()

            tool_log = session.scalars(select(ToolCallLog)).one()

        self.assertEqual("xiaohongshu-mcp.get_feed_detail", result.state.requested_tool_name)
        self.assertEqual("xiaohongshu_note", result.state.source_type)
        self.assertEqual([{"feed_id": "abc123", "xsec_token": "token456"}], calls)
        self.assertEqual(ToolCallStatus.SUCCEEDED, tool_log.status)
        self.assertEqual("xiaohongshu-mcp.get_feed_detail", tool_log.tool_name)

    def test_agent_run_uses_llm_client_for_final_response(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow

        class FakeLLMClient:
            def __init__(self) -> None:
                self.messages = None

            def complete(self, *, messages):
                from app.infrastructure.llm.chat_client import LLMChatCompletion

                self.messages = messages
                return LLMChatCompletion(content="百炼模型回复：我会先确认你的岗位目标。", usage={"total_tokens": 42})

        with self.Session() as session:
            session_id = self._session_id(session)
            fake_llm = FakeLLMClient()
            dependencies = self._dependencies(session, llm_client=fake_llm)

            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="我想找 Java 后端秋招"),
                dependencies=dependencies,
            )
            session.commit()

        self.assertEqual("百炼模型回复：我会先确认你的岗位目标。", result.state.final_response)
        self.assertEqual("llm", result.state.response_mode)
        self.assertIsNotNone(fake_llm.messages)
        self.assertEqual("user", fake_llm.messages[-1]["role"])
        self.assertEqual("我想找 Java 后端秋招", fake_llm.messages[-1]["content"])

    def test_agent_run_auto_compacts_when_context_exceeds_budget_before_llm_call(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.domains.conversations.models import AgentContextSummary, AgentMessageRole
        from app.domains.conversations.schemas import AgentMessageCreate

        class FakeLLMClient:
            def __init__(self) -> None:
                self.messages = None

            def complete(self, *, messages):
                from app.infrastructure.llm.chat_client import LLMChatCompletion

                self.messages = messages
                return LLMChatCompletion(content="ok")

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(session, llm_client=FakeLLMClient())
            for index in range(8):
                dependencies.conversation_service.append_message(
                    session_id,
                    AgentMessageCreate(
                        role=AgentMessageRole.USER if index % 2 == 0 else AgentMessageRole.ASSISTANT,
                        content_text=f"large historical message {index}",
                        visible_content_text=f"large historical message {index}",
                        token_estimate=10_000,
                    ),
                )

            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="new turn after a long transcript"),
                dependencies=dependencies,
            )
            session.commit()

            summaries = list(session.scalars(select(AgentContextSummary)).all())
            messages = dependencies.conversation_service.list_messages(session_id, limit=20)

        self.assertEqual(1, len(summaries))
        self.assertEqual(summaries[0].id, result.state.latest_summary_id)
        self.assertTrue(result.state.context_metadata["auto_compacted"])
        self.assertGreater(result.state.context_metadata["auto_compacted_message_count"], 0)
        self.assertFalse(result.state.need_compaction)
        self.assertTrue(any(message.compacted_by_summary_id == summaries[0].id for message in messages))

    def test_checkpoint_store_makes_checkpoint_timestamps_monotonic(self) -> None:
        from app.agent_runtime.checkpoints import AgentCheckpointStore
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.domains.automation.models import WorkflowCheckpoint

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(session)
            result = run_agent_workflow(
                AgentRunCommand(
                    session_id=session_id,
                    user_message="check checkpoint ordering",
                    requested_tool_name="sessions_search",
                    source_type="agent_chat",
                ),
                dependencies=dependencies,
            )
            session.commit()

            checkpoints = list(
                session.scalars(
                    select(WorkflowCheckpoint)
                    .where(WorkflowCheckpoint.workflow_run_id == result.workflow_run_id)
                    .order_by(WorkflowCheckpoint.created_at.asc())
                ).all()
            )
            latest = AgentCheckpointStore(
                session=session,
                automation_service=dependencies.automation_service,
            ).load_latest(result.workflow_run_id)

        timestamps = [checkpoint.created_at for checkpoint in checkpoints]
        self.assertEqual(len(timestamps), len(set(timestamps)))
        self.assertEqual("final_response", latest.checkpoint_key)
        self.assertEqual(result.state.tool_call_ids, latest.state.tool_call_ids)

    def test_create_agent_graph_exposes_expected_node_order(self) -> None:
        from app.agent_runtime.graph_factory import create_agent_graph

        graph = create_agent_graph()

        self.assertEqual(
            ["build_context", "plan_or_reply", "maybe_tool", "wait_confirmation", "final_response"],
            graph.node_order,
        )
        self.assertIsNotNone(graph.compiled_graph)
