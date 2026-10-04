import sys
import unittest
import shutil
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class AgentRuntimeGraphTest(unittest.TestCase):
    def setUp(self) -> None:
        from app.db.base import Base
        import app.agent_runtime.durable_state.models  # noqa: F401
        import app.agent_runtime.external_tasks.models  # noqa: F401
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

    def _dependencies(
        self,
        session,
        *,
        memory_repository=None,
        skill_repository=None,
        conversation_service=None,
        llm_client=None,
        intent_detector=None,
        execution_planner=None,
        capability_routing_middleware=None,
        durable_state_service=None,
    ):
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
            conversation_service=conversation_service or ConversationService(ConversationRepository(session)),
            registry=create_default_agent_tool_registry(),
            guard=AgentToolRuntimeGuard(policy=AgentToolPolicy(max_tool_calls=10)),
            memory_repository=memory_repository,
            skill_repository=skill_repository,
            db_session=session,
            llm_client=llm_client,
            intent_detector=intent_detector,
            execution_planner=execution_planner,
            capability_routing_middleware=capability_routing_middleware,
            durable_state_service=durable_state_service,
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

    def test_local_company_overview_uses_requested_sample_count_from_user_message(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, _resolved_tool_input
        from app.agent_runtime.state import AgentState
        from app.agent_runtime.tool_registry import LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL

        for message, expected_limit in (
            ("给我看一下有哪些公司，给我20个就行", 20),
            ("列出三十七家公司", 37),
            ("给我80家公司", 50),
        ):
            with self.subTest(message=message):
                command = AgentRunCommand(
                    session_id="session-1",
                    user_message=message,
                    requested_tool_name=LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL,
                )
                state = AgentState(
                    session_id="session-1",
                    workflow_run_id="workflow-1",
                    agent_run_id="agent-run-1",
                    user_message=message,
                    current_step="maybe_tool",
                    requested_tool_name=LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL,
                )

                self.assertEqual({"sample_limit": expected_limit}, _resolved_tool_input(command, state))

    def test_database_company_list_uses_requested_limit_from_user_message(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, _resolved_tool_input
        from app.agent_runtime.state import AgentState
        from app.agent_runtime.tool_registry import DATABASE_COMPANY_LIST_TOOL

        for message, expected_limit in (
            ("给我看一下有哪些公司，给我20个就行", 20),
            ("列出三十七家公司", 37),
            ("给我80家公司", 50),
        ):
            with self.subTest(message=message):
                command = AgentRunCommand(
                    session_id="session-1",
                    user_message=message,
                    requested_tool_name=DATABASE_COMPANY_LIST_TOOL,
                )
                state = AgentState(
                    session_id="session-1",
                    workflow_run_id="workflow-1",
                    agent_run_id="agent-run-1",
                    user_message=message,
                    current_step="maybe_tool",
                    requested_tool_name=DATABASE_COMPANY_LIST_TOOL,
                )

                self.assertEqual({"limit": expected_limit}, _resolved_tool_input(command, state))

    def test_explicit_required_capability_skips_broad_tool_choice_loop(self) -> None:
        from app.agent_runtime.graph_factory import _should_offer_tool_choice_before_specialized_routing
        from app.agent_runtime.state import AgentState

        state = AgentState(
            session_id="session-1",
            workflow_run_id="workflow-1",
            agent_run_id="agent-run-1",
            user_message="只通过 DBX 只读子 Agent 查询公司展览公司数",
            current_step="plan_or_reply",
            context_metadata={
                "intent_frame": {
                    "intent": "filesystem_operation",
                    "confidence": 0.98,
                    "required_capability": "agent.dbx_readonly",
                },
                "context_pack": {
                    "required_capability": "agent.dbx_readonly",
                    "allowed_capabilities": ["agent.dbx_readonly"],
                },
            },
        )

        self.assertFalse(_should_offer_tool_choice_before_specialized_routing(state))

    def test_database_company_list_runtime_helpers_are_labeled_as_local_collection(self) -> None:
        from app.agent_runtime.graph_factory import (
            _runtime_reasoning_summary,
            format_runtime_capability_name,
        )
        from app.api.v1.agent import _plan_stage_index_for_tool_name
        from app.agent_runtime.tool_registry import DATABASE_COMPANY_LIST_TOOL

        self.assertEqual("本地公司列表", format_runtime_capability_name(DATABASE_COMPANY_LIST_TOOL))
        self.assertEqual(2, _plan_stage_index_for_tool_name(DATABASE_COMPANY_LIST_TOOL))
        self.assertIn(
            "本地公司列表",
            _runtime_reasoning_summary(
                DATABASE_COMPANY_LIST_TOOL,
                tool_input={"limit": 20},
                user_message="数据库里有哪些公司，给我20个",
            ),
        )

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

    def test_context_builder_loaded_skill_and_history_are_recorded_as_memory_snapshots(self) -> None:
        from app.agent_runtime.durable_state.models import AgentMemorySnapshot
        from app.agent_runtime.durable_state.repository import SqlAlchemyDurableStateRepository
        from app.agent_runtime.durable_state.service import DurableStateService
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.domains.agent_memory.schemas import AgentSkillCreate
        from app.domains.conversations.models import AgentMessageRole
        from app.domains.conversations.schemas import AgentMessageCreate

        with self.Session() as session:
            session_id = self._session_id(session)
            skill_repository = self._skill_repository(session)
            skill = skill_repository.create_skill(
                AgentSkillCreate(
                    name="java-context-snapshot",
                    title="Java Context Snapshot",
                    description="Use this skill when the user asks for Java campus recruiting leads.",
                    category="job_discovery",
                )
            )
            durable_state_service = DurableStateService(SqlAlchemyDurableStateRepository(session))
            dependencies = self._dependencies(
                session,
                skill_repository=skill_repository,
                durable_state_service=durable_state_service,
            )
            previous_message = dependencies.conversation_service.append_message(
                session_id,
                AgentMessageCreate(
                    role=AgentMessageRole.USER,
                    content_text="previous Java campus target context",
                    visible_content_text="previous Java campus target context",
                    token_estimate=8,
                ),
            )

            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="Java"),
                dependencies=dependencies,
            )
            session.commit()

            snapshots = list(session.scalars(select(AgentMemorySnapshot)).all())

        snapshots_by_memory_id = {snapshot.memory_id: snapshot for snapshot in snapshots}
        self.assertEqual("final_response", result.state.current_step)
        self.assertIn(skill.id, snapshots_by_memory_id)
        self.assertIn(previous_message.id, snapshots_by_memory_id)
        self.assertEqual("agent_skill", snapshots_by_memory_id[skill.id].source_type)
        self.assertIn("ContextBuilder loaded skill", snapshots_by_memory_id[skill.id].usage_reason)
        self.assertEqual("session_history", snapshots_by_memory_id[previous_message.id].source_type)
        self.assertIn("ContextBuilder loaded session history", snapshots_by_memory_id[previous_message.id].usage_reason)
        self.assertFalse(snapshots_by_memory_id[skill.id].passed_to_executor)
        self.assertFalse(snapshots_by_memory_id[previous_message.id].passed_to_executor)

    def test_context_builder_loaded_long_term_memory_is_recorded_as_memory_snapshot(self) -> None:
        from app.agent_runtime.durable_state.models import AgentMemorySnapshot
        from app.agent_runtime.durable_state.repository import SqlAlchemyDurableStateRepository
        from app.agent_runtime.durable_state.service import DurableStateService
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.domains.agent_memory.models import AgentMemory, AgentMemoryStatus
        from app.domains.agent_memory.repository import AgentMemoryRepository

        with self.Session() as session:
            session_id = self._session_id(session)
            memory = AgentMemory(
                id="memory-submit-confirmation",
                memory_type="user_preference",
                scope="application_submission",
                title="投递前必须用户确认",
                content="任何岗位最终提交前都必须等待用户确认。",
                source_type="user_profile",
                status=AgentMemoryStatus.ACTIVE,
                importance=95,
            )
            session.add(memory)
            session.commit()

            durable_state_service = DurableStateService(SqlAlchemyDurableStateRepository(session))
            dependencies = self._dependencies(
                session,
                memory_repository=AgentMemoryRepository(session),
                durable_state_service=durable_state_service,
            )

            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="帮我投递腾讯 Java 岗位"),
                dependencies=dependencies,
            )
            session.commit()

            snapshots = list(session.scalars(select(AgentMemorySnapshot)).all())

        snapshots_by_memory_id = {snapshot.memory_id: snapshot for snapshot in snapshots}
        self.assertEqual([memory.id], result.state.loaded_memory_ids)
        self.assertIn(memory.id, snapshots_by_memory_id)
        self.assertEqual("agent_memory", snapshots_by_memory_id[memory.id].source_type)
        self.assertIn("ContextBuilder loaded long-term memory", snapshots_by_memory_id[memory.id].usage_reason)
        self.assertFalse(snapshots_by_memory_id[memory.id].passed_to_executor)

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

    def test_tool_result_declared_resource_effects_are_promoted_without_tool_specific_runtime_code(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry
        from app.domains.automation.models import ToolCallLog, ToolCallStatus

        effect = {
            "resource_type": "file",
            "action": "created",
            "operation": "export_pdf",
            "source_path": "C:/简历/resume.tex",
            "target_path": "C:/简历/resume.pdf",
            "focus_path": "C:/简历/resume.pdf",
            "aliases": ["刚才生成的 PDF"],
        }

        def export_handler(_session, **_arguments):
            return {
                "tool_name": "resume.export_pdf",
                "ok": True,
                "summary": "PDF 已导出。",
                "resource_effects": [effect],
            }

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(session)
            registry = AgentToolRegistry(dependencies.registry.list_definitions())
            registry.register(
                AgentToolDefinition(
                    name="resume.export_pdf",
                    description="Export a resume source file to PDF and report the created artifact.",
                    input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "additionalProperties": False},
                    output_schema={"type": "object"},
                    handler=export_handler,
                )
            )
            dependencies = dependencies.with_registry(registry)

            result = run_agent_workflow(
                AgentRunCommand(
                    session_id=session_id,
                    user_message="导出 PDF",
                    requested_tool_name="resume.export_pdf",
                    tool_input={"path": "C:/简历/resume.tex"},
                ),
                dependencies=dependencies,
            )
            session.commit()
            tool_logs = list(session.scalars(select(ToolCallLog)).all())

        self.assertEqual("final_response", result.state.current_step)
        self.assertTrue(tool_logs)
        self.assertTrue(all(log.status == ToolCallStatus.SUCCEEDED for log in tool_logs))
        self.assertEqual("C:/简历/resume.pdf", result.state.context_metadata["active_resource"]["focus_path"])
        self.assertEqual("export_pdf", result.state.context_metadata["active_resource"]["operation"])
        self.assertEqual("C:/简历/resume.pdf", result.state.context_metadata["artifact_context"]["active_artifact"]["focus_path"])

    def test_tool_result_contract_template_promotes_resource_effect_without_result_effect_field(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry, AgentToolSemanticProfile
        from app.domains.automation.models import ToolCallLog, ToolCallStatus

        def export_handler(_session, **arguments):
            return {
                "tool_name": "resume.export_pdf",
                "ok": True,
                "summary": "PDF 已导出。",
                "result": {"output_path": arguments["path"].replace(".tex", ".pdf")},
            }

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(session)
            registry = AgentToolRegistry(dependencies.registry.list_definitions())
            registry.register(
                AgentToolDefinition(
                    name="resume.export_pdf",
                    description="Export a resume source file to PDF and declare its artifact contract.",
                    input_schema={
                        "type": "object",
                        "required": ["path"],
                        "properties": {"path": {"type": "string"}},
                        "additionalProperties": False,
                    },
                    output_schema={"type": "object"},
                    handler=export_handler,
                    semantic_profile=AgentToolSemanticProfile(
                        intent="export_resume_pdf",
                        target_type="file_artifact",
                        result_contract={
                            "resource_effect_templates": [
                                {
                                    "resource_type": "file",
                                    "action": "created",
                                    "operation": "export_pdf",
                                    "source_path": "$input.path",
                                    "target_path": "$result.result.output_path",
                                    "focus_path": "$result.result.output_path",
                                    "aliases": ["刚才生成的 PDF", "导出的 PDF"],
                                }
                            ]
                        },
                    ),
                )
            )
            dependencies = dependencies.with_registry(registry)

            result = run_agent_workflow(
                AgentRunCommand(
                    session_id=session_id,
                    user_message="导出 PDF",
                    requested_tool_name="resume.export_pdf",
                    tool_input={"path": "C:/简历/resume.tex"},
                ),
                dependencies=dependencies,
            )
            session.commit()
            tool_log = session.scalars(select(ToolCallLog)).one()

        self.assertEqual("final_response", result.state.current_step)
        self.assertEqual(ToolCallStatus.SUCCEEDED, tool_log.status)
        self.assertEqual("C:/简历/resume.pdf", result.state.context_metadata["active_resource"]["focus_path"])
        self.assertEqual("C:/简历/resume.tex", result.state.context_metadata["active_resource"]["source_path"])
        self.assertEqual("export_pdf", result.state.context_metadata["active_resource"]["operation"])

    def test_stale_approval_continuation_does_not_mark_pending_request_approved(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, StaleApprovalContinuationError, continue_agent_workflow_after_approval, run_agent_workflow
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry, AgentToolRiskLevel
        from app.domains.automation.models import ApprovalRequest, ApprovalRequestStatus, WorkflowRun, WorkflowRunStatus, utc_now

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
            approval = session.scalars(select(ApprovalRequest)).one()
            workflow = session.get(WorkflowRun, result.workflow_run_id)
            assert workflow is not None
            workflow.status = WorkflowRunStatus.COMPLETED
            workflow.current_step = "final_response"
            workflow.completed_at = utc_now()
            dependencies.checkpoint_store.save(
                workflow_run_id=workflow.id,
                checkpoint_key="final_response",
                state=result.state.with_updates(current_step="final_response", final_response="already finished"),
            )

            with self.assertRaises(StaleApprovalContinuationError) as raised:
                continue_agent_workflow_after_approval(
                    approval.id,
                    approved=True,
                    decision_reason="clicked stale button",
                    dependencies=dependencies,
                )
            session.flush()

        self.assertEqual("STALE_APPROVAL_REQUEST", raised.exception.error_code)
        self.assertEqual(ApprovalRequestStatus.PENDING, approval.status)
        self.assertEqual(WorkflowRunStatus.COMPLETED, workflow.status)
        self.assertEqual("final_response", workflow.current_step)

    def test_skill_permission_snapshot_denies_tool_before_execution(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.domains.agent_memory.schemas import AgentSkillCreate
        from app.domains.automation.models import ApprovalRequest, ToolCallLog

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
        self.assertEqual("agent_runtime", tool_log.output_payload["execution"])
        self.assertEqual("agent_tool_registry", tool_log.output_payload["agent_runtime"]["executor_id"])
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
        self.assertEqual("agent_runtime", tool_log.output_payload["execution"])
        self.assertEqual("agent_tool_registry", tool_log.output_payload["agent_runtime"]["executor_id"])
        self.assertEqual("Example", tool_log.output_payload["result"]["result"]["title"])
        self.assertEqual([tool_log.id], result.state.tool_call_ids)

    def test_unavailable_tool_handler_records_structured_failure_envelope(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry
        from app.domains.automation.models import ToolCallLog, ToolCallStatus
        from app.domains.conversations.models import AgentMessageRole

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(session)
            registry = AgentToolRegistry(dependencies.registry.list_definitions())
            registry.register(
                AgentToolDefinition(
                    name="custom.no_handler",
                    description="A registered tool without an executable handler.",
                    input_schema={"type": "object", "properties": {}, "additionalProperties": False},
                    output_schema={"type": "object"},
                    handler=None,
                    allowed_source_types=frozenset({"agent_chat"}),
                )
            )

            result = run_agent_workflow(
                AgentRunCommand(
                    session_id=session_id,
                    user_message="调用不可用工具",
                    requested_tool_name="custom.no_handler",
                    source_type="agent_chat",
                ),
                dependencies=dependencies.with_registry(registry),
            )
            session.commit()

            tool_log = session.scalars(select(ToolCallLog)).one()
            messages = dependencies.conversation_service.list_messages(session_id, limit=20)

        self.assertEqual(ToolCallStatus.FAILED, tool_log.status)
        self.assertEqual("handler_unavailable", tool_log.output_payload["execution"])
        envelope = tool_log.output_payload["result"]["result_envelope"]
        self.assertEqual("failed", envelope["status"])
        self.assertEqual("custom.no_handler", envelope["capability"])
        self.assertEqual("TOOL_HANDLER_UNAVAILABLE", envelope["error_code"])
        self.assertFalse(envelope["retryable"])
        self.assertEqual("select_alternative_tool", envelope["next_action"])

        tool_results = [message for message in messages if message.role == AgentMessageRole.TOOL_RESULT]
        self.assertEqual(1, len(tool_results))
        self.assertEqual("TOOL_HANDLER_UNAVAILABLE", tool_results[0].content_json["result"]["result_envelope"]["error_code"])
        self.assertEqual([tool_log.id], result.state.tool_call_ids)

    def test_unstarted_tool_result_includes_goal_validation_recovery_hint(self) -> None:
        from app.agent_runtime.agent_as_tool import FILESYSTEM_SKILL_CAPABILITY
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry
        from app.domains.automation.models import ToolCallLog, ToolCallStatus
        from app.domains.conversations.models import AgentMessageRole
        from app.domains.conversations.schemas import AgentMessageCreate

        original_path = "C:/Users/phoenix/Documents/Obsidian Vault/简历/刘汉卿-后端开发-AI-Agent平台简历.tex"

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(session)
            registry = AgentToolRegistry(dependencies.registry.list_definitions())
            registry.register(
                AgentToolDefinition(
                    name="custom.no_handler",
                    description="A registered tool without an executable handler.",
                    input_schema={"type": "object", "properties": {}, "additionalProperties": False},
                    output_schema={"type": "object"},
                    handler=None,
                    allowed_source_types=frozenset({"agent_chat"}),
                )
            )
            dependencies.conversation_service.append_message(
                session_id,
                AgentMessageCreate(
                    role=AgentMessageRole.USER,
                    content_text=f"刚才查看这个文件的名字：{original_path}",
                    visible_content_text=f"刚才查看这个文件的名字：{original_path}",
                    token_estimate=12,
                ),
            )

            result = run_agent_workflow(
                AgentRunCommand(
                    session_id=session_id,
                    user_message="把这个文件名字改成 刘汉卿-后端开发-AI-Agent.tex",
                    requested_tool_name="custom.no_handler",
                    source_type="agent_chat",
                ),
                dependencies=dependencies.with_registry(registry),
            )
            session.commit()
            tool_log = session.scalars(select(ToolCallLog)).one()

        result_envelope = tool_log.output_payload["result"]["result_envelope"]

        self.assertEqual(ToolCallStatus.FAILED, tool_log.status)
        self.assertIsNone(result.state.context_metadata["goal_state"]["expected_operation"])
        self.assertEqual("TOOL_HANDLER_UNAVAILABLE", result_envelope["error_code"])
        self.assertFalse(result_envelope["retryable"])
        self.assertEqual("select_alternative_tool", result_envelope["next_action"])

    def test_web_search_can_be_dispatched_to_registered_claude_sdk_agent_executor(self) -> None:
        from app.agent_runtime.agent_as_tool import CLAUDE_SDK_AGENT_EXECUTOR_ID, AgentRuntimeContext, AgentTask, StandardAgentResult
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL, AgentToolDefinition, AgentToolRegistry
        from app.domains.automation.models import ToolCallLog, ToolCallStatus

        calls = []

        class FakeClaudeSdkAgent:
            def call(self, task: AgentTask, context: AgentRuntimeContext) -> StandardAgentResult:
                calls.append({"task": task, "context": context})
                return StandardAgentResult(
                    status="succeeded",
                    summary="腾讯校招官网：https://join.qq.com/",
                    observation="腾讯校招官网可查看产品岗。",
                    evidence=[{"type": "url", "title": "腾讯校招", "url": "https://join.qq.com/"}],
                    raw_result={
                        "tool_name": EXTERNAL_WEB_SEARCH_TOOL,
                        "ok": True,
                        "result": {
                            "executor_name": CLAUDE_SDK_AGENT_EXECUTOR_ID,
                            "query": task.input_payload["query"],
                            "answer": "腾讯校招官网：https://join.qq.com/",
                            "artifacts": [{"type": "url", "title": "腾讯校招", "url": "https://join.qq.com/"}],
                        },
                    },
                )

        def legacy_handler(_session, **_arguments):
            raise AssertionError("legacy handler should not run when claude-sdk-agent is registered for web search")

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(session)
            registry = AgentToolRegistry(
                definition
                for definition in dependencies.registry.list_definitions()
                if definition.name != EXTERNAL_WEB_SEARCH_TOOL
            )
            registry.register(
                AgentToolDefinition(
                    name=EXTERNAL_WEB_SEARCH_TOOL,
                    description="Search the public web through an external agent.",
                    input_schema={"type": "object", "required": ["query"], "properties": {"query": {"type": "string"}}},
                    output_schema={"type": "object", "required": ["tool_name", "ok", "result"]},
                    handler=legacy_handler,
                    allowed_source_types=frozenset({"agent_chat", "web_search"}),
                )
            )
            dependencies = dependencies.with_registry(registry).with_agent_runtime(
                executors={CLAUDE_SDK_AGENT_EXECUTOR_ID: FakeClaudeSdkAgent()},
                capability_executor_ids={EXTERNAL_WEB_SEARCH_TOOL: CLAUDE_SDK_AGENT_EXECUTOR_ID},
            )

            result = run_agent_workflow(
                AgentRunCommand(
                    session_id=session_id,
                    user_message="搜一下腾讯校招官网",
                    requested_tool_name=EXTERNAL_WEB_SEARCH_TOOL,
                    source_type="agent_chat",
                    tool_input={"query": "腾讯 校园招聘 官网"},
                ),
                dependencies=dependencies,
            )
            session.commit()
            tool_log = session.scalars(select(ToolCallLog)).one()

        self.assertEqual(ToolCallStatus.SUCCEEDED, tool_log.status)
        self.assertEqual("agent_runtime", tool_log.output_payload["execution"])
        self.assertEqual(CLAUDE_SDK_AGENT_EXECUTOR_ID, tool_log.output_payload["agent_runtime"]["executor_id"])
        self.assertEqual(CLAUDE_SDK_AGENT_EXECUTOR_ID, tool_log.output_payload["result"]["result_envelope"]["executor"])
        self.assertEqual("腾讯 校园招聘 官网", calls[0]["task"].input_payload["query"])
        self.assertEqual(CLAUDE_SDK_AGENT_EXECUTOR_ID, calls[0]["context"].namespace)
        self.assertEqual([tool_log.id], result.state.tool_call_ids)

    def test_sdk_agent_tool_approval_interruption_creates_waiting_approval(self) -> None:
        from app.agent_runtime.agent_as_tool import OPENAI_SDK_AGENT_EXECUTOR_ID, AgentRuntimeContext, AgentTask
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.sdk_agents.schemas import SdkAgentResultEnvelope, SdkToolApprovalRequest
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry, FILESYSTEM_DELETE_PATH_TOOL
        from app.domains.automation.models import ApprovalRequest, ToolCallLog, ToolCallStatus, WorkflowRun, WorkflowRunStatus

        class FakeOpenAISdkAgent:
            def call(self, task: AgentTask, context: AgentRuntimeContext):
                return SdkAgentResultEnvelope(
                    task_id=context.task_id,
                    capability_id=task.capability_id,
                    subagent_name="FileAnalysisAgent",
                    status="needs_approval",
                    summary="子 Agent 想删除临时文件，等待用户确认。",
                    approval_request=SdkToolApprovalRequest(
                        approval_type="tool_call",
                        tool_name=FILESYSTEM_DELETE_PATH_TOOL,
                        tool_input={"path": "F:/pythonProject/OfferMaster/runtime/tmp-cache.json", "force": True},
                        reason="删除文件属于高风险动作，必须用户确认。",
                        suggested_user_message="确认删除 runtime/tmp-cache.json 吗？",
                    ),
                    raw_trace_ref="sdk-run-trace-1",
                ).to_standard_agent_result()

        with self.Session() as session:
            session_id = self._session_id(session)
            outer_capability = "agent.file_analysis"
            dependencies = self._dependencies(session)
            registry = AgentToolRegistry(
                definition
                for definition in dependencies.registry.list_definitions()
                if definition.name != outer_capability
            )
            registry.register(
                AgentToolDefinition(
                    name=outer_capability,
                    description="Delegate local file analysis to a child agent.",
                    input_schema={
                        "type": "object",
                        "required": ["task"],
                        "properties": {"task": {"type": "string"}},
                        "additionalProperties": False,
                    },
                    output_schema={"type": "object"},
                    handler=None,
                    allowed_source_types=frozenset({"agent_chat"}),
                )
            )
            dependencies = dependencies.with_registry(registry).with_agent_runtime(
                executors={OPENAI_SDK_AGENT_EXECUTOR_ID: FakeOpenAISdkAgent()},
                capability_executor_ids={outer_capability: OPENAI_SDK_AGENT_EXECUTOR_ID},
            )

            result = run_agent_workflow(
                AgentRunCommand(
                    session_id=session_id,
                    user_message="读取缓存并清理无用临时文件",
                    requested_tool_name=outer_capability,
                    source_type="agent_chat",
                    tool_input={"task": "读取缓存并清理无用临时文件"},
                ),
                dependencies=dependencies,
            )
            session.commit()
            workflow = session.get(WorkflowRun, result.workflow_run_id)
            approval = session.scalars(select(ApprovalRequest)).one()
            tool_log = session.scalars(select(ToolCallLog)).one()

        self.assertEqual(WorkflowRunStatus.WAITING_USER, workflow.status)
        self.assertEqual("wait_confirmation", workflow.current_step)
        self.assertEqual("wait_confirmation", result.state.current_step)
        self.assertEqual(approval.id, result.state.approval_request_id)
        self.assertEqual(FILESYSTEM_DELETE_PATH_TOOL, approval.action_type)
        self.assertTrue(approval.payload["sdk_agent_approval"])
        self.assertEqual(FILESYSTEM_DELETE_PATH_TOOL, approval.payload["requested_tool_name"])
        self.assertEqual(outer_capability, approval.payload["outer_capability"])
        self.assertEqual({"task": "读取缓存并清理无用临时文件"}, approval.payload["outer_tool_input"])
        self.assertEqual("F:/pythonProject/OfferMaster/runtime/tmp-cache.json", approval.payload["tool_input"]["path"])
        self.assertEqual(ToolCallStatus.BLOCKED, tool_log.status)
        self.assertEqual(outer_capability, tool_log.tool_name)
        self.assertIn("SDK_AGENT_TOOL_APPROVAL_REQUIRED", tool_log.error)
        self.assertEqual([tool_log.id], result.state.tool_call_ids)

    def test_approved_sdk_agent_tool_interruption_executes_only_inner_tool_once(self) -> None:
        from app.agent_runtime.agent_as_tool import OPENAI_SDK_AGENT_EXECUTOR_ID, AgentRuntimeContext, AgentTask
        from app.agent_runtime.graph_factory import AgentRunCommand, continue_agent_workflow_after_approval, run_agent_workflow
        from app.agent_runtime.sdk_agents.schemas import SdkAgentResultEnvelope, SdkToolApprovalRequest
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry, AgentToolRiskLevel
        from app.domains.automation.models import ApprovalRequest, ToolCallLog, ToolCallStatus, WorkflowRun, WorkflowRunStatus

        outer_capability = "agent.file_analysis"
        inner_tool = "sdk.inner_delete"
        sdk_calls = []
        inner_calls = []

        class FakeOpenAISdkAgent:
            def call(self, task: AgentTask, context: AgentRuntimeContext):
                sdk_calls.append({"task": task, "context": context})
                return SdkAgentResultEnvelope(
                    task_id=context.task_id,
                    capability_id=task.capability_id,
                    subagent_name="FileAnalysisAgent",
                    status="needs_approval",
                    summary="子 Agent 想删除临时文件，等待用户确认。",
                    approval_request=SdkToolApprovalRequest(
                        approval_type="tool_call",
                        tool_name=inner_tool,
                        tool_input={"path": "runtime/tmp-cache.json", "force": True},
                        reason="删除文件属于高风险动作，必须用户确认。",
                        suggested_user_message="确认删除 runtime/tmp-cache.json 吗？",
                    ),
                    raw_trace_ref="sdk-run-trace-approve",
                ).to_standard_agent_result()

        def fake_inner_delete(_session, *, path: str, force: bool = False):
            inner_calls.append({"path": path, "force": force})
            return {"tool_name": inner_tool, "ok": True, "result": {"deleted": path, "force": force}}

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(session)
            registry = AgentToolRegistry(dependencies.registry.list_definitions())
            registry.register(
                AgentToolDefinition(
                    name=outer_capability,
                    description="Delegate file analysis to the SDK sub-agent.",
                    input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "additionalProperties": False},
                    output_schema={"type": "object"},
                    handler=None,
                    allowed_source_types=frozenset({"agent_chat"}),
                )
            )
            registry.register(
                AgentToolDefinition(
                    name=inner_tool,
                    description="Delete a file requested by the SDK sub-agent after approval.",
                    input_schema={
                        "type": "object",
                        "required": ["path"],
                        "properties": {"path": {"type": "string"}, "force": {"type": "boolean"}},
                        "additionalProperties": False,
                    },
                    output_schema={"type": "object"},
                    handler=fake_inner_delete,
                    risk_level=AgentToolRiskLevel.HIGH,
                    requires_confirmation=True,
                    allowed_source_types=frozenset({"agent_chat"}),
                )
            )
            dependencies = dependencies.with_registry(registry).with_agent_runtime(
                executors={OPENAI_SDK_AGENT_EXECUTOR_ID: FakeOpenAISdkAgent()},
                capability_executor_ids={outer_capability: OPENAI_SDK_AGENT_EXECUTOR_ID},
            )

            first = run_agent_workflow(
                AgentRunCommand(
                    session_id=session_id,
                    user_message="分析缓存目录并清理无用临时文件",
                    requested_tool_name=outer_capability,
                    source_type="agent_chat",
                    tool_input={"path": "runtime"},
                ),
                dependencies=dependencies,
            )
            approval = session.scalars(select(ApprovalRequest)).one()
            workflow = session.get(WorkflowRun, first.workflow_run_id)

            self.assertEqual(WorkflowRunStatus.WAITING_USER, workflow.status)
            self.assertEqual("wait_confirmation", first.state.current_step)
            self.assertEqual([], inner_calls)

            continued = continue_agent_workflow_after_approval(
                approval.id,
                approved=True,
                decision_reason="allow one cleanup",
                dependencies=dependencies,
            )
            session.commit()
            workflow = session.get(WorkflowRun, first.workflow_run_id)
            tool_logs = list(session.scalars(select(ToolCallLog)).all())

        self.assertEqual(1, len(sdk_calls))
        self.assertEqual([{"path": "runtime/tmp-cache.json", "force": True}], inner_calls)
        self.assertEqual(WorkflowRunStatus.COMPLETED, workflow.status)
        self.assertEqual("final_response", continued.state.current_step)
        self.assertEqual(inner_tool, continued.state.requested_tool_name)
        self.assertEqual([ToolCallStatus.BLOCKED, ToolCallStatus.SUCCEEDED], [log.status for log in tool_logs])
        self.assertEqual([outer_capability, inner_tool], [log.tool_name for log in tool_logs])
        self.assertEqual([log.id for log in tool_logs], continued.state.tool_call_ids)
        decision = continued.state.context_metadata["sdk_agent_approval_decision"]
        self.assertTrue(decision["approved"])
        self.assertEqual(inner_tool, decision["requested_tool_name"])
        self.assertEqual(outer_capability, decision["outer_capability"])
        self.assertEqual("allow one cleanup", decision["decision_reason"])

    def test_rejected_sdk_agent_tool_interruption_stops_without_inner_tool_execution(self) -> None:
        from app.agent_runtime.agent_as_tool import OPENAI_SDK_AGENT_EXECUTOR_ID, AgentRuntimeContext, AgentTask
        from app.agent_runtime.graph_factory import AgentRunCommand, continue_agent_workflow_after_approval, run_agent_workflow
        from app.agent_runtime.sdk_agents.schemas import SdkAgentResultEnvelope, SdkToolApprovalRequest
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry, AgentToolRiskLevel
        from app.domains.automation.models import ApprovalRequest, WorkflowRun, WorkflowRunStatus

        outer_capability = "agent.file_analysis"
        inner_tool = "sdk.inner_delete"
        sdk_calls = []
        inner_calls = []

        class FakeOpenAISdkAgent:
            def call(self, task: AgentTask, context: AgentRuntimeContext):
                sdk_calls.append({"task": task, "context": context})
                return SdkAgentResultEnvelope(
                    task_id=context.task_id,
                    capability_id=task.capability_id,
                    subagent_name="FileAnalysisAgent",
                    status="needs_approval",
                    summary="子 Agent 想删除临时文件，等待用户确认。",
                    approval_request=SdkToolApprovalRequest(
                        approval_type="tool_call",
                        tool_name=inner_tool,
                        tool_input={"path": "runtime/tmp-cache.json", "force": True},
                        reason="删除文件属于高风险动作，必须用户确认。",
                    ),
                    raw_trace_ref="sdk-run-trace-reject",
                ).to_standard_agent_result()

        def fake_inner_delete(_session, *, path: str, force: bool = False):
            inner_calls.append({"path": path, "force": force})
            return {"tool_name": inner_tool, "ok": True, "result": {"deleted": path, "force": force}}

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(session)
            registry = AgentToolRegistry(dependencies.registry.list_definitions())
            registry.register(
                AgentToolDefinition(
                    name=outer_capability,
                    description="Delegate file analysis to the SDK sub-agent.",
                    input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "additionalProperties": False},
                    output_schema={"type": "object"},
                    handler=None,
                    allowed_source_types=frozenset({"agent_chat"}),
                )
            )
            registry.register(
                AgentToolDefinition(
                    name=inner_tool,
                    description="Delete a file requested by the SDK sub-agent after approval.",
                    input_schema={
                        "type": "object",
                        "required": ["path"],
                        "properties": {"path": {"type": "string"}, "force": {"type": "boolean"}},
                        "additionalProperties": False,
                    },
                    output_schema={"type": "object"},
                    handler=fake_inner_delete,
                    risk_level=AgentToolRiskLevel.HIGH,
                    requires_confirmation=True,
                    allowed_source_types=frozenset({"agent_chat"}),
                )
            )
            dependencies = dependencies.with_registry(registry).with_agent_runtime(
                executors={OPENAI_SDK_AGENT_EXECUTOR_ID: FakeOpenAISdkAgent()},
                capability_executor_ids={outer_capability: OPENAI_SDK_AGENT_EXECUTOR_ID},
            )

            first = run_agent_workflow(
                AgentRunCommand(
                    session_id=session_id,
                    user_message="分析缓存目录并清理无用临时文件",
                    requested_tool_name=outer_capability,
                    source_type="agent_chat",
                    tool_input={"path": "runtime"},
                ),
                dependencies=dependencies,
            )
            approval = session.scalars(select(ApprovalRequest)).one()
            continued = continue_agent_workflow_after_approval(
                approval.id,
                approved=False,
                decision_reason="not this cleanup",
                dependencies=dependencies,
            )
            session.commit()
            workflow = session.get(WorkflowRun, first.workflow_run_id)

        self.assertEqual(1, len(sdk_calls))
        self.assertEqual([], inner_calls)
        self.assertEqual(WorkflowRunStatus.CANCELED, workflow.status)
        self.assertEqual("approval_rejected", continued.state.current_step)
        self.assertIn(inner_tool, continued.state.final_response)
        self.assertIn("not this cleanup", continued.state.final_response)
        decision = continued.state.context_metadata["sdk_agent_approval_decision"]
        self.assertFalse(decision["approved"])
        self.assertEqual("not this cleanup", decision["decision_reason"])
        self.assertEqual(inner_tool, decision["requested_tool_name"])
        self.assertEqual(outer_capability, decision["outer_capability"])

    def test_approved_sdk_agent_run_state_resume_calls_executor_instead_of_inner_tool(self) -> None:
        from app.agent_runtime.agent_as_tool import OPENAI_SDK_AGENT_EXECUTOR_ID, AgentRuntimeContext, AgentTask, StandardAgentResult
        from app.agent_runtime.graph_factory import AgentRunCommand, continue_agent_workflow_after_approval, run_agent_workflow
        from app.agent_runtime.sdk_agents.schemas import SdkAgentResultEnvelope, SdkToolApprovalRequest
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry, AgentToolRiskLevel
        from app.domains.automation.models import ApprovalRequest, ToolCallLog, ToolCallStatus, WorkflowRun, WorkflowRunStatus

        outer_capability = "agent.file_analysis"
        inner_tool = "sdk.inner_delete"
        sdk_calls = []
        resume_calls = []
        inner_calls = []

        class FakeResumableOpenAISdkAgent:
            def call(self, task: AgentTask, context: AgentRuntimeContext):
                sdk_calls.append({"task": task, "context": context})
                return SdkAgentResultEnvelope(
                    task_id=context.task_id,
                    capability_id=task.capability_id,
                    subagent_name="FileAnalysisAgent",
                    status="needs_approval",
                    summary="子 Agent 想删除临时文件，等待用户确认。",
                    approval_request=SdkToolApprovalRequest(
                        approval_type="tool_call",
                        tool_name=inner_tool,
                        tool_input={"path": "runtime/tmp-cache.json", "force": True},
                        reason="删除文件属于高风险动作，必须用户确认。",
                    ),
                    raw_trace_ref="sdk-run-state-trace-approve",
                    metadata={
                        "run_state": {"format": "json", "value": {"cursor": "sdk-state-approve"}},
                        "task_envelope": {
                            "task_id": context.task_id,
                            "trace_id": context.run_id,
                            "capability_id": task.capability_id,
                            "subagent_name": "FileAnalysisAgent",
                            "goal": task.goal,
                            "input_payload": dict(task.input_payload),
                            "allowed_tools": [inner_tool],
                            "risk_policy": {"mutation_tools": "approval_required"},
                            "constraints": [],
                            "expected_output": [],
                            "max_turns": 5,
                            "context_refs": {"session_id": context.session_id, "workflow_run_id": context.run_id},
                            "metadata": {"agent_run_id": context.metadata.get("agent_run_id")},
                        },
                    },
                ).to_standard_agent_result()

            def resume_after_approval(self, approval_payload, *, approved: bool, context: AgentRuntimeContext) -> StandardAgentResult:
                resume_calls.append({"approval_payload": approval_payload, "approved": approved, "context": context})
                return StandardAgentResult(
                    status="succeeded",
                    summary="SDK 子 Agent 已恢复 RunState，并完成清理结果汇总。",
                    raw_result={
                        "tool_name": outer_capability,
                        "ok": True,
                        "result": {"resumed": True, "approved": approved, "path": "runtime/tmp-cache.json"},
                        "resource_effects": [
                            {
                                "resource_type": "file",
                                "action": "created",
                                "operation": "sdk_agent_cleanup_report",
                                "target_path": "artifact-sandbox://run-1-tool-1/output/cleanup-report.md",
                                "focus_path": "artifact-sandbox://run-1-tool-1/output/cleanup-report.md",
                                "aliases": ["刚才生成的清理报告", "子 Agent 清理报告"],
                            }
                        ],
                    },
                )

        def fake_inner_delete(_session, *, path: str, force: bool = False):
            inner_calls.append({"path": path, "force": force})
            raise AssertionError("RunState approval should resume the SDK runner instead of executing the inner tool here")

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(session)
            registry = AgentToolRegistry(dependencies.registry.list_definitions())
            registry.register(
                AgentToolDefinition(
                    name=outer_capability,
                    description="Delegate file analysis to the SDK sub-agent.",
                    input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "additionalProperties": False},
                    output_schema={"type": "object"},
                    handler=None,
                    allowed_source_types=frozenset({"agent_chat"}),
                )
            )
            registry.register(
                AgentToolDefinition(
                    name=inner_tool,
                    description="Delete a file requested by the SDK sub-agent after approval.",
                    input_schema={
                        "type": "object",
                        "required": ["path"],
                        "properties": {"path": {"type": "string"}, "force": {"type": "boolean"}},
                        "additionalProperties": False,
                    },
                    output_schema={"type": "object"},
                    handler=fake_inner_delete,
                    risk_level=AgentToolRiskLevel.HIGH,
                    requires_confirmation=True,
                    allowed_source_types=frozenset({"agent_chat"}),
                )
            )
            dependencies = dependencies.with_registry(registry).with_agent_runtime(
                executors={OPENAI_SDK_AGENT_EXECUTOR_ID: FakeResumableOpenAISdkAgent()},
                capability_executor_ids={outer_capability: OPENAI_SDK_AGENT_EXECUTOR_ID},
            )

            first = run_agent_workflow(
                AgentRunCommand(
                    session_id=session_id,
                    user_message="分析缓存目录并清理无用临时文件",
                    requested_tool_name=outer_capability,
                    source_type="agent_chat",
                    tool_input={"path": "runtime"},
                ),
                dependencies=dependencies,
            )
            approval = session.scalars(select(ApprovalRequest)).one()
            self.assertEqual({"format": "json", "value": {"cursor": "sdk-state-approve"}}, approval.payload["sdk_agent_result"]["metadata"]["run_state"])

            continued = continue_agent_workflow_after_approval(
                approval.id,
                approved=True,
                decision_reason="allow SDK resume only",
                dependencies=dependencies,
            )
            session.commit()
            workflow = session.get(WorkflowRun, first.workflow_run_id)
            tool_logs = list(session.scalars(select(ToolCallLog)).all())

        self.assertEqual(1, len(sdk_calls))
        self.assertEqual(1, len(resume_calls))
        self.assertTrue(resume_calls[0]["approved"])
        self.assertEqual("allow SDK resume only", resume_calls[0]["context"].metadata["decision_reason"])
        self.assertEqual([], inner_calls)
        self.assertEqual(WorkflowRunStatus.COMPLETED, workflow.status)
        self.assertEqual("final_response", continued.state.current_step)
        self.assertEqual("SDK 子 Agent 已恢复 RunState，并完成清理结果汇总。", continued.state.final_response)
        self.assertEqual([ToolCallStatus.BLOCKED, ToolCallStatus.SUCCEEDED], [log.status for log in tool_logs])
        self.assertEqual([outer_capability, outer_capability], [log.tool_name for log in tool_logs])
        self.assertEqual([log.id for log in tool_logs], continued.state.tool_call_ids)
        decision = continued.state.context_metadata["sdk_agent_approval_decision"]
        self.assertTrue(decision["approved"])
        self.assertEqual(inner_tool, decision["requested_tool_name"])
        self.assertEqual("allow SDK resume only", decision["decision_reason"])
        self.assertEqual("sdk_agent_cleanup_report", continued.state.context_metadata["active_resource"]["operation"])
        self.assertEqual(
            "artifact-sandbox://run-1-tool-1/output/cleanup-report.md",
            continued.state.context_metadata["active_resource"]["focus_path"],
        )
        self.assertNotIn("sdk_agent_approval", continued.state.context_metadata)

    def test_rejected_sdk_agent_run_state_resume_returns_subagent_alternative(self) -> None:
        from app.agent_runtime.agent_as_tool import OPENAI_SDK_AGENT_EXECUTOR_ID, AgentRuntimeContext, AgentTask, StandardAgentResult
        from app.agent_runtime.graph_factory import AgentRunCommand, continue_agent_workflow_after_approval, run_agent_workflow
        from app.agent_runtime.sdk_agents.schemas import SdkAgentResultEnvelope, SdkToolApprovalRequest
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry, AgentToolRiskLevel
        from app.domains.automation.models import ApprovalRequest, WorkflowRun, WorkflowRunStatus

        outer_capability = "agent.file_analysis"
        inner_tool = "sdk.inner_delete"
        resume_calls = []
        inner_calls = []

        class FakeResumableOpenAISdkAgent:
            def call(self, task: AgentTask, context: AgentRuntimeContext):
                return SdkAgentResultEnvelope(
                    task_id=context.task_id,
                    capability_id=task.capability_id,
                    subagent_name="FileAnalysisAgent",
                    status="needs_approval",
                    summary="子 Agent 想删除临时文件，等待用户确认。",
                    approval_request=SdkToolApprovalRequest(
                        approval_type="tool_call",
                        tool_name=inner_tool,
                        tool_input={"path": "runtime/tmp-cache.json", "force": True},
                        reason="删除文件属于高风险动作，必须用户确认。",
                    ),
                    metadata={"run_state": {"format": "json", "value": {"cursor": "sdk-state-reject"}}},
                ).to_standard_agent_result()

            def resume_after_approval(self, approval_payload, *, approved: bool, context: AgentRuntimeContext) -> StandardAgentResult:
                resume_calls.append({"approval_payload": approval_payload, "approved": approved, "context": context})
                return StandardAgentResult(
                    status="succeeded",
                    summary="用户拒绝删除后，SDK 子 Agent 改为提供手动清理建议。",
                    raw_result={"tool_name": outer_capability, "ok": True, "result": {"alternative": "manual_cleanup"}},
                )

        def fake_inner_delete(_session, *, path: str, force: bool = False):
            inner_calls.append({"path": path, "force": force})
            return {"tool_name": inner_tool, "ok": True, "result": {"deleted": path}}

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(session)
            registry = AgentToolRegistry(dependencies.registry.list_definitions())
            registry.register(
                AgentToolDefinition(
                    name=outer_capability,
                    description="Delegate file analysis to the SDK sub-agent.",
                    input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "additionalProperties": False},
                    output_schema={"type": "object"},
                    handler=None,
                    allowed_source_types=frozenset({"agent_chat"}),
                )
            )
            registry.register(
                AgentToolDefinition(
                    name=inner_tool,
                    description="Delete a file requested by the SDK sub-agent after approval.",
                    input_schema={"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}}, "additionalProperties": False},
                    output_schema={"type": "object"},
                    handler=fake_inner_delete,
                    risk_level=AgentToolRiskLevel.HIGH,
                    requires_confirmation=True,
                    allowed_source_types=frozenset({"agent_chat"}),
                )
            )
            dependencies = dependencies.with_registry(registry).with_agent_runtime(
                executors={OPENAI_SDK_AGENT_EXECUTOR_ID: FakeResumableOpenAISdkAgent()},
                capability_executor_ids={outer_capability: OPENAI_SDK_AGENT_EXECUTOR_ID},
            )

            first = run_agent_workflow(
                AgentRunCommand(
                    session_id=session_id,
                    user_message="分析缓存目录并清理无用临时文件",
                    requested_tool_name=outer_capability,
                    source_type="agent_chat",
                    tool_input={"path": "runtime"},
                ),
                dependencies=dependencies,
            )
            approval = session.scalars(select(ApprovalRequest)).one()
            continued = continue_agent_workflow_after_approval(
                approval.id,
                approved=False,
                decision_reason="do not delete files",
                dependencies=dependencies,
            )
            session.commit()
            workflow = session.get(WorkflowRun, first.workflow_run_id)

        self.assertEqual(1, len(resume_calls))
        self.assertFalse(resume_calls[0]["approved"])
        self.assertEqual("do not delete files", resume_calls[0]["context"].metadata["decision_reason"])
        self.assertEqual([], inner_calls)
        self.assertEqual(WorkflowRunStatus.COMPLETED, workflow.status)
        self.assertEqual("final_response", continued.state.current_step)
        self.assertIn("手动清理建议", continued.state.final_response)
        decision = continued.state.context_metadata["sdk_agent_approval_decision"]
        self.assertFalse(decision["approved"])
        self.assertEqual("do not delete files", decision["decision_reason"])


    def test_agent_runtime_tool_call_records_transient_retry_metadata_after_recovery(self) -> None:
        from app.agent_runtime.agent_as_tool import CLAUDE_SDK_AGENT_EXECUTOR_ID, AgentRuntimeContext, AgentTask, StandardAgentResult
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL, AgentToolDefinition, AgentToolRegistry
        from app.domains.automation.models import ToolCallLog, ToolCallStatus

        calls = []

        class FlakyClaudeSdkAgent:
            def call(self, task: AgentTask, context: AgentRuntimeContext) -> StandardAgentResult:
                calls.append({"task": task, "context": context})
                if len(calls) < 3:
                    return StandardAgentResult(
                        status="failed",
                        summary="SDK 限流，稍后重试",
                        raw_result={
                            "tool_name": EXTERNAL_WEB_SEARCH_TOOL,
                            "ok": False,
                            "error": "429 rate limit",
                            "error_type": "RateLimitError",
                        },
                    )
                return StandardAgentResult(
                    status="succeeded",
                    summary="腾讯校招官网：https://join.qq.com/",
                    raw_result={
                        "tool_name": EXTERNAL_WEB_SEARCH_TOOL,
                        "ok": True,
                        "result": {"answer": "腾讯校招官网：https://join.qq.com/"},
                    },
                )

        def legacy_handler(_session, **_arguments):
            raise AssertionError("legacy handler should not run when claude-sdk-agent is registered for web search")

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(session)
            registry = AgentToolRegistry(
                definition
                for definition in dependencies.registry.list_definitions()
                if definition.name != EXTERNAL_WEB_SEARCH_TOOL
            )
            registry.register(
                AgentToolDefinition(
                    name=EXTERNAL_WEB_SEARCH_TOOL,
                    description="Search the public web through an external agent.",
                    input_schema={"type": "object", "required": ["query"], "properties": {"query": {"type": "string"}}},
                    output_schema={"type": "object", "required": ["tool_name", "ok", "result"]},
                    handler=legacy_handler,
                    allowed_source_types=frozenset({"agent_chat", "web_search"}),
                )
            )
            dependencies = dependencies.with_registry(registry).with_agent_runtime(
                executors={CLAUDE_SDK_AGENT_EXECUTOR_ID: FlakyClaudeSdkAgent()},
                capability_executor_ids={EXTERNAL_WEB_SEARCH_TOOL: CLAUDE_SDK_AGENT_EXECUTOR_ID},
            )

            result = run_agent_workflow(
                AgentRunCommand(
                    session_id=session_id,
                    user_message="搜一下腾讯校招官网",
                    requested_tool_name=EXTERNAL_WEB_SEARCH_TOOL,
                    source_type="agent_chat",
                    tool_input={"query": "腾讯 校园招聘 官网"},
                ),
                dependencies=dependencies,
            )
            session.commit()
            tool_log = session.scalars(select(ToolCallLog)).one()

        retry = tool_log.output_payload["result"]["runtime_retry"]
        self.assertEqual(ToolCallStatus.SUCCEEDED, tool_log.status)
        self.assertEqual(3, len(calls))
        self.assertTrue(retry["recovered"])
        self.assertEqual(3, retry["attempts"])
        self.assertEqual("RateLimitError", retry["errors"][0]["error_type"])
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

    def test_tool_choice_loop_enters_xiaohongshu_search_from_declared_content_tool(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.tool_registry import AgentToolCandidateProfile, AgentToolDefinition, AgentToolRegistry
        from app.agent_runtime.understanding.schemas import IntentFrame
        from app.domains.agent_memory.schemas import AgentSkillCreate
        from app.domains.automation.models import ToolCallLog, ToolCallStatus
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall
        from app.mcp_gateway.client import MCPToolCallResult

        calls = []

        class FakeNormalIntentDetector:
            def detect(self, _message):
                return IntentFrame(intent="normal_chat", confidence=0.0)

        class LegacyRoutingShouldNotRun:
            def decide(self, *, user_message, intent_frame, context_pack):  # pragma: no cover - failing path
                raise AssertionError("legacy capability routing should not run before tool choice loop")

        test_case = self

        class FakeXiaohongshuSearchLLM:
            def __init__(self) -> None:
                self.calls = []

            def complete(self, *, messages, tools=None, tool_choice=None):
                self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
                combined = "\n".join(str(message.get("content") or "") for message in messages)
                if tools and "小红书秋招笔记" not in combined:
                    test_case.assertIn(
                        "xiaohongshu_mcp_search_feeds",
                        [tool["function"]["name"] for tool in tools],
                    )
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-xhs-search",
                                name="xiaohongshu_mcp_search_feeds",
                                arguments={"keyword": "2027 秋招 Java 岗位"},
                            )
                        ],
                    )
                return LLMChatCompletion(content="已从小红书找到 2027 秋招 Java 岗位相关笔记。")

        def fake_search(_session, *, keyword: str, filters=None) -> MCPToolCallResult:
            calls.append({"keyword": keyword, "filters": filters})
            return MCPToolCallResult(
                tool_name="xiaohongshu-mcp.search_feeds",
                ok=True,
                result={"items": [{"title": "小红书秋招笔记"}]},
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
                    metadata_json={
                        "allowed_tools": ["xiaohongshu-mcp.search_feeds"],
                        "source_types": ["agent_chat", "xiaohongshu_note"],
                    },
                    sections={"workflow": "Search Xiaohongshu recruiting notes."},
                )
            )
            fake_llm = FakeXiaohongshuSearchLLM()
            dependencies = self._dependencies(
                session,
                skill_repository=skill_repository,
                llm_client=fake_llm,
                intent_detector=FakeNormalIntentDetector(),
                capability_routing_middleware=LegacyRoutingShouldNotRun(),
            )
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
                    candidate_profile=AgentToolCandidateProfile(
                        categories=frozenset({"xiaohongshu_content_search", "content_source_search"}),
                        keywords=frozenset({"小红书", "红书", "搜索笔记"}),
                        examples=("请在小红书搜索 2027 秋招 Java 岗位",),
                    ),
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
        self.assertEqual("agent_chat", result.state.source_type)
        self.assertEqual("llm_tool_choice_loop", result.state.response_mode)
        self.assertEqual("已从小红书找到 2027 秋招 Java 岗位相关笔记。", result.state.final_response)
        self.assertIn(
            "xiaohongshu-mcp.search_feeds",
            result.state.context_metadata["tool_candidate_selection"]["capabilities"],
        )
        self.assertNotIn("capability_routing", result.state.context_metadata)
        self.assertEqual([{"keyword": "2027 秋招 Java 岗位", "filters": None}], calls)
        self.assertEqual(ToolCallStatus.SUCCEEDED, tool_log.status)
        self.assertEqual("xiaohongshu-mcp.search_feeds", tool_log.tool_name)
        self.assertEqual(2, len(fake_llm.calls))

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
                    requested_tool_name="xiaohongshu-mcp.search_feeds",
                    source_type="xiaohongshu_note",
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

    def test_agent_does_not_auto_select_offerio_company_jobs_sync_from_update_request(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry
        from app.domains.automation.models import ApprovalRequest, ToolCallLog

        calls = []

        def fake_sync(_session, *, limit: int = 50, source_id: str | None = None):
            calls.append({"limit": limit, "source_id": source_id})
            return {
                "tool_name": "offerio.sync_company_jobs",
                "ok": True,
                "result": {
                    "source_name": "OfferIO 公司聚合岗位库",
                    "status": "succeeded",
                    "fetched_count": 1,
                    "extracted_count": 1,
                    "failed_count": 0,
                },
            }

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(session)
            registry = AgentToolRegistry(
                definition
                for definition in dependencies.registry.list_definitions()
                if definition.name != "offerio.sync_company_jobs"
            )
            registry.register(
                AgentToolDefinition(
                    name="offerio.sync_company_jobs",
                    description="Sync OfferIO company aggregated campus recruiting jobs into job leads.",
                    input_schema={"type": "object", "properties": {"limit": {"type": "integer"}}},
                    output_schema={"type": "object", "required": ["tool_name", "ok"]},
                    handler=fake_sync,
                    allowed_source_types=frozenset({"agent_chat", "official_api", "job_discovery"}),
                )
            )

            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="请从 OfferIO 公司聚合岗位库更新一下岗位"),
                dependencies=dependencies.with_registry(registry),
            )
            session.commit()

            tool_logs = list(session.scalars(select(ToolCallLog)).all())

        self.assertIsNone(result.state.requested_tool_name)
        self.assertEqual("agent_chat", result.state.source_type)
        self.assertEqual([], calls)
        self.assertEqual([], tool_logs)
        self.assertEqual("deterministic_stub", result.state.response_mode)

    def test_agent_does_not_auto_select_external_web_search_for_recent_search_request(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL, AgentToolDefinition, AgentToolRegistry
        from app.domains.automation.models import ToolCallLog

        calls = []

        def fake_web_search(_session, *, query: str, max_results: int = 5):
            calls.append({"query": query, "max_results": max_results})
            return {
                "tool_name": EXTERNAL_WEB_SEARCH_TOOL,
                "ok": True,
                "result": {
                    "executor_name": "claude-sdk-agent",
                    "query": query,
                    "answer": "联网搜索结果：梅西最近一场比赛信息。",
                    "sources": [{"title": "MLS", "url": "https://example.com/messi"}],
                },
            }

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(session)
            registry = AgentToolRegistry(
                definition
                for definition in dependencies.registry.list_definitions()
                if definition.name != EXTERNAL_WEB_SEARCH_TOOL
            )
            registry.register(
                AgentToolDefinition(
                    name=EXTERNAL_WEB_SEARCH_TOOL,
                    description="Search the public web through an external agent.",
                    input_schema={"type": "object", "required": ["query"]},
                    output_schema={"type": "object", "required": ["tool_name", "ok", "result"]},
                    handler=fake_web_search,
                    allowed_source_types=frozenset({"agent_chat", "web_search"}),
                )
            )

            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="你给我搜索一下梅西最近的比赛"),
                dependencies=dependencies.with_registry(registry),
            )
            session.commit()

            tool_logs = list(session.scalars(select(ToolCallLog)).all())

        self.assertIsNone(result.state.requested_tool_name)
        self.assertEqual("agent_chat", result.state.source_type)
        self.assertEqual([], tool_logs)
        self.assertEqual([], calls)
        self.assertEqual("deterministic_stub", result.state.response_mode)

    def test_agent_context_engineering_records_campus_search_intent_without_running_tool(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.understanding.intent_detector import HybridIntentDetector
        from app.domains.automation.models import ToolCallLog
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        class FakeIntentLLM:
            def complete(self, *, messages):
                return LLMChatCompletion(
                    content=(
                        '{"intent":"campus_recruiting_search","confidence":0.95,'
                        '"needs_external_info":true,"risk_level":"low",'
                        '"entities":{"company_names":["中科曙光"],"keywords":["校园招聘"],"time_range":"latest"}}'
                    )
                )

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(
                session,
                intent_detector=HybridIntentDetector(llm_client=FakeIntentLLM()),
            )

            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="你给我搜一下中科曙光的校园招聘信息"),
                dependencies=dependencies,
            )
            session.commit()
            tool_logs = list(session.scalars(select(ToolCallLog)).all())

        context_pack = result.state.context_metadata["context_pack"]
        intent_frame = result.state.context_metadata["intent_frame"]
        self.assertEqual("normal_chat", intent_frame["intent"])
        self.assertEqual([], intent_frame["entities"]["company_names"])
        self.assertEqual("model_driven", context_pack["capability_selection_mode"])
        self.assertIn("external.web_search", context_pack["allowed_capabilities"])
        self.assertIn("offerio.sync_company_jobs", context_pack["allowed_capabilities"])
        self.assertNotIn("offerio.sync_company_jobs", context_pack["excluded_capabilities"])
        self.assertIsNone(result.state.requested_tool_name)
        self.assertEqual([], result.state.tool_call_ids)
        self.assertEqual([], tool_logs)

    def test_capability_routing_middleware_keeps_normal_chat_out_of_execution_planner(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.routing.capability_routing_middleware import CapabilityRoutingMiddleware
        from app.infrastructure.llm.chat_client import LLMChatCompletion

        class PlannerShouldNotRun:
            def plan(self, *, user_message, context_pack):  # pragma: no cover - failing path
                raise AssertionError("planner should not run for chat_direct route")

        class FakeLLMClient:
            def __init__(self) -> None:
                self.calls = []

            def complete(self, *, messages, tools=None, tool_choice=None):
                self.calls.append(messages)
                return LLMChatCompletion(content="Planner Gate 是规划器启用门控。")

        with self.Session() as session:
            session_id = self._session_id(session)
            llm_client = FakeLLMClient()
            dependencies = self._dependencies(
                session,
                llm_client=llm_client,
                execution_planner=PlannerShouldNotRun(),
                capability_routing_middleware=CapabilityRoutingMiddleware(),
            )

            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="Planner Gate 是什么？"),
                dependencies=dependencies,
            )
            session.commit()

        self.assertNotIn("capability_routing", result.state.context_metadata)
        self.assertNotIn("execution_plan", result.state.context_metadata)
        self.assertEqual("llm_tool_choice_loop", result.state.response_mode)
        self.assertEqual("Planner Gate 是规划器启用门控。", result.state.final_response)
        self.assertEqual(1, len(llm_client.calls))

    def test_capability_routing_middleware_dispatches_campus_search_to_external_agent_tool(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.routing.capability_routing_middleware import CapabilityRoutingMiddleware
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL, AgentToolDefinition, AgentToolRegistry
        from app.agent_runtime.understanding.intent_detector import HybridIntentDetector
        from app.domains.automation.models import ToolCallLog, ToolCallStatus
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        class FakeIntentLLM:
            def complete(self, *, messages):
                return LLMChatCompletion(
                    content=(
                        '{"intent":"campus_recruiting_search","confidence":0.95,'
                        '"needs_external_info":true,"risk_level":"low",'
                        '"entities":{"company_names":["中科曙光"],"keywords":["校园招聘"],"time_range":"latest"}}'
                    )
                )

        search_calls = []

        class FakeMainLLM:
            def __init__(self) -> None:
                self.calls = []

            def complete(self, *, messages, tools=None, tool_choice=None):
                self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
                if tools is not None and not any(message.get("role") == "tool" for message in messages):
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-campus-search",
                                name="external_web_search",
                                arguments={"query": "中科曙光 校园招聘 官网", "max_results": 5},
                            )
                        ],
                    )
                return LLMChatCompletion(content="已完成中科曙光校园招聘信息搜索，官网结果已返回。")

        def fake_web_search(_session, *, query: str, max_results: int = 5):
            search_calls.append({"query": query, "max_results": max_results})
            return {
                "tool_name": EXTERNAL_WEB_SEARCH_TOOL,
                "ok": True,
                "result": {
                    "executor_name": "claude-sdk-agent",
                    "answer": "联网搜索结果：中科曙光校园招聘官网：https://jobs.example.com/sugon",
                    "sources": [{"title": "中科曙光校园招聘官网", "url": "https://jobs.example.com/sugon"}],
                },
            }

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(
                session,
                llm_client=FakeMainLLM(),
                intent_detector=HybridIntentDetector(llm_client=FakeIntentLLM()),
                capability_routing_middleware=CapabilityRoutingMiddleware(),
            )
            registry = AgentToolRegistry(
                definition
                for definition in dependencies.registry.list_definitions()
                if definition.name != EXTERNAL_WEB_SEARCH_TOOL
            )
            registry.register(
                AgentToolDefinition(
                    name=EXTERNAL_WEB_SEARCH_TOOL,
                    description="Search the public web through an external agent.",
                    input_schema={
                        "type": "object",
                        "required": ["query"],
                        "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
                    },
                    output_schema={"type": "object", "required": ["tool_name", "ok", "result"]},
                    handler=fake_web_search,
                    allowed_source_types=frozenset({"agent_chat", "web_search"}),
                )
            )

            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="你给我搜一下中科曙光的校园招聘信息"),
                dependencies=dependencies.with_registry(registry),
            )
            session.commit()
            tool_log = session.scalars(select(ToolCallLog)).one()

        self.assertNotIn("capability_routing", result.state.context_metadata)
        self.assertIn(EXTERNAL_WEB_SEARCH_TOOL, result.state.context_metadata["tool_candidate_selection"]["capabilities"])
        self.assertEqual([{"query": "中科曙光 校园招聘 官网", "max_results": 5}], search_calls)
        self.assertEqual(EXTERNAL_WEB_SEARCH_TOOL, result.state.requested_tool_name)
        self.assertEqual("llm_tool_choice_loop", result.state.response_mode)
        self.assertIn("校园招聘信息", result.state.final_response)
        self.assertEqual(ToolCallStatus.SUCCEEDED, tool_log.status)
        envelope = tool_log.output_payload["result"]["result_envelope"]
        self.assertEqual("succeeded", envelope["status"])
        self.assertEqual(EXTERNAL_WEB_SEARCH_TOOL, envelope["capability"])
        self.assertEqual("claude-sdk-agent", envelope["executor"])
        self.assertIn("中科曙光校园招聘官网", envelope["summary"])
        self.assertEqual("low", envelope["risk_level"])

    def test_external_search_result_is_synthesized_by_main_llm_before_final_answer(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.routing.capability_routing_middleware import CapabilityRoutingMiddleware
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL, AgentToolDefinition, AgentToolRegistry
        from app.agent_runtime.understanding.intent_detector import HybridIntentDetector
        from app.domains.automation.models import ToolCallLog, ToolCallStatus
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

        class FakeMainLLM:
            def __init__(self) -> None:
                self.calls = []

            def complete(self, *, messages, tools=None, tool_choice=None):
                self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
                combined = "\n".join(str(message.get("content") or "") for message in messages)
                if tools is not None and not any(message.get("role") == "tool" for message in messages):
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-gongniu-search",
                                name="external_web_search",
                                arguments={"query": "公牛集团 校园招聘 官网", "max_results": 5},
                            )
                        ],
                    )
                # The model-driven loop receives the tool observation directly
                # and synthesizes from it in the next turn. Do not couple this
                # test to the legacy post-loop synthesis prompt shape.
                test_case.assertTrue(any(message.get("role") == "tool" for message in messages))
                test_case.assertIn("芝加哥公牛队", combined)
                test_case.assertIn("公牛集团校园招聘", combined)
                test_case.assertIn("https://campus.gongniu.cn/", combined)
                return LLMChatCompletion(content="公牛集团校招入口：https://campus.gongniu.cn/。")

        search_calls = []

        def fake_web_search(_session, *, query: str, max_results: int = 5):
            search_calls.append({"query": query, "max_results": max_results})
            return {
                "tool_name": EXTERNAL_WEB_SEARCH_TOOL,
                "ok": True,
                "result": {
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
                },
            }

        with self.Session() as session:
            session_id = self._session_id(session)
            fake_llm = FakeMainLLM()
            dependencies = self._dependencies(
                session,
                llm_client=fake_llm,
                intent_detector=HybridIntentDetector(llm_client=FakeIntentLLM()),
                capability_routing_middleware=CapabilityRoutingMiddleware(),
            )
            registry = AgentToolRegistry(
                definition
                for definition in dependencies.registry.list_definitions()
                if definition.name != EXTERNAL_WEB_SEARCH_TOOL
            )
            registry.register(
                AgentToolDefinition(
                    name=EXTERNAL_WEB_SEARCH_TOOL,
                    description="Search the public web through an external agent.",
                    input_schema={"type": "object", "required": ["query"], "properties": {"query": {"type": "string"}}},
                    output_schema={"type": "object", "required": ["tool_name", "ok", "result"]},
                    handler=fake_web_search,
                    allowed_source_types=frozenset({"agent_chat", "web_search"}),
                )
            )

            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="你搜索一下公牛的校园招聘"),
                dependencies=dependencies.with_registry(registry),
            )
            session.commit()
            tool_logs = list(session.scalars(select(ToolCallLog)).all())

        self.assertEqual("llm_tool_choice_loop", result.state.response_mode)
        self.assertEqual("公牛集团校招入口：https://campus.gongniu.cn/。", result.state.final_response)
        self.assertNotIn("NBA", result.state.final_response)
        self.assertNotIn("芝加哥", result.state.final_response)
        self.assertNotIn("过滤", result.state.final_response)
        self.assertGreaterEqual(len(fake_llm.calls), 1)
        self.assertTrue(tool_logs)
        self.assertTrue(all(log.status == ToolCallStatus.SUCCEEDED for log in tool_logs))

    def test_external_search_synthesis_does_not_expose_irrelevant_noisy_results(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.routing.capability_routing_middleware import CapabilityRoutingMiddleware
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL, AgentToolDefinition, AgentToolRegistry
        from app.agent_runtime.understanding.intent_detector import HybridIntentDetector
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        class FakeIntentLLM:
            def complete(self, *, messages):
                return LLMChatCompletion(
                    content=(
                        '{"intent":"campus_recruiting_search","confidence":0.95,'
                        '"needs_external_info":true,"risk_level":"low",'
                        '"entities":{"company_names":["公牛集团"],"keywords":["2026年校园秋招"],"time_range":"2026"}}'
                    )
                )

        test_case = self

        class FakeMainLLM:
            def complete(self, *, messages, tools=None, tool_choice=None):
                combined = "\n".join(str(message.get("content") or "") for message in messages)
                if tools is not None and not any(message.get("role") == "tool" for message in messages):
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-gongniu-noisy-search",
                                name="external_web_search",
                                arguments={"query": "公牛集团 2026 校园秋招", "max_results": 5},
                            )
                        ],
                    )
                # The model-driven loop receives noisy observations directly;
                # the test verifies the model can synthesize a safe answer
                # without depending on the old post-loop prompt.
                test_case.assertTrue(any(message.get("role") == "tool" for message in messages))
                test_case.assertIn("汉字", combined)
                test_case.assertIn("我的世界", combined)
                return LLMChatCompletion(content="我没有找到公牛集团 2026 年校园秋招的可靠招聘入口或官方公告。建议继续关注公牛集团官网招聘页、官方招聘公众号和高校就业网。")

        search_calls = []

        def fake_web_search(_session, *, query: str, max_results: int = 5):
            search_calls.append({"query": query, "max_results": max_results})
            return {
                "tool_name": EXTERNAL_WEB_SEARCH_TOOL,
                "ok": True,
                "result": {
                    "executor_name": "claude-sdk-agent",
                    "answer": (
                        "联网搜索结果：\n"
                        "- 我_百度百科：汉字“我”的解释\n"
                        "- 张国荣歌曲《我》_百度百科\n"
                        "- Minecraft 我的世界 官网"
                    ),
                    "sources": [
                        {"title": "我_百度百科", "url": "https://baike.baidu.com/item/%E6%88%91"},
                        {"title": "我的世界官网", "url": "https://www.minecraft.net/"},
                    ],
                },
            }

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(
                session,
                llm_client=FakeMainLLM(),
                intent_detector=HybridIntentDetector(llm_client=FakeIntentLLM()),
                capability_routing_middleware=CapabilityRoutingMiddleware(),
            )
            registry = AgentToolRegistry(
                definition
                for definition in dependencies.registry.list_definitions()
                if definition.name != EXTERNAL_WEB_SEARCH_TOOL
            )
            registry.register(
                AgentToolDefinition(
                    name=EXTERNAL_WEB_SEARCH_TOOL,
                    description="Search the public web through an external agent.",
                    input_schema={"type": "object", "required": ["query"], "properties": {"query": {"type": "string"}}},
                    output_schema={"type": "object", "required": ["tool_name", "ok", "result"]},
                    handler=fake_web_search,
                    allowed_source_types=frozenset({"agent_chat", "web_search"}),
                )
            )

            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="我要的是2026年的校园秋招信息"),
                dependencies=dependencies.with_registry(registry),
            )
            session.commit()

        self.assertEqual("tool_result_summary_unreliable", result.state.response_mode)
        self.assertNotIn("汉字", result.state.final_response)
        self.assertNotIn("张国荣", result.state.final_response)
        self.assertNotIn("我的世界", result.state.final_response)
        self.assertNotIn("这说明", result.state.final_response)
        self.assertNotIn("尚未发布", result.state.final_response)

    def test_capability_routing_middleware_dispatches_offerio_sync_to_local_workflow_tool(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.routing.capability_routing_middleware import CapabilityRoutingMiddleware
        from app.agent_runtime.tool_registry import OFFERIO_COMPANY_JOBS_TOOL, AgentToolDefinition, AgentToolRegistry
        from app.domains.automation.models import ToolCallLog, ToolCallStatus

        calls = []

        def fake_sync(_session, *, limit: int = 50, source_id: str | None = None):
            calls.append({"limit": limit, "source_id": source_id})
            return {
                "tool_name": OFFERIO_COMPANY_JOBS_TOOL,
                "ok": True,
                "result": {
                    "source_name": "OfferIO 公司聚合岗位库",
                    "status": "succeeded",
                    "fetched_count": 50,
                    "extracted_count": 50,
                    "failed_count": 0,
                },
            }

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(session, capability_routing_middleware=CapabilityRoutingMiddleware())
            registry = AgentToolRegistry(
                definition
                for definition in dependencies.registry.list_definitions()
                if definition.name != OFFERIO_COMPANY_JOBS_TOOL
            )
            registry.register(
                AgentToolDefinition(
                    name=OFFERIO_COMPANY_JOBS_TOOL,
                    description="Sync OfferIO company aggregated campus recruiting jobs into job leads.",
                    input_schema={"type": "object", "properties": {"limit": {"type": "integer"}}},
                    output_schema={"type": "object", "required": ["tool_name", "ok"]},
                    handler=fake_sync,
                    allowed_source_types=frozenset({"agent_chat", "official_api", "job_discovery"}),
                )
            )

            result = run_agent_workflow(
                AgentRunCommand(
                    session_id=session_id,
                    user_message="请从 OfferIO 公司聚合岗位库更新一下岗位",
                    requested_tool_name=OFFERIO_COMPANY_JOBS_TOOL,
                    tool_input={"limit": 1000},
                ),
                dependencies=dependencies.with_registry(registry),
            )
            session.commit()
            tool_log = session.scalars(select(ToolCallLog)).one()

        self.assertNotIn("capability_routing", result.state.context_metadata)
        self.assertEqual([{"limit": 1000, "source_id": None}], calls)
        self.assertEqual(OFFERIO_COMPANY_JOBS_TOOL, result.state.requested_tool_name)
        self.assertEqual("tool_result_summary", result.state.response_mode)
        self.assertIn("已从 OfferIO 公司聚合岗位库同步岗位", result.state.final_response)
        self.assertEqual(ToolCallStatus.SUCCEEDED, tool_log.status)

    def test_runtime_guard_blocks_routed_capability_outside_context_pack(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.routing.schemas import RouteDecision
        from app.agent_runtime.tool_registry import OFFERIO_COMPANY_JOBS_TOOL, AgentToolDefinition, AgentToolRegistry
        from app.agent_runtime.understanding.intent_detector import HybridIntentDetector
        from app.domains.automation.models import ToolCallLog
        from app.infrastructure.llm.chat_client import LLMChatCompletion

        class FakeIntentLLM:
            def complete(self, *, messages):
                return LLMChatCompletion(
                    content=(
                        '{"intent":"campus_recruiting_search","confidence":0.95,'
                        '"needs_external_info":true,"risk_level":"low",'
                        '"entities":{"company_names":["中科曙光"],"keywords":["校园招聘"],"time_range":"latest"}}'
                    )
                )

        class BadRoutingMiddleware:
            def decide(self, *, user_message, intent_frame, context_pack):
                return RouteDecision(
                    route="local_workflow",
                    capability=OFFERIO_COMPANY_JOBS_TOOL,
                    executor_type="local_workflow",
                    executor_name="offerio_company_jobs_sync",
                    reason="bad route should be blocked",
                    allowed_capabilities=list(context_pack.get("allowed_capabilities") or []),
                    tool_input={"limit": 1000},
                )

        calls = []

        def fake_sync(_session, *, limit: int = 50, source_id: str | None = None):
            calls.append({"limit": limit, "source_id": source_id})
            return {"tool_name": OFFERIO_COMPANY_JOBS_TOOL, "ok": True, "result": {}}

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(
                session,
                intent_detector=HybridIntentDetector(llm_client=FakeIntentLLM()),
                capability_routing_middleware=BadRoutingMiddleware(),
            )
            registry = AgentToolRegistry(
                definition
                for definition in dependencies.registry.list_definitions()
                if definition.name != OFFERIO_COMPANY_JOBS_TOOL
            )
            registry.register(
                AgentToolDefinition(
                    name=OFFERIO_COMPANY_JOBS_TOOL,
                    description="Sync OfferIO company aggregated campus recruiting jobs into job leads.",
                    input_schema={"type": "object", "properties": {"limit": {"type": "integer"}}},
                    output_schema={"type": "object", "required": ["tool_name", "ok"]},
                    handler=fake_sync,
                    allowed_source_types=frozenset({"agent_chat", "official_api", "job_discovery"}),
                )
            )

            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="你给我搜一下中科曙光的校园招聘信息"),
                dependencies=dependencies.with_registry(registry),
            )
            session.commit()
            tool_logs = list(session.scalars(select(ToolCallLog)).all())

        self.assertEqual([], calls)
        self.assertEqual([], tool_logs)
        # Ordinary agent chat no longer executes the legacy capability router;
        # the model receives the complete registered catalog and runtime owns
        # validation only after the model submits a structured call.
        self.assertEqual("deterministic_stub", result.state.response_mode)
        self.assertNotIn("capability_routing_guard", result.state.context_metadata)

    def test_runtime_guard_blocks_routed_capability_with_invalid_tool_input(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.routing.schemas import RouteDecision
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL, AgentToolDefinition, AgentToolRegistry
        from app.agent_runtime.understanding.intent_detector import HybridIntentDetector
        from app.domains.automation.models import ToolCallLog
        from app.infrastructure.llm.chat_client import LLMChatCompletion

        class FakeIntentLLM:
            def complete(self, *, messages):
                return LLMChatCompletion(
                    content=(
                        '{"intent":"campus_recruiting_search","confidence":0.95,'
                        '"needs_external_info":true,"risk_level":"low",'
                        '"entities":{"company_names":["中科曙光"],"keywords":["校园招聘"],"time_range":"latest"}}'
                    )
                )

        class BadInputRoutingMiddleware:
            def decide(self, *, user_message, intent_frame, context_pack):
                return RouteDecision(
                    route="external_agent",
                    capability=EXTERNAL_WEB_SEARCH_TOOL,
                    executor_type="external_agent",
                    executor_name="claude_sdk_agent",
                    reason="bad input should be blocked",
                    allowed_capabilities=list(context_pack.get("allowed_capabilities") or []),
                    tool_input={"max_results": 5},
                )

        calls = []

        def fake_web_search(_session, *, query: str, max_results: int = 5):
            calls.append({"query": query, "max_results": max_results})
            return {"tool_name": EXTERNAL_WEB_SEARCH_TOOL, "ok": True, "result": {}}

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(
                session,
                intent_detector=HybridIntentDetector(llm_client=FakeIntentLLM()),
                capability_routing_middleware=BadInputRoutingMiddleware(),
            )
            registry = AgentToolRegistry(
                definition
                for definition in dependencies.registry.list_definitions()
                if definition.name != EXTERNAL_WEB_SEARCH_TOOL
            )
            registry.register(
                AgentToolDefinition(
                    name=EXTERNAL_WEB_SEARCH_TOOL,
                    description="Search the public web through an external agent.",
                    input_schema={
                        "type": "object",
                        "required": ["query"],
                        "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
                        "additionalProperties": False,
                    },
                    output_schema={"type": "object", "required": ["tool_name", "ok", "result"]},
                    handler=fake_web_search,
                    allowed_source_types=frozenset({"agent_chat", "web_search"}),
                )
            )

            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="你给我搜一下中科曙光的校园招聘信息"),
                dependencies=dependencies.with_registry(registry),
            )
            session.commit()
            tool_logs = list(session.scalars(select(ToolCallLog)).all())

        self.assertEqual([], calls)
        self.assertEqual([], tool_logs)
        self.assertEqual("deterministic_stub", result.state.response_mode)
        self.assertNotIn("capability_routing_guard", result.state.context_metadata)

    def test_capability_routing_high_risk_route_asks_user_without_running_tools_or_llm(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.routing.capability_routing_middleware import CapabilityRoutingMiddleware
        from app.agent_runtime.understanding.intent_detector import HybridIntentDetector
        from app.domains.automation.models import ToolCallLog
        from app.infrastructure.llm.chat_client import LLMChatCompletion

        class FakeIntentLLM:
            def complete(self, *, messages):
                return LLMChatCompletion(
                    content=(
                        '{"intent":"application_entry_discovery","confidence":0.95,'
                        '"needs_external_info":true,"risk_level":"high",'
                        '"entities":{"job_ids":["lead-risk-1"]}}'
                    )
                )

        class LLMShouldNotRun:
            def complete(self, *, messages, tools=None, tool_choice=None):
                return LLMChatCompletion(content="提交简历属于高风险操作，需要你确认后才能继续。")

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(
                session,
                llm_client=LLMShouldNotRun(),
                intent_detector=HybridIntentDetector(llm_client=FakeIntentLLM()),
                capability_routing_middleware=CapabilityRoutingMiddleware(),
            )

            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="帮我上传简历并提交这个岗位 job_id=lead-risk-1"),
                dependencies=dependencies,
            )
            session.commit()
            tool_logs = list(session.scalars(select(ToolCallLog)).all())

        self.assertNotIn("capability_routing", result.state.context_metadata)
        self.assertEqual([], tool_logs)
        self.assertEqual("llm_tool_choice_loop", result.state.response_mode)
        self.assertIn("需要你确认", result.state.final_response)

    def test_ambiguous_search_route_asks_user_before_running_external_search(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.routing.capability_routing_middleware import CapabilityRoutingMiddleware
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL, AgentToolDefinition, AgentToolRegistry
        from app.agent_runtime.understanding.intent_detector import HybridIntentDetector
        from app.domains.automation.models import ToolCallLog
        from app.infrastructure.llm.chat_client import LLMChatCompletion

        class FakeIntentLLM:
            def complete(self, *, messages):
                return LLMChatCompletion(
                    content=(
                        '{"intent":"campus_recruiting_search","confidence":0.62,'
                        '"needs_external_info":true,"risk_level":"low",'
                        '"candidate_intents":["campus_recruiting_search","external_agent_task"],'
                        '"entities":{"company_names":["公牛"],"keywords":[],"time_range":"latest"}}'
                    )
                )

        class LLMShouldNotRun:
            def complete(self, *, messages, tools=None, tool_choice=None):
                return LLMChatCompletion(content="请明确你要查询公牛集团还是芝加哥公牛队。")

        calls = []

        def fake_web_search(_session, *, query: str, max_results: int = 5):
            calls.append({"query": query, "max_results": max_results})
            return {"tool_name": EXTERNAL_WEB_SEARCH_TOOL, "ok": True, "result": {}}

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(
                session,
                llm_client=LLMShouldNotRun(),
                intent_detector=HybridIntentDetector(llm_client=FakeIntentLLM()),
                capability_routing_middleware=CapabilityRoutingMiddleware(),
            )
            registry = AgentToolRegistry(
                definition
                for definition in dependencies.registry.list_definitions()
                if definition.name != EXTERNAL_WEB_SEARCH_TOOL
            )
            registry.register(
                AgentToolDefinition(
                    name=EXTERNAL_WEB_SEARCH_TOOL,
                    description="Search the public web through an external agent.",
                    input_schema={"type": "object", "required": ["query"]},
                    output_schema={"type": "object", "required": ["tool_name", "ok", "result"]},
                    handler=fake_web_search,
                    allowed_source_types=frozenset({"agent_chat", "web_search"}),
                )
            )

            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="帮我搜一下公牛"),
                dependencies=dependencies.with_registry(registry),
            )
            session.commit()
            tool_logs = list(session.scalars(select(ToolCallLog)).all())

        self.assertNotIn("capability_routing", result.state.context_metadata)
        self.assertEqual([], calls)
        self.assertEqual([], tool_logs)
        self.assertEqual("llm_tool_choice_loop", result.state.response_mode)
        self.assertIn("公牛集团", result.state.final_response)
        self.assertIn("芝加哥公牛队", result.state.final_response)

    def test_native_tool_loop_executes_allowed_web_search_once_and_finalizes_with_observation(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL, AgentToolDefinition, AgentToolRegistry
        from app.agent_runtime.understanding.intent_detector import HybridIntentDetector
        from app.domains.automation.models import ToolCallLog, ToolCallStatus
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        class FakeIntentLLM:
            def complete(self, *, messages):
                return LLMChatCompletion(
                    content=(
                        '{"intent":"campus_recruiting_search","confidence":0.95,'
                        '"needs_external_info":true,"risk_level":"low",'
                        '"entities":{"company_names":["中科曙光"],"keywords":["校园招聘"],"time_range":"latest"}}'
                    )
                )

        test_case = self

        class FakeToolLoopLLM:
            def __init__(self) -> None:
                self.calls = []

            def complete(self, *, messages, tools=None, tool_choice=None):
                self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
                if tools and not any(message.get("role") == "tool" for message in messages):
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-search-1",
                                name="external_web_search",
                                arguments={"query": "中科曙光 校园招聘 秋招 官网", "max_results": 5},
                            )
                        ],
                    )
                combined = "\n".join(str(message.get("content") or "") for message in messages)
                test_case.assertIn("工具 external.web_search 执行状态：succeeded", combined)
                test_case.assertIn("中科曙光校园招聘官网", combined)
                return LLMChatCompletion(content="已找到中科曙光校园招聘官网：https://jobs.example.com/sugon")

        search_calls = []

        def fake_web_search(_session, *, query: str, max_results: int = 5):
            search_calls.append({"query": query, "max_results": max_results})
            return {
                "tool_name": EXTERNAL_WEB_SEARCH_TOOL,
                "ok": True,
                "result": {
                    "answer": "中科曙光校园招聘官网：https://jobs.example.com/sugon",
                    "sources": [{"title": "中科曙光校园招聘官网", "url": "https://jobs.example.com/sugon"}],
                },
            }

        with self.Session() as session:
            session_id = self._session_id(session)
            fake_llm = FakeToolLoopLLM()
            dependencies = self._dependencies(
                session,
                llm_client=fake_llm,
                intent_detector=HybridIntentDetector(llm_client=FakeIntentLLM()),
            )
            registry = AgentToolRegistry(
                definition for definition in dependencies.registry.list_definitions() if definition.name != EXTERNAL_WEB_SEARCH_TOOL
            )
            registry.register(
                AgentToolDefinition(
                    name=EXTERNAL_WEB_SEARCH_TOOL,
                    description="Search the public web through an external agent.",
                    input_schema={
                        "type": "object",
                        "required": ["query"],
                        "properties": {
                            "query": {"type": "string"},
                            "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
                        },
                        "additionalProperties": False,
                    },
                    output_schema={"type": "object", "required": ["tool_name", "ok", "result"]},
                    handler=fake_web_search,
                    allowed_source_types=frozenset({"agent_chat", "web_search"}),
                )
            )

            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="你给我搜一下中科曙光的校园招聘信息"),
                dependencies=dependencies.with_registry(registry),
            )
            session.commit()
            tool_log = session.scalars(select(ToolCallLog)).one()

        self.assertEqual([{"query": "中科曙光 校园招聘 秋招 官网", "max_results": 5}], search_calls)
        self.assertEqual(EXTERNAL_WEB_SEARCH_TOOL, result.state.requested_tool_name)
        self.assertEqual("llm_tool_choice_loop", result.state.response_mode)
        self.assertEqual("已找到中科曙光校园招聘官网：https://jobs.example.com/sugon", result.state.final_response)
        self.assertEqual(ToolCallStatus.SUCCEEDED, tool_log.status)
        self.assertEqual(EXTERNAL_WEB_SEARCH_TOOL, tool_log.tool_name)
        self.assertEqual(2, len(fake_llm.calls))
        self.assertEqual("auto", fake_llm.calls[0]["tool_choice"])
        self.assertIn(
            "external_web_search",
            [tool["function"]["name"] for tool in fake_llm.calls[0]["tools"]],
        )

    def test_native_blocked_tool_call_is_recorded_as_structured_result_pair(self) -> None:
        from app.agent_runtime.graph_factory import AgentState, _blocked_native_tool_call_response
        from app.domains.automation.models import ToolCallLog, ToolCallStatus, WorkflowRunStatus
        from app.domains.conversations.models import AgentMessageRole
        from app.domains.automation.schemas import WorkflowRunCreate

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(session)
            workflow = dependencies.automation_service.start_workflow(
                WorkflowRunCreate(
                    workflow_type="agent_chat",
                    current_step="plan_or_reply",
                    user_goal="调用一个未允许的工具",
                )
            )
            state = AgentState(
                session_id=session_id,
                workflow_run_id=workflow.id,
                agent_run_id="agent-run-native-blocked",
                user_message="调用一个未允许的工具",
                current_step="plan_or_reply",
                context_metadata={
                    "context_pack": {
                        "allowed_capabilities": ["external.web_search"],
                    }
                },
            )

            blocked = _blocked_native_tool_call_response(
                state,
                dependencies=dependencies,
                requested_tool_name="filesystem.delete_file",
                tool_input={"path": "C:/tmp/forbidden.txt"},
                tool_call_id="call-blocked-1",
                reason="Model requested a tool outside this turn's ContextPack.",
                details={"requested_tool_name": "filesystem.delete_file"},
            )
            session.commit()
            tool_log = session.scalars(select(ToolCallLog)).one()
            messages = dependencies.conversation_service.list_messages(session_id, limit=20)

        self.assertEqual(ToolCallStatus.BLOCKED, tool_log.status)
        self.assertEqual("filesystem.delete_file", tool_log.tool_name)
        self.assertEqual("native_tool_loop", tool_log.output_payload["execution"])
        result_payload = tool_log.output_payload["result"]
        self.assertEqual("NATIVE_TOOL_CALL_BLOCKED", result_payload["error_code"])
        self.assertFalse(result_payload["ok"])
        self.assertTrue(result_payload["retryable"])
        self.assertEqual("select_alternative_tool", result_payload["next_action"])
        self.assertEqual([tool_log.id], blocked.tool_call_ids)
        self.assertEqual("filesystem.delete_file", blocked.llm_messages[-1]["metadata"]["content_json"]["tool_name"])
        self.assertEqual("blocked", blocked.llm_messages[-1]["metadata"]["content_json"]["status"])
        tool_messages = [message for message in messages if message.tool_call_log_id == tool_log.id]
        self.assertEqual([AgentMessageRole.TOOL_CALL, AgentMessageRole.TOOL_RESULT], [message.role for message in tool_messages])
        self.assertEqual("call-blocked-1", tool_messages[0].content_json["tool_call_id"])
        self.assertEqual("call-blocked-1", tool_messages[1].content_json["tool_call_id"])
        self.assertEqual(WorkflowRunStatus.RUNNING, workflow.status)

    def test_native_blocked_tool_call_returns_observation_to_model_for_recovery(self) -> None:
        from app.agent_runtime.graph_factory import _native_tool_loop_node
        from app.agent_runtime.state import AgentState
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL, AgentToolDefinition, AgentToolRegistry
        from app.domains.automation.models import ToolCallLog, ToolCallStatus
        from app.domains.automation.schemas import WorkflowRunCreate
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        test_case = self

        class FakeNativeLoopLLM:
            def __init__(self) -> None:
                self.calls = []

            def complete(self, *, messages, tools=None, tool_choice=None):
                combined = "\n".join(str(message.get("content") or "") for message in messages)
                self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
                if len(self.calls) == 1:
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-invalid-1",
                                name="filesystem_delete_file",
                                arguments={"path": "C:/tmp/forbidden.txt"},
                            )
                        ],
                    )
                if len(self.calls) == 2:
                    test_case.assertIn("NATIVE_TOOL_CALL_BLOCKED", combined)
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-search-after-block",
                                name="external_web_search",
                                arguments={"query": "中科曙光 校园招聘 官网", "max_results": 3},
                            )
                        ],
                    )
                    test_case.assertIn("Tool result: external.web_search succeeded", combined)
                return LLMChatCompletion(content="已根据修正后的工具调用完成搜索。")

        def fake_web_search(_session, *, query: str, max_results: int = 5):
            return {
                "tool_name": EXTERNAL_WEB_SEARCH_TOOL,
                "ok": True,
                "result": {"answer": f"{query}：官方入口", "sources": []},
            }

        with self.Session() as session:
            session_id = self._session_id(session)
            fake_llm = FakeNativeLoopLLM()
            dependencies = self._dependencies(session, llm_client=fake_llm)
            registry = AgentToolRegistry(
                definition for definition in dependencies.registry.list_definitions() if definition.name != EXTERNAL_WEB_SEARCH_TOOL
            )
            registry.register(
                AgentToolDefinition(
                    name=EXTERNAL_WEB_SEARCH_TOOL,
                    description="Search the public web.",
                    input_schema={
                        "type": "object",
                        "required": ["query"],
                        "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
                        "additionalProperties": False,
                    },
                    output_schema={"type": "object"},
                    handler=fake_web_search,
                    allowed_source_types=frozenset({"agent_chat"}),
                )
            )
            workflow = dependencies.automation_service.start_workflow(
                WorkflowRunCreate(
                    workflow_type="agent_chat",
                    current_step="plan_or_reply",
                    user_goal="搜索校园招聘信息",
                )
            )
            state = AgentState(
                session_id=session_id,
                workflow_run_id=workflow.id,
                agent_run_id="agent-run-native-recovery",
                user_message="搜索校园招聘信息",
                current_step="plan_or_reply",
                llm_messages=[{"role": "user", "content": "搜索校园招聘信息"}],
                context_metadata={
                    "context_pack": {
                        "risk_level": "low",
                        "allowed_capabilities": [EXTERNAL_WEB_SEARCH_TOOL],
                        "intent_frame": {"entities": {"company_names": ["中科曙光", "歌尔"]}},
                    }
                },
            )

            result = _native_tool_loop_node(state, dependencies=dependencies.with_registry(registry))
            session.commit()
            tool_logs = list(session.scalars(select(ToolCallLog)).all())

        self.assertEqual("已根据修正后的工具调用完成搜索。", result.final_response)
        self.assertEqual("llm_tool_loop", result.response_mode)
        self.assertGreaterEqual(len(fake_llm.calls), 3)
        self.assertEqual(2, len(tool_logs))
        self.assertEqual([ToolCallStatus.BLOCKED, ToolCallStatus.SUCCEEDED], [log.status for log in tool_logs])
        self.assertEqual(2, len(result.tool_call_ids))
        self.assertEqual("blocked", result.context_metadata["loop_agent"]["trace"][0]["observation_status"])
        self.assertEqual("succeeded", result.context_metadata["loop_agent"]["trace"][1]["observation_status"])

    def test_tool_choice_loop_enters_web_search_for_public_company_question_without_fixed_intent(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL, AgentToolDefinition, AgentToolRegistry
        from app.domains.automation.models import ToolCallLog, ToolCallStatus
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        test_case = self

        class FakeToolChoiceLLM:
            def __init__(self) -> None:
                self.calls = []

            def complete(self, *, messages, tools=None, tool_choice=None):
                self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
                combined = "\n".join(str(message.get("content") or "") for message in messages)
                if tools and "Canonical 是 Ubuntu 背后的公司" not in combined:
                    test_case.assertEqual("auto", tool_choice)
                    test_case.assertIn("external_web_search", [tool["function"]["name"] for tool in tools])
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-canonical-public-info",
                                name="external_web_search",
                                arguments={"query": "Canonical Ltd. 主要业务", "max_results": 3},
                            )
                        ],
                    )
                test_case.assertIn("Canonical 是 Ubuntu 背后的公司", combined)
                return LLMChatCompletion(content="Canonical Ltd. 主要做 Ubuntu、企业 Linux、云基础设施和安全支持。")

        search_calls = []

        def fake_web_search(_session, *, query: str, max_results: int = 5):
            search_calls.append({"query": query, "max_results": max_results})
            return {
                "tool_name": EXTERNAL_WEB_SEARCH_TOOL,
                "ok": True,
                "result": {
                    "executor_name": "fake-public-search",
                    "query": query,
                    "answer": "Canonical 是 Ubuntu 背后的公司，主要提供企业 Linux、云基础设施和安全支持。",
                    "sources": [{"title": "Canonical", "url": "https://canonical.com/"}],
                },
            }

        with self.Session() as session:
            session_id = self._session_id(session)
            fake_llm = FakeToolChoiceLLM()
            dependencies = self._dependencies(session, llm_client=fake_llm)
            registry = AgentToolRegistry(
                definition
                for definition in dependencies.registry.list_definitions()
                if definition.name != EXTERNAL_WEB_SEARCH_TOOL
            )
            registry.register(
                AgentToolDefinition(
                    name=EXTERNAL_WEB_SEARCH_TOOL,
                    description="Search the public web through an external agent.",
                    input_schema={
                        "type": "object",
                        "required": ["query"],
                        "properties": {
                            "query": {"type": "string"},
                            "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
                        },
                        "additionalProperties": False,
                    },
                    output_schema={"type": "object", "required": ["tool_name", "ok", "result"]},
                    handler=fake_web_search,
                    allowed_source_types=frozenset({"agent_chat", "web_search"}),
                )
            )

            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="Canonical Ltd. 是做什么的？主要业务是什么？"),
                dependencies=dependencies.with_registry(registry),
            )
            session.commit()
            tool_log = session.scalars(select(ToolCallLog)).one()

        self.assertEqual([{"query": "Canonical Ltd. 主要业务", "max_results": 3}], search_calls)
        self.assertEqual(EXTERNAL_WEB_SEARCH_TOOL, result.state.requested_tool_name)
        self.assertEqual("llm_tool_choice_loop", result.state.response_mode)
        self.assertEqual("Canonical Ltd. 主要做 Ubuntu、企业 Linux、云基础设施和安全支持。", result.state.final_response)
        self.assertEqual(ToolCallStatus.SUCCEEDED, tool_log.status)
        self.assertEqual(EXTERNAL_WEB_SEARCH_TOOL, tool_log.tool_name)
        self.assertEqual(2, len(fake_llm.calls))
        self.assertEqual("model_final", result.state.context_metadata["tool_choice_loop"]["stop_reason"])

    def test_tool_choice_loop_streams_candidate_and_model_decision_events(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL, AgentToolDefinition, AgentToolRegistry
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        class FakeToolChoiceLLM:
            def complete(self, *, messages, tools=None, tool_choice=None):
                combined = "\n".join(str(message.get("content") or "") for message in messages)
                if tools and "Canonical 是 Ubuntu 背后的公司" not in combined:
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-canonical-event-test",
                                name="external_web_search",
                                arguments={"query": "Canonical Ltd. 主要业务", "max_results": 3},
                            )
                        ],
                    )
                return LLMChatCompletion(content="Canonical Ltd. 主要做 Ubuntu。")

        def fake_web_search(_session, *, query: str, max_results: int = 5):
            return {
                "tool_name": EXTERNAL_WEB_SEARCH_TOOL,
                "ok": True,
                "result": {
                    "executor_name": "fake-public-search",
                    "query": query,
                    "answer": "Canonical 是 Ubuntu 背后的公司。",
                    "sources": [{"title": "Canonical", "url": "https://canonical.com/"}],
                },
            }

        emitted_events = []
        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(session, llm_client=FakeToolChoiceLLM())
            registry = AgentToolRegistry(
                definition
                for definition in dependencies.registry.list_definitions()
                if definition.name != EXTERNAL_WEB_SEARCH_TOOL
            )
            registry.register(
                AgentToolDefinition(
                    name=EXTERNAL_WEB_SEARCH_TOOL,
                    description="Search the public web through an external agent.",
                    input_schema={
                        "type": "object",
                        "required": ["query"],
                        "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
                        "additionalProperties": False,
                    },
                    output_schema={"type": "object", "required": ["tool_name", "ok", "result"]},
                    handler=fake_web_search,
                    allowed_source_types=frozenset({"agent_chat", "web_search"}),
                )
            )

            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="Canonical Ltd. 是做什么的？主要业务是什么？"),
                dependencies=dependencies.with_registry(registry).with_event_sink(emitted_events.append),
            )
            session.commit()

        self.assertEqual("llm_tool_choice_loop", result.state.response_mode)
        event_types = [event["event_type"] for event in emitted_events]
        self.assertIn("candidate_capabilities", event_types)
        self.assertIn("model_decision", event_types)
        self.assertIn("turn_started", event_types)
        self.assertIn("turn_finished", event_types)
        candidate_event = next(event for event in emitted_events if event["event_type"] == "candidate_capabilities")
        self.assertIn(EXTERNAL_WEB_SEARCH_TOOL, candidate_event["candidate_capabilities"])
        decision_event = next(event for event in emitted_events if event["event_type"] == "model_decision")
        self.assertEqual(EXTERNAL_WEB_SEARCH_TOOL, decision_event["tool_name"])
        self.assertEqual(EXTERNAL_WEB_SEARCH_TOOL, decision_event["capability"])

    def test_tool_choice_loop_runs_before_legacy_capability_routing(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL, AgentToolDefinition, AgentToolRegistry
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        class LegacyRoutingShouldNotRun:
            def decide(self, *, user_message, intent_frame, context_pack):  # pragma: no cover - failing path
                raise AssertionError("legacy capability routing should not run before tool choice loop")

        class FakeToolChoiceLLM:
            def complete(self, *, messages, tools=None, tool_choice=None):
                combined = "\n".join(str(message.get("content") or "") for message in messages)
                if tools and "Canonical 是 Ubuntu 背后的公司" not in combined:
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-canonical-new-route",
                                name="external_web_search",
                                arguments={"query": "Canonical Ltd. 主要业务", "max_results": 3},
                            )
                        ],
                    )
                return LLMChatCompletion(content="Canonical Ltd. 主要做 Ubuntu。")

        def fake_web_search(_session, *, query: str, max_results: int = 5):
            return {
                "tool_name": EXTERNAL_WEB_SEARCH_TOOL,
                "ok": True,
                "result": {
                    "executor_name": "fake-public-search",
                    "query": query,
                    "answer": "Canonical 是 Ubuntu 背后的公司。",
                    "sources": [{"title": "Canonical", "url": "https://canonical.com/"}],
                },
            }

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(
                session,
                llm_client=FakeToolChoiceLLM(),
                capability_routing_middleware=LegacyRoutingShouldNotRun(),
            )
            registry = AgentToolRegistry(
                definition
                for definition in dependencies.registry.list_definitions()
                if definition.name != EXTERNAL_WEB_SEARCH_TOOL
            )
            registry.register(
                AgentToolDefinition(
                    name=EXTERNAL_WEB_SEARCH_TOOL,
                    description="Search the public web through an external agent.",
                    input_schema={
                        "type": "object",
                        "required": ["query"],
                        "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
                        "additionalProperties": False,
                    },
                    output_schema={"type": "object", "required": ["tool_name", "ok", "result"]},
                    handler=fake_web_search,
                    allowed_source_types=frozenset({"agent_chat", "web_search"}),
                )
            )

            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="Canonical Ltd. 是做什么的？主要业务是什么？"),
                dependencies=dependencies.with_registry(registry),
            )
            session.commit()

        self.assertEqual("llm_tool_choice_loop", result.state.response_mode)
        self.assertEqual(EXTERNAL_WEB_SEARCH_TOOL, result.state.requested_tool_name)
        self.assertNotIn("capability_routing", result.state.context_metadata)

    def test_tool_choice_loop_enters_web_search_for_realtime_question_without_fixed_intent(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL, AgentToolDefinition, AgentToolRegistry
        from app.agent_runtime.understanding.schemas import IntentFrame
        from app.domains.automation.models import ToolCallLog, ToolCallStatus
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        class FakeNormalIntentDetector:
            def detect(self, _message):
                return IntentFrame(intent="normal_chat", confidence=0.0)

        test_case = self

        class FakeRealtimeLLM:
            def __init__(self) -> None:
                self.calls = []

            def complete(self, *, messages, tools=None, tool_choice=None):
                self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
                combined = "\n".join(str(message.get("content") or "") for message in messages)
                if tools and "梅西今天有比赛" not in combined:
                    self_tool_names = [tool["function"]["name"] for tool in tools]
                    self_tool_names.sort()
                    test_case.assertIn("external_web_search", self_tool_names)
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-messi-match",
                                name="external_web_search",
                                arguments={"query": "梅西 今天 比赛 赛程 结果", "max_results": 5},
                            )
                        ],
                    )
                return LLMChatCompletion(content="梅西今天有比赛，具体时间以官方赛程为准。")

        search_calls = []

        def fake_web_search(_session, *, query: str, max_results: int = 5):
            search_calls.append({"query": query, "max_results": max_results})
            return {
                "tool_name": EXTERNAL_WEB_SEARCH_TOOL,
                "ok": True,
                "result": {
                    "query": query,
                    "answer": "梅西今天有比赛，开球时间为当地时间晚上。",
                    "sources": [{"title": "Match schedule", "url": "https://example.com/match"}],
                },
            }

        with self.Session() as session:
            session_id = self._session_id(session)
            fake_llm = FakeRealtimeLLM()
            dependencies = self._dependencies(session, llm_client=fake_llm, intent_detector=FakeNormalIntentDetector())
            registry = AgentToolRegistry(
                definition
                for definition in dependencies.registry.list_definitions()
                if definition.name != EXTERNAL_WEB_SEARCH_TOOL
            )
            registry.register(
                AgentToolDefinition(
                    name=EXTERNAL_WEB_SEARCH_TOOL,
                    description="Search the public web through an external agent.",
                    input_schema={
                        "type": "object",
                        "required": ["query"],
                        "properties": {
                            "query": {"type": "string"},
                            "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
                        },
                        "additionalProperties": False,
                    },
                    output_schema={"type": "object", "required": ["tool_name", "ok", "result"]},
                    handler=fake_web_search,
                    allowed_source_types=frozenset({"agent_chat", "web_search"}),
                )
            )

            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="给我查一下梅西今天的比赛"),
                dependencies=dependencies.with_registry(registry),
            )
            session.commit()
            tool_log = session.scalars(select(ToolCallLog)).one()

        self.assertEqual(1, len(search_calls))
        self.assertEqual(5, search_calls[0]["max_results"])
        self.assertIn("Lionel Messi", search_calls[0]["query"])
        self.assertIn("梅西", search_calls[0]["query"])
        self.assertIn("Inter Miami", search_calls[0]["query"])
        self.assertIn("football fixtures", search_calls[0]["query"])
        self.assertIn(date.today().isoformat(), search_calls[0]["query"])
        self.assertEqual(EXTERNAL_WEB_SEARCH_TOOL, result.state.requested_tool_name)
        self.assertEqual("llm_tool_choice_loop", result.state.response_mode)
        self.assertEqual("梅西今天有比赛，具体时间以官方赛程为准。", result.state.final_response)
        self.assertEqual(ToolCallStatus.SUCCEEDED, tool_log.status)


    def test_tool_choice_loop_changes_web_search_query_after_off_target_result(self) -> None:
        # Native structured tool-call regression coverage.
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL, AgentToolDefinition, AgentToolRegistry
        from app.agent_runtime.understanding.schemas import IntentFrame
        from app.domains.automation.models import ToolCallLog
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        class FakeNormalIntentDetector:
            def detect(self, _message):
                return IntentFrame(intent="normal_chat", confidence=0.0)

        class FakeLLM:
            def __init__(self) -> None:
                self.calls = []

            def complete(self, *, messages, tools=None, tool_choice=None):
                self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
                combined = "\n".join(str(message.get("content") or "") for message in messages)
                if tools and "Al Nassr fixtures official" not in combined:
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-ronaldo-fixtures",
                                name="external_web_search",
                                arguments={"query": "Cristiano Ronaldo Al Nassr fixtures", "max_results": 5},
                            )
                        ],
                    )
                    return LLMChatCompletion(
                        content='Tool call: external.web_search{"query":"C罗 本周 比赛日程","max_results":5}'
                    )
                return LLMChatCompletion(content="C 罗本周赛程以官方公布为准。")

        search_calls = []

        def fake_web_search(_session, *, query: str, max_results: int = 5):
            search_calls.append({"query": query, "max_results": max_results})
            if len(search_calls) == 1:
                return {
                    "tool_name": EXTERNAL_WEB_SEARCH_TOOL,
                    "ok": True,
                    "result": {
                        "query": query,
                        "answer": "检索结果均为UTF-8编码转换类工具网站，与足球赛程无关。",
                        "sources": [{"title": "UTF-8 编码转换", "url": "https://example.com/utf8"}],
                    },
                }
            return {
                "tool_name": EXTERNAL_WEB_SEARCH_TOOL,
                "ok": True,
                "result": {
                    "query": query,
                    "answer": "Al Nassr fixtures official：C 罗本周赛程以官方公布为准。",
                    "sources": [{"title": "Al Nassr fixtures official", "url": "https://example.com/al-nassr"}],
                },
            }

        with self.Session() as session:
            session_id = self._session_id(session)
            fake_llm = FakeLLM()
            dependencies = self._dependencies(session, llm_client=fake_llm, intent_detector=FakeNormalIntentDetector())
            registry = AgentToolRegistry(
                definition
                for definition in dependencies.registry.list_definitions()
                if definition.name != EXTERNAL_WEB_SEARCH_TOOL
            )
            registry.register(
                AgentToolDefinition(
                    name=EXTERNAL_WEB_SEARCH_TOOL,
                    description="Search the public web through an external agent.",
                    input_schema={
                        "type": "object",
                        "required": ["query"],
                        "properties": {
                            "query": {"type": "string"},
                            "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
                        },
                        "additionalProperties": False,
                    },
                    output_schema={"type": "object", "required": ["tool_name", "ok", "result"]},
                    handler=fake_web_search,
                    allowed_source_types=frozenset({"agent_chat", "web_search"}),
                )
            )

            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="你看一下c罗这个星期有什么比赛吗"),
                dependencies=dependencies.with_registry(registry),
            )
            session.commit()
            tool_logs = session.scalars(select(ToolCallLog)).all()

        self.assertEqual(2, len(search_calls))
        self.assertNotEqual(search_calls[0]["query"], search_calls[1]["query"])
        self.assertIn("Cristiano Ronaldo", search_calls[1]["query"])
        self.assertIn("Al Nassr", search_calls[1]["query"])
        self.assertIn("fixtures", search_calls[1]["query"])
        self.assertNotIn("校园招聘", search_calls[1]["query"])
        self.assertEqual(2, len(tool_logs))
        trace = result.state.context_metadata["tool_choice_loop"]["trace"]
        self.assertEqual("retry", trace[0]["metadata"]["observation"]["suggested_next_decision"]["metadata"]["reflection"]["next_action"])

    def test_tool_choice_loop_does_not_summarize_bad_web_search_as_answer(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL, AgentToolDefinition, AgentToolRegistry
        from app.agent_runtime.understanding.schemas import IntentFrame
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        class FakeNormalIntentDetector:
            def detect(self, _message):
                return IntentFrame(intent="normal_chat", confidence=0.0)

        class FakeLLM:
            def complete(self, *, messages, tools=None, tool_choice=None):
                if tools and not any(message.get("role") == "tool" for message in messages):
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-ronaldo-bad-search",
                                name="external_web_search",
                                arguments={"query": "Cristiano Ronaldo weekly fixtures", "max_results": 5},
                            )
                        ],
                    )
                    return LLMChatCompletion(
                        content='Tool call: external.web_search{"query":"C罗 本周 比赛日程","max_results":5}'
                    )
                return LLMChatCompletion(content="C 罗通常周末比赛，本周可能有沙特联赛。")

        search_calls = []

        def fake_bad_search(_session, *, query: str, max_results: int = 5):
            search_calls.append({"query": query, "max_results": max_results})
            return {
                "tool_name": EXTERNAL_WEB_SEARCH_TOOL,
                "ok": True,
                "result": {
                    "query": query,
                    "answer": "检索结果均为贝锐向日葵远程控制软件相关页面，与足球赛程无关。",
                    "sources": [{"title": "向日葵远程控制", "url": "https://example.com/sunlogin"}],
                },
            }

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(session, llm_client=FakeLLM(), intent_detector=FakeNormalIntentDetector())
            registry = AgentToolRegistry(
                definition
                for definition in dependencies.registry.list_definitions()
                if definition.name != EXTERNAL_WEB_SEARCH_TOOL
            )
            registry.register(
                AgentToolDefinition(
                    name=EXTERNAL_WEB_SEARCH_TOOL,
                    description="Search the public web through an external agent.",
                    input_schema={
                        "type": "object",
                        "required": ["query"],
                        "properties": {
                            "query": {"type": "string"},
                            "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
                        },
                        "additionalProperties": False,
                    },
                    output_schema={"type": "object", "required": ["tool_name", "ok", "result"]},
                    handler=fake_bad_search,
                    allowed_source_types=frozenset({"agent_chat", "web_search"}),
                )
            )

            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="你看一下c罗这个星期有什么比赛吗"),
                dependencies=dependencies.with_registry(registry),
            )

        self.assertEqual(2, len(search_calls))
        self.assertIn("没有找到可靠", result.state.final_response)
        self.assertIn("联网搜索", result.state.final_response)
        self.assertNotIn("通常周末", result.state.final_response)
        self.assertEqual("tool_result_summary_unreliable", result.state.response_mode)

    def test_tool_choice_loop_enters_local_company_database_without_fixed_intent(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.tool_registry import DATABASE_COMPANY_LIST_TOOL
        from app.agent_runtime.understanding.schemas import IntentFrame
        from app.domains.automation.models import ToolCallLog, ToolCallStatus
        from app.domains.jobs.models import Company
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        class FakeNormalIntentDetector:
            def detect(self, _message):
                return IntentFrame(intent="normal_chat", confidence=0.0)

        test_case = self

        class FakeCompanyDatabaseLLM:
            def __init__(self) -> None:
                self.calls = []

            def complete(self, *, messages, tools=None, tool_choice=None):
                self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
                combined = "\n".join(str(message.get("content") or "") for message in messages)
                if tools and "腾讯" not in combined:
                    test_case.assertIn("database_company_list", [tool["function"]["name"] for tool in tools])
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-database-company-list",
                                name="database_company_list",
                                arguments={"limit": 20},
                            )
                        ],
                    )
                return LLMChatCompletion(content="下面是本地数据库里的公司表格，包含腾讯和 Canonical Ltd.。")

        with self.Session() as session:
            session.add_all(
                [
                    Company(name="Canonical Ltd.", normalized_name="canonical ltd"),
                    Company(name="腾讯", normalized_name="tencent"),
                ]
            )
            session.commit()
            session_id = self._session_id(session)
            fake_llm = FakeCompanyDatabaseLLM()
            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="数据库里有哪些公司，给我20个"),
                dependencies=self._dependencies(
                    session,
                    llm_client=fake_llm,
                    intent_detector=FakeNormalIntentDetector(),
                ),
            )
            session.commit()
            tool_log = session.scalars(select(ToolCallLog)).one()

        self.assertEqual(DATABASE_COMPANY_LIST_TOOL, result.state.requested_tool_name)
        self.assertEqual("llm_tool_choice_loop", result.state.response_mode)
        self.assertEqual("下面是本地数据库里的公司表格，包含腾讯和 Canonical Ltd.。", result.state.final_response)
        self.assertEqual(ToolCallStatus.SUCCEEDED, tool_log.status)
        self.assertEqual(DATABASE_COMPANY_LIST_TOOL, tool_log.tool_name)
        self.assertEqual({"limit": 20}, tool_log.input_payload)
        self.assertEqual(2, len(fake_llm.calls))

    def test_tool_choice_loop_enters_local_job_source_overview_without_fixed_intent(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.tool_registry import LOCAL_JOB_SOURCE_OVERVIEW_TOOL, AgentToolDefinition, AgentToolRegistry
        from app.agent_runtime.understanding.schemas import IntentFrame
        from app.domains.automation.models import ToolCallLog, ToolCallStatus
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        class FakeNormalIntentDetector:
            def detect(self, _message):
                return IntentFrame(intent="normal_chat", confidence=0.0)

        test_case = self
        job_source_calls = []

        class FakeJobSourceLLM:
            def __init__(self) -> None:
                self.calls = []

            def complete(self, *, messages, tools=None, tool_choice=None):
                self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
                combined = "\n".join(str(message.get("content") or "") for message in messages)
                if tools and "source_count" not in combined:
                    test_case.assertIn("local_job_source_overview", [tool["function"]["name"] for tool in tools])
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-local-job-source-overview",
                                name="local_job_source_overview",
                                arguments={"sample_limit": 20, "include_external_job_board": True},
                            )
                        ],
                    )
                return LLMChatCompletion(content="来源库共有 7 条，下面按表格列出主要来源。")

        def fake_job_source_overview(_session, *, sample_limit: int = 10, include_external_job_board: bool = True):
            job_source_calls.append(
                {"sample_limit": sample_limit, "include_external_job_board": include_external_job_board}
            )
            return {
                "tool_name": LOCAL_JOB_SOURCE_OVERVIEW_TOOL,
                "ok": True,
                "result": {
                    "source_count": 7,
                    "lead_count": 71,
                    "external_job_board": {"company_count": 1247},
                    "sample_sources": ["开放岗位来源库"],
                },
            }

        with self.Session() as session:
            session_id = self._session_id(session)
            fake_llm = FakeJobSourceLLM()
            dependencies = self._dependencies(session, llm_client=fake_llm, intent_detector=FakeNormalIntentDetector())
            registry = AgentToolRegistry(
                definition
                for definition in dependencies.registry.list_definitions()
                if definition.name != LOCAL_JOB_SOURCE_OVERVIEW_TOOL
            )
            registry.register(
                AgentToolDefinition(
                    name=LOCAL_JOB_SOURCE_OVERVIEW_TOOL,
                    description="Read a safe overview of local job sources.",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "sample_limit": {"type": "integer", "minimum": 1, "maximum": 50},
                            "include_external_job_board": {"type": "boolean"},
                        },
                        "additionalProperties": False,
                    },
                    output_schema={"type": "object", "required": ["tool_name", "ok", "result"]},
                    handler=fake_job_source_overview,
                    allowed_source_types=frozenset({"agent_chat", "job_discovery"}),
                )
            )

            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="岗位来源库现在有多少条，给我20个"),
                dependencies=dependencies.with_registry(registry),
            )
            session.commit()
            tool_log = session.scalars(select(ToolCallLog)).one()

        self.assertEqual([{"sample_limit": 20, "include_external_job_board": True}], job_source_calls)
        self.assertEqual(LOCAL_JOB_SOURCE_OVERVIEW_TOOL, result.state.requested_tool_name)
        self.assertEqual("llm_tool_choice_loop", result.state.response_mode)
        self.assertEqual("来源库共有 7 条，下面按表格列出主要来源。", result.state.final_response)
        self.assertEqual(ToolCallStatus.SUCCEEDED, tool_log.status)
        self.assertEqual(LOCAL_JOB_SOURCE_OVERVIEW_TOOL, tool_log.tool_name)
        self.assertEqual({"sample_limit": 20, "include_external_job_board": True}, tool_log.input_payload)
        self.assertEqual(2, len(fake_llm.calls))

    def test_tool_choice_loop_can_offer_declared_sub_agent_capability(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentCapabilityDefinition, AgentRuntimeContext, AgentTask, StandardAgentResult
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.tool_registry import AgentToolCandidateProfile
        from app.agent_runtime.understanding.schemas import IntentFrame
        from app.domains.automation.models import ToolCallLog, ToolCallStatus
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        class FakeNormalIntentDetector:
            def detect(self, _message):
                return IntentFrame(intent="normal_chat", confidence=0.0)

        class FakeOpenAIAgent:
            executor_id = "openai-sdk-agent"

            def __init__(self) -> None:
                self.calls = []

            def capabilities(self):
                return [
                    AgentCapabilityDefinition(
                        capability_id="resume.tailor",
                        name="简历优化",
                        description="根据目标岗位优化简历表达。",
                        executor_id=self.executor_id,
                        input_schema={
                            "type": "object",
                            "required": ["resume_text", "job_description"],
                            "properties": {
                                "resume_text": {"type": "string"},
                                "job_description": {"type": "string"},
                            },
                            "additionalProperties": False,
                        },
                        output_schema={"type": "object", "required": ["revised_resume"]},
                        risk_level="low",
                        allowed_source_types=frozenset({"agent_chat"}),
                        candidate_profile=AgentToolCandidateProfile(
                            categories=frozenset({"resume_tailoring", "content_processing"}),
                            keywords=frozenset({"优化简历", "改简历", "匹配 JD"}),
                            examples=("帮我优化这段简历，让它更适合腾讯后端岗位",),
                        ),
                    )
                ]

            def call(self, task: AgentTask, context: AgentRuntimeContext) -> StandardAgentResult:
                self.calls.append({"task": task, "context": context})
                return StandardAgentResult(
                    status="succeeded",
                    summary="已优化简历，突出 Java 后端项目经验。",
                    observation="已优化简历：突出分布式系统、Java 后端和项目结果。",
                    raw_result={
                        "revised_resume": "优化后的简历内容",
                        "metadata": {
                            "telemetry": {
                                "schema_version": "offer_master.sdk_agent_telemetry.v1",
                                "subagent_name": "OfferMasterSdkAgent",
                                "capability_id": "resume.tailor",
                                "status": "succeeded",
                                "allowed_tool_count": 2,
                            }
                        },
                    },
                )

        test_case = self

        class FakeResumeLLM:
            def __init__(self) -> None:
                self.calls = []

            def complete(self, *, messages, tools=None, tool_choice=None):
                self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
                combined = "\n".join(str(message.get("content") or "") for message in messages)
                if tools and "已优化简历" not in combined:
                    test_case.assertIn("resume_tailor", [tool["function"]["name"] for tool in tools])
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-resume-tailor",
                                name="resume_tailor",
                                arguments={
                                    "resume_text": "旧简历：做过 Java 项目。",
                                    "job_description": "腾讯后端岗位，要求 Java 和分布式系统。",
                                },
                            )
                        ],
                    )
                return LLMChatCompletion(content="已根据腾讯后端岗位优化简历，重点突出 Java 和分布式项目经验。")

        with self.Session() as session:
            session_id = self._session_id(session)
            fake_agent = FakeOpenAIAgent()
            fake_llm = FakeResumeLLM()
            dependencies = self._dependencies(
                session,
                llm_client=fake_llm,
                intent_detector=FakeNormalIntentDetector(),
            ).with_agent_runtime(executors={fake_agent.executor_id: fake_agent})

            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="帮我优化这段简历，让它更适合腾讯后端岗位"),
                dependencies=dependencies,
            )
            session.commit()
            tool_log = session.scalars(select(ToolCallLog)).one()

        self.assertEqual("resume.tailor", result.state.requested_tool_name)
        self.assertEqual("llm_tool_choice_loop", result.state.response_mode)
        self.assertEqual("已根据腾讯后端岗位优化简历，重点突出 Java 和分布式项目经验。", result.state.final_response)
        self.assertEqual(ToolCallStatus.SUCCEEDED, tool_log.status)
        self.assertEqual("resume.tailor", tool_log.tool_name)
        self.assertEqual("openai-sdk-agent", tool_log.output_payload["agent_runtime"]["executor_id"])
        self.assertEqual("offer_master.sdk_agent_telemetry.v1", tool_log.output_payload["agent_runtime"]["telemetry"]["schema_version"])
        self.assertEqual(2, tool_log.output_payload["agent_runtime"]["telemetry"]["allowed_tool_count"])
        self.assertEqual("resume.tailor", fake_agent.calls[0]["task"].capability_id)
        self.assertEqual("openai-sdk-agent", fake_agent.calls[0]["context"].namespace)
        self.assertEqual(2, len(fake_llm.calls))

    def test_main_model_can_delegate_to_always_available_chrome_and_dbx_agents(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentCapabilityDefinition, AgentRuntimeContext, AgentTask, StandardAgentResult
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.understanding.schemas import IntentFrame
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        class FakeNormalIntentDetector:
            def detect(self, _message):
                return IntentFrame(intent="normal_chat", confidence=0.0)

        class FakeChildAgent:
            executor_id = "openai-sdk-agent"

            def __init__(self) -> None:
                self.calls = []

            def capabilities(self):
                definitions = []
                for capability_id, name, description in (
                    ("agent.google_chrome", "Google Chrome 浏览器 Agent", "处理浏览器任务。"),
                    ("agent.dbx_readonly", "DBX 只读数据库 Agent", "处理只读数据库任务。"),
                ):
                    definitions.append(
                        AgentCapabilityDefinition(
                            capability_id=capability_id,
                            name=name,
                            description=description,
                            executor_id=self.executor_id,
                            input_schema={"type": "object", "required": ["task"], "properties": {"task": {"type": "string"}}},
                            output_schema={"type": "object"},
                            kind="agent",
                            always_available=True,
                            allowed_source_types=frozenset({"agent_chat"}),
                        )
                    )
                return definitions

            def call(self, task: AgentTask, context: AgentRuntimeContext) -> StandardAgentResult:
                self.calls.append({"task": task, "context": context})
                return StandardAgentResult(
                    status="succeeded",
                    summary=f"已完成 {task.capability_id} 委派任务。",
                    observation=f"child-result:{task.capability_id}",
                )

        test_case = self

        class FakeMainLLM:
            def __init__(self) -> None:
                self.calls = []

            def complete(self, *, messages, tools=None, tool_choice=None):
                self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
                if len(self.calls) % 2 == 1:
                    tool_names = [tool["function"]["name"] for tool in tools or []]
                    test_case.assertIn("agent_dbx_readonly", tool_names)
                    test_case.assertIn("agent_google_chrome", tool_names)
                    test_case.assertFalse(any(
                        name.startswith(("mcp.chrome.", "mcp.google_chrome.", "mcp.google-chrome.", "mcp.dbx.", "mcp.dropbox."))
                        for name in tool_names
                    ))
                    capability_id = "agent_dbx_readonly" if len(self.calls) == 1 else "agent.google_chrome"
                    alias = capability_id.replace(".", "_")
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[LLMToolCall(id=f"call-{len(self.calls)}", name=alias, arguments={"task": "完成用户要求的只读任务"})],
                    )
                return LLMChatCompletion(content="已从对应子 Agent 获得结果并完成回复。")

        with self.Session() as session:
            fake_agent = FakeChildAgent()
            fake_llm = FakeMainLLM()
            dependencies = self._dependencies(
                session,
                llm_client=fake_llm,
                intent_detector=FakeNormalIntentDetector(),
            ).with_agent_runtime(executors={fake_agent.executor_id: fake_agent})
            dbx_session_id = self._session_id(session)
            chrome_session_id = self._session_id(session)

            dbx_result = run_agent_workflow(
                AgentRunCommand(session_id=dbx_session_id, user_message="请通过 DBX 查询当前数据库连接"),
                dependencies=dependencies,
            )
            chrome_result = run_agent_workflow(
                AgentRunCommand(session_id=chrome_session_id, user_message="请用 Chrome 查看当前页面标题"),
                dependencies=dependencies,
            )
            session.commit()

        self.assertEqual("llm_tool_choice_loop", dbx_result.state.response_mode)
        self.assertEqual("llm_tool_choice_loop", chrome_result.state.response_mode)
        self.assertEqual(
            ["agent.dbx_readonly", "agent.google_chrome"],
            [call["task"].capability_id for call in fake_agent.calls],
        )
        self.assertEqual("openai-sdk-agent", fake_agent.calls[0]["context"].namespace)
        self.assertEqual("openai-sdk-agent", fake_agent.calls[1]["context"].namespace)

    def test_dbx_child_agent_does_not_require_main_database_session(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentCapabilityDefinition, AgentRuntimeContext, AgentTask, StandardAgentResult
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.domains.automation.models import ToolCallLog, ToolCallStatus

        class FakeChildAgent:
            executor_id = "openai-sdk-agent"

            def __init__(self) -> None:
                self.calls = []

            def capabilities(self):
                return [
                    AgentCapabilityDefinition(
                        capability_id="agent.dbx_readonly",
                        name="DBX 只读数据库 Agent",
                        description="处理只读数据库任务。",
                        executor_id=self.executor_id,
                        input_schema={"type": "object", "required": ["task"], "properties": {"task": {"type": "string"}}},
                        output_schema={"type": "object"},
                        kind="agent",
                        always_available=True,
                        allowed_source_types=frozenset({"agent_chat"}),
                    )
                ]

            def call(self, task: AgentTask, context: AgentRuntimeContext) -> StandardAgentResult:
                self.calls.append((task, context))
                return StandardAgentResult(
                    status="succeeded",
                    summary="DBX read completed",
                    raw_result={
                        "schema_version": "offer_master.sdk_agent_result.v1",
                        "status": "succeeded",
                        "trace_summary": {"tool_call_count": 1, "operation_refs": ["tool:dbx_list_connections#dbx-call-1"]},
                        "metadata": {"telemetry": {"subagent_name": "DbxReadOnlyAgent", "tool_call_count": 1}},
                    },
                )

        with self.Session() as session:
            session_id = self._session_id(session)
            fake_agent = FakeChildAgent()
            events = []
            dependencies = self._dependencies(session).with_agent_runtime(
                executors={fake_agent.executor_id: fake_agent}
            ).with_event_sink(events.append)
            result = run_agent_workflow(
                AgentRunCommand(
                    session_id=session_id,
                    user_message="只读查询 DBX 连接",
                    requested_tool_name="agent.dbx_readonly",
                    source_type="agent_chat",
                    tool_input={"task": "先发现可用 DBX 连接"},
                ),
                dependencies=dependencies,
            )
            session.commit()
            tool_log = session.scalars(select(ToolCallLog)).one()

        self.assertEqual(1, len(fake_agent.calls))
        self.assertEqual(ToolCallStatus.SUCCEEDED, tool_log.status)
        self.assertEqual("agent_runtime", tool_log.output_payload["execution"])
        self.assertEqual("DbxReadOnlyAgent", tool_log.output_payload["agent_runtime"]["telemetry"]["subagent_name"])
        self.assertEqual(1, tool_log.output_payload["agent_runtime"]["telemetry"]["tool_call_count"])
        self.assertEqual(["subagent_started", "subagent_finished"], [event["event_type"] for event in events if event["event_type"].startswith("subagent_")])

    @unittest.skip("Superseded: native structured tool calls own the replacement operation and path.")
    def test_tool_choice_loop_reuses_recent_file_path_for_exact_resume_name_replacement(self) -> None:
        from app.agent_runtime.agent_as_tool import FILESYSTEM_SKILL_CAPABILITY
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.tool_registry import AgentToolCandidateProfile, AgentToolDefinition, AgentToolRegistry, AgentToolRiskLevel
        from app.domains.automation.models import ApprovalRequest, ToolCallLog
        from app.domains.conversations.models import AgentMessageRole
        from app.domains.conversations.schemas import AgentMessageCreate
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        resume_path = "C:/Users/phoenix/Documents/Obsidian Vault/简历/resume.tex"
        read_payload = "\\name{刘汉卿}\n\\section{项目}原文保持不变"
        expected_write_payload = read_payload.replace("刘汉卿", "王爷")

        class FakeLLM:
            def __init__(self, test_case: AgentRuntimeGraphTest) -> None:
                self.calls = []
                self._test_case = test_case

            def complete(self, *, messages, tools=None, tool_choice=None):
                self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
                combined = "\n".join(str(message.get("content") or "") for message in messages)
                if tools and len(self.calls) == 1:
                    self._test_case.assertIn(resume_path, combined)
                    tool_names = [tool["function"]["name"] for tool in tools]
                    self._test_case.assertIn("skill_filesystem", tool_names)
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-read-resume",
                                name="skill_filesystem",
                                arguments={"path": resume_path, "old_text": "刘汉卿", "new_text": "王爷", "encoding": "utf-8"},
                            )
                        ],
                    )
                return LLMChatCompletion(content="已准备好写回，仅替换姓名。")

        read_calls = []
        write_calls = []

        def fake_read_file(_session, *, path: str, encoding: str = "auto", offset: int = 0, limit: int = 200):
            read_calls.append({"path": path, "encoding": encoding, "offset": offset, "limit": limit})
            return {"tool_name": "filesystem.read_file", "ok": True, "result": {"content": read_payload}}

        def fake_write_text(_session, *, path: str, text: str, encoding: str = "utf-8", overwrite: bool = False):
            write_calls.append({"path": path, "text": text, "encoding": encoding, "overwrite": overwrite})
            return {"tool_name": "filesystem.write_text", "ok": True, "result": {"path": path, "bytes_written": len(text)}}

        registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name="filesystem.read_file",
                    description="读取用户指定的本地文件。",
                    input_schema={
                        "type": "object",
                        "required": ["path"],
                        "properties": {
                            "path": {"type": "string"},
                            "encoding": {"type": "string"},
                            "offset": {"type": "integer"},
                            "limit": {"type": "integer"},
                        },
                        "additionalProperties": False,
                    },
                    output_schema={"type": "object"},
                    handler=fake_read_file,
                    allowed_source_types=frozenset({"agent_chat"}),
                    candidate_profile=AgentToolCandidateProfile(categories=frozenset({"filesystem_read", "filesystem_operation"})),
                ),
                AgentToolDefinition(
                    name="filesystem.write_text",
                    description="写入用户指定的本地文件。",
                    input_schema={
                        "type": "object",
                        "required": ["path", "text"],
                        "properties": {
                            "path": {"type": "string"},
                            "text": {"type": "string"},
                            "encoding": {"type": "string"},
                            "overwrite": {"type": "boolean"},
                        },
                        "additionalProperties": False,
                    },
                    output_schema={"type": "object"},
                    handler=fake_write_text,
                    risk_level=AgentToolRiskLevel.HIGH,
                    requires_confirmation=True,
                    allowed_source_types=frozenset({"agent_chat"}),
                    candidate_profile=AgentToolCandidateProfile(categories=frozenset({"filesystem_write", "filesystem_operation"})),
                ),
            ]
        )

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(session, llm_client=FakeLLM(self)).with_registry(registry)
            dependencies.conversation_service.append_message(
                session_id,
                AgentMessageCreate(
                    role=AgentMessageRole.USER,
                    content_text=f"这是简历路径：{resume_path}",
                    visible_content_text=f"这是简历路径：{resume_path}",
                    token_estimate=12,
                ),
            )

            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="你把简历的名字给我换了，其他的啥都不要动"),
                dependencies=dependencies,
            )
            session.commit()
            tool_logs = list(session.scalars(select(ToolCallLog)).all())
            approval = session.scalars(select(ApprovalRequest)).one()

        self.assertEqual("wait_confirmation", result.state.current_step)
        self.assertEqual(FILESYSTEM_SKILL_CAPABILITY, result.state.requested_tool_name)
        self.assertEqual([], read_calls)
        self.assertEqual([], write_calls)
        self.assertEqual([], tool_logs)
        self.assertEqual("replace_text", approval.payload["approval_payload"]["operation"])
        self.assertEqual("刘汉卿", approval.payload["approval_payload"]["old_text"])
        self.assertEqual("王爷", approval.payload["approval_payload"]["new_text"])
        self.assertEqual(resume_path, approval.payload["approval_payload"]["path"])

    def test_tool_choice_loop_reuses_recent_file_path_for_short_read_content_followup(self) -> None:
        from app.agent_runtime.agent_as_tool import FILESYSTEM_SKILL_CAPABILITY, FILESYSTEM_SKILL_EXECUTOR_ID
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.skills.filesystem_executor import FilesystemSkillExecutor
        from app.agent_runtime.tool_registry import AgentToolCandidateProfile, AgentToolDefinition, AgentToolRegistry
        from app.domains.automation.models import ToolCallLog, ToolCallStatus
        from app.domains.conversations.models import AgentMessageRole
        from app.domains.conversations.schemas import AgentMessageCreate
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        resume_path = "C:/Users/phoenix/Documents/Obsidian Vault/简历/resume.tex"
        read_payload = "姓名：刘汉卿\n项目：OfferMaster Agent Loop"
        _write_read_script(self.skill_root, read_payload)

        class FakeLLM:
            def __init__(self, test_case: AgentRuntimeGraphTest) -> None:
                self.calls = []
                self._test_case = test_case

            def complete(self, *, messages, tools=None, tool_choice=None):
                self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
                combined = "\n".join(str(message.get("content") or "") for message in messages)
                if tools and len(self.calls) == 1:
                    self._test_case.assertIn(resume_path, combined)
                    tool_names = [tool["function"]["name"] for tool in tools]
                    self._test_case.assertIn("skill_filesystem", tool_names)
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-read-resume-content",
                                name="skill_filesystem",
                                arguments={"operation": "read_file", "path": resume_path, "encoding": "utf-8", "limit": 500},
                            )
                        ],
                    )
                return LLMChatCompletion(content=f"读取到文件内容：\n{read_payload}")

        read_calls = []

        def fake_read_file(_session, *, path: str, encoding: str = "auto", offset: int = 0, limit: int = 200):
            read_calls.append({"path": path, "encoding": encoding, "offset": offset, "limit": limit})
            return {"tool_name": "filesystem.read_file", "ok": True, "result": {"content": read_payload}}

        registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name="filesystem.path_exists",
                    description="检查用户指定的本地路径是否存在。",
                    input_schema={
                        "type": "object",
                        "required": ["path"],
                        "properties": {"path": {"type": "string"}},
                        "additionalProperties": False,
                    },
                    output_schema={"type": "object"},
                    handler=lambda _session, *, path: {"tool_name": "filesystem.path_exists", "ok": True, "result": {"exists": True}},
                    allowed_source_types=frozenset({"agent_chat"}),
                    candidate_profile=AgentToolCandidateProfile(categories=frozenset({"filesystem_stat", "filesystem_operation"})),
                ),
                AgentToolDefinition(
                    name="filesystem.read_file",
                    description="读取用户指定的本地文件内容。",
                    input_schema={
                        "type": "object",
                        "required": ["path"],
                        "properties": {
                            "path": {"type": "string"},
                            "encoding": {"type": "string"},
                            "offset": {"type": "integer"},
                            "limit": {"type": "integer"},
                        },
                        "additionalProperties": False,
                    },
                    output_schema={"type": "object"},
                    handler=fake_read_file,
                    allowed_source_types=frozenset({"agent_chat"}),
                    candidate_profile=AgentToolCandidateProfile(categories=frozenset({"filesystem_read", "filesystem_operation"})),
                ),
            ]
        )

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(session, llm_client=FakeLLM(self)).with_registry(registry).with_agent_runtime(
                executors={FILESYSTEM_SKILL_EXECUTOR_ID: FilesystemSkillExecutor(script_root=self.skill_root, session_provider=lambda: session)}
            )
            dependencies.conversation_service.append_message(
                session_id,
                AgentMessageCreate(
                    role=AgentMessageRole.USER,
                    content_text=f"这是简历路径：{resume_path}",
                    visible_content_text=f"这是简历路径：{resume_path}",
                    token_estimate=12,
                ),
            )

            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="读取内容"),
                dependencies=dependencies,
            )
            session.commit()
            tool_log = session.scalars(select(ToolCallLog)).one()

        self.assertEqual(FILESYSTEM_SKILL_CAPABILITY, result.state.requested_tool_name)
        self.assertEqual("llm_tool_choice_loop", result.state.response_mode)
        self.assertIn("姓名：刘汉卿", result.state.final_response)
        self.assertEqual([], read_calls)
        self.assertEqual(ToolCallStatus.SUCCEEDED, tool_log.status)
        self.assertEqual(FILESYSTEM_SKILL_CAPABILITY, tool_log.tool_name)
        self.assertEqual("read_file.py", tool_log.output_payload["result"]["internal_script"])
        self.assertEqual(resume_path, tool_log.output_payload["result"]["arguments"]["path"])

    def test_capability_routing_executes_filesystem_path_exists_through_skill(self) -> None:
        from app.agent_runtime.agent_as_tool import FILESYSTEM_SKILL_CAPABILITY, FILESYSTEM_SKILL_EXECUTOR_ID
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.routing.capability_routing_middleware import CapabilityRoutingMiddleware
        from app.agent_runtime.skills.filesystem_executor import FilesystemSkillExecutor
        from app.domains.automation.models import ToolCallLog, ToolCallStatus
        from app.domains.conversations.models import AgentMessageRole
        from app.domains.conversations.schemas import AgentMessageCreate
        from app.domains.conversations.models import AgentMessageRole
        from app.domains.conversations.schemas import AgentMessageCreate

        resume_path = "C:/Users/phoenix/Documents/Obsidian Vault/简历/resume.tex"
        _write_path_exists_script(self.skill_root)

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(
                session,
                capability_routing_middleware=CapabilityRoutingMiddleware(),
            ).with_agent_runtime(
                executors={FILESYSTEM_SKILL_EXECUTOR_ID: FilesystemSkillExecutor(script_root=self.skill_root, session_provider=lambda: session)}
            )

            result = run_agent_workflow(
                AgentRunCommand(
                    session_id=session_id,
                    user_message=f"{resume_path} 你看下这个文件是否存在",
                    requested_tool_name=FILESYSTEM_SKILL_CAPABILITY,
                    tool_input={
                        "user_task": f"{resume_path} 你看下这个文件是否存在",
                        "operation": "path_exists",
                        "path": resume_path,
                    },
                ),
                dependencies=dependencies,
            )
            session.commit()
            tool_log = session.scalars(select(ToolCallLog)).one()

        self.assertEqual(FILESYSTEM_SKILL_CAPABILITY, result.state.requested_tool_name)
        self.assertEqual("final_response", result.state.current_step)
        self.assertEqual(ToolCallStatus.SUCCEEDED, tool_log.status)
        self.assertEqual(FILESYSTEM_SKILL_CAPABILITY, tool_log.tool_name)
        self.assertEqual("path_exists", tool_log.output_payload["result"]["operation"])
        self.assertEqual("path_exists.py", tool_log.output_payload["result"]["internal_script"])
        self.assertTrue(tool_log.output_payload["result"]["result"]["result"]["exists"])



    def test_filesystem_unknown_operation_failure_is_not_reported_as_completed(self) -> None:
        from app.agent_runtime.agent_as_tool import (
            FILESYSTEM_SKILL_CAPABILITY,
            FILESYSTEM_SKILL_EXECUTOR_ID,
            AgentRuntimeContext,
            AgentTask,
            StandardAgentResult,
        )
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow

        class FakeFilesystemExecutor:
            def call(self, task: AgentTask, context: AgentRuntimeContext) -> StandardAgentResult:
                return StandardAgentResult(
                    status="failed",
                    summary="filesystem Skill 暂时无法判断应该执行哪个文件动作。",
                    missing_information=["filesystem_operation"],
                    raw_result={"tool_name": FILESYSTEM_SKILL_CAPABILITY, "ok": False, "operation": "unknown"},
                )

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(session).with_agent_runtime(
                executors={FILESYSTEM_SKILL_EXECUTOR_ID: FakeFilesystemExecutor()}
            )

            result = run_agent_workflow(
                AgentRunCommand(
                    session_id=session_id,
                    user_message="这个文件你处理一下",
                    requested_tool_name=FILESYSTEM_SKILL_CAPABILITY,
                ),
                dependencies=dependencies,
            )

        self.assertIn("暂时无法判断", result.state.final_response)
        self.assertNotIn("已完成", result.state.final_response)

    def test_filesystem_operation_semantic_gap_is_not_rewritten_as_missing_tool_argument(self) -> None:
        """High-level intent gaps must stay user-facing clarifications, not schema errors."""
        from app.agent_runtime.agent_as_tool import StandardAgentResult
        from app.agent_runtime.graph_factory import _missing_runtime_tool_input_fields

        agent_result = StandardAgentResult(
            status="failed",
            summary="filesystem Skill 未完成：文件动作证据不足。",
            missing_information=["filesystem_operation"],
            raw_result={
                "tool_name": "skill.filesystem",
                "ok": False,
                "operation": "unknown",
                "recoverable": True,
                "next_action": "ask_user",
            },
        )

        missing = _missing_runtime_tool_input_fields(
            agent_result=agent_result,
            result_payload=agent_result.raw_result,
        )

        self.assertEqual((), missing)

    def test_path_only_filesystem_request_asks_for_action_in_plain_language(self) -> None:
        """A bare path is an entity reference; it is not a missing schema field."""
        from app.agent_runtime.agent_as_tool import FILESYSTEM_SKILL_CAPABILITY, FILESYSTEM_SKILL_EXECUTOR_ID
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.skills.filesystem_executor import FilesystemSkillExecutor

        path = r"C:\Users\phoenix\Documents\Obsidian Vault\简历\一下.tex"

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(session).with_agent_runtime(
                executors={FILESYSTEM_SKILL_EXECUTOR_ID: FilesystemSkillExecutor(script_root=self.skill_root)}
            )

            result = run_agent_workflow(
                AgentRunCommand(
                    session_id=session_id,
                    user_message=path,
                    requested_tool_name=FILESYSTEM_SKILL_CAPABILITY,
                    tool_input={"user_task": path, "path": path},
                ),
                dependencies=dependencies,
            )

        self.assertNotIn("缺少 filesystem_operation", result.state.final_response)
        self.assertIn("结构化 operation", result.state.final_response)
        self.assertNotIn("已完成", result.state.final_response)

    def test_filesystem_no_dead_end_failure_is_summarized_as_unfinished(self) -> None:
        from app.agent_runtime.agent_as_tool import (
            FILESYSTEM_SKILL_CAPABILITY,
            FILESYSTEM_SKILL_EXECUTOR_ID,
            AgentRuntimeContext,
            AgentTask,
            StandardAgentResult,
        )
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow

        class FakeFilesystemExecutor:
            def call(self, task: AgentTask, context: AgentRuntimeContext) -> StandardAgentResult:
                return StandardAgentResult(
                    status="failed",
                    summary="filesystem Skill 未完成：文件动作证据不足。下一步：向用户澄清要执行的具体文件动作。",
                    missing_information=["filesystem_operation"],
                    raw_result={
                        "tool_name": FILESYSTEM_SKILL_CAPABILITY,
                        "ok": False,
                        "operation": "unknown",
                        "recoverable": True,
                        "next_action": "ask_user",
                        "summary": "filesystem Skill 未完成：文件动作证据不足。下一步：向用户澄清要执行的具体文件动作。",
                    },
                )

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(session).with_agent_runtime(
                executors={FILESYSTEM_SKILL_EXECUTOR_ID: FakeFilesystemExecutor()}
            )

            result = run_agent_workflow(
                AgentRunCommand(
                    session_id=session_id,
                    user_message="这个文件你处理一下",
                    requested_tool_name=FILESYSTEM_SKILL_CAPABILITY,
                ),
                dependencies=dependencies,
            )

        self.assertIn("未完成", result.state.final_response)
        self.assertIn("下一步", result.state.final_response)
        self.assertNotIn("已完成", result.state.final_response)

    def test_read_before_write_recovery_wins_over_goal_validation_ask_user(self) -> None:
        from app.agent_runtime.agent_as_tool import (
            FILESYSTEM_SKILL_CAPABILITY,
            FILESYSTEM_SKILL_EXECUTOR_ID,
            AgentRuntimeContext,
            AgentTask,
            StandardAgentResult,
        )
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow

        source_path = "C:/tmp/e2e-source.tex"
        calls: list[str] = []

        class FakeFilesystemExecutor:
            def call(self, task: AgentTask, context: AgentRuntimeContext) -> StandardAgentResult:
                operation = str(task.input_payload.get("operation") or "")
                calls.append(operation)
                if operation == "read_file":
                    return StandardAgentResult(
                        status="succeeded",
                        summary="已读取文件内容。",
                        raw_result={
                            "tool_name": FILESYSTEM_SKILL_CAPABILITY,
                            "ok": True,
                            "operation": "read_file",
                            "arguments": {"path": source_path},
                            "result": {
                                "tool_name": "filesystem.read_file",
                                "ok": True,
                                "result": {"content": "Name: Liu Hanqing\nFocus: Java backend"},
                            },
                        },
                    )
                return StandardAgentResult(
                    status="failed",
                    summary="filesystem Skill 未完成：需要先读取文件内容。",
                    missing_information=["dst"],
                    requires_user_action=False,
                    raw_result={
                        "tool_name": FILESYSTEM_SKILL_CAPABILITY,
                        "ok": False,
                        "operation": "copy_file",
                        "error": "缺少目标文件名",
                        "error_code": "CONTENT_BASED_NAME_REQUIRES_READ",
                        "next_action": "read_before_write",
                        "recoverable": True,
                        "summary": "filesystem Skill 未完成：需要先读取文件内容。",
                        "missing_args": ["dst"],
                        "read_before_write": {
                            "required": True,
                            "operation": "read_file",
                            "path": source_path,
                        },
                    },
                )

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(session).with_agent_runtime(
                executors={FILESYSTEM_SKILL_EXECUTOR_ID: FakeFilesystemExecutor()}
            )
            result = run_agent_workflow(
                AgentRunCommand(
                    session_id=session_id,
                    user_message="复制一下这个文件，根据文件内容你自己起一个合适的名字。",
                    requested_tool_name=FILESYSTEM_SKILL_CAPABILITY,
                    tool_input={
                        "operation": "copy_file",
                        "src": source_path,
                        "operation_intent": {
                            "destination": {"kind": "directory", "path": "C:/tmp", "reference": "same_directory"},
                            "name_policy": "content_based",
                            "user_delegated_name": True,
                        },
                    },
                ),
                dependencies=dependencies,
            )

        self.assertEqual(["copy_file", "read_file"], calls)
        self.assertNotEqual("wait_user_input", result.state.current_step)
        self.assertIn("Name: Liu Hanqing", result.state.final_response)

    def test_filesystem_skill_stays_on_main_runtime_when_sdk_advertises_same_capability(self) -> None:
        from app.agent_runtime.agent_as_tool import FILESYSTEM_SKILL_CAPABILITY, FILESYSTEM_SKILL_EXECUTOR_ID
        from app.agent_runtime.graph_factory import (
            AgentRunCommand,
            _agent_runtime_executor_id,
            _runtime_capability_executor_ids,
        )

        with self.Session() as session:
            dependencies = self._dependencies(session).with_agent_runtime(
                capability_executor_ids={FILESYSTEM_SKILL_CAPABILITY: "openai-sdk-agent"}
            )
            command = AgentRunCommand(
                session_id="session-1",
                user_message="复制一个本地文件",
                requested_tool_name=FILESYSTEM_SKILL_CAPABILITY,
            )

            self.assertEqual(
                FILESYSTEM_SKILL_EXECUTOR_ID,
                _runtime_capability_executor_ids(dependencies)[FILESYSTEM_SKILL_CAPABILITY],
            )
            self.assertEqual(
                FILESYSTEM_SKILL_EXECUTOR_ID,
                _agent_runtime_executor_id(command, dependencies=dependencies),
            )

    def test_read_before_write_recovery_grounds_model_before_copy_approval(self) -> None:
        from app.agent_runtime.agent_as_tool import FILESYSTEM_SKILL_CAPABILITY, FILESYSTEM_SKILL_EXECUTOR_ID
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.skills.filesystem_executor import FilesystemSkillExecutor
        from app.domains.automation.models import ApprovalRequest
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        source_path = self.skill_root / "e2e-source.tex"
        expected_dst = self.skill_root / "LiuHanqing_JavaBackend_OfferMasterLoop.tex"
        source_path.write_text("Name: Liu Hanqing\nFocus: Java backend\nProject: OfferMaster Agent Loop", encoding="utf-8")
        _write_read_script(self.skill_root, source_path.read_text(encoding="utf-8"))
        _write_copy_script(self.skill_root)

        test_case = self

        class FakeLLM:
            def __init__(self, test_case: AgentRuntimeGraphTest) -> None:
                self.calls = []
                self._test_case = test_case

            def complete(self, *, messages, tools=None, tool_choice=None):
                self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
                combined = "\n".join(str(message.get("content") or "") for message in messages)
                self._test_case.assertIsNotNone(tools)
                self._test_case.assertIn("Name: Liu Hanqing", combined)
                self._test_case.assertIn("Focus: Java backend", combined)
                self._test_case.assertIn("skill_filesystem", [tool["function"]["name"] for tool in tools])
                return LLMChatCompletion(
                    content="",
                    tool_calls=[
                        LLMToolCall(
                            id="call-copy-after-read",
                            name="skill_filesystem",
                            arguments={
                                "operation": "copy_file",
                                "src": str(source_path),
                                "operation_intent": {
                                    "destination": {
                                        "kind": "file",
                                        "path": str(expected_dst),
                                        "reference": "model_selected_from_content",
                                    },
                                    "name_policy": "content_based",
                                    "user_delegated_name": True,
                                    "name_intent": {
                                        "mode": "model_proposed",
                                        "filename_stem": expected_dst.stem,
                                        "source_basis": "file_content",
                                    },
                                },
                            },
                        )
                    ],
                )

        with self.Session() as session:
            session_id = self._session_id(session)
            fake_llm = FakeLLM(self)
            dependencies = self._dependencies(session, llm_client=fake_llm).with_agent_runtime(
                executors={
                    FILESYSTEM_SKILL_EXECUTOR_ID: FilesystemSkillExecutor(
                        script_root=self.skill_root,
                        session_provider=lambda: session,
                    )
                }
            )
            result = run_agent_workflow(
                AgentRunCommand(
                    session_id=session_id,
                    user_message="复制一下这个文件，根据文件内容你自己起一个合适的名字。",
                    requested_tool_name=FILESYSTEM_SKILL_CAPABILITY,
                    tool_input={
                        "operation": "copy_file",
                        "src": str(source_path),
                        "operation_intent": {
                            "destination": {"kind": "directory", "path": str(self.skill_root), "reference": "same_directory"},
                            "name_policy": "content_based",
                            "user_delegated_name": True,
                        },
                    },
                ),
                dependencies=dependencies,
            )
            session.commit()
            approval = session.query(ApprovalRequest).one()

        self.assertEqual("wait_confirmation", result.state.current_step)
        self.assertEqual(1, len(fake_llm.calls))
        self.assertEqual(FILESYSTEM_SKILL_CAPABILITY, approval.payload["requested_tool_name"])
        self.assertEqual(str(expected_dst), approval.payload["approval_payload"]["dst"])
        self.assertFalse(expected_dst.exists())

    def test_filesystem_read_then_model_final_cannot_end_before_write_call(self) -> None:
        """A content-based copy must keep the native loop alive after a read observation."""
        from app.agent_runtime.agent_as_tool import FILESYSTEM_SKILL_CAPABILITY, FILESYSTEM_SKILL_EXECUTOR_ID
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.skills.filesystem_executor import FilesystemSkillExecutor
        from app.domains.automation.models import ApprovalRequest
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        source_path = self.skill_root / "premature-final-source.tex"
        expected_dst = self.skill_root / "LiuHanqing_AI_Agent_Resume.tex"
        source_path.write_text("Name: Liu Hanqing\\nFocus: AI Agent", encoding="utf-8")
        _write_read_script(self.skill_root, source_path.read_text(encoding="utf-8"))
        _write_copy_script(self.skill_root)

        test_case = self

        class FakeLLM:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, *, messages, tools=None, tool_choice=None):
                self.calls += 1
                combined = "\\n".join(str(message.get("content") or "") for message in messages)
                test_case.assertIn("skill_filesystem", [tool["function"]["name"] for tool in tools or []])
                if self.calls == 1:
                    test_case.assertIn("Name: Liu Hanqing", combined)
                    return LLMChatCompletion(content="我建议命名为 LiuHanqing_AI_Agent_Resume.tex，等待确认。")
                test_case.assertIn("运行时阻止了本轮提前结束", combined)
                return LLMChatCompletion(
                    content="",
                    tool_calls=[
                        LLMToolCall(
                            id="call-copy-after-runtime-guard",
                            name="skill_filesystem",
                            arguments={
                                "operation": "copy_file",
                                "src": str(source_path),
                                "operation_intent": {
                                    "destination": {"kind": "directory", "path": str(test_case.skill_root)},
                                    "name_policy": "content_based",
                                    "user_delegated_name": True,
                                    "name_intent": {
                                        "mode": "model_proposed",
                                        "filename_stem": expected_dst.stem,
                                        "source_basis": "file_content",
                                    },
                                },
                            },
                        )
                    ],
                )

        with self.Session() as session:
            session_id = self._session_id(session)
            fake_llm = FakeLLM()
            dependencies = self._dependencies(session, llm_client=fake_llm).with_agent_runtime(
                executors={
                    FILESYSTEM_SKILL_EXECUTOR_ID: FilesystemSkillExecutor(
                        script_root=self.skill_root,
                        session_provider=lambda: session,
                    )
                }
            )
            result = run_agent_workflow(
                AgentRunCommand(
                    session_id=session_id,
                    user_message="复制一下这个文件，根据文件内容你自己起一个合适的名字。",
                    requested_tool_name=FILESYSTEM_SKILL_CAPABILITY,
                    source_type="agent_chat",
                    tool_input={
                        "operation": "copy_file",
                        "src": str(source_path),
                        "operation_intent": {
                            "destination": {"kind": "directory", "path": str(self.skill_root), "reference": "same_directory"},
                            "name_policy": "content_based",
                            "user_delegated_name": True,
                        },
                    },
                ),
                dependencies=dependencies,
            )
            session.commit()
            approval = session.query(ApprovalRequest).one()

        self.assertEqual(2, fake_llm.calls)
        self.assertEqual("wait_confirmation", result.state.current_step)
        self.assertEqual(str(expected_dst), approval.payload["approval_payload"]["dst"])
        self.assertFalse(expected_dst.exists())

    def test_coarse_route_read_after_missing_operation_still_requires_write_call(self) -> None:
        """A coarse skill miss must not let a later read end a delegated copy task."""
        from app.agent_runtime.agent_as_tool import FILESYSTEM_SKILL_CAPABILITY, FILESYSTEM_SKILL_EXECUTOR_ID
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.skills.filesystem_executor import FilesystemSkillExecutor
        from app.domains.automation.models import ApprovalRequest
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        source_path = self.skill_root / "coarse-premature-final-source.tex"
        expected_dst = self.skill_root / "LiuHanqing_AI_Agent_Resume.tex"
        source_path.write_text("Name: Liu Hanqing\\nFocus: AI Agent", encoding="utf-8")
        _write_read_script(self.skill_root, source_path.read_text(encoding="utf-8"))
        _write_copy_script(self.skill_root)

        test_case = self

        class FakeLLM:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, *, messages, tools=None, tool_choice=None):
                self.calls += 1
                test_case.assertIn("skill_filesystem", [tool["function"]["name"] for tool in tools or []])
                combined = "\\n".join(str(message.get("content") or "") for message in messages)
                if self.calls == 1:
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-missing-operation",
                                name="skill_filesystem",
                                arguments={"user_task": "复制文件并根据内容起名", "path": str(source_path)},
                            )
                        ],
                    )
                if self.calls == 2:
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-read-after-missing-operation",
                                name="skill_filesystem",
                                arguments={"operation": "read_file", "path": str(source_path)},
                            )
                        ],
                    )
                if self.calls == 3:
                    test_case.assertIn("运行时阻止了本轮提前结束", combined)
                    return LLMChatCompletion(content="我建议命名为 LiuHanqing_AI_Agent_Resume.tex，等待确认。")
                test_case.assertIn("运行时阻止了本轮提前结束", combined)
                return LLMChatCompletion(
                    content="",
                    tool_calls=[
                        LLMToolCall(
                            id="call-copy-after-coarse-guard",
                            name="skill_filesystem",
                            arguments={
                                "operation": "copy_file",
                                "src": str(source_path),
                                "operation_intent": {
                                "destination": {"kind": "directory", "path": str(test_case.skill_root)},
                                    "name_policy": "content_based",
                                    "user_delegated_name": True,
                                    "name_intent": {
                                        "mode": "model_proposed",
                                        "filename_stem": expected_dst.stem,
                                        "source_basis": "file_content",
                                    },
                                },
                            },
                        )
                    ],
                )

        with self.Session() as session:
            session_id = self._session_id(session)
            fake_llm = FakeLLM()
            dependencies = self._dependencies(session, llm_client=fake_llm).with_agent_runtime(
                executors={
                    FILESYSTEM_SKILL_EXECUTOR_ID: FilesystemSkillExecutor(
                        script_root=self.skill_root,
                        session_provider=lambda: session,
                    )
                }
            )
            result = run_agent_workflow(
                AgentRunCommand(
                    session_id=session_id,
                    user_message=f"复制这个文件 {source_path}，读取后根据内容自己起一个合适的名字",
                ),
                dependencies=dependencies,
            )
            session.commit()
            approval = session.query(ApprovalRequest).one()

        self.assertEqual(4, fake_llm.calls)
        self.assertEqual("wait_confirmation", result.state.current_step)
        self.assertEqual(str(expected_dst), approval.payload["approval_payload"]["dst"])
        self.assertFalse(expected_dst.exists())

    def test_coarse_filesystem_route_returns_to_model_for_operation_and_name(self) -> None:
        from app.agent_runtime.agent_as_tool import FILESYSTEM_SKILL_CAPABILITY, FILESYSTEM_SKILL_EXECUTOR_ID
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.skills.filesystem_executor import FilesystemSkillExecutor
        from app.domains.automation.models import ApprovalRequest
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        source_path = self.skill_root / "coarse-route-source.tex"
        expected_dst = self.skill_root / "LiuHanqing_AI_Agent_Resume.tex"
        source_path.write_text("Name: Liu Hanqing\nFocus: AI Agent", encoding="utf-8")
        _write_read_script(self.skill_root, source_path.read_text(encoding="utf-8"))
        _write_copy_script(self.skill_root)

        test_case = self

        class FakeLLM:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, *, messages, tools=None, tool_choice=None):
                self.calls += 1
                test_case.assertIn("skill_filesystem", [tool["function"]["name"] for tool in tools or []])
                combined = "\n".join(str(message.get("content") or "") for message in messages)
                if self.calls == 1:
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-read-after-coarse-route",
                                name="skill_filesystem",
                                arguments={"operation": "read_file", "path": str(source_path)},
                            )
                        ],
                    )
                test_case.assertIn("Name: Liu Hanqing", combined)
                return LLMChatCompletion(
                    content="",
                    tool_calls=[
                        LLMToolCall(
                            id="call-copy-after-model-read",
                            name="skill_filesystem",
                            arguments={
                                "operation": "copy_file",
                                "src": str(source_path),
                                "operation_intent": {
                                    "destination": {"kind": "directory", "path": str(test_case.skill_root)},
                                    "name_policy": "content_based",
                                    "user_delegated_name": True,
                                    "name_intent": {
                                        "mode": "model_proposed",
                                        "filename_stem": "LiuHanqing_AI_Agent_Resume",
                                        "source_basis": "file_content",
                                    },
                                },
                            },
                        )
                    ],
                )

        fake_llm = FakeLLM()
        with self.Session() as session:
            session_id = self._session_id(session)
            from app.agent_runtime.routing.capability_routing_middleware import CapabilityRoutingMiddleware

            dependencies = self._dependencies(
                session,
                llm_client=fake_llm,
                capability_routing_middleware=CapabilityRoutingMiddleware(),
            ).with_agent_runtime(
                executors={
                    FILESYSTEM_SKILL_EXECUTOR_ID: FilesystemSkillExecutor(
                        script_root=self.skill_root,
                        session_provider=lambda: session,
                    )
                }
            )
            result = run_agent_workflow(
                AgentRunCommand(
                    session_id=session_id,
                    user_message=f"复制这个文件 {source_path}，读取后根据内容自己起一个合适的名字",
                ),
                dependencies=dependencies,
            )
            session.commit()
            approval = session.query(ApprovalRequest).one()

        self.assertEqual(2, fake_llm.calls)
        self.assertEqual("wait_confirmation", result.state.current_step)
        self.assertEqual(str(expected_dst), approval.payload["approval_payload"]["dst"])
        self.assertFalse(expected_dst.exists())



    def test_filesystem_read_result_is_observation_for_document_analysis_requests(self) -> None:
        """read_file should feed final synthesis, not directly become the chat answer."""
        from app.agent_runtime.agent_as_tool import FILESYSTEM_SKILL_CAPABILITY, FILESYSTEM_SKILL_EXECUTOR_ID
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.skills.filesystem_executor import FilesystemSkillExecutor
        from app.domains.automation.models import ToolCallLog, ToolCallStatus
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        resume_path = "C:/Users/phoenix/Documents/Obsidian Vault/简历/resume.tex"
        read_payload = "\\documentclass{article}\n\\section{项目经历}OfferMaster Agent Runtime\n\\section{技能}Python, FastAPI"
        expected_answer = "这份简历的后端匹配点主要是 FastAPI、Python 和 Agent Runtime 项目经验。"
        _write_read_script(self.skill_root, read_payload)

        class FakeAnalysisLLM:
            def __init__(self, test_case: AgentRuntimeGraphTest) -> None:
                self.calls = []
                self._test_case = test_case

            def complete(self, *, messages, tools=None, tool_choice=None):
                self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
                self._test_case.assertIsNotNone(messages)
                return LLMChatCompletion(content=expected_answer)

        with self.Session() as session:
            session_id = self._session_id(session)
            fake_llm = FakeAnalysisLLM(self)
            dependencies = self._dependencies(session, llm_client=fake_llm).with_agent_runtime(
                executors={FILESYSTEM_SKILL_EXECUTOR_ID: FilesystemSkillExecutor(script_root=self.skill_root, session_provider=lambda: session)}
            )

            result = run_agent_workflow(
                AgentRunCommand(
                    session_id=session_id,
                    user_message="分析这个文件内容哪里适合 Java 后端岗位",
                    requested_tool_name=FILESYSTEM_SKILL_CAPABILITY,
                    source_type="agent_chat",
                    tool_input={"operation": "read_file", "path": resume_path, "encoding": "utf-8", "limit": 500},
                ),
                dependencies=dependencies,
            )
            session.commit()
            tool_log = session.scalars(select(ToolCallLog)).one()

        self.assertEqual(ToolCallStatus.SUCCEEDED, tool_log.status)
        self.assertEqual("read_file", tool_log.output_payload["result"]["operation"])
        self.assertIn(result.state.response_mode, {"llm_tool_observation_final_answer", "llm_filesystem_answer_synthesis"})
        self.assertEqual(expected_answer, result.state.final_response)
        self.assertNotIn("\\documentclass", result.state.final_response)
        self.assertEqual(1, len(fake_llm.calls))


    def test_tool_choice_loop_reuses_recent_file_path_for_pronoun_open_followup(self) -> None:
        from app.agent_runtime.agent_as_tool import FILESYSTEM_SKILL_CAPABILITY, FILESYSTEM_SKILL_EXECUTOR_ID
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.skills.filesystem_executor import FilesystemSkillExecutor
        from app.agent_runtime.tool_registry import AgentToolCandidateProfile, AgentToolDefinition, AgentToolRegistry
        from app.domains.automation.models import ToolCallLog, ToolCallStatus
        from app.domains.conversations.models import AgentMessageRole
        from app.domains.conversations.schemas import AgentMessageCreate
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        resume_path = "C:/Users/phoenix/Documents/Obsidian Vault/简历/resume.tex"
        read_payload = "姓名：刘汉卿\n项目：OfferMaster Agent Loop"
        _write_read_script(self.skill_root, read_payload)

        class FakeLLM:
            def __init__(self, test_case: AgentRuntimeGraphTest) -> None:
                self.calls = []
                self._test_case = test_case

            def complete(self, *, messages, tools=None, tool_choice=None):
                self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
                combined = "\n".join(str(message.get("content") or "") for message in messages)
                if tools and len(self.calls) == 1:
                    self._test_case.assertIn(resume_path, combined)
                    self._test_case.assertIn("skill_filesystem", [tool["function"]["name"] for tool in tools])
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-open-resume-pronoun",
                                name="skill_filesystem",
                                arguments={"operation": "read_file", "path": resume_path, "encoding": "utf-8", "limit": 500},
                            )
                        ],
                    )
                if tools is None and len(self.calls) == 1:  # pragma: no cover - regression guard.
                    raise AssertionError("contextual file follow-up should enter the tool choice loop")
                return LLMChatCompletion(content=f"读取到文件内容：\n{read_payload}")

        read_calls = []

        def fake_read_file(_session, *, path: str, encoding: str = "auto", offset: int = 0, limit: int = 200):
            read_calls.append({"path": path, "encoding": encoding, "offset": offset, "limit": limit})
            return {"tool_name": "filesystem.read_file", "ok": True, "result": {"content": read_payload}}

        registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name="filesystem.read_file",
                    description="读取用户指定的本地文件内容。",
                    input_schema={
                        "type": "object",
                        "required": ["path"],
                        "properties": {
                            "path": {"type": "string"},
                            "encoding": {"type": "string"},
                            "offset": {"type": "integer"},
                            "limit": {"type": "integer"},
                        },
                        "additionalProperties": False,
                    },
                    output_schema={"type": "object"},
                    handler=fake_read_file,
                    allowed_source_types=frozenset({"agent_chat"}),
                    candidate_profile=AgentToolCandidateProfile(categories=frozenset({"filesystem_read", "filesystem_operation"})),
                ),
            ]
        )

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(session, llm_client=FakeLLM(self)).with_registry(registry).with_agent_runtime(
                executors={FILESYSTEM_SKILL_EXECUTOR_ID: FilesystemSkillExecutor(script_root=self.skill_root, session_provider=lambda: session)}
            )
            dependencies.conversation_service.append_message(
                session_id,
                AgentMessageCreate(
                    role=AgentMessageRole.USER,
                    content_text=f"这是简历路径：{resume_path}",
                    visible_content_text=f"这是简历路径：{resume_path}",
                    token_estimate=12,
                ),
            )

            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="打开它"),
                dependencies=dependencies,
            )
            session.commit()
            tool_log = session.scalars(select(ToolCallLog)).one()

        self.assertEqual(FILESYSTEM_SKILL_CAPABILITY, result.state.requested_tool_name)
        self.assertEqual("llm_tool_choice_loop", result.state.response_mode)
        self.assertEqual([], read_calls)
        self.assertEqual(ToolCallStatus.SUCCEEDED, tool_log.status)
        self.assertEqual("read_file.py", tool_log.output_payload["result"]["internal_script"])

    def test_tool_choice_loop_completes_missing_file_path_before_execution(self) -> None:
        from app.agent_runtime.agent_as_tool import FILESYSTEM_SKILL_CAPABILITY, FILESYSTEM_SKILL_EXECUTOR_ID
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.skills.filesystem_executor import FilesystemSkillExecutor
        from app.agent_runtime.tool_registry import AgentToolCandidateProfile, AgentToolDefinition, AgentToolRegistry
        from app.domains.automation.models import ToolCallLog, ToolCallStatus
        from app.domains.conversations.models import AgentMessageRole
        from app.domains.conversations.schemas import AgentMessageCreate
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        resume_path = "C:/Users/phoenix/Documents/Obsidian Vault/简历/resume.tex"
        read_payload = "姓名：刘汉卿\n项目：OfferMaster Agent Loop"
        _write_read_script(self.skill_root, read_payload)

        class FakeLLM:
            def __init__(self, test_case: AgentRuntimeGraphTest) -> None:
                self.calls = []
                self._test_case = test_case

            def complete(self, *, messages, tools=None, tool_choice=None):
                self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
                if tools and len(self.calls) == 1:
                    self._test_case.assertIn("skill_filesystem", [tool["function"]["name"] for tool in tools])
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-read-with-missing-path",
                                name="skill_filesystem",
                                arguments={"operation": "read_file", "encoding": "utf-8", "limit": 500},
                            )
                        ],
                    )
                return LLMChatCompletion(content=f"读取到文件内容：\n{read_payload}")

        read_calls = []

        def fake_read_file(_session, *, path: str, encoding: str = "auto", offset: int = 0, limit: int = 200):
            read_calls.append({"path": path, "encoding": encoding, "offset": offset, "limit": limit})
            return {"tool_name": "filesystem.read_file", "ok": True, "result": {"content": read_payload}}

        registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name="filesystem.read_file",
                    description="读取用户指定的本地文件内容。",
                    input_schema={
                        "type": "object",
                        "required": ["path"],
                        "properties": {
                            "path": {"type": "string"},
                            "encoding": {"type": "string"},
                            "offset": {"type": "integer"},
                            "limit": {"type": "integer"},
                        },
                        "additionalProperties": False,
                    },
                    output_schema={"type": "object"},
                    handler=fake_read_file,
                    allowed_source_types=frozenset({"agent_chat"}),
                    candidate_profile=AgentToolCandidateProfile(categories=frozenset({"filesystem_read", "filesystem_operation"})),
                ),
            ]
        )

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(session, llm_client=FakeLLM(self)).with_registry(registry).with_agent_runtime(
                executors={FILESYSTEM_SKILL_EXECUTOR_ID: FilesystemSkillExecutor(script_root=self.skill_root, session_provider=lambda: session)}
            )
            dependencies.conversation_service.append_message(
                session_id,
                AgentMessageCreate(
                    role=AgentMessageRole.USER,
                    content_text=f"这是简历路径：{resume_path}",
                    visible_content_text=f"这是简历路径：{resume_path}",
                    token_estimate=12,
                ),
            )

            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="读取内容"),
                dependencies=dependencies,
            )
            session.commit()
            tool_log = session.scalars(select(ToolCallLog)).one()

        self.assertEqual(FILESYSTEM_SKILL_CAPABILITY, result.state.requested_tool_name)
        self.assertEqual("llm_tool_choice_loop", result.state.response_mode)
        self.assertEqual([], read_calls)
        self.assertEqual("read_file.py", tool_log.output_payload["result"]["internal_script"])
        self.assertEqual(resume_path, tool_log.output_payload["result"]["arguments"]["path"])
        self.assertEqual(ToolCallStatus.SUCCEEDED, tool_log.status)

    @unittest.skip("Superseded: high-risk filesystem calls must provide structured operation and path fields.")
    def test_tool_choice_loop_preserves_completed_input_for_high_risk_confirmation(self) -> None:
        from app.agent_runtime.agent_as_tool import FILESYSTEM_SKILL_CAPABILITY
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.tool_registry import AgentToolCandidateProfile, AgentToolDefinition, AgentToolRegistry, AgentToolRiskLevel
        from app.domains.automation.models import ApprovalRequest, ToolCallLog
        from app.domains.conversations.models import AgentMessageRole
        from app.domains.conversations.schemas import AgentMessageCreate
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        resume_path = "C:/Users/phoenix/Documents/Obsidian Vault/简历/刘汉卿-后端开发-AI-Agent平台简历.tex"

        class FakeLLM:
            def complete(self, *, messages, tools=None, tool_choice=None):
                if tools and not any(message.get("role") == "tool" for message in messages):
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-replace-missing-path",
                                name="skill_filesystem",
                                arguments={"old_text": "刘汉卿", "new_text": "王爷"},
                            )
                        ],
                    )
                return LLMChatCompletion(content="should not run")

        registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name="filesystem.replace_text",
                    description="精确替换用户指定文件中的文本。",
                    input_schema={
                        "type": "object",
                        "required": ["path", "old_text", "new_text"],
                        "properties": {
                            "path": {"type": "string"},
                            "old_text": {"type": "string"},
                            "new_text": {"type": "string"},
                            "encoding": {"type": "string"},
                        },
                        "additionalProperties": False,
                    },
                    output_schema={"type": "object"},
                    handler=lambda _session, **arguments: {"tool_name": "filesystem.replace_text", "ok": True, "result": arguments},
                    risk_level=AgentToolRiskLevel.HIGH,
                    requires_confirmation=True,
                    allowed_source_types=frozenset({"agent_chat"}),
                    candidate_profile=AgentToolCandidateProfile(categories=frozenset({"filesystem_replace", "filesystem_write", "filesystem_operation"})),
                ),
            ]
        )

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(session, llm_client=FakeLLM()).with_registry(registry)
            dependencies.conversation_service.append_message(
                session_id,
                AgentMessageCreate(
                    role=AgentMessageRole.USER,
                    content_text=f"这是简历路径：{resume_path}",
                    visible_content_text=f"这是简历路径：{resume_path}",
                    token_estimate=12,
                ),
            )

            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="把简历名字改为王爷，其他不要动"),
                dependencies=dependencies,
            )
            session.commit()
            approval = session.scalars(select(ApprovalRequest)).one()
            tool_logs = list(session.scalars(select(ToolCallLog)).all())

        self.assertEqual("wait_confirmation", result.state.current_step)
        self.assertEqual(FILESYSTEM_SKILL_CAPABILITY, result.state.requested_tool_name)
        self.assertEqual([], tool_logs)
        self.assertEqual("replace_text", approval.payload["approval_payload"]["operation"])
        self.assertEqual("刘汉卿", approval.payload["approval_payload"]["old_text"])
        self.assertEqual("王爷", approval.payload["approval_payload"]["new_text"])
        self.assertEqual(resume_path, approval.payload["approval_payload"]["path"])

    def test_tool_choice_loop_uses_active_filename_context_for_contextual_rename(self) -> None:
        """Catches regressions where a filename-focused follow-up picks read/replace instead of rename."""
        from app.agent_runtime.agent_as_tool import FILESYSTEM_SKILL_CAPABILITY, FILESYSTEM_SKILL_EXECUTOR_ID
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.tool_registry import AgentToolCandidateProfile, AgentToolDefinition, AgentToolRegistry, AgentToolRiskLevel
        from app.domains.automation.models import ApprovalRequest, ToolCallLog
        from app.domains.conversations.models import AgentMessageRole
        from app.domains.conversations.schemas import AgentMessageCreate
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        original_path = "C:/Users/phoenix/Documents/Obsidian Vault/简历/刘汉卿-后端开发-AI-Agent平台简历.tex"
        expected_dst = "C:/Users/phoenix/Documents/Obsidian Vault/简历/刘汉卿-后端开发-AI-Agent.tex"

        class FakeLLM:
            def __init__(self, test_case: AgentRuntimeGraphTest) -> None:
                self.calls = []
                self._test_case = test_case

            def complete(self, *, messages, tools=None, tool_choice=None):
                self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
                combined = "\n".join(str(message.get("content") or "") for message in messages)
                if tools and not any(message.get("role") == "tool" for message in messages):
                    self._test_case.assertIn(original_path, combined)
                    self._test_case.assertIn("skill_filesystem", [tool["function"]["name"] for tool in tools])
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-rename-active-file",
                                name="skill_filesystem",
                                arguments={"operation": "rename_file", "src": original_path, "dst": expected_dst},
                            )
                        ],
                    )
                return LLMChatCompletion(content="should not run")

        registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name="filesystem.move_file",
                    description="重命名用户当前关注的本地文件。",
                    input_schema={
                        "type": "object",
                        "required": ["src", "dst"],
                        "properties": {
                            "src": {"type": "string"},
                            "dst": {"type": "string"},
                            "overwrite": {"type": "boolean", "default": False},
                        },
                        "additionalProperties": False,
                    },
                    output_schema={"type": "object"},
                    handler=lambda _session, **arguments: {"tool_name": "filesystem.move_file", "ok": True, "result": arguments},
                    risk_level=AgentToolRiskLevel.HIGH,
                    requires_confirmation=True,
                    allowed_source_types=frozenset({"agent_chat"}),
                    candidate_profile=AgentToolCandidateProfile(categories=frozenset({"filesystem_move", "filesystem_operation"})),
                ),
            ]
        )

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(session, llm_client=FakeLLM(self)).with_registry(registry)
            dependencies.conversation_service.append_message(
                session_id,
                AgentMessageCreate(
                    role=AgentMessageRole.USER,
                    content_text=f"刚才查看这个文件的名字：{original_path}",
                    visible_content_text=f"刚才查看这个文件的名字：{original_path}",
                    token_estimate=12,
                ),
            )

            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="把名字改成刘汉卿-后端开发-AI-Agent"),
                dependencies=dependencies,
            )
            session.commit()
            approval = session.scalars(select(ApprovalRequest)).one()
            tool_logs = list(session.scalars(select(ToolCallLog)).all())

        self.assertEqual("wait_confirmation", result.state.current_step)
        self.assertEqual(FILESYSTEM_SKILL_CAPABILITY, result.state.requested_tool_name)
        self.assertEqual([], tool_logs)
        self.assertEqual(FILESYSTEM_SKILL_CAPABILITY, approval.payload["requested_tool_name"])
        self.assertEqual(FILESYSTEM_SKILL_EXECUTOR_ID, approval.payload["executor_id"])
        self.assertEqual(original_path, approval.payload["approval_payload"]["src"])
        self.assertEqual(expected_dst, approval.payload["approval_payload"]["dst"])
        self.assertFalse(approval.payload["approval_payload"]["overwrite"])
        self.assertIn(result.state.context_metadata["active_file"]["last_focus"], {"file", "filename"})

    def test_tool_choice_loop_routes_filesystem_followup_through_skill_capability(self) -> None:
        """Skill-as-Capability keeps filesystem scripts out of the main agent tool choice surface."""
        from app.agent_runtime.agent_as_tool import FILESYSTEM_SKILL_CAPABILITY, FILESYSTEM_SKILL_EXECUTOR_ID
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.domains.automation.models import ApprovalRequest, ToolCallLog
        from app.domains.conversations.models import AgentMessageRole
        from app.domains.conversations.schemas import AgentMessageCreate
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        original_path = "C:/Users/phoenix/Documents/Obsidian Vault/简历/刘汉卿-后端开发-AI-Agent平台简历.tex"
        expected_dst = "C:/Users/phoenix/Documents/Obsidian Vault/简历/刘汉卿-后端开发-AI-Agent.tex"

        class FakeLLM:
            def __init__(self, test_case: AgentRuntimeGraphTest) -> None:
                self.calls = []
                self._test_case = test_case

            def complete(self, *, messages, tools=None, tool_choice=None):
                self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
                if tools and not any(message.get("role") == "tool" for message in messages):
                    self._test_case.assertIn("skill_filesystem", [tool["function"]["name"] for tool in tools])
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-filesystem-skill",
                                name="skill_filesystem",
                                arguments={
                                    "user_task": "把名字改成刘汉卿-后端开发-AI-Agent",
                                    "operation": "rename_file",
                                    "src": original_path,
                                    "dst": expected_dst,
                                },
                            )
                        ],
                    )
                return LLMChatCompletion(content="should not run")

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(session, llm_client=FakeLLM(self))
            dependencies.conversation_service.append_message(
                session_id,
                AgentMessageCreate(
                    role=AgentMessageRole.USER,
                    content_text=f"刚才查看这个文件的名字：{original_path}",
                    visible_content_text=f"刚才查看这个文件的名字：{original_path}",
                    token_estimate=12,
                ),
            )

            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="把名字改成刘汉卿-后端开发-AI-Agent"),
                dependencies=dependencies,
            )
            session.commit()
            approval = session.scalars(select(ApprovalRequest)).one()
            tool_logs = list(session.scalars(select(ToolCallLog)).all())

        self.assertEqual("wait_confirmation", result.state.current_step)
        self.assertEqual(FILESYSTEM_SKILL_CAPABILITY, result.state.requested_tool_name)
        self.assertEqual([], tool_logs)
        self.assertEqual(FILESYSTEM_SKILL_CAPABILITY, approval.payload["requested_tool_name"])
        self.assertEqual(FILESYSTEM_SKILL_EXECUTOR_ID, approval.payload["executor_id"])
        self.assertEqual(original_path, approval.payload["tool_input"]["context_metadata"]["active_file"]["path"])
        self.assertEqual(expected_dst, approval.payload["approval_payload"]["dst"])

    def test_approved_filesystem_skill_rename_runs_internal_script_once(self) -> None:
        """Approval resumes the same high-level Skill; the main agent never calls move_file directly."""
        from app.agent_runtime.agent_as_tool import FILESYSTEM_SKILL_CAPABILITY, FILESYSTEM_SKILL_EXECUTOR_ID
        from app.agent_runtime.graph_factory import AgentRunCommand, continue_agent_workflow_after_approval, run_agent_workflow
        from app.agent_runtime.skills.filesystem_executor import FilesystemSkillExecutor
        from app.domains.automation.models import ApprovalRequest, ToolCallLog, ToolCallStatus
        from app.domains.conversations.models import AgentMessageRole
        from app.domains.conversations.schemas import AgentMessageCreate
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        _write_move_script(self.skill_root)
        src = self.skill_root / "刘汉卿-后端开发-AI-Agent平台简历.tex"
        src.write_text("resume", encoding="utf-8")
        original_path = str(src)
        expected_dst = str(self.skill_root / "刘汉卿-后端开发-AI-Agent.tex")

        class FakeLLM:
            def __init__(self) -> None:
                self.calls = []

            def complete(self, *, messages, tools=None, tool_choice=None):
                self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
                if tools and not any(message.get("role") == "tool" for message in messages):
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-filesystem-skill",
                                name="skill_filesystem",
                                arguments={"operation": "rename_file", "src": original_path, "dst": expected_dst},
                            )
                        ],
                    )
                return LLMChatCompletion(content="文件名已经按确认后的操作处理。")

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(session, llm_client=FakeLLM()).with_agent_runtime(
                executors={
                    FILESYSTEM_SKILL_EXECUTOR_ID: FilesystemSkillExecutor(
                        script_root=self.skill_root,
                        session_provider=lambda: session,
                    )
                }
            )
            dependencies.conversation_service.append_message(
                session_id,
                AgentMessageCreate(
                    role=AgentMessageRole.USER,
                    content_text=f"刚才查看这个文件的名字：{original_path}",
                    visible_content_text=f"刚才查看这个文件的名字：{original_path}",
                    token_estimate=12,
                ),
            )

            first = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="把名字改成刘汉卿-后端开发-AI-Agent"),
                dependencies=dependencies,
            )
            approval = session.scalars(select(ApprovalRequest)).one()
            continued = continue_agent_workflow_after_approval(
                approval.id,
                approved=True,
                decision_reason="确认重命名",
                dependencies=dependencies,
            )
            session.commit()
            tool_logs = list(session.scalars(select(ToolCallLog)).all())

        self.assertEqual("wait_confirmation", first.state.current_step)
        self.assertEqual(FILESYSTEM_SKILL_CAPABILITY, first.state.requested_tool_name)
        self.assertEqual("final_response", continued.state.current_step)
        self.assertEqual(FILESYSTEM_SKILL_CAPABILITY, continued.state.requested_tool_name)
        self.assertEqual([ToolCallStatus.SUCCEEDED], [log.status for log in tool_logs])
        self.assertEqual([FILESYSTEM_SKILL_CAPABILITY], [log.tool_name for log in tool_logs])
        self.assertEqual("move_file.py", tool_logs[0].output_payload["result"]["internal_script"])
        self.assertEqual(expected_dst, tool_logs[0].output_payload["result"]["arguments"]["dst"])
        self.assertFalse(src.exists())
        self.assertTrue(Path(expected_dst).exists())
        self.assertTrue(tool_logs[0].output_payload["result"]["filesystem_trace"]["postcheck"]["completed"])





    def test_runtime_result_summary_includes_filesystem_postcheck_trace(self) -> None:
        from app.agent_runtime.graph_factory import _runtime_result_summary_metadata, _runtime_result_summary_text

        payload = {
            "tool_name": "skill.filesystem",
            "ok": False,
            "operation": "rename_file",
            "error": "FILESYSTEM_POSTCHECK_FAILED: target_missing_after_rename",
            "filesystem_trace": {
                "operation": "rename_file",
                "precheck": {"source_path": "A.tex", "source_exists_before": True, "target_path": "B.tex"},
                "script": {"internal_script": "move_file.py", "ok": True},
                "postcheck": {
                    "completed": False,
                    "source_path": "A.tex",
                    "target_path": "B.tex",
                    "source_exists_after": True,
                    "target_exists_after": False,
                    "reason": "target_missing_after_rename",
                },
            },
        }

        summary = _runtime_result_summary_metadata(payload)
        text = _runtime_result_summary_text(summary)

        self.assertEqual("rename_file", summary["filesystem_trace"]["operation"])
        self.assertFalse(summary["filesystem_trace"]["postcheck"]["completed"])
        self.assertIn("复核失败", text)
        self.assertIn("A.tex -> B.tex", text)

    def test_filesystem_result_overrides_stale_prepared_generic_final_response(self) -> None:
        from app.agent_runtime.graph_factory import _generate_final_response
        from app.agent_runtime.state import AgentState

        target = r"C:\Users\phoenix\Documents\简历\刘汉卿-AI-Agent平台后端简历.tex"
        state = AgentState(
            session_id="session-final-answer",
            workflow_run_id="workflow-final-answer",
            agent_run_id="agent-final-answer",
            user_message="你自己起一个名字，合适的名字根据那个文件中的内容",
            current_step="final_response",
            requested_tool_name="skill.filesystem",
            final_response="OfferMaster",
            response_mode="llm",
            llm_messages=[
                {
                    "role": "assistant",
                    "content": "Tool result: skill.filesystem succeeded",
                    "metadata": {
                        "content_json": {
                            "tool_name": "skill.filesystem",
                            "status": "succeeded",
                            "result": {
                                "tool_name": "skill.filesystem",
                                "ok": True,
                                "operation": "rename_file",
                                "arguments": {"src": r"C:\Users\phoenix\Documents\简历\一下.tex", "dst": target},
                                "filesystem_trace": {
                                    "postcheck": {
                                        "completed": True,
                                        "source_path": r"C:\Users\phoenix\Documents\简历\一下.tex",
                                        "target_path": target,
                                    }
                                },
                            },
                        }
                    },
                }
            ],
            context_metadata={
                "goal_state": {
                    "intent": "filesystem_operation",
                    "expected_operation": "rename_file",
                    "target": {"target_name": target},
                }
            },
        )

        with self.Session() as session:
            dependencies = self._dependencies(session)
            content, _mode = _generate_final_response(state, dependencies=dependencies)

        self.assertIn(target, content)

    def test_tool_choice_loop_reuses_recent_company_context_for_pronoun_public_lookup(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL, AgentToolDefinition, AgentToolRegistry
        from app.domains.automation.models import ToolCallLog, ToolCallStatus
        from app.domains.conversations.models import AgentMessageRole
        from app.domains.conversations.schemas import AgentMessageCreate
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        class FakeLLM:
            def __init__(self, test_case: AgentRuntimeGraphTest) -> None:
                self.calls = []
                self._test_case = test_case

            def complete(self, *, messages, tools=None, tool_choice=None):
                self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
                combined = "\n".join(str(message.get("content") or "") for message in messages)
                if tools and "Canonical 是 Ubuntu 背后的公司" not in combined:
                    self._test_case.assertIn("Canonical Ltd.", combined)
                    self._test_case.assertIn("external_web_search", [tool["function"]["name"] for tool in tools])
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-pronoun-company-public-info",
                                name="external_web_search",
                                arguments={"query": "Canonical Ltd. 主要业务", "max_results": 3},
                            )
                        ],
                    )
                if tools is None and len(self.calls) == 1:  # pragma: no cover - regression guard.
                    raise AssertionError("contextual public lookup should enter the tool choice loop")
                return LLMChatCompletion(content="Canonical Ltd. 主要做 Ubuntu、企业 Linux 和云基础设施。")

        search_calls = []

        def fake_web_search(_session, *, query: str, max_results: int = 5):
            search_calls.append({"query": query, "max_results": max_results})
            return {
                "tool_name": EXTERNAL_WEB_SEARCH_TOOL,
                "ok": True,
                "result": {
                    "query": query,
                    "answer": "Canonical 是 Ubuntu 背后的公司，主要做企业 Linux 和云基础设施。",
                    "sources": [{"title": "Canonical", "url": "https://canonical.com/"}],
                },
            }

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(session, llm_client=FakeLLM(self))
            registry = AgentToolRegistry(
                definition
                for definition in dependencies.registry.list_definitions()
                if definition.name != EXTERNAL_WEB_SEARCH_TOOL
            )
            registry.register(
                AgentToolDefinition(
                    name=EXTERNAL_WEB_SEARCH_TOOL,
                    description="Search the public web through an external agent.",
                    input_schema={
                        "type": "object",
                        "required": ["query"],
                        "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
                        "additionalProperties": False,
                    },
                    output_schema={"type": "object"},
                    handler=fake_web_search,
                    allowed_source_types=frozenset({"agent_chat", "web_search"}),
                )
            )
            dependencies = dependencies.with_registry(registry)
            dependencies.conversation_service.append_message(
                session_id,
                AgentMessageCreate(
                    role=AgentMessageRole.USER,
                    content_text="我想了解 Canonical Ltd. 这个公司",
                    visible_content_text="我想了解 Canonical Ltd. 这个公司",
                    token_estimate=10,
                ),
            )

            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="查一下它主要业务"),
                dependencies=dependencies,
            )
            session.commit()
            tool_log = session.scalars(select(ToolCallLog)).one()

        self.assertEqual([{"query": "Canonical Ltd. 主要业务", "max_results": 3}], search_calls)
        self.assertEqual(EXTERNAL_WEB_SEARCH_TOOL, result.state.requested_tool_name)
        self.assertEqual("llm_tool_choice_loop", result.state.response_mode)
        self.assertEqual(ToolCallStatus.SUCCEEDED, tool_log.status)
        self.assertEqual(EXTERNAL_WEB_SEARCH_TOOL, tool_log.tool_name)

    def test_native_tool_loop_can_execute_multiple_tool_steps_before_final_answer(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL, AgentToolDefinition, AgentToolRegistry
        from app.agent_runtime.understanding.intent_detector import HybridIntentDetector
        from app.domains.automation.models import ToolCallLog, ToolCallStatus
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        class FakeIntentLLM:
            def complete(self, *, messages):
                return LLMChatCompletion(
                    content=(
                        '{"intent":"campus_recruiting_search","confidence":0.95,'
                        '"needs_external_info":true,"risk_level":"low",'
                        '"entities":{"company_names":["腾讯","京东"],"keywords":["校园招聘"],"time_range":"latest"}}'
                    )
                )

        class FakeMultiStepToolLoopLLM:
            def __init__(self) -> None:
                self.calls = []

            def complete(self, *, messages, tools=None, tool_choice=None):
                self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
                combined = "\n".join(str(message.get("content") or "") for message in messages)
                if tools and "腾讯校招官网" not in combined:
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-tencent",
                                name="external_web_search",
                                arguments={"query": "腾讯 校园招聘 官网", "max_results": 5},
                            )
                        ],
                    )
                if tools and "京东校招官网" not in combined:
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-jd",
                                name="external_web_search",
                                arguments={"query": "京东 校园招聘 官网", "max_results": 5},
                            )
                        ],
                    )
                if tools and not any(message.get("role") == "tool" for message in messages):
                    return LLMChatCompletion(content="腾讯校招官网：https://join.qq.com/；京东校招官网：https://campus.jd.com/")
                if "腾讯校招官网" in combined and "京东校招官网" in combined:
                    return LLMChatCompletion(content="腾讯校招官网：https://join.qq.com/；京东校招官网：https://campus.jd.com/")
                return LLMChatCompletion(content="premature final answer")

        search_calls = []

        def fake_web_search(_session, *, query: str, max_results: int = 5):
            search_calls.append({"query": query, "max_results": max_results})
            if "腾讯" in query:
                answer = "腾讯校招官网：https://join.qq.com/"
                source = {"title": "腾讯校招官网", "url": "https://join.qq.com/"}
            else:
                answer = "京东校招官网：https://campus.jd.com/"
                source = {"title": "京东校招官网", "url": "https://campus.jd.com/"}
            return {"tool_name": EXTERNAL_WEB_SEARCH_TOOL, "ok": True, "result": {"answer": answer, "sources": [source]}}

        with self.Session() as session:
            session_id = self._session_id(session)
            fake_llm = FakeMultiStepToolLoopLLM()
            dependencies = self._dependencies(
                session,
                llm_client=fake_llm,
                intent_detector=HybridIntentDetector(llm_client=FakeIntentLLM()),
            )
            registry = AgentToolRegistry(
                definition for definition in dependencies.registry.list_definitions() if definition.name != EXTERNAL_WEB_SEARCH_TOOL
            )
            registry.register(
                AgentToolDefinition(
                    name=EXTERNAL_WEB_SEARCH_TOOL,
                    description="Search the public web through an external agent.",
                    input_schema={
                        "type": "object",
                        "required": ["query"],
                        "properties": {
                            "query": {"type": "string"},
                            "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
                        },
                        "additionalProperties": False,
                    },
                    output_schema={"type": "object", "required": ["tool_name", "ok", "result"]},
                    handler=fake_web_search,
                    allowed_source_types=frozenset({"agent_chat", "web_search"}),
                )
            )

            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="分别查一下腾讯和京东的校园招聘官网"),
                dependencies=dependencies.with_registry(registry),
            )
            session.commit()
            tool_logs = list(session.scalars(select(ToolCallLog)).all())

        self.assertEqual(
            [
                {"query": "腾讯 校园招聘 官网", "max_results": 5},
                {"query": "京东 校园招聘 官网", "max_results": 5},
            ],
            search_calls,
        )
        self.assertEqual("llm_tool_choice_loop", result.state.response_mode)
        self.assertIn("腾讯校招官网", result.state.final_response)
        self.assertIn("京东校招官网", result.state.final_response)
        self.assertEqual([ToolCallStatus.SUCCEEDED, ToolCallStatus.SUCCEEDED], [log.status for log in tool_logs])
        self.assertEqual(3, len(fake_llm.calls))
        self.assertEqual(2, result.state.context_metadata["tool_choice_loop"]["executed_step_count"])
        loop_agent = result.state.context_metadata["tool_choice_loop"]
        self.assertTrue(loop_agent["enabled"])
        self.assertEqual("runtime_controlled", loop_agent["control_mode"])
        self.assertEqual("model_final", loop_agent["stop_reason"])
        self.assertEqual(2, loop_agent["executed_step_count"])
        self.assertEqual(["external.web_search", "external.web_search"], [step["capability"] for step in loop_agent["trace"]])
        self.assertNotIn("thought", loop_agent["trace"][0])

    def test_native_tool_loop_records_reflection_retry_when_web_search_result_is_off_target(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL, AgentToolDefinition, AgentToolRegistry
        from app.agent_runtime.understanding.intent_detector import HybridIntentDetector
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        class FakeIntentLLM:
            def complete(self, *, messages):
                return LLMChatCompletion(
                    content=(
                        '{"intent":"campus_recruiting_search","confidence":0.95,'
                        '"needs_external_info":true,"risk_level":"low",'
                        '"entities":{"company_names":["中科曙光"],"keywords":["校园招聘"],"time_range":"latest"}}'
                    )
                )

        class FakeToolLoopLLM:
            def __init__(self) -> None:
                self.calls = []

            def complete(self, *, messages, tools=None, tool_choice=None):
                self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
                if tools and not any(message.get("role") == "tool" for message in messages):
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-bad-sugon",
                                name="external_web_search",
                                arguments={"query": "中科曙光 校园招聘 官网", "max_results": 5},
                            )
                        ],
                    )
                return LLMChatCompletion(content="未找到可靠官网，建议换关键词重试。")

        search_calls = []

        def fake_web_search(_session, *, query: str, max_results: int = 5):
            search_calls.append({"query": query, "max_results": max_results})
            return {
                "tool_name": EXTERNAL_WEB_SEARCH_TOOL,
                "ok": True,
                "result": {
                    "answer": "中（汉语汉字）_百度百科",
                    "sources": [{"title": "中（汉语汉字）_百度百科", "url": "https://baike.baidu.com/item/中"}],
                },
            }

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(
                session,
                llm_client=FakeToolLoopLLM(),
                intent_detector=HybridIntentDetector(llm_client=FakeIntentLLM()),
            )
            registry = AgentToolRegistry(
                definition for definition in dependencies.registry.list_definitions() if definition.name != EXTERNAL_WEB_SEARCH_TOOL
            )
            registry.register(
                AgentToolDefinition(
                    name=EXTERNAL_WEB_SEARCH_TOOL,
                    description="Search the public web through an external agent.",
                    input_schema={
                        "type": "object",
                        "required": ["query"],
                        "properties": {
                            "query": {"type": "string"},
                            "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
                        },
                        "additionalProperties": False,
                    },
                    output_schema={"type": "object", "required": ["tool_name", "ok", "result"]},
                    handler=fake_web_search,
                    allowed_source_types=frozenset({"agent_chat", "web_search"}),
                )
            )

            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="搜一下中科曙光校园招聘官网"),
                dependencies=dependencies.with_registry(registry),
            )
            session.commit()

        self.assertEqual(
            [
                {"query": "中科曙光 校园招聘 官网", "max_results": 5},
                {"query": "中科曙光 校园招聘 官网 2026", "max_results": 5},
            ],
            search_calls,
        )
        tool_choice_loop = result.state.context_metadata["tool_choice_loop"]
        self.assertEqual(2, tool_choice_loop["executed_step_count"])
        self.assertEqual(2, len(tool_choice_loop["trace"]))
        trace_entry = tool_choice_loop["trace"][0]
        reflection = trace_entry["metadata"]["observation"]["metadata"]["reflection"]
        self.assertEqual("bad", reflection["quality"])
        self.assertEqual("retry", reflection["next_action"])
        self.assertIn("中科曙光", reflection["suggested_input_patch"]["query"])
        self.assertIn("校园招聘", reflection["suggested_input_patch"]["query"])

    def test_reflection_retry_budget_is_configurable_but_capped(self) -> None:
        from app.agent_runtime.graph_factory import _native_tool_loop_reflection_retry_budget
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL

        self.assertEqual(
            3,
            _native_tool_loop_reflection_retry_budget(
                {"loop_agent": {"reflection_retry_budget": None}},
                [EXTERNAL_WEB_SEARCH_TOOL],
            ),
        )
        self.assertEqual(
            0,
            _native_tool_loop_reflection_retry_budget(
                {"loop_agent": {"reflection_retry_budget": 0}},
                [EXTERNAL_WEB_SEARCH_TOOL],
            ),
        )
        self.assertEqual(
            3,
            _native_tool_loop_reflection_retry_budget(
                {"loop_agent": {"reflection_retry_budget": 9}},
                [EXTERNAL_WEB_SEARCH_TOOL],
            ),
        )
        self.assertEqual(
            2,
            _native_tool_loop_reflection_retry_budget(
                {"reflection_retry_budget": "2"},
                [EXTERNAL_WEB_SEARCH_TOOL],
            ),
        )
        self.assertEqual(
            0,
            _native_tool_loop_reflection_retry_budget(
                {"loop_agent": {"reflection_retry_budget": 2}},
                ["applications.find_apply_entry"],
            ),
        )

    def test_native_tool_loop_retries_web_search_with_reflection_suggested_query(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL, AgentToolDefinition, AgentToolRegistry
        from app.agent_runtime.understanding.intent_detector import HybridIntentDetector
        from app.domains.automation.models import ToolCallLog, ToolCallStatus
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        class FakeIntentLLM:
            def complete(self, *, messages):
                return LLMChatCompletion(
                    content=(
                        '{"intent":"campus_recruiting_search","confidence":0.95,'
                        '"needs_external_info":true,"risk_level":"low",'
                        '"entities":{"company_names":["中科曙光"],"keywords":["校园招聘"],"time_range":"latest"}}'
                    )
                )

        class FakeToolLoopLLM:
            def __init__(self) -> None:
                self.calls = []

            def complete(self, *, messages, tools=None, tool_choice=None):
                self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
                if tools and not any(message.get("role") == "tool" for message in messages):
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-bad-sugon",
                                name="external_web_search",
                                arguments={"query": "中科曙光 招聘", "max_results": 5},
                            )
                        ],
                    )
                combined = "\n".join(str(message.get("content") or "") for message in messages)
                self.final_messages = combined
                return LLMChatCompletion(content="已找到中科曙光校招官网：https://jobs.example.com/sugon")

        search_calls = []

        def fake_web_search(_session, *, query: str, max_results: int = 5):
            search_calls.append({"query": query, "max_results": max_results})
            if query == "中科曙光 招聘":
                return {
                    "tool_name": EXTERNAL_WEB_SEARCH_TOOL,
                    "ok": True,
                    "result": {
                        "answer": "中（汉语汉字）_百度百科",
                        "sources": [{"title": "中（汉语汉字）_百度百科", "url": "https://baike.baidu.com/item/中"}],
                    },
                }
            return {
                "tool_name": EXTERNAL_WEB_SEARCH_TOOL,
                "ok": True,
                "result": {
                    "answer": "中科曙光校园招聘官网：https://jobs.example.com/sugon",
                    "sources": [{"title": "中科曙光校园招聘官网", "url": "https://jobs.example.com/sugon"}],
                },
            }

        with self.Session() as session:
            session_id = self._session_id(session)
            fake_llm = FakeToolLoopLLM()
            dependencies = self._dependencies(
                session,
                llm_client=fake_llm,
                intent_detector=HybridIntentDetector(llm_client=FakeIntentLLM()),
            )
            registry = AgentToolRegistry(
                definition for definition in dependencies.registry.list_definitions() if definition.name != EXTERNAL_WEB_SEARCH_TOOL
            )
            registry.register(
                AgentToolDefinition(
                    name=EXTERNAL_WEB_SEARCH_TOOL,
                    description="Search the public web through an external agent.",
                    input_schema={
                        "type": "object",
                        "required": ["query"],
                        "properties": {
                            "query": {"type": "string"},
                            "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
                        },
                        "additionalProperties": False,
                    },
                    output_schema={"type": "object", "required": ["tool_name", "ok", "result"]},
                    handler=fake_web_search,
                    allowed_source_types=frozenset({"agent_chat", "web_search"}),
                )
            )

            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="搜一下中科曙光校园招聘官网"),
                dependencies=dependencies.with_registry(registry),
            )
            session.commit()
            tool_logs = list(session.scalars(select(ToolCallLog)).all())

        self.assertEqual(
            [
                {"query": "中科曙光 招聘", "max_results": 5},
                {"query": "中科曙光 校园招聘 官网 2026", "max_results": 5},
            ],
            search_calls,
        )
        self.assertEqual([ToolCallStatus.SUCCEEDED, ToolCallStatus.SUCCEEDED], [log.status for log in tool_logs])
        self.assertEqual("llm_tool_choice_loop", result.state.response_mode)
        self.assertIn("中科曙光校招官网", result.state.final_response)
        self.assertIn("中科曙光校园招聘官网", fake_llm.final_messages)
        loop_agent = result.state.context_metadata["tool_choice_loop"]
        self.assertEqual(2, loop_agent["executed_step_count"])
        self.assertEqual(
            ["retry", "continue"],
            [step["metadata"]["observation"]["metadata"]["reflection"]["next_action"] for step in loop_agent["trace"]],
        )

    def test_execution_planner_executes_capability_action_without_native_tool_call_selection(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.planning.schemas import ExecutionPlan, ExecutionPlannerAction
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL, AgentToolDefinition, AgentToolRegistry
        from app.agent_runtime.understanding.intent_detector import HybridIntentDetector
        from app.domains.automation.models import ToolCallLog, ToolCallStatus
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        class FakeIntentLLM:
            def complete(self, *, messages):
                return LLMChatCompletion(
                    content=(
                        '{"intent":"campus_recruiting_search","confidence":0.95,'
                        '"needs_external_info":true,"risk_level":"low",'
                        '"entities":{"company_names":["经纬恒润"],"keywords":["校园招聘"],"time_range":"latest"}}'
                    )
                )

        class FakeExecutionPlanner:
            def __init__(self) -> None:
                self.calls = []

            def plan(self, *, user_message, context_pack):
                self.calls.append({"user_message": user_message, "context_pack": context_pack})
                return ExecutionPlan(
                    mode="simple_tool_call",
                    confidence=0.92,
                    risk_level="low",
                    actions=[
                        ExecutionPlannerAction(
                            type="call_capability",
                            capability=EXTERNAL_WEB_SEARCH_TOOL,
                            arguments={"query": "经纬恒润 校园招聘 官网", "max_results": 5},
                            reason="需要查询公司校招入口",
                        )
                    ],
                    reason="用户要求查询最新校招信息",
                )

        test_case = self

        class FakeFinalLLM:
            def __init__(self) -> None:
                self.calls = []

            def complete(self, *, messages, tools=None, tool_choice=None):
                self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
                test_case.assertIsNotNone(tools)
                combined = "\n".join(str(message.get("content") or "") for message in messages)
                if not any(message.get("role") == "tool" for message in messages):
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-hirain-search",
                                name="external_web_search",
                                arguments={"query": "经纬恒润 校园招聘 官网", "max_results": 5},
                            )
                        ],
                    )
                test_case.assertIn("工具 external.web_search 执行状态：succeeded", combined)
                test_case.assertIn("经纬恒润校园招聘官网", combined)
                return LLMChatCompletion(content="已找到经纬恒润校园招聘官网：https://jobs.example.com/hirain")

        search_calls = []

        def fake_web_search(_session, *, query: str, max_results: int = 5):
            search_calls.append({"query": query, "max_results": max_results})
            return {
                "tool_name": EXTERNAL_WEB_SEARCH_TOOL,
                "ok": True,
                "result": {
                    "answer": "经纬恒润校园招聘官网：https://jobs.example.com/hirain",
                    "sources": [{"title": "经纬恒润校园招聘官网", "url": "https://jobs.example.com/hirain"}],
                },
            }

        with self.Session() as session:
            session_id = self._session_id(session)
            planner = FakeExecutionPlanner()
            final_llm = FakeFinalLLM()
            dependencies = self._dependencies(
                session,
                llm_client=final_llm,
                intent_detector=HybridIntentDetector(llm_client=FakeIntentLLM()),
                execution_planner=planner,
            )
            registry = AgentToolRegistry(
                definition
                for definition in dependencies.registry.list_definitions()
                if definition.name != EXTERNAL_WEB_SEARCH_TOOL
            )
            registry.register(
                AgentToolDefinition(
                    name=EXTERNAL_WEB_SEARCH_TOOL,
                    description="Search the public web through an external agent.",
                    input_schema={
                        "type": "object",
                        "required": ["query"],
                        "properties": {
                            "query": {"type": "string"},
                            "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
                        },
                        "additionalProperties": False,
                    },
                    output_schema={"type": "object", "required": ["tool_name", "ok", "result"]},
                    handler=fake_web_search,
                    allowed_source_types=frozenset({"agent_chat", "web_search"}),
                )
            )

            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="查一下经纬恒润的校园招聘信息"),
                dependencies=dependencies.with_registry(registry),
            )
            session.commit()
            tool_log = session.scalars(select(ToolCallLog)).one()

        self.assertEqual([{"query": "经纬恒润 校园招聘 官网", "max_results": 5}], search_calls)
        self.assertEqual(0, len(planner.calls))
        self.assertEqual(EXTERNAL_WEB_SEARCH_TOOL, result.state.requested_tool_name)
        self.assertEqual("llm_tool_choice_loop", result.state.response_mode)
        self.assertEqual("已找到经纬恒润校园招聘官网：https://jobs.example.com/hirain", result.state.final_response)
        self.assertNotIn("execution_plan", result.state.context_metadata)
        self.assertEqual(ToolCallStatus.SUCCEEDED, tool_log.status)
        self.assertEqual(2, len(final_llm.calls))

    def test_native_tool_loop_preserves_tool_input_through_confirmation(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, continue_agent_workflow_after_approval, run_agent_workflow
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL, AgentToolDefinition, AgentToolRegistry
        from app.agent_runtime.understanding.intent_detector import HybridIntentDetector
        from app.domains.automation.models import ApprovalRequest, ToolCallLog, ToolCallStatus, WorkflowRun, WorkflowRunStatus
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        class FakeIntentLLM:
            def complete(self, *, messages):
                return LLMChatCompletion(
                    content=(
                        '{"intent":"campus_recruiting_search","confidence":0.95,'
                        '"needs_external_info":true,"risk_level":"low",'
                        '"entities":{"company_names":["中科曙光"],"keywords":["校园招聘"],"time_range":"latest"}}'
                    )
                )

        class FakeToolLoopLLM:
            def __init__(self) -> None:
                self.calls = []

            def complete(self, *, messages, tools=None, tool_choice=None):
                self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
                if tools and not any(message.get("role") == "tool" for message in messages):
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-search-confirmed",
                                name="external_web_search",
                                arguments={"query": "中科曙光 校园招聘 秋招 官网", "max_results": 5},
                            )
                        ],
                    )
                return LLMChatCompletion(content="确认后已完成联网搜索：中科曙光校园招聘官网")

        search_calls = []

        def fake_web_search(_session, *, query: str, max_results: int = 5):
            search_calls.append({"query": query, "max_results": max_results})
            return {
                "tool_name": EXTERNAL_WEB_SEARCH_TOOL,
                "ok": True,
                "result": {"answer": "中科曙光校园招聘官网", "sources": []},
            }

        with self.Session() as session:
            session_id = self._session_id(session)
            fake_llm = FakeToolLoopLLM()
            dependencies = self._dependencies(
                session,
                llm_client=fake_llm,
                intent_detector=HybridIntentDetector(llm_client=FakeIntentLLM()),
            )
            registry = AgentToolRegistry(
                definition
                for definition in dependencies.registry.list_definitions()
                if definition.name != EXTERNAL_WEB_SEARCH_TOOL
            )
            registry.register(
                AgentToolDefinition(
                    name=EXTERNAL_WEB_SEARCH_TOOL,
                    description="Search the public web through an external agent.",
                    input_schema={
                        "type": "object",
                        "required": ["query"],
                        "properties": {
                            "query": {"type": "string"},
                            "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
                        },
                        "additionalProperties": False,
                    },
                    output_schema={"type": "object", "required": ["tool_name", "ok", "result"]},
                    handler=fake_web_search,
                    requires_confirmation=True,
                    allowed_source_types=frozenset({"agent_chat", "web_search"}),
                )
            )
            dependencies = dependencies.with_registry(registry)

            first = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="你给我搜一下中科曙光的校园招聘信息"),
                dependencies=dependencies,
            )
            approval = session.scalars(select(ApprovalRequest)).one()
            workflow = session.get(WorkflowRun, first.workflow_run_id)

            self.assertEqual(WorkflowRunStatus.WAITING_USER, workflow.status)
            self.assertEqual("wait_confirmation", first.state.current_step)
            self.assertEqual({"query": "中科曙光 校园招聘 秋招 官网", "max_results": 5}, approval.payload["tool_input"])
            self.assertEqual([], search_calls)

            continued = continue_agent_workflow_after_approval(
                approval.id,
                approved=True,
                decision_reason="allow search",
                dependencies=dependencies,
            )
            session.commit()
            tool_log = session.scalars(select(ToolCallLog)).one()

        self.assertEqual([{"query": "中科曙光 校园招聘 秋招 官网", "max_results": 5}], search_calls)
        self.assertEqual("final_response", continued.state.current_step)
        self.assertEqual("llm_tool_result_summary", continued.state.response_mode)
        self.assertEqual("确认后已完成联网搜索：中科曙光校园招聘官网", continued.state.final_response)
        self.assertEqual(ToolCallStatus.SUCCEEDED, tool_log.status)
        self.assertEqual(2, len(fake_llm.calls))

    def test_agent_does_not_auto_select_external_web_search_for_campus_recruiting_search_request(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL, AgentToolDefinition, AgentToolRegistry
        from app.domains.automation.models import ToolCallLog

        calls = []

        def fake_web_search(_session, *, query: str, max_results: int = 5):
            calls.append({"query": query, "max_results": max_results})
            return {
                "tool_name": EXTERNAL_WEB_SEARCH_TOOL,
                "ok": True,
                "result": {
                    "executor_name": "http-web-search-fallback",
                    "query": query,
                    "answer": "联网搜索结果：腾讯校招官方入口：https://join.qq.com/",
                    "sources": [{"title": "腾讯校招", "url": "https://join.qq.com/"}],
                },
            }

        with self.Session() as session:
            session_id = self._session_id(session)
            dependencies = self._dependencies(session)
            registry = AgentToolRegistry(
                definition
                for definition in dependencies.registry.list_definitions()
                if definition.name != EXTERNAL_WEB_SEARCH_TOOL
            )
            registry.register(
                AgentToolDefinition(
                    name=EXTERNAL_WEB_SEARCH_TOOL,
                    description="Search the public web through an external agent.",
                    input_schema={"type": "object", "required": ["query"]},
                    output_schema={"type": "object", "required": ["tool_name", "ok", "result"]},
                    handler=fake_web_search,
                    allowed_source_types=frozenset({"agent_chat", "web_search"}),
                )
            )

            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="帮我搜一下腾讯秋招校园招聘信息"),
                dependencies=dependencies.with_registry(registry),
            )
            session.commit()

            tool_logs = list(session.scalars(select(ToolCallLog)).all())

        self.assertIsNone(result.state.requested_tool_name)
        self.assertEqual("agent_chat", result.state.source_type)
        self.assertEqual([], tool_logs)
        self.assertEqual([], calls)
        self.assertEqual("deterministic_stub", result.state.response_mode)

    def test_agent_does_not_auto_select_find_apply_entry_from_job_id_request(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.domains.automation.models import ToolCallLog
        from app.domains.jobs.models import (
            JobLead,
            JobLeadStatus,
            JobSource,
            JobSourceFetchMode,
            JobSourceTrustLevel,
            JobSourceType,
        )

        with self.Session() as session:
            source = JobSource(
                name="Campus leads",
                source_type=JobSourceType.OFFICIAL_API,
                entry_url="https://example.com/jobs",
                trust_level=JobSourceTrustLevel.MEDIUM_HIGH,
                fetch_mode=JobSourceFetchMode.OFFICIAL_API,
            )
            session.add(
                JobLead(
                    id="lead-apply-1",
                    source=source,
                    lead_hash="lead-apply-1",
                    company_name="Tencent",
                    title="Backend Engineer Intern",
                    source_url="https://careers.tencent.com/job/1",
                    apply_url="https://careers.tencent.com/apply/1",
                    jd_text="Campus backend role requiring Java and distributed systems.",
                    skills=[],
                    trust_level=JobSourceTrustLevel.MEDIUM_HIGH,
                    verification_status=JobLeadStatus.VERIFIED,
                )
            )
            session.flush()
            session_id = self._session_id(session)

            result = run_agent_workflow(
                AgentRunCommand(
                    session_id=session_id,
                    user_message="open application entry for job_id=lead-apply-1",
                ),
                dependencies=self._dependencies(session),
            )
            session.commit()

            tool_logs = list(session.scalars(select(ToolCallLog)).all())

        self.assertIsNone(result.state.requested_tool_name)
        self.assertEqual("agent_chat", result.state.source_type)
        self.assertEqual([], tool_logs)
        self.assertEqual("deterministic_stub", result.state.response_mode)

    def test_explicit_tool_execution_records_durable_orchestration_step(self) -> None:
        from app.agent_runtime.durable_state.models import AgentStepState, AgentTaskState
        from app.agent_runtime.durable_state.repository import SqlAlchemyDurableStateRepository
        from app.agent_runtime.durable_state.schemas import AgentStepStatus, AgentTaskStatus
        from app.agent_runtime.durable_state.service import DurableStateService
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow

        with self.Session() as session:
            session_id = self._session_id(session)
            durable_state_service = DurableStateService(SqlAlchemyDurableStateRepository(session))
            dependencies = self._dependencies(session, durable_state_service=durable_state_service)

            result = run_agent_workflow(
                AgentRunCommand(
                    session_id=session_id,
                    user_message="查一下这轮会话里的岗位线索",
                    requested_tool_name="sessions_search",
                    source_type="agent_chat",
                ),
                dependencies=dependencies,
            )
            session.commit()

            task = session.get(AgentTaskState, result.workflow_run_id)
            steps = list(session.scalars(select(AgentStepState)).all())

        self.assertIsNotNone(task)
        self.assertEqual(AgentTaskStatus.RUNNING, task.status)
        self.assertEqual("sessions_search", task.capability)
        self.assertEqual(1, len(steps))
        self.assertEqual(task.current_step_id, steps[0].id)
        self.assertEqual(AgentStepStatus.SUCCEEDED, steps[0].status)
        self.assertEqual("sessions_search", steps[0].capability)
        self.assertEqual(result.state.tool_call_ids[0], steps[0].tool_call_log_id)
        self.assertEqual({"query": "查一下这轮会话里的岗位线索", "limit": 10}, steps[0].input_payload["tool_input"])

    def test_memory_search_tool_records_memory_snapshot_for_durable_step(self) -> None:
        from app.agent_runtime.durable_state.models import AgentMemorySnapshot
        from app.agent_runtime.durable_state.repository import SqlAlchemyDurableStateRepository
        from app.agent_runtime.durable_state.service import DurableStateService
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.domains.agent_memory.models import AgentMemory, AgentMemoryStatus

        with self.Session() as session:
            session_id = self._session_id(session)
            session.add(
                AgentMemory(
                    id="memory-java-backend",
                    memory_type="user_preference",
                    scope="candidate_profile",
                    title="Java 后端偏好",
                    content="用户偏好 Java 后端和分布式系统方向。",
                    source_type="user_profile",
                    status=AgentMemoryStatus.ACTIVE,
                    importance=10,
                    metadata_json={"source": "test"},
                )
            )
            session.commit()

        with self.Session() as session:
            durable_state_service = DurableStateService(SqlAlchemyDurableStateRepository(session))
            dependencies = self._dependencies(session, durable_state_service=durable_state_service)

            result = run_agent_workflow(
                AgentRunCommand(
                    session_id=session_id,
                    user_message="Java 后端偏好",
                    requested_tool_name="memory_search",
                    source_type="agent_chat",
                ),
                dependencies=dependencies,
            )
            session.commit()

            snapshots = list(session.scalars(select(AgentMemorySnapshot)).all())

        self.assertEqual("final_response", result.state.current_step)
        self.assertEqual(1, len(snapshots))
        self.assertEqual("memory-java-backend", snapshots[0].memory_id)
        self.assertEqual("agent_memory", snapshots[0].source_type)
        self.assertIn("memory_search matched", snapshots[0].usage_reason)
        self.assertFalse(snapshots[0].passed_to_executor)

    def test_tool_result_envelope_artifacts_are_indexed_for_durable_step(self) -> None:
        from app.agent_runtime.durable_state.models import AgentArtifactIndex
        from app.agent_runtime.durable_state.repository import SqlAlchemyDurableStateRepository
        from app.agent_runtime.durable_state.service import DurableStateService
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL, AgentToolDefinition, AgentToolRegistry

        def fake_web_search(_session, *, query: str, max_results: int = 5):
            return {
                "tool_name": EXTERNAL_WEB_SEARCH_TOOL,
                "ok": True,
                "result": {
                    "executor_name": "claude-sdk-agent",
                    "query": query,
                    "answer": "腾讯校招官网：https://join.qq.com/",
                    "artifacts": [
                        {"type": "url", "title": "腾讯校招", "url": "https://join.qq.com/"},
                    ],
                },
            }

        with self.Session() as session:
            session_id = self._session_id(session)
            durable_state_service = DurableStateService(SqlAlchemyDurableStateRepository(session))
            dependencies = self._dependencies(session, durable_state_service=durable_state_service)
            registry = AgentToolRegistry(
                definition
                for definition in dependencies.registry.list_definitions()
                if definition.name != EXTERNAL_WEB_SEARCH_TOOL
            )
            registry.register(
                AgentToolDefinition(
                    name=EXTERNAL_WEB_SEARCH_TOOL,
                    description="Search the public web through an external agent.",
                    input_schema={"type": "object", "required": ["query"]},
                    output_schema={"type": "object", "required": ["tool_name", "ok", "result"]},
                    handler=fake_web_search,
                    allowed_source_types=frozenset({"agent_chat", "web_search"}),
                )
            )

            result = run_agent_workflow(
                AgentRunCommand(
                    session_id=session_id,
                    user_message="搜一下腾讯校招官网",
                    requested_tool_name=EXTERNAL_WEB_SEARCH_TOOL,
                    source_type="agent_chat",
                    tool_input={"query": "腾讯 校园招聘 官网", "max_results": 5},
                ),
                dependencies=dependencies.with_registry(registry),
            )
            session.commit()

            artifacts = list(session.scalars(select(AgentArtifactIndex)).all())

        self.assertEqual("final_response", result.state.current_step)
        self.assertEqual(1, len(artifacts))
        self.assertEqual("url", artifacts[0].artifact_type)
        self.assertEqual("https://join.qq.com/", artifacts[0].uri)
        self.assertEqual("claude-sdk-agent", artifacts[0].artifact_metadata["executor"])

    def test_tool_summary_reports_apply_entry_when_external_dispatch_succeeds(self) -> None:
        from app.agent_runtime.graph_factory import tool_result_summary_response
        from app.agent_runtime.state import AgentState
        from app.agent_runtime.tool_registry import APPLICATION_FIND_APPLY_ENTRY_TOOL

        state = AgentState(
            session_id="session-apply-summary",
            workflow_run_id="workflow-apply-summary",
            agent_run_id="agent-run-apply-summary",
            user_message="open application entry for job_id=lead-apply-1",
            current_step="maybe_tool",
            requested_tool_name=APPLICATION_FIND_APPLY_ENTRY_TOOL,
            llm_messages=[
                {
                    "role": "assistant",
                    "content": "Tool result: applications.find_apply_entry succeeded",
                    "metadata": {
                        "content_json": {
                            "tool_name": APPLICATION_FIND_APPLY_ENTRY_TOOL,
                            "status": "succeeded",
                            "result": {
                                "tool_name": APPLICATION_FIND_APPLY_ENTRY_TOOL,
                                "ok": True,
                                "result": {
                                    "task_id": "external-task-apply-1",
                                    "status": "succeeded",
                                    "task_envelope": {
                                        "job": {
                                            "job_id": "lead-apply-1",
                                            "company_name": "Tencent",
                                            "title": "Backend Engineer Intern",
                                        }
                                    },
                                    "dispatch": {
                                        "ok": True,
                                        "executor_name": "claude-sdk-agent",
                                        "status": "succeeded",
                                        "result_status": "found_opened",
                                        "apply_url": "https://careers.tencent.com/apply/1",
                                        "next_action": "wait_user_review",
                                    },
                                },
                            },
                        }
                    },
                }
            ],
        )

        response = tool_result_summary_response(state)

        self.assertIsNotNone(response)
        content, mode = response
        self.assertEqual("tool_result_summary", mode)
        self.assertIn("已找到申请入口", content)
        self.assertIn("https://careers.tencent.com/apply/1", content)
        self.assertIn("claude-sdk-agent", content)
        self.assertIn("停在最终提交前", content)

    def test_tool_summary_rejects_company_overview_when_user_asked_specific_company(self) -> None:
        from app.agent_runtime.graph_factory import tool_result_summary_response
        from app.agent_runtime.state import AgentState
        from app.agent_runtime.tool_registry import LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL

        state = AgentState(
            session_id="session-company-specific-summary",
            workflow_run_id="workflow-company-specific-summary",
            agent_run_id="agent-run-company-specific-summary",
            user_message="你给我看一下数据库中关于京东这个公司的信息有什么",
            current_step="maybe_tool",
            requested_tool_name=LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL,
            llm_messages=[
                {
                    "role": "assistant",
                    "content": "Tool result: local.company_database_overview succeeded",
                    "metadata": {
                        "content_json": {
                            "tool_name": LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL,
                            "status": "succeeded",
                            "result": {
                                "tool_name": LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL,
                                "ok": True,
                                "result": {
                                    "company_count": 2,
                                    "job_count": 3,
                                    "job_lead_count": 71,
                                    "job_lead_company_count": 71,
                                    "recruiting_signal_count": 11,
                                    "recruiting_signal_company_count": 10,
                                    "sample_companies": ["Canonical Ltd.", "腾讯"],
                                    "sample_lead_companies": ["3PEAK", "deeproute.ai"],
                                    "sample_signal_companies": ["中国平安", "基恩士"],
                                    "company_rows": [
                                        {
                                            "tier": "正式企业",
                                            "company_name": "Canonical Ltd.",
                                            "known_info": "企业档案、正式岗位",
                                            "quantity": "2 条岗位",
                                            "status": "可用于推荐",
                                        },
                                        {
                                            "tier": "正式企业",
                                            "company_name": "腾讯",
                                            "known_info": "企业档案、正式岗位",
                                            "quantity": "1 条岗位",
                                            "status": "可用于推荐",
                                        },
                                    ],
                                },
                            },
                        }
                    },
                }
            ],
        )

        response = tool_result_summary_response(state)

        self.assertIsNotNone(response)
        content, mode = response
        self.assertEqual("tool_result_summary_insufficient", mode)
        self.assertIn("京东", content)
        self.assertIn("不能把全库概览当成答案", content)
        self.assertNotIn("我先按公司档次列出来", content)
        self.assertNotIn("Canonical Ltd.", content)
        self.assertNotIn("腾讯", content)

    def test_tool_summary_formats_database_company_list_as_markdown_table(self) -> None:
        from app.agent_runtime.graph_factory import tool_result_summary_response
        from app.agent_runtime.state import AgentState
        from app.agent_runtime.tool_registry import DATABASE_COMPANY_LIST_TOOL

        state = AgentState(
            session_id="session-company-list-summary",
            workflow_run_id="workflow-company-list-summary",
            agent_run_id="agent-run-company-list-summary",
            user_message="给我看一下有哪些公司，给我20个就行",
            current_step="maybe_tool",
            requested_tool_name=DATABASE_COMPANY_LIST_TOOL,
            llm_messages=[
                {
                    "role": "assistant",
                    "content": "Tool result: database.company_list succeeded",
                    "metadata": {
                        "content_json": {
                            "tool_name": DATABASE_COMPANY_LIST_TOOL,
                            "status": "succeeded",
                            "result": {
                                "tool_name": DATABASE_COMPANY_LIST_TOOL,
                                "ok": True,
                                "result": {
                                    "total_count": 3,
                                    "count": 2,
                                    "companies": [
                                        {
                                            "company_name": "阿里巴巴",
                                            "has_profile": True,
                                            "job_count": 2,
                                            "lead_count": 0,
                                            "signal_count": 1,
                                            "total_record_count": 3,
                                        },
                                        {
                                            "company_name": "腾讯",
                                            "has_profile": True,
                                            "job_count": 1,
                                            "lead_count": 2,
                                            "signal_count": 1,
                                            "total_record_count": 4,
                                        },
                                    ],
                                },
                            },
                        }
                    },
                }
            ],
        )

        response = tool_result_summary_response(state)

        self.assertIsNotNone(response)
        content, mode = response
        self.assertEqual("tool_result_summary", mode)
        self.assertIn("当前本地数据库去重公司共 3 家", content)
        self.assertIn("| 公司 | 企业档案 | 正式岗位 | 岗位线索 | 招聘来源 | 已有记录 |", content)
        self.assertIn("| 阿里巴巴 | 有 | 2 | 0 | 1 | 3 |", content)
        self.assertIn("| 腾讯 | 有 | 1 | 2 | 1 | 4 |", content)
        self.assertIn("本次展示 2 家", content)

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





    def test_agent_run_rewrites_false_tool_execution_claim_when_no_tool_ran(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow

        class FakeLLMClient:
            def complete(self, *, messages):
                from app.infrastructure.llm.chat_client import LLMChatCompletion

                return LLMChatCompletion(content="我已经调用联网搜索，查到了 Canonical 的主要业务。")

        with self.Session() as session:
            session_id = self._session_id(session)
            result = run_agent_workflow(
                AgentRunCommand(session_id=session_id, user_message="帮我写一句求职备注"),
                dependencies=self._dependencies(session, llm_client=FakeLLMClient()),
            )
            session.commit()

        self.assertNotIn("我已经调用联网搜索", result.state.final_response)
        self.assertIn("本轮没有真实工具执行记录", result.state.final_response)
        self.assertEqual("false_tool_claim_fallback", result.state.response_mode)
        self.assertTrue(result.state.context_metadata["output_sanitizer"]["removed_false_tool_claim"])

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

    def test_agent_run_auto_compaction_flushes_memory_before_summary(self) -> None:
        from app.agent_runtime.graph_factory import AgentRunCommand, run_agent_workflow
        from app.domains.conversations.models import AgentMessageRole
        from app.domains.conversations.repository import ConversationRepository
        from app.domains.conversations.schemas import AgentMessageCreate
        from app.domains.conversations.service import (
            ConversationService,
            PreCompactionMemoryFlushResult,
        )

        class FakeLLMClient:
            def complete(self, *, messages):
                from app.infrastructure.llm.chat_client import LLMChatCompletion

                return LLMChatCompletion(content="ok")

        flush_commands = []

        def flush_memory(command):
            flush_commands.append(command)
            return PreCompactionMemoryFlushResult(
                reviewed_message_count=len(command.message_ids),
                created_candidate_ids=["candidate-auto-compact"],
                pending_candidate_ids=["candidate-auto-compact"],
            )

        with self.Session() as session:
            conversation_service = ConversationService(
                ConversationRepository(session),
                pre_compaction_memory_flush=flush_memory,
            )
            created = conversation_service.create_session(title="auto compact flush", primary_intent="agent_chat")
            session_id = created.id
            for index in range(8):
                conversation_service.append_message(
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
                dependencies=self._dependencies(
                    session,
                    conversation_service=conversation_service,
                    llm_client=FakeLLMClient(),
                ),
            )
            session.commit()

        self.assertEqual(1, len(flush_commands))
        self.assertEqual(session_id, flush_commands[0].session_id)
        self.assertEqual(result.workflow_run_id, flush_commands[0].workflow_run_id)
        self.assertEqual(result.state.agent_run_id, flush_commands[0].agent_run_id)
        self.assertEqual("agent_chat", flush_commands[0].target_scope)
        self.assertGreater(len(flush_commands[0].message_ids), 0)
        self.assertEqual(
            ["candidate-auto-compact"],
            result.state.context_metadata["auto_compaction_memory_flush"]["created_candidate_ids"],
        )

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

    def test_checkpoint_store_prefers_workflow_step_when_database_timestamps_collide(self) -> None:
        from app.agent_runtime.checkpoints import AgentCheckpointStore
        from app.agent_runtime.state import AgentState
        from app.domains.automation.models import WorkflowCheckpoint, WorkflowRun, WorkflowRunStatus

        with self.Session() as session:
            workflow = WorkflowRun(
                workflow_type="agent_chat",
                status=WorkflowRunStatus.WAITING_USER,
                current_step="wait_confirmation",
            )
            session.add(workflow)
            session.flush()
            timestamp = datetime(2026, 9, 20, 15, 42, 50)

            def checkpoint_state(step: str, approval_request_id: str | None = None) -> dict[str, object]:
                return AgentState(
                    session_id="session-colliding-checkpoints",
                    workflow_run_id=workflow.id,
                    agent_run_id="agent-run-colliding-checkpoints",
                    user_message="continue",
                    current_step=step,
                    approval_request_id=approval_request_id,
                ).to_checkpoint_state()

            session.add_all(
                [
                    WorkflowCheckpoint(
                        workflow_run_id=workflow.id,
                        checkpoint_key="build_context",
                        state=checkpoint_state("build_context"),
                        created_at=timestamp,
                    ),
                    WorkflowCheckpoint(
                        workflow_run_id=workflow.id,
                        checkpoint_key="wait_confirmation",
                        state=checkpoint_state("wait_confirmation", "approval-1"),
                        created_at=timestamp,
                    ),
                ]
            )
            session.commit()

            latest = AgentCheckpointStore(
                session=session,
                automation_service=self._dependencies(session).automation_service,
            ).load_latest(workflow.id)

        self.assertEqual("wait_confirmation", latest.checkpoint_key)
        self.assertEqual("approval-1", latest.state.approval_request_id)

    def test_create_agent_graph_exposes_expected_node_order(self) -> None:
        from app.agent_runtime.graph_factory import create_agent_graph

        graph = create_agent_graph()

        self.assertEqual(
            ["build_context", "plan_or_reply", "maybe_tool", "wait_confirmation", "final_response"],
            graph.node_order,
        )
        self.assertIsNotNone(graph.compiled_graph)


def _write_read_script(root: Path, content: str) -> Path:
    """Create the filesystem Skill's internal read script for graph-level tests."""

    scripts_dir = root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / "read_file.py").write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--path', required=True)\n"
        "parser.add_argument('--offset', type=int, default=0)\n"
        "parser.add_argument('--limit', type=int, default=200)\n"
        "parser.add_argument('--encoding', default='auto')\n"
        "args = parser.parse_args()\n"
        f"content = {content!r}\n"
        "print(content[args.offset:args.offset + args.limit], end='')\n",
        encoding="utf-8",
    )
    return root


def _write_path_exists_script(root: Path) -> Path:
    """Create the filesystem Skill's internal path_exists script for graph-level tests."""

    scripts_dir = root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / "path_exists.py").write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--path', required=True)\n"
        "args = parser.parse_args()\n"
        "print(f'EXISTS: file {args.path}')\n",
        encoding="utf-8",
    )
    return root


def _write_move_script(root: Path) -> Path:
    """Create the filesystem Skill's internal move script for graph-level tests."""

    scripts_dir = root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / "move_file.py").write_text(
        "import argparse\n"
        "from pathlib import Path\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--src', required=True)\n"
        "parser.add_argument('--dst', required=True)\n"
        "parser.add_argument('--overwrite', action='store_true')\n"
        "args = parser.parse_args()\n"
        "src = Path(args.src)\n"
        "dst = Path(args.dst)\n"
        "if dst.exists() and not args.overwrite:\n"
        "    print('ERROR: destination exists')\n"
        "    raise SystemExit(1)\n"
        "dst.parent.mkdir(parents=True, exist_ok=True)\n"
        "src.replace(dst)\n"
        "print(f'MOVED:{src}->{dst}')\n",
        encoding="utf-8",
    )
    return root


def _write_noop_move_script(root: Path) -> Path:
    """Create a broken move script that claims success without changing disk state."""

    scripts_dir = root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / "move_file.py").write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--src', required=True)\n"
        "parser.add_argument('--dst', required=True)\n"
        "parser.add_argument('--overwrite', action='store_true')\n"
        "args = parser.parse_args()\n"
        "print(f'MOVED:{args.src}->{args.dst}')\n",
        encoding="utf-8",
    )
    return root


def _write_copy_script(root: Path) -> Path:
    """Create the filesystem Skill's internal copy script for graph-level tests."""

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
        "    print('ERROR: destination exists')\n"
        "    raise SystemExit(1)\n"
        "dst.parent.mkdir(parents=True, exist_ok=True)\n"
        "shutil.copy2(src, dst)\n"
        "print(f'COPIED:{src}->{dst}')\n",
        encoding="utf-8",
    )
    return root
