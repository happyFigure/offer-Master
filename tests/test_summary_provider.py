import sys
import unittest
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class SummaryProviderTest(unittest.TestCase):
    def setUp(self):
        from app.db.base import Base
        import app.domains.agent_memory.models  # noqa: F401
        import app.domains.automation.models  # noqa: F401
        import app.domains.conversations.models  # noqa: F401

        self.engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

    def tearDown(self):
        self.engine.dispose()

    def _build_plan(self):
        from app.agent_runtime.memory.compaction import CompactionConfig, prepare_compaction
        from app.domains.conversations.models import AgentMessageKind, AgentMessageRole
        from app.domains.conversations.schemas import AgentMessageCreate
        from app.domains.conversations.repository import ConversationRepository
        from app.domains.conversations.service import ConversationService

        with self.Session() as session:
            service = ConversationService(ConversationRepository(session))
            conversation = service.create_session(title="摘要 provider", primary_intent="agent_memory")
            first = service.append_message(
                conversation.id,
                AgentMessageCreate(
                    role=AgentMessageRole.USER,
                    message_kind=AgentMessageKind.USER_TEXT,
                    content_text="用户目标：找 Java 后端秋招岗位，投递前必须确认。",
                    visible_content_text="用户目标：找 Java 后端秋招岗位，投递前必须确认。",
                    token_estimate=800,
                ),
            )
            kept = service.append_message(
                conversation.id,
                AgentMessageCreate(
                    role=AgentMessageRole.ASSISTANT,
                    content_text="最近回复：继续查询岗位。",
                    visible_content_text="最近回复：继续查询岗位。",
                    token_estimate=50,
                ),
            )
            session.flush()
            plan = prepare_compaction(
                [first, kept],
                latest_summary="Goal:\n- 之前已经确认是秋招求职助手。",
                config=CompactionConfig(context_window=1000, reserve_tokens=0, keep_recent_tokens=100),
            )
            return plan, first.id, kept.id

    def test_deterministic_summary_provider_preserves_existing_summary_shape(self):
        from app.agent_runtime.memory.compaction import CompactionConfig, prepare_compaction
        from app.agent_runtime.memory.summary_provider import DeterministicSummaryProvider
        from app.domains.conversations.models import AgentMessageKind, AgentMessageRole, AgentSession
        from app.domains.conversations.schemas import AgentMessageCreate
        from app.domains.conversations.repository import ConversationRepository
        from app.domains.conversations.service import ConversationService

        with self.Session() as session:
            service = ConversationService(ConversationRepository(session))
            conversation = service.create_session(title="摘要 provider", primary_intent="agent_memory")
            first = service.append_message(
                conversation.id,
                AgentMessageCreate(
                    role=AgentMessageRole.USER,
                    message_kind=AgentMessageKind.USER_TEXT,
                    content_text="用户目标：找 Java 后端秋招岗位",
                    visible_content_text="用户目标：找 Java 后端秋招岗位",
                    token_estimate=800,
                ),
            )
            kept = service.append_message(
                conversation.id,
                AgentMessageCreate(
                    role=AgentMessageRole.ASSISTANT,
                    content_text="最近回复：继续查询岗位。",
                    visible_content_text="最近回复：继续查询岗位。",
                    token_estimate=50,
                ),
            )
            session.flush()

            plan = prepare_compaction(
                [first, kept],
                latest_summary=None,
                config=CompactionConfig(context_window=1000, reserve_tokens=0, keep_recent_tokens=100),
            )
            result = DeterministicSummaryProvider().summarize(plan)

        self.assertEqual("deterministic_compactor", result.created_by)
        self.assertIn("Goal:", result.summary_text)
        self.assertIn("用户目标：找 Java 后端秋招岗位", result.summary_text)
        self.assertEqual("Preserve older conversation context for future Agent turns.", result.summary_json["Goal"])
        self.assertEqual([first.id], result.summary_json["Progress"]["covered_message_ids"])
        self.assertEqual(kept.id, result.summary_json["Key Decisions"]["first_kept_message_id"])
        self.assertEqual("deterministic", result.metadata_json["summary_provider"])

    def test_conversation_service_uses_injected_summary_provider(self):
        from app.agent_runtime.memory.compaction import CompactionPlan
        from app.agent_runtime.memory.summary_provider import SummaryProviderResult
        from app.domains.conversations.models import AgentMessageRole
        from app.domains.conversations.repository import ConversationRepository
        from app.domains.conversations.schemas import AgentMessageCreate
        from app.domains.conversations.service import ConversationService
        from app.agent_runtime.memory.compaction import CompactionConfig

        class StubSummaryProvider:
            def summarize(self, plan: CompactionPlan) -> SummaryProviderResult:
                return SummaryProviderResult(
                    summary_text="Goal:\n- Stub provider summary.\n\nNext Steps:\n- Continue from injected provider.",
                    summary_json={"Goal": "Stub provider summary."},
                    created_by="stub_summary_provider",
                    metadata_json={"summary_provider": "stub", "mode": "test"},
                )

        with self.Session() as session:
            service = ConversationService(
                ConversationRepository(session),
                summary_provider=StubSummaryProvider(),
            )
            conversation = service.create_session(title="可插拔摘要", primary_intent="agent_memory")
            service.append_message(
                conversation.id,
                AgentMessageCreate(
                    role=AgentMessageRole.USER,
                    content_text="旧消息：需要被 provider 摘要。",
                    visible_content_text="旧消息：需要被 provider 摘要。",
                    token_estimate=500,
                ),
            )
            service.append_message(
                conversation.id,
                AgentMessageCreate(
                    role=AgentMessageRole.ASSISTANT,
                    content_text="最近消息：保留原文。",
                    visible_content_text="最近消息：保留原文。",
                    token_estimate=10,
                ),
            )

            compact_result = service.compact_session(
                conversation.id,
                CompactionConfig(context_window=1000, reserve_tokens=0, keep_recent_tokens=100),
            )
            session.commit()

        self.assertEqual("stub_summary_provider", compact_result.summary.created_by)
        self.assertEqual({"Goal": "Stub provider summary."}, compact_result.summary.summary_json)
        self.assertEqual("stub", compact_result.summary.metadata_json["summary_provider"])
        self.assertEqual("test", compact_result.summary.metadata_json["mode"])
        self.assertIn("token_estimate_before", compact_result.summary.metadata_json)

    def test_compact_session_flushes_memory_candidates_before_summary_provider_runs(self):
        from app.agent_runtime.memory.compaction import CompactionConfig, CompactionPlan
        from app.agent_runtime.memory.summary_provider import SummaryProviderResult
        from app.domains.conversations.models import AgentMessageRole
        from app.domains.conversations.repository import ConversationRepository
        from app.domains.conversations.schemas import AgentMessageCreate
        from app.domains.conversations.service import ConversationService, PreCompactionMemoryFlushResult

        state = {"flushed": False, "command": None}

        class SummaryProviderAfterFlush:
            def summarize(self, plan: CompactionPlan) -> SummaryProviderResult:
                if not state["flushed"]:
                    raise AssertionError("pre-compaction memory flush must run before summary generation")
                return SummaryProviderResult(
                    summary_text="Goal:\n- Summary after silent memory flush.",
                    summary_json={"Goal": "Summary after silent memory flush."},
                    created_by="summary_after_flush",
                    metadata_json={"summary_provider": "stub"},
                )

        def flush_memory(command):
            state["flushed"] = True
            state["command"] = command
            return PreCompactionMemoryFlushResult(
                reviewed_message_count=len(command.message_ids),
                reviewed_tool_call_count=0,
                created_candidate_ids=["candidate-1"],
                pending_candidate_ids=["candidate-1"],
                promoted_memory_ids=[],
                merged_memory_ids=[],
                skipped_reasons=[],
            )

        with self.Session() as session:
            service = ConversationService(
                ConversationRepository(session),
                summary_provider=SummaryProviderAfterFlush(),
                pre_compaction_memory_flush=flush_memory,
            )
            conversation = service.create_session(title="压缩前刷新", primary_intent="agent_memory")
            old_message = service.append_message(
                conversation.id,
                AgentMessageCreate(
                    role=AgentMessageRole.USER,
                    content_text="投递前一定要让我确认。",
                    visible_content_text="投递前一定要让我确认。",
                    token_estimate=500,
                ),
            )
            kept_message = service.append_message(
                conversation.id,
                AgentMessageCreate(
                    role=AgentMessageRole.ASSISTANT,
                    content_text="最近消息保留原文。",
                    visible_content_text="最近消息保留原文。",
                    token_estimate=10,
                ),
            )

            compact_result = service.compact_session(
                conversation.id,
                CompactionConfig(context_window=1000, reserve_tokens=0, keep_recent_tokens=100),
                workflow_run_id="workflow-flush-1",
                agent_run_id="agent-run-flush-1",
                target_scope="job_discovery",
            )
            session.commit()

        self.assertTrue(state["flushed"])
        self.assertEqual(conversation.id, state["command"].session_id)
        self.assertEqual("workflow-flush-1", state["command"].workflow_run_id)
        self.assertEqual("agent-run-flush-1", state["command"].agent_run_id)
        self.assertEqual("job_discovery", state["command"].target_scope)
        self.assertEqual([old_message.id], state["command"].message_ids)
        self.assertNotIn(kept_message.id, state["command"].message_ids)
        self.assertEqual("summary_after_flush", compact_result.summary.created_by)
        self.assertEqual(
            {
                "reviewed_message_count": 1,
                "reviewed_tool_call_count": 0,
                "created_candidate_ids": ["candidate-1"],
                "pending_candidate_ids": ["candidate-1"],
                "promoted_memory_ids": [],
                "merged_memory_ids": [],
                "skipped_reasons": [],
            },
            compact_result.summary.metadata_json["pre_compaction_memory_flush"],
        )

    def test_compact_session_continues_when_pre_compaction_memory_flush_fails(self):
        from app.agent_runtime.memory.compaction import CompactionConfig
        from app.domains.conversations.models import AgentMessageRole
        from app.domains.conversations.repository import ConversationRepository
        from app.domains.conversations.schemas import AgentMessageCreate
        from app.domains.conversations.service import ConversationService

        def broken_flush(_command):
            raise RuntimeError("learning database temporarily unavailable")

        with self.Session() as session:
            service = ConversationService(
                ConversationRepository(session),
                pre_compaction_memory_flush=broken_flush,
            )
            conversation = service.create_session(title="刷新失败仍压缩", primary_intent="agent_memory")
            service.append_message(
                conversation.id,
                AgentMessageCreate(
                    role=AgentMessageRole.USER,
                    content_text="旧消息：投递前一定要让我确认。",
                    visible_content_text="旧消息：投递前一定要让我确认。",
                    token_estimate=500,
                ),
            )
            service.append_message(
                conversation.id,
                AgentMessageCreate(
                    role=AgentMessageRole.ASSISTANT,
                    content_text="最近消息：继续。",
                    visible_content_text="最近消息：继续。",
                    token_estimate=10,
                ),
            )

            compact_result = service.compact_session(
                conversation.id,
                CompactionConfig(context_window=1000, reserve_tokens=0, keep_recent_tokens=100),
                workflow_run_id="workflow-broken-flush",
                agent_run_id="agent-run-broken-flush",
            )
            session.commit()

        self.assertIsNotNone(compact_result.summary.id)
        flush_metadata = compact_result.summary.metadata_json["pre_compaction_memory_flush"]
        self.assertEqual(["pre_compaction_memory_flush_failed"], flush_metadata["skipped_reasons"])
        self.assertIn("learning database temporarily unavailable", flush_metadata["error"])

    def test_llm_summary_provider_uses_llm_and_parses_structured_json(self):
        from app.agent_runtime.memory.summary_provider import LLMSummaryProvider
        from app.infrastructure.llm.chat_client import LLMChatCompletion

        class FakeLLMClient:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []

            def complete(self, *, messages, tools=None, tool_choice=None, extra_body=None):
                self.calls.append(
                    {
                        "messages": messages,
                        "tools": tools,
                        "tool_choice": tool_choice,
                        "extra_body": extra_body,
                    }
                )
                return LLMChatCompletion(
                    content=(
                        "```json\n"
                        '{"Goal":["继续围绕 Java 后端秋招找岗位"],'
                        '"Constraints & Preferences":["投递前必须确认"],'
                        '"Progress":["已记录用户目标和最近上下文"],'
                        '"Key Decisions":["旧消息压成结构化摘要，最近消息保留原文"],'
                        '"Next Steps":["继续查询可信岗位来源"],'
                        '"Critical Context":["用户关注 Java / Agent / 后端"],'
                        '"Retrieval Hints":["Java 秋招","投递确认"]}'
                        "\n```"
                    ),
                    usage={"prompt_tokens": 100, "completion_tokens": 60},
                )

        plan, first_id, _ = self._build_plan()
        llm_client = FakeLLMClient()

        result = LLMSummaryProvider(llm_client=llm_client).summarize(plan)

        self.assertEqual("llm_summary_provider", result.created_by)
        self.assertEqual("llm", result.metadata_json["summary_provider"])
        self.assertEqual({"prompt_tokens": 100, "completion_tokens": 60}, result.metadata_json["llm_usage"])
        self.assertIn("Goal:", result.summary_text)
        self.assertIn("继续围绕 Java 后端秋招找岗位", result.summary_text)
        self.assertEqual(["投递前必须确认"], result.summary_json["Constraints & Preferences"])
        self.assertIn(first_id, llm_client.calls[0]["messages"][1]["content"])
        self.assertIsNone(llm_client.calls[0]["tools"])
        self.assertIsNone(llm_client.calls[0]["tool_choice"])

    def test_hybrid_summary_provider_falls_back_when_llm_provider_raises(self):
        from app.agent_runtime.memory.summary_provider import DeterministicSummaryProvider, HybridSummaryProvider

        class BrokenProvider:
            name = "llm"

            def summarize(self, plan):
                raise RuntimeError("provider down")

        plan, first_id, _ = self._build_plan()

        result = HybridSummaryProvider(
            primary=BrokenProvider(),
            fallback=DeterministicSummaryProvider(),
        ).summarize(plan)

        self.assertEqual("hybrid_summary_provider", result.created_by)
        self.assertEqual("hybrid", result.metadata_json["summary_provider"])
        self.assertEqual("fallback", result.metadata_json["summary_provider_mode"])
        self.assertEqual("deterministic", result.metadata_json["fallback_provider"])
        self.assertIn("provider down", result.metadata_json["fallback_reason"])
        self.assertIn(first_id, result.summary_text)

    def test_hybrid_summary_provider_falls_back_when_llm_output_is_invalid(self):
        from app.agent_runtime.memory.summary_provider import DeterministicSummaryProvider, HybridSummaryProvider, LLMSummaryProvider
        from app.infrastructure.llm.chat_client import LLMChatCompletion

        class BadLLMClient:
            def complete(self, *, messages, tools=None, tool_choice=None, extra_body=None):
                return LLMChatCompletion(content='{"Goal":["只有目标，没有其他必填字段"]}')

        plan, first_id, _ = self._build_plan()

        result = HybridSummaryProvider(
            primary=LLMSummaryProvider(llm_client=BadLLMClient()),
            fallback=DeterministicSummaryProvider(),
        ).summarize(plan)

        self.assertEqual("fallback", result.metadata_json["summary_provider_mode"])
        self.assertIn("missing", result.metadata_json["fallback_reason"].lower())
        self.assertIn(first_id, result.summary_text)

    def test_agent_summary_provider_factory_uses_hybrid_when_configured(self):
        from app.api.v1.agent import _build_agent_summary_provider
        from app.core.config import Settings
        from app.infrastructure.llm.chat_client import LLMChatCompletion

        class FakeLLMClient:
            def complete(self, *, messages, tools=None, tool_choice=None, extra_body=None):
                return LLMChatCompletion(
                    content=(
                        '{"Goal":["按用户目标找秋招岗位"],'
                        '"Constraints & Preferences":["投递前确认"],'
                        '"Progress":["已压缩旧消息"],'
                        '"Key Decisions":["摘要由 LLM 生成，失败走确定性回退"],'
                        '"Next Steps":["继续保留最近原文"],'
                        '"Critical Context":["Java 后端秋招"],'
                        '"Retrieval Hints":["Java","秋招"]}'
                    )
                )

        settings = Settings(
            _env_file=None,
            JOBPILOT_LLM_API_KEY="sk-test",
            JOBPILOT_AGENT_SUMMARY_PROVIDER="hybrid",
        )
        plan, _, _ = self._build_plan()

        provider = _build_agent_summary_provider(settings, llm_client=FakeLLMClient())
        result = provider.summarize(plan)

        self.assertEqual("hybrid", result.metadata_json["summary_provider"])
        self.assertEqual("primary", result.metadata_json["summary_provider_mode"])
        self.assertEqual("llm", result.metadata_json["primary_provider"])
        self.assertIn("按用户目标找秋招岗位", result.summary_text)

    def test_agent_summary_provider_factory_defaults_to_deterministic(self):
        from app.api.v1.agent import _build_agent_summary_provider
        from app.agent_runtime.memory.summary_provider import DeterministicSummaryProvider
        from app.core.config import Settings

        settings = Settings(_env_file=None)

        provider = _build_agent_summary_provider(settings, llm_client=None)

        self.assertIsInstance(provider, DeterministicSummaryProvider)

    def test_pre_compaction_memory_flush_adapter_runs_consolidation_service(self):
        from app.api.v1.agent import _build_pre_compaction_memory_flush
        from app.domains.agent_memory.models import AgentLearningCandidate
        from app.domains.agent_memory.repository import AgentMemoryRepository
        from app.domains.automation.models import WorkflowRun, WorkflowRunStatus
        from app.domains.conversations.models import AgentMessage, AgentMessageKind, AgentMessageRole, AgentSession
        from app.domains.conversations.service import PreCompactionMemoryFlushCommand

        with self.Session() as session:
            agent_session = AgentSession(id="session-adapter", title="adapter flush")
            workflow_run = WorkflowRun(
                id="workflow-adapter",
                workflow_type="agent_memory",
                status=WorkflowRunStatus.RUNNING,
                current_step="build_context",
                user_goal="压缩前刷新记忆",
            )
            old_message = AgentMessage(
                id="message-adapter-boundary",
                session_id=agent_session.id,
                role=AgentMessageRole.USER,
                message_kind=AgentMessageKind.USER_TEXT,
                content_text="投递前一定要让我确认，不能自动提交。",
                visible_content_text="投递前一定要让我确认，不能自动提交。",
            )
            session.add_all([agent_session, workflow_run, old_message])
            session.flush()

            flush = _build_pre_compaction_memory_flush(session, AgentMemoryRepository(session))
            result = flush(
                PreCompactionMemoryFlushCommand(
                    session_id=agent_session.id,
                    workflow_run_id=workflow_run.id,
                    agent_run_id="agent-run-adapter",
                    target_scope="job_discovery",
                    message_ids=[old_message.id],
                )
            )
            session.commit()

            candidates = list(session.scalars(select(AgentLearningCandidate)).all())

        self.assertEqual(1, result.reviewed_message_count)
        self.assertEqual(1, len(result.created_candidate_ids))
        self.assertEqual(1, len(result.pending_candidate_ids))
        self.assertEqual(1, len(candidates))
        self.assertEqual("投递前必须用户确认", candidates[0].candidate_title)
        self.assertEqual("workflow-adapter", candidates[0].source_workflow_run_id)


if __name__ == "__main__":
    unittest.main()
