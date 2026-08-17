import sys
import unittest
import warnings
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))
try:
    from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

    warnings.simplefilter("ignore", LangChainPendingDeprecationWarning)
except ImportError:
    pass


class FakeFetcher:
    tool_name = "HTTPArticleFetcher"

    def __init__(self, text: str | None = None):
        self.text = text or (
            "JD.com 2027 campus recruiting opens Java backend engineer and Agent platform "
            "engineering roles in Beijing. Students can apply from the official career page."
        )
        self.calls = 0

    def fetch(self, url, context, *, domain_health=None):
        from app.domains.jobs.schemas import ToolResult, ToolSuggestedNextAction

        self.calls += 1
        return ToolResult(
            ok=True,
            stage=context.stage,
            tool_name=self.tool_name,
            suggested_next_action=ToolSuggestedNextAction.CONTINUE_WORKFLOW,
            cost={
                "tool_calls": context.tool_call_count + 1,
                "llm_calls": context.llm_call_count,
                "fetch_attempts_for_stage": context.fetch_attempts_for_stage + 1,
                "mcp_requests": context.mcp_request_count,
            },
            artifacts={
                "url": url,
                "final_url": url,
                "status_code": 200,
                "title": "JD 2027 Campus Recruiting",
                "text": self.text,
                "content_length": len(self.text),
                "candidate_links": ["https://career.example.com/apply/java-backend"],
                "extraction_method": "fake_http",
            },
        )


class FakeSocialProvider:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.calls = 0

    def extract(self, source_id, raw_lead_id, raw_content, source_url, trust_level):
        from app.domains.jobs.schemas import JobLeadCreate

        self.calls += 1
        if self.fail:
            raise RuntimeError("LLM extraction failed")
        return [
            JobLeadCreate(
                source_id=source_id,
                raw_lead_id=raw_lead_id,
                company_name="JD.com",
                title="Java Backend Engineer",
                city="Beijing",
                job_direction="backend",
                graduation_year="2027",
                source_url=source_url,
                apply_url="https://career.example.com/apply/java-backend",
                job_type="campus",
                jd_text="Java backend and Agent workflow platform engineering.",
                skills=["Java", "Spring", "Agent"],
                confidence_score=86,
                trust_level=trust_level,
            )
        ]


class UrlImportWorkflowTest(unittest.TestCase):
    def setUp(self):
        from app.db.base import Base
        from app.domains.applications import models as application_models  # noqa: F401
        from app.domains.automation import models as automation_models  # noqa: F401
        from app.domains.jobs import models as job_models  # noqa: F401

        self.engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

    def tearDown(self):
        self.engine.dispose()

    def test_url_import_workflow_success_saves_run_checkpoints_tool_logs_raw_and_leads(self):
        from app.domains.automation.models import ToolCallLog, WorkflowCheckpoint, WorkflowRun
        from app.domains.jobs.models import JobLead, RawJobLead, UrlImportRun
        from app.agent_runtime.workflows.url_import import (
            UrlImportCommand,
            build_url_import_dependencies,
            run_url_import_workflow,
        )

        with self.Session() as session:
            fetcher = FakeFetcher()
            provider = FakeSocialProvider()
            result = run_url_import_workflow(
                UrlImportCommand(url="https://career.example.com/jobs/java?utm_source=xhs"),
                dependencies=build_url_import_dependencies(
                    session,
                    fetchers={"HTTPArticleFetcher": fetcher},
                    social_provider=provider,
                ),
            )
            session.commit()

            workflow = session.get(WorkflowRun, result.workflow_run.id)
            url_run = session.get(UrlImportRun, result.url_import_run.id)
            raw_items = session.scalars(select(RawJobLead)).all()
            leads = session.scalars(select(JobLead)).all()
            checkpoints = session.scalars(select(WorkflowCheckpoint)).all()
            tool_logs = session.scalars(select(ToolCallLog).order_by(ToolCallLog.created_at)).all()

        self.assertEqual("completed", workflow.status)
        self.assertEqual("completed", workflow.current_step)
        self.assertEqual("succeeded", url_run.status)
        self.assertEqual("completed", url_run.current_stage)
        self.assertEqual("https://career.example.com/jobs/java", url_run.normalized_url)
        self.assertEqual("official_career_site", url_run.source_type)
        self.assertEqual(raw_items[0].id, url_run.raw_job_lead_id)
        self.assertEqual(1, url_run.extracted_count)
        self.assertEqual(1, len(raw_items))
        self.assertEqual("extracted", raw_items[0].status)
        self.assertEqual(1, len(leads))
        self.assertEqual("JD.com", leads[0].company_name)
        self.assertEqual("unverified", leads[0].verification_status)
        self.assertEqual(1, fetcher.calls)
        self.assertEqual(1, provider.calls)
        self.assertIn("normalize_url", {checkpoint.checkpoint_key for checkpoint in checkpoints})
        self.assertIn("save_job_leads", {checkpoint.checkpoint_key for checkpoint in checkpoints})
        self.assertEqual(["HTTPArticleFetcher", "BailianJobLeadExtractor"], [log.tool_name for log in tool_logs])
        self.assertEqual(["succeeded", "succeeded"], [log.status for log in tool_logs])

    def test_llm_failure_preserves_raw_and_marks_run_recoverable(self):
        from app.domains.automation.models import ToolCallLog, WorkflowRun
        from app.domains.jobs.models import RawJobLead, UrlImportRun
        from app.agent_runtime.workflows.url_import import (
            UrlImportCommand,
            build_url_import_dependencies,
            run_url_import_workflow,
        )

        with self.Session() as session:
            result = run_url_import_workflow(
                UrlImportCommand(url="https://career.example.com/jobs/java"),
                dependencies=build_url_import_dependencies(
                    session,
                    fetchers={"HTTPArticleFetcher": FakeFetcher()},
                    social_provider=FakeSocialProvider(fail=True),
                ),
            )
            session.commit()

            workflow = session.get(WorkflowRun, result.workflow_run.id)
            url_run = session.get(UrlImportRun, result.url_import_run.id)
            raw_items = session.scalars(select(RawJobLead)).all()
            tool_logs = session.scalars(select(ToolCallLog).order_by(ToolCallLog.created_at)).all()

        self.assertEqual("failed_recoverable", workflow.status)
        self.assertEqual("partial", url_run.status)
        self.assertEqual("extract_job_leads", url_run.current_stage)
        self.assertEqual("LLM_EXTRACTION_FAILED", url_run.error_code)
        self.assertEqual(raw_items[0].id, url_run.raw_job_lead_id)
        self.assertEqual("captured", raw_items[0].status)
        self.assertEqual(0, url_run.extracted_count)
        self.assertEqual(["succeeded", "failed"], [log.status for log in tool_logs])
        self.assertEqual("LLM_EXTRACTION_FAILED", tool_logs[-1].output_payload["error_code"])

    def test_resume_uses_saved_raw_checkpoint_without_refetching(self):
        from app.domains.automation.models import ToolCallLog, WorkflowRun
        from app.domains.jobs.models import JobLead, RawJobLead, UrlImportRun
        from app.agent_runtime.workflows.url_import import (
            UrlImportCommand,
            build_url_import_dependencies,
            resume_url_import_workflow,
            run_url_import_workflow,
        )

        with self.Session() as session:
            fetcher = FakeFetcher()
            failed = run_url_import_workflow(
                UrlImportCommand(url="https://career.example.com/jobs/java"),
                dependencies=build_url_import_dependencies(
                    session,
                    fetchers={"HTTPArticleFetcher": fetcher},
                    social_provider=FakeSocialProvider(fail=True),
                ),
            )
            resumed = resume_url_import_workflow(
                failed.url_import_run.id,
                dependencies=build_url_import_dependencies(
                    session,
                    fetchers={"HTTPArticleFetcher": fetcher},
                    social_provider=FakeSocialProvider(),
                ),
            )
            session.commit()

            workflow = session.get(WorkflowRun, resumed.workflow_run.id)
            url_run = session.get(UrlImportRun, resumed.url_import_run.id)
            raw_item = session.scalars(select(RawJobLead)).one()
            leads = session.scalars(select(JobLead)).all()
            tool_logs = session.scalars(select(ToolCallLog).order_by(ToolCallLog.created_at)).all()

        self.assertEqual(1, fetcher.calls)
        self.assertEqual("completed", workflow.status)
        self.assertEqual("succeeded", url_run.status)
        self.assertEqual("extracted", raw_item.status)
        self.assertEqual(1, len(leads))
        self.assertEqual(["HTTPArticleFetcher", "BailianJobLeadExtractor", "BailianJobLeadExtractor"], [log.tool_name for log in tool_logs])

    def test_duplicate_url_creates_duplicate_run_without_fetching_again(self):
        from app.domains.jobs.models import UrlImportRun
        from app.agent_runtime.workflows.url_import import (
            UrlImportCommand,
            build_url_import_dependencies,
            run_url_import_workflow,
        )

        with self.Session() as session:
            first_fetcher = FakeFetcher()
            first = run_url_import_workflow(
                UrlImportCommand(url="https://career.example.com/jobs/java?utm_source=xhs"),
                dependencies=build_url_import_dependencies(
                    session,
                    fetchers={"HTTPArticleFetcher": first_fetcher},
                    social_provider=FakeSocialProvider(),
                ),
            )
            second_fetcher = FakeFetcher()
            second = run_url_import_workflow(
                UrlImportCommand(url="https://career.example.com/jobs/java?utm_campaign=campus"),
                dependencies=build_url_import_dependencies(
                    session,
                    fetchers={"HTTPArticleFetcher": second_fetcher},
                    social_provider=FakeSocialProvider(),
                ),
            )
            session.commit()

            runs = session.scalars(select(UrlImportRun).order_by(UrlImportRun.started_at)).all()

        self.assertEqual("succeeded", first.url_import_run.status)
        self.assertEqual("duplicate", second.url_import_run.status)
        self.assertEqual(first.url_import_run.id, second.url_import_run.duplicate_of_run_id)
        self.assertEqual(1, first_fetcher.calls)
        self.assertEqual(0, second_fetcher.calls)
        self.assertEqual(2, len(runs))

    def test_url_import_graph_invokes_workflow(self):
        from app.agent_runtime.workflows.url_import import (
            UrlImportCommand,
            build_url_import_dependencies,
            build_url_import_graph,
        )

        with self.Session() as session:
            graph = build_url_import_graph(
                dependencies=build_url_import_dependencies(
                    session,
                    fetchers={"HTTPArticleFetcher": FakeFetcher()},
                    social_provider=FakeSocialProvider(),
                )
            )
            state = graph.invoke({"command": UrlImportCommand(url="https://career.example.com/jobs/java")})
            session.commit()

        self.assertEqual("succeeded", state["result"].url_import_run.status)


if __name__ == "__main__":
    unittest.main()
