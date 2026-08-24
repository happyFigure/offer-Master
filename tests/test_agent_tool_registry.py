from unittest import TestCase


class AgentToolRegistryTest(TestCase):
    def test_default_registry_registers_memory_and_content_source_tools_in_stable_order(self) -> None:
        from app.agent_runtime.tool_registry import create_default_agent_tool_registry

        registry = create_default_agent_tool_registry()
        definitions = registry.list_definitions()

        self.assertEqual(
            [
                "applications.find_apply_entry",
                "external.web_search",
                "local.company_database_overview",
                "local.job_source_overview",
                "memory_get",
                "memory_search",
                "offerio.sync_company_jobs",
                "sessions_history",
                "sessions_search",
                "weixin-articles-mcp.read_article",
                "xiaohongshu-mcp.get_feed_detail",
                "xiaohongshu-mcp.search_feeds",
            ],
            [definition.name for definition in definitions],
        )
        self.assertTrue(all(definition.input_schema for definition in definitions))
        self.assertTrue(all(definition.output_schema for definition in definitions))
        self.assertTrue(all(not definition.requires_confirmation for definition in definitions))

    def test_external_web_search_normalizes_cristiano_ronaldo_alias_before_executor(self) -> None:
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL, create_external_web_search_agent_tool_definitions

        captured = []

        def fake_executor(query: str, max_results: int):
            captured.append({"query": query, "max_results": max_results})
            return {"executor_name": "fake-search", "answer": "ok", "sources": []}

        definition = create_external_web_search_agent_tool_definitions(external_web_search_executor=fake_executor)[0]

        result = definition.handler(None, query="C罗 2026年8月24日到30日 比赛日程", max_results=5)

        self.assertTrue(result["ok"])
        self.assertEqual(EXTERNAL_WEB_SEARCH_TOOL, result["tool_name"])
        self.assertEqual(1, len(captured))
        self.assertIn("Cristiano Ronaldo", captured[0]["query"])
        self.assertIn("Al Nassr", captured[0]["query"])
        self.assertIn("football", captured[0]["query"])
        self.assertIn("C罗", result["result"]["original_query"])
        self.assertEqual(captured[0]["query"], result["result"]["query"])

    def test_local_company_database_overview_tool_reads_counts_without_writing(self) -> None:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from app.agent_runtime.tool_registry import (
            LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL,
            create_local_company_database_agent_tool_definitions,
        )
        from app.db.base import Base
        import app.domains.agent_memory.models  # noqa: F401
        import app.domains.applications.models  # noqa: F401
        import app.domains.automation.models  # noqa: F401
        import app.domains.conversations.models  # noqa: F401
        from app.domains.jobs.models import (
            Company,
            Job,
            JobLead,
            JobLeadStatus,
            JobSource,
            JobSourceFetchMode,
            JobSourceTrustLevel,
            JobSourceType,
            RecruitingSignal,
            RecruitingSignalType,
        )

        engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine, expire_on_commit=False, future=True)

        try:
            with Session() as session:
                source = JobSource(
                    name="Local recruiting source",
                    source_type=JobSourceType.OFFICIAL_API,
                    entry_url="https://example.com/jobs",
                    trust_level=JobSourceTrustLevel.MEDIUM_HIGH,
                    fetch_mode=JobSourceFetchMode.OFFICIAL_API,
                )
                tencent = Company(name="Tencent", normalized_name="tencent")
                alibaba = Company(name="Alibaba", normalized_name="alibaba")
                session.add_all(
                    [
                        source,
                        tencent,
                        alibaba,
                        Job(
                            company=tencent,
                            title="Backend Engineer Intern",
                            source="manual",
                            source_job_id="job-1",
                            skills=[],
                        ),
                        JobLead(
                            source=source,
                            lead_hash="lead-1",
                            company_name="Tencent",
                            title="Tencent 校招岗位聚合",
                            skills=[],
                            trust_level=JobSourceTrustLevel.MEDIUM_HIGH,
                            verification_status=JobLeadStatus.VERIFIED,
                        ),
                        JobLead(
                            source=source,
                            lead_hash="lead-2",
                            company_name="ByteDance",
                            title="ByteDance 校招岗位聚合",
                            skills=[],
                            trust_level=JobSourceTrustLevel.MEDIUM_HIGH,
                            verification_status=JobLeadStatus.VERIFIED,
                        ),
                        RecruitingSignal(
                            source=source,
                            signal_hash="signal-1",
                            company_name="Meituan",
                            normalized_company_name="meituan",
                            signal_type=RecruitingSignalType.CAMPUS_RECRUITMENT_OPEN,
                            trust_level=JobSourceTrustLevel.MEDIUM_HIGH,
                        ),
                        RecruitingSignal(
                            source=source,
                            signal_hash="signal-2",
                            company_name="Tencent",
                            normalized_company_name="tencent",
                            signal_type=RecruitingSignalType.CAMPUS_RECRUITMENT_OPEN,
                            trust_level=JobSourceTrustLevel.MEDIUM_HIGH,
                        ),
                    ]
                )
                session.commit()

                definitions = {definition.name: definition for definition in create_local_company_database_agent_tool_definitions()}
                result = definitions[LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL].handler(session, sample_limit=5)
        finally:
            engine.dispose()

        self.assertTrue(result["ok"])
        self.assertEqual(LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL, result["tool_name"])
        self.assertEqual(
            {
                "company_count": 2,
                "job_count": 1,
                "job_lead_count": 2,
                "job_lead_company_count": 2,
                "recruiting_signal_count": 2,
                "recruiting_signal_company_count": 2,
            },
            {key: result["result"][key] for key in (
                "company_count",
                "job_count",
                "job_lead_count",
                "job_lead_company_count",
                "recruiting_signal_count",
                "recruiting_signal_company_count",
            )},
        )
        self.assertEqual(["Alibaba", "Tencent"], result["result"]["sample_companies"])
        self.assertEqual(["ByteDance", "Tencent"], result["result"]["sample_lead_companies"])
        self.assertEqual(["Meituan", "Tencent"], result["result"]["sample_signal_companies"])
        self.assertEqual(
            [
                {
                    "tier": "正式企业",
                    "company_name": "Alibaba",
                    "known_info": "企业档案",
                    "quantity": "0 条岗位",
                    "status": "可补充岗位后用于推荐",
                },
                {
                    "tier": "正式企业",
                    "company_name": "Tencent",
                    "known_info": "企业档案、正式岗位、岗位线索、校招来源",
                    "quantity": "1 条岗位，1 条线索，1 条来源",
                    "status": "可用于推荐",
                },
                {
                    "tier": "岗位线索企业",
                    "company_name": "ByteDance",
                    "known_info": "岗位线索",
                    "quantity": "1 条线索",
                    "status": "待补全企业档案",
                },
                {
                    "tier": "校招来源企业",
                    "company_name": "Meituan",
                    "known_info": "校招来源",
                    "quantity": "1 条来源",
                    "status": "可继续验证",
                },
            ],
            result["result"]["company_rows"],
        )

    def test_local_company_database_summary_renders_company_rows_as_markdown_table(self) -> None:
        from app.agent_runtime.graph_factory import _company_database_overview_summary_response

        response = _company_database_overview_summary_response(
            {
                "status": "succeeded",
                "result": {
                    "ok": True,
                    "result": {
                        "company_count": 2,
                        "job_count": 1,
                        "job_lead_count": 2,
                        "job_lead_company_count": 2,
                        "recruiting_signal_count": 2,
                        "recruiting_signal_company_count": 2,
                        "company_rows": [
                            {
                                "tier": "正式企业",
                                "company_name": "Tencent",
                                "known_info": "企业档案、正式岗位",
                                "quantity": "1 条岗位",
                                "status": "可用于推荐",
                            },
                            {
                                "tier": "岗位线索企业",
                                "company_name": "ByteDance",
                                "known_info": "岗位线索",
                                "quantity": "1 条线索",
                                "status": "待补全企业档案",
                            },
                        ],
                    },
                },
            }
        )

        self.assertIn("| 档次 | 公司 | 已有信息 | 数量 | 状态 |", response)
        self.assertIn("| --- | --- | --- | --- | --- |", response)
        self.assertIn("| 正式企业 | Tencent | 企业档案、正式岗位 | 1 条岗位 | 可用于推荐 |", response)
        self.assertIn("| 岗位线索企业 | ByteDance | 岗位线索 | 1 条线索 | 待补全企业档案 |", response)

    def test_local_job_source_overview_tool_reads_local_sources_and_offerio_board_totals(self) -> None:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from app.agent_runtime.tool_registry import (
            LOCAL_JOB_SOURCE_OVERVIEW_TOOL,
            create_local_job_source_agent_tool_definitions,
        )
        from app.db.base import Base
        import app.domains.agent_memory.models  # noqa: F401
        import app.domains.applications.models  # noqa: F401
        import app.domains.automation.models  # noqa: F401
        import app.domains.conversations.models  # noqa: F401
        from app.domains.jobs.models import JobSource, JobSourceFetchMode, JobSourceTrustLevel, JobSourceType
        from app.domains.jobs.providers.offerio import OfferIOPage

        class FakeOfferIOProvider:
            def list_company_openings(self, **kwargs):
                return OfferIOPage(items=[], page=kwargs["page"], page_size=kwargs["page_size"], total=1247, total_pages=1247)

            def list_companies(self, **kwargs):
                return OfferIOPage(items=[], page=kwargs["page"], page_size=kwargs["page_size"], total=987, total_pages=987)

        engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine, expire_on_commit=False, future=True)

        try:
            with Session() as session:
                session.add_all(
                    [
                        JobSource(
                            name="OfferIO 公司聚合岗位库",
                            source_type=JobSourceType.OFFICIAL_API,
                            entry_url="https://offerio.work/api/recruitment/job-companies?page=1&pageSize=50",
                            enabled=True,
                            trust_level=JobSourceTrustLevel.MEDIUM_HIGH,
                            fetch_mode=JobSourceFetchMode.OFFICIAL_API,
                        ),
                        JobSource(
                            name="高校就业网",
                            source_type=JobSourceType.UNIVERSITY_CAREER_SITE,
                            entry_url="https://career.example.edu/jobs",
                            enabled=True,
                            trust_level=JobSourceTrustLevel.HIGH,
                            fetch_mode=JobSourceFetchMode.PUBLIC_HTML,
                        ),
                        JobSource(
                            name="旧公众号来源",
                            source_type=JobSourceType.WECHAT_ACCOUNT,
                            enabled=False,
                            trust_level=JobSourceTrustLevel.MEDIUM,
                            fetch_mode=JobSourceFetchMode.MCP_VISIBLE_PAGE,
                        ),
                    ]
                )
                session.commit()

                definitions = {
                    definition.name: definition
                    for definition in create_local_job_source_agent_tool_definitions(offerio_provider_factory=FakeOfferIOProvider)
                }
                result = definitions[LOCAL_JOB_SOURCE_OVERVIEW_TOOL].handler(session, sample_limit=5)
        finally:
            engine.dispose()

        self.assertTrue(result["ok"])
        self.assertEqual(LOCAL_JOB_SOURCE_OVERVIEW_TOOL, result["tool_name"])
        self.assertEqual(3, result["result"]["source_count"])
        self.assertEqual(2, result["result"]["enabled_source_count"])
        self.assertEqual(1, result["result"]["disabled_source_count"])
        self.assertEqual(3, result["result"]["unsynced_source_count"])
        self.assertEqual(1247, result["result"]["external_job_board"]["offerio_company_openings_total"])
        self.assertEqual(987, result["result"]["external_job_board"]["offerio_company_jobs_total"])
        self.assertEqual("official_api", result["result"]["sample_sources"][0]["source_type"])

    def test_application_find_apply_entry_tool_queues_external_task(self) -> None:
        from sqlalchemy import create_engine, select
        from sqlalchemy.orm import sessionmaker

        from app.agent_runtime.external_tasks.models import ExternalAgentTask, ExternalAgentTaskEvent
        from app.agent_runtime.tool_registry import APPLICATION_FIND_APPLY_ENTRY_TOOL, create_application_agent_tool_definitions
        from app.db.base import Base
        import app.domains.agent_memory.models  # noqa: F401
        import app.domains.applications.models  # noqa: F401
        import app.domains.automation.models  # noqa: F401
        import app.domains.conversations.models  # noqa: F401
        import app.domains.jobs.models  # noqa: F401

        engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine, expire_on_commit=False, future=True)

        try:
            with Session() as session:
                definitions = {
                    definition.name: definition
                    for definition in create_application_agent_tool_definitions()
                }
                result = definitions[APPLICATION_FIND_APPLY_ENTRY_TOOL].handler(
                    session,
                    task_id="task-from-tool-1",
                    trace_id="trace-from-tool-1",
                    job_id="job-lead-1",
                    company_name="Tencent",
                    title="Backend Engineer Intern",
                    source_url="https://careers.tencent.com/job/1",
                    apply_url_candidate="https://careers.tencent.com/apply/1",
                    profile_id="default",
                    resume_version_id="resume-v3",
                )
                session.commit()

                task = session.scalars(select(ExternalAgentTask)).one()
                events = list(session.scalars(select(ExternalAgentTaskEvent)).all())
        finally:
            engine.dispose()

        self.assertTrue(result["ok"])
        self.assertEqual(APPLICATION_FIND_APPLY_ENTRY_TOOL, result["tool_name"])
        self.assertEqual("task-from-tool-1", result["result"]["task_id"])
        self.assertEqual("queued", result["result"]["status"])
        self.assertEqual("external_agent_dispatch", result["result"]["next_action"])
        self.assertEqual("task-from-tool-1", task.id)
        self.assertEqual("find_apply_entry", task.task_type)
        self.assertEqual("trace-from-tool-1", task.trace_id)
        self.assertEqual("Tencent", task.input_payload["job"]["company_name"])
        self.assertEqual(["task_queued"], [event.event_type for event in events])

    def test_application_find_apply_entry_tool_can_dispatch_queued_task(self) -> None:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        import app.agent_runtime.external_tasks.models  # noqa: F401
        from app.agent_runtime.tool_registry import APPLICATION_FIND_APPLY_ENTRY_TOOL, create_application_agent_tool_definitions
        from app.db.base import Base
        import app.domains.agent_memory.models  # noqa: F401
        import app.domains.applications.models  # noqa: F401
        import app.domains.automation.models  # noqa: F401
        import app.domains.conversations.models  # noqa: F401
        import app.domains.jobs.models  # noqa: F401

        dispatched = []

        def fake_dispatcher(_session, task_id):
            dispatched.append(task_id)
            return {
                "ok": True,
                "executor_name": "claude-sdk-agent",
                "task_id": task_id,
                "status": "succeeded",
                "result_status": "found_opened",
                "apply_url": "https://careers.tencent.com/apply/1",
                "next_action": "wait_user_review",
            }

        engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine, expire_on_commit=False, future=True)

        try:
            with Session() as session:
                definitions = {
                    definition.name: definition
                    for definition in create_application_agent_tool_definitions(external_task_dispatcher=fake_dispatcher)
                }
                result = definitions[APPLICATION_FIND_APPLY_ENTRY_TOOL].handler(
                    session,
                    task_id="task-dispatch-from-tool-1",
                    trace_id="trace-dispatch-from-tool-1",
                    job_id="job-lead-1",
                    company_name="Tencent",
                    title="Backend Engineer Intern",
                    source_url="https://careers.tencent.com/job/1",
                    profile_id="default",
                    resume_version_id="resume-v3",
                )
        finally:
            engine.dispose()

        self.assertEqual(["task-dispatch-from-tool-1"], dispatched)
        self.assertEqual("external_agent_completed", result["result"]["next_action"])
        self.assertEqual("claude-sdk-agent", result["result"]["dispatch"]["executor_name"])
        self.assertEqual("https://careers.tencent.com/apply/1", result["result"]["dispatch"]["apply_url"])
        self.assertEqual("succeeded", result["result"]["result_envelope"]["status"])
        self.assertEqual(APPLICATION_FIND_APPLY_ENTRY_TOOL, result["result"]["result_envelope"]["capability"])
        self.assertEqual("claude-sdk-agent", result["result"]["result_envelope"]["executor"])
        self.assertIn("Tencent - Backend Engineer Intern", result["result"]["result_envelope"]["summary"])
        self.assertTrue(result["result"]["result_envelope"]["requires_user_action"])

    def test_offerio_company_jobs_tool_creates_default_source_and_imports_leads(self) -> None:
        from sqlalchemy import create_engine, select
        from sqlalchemy.orm import sessionmaker

        from app.agent_runtime.tool_registry import create_job_source_agent_tool_definitions
        from app.db.base import Base
        import app.domains.agent_memory.models  # noqa: F401
        import app.domains.applications.models  # noqa: F401
        import app.domains.automation.models  # noqa: F401
        import app.domains.conversations.models  # noqa: F401
        from app.domains.jobs.models import JobLead, JobSource, RawJobLead, SourceSyncRun
        from app.domains.jobs.providers.offerio import OfferIOCompany, OfferIOPage

        class FakeOfferIOProvider:
            def __init__(self) -> None:
                self.kwargs = None

            def list_companies(self, **kwargs):
                self.kwargs = kwargs
                return OfferIOPage(
                    items=[
                        OfferIOCompany(
                            name="Tencent",
                            company_nature="private",
                            industry="internet/software",
                            locations="Shenzhen",
                            job_count=98,
                            updated_at="2026-08-17",
                            raw_payload={"company": "Tencent"},
                        )
                    ],
                    page=1,
                    page_size=kwargs["page_size"],
                    total=1,
                    total_pages=1,
                )

        provider = FakeOfferIOProvider()
        engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine, expire_on_commit=False, future=True)

        try:
            with Session() as session:
                definitions = {
                    definition.name: definition
                    for definition in create_job_source_agent_tool_definitions(offerio_provider_factory=lambda: provider)
                }

                result = definitions["offerio.sync_company_jobs"].handler(session, limit=25)
                session.commit()

                source = session.scalars(select(JobSource)).one()
                run = session.scalars(select(SourceSyncRun)).one()
                raw_lead = session.scalars(select(RawJobLead)).one()
                lead = session.scalars(select(JobLead)).one()
        finally:
            engine.dispose()

        self.assertTrue(result["ok"])
        self.assertEqual("offerio.sync_company_jobs", result["tool_name"])
        self.assertEqual("succeeded", result["result"]["status"])
        self.assertEqual(1, result["result"]["extracted_count"])
        self.assertEqual({"job_type": "校招", "page": 1, "page_size": 50}, provider.kwargs)
        self.assertEqual("OfferIO 公司聚合岗位库", source.name)
        self.assertEqual("https://offerio.work/api/recruitment/job-companies?jobType=校招&page=1&pageSize=50", source.entry_url)
        self.assertEqual("succeeded", run.status)
        self.assertEqual("application/json", raw_lead.content_type)
        self.assertEqual("Tencent", lead.company_name)
        self.assertEqual("Tencent 校招岗位聚合（98 个）", lead.title)

    def test_offerio_company_jobs_tool_reuses_existing_chinese_source(self) -> None:
        from sqlalchemy import create_engine, select
        from sqlalchemy.orm import sessionmaker

        from app.agent_runtime.tool_registry import OFFERIO_COMPANY_JOBS_ENTRY_URL, create_job_source_agent_tool_definitions
        from app.db.base import Base
        import app.domains.agent_memory.models  # noqa: F401
        import app.domains.applications.models  # noqa: F401
        import app.domains.automation.models  # noqa: F401
        import app.domains.conversations.models  # noqa: F401
        from app.domains.jobs.models import JobSource, JobSourceFetchMode, JobSourceTrustLevel, JobSourceType
        from app.domains.jobs.providers.offerio import OfferIOCompany, OfferIOPage

        class FakeOfferIOProvider:
            def list_companies(self, **kwargs):
                return OfferIOPage(
                    items=[
                        OfferIOCompany(
                            name="Alibaba",
                            company_nature="private",
                            industry="internet/software",
                            locations="Hangzhou",
                            job_count=42,
                            updated_at="2026-08-17",
                            raw_payload={"company": "Alibaba"},
                        )
                    ],
                    page=1,
                    page_size=kwargs["page_size"],
                    total=1,
                    total_pages=1,
                )

        engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine, expire_on_commit=False, future=True)

        try:
            with Session() as session:
                existing_source = JobSource(
                    name="OfferIO 公司聚合岗位库",
                    source_type=JobSourceType.OFFICIAL_API,
                    entry_url=OFFERIO_COMPANY_JOBS_ENTRY_URL,
                    trust_level=JobSourceTrustLevel.MEDIUM_HIGH,
                    fetch_mode=JobSourceFetchMode.OFFICIAL_API,
                    raw_payload={"configured_by": "enabled_sources"},
                )
                session.add(existing_source)
                session.flush()
                existing_source_id = existing_source.id
                definitions = {
                    definition.name: definition
                    for definition in create_job_source_agent_tool_definitions(offerio_provider_factory=FakeOfferIOProvider)
                }

                result = definitions["offerio.sync_company_jobs"].handler(session, limit=50)
                session.commit()

                sources = list(session.scalars(select(JobSource).order_by(JobSource.name)).all())
        finally:
            engine.dispose()

        self.assertTrue(result["ok"])
        self.assertEqual(existing_source_id, result["result"]["source_id"])
        self.assertEqual("OfferIO 公司聚合岗位库", result["result"]["source_name"])
        self.assertEqual(["OfferIO 公司聚合岗位库"], [source.name for source in sources])

    def test_content_source_tool_definitions_call_their_adapter_methods(self) -> None:
        from app.agent_runtime.tool_registry import create_content_source_agent_tool_definitions
        from app.mcp_gateway.client import MCPToolCallResult

        class FakeContentSourceClient:
            def __init__(self) -> None:
                self.calls = []

            def read_weixin_article(self, *, url: str) -> MCPToolCallResult:
                self.calls.append(("read_weixin_article", {"url": url}))
                return MCPToolCallResult(
                    tool_name="weixin-articles-mcp.read_article",
                    ok=True,
                    result={"title": "Tencent 2027"},
                )

            def search_xiaohongshu_feeds(self, *, keyword: str, filters: dict | None = None) -> MCPToolCallResult:
                self.calls.append(("search_xiaohongshu_feeds", {"keyword": keyword, "filters": filters}))
                return MCPToolCallResult(
                    tool_name="xiaohongshu-mcp.search_feeds",
                    ok=True,
                    result={"items": [{"title": "2027 秋招"}]},
                )

            def get_xiaohongshu_feed_detail(self, **arguments) -> MCPToolCallResult:
                self.calls.append(("get_xiaohongshu_feed_detail", arguments))
                return MCPToolCallResult(
                    tool_name="xiaohongshu-mcp.get_feed_detail",
                    ok=True,
                    result={"feed_id": arguments["feed_id"], "text": "招聘信息"},
                )

        fake_client = FakeContentSourceClient()
        definitions = {definition.name: definition for definition in create_content_source_agent_tool_definitions(fake_client)}

        self.assertEqual(
            {"weixin-articles-mcp.read_article", "xiaohongshu-mcp.search_feeds", "xiaohongshu-mcp.get_feed_detail"},
            set(definitions),
        )

        wechat_result = definitions["weixin-articles-mcp.read_article"].handler(None, url="https://mp.weixin.qq.com/s/example")
        search_result = definitions["xiaohongshu-mcp.search_feeds"].handler(None, keyword="2027 秋招 Java")
        detail_result = definitions["xiaohongshu-mcp.get_feed_detail"].handler(
            None,
            feed_id="abc",
            xsec_token="token",
        )

        self.assertTrue(wechat_result.ok)
        self.assertTrue(search_result.ok)
        self.assertTrue(detail_result.ok)
        self.assertEqual(
            [
                ("read_weixin_article", {"url": "https://mp.weixin.qq.com/s/example"}),
                ("search_xiaohongshu_feeds", {"keyword": "2027 秋招 Java", "filters": None}),
                ("get_xiaohongshu_feed_detail", {"feed_id": "abc", "xsec_token": "token"}),
            ],
            fake_client.calls,
        )

    def test_content_source_client_delegates_xiaohongshu_tools_to_mcp_gateway(self) -> None:
        from app.mcp_gateway.client import MCPToolCallResult
        from app.mcp_gateway.content_source_client import ContentSourceMCPClient

        class FakeMCPGatewayClient:
            def __init__(self) -> None:
                self.calls = []

            def call_tool(self, *, tool_name: str, arguments: dict) -> MCPToolCallResult:
                self.calls.append({"tool_name": tool_name, "arguments": arguments})
                return MCPToolCallResult(tool_name=tool_name, ok=True, result={"ok": True})

        fake_gateway = FakeMCPGatewayClient()
        client = ContentSourceMCPClient(mcp_client=fake_gateway)

        search_result = client.search_xiaohongshu_feeds(keyword="2027 autumn recruit")
        detail_result = client.get_xiaohongshu_feed_detail(feed_id="abc", xsec_token="token")

        self.assertTrue(search_result.ok)
        self.assertTrue(detail_result.ok)
        self.assertEqual(
            [
                {"tool_name": "xiaohongshu-mcp.search_feeds", "arguments": {"keyword": "2027 autumn recruit", "filters": None}},
                {"tool_name": "xiaohongshu-mcp.get_feed_detail", "arguments": {"feed_id": "abc", "xsec_token": "token"}},
            ],
            fake_gateway.calls,
        )

    def test_content_source_client_calls_xiaohongshu_rest_api_when_base_url_configured(self) -> None:
        from app.mcp_gateway.content_source_client import ContentSourceMCPClient

        class FakeResponse:
            status_code = 200

            def __init__(self, payload: dict) -> None:
                self._payload = payload

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return self._payload

        class FakeHTTPClient:
            def __init__(self) -> None:
                self.calls = []

            def post(self, url: str, *, json: dict, headers: dict, timeout: float) -> FakeResponse:
                self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
                return FakeResponse({"success": True, "data": {"echo": json}, "message": "ok"})

        http_client = FakeHTTPClient()
        client = ContentSourceMCPClient(
            xiaohongshu_base_url="http://127.0.0.1:18060/",
            xiaohongshu_auth_token="secret-token",
            http_client=http_client,
        )

        search_result = client.search_xiaohongshu_feeds(keyword="2027 autumn recruit", filters={"sort_by": "latest"})
        detail_result = client.get_xiaohongshu_feed_detail(feed_id="abc", xsec_token="token", include_comments=True, comment_limit=20)

        self.assertTrue(search_result.ok)
        self.assertTrue(detail_result.ok)
        self.assertEqual(
            [
                {
                    "url": "http://127.0.0.1:18060/api/v1/feeds/search",
                    "json": {"keyword": "2027 autumn recruit", "filters": {"sort_by": "latest"}},
                    "headers": {"Authorization": "Bearer secret-token"},
                    "timeout": 30.0,
                },
                {
                    "url": "http://127.0.0.1:18060/api/v1/feeds/detail",
                    "json": {"feed_id": "abc", "xsec_token": "token", "load_all_comments": True, "limit": 20},
                    "headers": {"Authorization": "Bearer secret-token"},
                    "timeout": 30.0,
                },
            ],
            http_client.calls,
        )

    def test_guard_blocks_unregistered_tool_with_structured_error(self) -> None:
        from app.agent_runtime.guardrails import AgentToolCallContext, AgentToolRuntimeGuard
        from app.agent_runtime.tool_registry import AgentToolRegistry

        result = AgentToolRuntimeGuard().pre_check(
            AgentToolCallContext(stage="plan", tool_name="unknown_tool", source_type="agent_chat"),
            registry=AgentToolRegistry(),
        )

        self.assertFalse(result.ok)
        self.assertEqual("TOOL_NOT_REGISTERED", result.error_code)
        self.assertEqual("plan", result.stage)
        self.assertEqual("unknown_tool", result.tool_name)
        self.assertEqual("stop", result.next_action)
        self.assertIn("unknown_tool", result.reason)
        self.assertIn("not registered", result.user_message)

    def test_guard_blocks_tool_budget_exceeded(self) -> None:
        from app.agent_runtime.guardrails import AgentToolCallContext, AgentToolPolicy, AgentToolRuntimeGuard
        from app.agent_runtime.tool_registry import create_default_agent_tool_registry

        result = AgentToolRuntimeGuard(policy=AgentToolPolicy(max_tool_calls=10)).pre_check(
            AgentToolCallContext(
                stage="recall",
                tool_name="sessions_search",
                source_type="agent_chat",
                tool_call_count=10,
            ),
            registry=create_default_agent_tool_registry(),
        )

        self.assertFalse(result.ok)
        self.assertEqual("TOOL_BUDGET_EXCEEDED", result.error_code)
        self.assertEqual("stop", result.next_action)
        self.assertEqual({"tool_calls": 10, "max_tool_calls": 10}, result.cost)
        self.assertEqual("max_tool_calls", result.error_details["budget_name"])

    def test_guard_blocks_source_type_not_allowed(self) -> None:
        from app.agent_runtime.guardrails import AgentToolCallContext, AgentToolRuntimeGuard
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry

        registry = AgentToolRegistry()
        registry.register(
            AgentToolDefinition(
                name="wechat_visible_page",
                description="Read a user-visible WeChat page.",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                allowed_source_types=frozenset({"wechat_article"}),
            )
        )

        result = AgentToolRuntimeGuard().pre_check(
            AgentToolCallContext(
                stage="fetch",
                tool_name="wechat_visible_page",
                source_type="xiaohongshu_note",
            ),
            registry=registry,
        )

        self.assertFalse(result.ok)
        self.assertEqual("TOOL_SOURCE_TYPE_NOT_ALLOWED", result.error_code)
        self.assertEqual("select_alternative_tool", result.next_action)
        self.assertEqual("xiaohongshu_note", result.error_details["source_type"])
        self.assertEqual(["wechat_article"], result.error_details["allowed_source_types"])

    def test_guard_requires_confirmation_for_high_risk_tool(self) -> None:
        from app.agent_runtime.guardrails import AgentToolCallContext, AgentToolRuntimeGuard
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry, AgentToolRiskLevel

        registry = AgentToolRegistry()
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

        blocked = AgentToolRuntimeGuard().pre_check(
            AgentToolCallContext(
                stage="apply",
                tool_name="submit_application",
                source_type="application",
                user_confirmed=False,
            ),
            registry=registry,
        )
        allowed = AgentToolRuntimeGuard().pre_check(
            AgentToolCallContext(
                stage="apply",
                tool_name="submit_application",
                source_type="application",
                user_confirmed=True,
            ),
            registry=registry,
        )

        self.assertFalse(blocked.ok)
        self.assertEqual("TOOL_USER_CONFIRMATION_REQUIRED", blocked.error_code)
        self.assertEqual("request_user_confirmation", blocked.next_action)
        self.assertTrue(allowed.ok)
        self.assertEqual("continue", allowed.next_action)

    def test_mcp_tool_policy_normalizes_allowlist_and_confirmation_boundary(self) -> None:
        from app.mcp_gateway.tool_policy import MCPToolPolicy

        policy = MCPToolPolicy.from_allowlist([" open_page ", "read_page", "fill_form", ""])

        self.assertEqual(["open_page", "read_page", "fill_form"], policy.allowed_tool_names())
        self.assertTrue(policy.is_allowed("open_page"))
        self.assertFalse(policy.is_allowed("unknown_tool"))
        self.assertFalse(policy.requires_confirmation("open_page"))
        self.assertTrue(policy.requires_confirmation("fill_form"))
