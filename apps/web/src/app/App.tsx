import {
  Activity,
  AlertTriangle,
  BadgeCheck,
  BellDot,
  Bot,
  BriefcaseBusiness,
  CheckCircle2,
  ChevronRight,
  CircleDot,
  Clock3,
  DatabaseZap,
  ExternalLink,
  FileSearch,
  Gauge,
  Layers3,
  Loader2,
  LucideIcon,
  MessageCircle,
  Network,
  Plus,
  RadioTower,
  RefreshCcw,
  Search,
  Send,
  ShieldCheck,
  Sparkles,
  Workflow,
} from "lucide-react";
import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";

import { createJobSource, listJobSources, syncJobSource } from "../api/jobSources";
import { extractJobLeads, listJobLeads, verifyAndConvertJobLead, verifyJobLead } from "../api/jobLeads";
import type {
  JobLead,
  JobLeadFilters,
  JobLeadStatus,
  JobSource,
  JobSourceFetchMode,
  JobSourceTrustLevel,
  JobSourceType,
} from "../types/jobs";

type PageId = "chat" | "dashboard" | "sources" | "leads" | "pipeline" | "guardrails";
type NoticeKind = "success" | "warning" | "danger" | "info";

interface Notice {
  kind: NoticeKind;
  message: string;
}

interface SourceDraft {
  name: string;
  source_type: JobSourceType;
  entry_url: string;
  sync_interval_hours: number;
  trust_level: JobSourceTrustLevel;
  fetch_mode: JobSourceFetchMode;
  notes: string;
}

interface LeadFilterDraft {
  keyword: string;
  verification_status: JobLeadStatus | "";
  source_type: JobSourceType | "";
  trust_level: JobSourceTrustLevel | "";
  graduation_year: string;
}

interface ExtractDraft {
  source_id: string;
  source_url: string;
  raw_content: string;
}

interface ChatMessage {
  id: string;
  role: "assistant" | "user";
  content: string;
  meta?: string;
}

const NAV_ITEMS: Array<{ id: PageId; label: string; description: string; icon: LucideIcon }> = [
  { id: "chat", label: "AI 对话", description: "简历/岗位/面试问答", icon: Bot },
  { id: "dashboard", label: "总览", description: "同步态势与下一步", icon: Gauge },
  { id: "sources", label: "信息源", description: "高校/企业/社媒入口", icon: RadioTower },
  { id: "leads", label: "岗位线索", description: "抽取、筛选、验证", icon: FileSearch },
  { id: "pipeline", label: "投递管线", description: "正式岗位与申请", icon: BriefcaseBusiness },
  { id: "guardrails", label: "边界设置", description: "MCP 与确认边界", icon: ShieldCheck },
];

const SOURCE_TYPE_OPTIONS: Array<{ value: JobSourceType; label: string }> = [
  { value: "university_career_site", label: "高校就业网" },
  { value: "official_career_site", label: "企业招聘官网" },
  { value: "xiaohongshu_note", label: "小红书笔记" },
  { value: "wechat_article", label: "公众号文章" },
  { value: "job_board_visible_page", label: "招聘平台可见页" },
  { value: "official_api", label: "官方 API" },
  { value: "manual_clip", label: "手动剪贴" },
];

const FETCH_MODE_OPTIONS: Array<{ value: JobSourceFetchMode; label: string }> = [
  { value: "public_html", label: "公开 HTML" },
  { value: "manual_clip", label: "手动剪贴" },
  { value: "mcp_visible_page", label: "MCP 可见页面" },
  { value: "official_api", label: "官方 API" },
];

const TRUST_OPTIONS: Array<{ value: JobSourceTrustLevel; label: string }> = [
  { value: "high", label: "高" },
  { value: "medium_high", label: "中高" },
  { value: "medium", label: "中" },
  { value: "low_medium", label: "中低" },
  { value: "low", label: "低" },
];

const STATUS_OPTIONS: Array<{ value: JobLeadStatus; label: string }> = [
  { value: "unverified", label: "未验证" },
  { value: "pending_review", label: "待复核" },
  { value: "verified", label: "已验证" },
  { value: "converted", label: "已转正式岗位" },
  { value: "expired", label: "已过期" },
  { value: "invalid", label: "无效" },
];

const INITIAL_SOURCE_DRAFT: SourceDraft = {
  name: "",
  source_type: "university_career_site",
  entry_url: "",
  sync_interval_hours: 24,
  trust_level: "medium_high",
  fetch_mode: "public_html",
  notes: "",
};

const INITIAL_FILTERS: LeadFilterDraft = {
  keyword: "",
  verification_status: "",
  source_type: "",
  trust_level: "",
  graduation_year: "2027",
};

const INITIAL_CHAT_MESSAGES: ChatMessage[] = [
  {
    id: "assistant-welcome",
    role: "assistant",
    content: "我是 OfferMaster 的 AI 求职助手前端壳子。后续接入 Chat API 后，我可以结合你的简历、岗位线索和面试知识库给出建议。",
    meta: "前端预览",
  },
];

const CHAT_PROMPTS = ["帮我分析这个岗位是否适合我", "帮我优化简历项目描述", "模拟 Java 后端面试", "根据岗位生成投递建议"];

function App() {
  const [activePage, setActivePage] = useState<PageId>(() => getInitialPage());
  const [sources, setSources] = useState<JobSource[]>([]);
  const [leads, setLeads] = useState<JobLead[]>([]);
  const [notice, setNotice] = useState<Notice | null>(null);
  const [loading, setLoading] = useState(true);
  const [workingAction, setWorkingAction] = useState<string | null>(null);
  const [sourceDraft, setSourceDraft] = useState<SourceDraft>(INITIAL_SOURCE_DRAFT);
  const [leadFilters, setLeadFilters] = useState<LeadFilterDraft>(INITIAL_FILTERS);
  const [extractDraft, setExtractDraft] = useState<ExtractDraft>({
    source_id: "",
    source_url: "",
    raw_content: "",
  });
  const [chatDraft, setChatDraft] = useState("");
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>(INITIAL_CHAT_MESSAGES);

  const refreshData = useCallback(async (filters?: JobLeadFilters) => {
    const [nextSources, nextLeads] = await Promise.all([
      listJobSources(),
      listJobLeads(filters ?? { graduation_year: INITIAL_FILTERS.graduation_year, limit: 80 }),
    ]);
    setSources(nextSources);
    setLeads(nextLeads);
    setExtractDraft((current) => ({
      ...current,
      source_id: current.source_id || nextSources[0]?.id || "",
    }));
  }, []);

  useEffect(() => {
    const handleHashChange = () => setActivePage(getInitialPage());
    window.addEventListener("hashchange", handleHashChange);
    return () => window.removeEventListener("hashchange", handleHashChange);
  }, []);

  useEffect(() => {
    refreshData()
      .catch((error: unknown) => {
        setNotice({ kind: "warning", message: `后端暂未连接：${toDisplayError(error)}` });
      })
      .finally(() => setLoading(false));
  }, [refreshData]);

  const summary = useMemo(() => buildSummary(sources, leads), [sources, leads]);

  const navigate = (pageId: PageId) => {
    window.location.hash = pageId;
    setActivePage(pageId);
  };

  const runAction = async (actionName: string, action: () => Promise<string>) => {
    setNotice(null);
    setWorkingAction(actionName);

    try {
      const message = await action();
      setNotice({ kind: "success", message });
    } catch (error: unknown) {
      setNotice({ kind: "danger", message: toDisplayError(error) });
    } finally {
      setWorkingAction(null);
    }
  };

  const handleRefresh = () =>
    runAction("refresh", async () => {
      await refreshData(buildLeadFilters(leadFilters));
      return "已刷新岗位信息源与线索列表。";
    });

  const handleCreateSource = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    void runAction("create-source", async () => {
      if (!sourceDraft.name.trim()) {
        throw new Error("请先填写信息源名称。");
      }

      await createJobSource({
        name: sourceDraft.name.trim(),
        source_type: sourceDraft.source_type,
        entry_url: sourceDraft.entry_url.trim() || null,
        enabled: true,
        sync_interval_hours: sourceDraft.sync_interval_hours,
        trust_level: sourceDraft.trust_level,
        fetch_mode: sourceDraft.fetch_mode,
        notes: sourceDraft.notes.trim() || null,
      });
      setSourceDraft(INITIAL_SOURCE_DRAFT);
      await refreshData(buildLeadFilters(leadFilters));
      return "信息源已创建，后续可加入定时同步。";
    });
  };

  const handleSyncSource = (source: JobSource) => {
    void runAction(`sync-${source.id}`, async () => {
      const result = await syncJobSource(source.id, 20);
      await refreshData(buildLeadFilters(leadFilters));
      return `${source.name} 同步完成：抓取 ${result.fetched_count} 条，抽取 ${result.extracted_count} 条。`;
    });
  };

  const handleSearchLeads = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    void runAction("search-leads", async () => {
      setLeads(await listJobLeads(buildLeadFilters(leadFilters)));
      return "岗位线索筛选已更新。";
    });
  };

  const handleExtractLeads = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    void runAction("extract-leads", async () => {
      if (!extractDraft.source_id) {
        throw new Error("请先选择一个信息源。没有信息源时，请先到“信息源”页面创建。");
      }
      if (!extractDraft.raw_content.trim()) {
        throw new Error("请粘贴小红书、公众号、牛客或群消息中的秋招汇总文本。");
      }

      const result = await extractJobLeads({
        source_id: extractDraft.source_id,
        source_url: extractDraft.source_url.trim() || null,
        raw_content: extractDraft.raw_content.trim(),
      });
      setExtractDraft((current) => ({ ...current, raw_content: "" }));
      await refreshData(buildLeadFilters(leadFilters));
      return `已抽取 ${result.extracted_count} 条岗位线索，进入未验证池。`;
    });
  };

  const handleMarkLeadStatus = (lead: JobLead, status: JobLeadStatus) => {
    void runAction(`${status}-${lead.id}`, async () => {
      await verifyJobLead(lead.id, status);
      await refreshData(buildLeadFilters(leadFilters));
      return `${lead.company_name} - ${lead.title} 已标记为${STATUS_LABELS[status]}。`;
    });
  };

  const handleVerifyAndConvert = (lead: JobLead) => {
    void runAction(`convert-${lead.id}`, async () => {
      const result = await verifyAndConvertJobLead(lead.id);
      await refreshData(buildLeadFilters(leadFilters));
      return `${result.job.company.name} - ${result.job.title} 已验证并转入正式岗位。`;
    });
  };

  const handleSendChat = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const content = chatDraft.trim();

    if (!content) {
      return;
    }

    const timestamp = Date.now();
    setChatMessages((current) => [
      ...current,
      { id: `user-${timestamp}`, role: "user", content, meta: "你" },
      {
        id: `assistant-placeholder-${timestamp}`,
        role: "assistant",
        content: "聊天后端还没有接入。这里先保留完整的对话页面交互，后续做到 Chat API、Conversation/Message 持久化后再返回真实模型回复。",
        meta: "占位回复",
      },
    ]);
    setChatDraft("");
  };

  return (
    <div className="app-root">
      <a className="skip-link" href="#main-content">
        跳到主内容
      </a>
      <div className="app-shell">
        <aside className="sidebar glass-panel" aria-label="主导航">
          <div className="brand-block">
            <div className="brand-mark" aria-hidden="true">
              <Sparkles size={20} />
            </div>
            <div>
              <p className="eyebrow">OfferMaster</p>
              <h1>秋招智能工作台</h1>
            </div>
          </div>

          <nav className="nav-list">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              return (
                <button
                  key={item.id}
                  className={`nav-item ${activePage === item.id ? "is-active" : ""}`}
                  type="button"
                  onClick={() => navigate(item.id)}
                >
                  <Icon size={18} aria-hidden="true" />
                  <span>
                    <strong>{item.label}</strong>
                    <small>{item.description}</small>
                  </span>
                </button>
              );
            })}
          </nav>

          <div className="sidebar-footer">
            <div className="status-dot-row">
              <span className={`status-dot ${notice?.kind === "danger" || notice?.kind === "warning" ? "is-warn" : ""}`} />
              <span>{notice?.kind === "danger" || notice?.kind === "warning" ? "等待后端连接" : "前端入口就绪"}</span>
            </div>
            <p>Nginx 统一入口：127.0.0.1:5173</p>
          </div>
        </aside>

        <main className="main-surface" id="main-content" tabIndex={-1}>
          <header className="topbar glass-panel">
            <div>
              <p className="eyebrow">Agent + Workflow</p>
              <h2>{PAGE_TITLES[activePage]}</h2>
              <p>{PAGE_DESCRIPTIONS[activePage]}</p>
            </div>
            <div className="topbar-actions">
              <button className="button button-ghost" type="button" onClick={handleRefresh} disabled={workingAction === "refresh"}>
                {workingAction === "refresh" ? <Loader2 className="spin" size={16} /> : <RefreshCcw size={16} />}
                刷新
              </button>
              <button className="button button-primary" type="button" onClick={() => navigate("leads")}>
                <Search size={16} />
                查线索
              </button>
            </div>
          </header>

          {notice ? (
            <div className={`notice notice-${notice.kind}`} role="status" aria-live="polite">
              {notice.kind === "danger" ? <AlertTriangle size={17} /> : <CircleDot size={17} />}
              <span>{notice.message}</span>
            </div>
          ) : null}

          {loading ? (
            <LoadingView />
          ) : (
            <div className="page-stack">
              {activePage === "chat" ? (
                <ChatPage
                  draft={chatDraft}
                  messages={chatMessages}
                  onDraftChange={setChatDraft}
                  onPromptSelect={setChatDraft}
                  onSubmit={handleSendChat}
                  navigate={navigate}
                />
              ) : null}
              {activePage === "dashboard" ? <DashboardPage summary={summary} sources={sources} leads={leads} navigate={navigate} /> : null}
              {activePage === "sources" ? (
                <SourcesPage
                  draft={sourceDraft}
                  onDraftChange={setSourceDraft}
                  sources={sources}
                  workingAction={workingAction}
                  onCreate={handleCreateSource}
                  onSync={handleSyncSource}
                />
              ) : null}
              {activePage === "leads" ? (
                <LeadsPage
                  extractDraft={extractDraft}
                  filters={leadFilters}
                  leads={leads}
                  sources={sources}
                  workingAction={workingAction}
                  onExtract={handleExtractLeads}
                  onExtractDraftChange={setExtractDraft}
                  onFiltersChange={setLeadFilters}
                  onSearch={handleSearchLeads}
                  onMarkStatus={handleMarkLeadStatus}
                  onVerifyAndConvert={handleVerifyAndConvert}
                />
              ) : null}
              {activePage === "pipeline" ? <PipelinePage leads={leads} navigate={navigate} /> : null}
              {activePage === "guardrails" ? <GuardrailsPage /> : null}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

function ChatPage({
  draft,
  messages,
  navigate,
  onDraftChange,
  onPromptSelect,
  onSubmit,
}: {
  draft: string;
  messages: ChatMessage[];
  navigate: (page: PageId) => void;
  onDraftChange: (value: string) => void;
  onPromptSelect: (value: string) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
}) {
  return (
    <section className="chat-layout">
      <div className="glass-panel chat-panel">
        <div className="chat-hero">
          <div className="chat-hero-icon" aria-hidden="true">
            <Bot size={22} />
          </div>
          <div>
            <p className="eyebrow">AI Career Copilot</p>
            <h3>AI 求职助手</h3>
            <p>先保留对话体验入口，后续再接入 Chat API、简历知识库和岗位上下文。</p>
          </div>
        </div>

        <div className="chat-message-list" aria-label="AI 对话消息">
          {messages.map((message) => (
            <article className={`chat-message chat-message-${message.role}`} key={message.id}>
              <div className="chat-avatar" aria-hidden="true">
                {message.role === "assistant" ? <Bot size={16} /> : <MessageCircle size={16} />}
              </div>
              <div className="chat-bubble">
                <div className="chat-meta">
                  <strong>{message.role === "assistant" ? "OfferMaster AI" : "你"}</strong>
                  {message.meta ? <span>{message.meta}</span> : null}
                </div>
                <p>{message.content}</p>
              </div>
            </article>
          ))}
        </div>

        <div className="prompt-row" aria-label="快捷问题">
          {CHAT_PROMPTS.map((prompt) => (
            <button className="prompt-chip" key={prompt} type="button" onClick={() => onPromptSelect(prompt)}>
              {prompt}
            </button>
          ))}
        </div>

        <form className="chat-composer" onSubmit={onSubmit}>
          <label htmlFor="chat-input">输入你想问 AI 的问题</label>
          <div className="chat-input-row">
            <textarea
              id="chat-input"
              value={draft}
              onChange={(event) => onDraftChange(event.target.value)}
              placeholder="例如：帮我分析 Java 后端秋招岗位和我的项目匹配度"
              rows={3}
            />
            <button className="button button-primary chat-send" type="submit" disabled={!draft.trim()}>
              <Send size={16} />
              发送
            </button>
          </div>
        </form>
      </div>

      <aside className="glass-panel chat-side-panel">
        <p className="eyebrow">Scope</p>
        <h3>当前只做前端壳子</h3>
        <div className="action-list">
          <div className="action-item">
            <CheckCircle2 size={18} aria-hidden="true" />
            <div>
              <strong>已实现</strong>
              <p>左侧 AI 对话入口、聊天页布局、输入框、快捷问题和本地占位回复。</p>
            </div>
          </div>
          <div className="action-item">
            <Clock3 size={18} aria-hidden="true" />
            <div>
              <strong>后续接入</strong>
              <p>Chat API、Conversation/Message 表、RAG 简历知识库和流式模型回复。</p>
            </div>
          </div>
          <div className="action-item">
            <ShieldCheck size={18} aria-hidden="true" />
            <div>
              <strong>边界不变</strong>
              <p>真实投递、MCP 自动化和高风险操作仍必须等待用户确认。</p>
            </div>
          </div>
        </div>
        <button className="button button-ghost full-width" type="button" onClick={() => navigate("leads")}>
          <FileSearch size={16} />
          查看岗位线索
        </button>
      </aside>
    </section>
  );
}

function DashboardPage({
  summary,
  sources,
  leads,
  navigate,
}: {
  summary: ReturnType<typeof buildSummary>;
  sources: JobSource[];
  leads: JobLead[];
  navigate: (page: PageId) => void;
}) {
  const latestLeads = leads.slice(0, 5);
  const unsyncedSources = sources.filter((source) => !source.last_synced_at).slice(0, 4);

  return (
    <>
      <section className="metric-grid" aria-label="关键指标">
        <StatCard icon={RadioTower} label="信息源" value={summary.totalSources} helper={`${summary.unsyncedSources} 个待首轮同步`} tone="cyan" />
        <StatCard icon={FileSearch} label="岗位线索" value={summary.totalLeads} helper={`${summary.unverifiedLeads} 条未验证`} tone="amber" />
        <StatCard icon={BadgeCheck} label="已验证" value={summary.verifiedLeads} helper="可转正式岗位" tone="green" />
        <StatCard icon={ShieldCheck} label="确认边界" value="ON" helper="投递前必须用户确认" tone="blue" />
      </section>

      <section className="dashboard-grid">
        <Panel className="span-7" title="岗位发现链路" eyebrow="Workflow Boundary" actionLabel="管理线索" onAction={() => navigate("leads")}>
          <div className="workflow-rail" aria-label="岗位线索处理流程">
            {WORKFLOW_STEPS.map((step, index) => {
              const Icon = step.icon;
              return (
                <article className="workflow-step" key={step.title}>
                  <div className="workflow-index">0{index + 1}</div>
                  <Icon size={18} aria-hidden="true" />
                  <div>
                    <strong>{step.title}</strong>
                    <p>{step.body}</p>
                  </div>
                </article>
              );
            })}
          </div>
        </Panel>

        <Panel className="span-5" title="下一步建议" eyebrow="Action Queue" actionLabel="添加来源" onAction={() => navigate("sources")}>
          <div className="action-list">
            {buildActionItems(summary).map((item) => {
              const Icon = item.icon;
              return (
                <div className="action-item" key={item.title}>
                  <Icon size={18} aria-hidden="true" />
                  <div>
                    <strong>{item.title}</strong>
                    <p>{item.body}</p>
                  </div>
                </div>
              );
            })}
          </div>
        </Panel>
      </section>

      <section className="dashboard-grid">
        <Panel className="span-7" title="最新线索" eyebrow="Recent Leads">
          {latestLeads.length ? (
            <div className="lead-strip-list">
              {latestLeads.map((lead) => (
                <LeadStrip key={lead.id} lead={lead} />
              ))}
            </div>
          ) : (
            <EmptyState icon={FileSearch} title="还没有岗位线索" body="先创建信息源，或把小红书/公众号的秋招汇总粘贴到“岗位线索”页进行抽取。" />
          )}
        </Panel>

        <Panel className="span-5" title="待首轮同步来源" eyebrow="Source Health">
          {unsyncedSources.length ? (
            <div className="compact-list">
              {unsyncedSources.map((source) => (
                <div className="compact-row" key={source.id}>
                  <div>
                    <strong>{source.name}</strong>
                    <p>{SOURCE_TYPE_LABELS[source.source_type]}</p>
                  </div>
                  <TrustPill level={source.trust_level} />
                </div>
              ))}
            </div>
          ) : (
            <EmptyState icon={CheckCircle2} title="来源状态清爽" body="当前启用来源均已有同步记录，后续可接入真正的定时调度。" />
          )}
        </Panel>
      </section>
    </>
  );
}

function SourcesPage({
  draft,
  onDraftChange,
  sources,
  workingAction,
  onCreate,
  onSync,
}: {
  draft: SourceDraft;
  onDraftChange: (draft: SourceDraft) => void;
  sources: JobSource[];
  workingAction: string | null;
  onCreate: (event: FormEvent<HTMLFormElement>) => void;
  onSync: (source: JobSource) => void;
}) {
  return (
    <section className="content-grid">
      <Panel className="span-4" title="新增信息源" eyebrow="Source Registry">
        <form className="form-stack" onSubmit={onCreate}>
          <label>
            <span>名称</span>
            <input value={draft.name} onChange={(event) => onDraftChange({ ...draft, name: event.target.value })} placeholder="例如：大连海事就业网" />
          </label>
          <label>
            <span>来源类型</span>
            <select
              value={draft.source_type}
              onChange={(event) => onDraftChange({ ...draft, source_type: event.target.value as JobSourceType })}
            >
              {SOURCE_TYPE_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>抓取方式</span>
            <select
              value={draft.fetch_mode}
              onChange={(event) => onDraftChange({ ...draft, fetch_mode: event.target.value as JobSourceFetchMode })}
            >
              {FETCH_MODE_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>入口 URL</span>
            <input value={draft.entry_url} onChange={(event) => onDraftChange({ ...draft, entry_url: event.target.value })} placeholder="https://..." />
          </label>
          <div className="form-row">
            <label>
              <span>可信度</span>
              <select
                value={draft.trust_level}
                onChange={(event) => onDraftChange({ ...draft, trust_level: event.target.value as JobSourceTrustLevel })}
              >
                {TRUST_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span>同步间隔</span>
              <input
                min={1}
                max={720}
                type="number"
                value={draft.sync_interval_hours}
                onChange={(event) => onDraftChange({ ...draft, sync_interval_hours: Number(event.target.value) })}
              />
            </label>
          </div>
          <label>
            <span>备注</span>
            <textarea value={draft.notes} onChange={(event) => onDraftChange({ ...draft, notes: event.target.value })} rows={3} placeholder="例如：公众号汇总可信度较高，但投递前仍需验证" />
          </label>
          <button className="button button-primary full-width" type="submit" disabled={workingAction === "create-source"}>
            {workingAction === "create-source" ? <Loader2 className="spin" size={16} /> : <Plus size={16} />}
            创建来源
          </button>
        </form>
      </Panel>

      <Panel className="span-8" title="已启用来源" eyebrow="Enabled Sources">
        {sources.length ? (
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>来源</th>
                  <th>类型</th>
                  <th>方式</th>
                  <th>可信度</th>
                  <th>最近同步</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                {sources.map((source) => (
                  <tr key={source.id}>
                    <td>
                      <div className="table-title">
                        <strong>{source.name}</strong>
                        {source.entry_url ? (
                          <a href={source.entry_url} target="_blank" rel="noreferrer">
                            打开来源 <ExternalLink size={13} aria-hidden="true" />
                          </a>
                        ) : (
                          <span>无入口 URL</span>
                        )}
                      </div>
                    </td>
                    <td>{SOURCE_TYPE_LABELS[source.source_type]}</td>
                    <td>{FETCH_MODE_LABELS[source.fetch_mode]}</td>
                    <td>
                      <TrustPill level={source.trust_level} />
                    </td>
                    <td>{formatDateTime(source.last_synced_at)}</td>
                    <td>
                      <button
                        className="button button-small button-ghost"
                        type="button"
                        onClick={() => onSync(source)}
                        disabled={workingAction === `sync-${source.id}`}
                      >
                        {workingAction === `sync-${source.id}` ? <Loader2 className="spin" size={14} /> : <RefreshCcw size={14} />}
                        同步
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState icon={RadioTower} title="还没有启用信息源" body="先把高校就业网、企业官网、公众号或小红书汇总入口录入进来。" />
        )}
      </Panel>
    </section>
  );
}

function LeadsPage({
  extractDraft,
  filters,
  leads,
  sources,
  workingAction,
  onExtract,
  onExtractDraftChange,
  onFiltersChange,
  onSearch,
  onMarkStatus,
  onVerifyAndConvert,
}: {
  extractDraft: ExtractDraft;
  filters: LeadFilterDraft;
  leads: JobLead[];
  sources: JobSource[];
  workingAction: string | null;
  onExtract: (event: FormEvent<HTMLFormElement>) => void;
  onExtractDraftChange: (draft: ExtractDraft) => void;
  onFiltersChange: (filters: LeadFilterDraft) => void;
  onSearch: (event: FormEvent<HTMLFormElement>) => void;
  onMarkStatus: (lead: JobLead, status: JobLeadStatus) => void;
  onVerifyAndConvert: (lead: JobLead) => void;
}) {
  return (
    <section className="content-grid">
      <Panel className="span-4" title="粘贴汇总抽取" eyebrow="LLM Extraction">
        <form className="form-stack" onSubmit={onExtract}>
          <label>
            <span>归属来源</span>
            <select
              value={extractDraft.source_id}
              onChange={(event) => onExtractDraftChange({ ...extractDraft, source_id: event.target.value })}
            >
              <option value="">选择信息源</option>
              {sources.map((source) => (
                <option key={source.id} value={source.id}>
                  {source.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>原文 URL</span>
            <input value={extractDraft.source_url} onChange={(event) => onExtractDraftChange({ ...extractDraft, source_url: event.target.value })} placeholder="可选：小红书/公众号链接" />
          </label>
          <label>
            <span>秋招汇总文本</span>
            <textarea
              className="large-textarea"
              value={extractDraft.raw_content}
              onChange={(event) => onExtractDraftChange({ ...extractDraft, raw_content: event.target.value })}
              placeholder="粘贴开放企业、岗位、网申入口、截止日期等文本"
              rows={10}
            />
          </label>
          <button className="button button-primary full-width" type="submit" disabled={workingAction === "extract-leads"}>
            {workingAction === "extract-leads" ? <Loader2 className="spin" size={16} /> : <Bot size={16} />}
            抽取岗位线索
          </button>
        </form>
      </Panel>

      <Panel className="span-8" title="线索池" eyebrow="Lead Inbox">
        <form className="filter-bar" onSubmit={onSearch}>
          <label>
            <span>关键词</span>
            <input value={filters.keyword} onChange={(event) => onFiltersChange({ ...filters, keyword: event.target.value })} placeholder="Java / Agent / 后端" />
          </label>
          <label>
            <span>状态</span>
            <select
              value={filters.verification_status}
              onChange={(event) => onFiltersChange({ ...filters, verification_status: event.target.value as JobLeadStatus | "" })}
            >
              <option value="">全部</option>
              {STATUS_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>来源</span>
            <select value={filters.source_type} onChange={(event) => onFiltersChange({ ...filters, source_type: event.target.value as JobSourceType | "" })}>
              <option value="">全部</option>
              {SOURCE_TYPE_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>届别</span>
            <input value={filters.graduation_year} onChange={(event) => onFiltersChange({ ...filters, graduation_year: event.target.value })} placeholder="2027" />
          </label>
          <button className="button button-ghost" type="submit" disabled={workingAction === "search-leads"}>
            {workingAction === "search-leads" ? <Loader2 className="spin" size={16} /> : <Search size={16} />}
            筛选
          </button>
        </form>

        {leads.length ? (
          <div className="lead-card-list">
            {leads.map((lead) => (
              <article className="lead-card" key={lead.id}>
                <div className="lead-card-main">
                  <div className="lead-card-title">
                    <div>
                      <p className="eyebrow">{lead.company_name}</p>
                      <h3>{lead.title}</h3>
                    </div>
                    <StatusPill status={lead.verification_status} />
                  </div>
                  <p>{lead.jd_text || "暂无 JD 摘要，投递前需要进入来源页或官网验证岗位详情。"}</p>
                  <div className="tag-row">
                    {lead.city ? <span>{lead.city}</span> : null}
                    {lead.job_direction ? <span>{lead.job_direction}</span> : null}
                    {lead.graduation_year ? <span>{lead.graduation_year} 届</span> : null}
                    {lead.deadline ? <span>截止 {lead.deadline}</span> : null}
                    <span>可信度 {TRUST_LABELS[lead.trust_level]}</span>
                  </div>
                  {lead.skills.length ? (
                    <div className="skill-row">
                      {lead.skills.slice(0, 6).map((skill) => (
                        <span key={`${lead.id}-${skill}`}>{skill}</span>
                      ))}
                    </div>
                  ) : null}
                </div>
                <div className="lead-actions">
                  {lead.source_url || lead.apply_url || lead.verified_url ? (
                    <a className="button button-small button-ghost" href={lead.apply_url || lead.verified_url || lead.source_url || "#"} target="_blank" rel="noreferrer">
                      <ExternalLink size={14} />
                      打开
                    </a>
                  ) : null}
                  <button className="button button-small button-ghost" type="button" onClick={() => onMarkStatus(lead, "verified")} disabled={workingAction === `verified-${lead.id}`}>
                    {workingAction === `verified-${lead.id}` ? <Loader2 className="spin" size={14} /> : <CheckCircle2 size={14} />}
                    标记已验证
                  </button>
                  <button
                    className="button button-small button-primary"
                    type="button"
                    onClick={() => onVerifyAndConvert(lead)}
                    disabled={workingAction === `convert-${lead.id}` || lead.verification_status === "converted"}
                  >
                    {workingAction === `convert-${lead.id}` ? <Loader2 className="spin" size={14} /> : <Send size={14} />}
                    验证并转岗位
                  </button>
                </div>
              </article>
            ))}
          </div>
        ) : (
          <EmptyState icon={FileSearch} title="暂无匹配线索" body="调整筛选条件，或先粘贴一批秋招汇总文本让模型抽取。" />
        )}
      </Panel>
    </section>
  );
}

function PipelinePage({ leads, navigate }: { leads: JobLead[]; navigate: (page: PageId) => void }) {
  const converted = leads.filter((lead) => lead.verification_status === "converted");
  const ready = leads.filter((lead) => lead.verification_status === "verified");

  return (
    <section className="dashboard-grid">
      <Panel className="span-6" title="正式岗位准备区" eyebrow="Application Pipeline">
        <div className="pipeline-board">
          <PipelineColumn title="已验证" count={ready.length} icon={BadgeCheck} />
          <PipelineColumn title="已转正式岗位" count={converted.length} icon={BriefcaseBusiness} />
          <PipelineColumn title="待用户确认" count={0} icon={ShieldCheck} />
        </div>
        <p className="panel-note">投递动作还未开发到当前阶段。这里先只展示“验证线索 → 正式岗位 → 用户确认”的边界，不会自动投递。</p>
      </Panel>
      <Panel className="span-6" title="下一阶段范围" eyebrow="Deferred Scope" actionLabel="回到线索池" onAction={() => navigate("leads")}>
        <div className="action-list">
          <div className="action-item">
            <BriefcaseBusiness size={18} aria-hidden="true" />
            <div>
              <strong>正式岗位列表与详情</strong>
              <p>接入 jobs API 后展示已确认岗位、岗位详情和来源追溯。</p>
            </div>
          </div>
          <div className="action-item">
            <BellDot size={18} aria-hidden="true" />
            <div>
              <strong>投递状态机</strong>
              <p>后续再加入网申中、笔试、面试、Offer 等申请事件。</p>
            </div>
          </div>
        </div>
      </Panel>
    </section>
  );
}

function GuardrailsPage() {
  return (
    <section className="dashboard-grid">
      <Panel className="span-6" title="自动化边界" eyebrow="Guardrails">
        <div className="guardrail-list">
          {GUARDRAILS.map((item) => {
            const Icon = item.icon;
            return (
              <article className="guardrail-card" key={item.title}>
                <Icon size={18} aria-hidden="true" />
                <div>
                  <strong>{item.title}</strong>
                  <p>{item.body}</p>
                </div>
              </article>
            );
          })}
        </div>
      </Panel>
      <Panel className="span-6" title="当前阶段" eyebrow="Implementation State">
        <div className="state-stack">
          <MetricRow label="DDD 边界" value="已保持" />
          <MetricRow label="LangGraph checkpoint" value="后端已落地" />
          <MetricRow label="MCP Gateway" value="浏览器自动化唯一入口" />
          <MetricRow label="真实投递" value="必须用户确认" />
        </div>
      </Panel>
    </section>
  );
}

function Panel({
  actionLabel,
  children,
  className = "",
  eyebrow,
  onAction,
  title,
}: {
  actionLabel?: string;
  children: React.ReactNode;
  className?: string;
  eyebrow: string;
  onAction?: () => void;
  title: string;
}) {
  return (
    <section className={`glass-panel content-panel ${className}`}>
      <div className="panel-heading">
        <div>
          <p className="eyebrow">{eyebrow}</p>
          <h3>{title}</h3>
        </div>
        {actionLabel && onAction ? (
          <button className="button button-small button-ghost" type="button" onClick={onAction}>
            {actionLabel}
            <ChevronRight size={14} />
          </button>
        ) : null}
      </div>
      {children}
    </section>
  );
}

function StatCard({ icon: Icon, label, value, helper, tone }: { icon: LucideIcon; label: string; value: string | number; helper: string; tone: string }) {
  return (
    <article className={`glass-panel stat-card stat-${tone}`}>
      <div className="stat-icon">
        <Icon size={18} aria-hidden="true" />
      </div>
      <div>
        <p>{label}</p>
        <strong>{value}</strong>
        <span>{helper}</span>
      </div>
    </article>
  );
}

function LoadingView() {
  return (
    <section className="loading-grid" aria-label="页面加载中">
      {Array.from({ length: 6 }, (_, index) => (
        <div className="glass-panel skeleton-card" key={index}>
          <span />
          <strong />
          <p />
        </div>
      ))}
    </section>
  );
}

function EmptyState({ icon: Icon, title, body }: { icon: LucideIcon; title: string; body: string }) {
  return (
    <div className="empty-state">
      <Icon size={28} aria-hidden="true" />
      <strong>{title}</strong>
      <p>{body}</p>
    </div>
  );
}

function StatusPill({ status }: { status: JobLeadStatus }) {
  return <span className={`pill status-${status}`}>{STATUS_LABELS[status]}</span>;
}

function TrustPill({ level }: { level: JobSourceTrustLevel }) {
  return <span className={`pill trust-${level}`}>{TRUST_LABELS[level]}</span>;
}

function LeadStrip({ lead }: { lead: JobLead }) {
  return (
    <article className="lead-strip">
      <div>
        <strong>{lead.company_name}</strong>
        <p>{lead.title}</p>
      </div>
      <StatusPill status={lead.verification_status} />
    </article>
  );
}

function PipelineColumn({ count, icon: Icon, title }: { count: number; icon: LucideIcon; title: string }) {
  return (
    <div className="pipeline-column">
      <Icon size={18} aria-hidden="true" />
      <span>{title}</span>
      <strong>{count}</strong>
    </div>
  );
}

function MetricRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function buildLeadFilters(filters: LeadFilterDraft): JobLeadFilters {
  return {
    keyword: filters.keyword.trim() || undefined,
    verification_status: filters.verification_status || undefined,
    source_type: filters.source_type || undefined,
    trust_level: filters.trust_level || undefined,
    graduation_year: filters.graduation_year.trim() || undefined,
    limit: 120,
  };
}

function buildSummary(sources: JobSource[], leads: JobLead[]) {
  const unverifiedLeads = leads.filter((lead) => ["unverified", "pending_review"].includes(lead.verification_status)).length;
  const verifiedLeads = leads.filter((lead) => lead.verification_status === "verified").length;

  return {
    totalSources: sources.length,
    unsyncedSources: sources.filter((source) => !source.last_synced_at).length,
    totalLeads: leads.length,
    unverifiedLeads,
    verifiedLeads,
    convertedLeads: leads.filter((lead) => lead.verification_status === "converted").length,
  };
}

function buildActionItems(summary: ReturnType<typeof buildSummary>) {
  const items = [];

  if (summary.totalSources === 0) {
    items.push({ icon: RadioTower, title: "先录入信息源", body: "把高校就业网、公众号、小红书汇总或企业官网加入来源池。" });
  }
  if (summary.unsyncedSources > 0) {
    items.push({ icon: RefreshCcw, title: "完成首轮同步", body: `${summary.unsyncedSources} 个来源还没有同步记录。` });
  }
  if (summary.unverifiedLeads > 0) {
    items.push({ icon: ShieldCheck, title: "验证线索", body: `${summary.unverifiedLeads} 条线索需要验证入口和开放状态。` });
  }
  if (summary.verifiedLeads > 0) {
    items.push({ icon: BriefcaseBusiness, title: "转正式岗位", body: `${summary.verifiedLeads} 条已验证线索可进入正式岗位池。` });
  }
  if (items.length === 0) {
    items.push({ icon: CheckCircle2, title: "当前队列为空", body: "可以继续添加新来源，扩大秋招信息覆盖。" });
  }

  return items;
}

function getInitialPage(): PageId {
  const hash = window.location.hash.replace("#", "") as PageId;
  return NAV_ITEMS.some((item) => item.id === hash) ? hash : "dashboard";
}

function formatDateTime(value: string | null): string {
  if (!value) {
    return "未同步";
  }

  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function toDisplayError(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return "未知错误";
}

const PAGE_TITLES: Record<PageId, string> = {
  chat: "AI 求职助手",
  dashboard: "秋招发现总览",
  sources: "岗位信息源管理",
  leads: "岗位线索工作台",
  pipeline: "投递管线预备区",
  guardrails: "自动化边界",
};

const PAGE_DESCRIPTIONS: Record<PageId, string> = {
  chat: "先放出对话入口和交互壳子，后续接入真实 Chat API 与简历知识库。",
  dashboard: "先广撒网收集来源，再让线索进入验证与用户确认流程。",
  sources: "管理高校就业网、企业官网、公众号、小红书和可见招聘页来源。",
  leads: "从非结构化汇总抽取岗位线索，并在投递前做懒加载验证。",
  pipeline: "正式投递还未自动化，当前只展示阶段边界与后续入口。",
  guardrails: "保留 DDD、LangGraph checkpoint、MCP Gateway 和用户确认边界。",
};

const SOURCE_TYPE_LABELS: Record<JobSourceType, string> = Object.fromEntries(SOURCE_TYPE_OPTIONS.map((option) => [option.value, option.label])) as Record<
  JobSourceType,
  string
>;

const FETCH_MODE_LABELS: Record<JobSourceFetchMode, string> = Object.fromEntries(FETCH_MODE_OPTIONS.map((option) => [option.value, option.label])) as Record<
  JobSourceFetchMode,
  string
>;

const TRUST_LABELS: Record<JobSourceTrustLevel, string> = Object.fromEntries(TRUST_OPTIONS.map((option) => [option.value, option.label])) as Record<
  JobSourceTrustLevel,
  string
>;

const STATUS_LABELS: Record<JobLeadStatus, string> = Object.fromEntries(STATUS_OPTIONS.map((option) => [option.value, option.label])) as Record<JobLeadStatus, string>;

const WORKFLOW_STEPS: Array<{ icon: LucideIcon; title: string; body: string }> = [
  { icon: Network, title: "来源登记", body: "管理广撒网入口，区分公开 HTML、手动剪贴、官方 API 与 MCP 可见页面。" },
  { icon: DatabaseZap, title: "脚本采集", body: "稳定来源由程序拉取，降低模型调用成本。" },
  { icon: Bot, title: "模型抽取", body: "小红书、公众号、牛客等汇总文本由大模型抽取结构化岗位线索。" },
  { icon: ShieldCheck, title: "验证确认", body: "投递前验证链接与开放状态，真实提交必须用户确认。" },
];

const GUARDRAILS: Array<{ icon: LucideIcon; title: string; body: string }> = [
  { icon: Layers3, title: "DDD 不下沉", body: "页面只调用 API，业务规则继续在 domain service 内闭环。" },
  { icon: Workflow, title: "LangGraph 留痕", body: "采集/抽取/验证工作流需要 checkpoint 与可恢复状态。" },
  { icon: ShieldCheck, title: "MCP Gateway", body: "登录态、验证码、动态页面只走用户可见的 MCP 浏览器边界。" },
  { icon: Activity, title: "高风险确认", body: "投递、撤回、发送消息等动作必须等待用户确认。" },
];

export default App;
