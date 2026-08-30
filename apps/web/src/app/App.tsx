import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BadgeCheck,
  Bot,
  BriefcaseBusiness,
  Building2,
  CheckCircle2,
  ChevronRight,
  CircleDot,
  Clock3,
  Copy,
  Cpu,
  DatabaseZap,
  Download,
  ExternalLink,
  FileSearch,
  Gauge,
  Layers3,
  Link2,
  Loader2,
  LucideIcon,
  MessageCircle,
  Network,
  Pencil,
  Plus,
  RadioTower,
  RefreshCcw,
  Save,
  Search,
  Send,
  ShieldCheck,
  Sparkles,
  Trash2,
  Workflow,
  Wrench,
  X,
} from "lucide-react";
import { FormEvent, KeyboardEvent, ReactNode, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";

import {
  approveAgentApproval,
  createAgentSession,
  deleteAgentSession,
  getAgentMessages,
  getAgentTaskPlan,
  listAgentSessions,
  rejectAgentApproval,
  streamAgentMessage,
  updateAgentSession,
} from "../api/agent";
import { getAgentRuntimePanel } from "../api/agentRuntime";
import { archiveAgentSkill, importAgentSkill, listAgentSkills, pinAgentSkill } from "../api/agentSkills";
import { createApplicationFromJob, listApplications, updateApplication } from "../api/applications";
import { listOfferIOCompanies, listOfferIOCompanyOpenings, listOfferIOJobs } from "../api/jobBoard";
import { createJobSource, disableJobSource, listJobSources, syncJobSource, updateJobSource } from "../api/jobSources";
import { extractJobLeads, listJobLeads, verifyAndConvertJobLead, verifyJobLead } from "../api/jobLeads";
import { listArticleCandidates } from "../api/recruitingSignals";
import { importJobLeadsFromUrl, listDomainHealth, listDomainHealthByDomain, pollUrlImportRun, submitVisiblePageContent } from "../api/urlImport";
import { AsciiArt } from "../components/ui/n-ascii";
import type {
  AgentApprovalRequiredPayload,
  AgentContextMetadata,
  AgentMessage,
  AgentSession,
  AgentStreamOuterSessionEvent,
  AgentStreamToolEvent,
  AgentTaskPlan,
  AgentTaskPlanStage,
} from "../types/agent";
import type { AgentRuntimeCapability, AgentRuntimeHealth, AgentRuntimeMember, AgentRuntimePanel } from "../types/agentRuntime";
import type { AgentSkill, AgentSkillAvailabilityState, AgentSkillImportInput } from "../types/agentSkills";
import type {
  ArticleCandidate,
  ApplicationBoardItem,
  ApplicationStatus,
  DomainHealth,
  ImportUrlAcceptedResponse,
  ImportUrlRun,
  JobImportDraft,
  JobLead,
  JobLeadFilters,
  JobLeadStatus,
  JobSource,
  JobSourceFetchMode,
  JobSourceTrustLevel,
  JobSourceType,
  OfferIOCompany,
  OfferIOCompanyOpening,
  OfferIOJob,
  UrlImportInput,
} from "../types/jobs";

type PageId = "chat" | "agents" | "skills" | "dashboard" | "sources" | "jobs" | "leads" | "pipeline" | "guardrails";
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

interface JobBoardFilterDraft {
  source_mode: "company_jobs" | "company_openings";
  keyword: string;
  industry: string;
  job_type: string;
  batch: string;
  target: string;
  company_nature: string;
  page: number;
  page_size: number;
}

interface ExtractDraft {
  source_id: string;
  source_url: string;
  raw_content: string;
}

interface UrlImportDraft {
  url: string;
  source_id: string;
  source_hint: JobSourceType | "";
  trust_level: JobSourceTrustLevel | "";
  force_refresh: boolean;
}

interface UrlImportProgress {
  accepted: ImportUrlAcceptedResponse | null;
  run: ImportUrlRun | null;
}

interface VisiblePageDraft {
  title: string;
  final_url: string;
  visible_text: string;
}

type UrlImportFlowState = "pending" | "running" | "done" | "blocked" | "failed";

interface UrlImportFlowStep {
  id: string;
  title: string;
  detail: string;
  icon: LucideIcon;
  state: UrlImportFlowState;
}

interface ChatMessage {
  id: string;
  role: "assistant" | "user";
  content: string;
  meta?: string;
}

type ChatRuntimeEventKind = "outer_session" | "tool";
type ChatRuntimeEventTone = "running" | "success" | "warning" | "danger" | "muted";
type ChatRuntimeEventGroup = "reasoning" | "tooling" | "observation" | "evidence" | "status";
type ChatRuntimeActorType = "main_model" | "sub_agent" | "local_tool" | "runtime" | "observation";

interface RuntimeEvidenceItem {
  title: string;
  url?: string;
}

interface RuntimeActorInfo {
  type: ChatRuntimeActorType;
  label: string;
  detail: string;
}

interface RuntimeActorVisual {
  Icon: LucideIcon;
  label: string;
  caption: string;
  className: string;
}

interface ChatRuntimeTimelineEvent {
  id: string;
  kind: ChatRuntimeEventKind;
  eventType: string;
  label: string;
  summary: string;
  status?: string | null;
  toolName?: string | null;
  stepIndex?: number | null;
  inputHint?: string | null;
  candidateNames?: string[];
  inputPreview?: Record<string, unknown> | null;
  resultSummary?: Record<string, unknown> | null;
  reflection?: Record<string, unknown> | null;
  evidence?: RuntimeEvidenceItem[];
  createdAt: number;
  tone: ChatRuntimeEventTone;
}

interface SkillImportDraft {
  source_path: string;
  category: string;
}

const NAV_ITEMS: Array<{ id: PageId; label: string; description: string; icon: LucideIcon }> = [
  { id: "chat", label: "AI 对话", description: "简历/岗位/面试问答", icon: Bot },
  { id: "agents", label: "Agent 面板", description: "成员与能力注册", icon: Network },
  { id: "skills", label: "Skill 管理", description: "内容源能力与依赖", icon: Layers3 },
  { id: "dashboard", label: "总览", description: "同步态势与下一步", icon: Gauge },
  { id: "sources", label: "信息源", description: "高校/企业/社媒入口", icon: RadioTower },
  { id: "jobs", label: "公司展览", description: "公司结构化浏览", icon: Building2 },
  { id: "leads", label: "线索导入", description: "文章/链接/手动兜底", icon: FileSearch },
  { id: "pipeline", label: "投递进度", description: "看板阶段可修改", icon: BriefcaseBusiness },
  { id: "guardrails", label: "边界设置", description: "MCP 与确认边界", icon: ShieldCheck },
];

const SOURCE_TYPE_OPTIONS: Array<{ value: JobSourceType; label: string }> = [
  { value: "university_career_site", label: "高校就业网" },
  { value: "official_career_site", label: "企业招聘官网" },
  { value: "xiaohongshu_note", label: "小红书笔记" },
  { value: "wechat_article", label: "公众号文章" },
  { value: "wechat_account", label: "微信公众号账号" },
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
  source_type: "wechat_account",
  entry_url: "",
  sync_interval_hours: 24,
  trust_level: "high",
  fetch_mode: "mcp_visible_page",
  notes: "",
};

const INITIAL_FILTERS: LeadFilterDraft = {
  keyword: "",
  verification_status: "",
  source_type: "",
  trust_level: "",
  graduation_year: "2027",
};

const INITIAL_JOB_BOARD_FILTERS: JobBoardFilterDraft = {
  source_mode: "company_openings",
  keyword: "",
  industry: "",
  job_type: "校招",
  batch: "秋招",
  target: "2027届",
  company_nature: "",
  page: 1,
  page_size: 50,
};

const INITIAL_URL_IMPORT_DRAFT: UrlImportDraft = {
  url: "",
  source_id: "",
  source_hint: "",
  trust_level: "medium_high",
  force_refresh: false,
};

const INITIAL_VISIBLE_PAGE_DRAFT: VisiblePageDraft = {
  title: "",
  final_url: "",
  visible_text: "",
};

const INITIAL_CHAT_MESSAGES: ChatMessage[] = [
  {
    id: "assistant-welcome",
    role: "assistant",
    content: "AI 会话记忆入口已接入。你可以先记录简历、目标岗位或秋招问题，当前回复来自后端会话 API。",
    meta: "会话 API",
  },
];

const RUNTIME_TOOL_LABELS: Record<string, string> = {
  agent_loop: "主 agent 循环",
  "external.web_search": "网页搜索",
  "local.company_database_overview": "本地企业库概览",
  "local.job_source_overview": "岗位来源概览",
  "offerio.sync_company_jobs": "OfferIO 岗位同步",
  "applications.find_apply_entry": "申请入口发现",
  "resume.tailor": "简历优化 Agent",
  "filesystem.list_dir": "查看目录",
  "filesystem.path_exists": "检查路径是否存在",
  "filesystem.path_stat": "查看文件信息",
  "filesystem.read_file": "读取文件",
  "filesystem.write_text": "写入文件",
  "filesystem.replace_text": "精确替换文本",
  "filesystem.copy_file": "复制文件",
  "filesystem.move_file": "移动/重命名文件",
  "filesystem.delete_path": "删除文件",
  "filesystem.make_dir": "创建目录",
  memory_search: "会话记忆检索",
};

const RUNTIME_EVENT_LABELS: Record<string, string> = {
  candidate_capabilities: "候选能力",
  reasoning_summary: "思考摘要",
  task_started: "任务开始",
  turn_started: "开始思考",
  model_decision: "模型选择能力",
  tool_input_preview: "工具输入",
  tool_started: "工具开始",
  tool_finished: "工具完成",
  tool_result_summary: "结果摘要",
  reflection_evaluation: "反思判断",
  evidence_selected: "证据选择",
  turn_finished: "观察结果",
  tool_reflection_retry: "结果不足，准备重试",
  textual_tool_call_recovered: "自动纠偏执行",
  textual_tool_call_blocked: "疑似工具调用",
  observation_insufficient: "观察不足",
  waiting_user: "等待用户确认或补充",
  task_finished: "任务结束",
};

const RUNTIME_EVENT_SUMMARIES: Record<string, (toolLabel: string) => string> = {
  candidate_capabilities: () => "主 agent 已把候选能力交给模型选择。",
  reasoning_summary: () => "主 agent 已生成本轮安全思考摘要。",
  task_started: () => "主 agent 开始处理本轮任务。",
  turn_started: () => "主 agent 正在判断下一步该直接回答还是调用能力。",
  model_decision: (toolLabel) => `模型判断下一步要使用：${toolLabel}。`,
  tool_input_preview: (toolLabel) => `准备传给 ${toolLabel} 的输入已整理。`,
  tool_started: (toolLabel) => `开始调用：${toolLabel}。`,
  tool_finished: (toolLabel) => `已观察到 ${toolLabel} 的执行结果。`,
  tool_result_summary: (toolLabel) => `${toolLabel} 的结果已整理成可读摘要。`,
  reflection_evaluation: () => "主 agent 已判断工具结果是否足够回答。",
  evidence_selected: () => "主 agent 已选择本轮回答要引用的证据。",
  turn_finished: (toolLabel) => `主 agent 已读取 ${toolLabel} 的观察结果。`,
  tool_reflection_retry: () => "当前结果不够好，主 agent 准备调整输入后重试。",
  textual_tool_call_recovered: () => "模型把工具调用写成了普通文字，运行时已转换为真实工具流程。",
  textual_tool_call_blocked: () => "模型把工具调用写成了普通文字，运行时已拦截，没有当作真实工具执行。",
  observation_insufficient: () => "当前观察结果还不足以完成任务，需要继续补充信息。",
  waiting_user: () => "当前步骤需要用户确认或补充信息后才能继续。",
  task_finished: () => "本轮 agent loop 已结束。",
};

const INITIAL_SKILL_IMPORT_DRAFT: SkillImportDraft = {
  source_path: "",
  category: "content_source",
};

const CHAT_PROMPTS = ["帮我分析这个岗位是否适合我", "帮我优化简历项目描述", "模拟 Java 后端面试", "根据岗位生成投递建议"];

interface ParsedMarkdownTable {
  headers: string[];
  rows: string[][];
}

type ChatContentBlock = { type: "text"; text: string } | { type: "table"; table: ParsedMarkdownTable };

function ChatMessageContent({ content }: { content: string }) {
  const blocks = useMemo(() => parseMarkdownTables(content), [content]);

  return (
    <div className="chat-message-content">
      {blocks.map((block, index) =>
        block.type === "table" ? (
          <ChatTableCard key={`table-${index}`} table={block.table} />
        ) : block.text.trim() ? (
          <p className="chat-message-text" key={`text-${index}`}>
            {renderInlineLinks(block.text.trim())}
          </p>
        ) : null,
      )}
    </div>
  );
}

function renderInlineLinks(text: string): ReactNode[] {
  const markdownLinkPattern = /\[([^\]\n]+)]\((https?:\/\/[^\s)]+)\)/gi;
  const nodes: ReactNode[] = [];
  let cursor = 0;
  let match: RegExpExecArray | null;

  while ((match = markdownLinkPattern.exec(text))) {
    const [raw, label, url] = match;
    const matchStart = match.index;

    if (matchStart > cursor) {
      nodes.push(...renderBareLinks(text.slice(cursor, matchStart), `text-${cursor}`));
    }

    if (isSafeHttpUrl(url)) {
      nodes.push(
        <a className="chat-inline-link" href={url} key={`markdown-link-${matchStart}`} target="_blank" rel="noreferrer">
          {label}
        </a>,
      );
    } else {
      nodes.push(raw);
    }

    cursor = matchStart + raw.length;
  }

  if (cursor < text.length) {
    nodes.push(...renderBareLinks(text.slice(cursor), `text-${cursor}`));
  }

  return nodes.length ? nodes : [text];
}

function renderBareLinks(text: string, keyPrefix: string): ReactNode[] {
  const bareUrlPattern = /https?:\/\/[^\s<>"'`]+/gi;
  const nodes: ReactNode[] = [];
  let cursor = 0;
  let match: RegExpExecArray | null;

  while ((match = bareUrlPattern.exec(text))) {
    const rawUrl = match[0];
    const matchStart = match.index;
    const { url, trailingText } = splitTrailingUrlText(rawUrl);

    if (!url || !isSafeHttpUrl(url)) {
      continue;
    }

    if (matchStart > cursor) {
      nodes.push(text.slice(cursor, matchStart));
    }

    nodes.push(
      <a className="chat-inline-link" href={url} key={`${keyPrefix}-link-${matchStart}`} target="_blank" rel="noreferrer">
        {url}
      </a>,
    );

    if (trailingText) {
      nodes.push(trailingText);
    }

    cursor = matchStart + rawUrl.length;
  }

  if (cursor < text.length) {
    nodes.push(text.slice(cursor));
  }

  return nodes.length ? nodes : [text];
}

function splitTrailingUrlText(rawUrl: string): { url: string; trailingText: string } {
  const trailingMatch = rawUrl.match(/[.,!?;:，。！？；：、)）\]]+$/);
  if (!trailingMatch?.index) {
    return { url: rawUrl, trailingText: "" };
  }
  return {
    url: rawUrl.slice(0, trailingMatch.index),
    trailingText: rawUrl.slice(trailingMatch.index),
  };
}

function isSafeHttpUrl(value: string): boolean {
  try {
    const url = new URL(value);
    return url.protocol === "http:" || url.protocol === "https:";
  } catch {
    return false;
  }
}

function ChatTableCard({ table }: { table: ParsedMarkdownTable }) {
  const tableText = markdownTableToText(table);

  const handleCopy = () => {
    void navigator.clipboard?.writeText(tableText);
  };

  const handleDownload = () => {
    const blob = new Blob([tableText], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "offermaster-table.md";
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <section className="chat-table-card" aria-label="表格">
      <div className="chat-table-card-header">
        <strong>表格</strong>
        <div className="chat-table-actions">
          <button type="button" aria-label="下载表格" title="下载表格" onClick={handleDownload}>
            <Download size={17} />
          </button>
          <button type="button" aria-label="复制表格" title="复制表格" onClick={handleCopy}>
            <Copy size={17} />
          </button>
        </div>
      </div>
      <div className="chat-table-scroll">
        <table>
          <thead>
            <tr>
              {table.headers.map((header) => (
                <th key={header}>{header}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.rows.map((row, rowIndex) => (
              <tr key={`row-${rowIndex}`}>
                {table.headers.map((_, cellIndex) => (
                  <td key={`cell-${cellIndex}`}>{row[cellIndex] ?? ""}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function App() {
  const [activePage, setActivePage] = useState<PageId>(() => getInitialPage());
  const [sources, setSources] = useState<JobSource[]>([]);
  const [leads, setLeads] = useState<JobLead[]>([]);
  const [articleCandidates, setArticleCandidates] = useState<ArticleCandidate[]>([]);
  const [notice, setNotice] = useState<Notice | null>(null);
  const [loading, setLoading] = useState(true);
  const [workingAction, setWorkingAction] = useState<string | null>(null);
  const [sourceDraft, setSourceDraft] = useState<SourceDraft>(INITIAL_SOURCE_DRAFT);
  const [editingSourceId, setEditingSourceId] = useState<string | null>(null);
  const [leadFilters, setLeadFilters] = useState<LeadFilterDraft>(INITIAL_FILTERS);
  const [urlImportDraft, setUrlImportDraft] = useState<UrlImportDraft>(INITIAL_URL_IMPORT_DRAFT);
  const [visiblePageDraft, setVisiblePageDraft] = useState<VisiblePageDraft>(INITIAL_VISIBLE_PAGE_DRAFT);
  const [urlImportProgress, setUrlImportProgress] = useState<UrlImportProgress>({ accepted: null, run: null });
  const [domainHealth, setDomainHealth] = useState<DomainHealth[]>([]);
  const [extractDraft, setExtractDraft] = useState<ExtractDraft>({
    source_id: "",
    source_url: "",
    raw_content: "",
  });
  const [chatDraft, setChatDraft] = useState("");
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>(INITIAL_CHAT_MESSAGES);
  const [agentSessions, setAgentSessions] = useState<AgentSession[]>([]);
  const [agentSession, setAgentSession] = useState<AgentSession | null>(null);
  const [agentRuntimePanel, setAgentRuntimePanel] = useState<AgentRuntimePanel | null>(null);
  const [agentSkills, setAgentSkills] = useState<AgentSkill[]>([]);
  const [skillImportDraft, setSkillImportDraft] = useState<SkillImportDraft>(INITIAL_SKILL_IMPORT_DRAFT);
  const [chatContextMetadata, setChatContextMetadata] = useState<AgentContextMetadata | null>(null);
  const [taskPlan, setTaskPlan] = useState<AgentTaskPlan | null>(null);
  const [pendingApproval, setPendingApproval] = useState<AgentApprovalRequiredPayload | null>(null);
  const [runtimeEvents, setRuntimeEvents] = useState<ChatRuntimeTimelineEvent[]>([]);
  const [chatLoading, setChatLoading] = useState(false);
  const [jobBoardFilters, setJobBoardFilters] = useState<JobBoardFilterDraft>(INITIAL_JOB_BOARD_FILTERS);
  const [offerioCompanies, setOfferioCompanies] = useState<OfferIOCompany[]>([]);
  const [offerioCompanyTotal, setOfferioCompanyTotal] = useState(0);
  const [offerioCompanyPage, setOfferioCompanyPage] = useState(1);
  const [offerioCompanyTotalPages, setOfferioCompanyTotalPages] = useState(1);
  const [offerioOpenings, setOfferioOpenings] = useState<OfferIOCompanyOpening[]>([]);
  const [offerioOpeningTotal, setOfferioOpeningTotal] = useState(0);
  const [offerioOpeningPage, setOfferioOpeningPage] = useState(1);
  const [offerioOpeningTotalPages, setOfferioOpeningTotalPages] = useState(1);
  const [offerioJobs, setOfferioJobs] = useState<OfferIOJob[]>([]);
  const [selectedOfferioCompany, setSelectedOfferioCompany] = useState<string | null>(null);
  const [applications, setApplications] = useState<ApplicationBoardItem[]>([]);

  const refreshData = useCallback(async (filters?: JobLeadFilters) => {
    const [nextSources, nextLeads, nextDomainHealth, nextCandidates] = await Promise.all([
      listJobSources(),
      listJobLeads(filters ?? { graduation_year: INITIAL_FILTERS.graduation_year, limit: 80 }),
      listDomainHealth(),
      listArticleCandidates({ limit: 80 }),
    ]);
    setSources(nextSources);
    setLeads(nextLeads);
    setDomainHealth(nextDomainHealth);
    setArticleCandidates(nextCandidates);
    setExtractDraft((current) => ({
      ...current,
      source_id: current.source_id || nextSources[0]?.id || "",
    }));
    setUrlImportDraft((current) => ({
      ...current,
      source_id: current.source_id || nextSources[0]?.id || "",
    }));
  }, []);

  const refreshActiveTaskPlan = useCallback(async (metadata: AgentContextMetadata | null): Promise<AgentTaskPlan | null> => {
    const taskId = extractActiveTaskId(metadata);
    if (!taskId) {
      setTaskPlan(null);
      return null;
    }
    try {
      const plan = await getAgentTaskPlan(taskId);
      setTaskPlan(plan);
      return plan;
    } catch {
      setTaskPlan(null);
      return null;
    }
  }, []);

  const loadAgentSession = useCallback(async (targetSessionId?: string) => {
    setChatLoading(true);

    try {
      const existingSessions = await listAgentSessions(50, 0);
      const session =
        existingSessions.find((item) => item.id === targetSessionId) ??
        existingSessions[0] ??
        (await createAgentSession({
          title: "AI 求职助手",
          primary_intent: "agent_chat",
        }));
      const nextSessions = existingSessions.some((item) => item.id === session.id) ? existingSessions : [session, ...existingSessions];
      const messages = await getAgentMessages(session.id, 100);
      const latestAssistantContext = [...messages]
        .reverse()
        .map(extractContextMetadata)
        .find((metadata): metadata is AgentContextMetadata => metadata !== null) ?? null;

      setAgentSessions(nextSessions);
      setAgentSession(session);
      setChatMessages(toChatMessages(messages));
      setChatContextMetadata(latestAssistantContext);
      setPendingApproval(null);
      setRuntimeEvents([]);
      await refreshActiveTaskPlan(latestAssistantContext);
    } finally {
      setChatLoading(false);
    }
  }, [refreshActiveTaskPlan]);

  const refreshSkills = useCallback(async () => {
    setAgentSkills(await listAgentSkills("active", 120));
  }, []);

  const refreshAgentRuntimePanel = useCallback(async () => {
    setAgentRuntimePanel(await getAgentRuntimePanel());
  }, []);

  const refreshApplications = useCallback(async () => {
    setApplications(await listApplications(120));
  }, []);

  const refreshJobBoard = useCallback(async (filters: JobBoardFilterDraft = INITIAL_JOB_BOARD_FILTERS) => {
    if (filters.source_mode === "company_openings") {
      const result = await listOfferIOCompanyOpenings({
        keyword: filters.keyword.trim() || undefined,
        industry: filters.industry.trim() || undefined,
        batch: filters.batch.trim() || undefined,
        target: filters.target.trim() || undefined,
        company_nature: filters.company_nature.trim() || undefined,
        page: filters.page,
        page_size: filters.page_size,
      });
      setOfferioOpenings(result.items);
      setOfferioOpeningTotal(result.total);
      setOfferioOpeningPage(result.page);
      setOfferioOpeningTotalPages(result.total_pages || 1);
      return;
    }

    const result = await listOfferIOCompanies({
      job_type: filters.job_type,
      keyword: filters.keyword.trim() || undefined,
      industry: filters.industry.trim() || undefined,
      page: filters.page,
      page_size: filters.page_size,
    });
    setOfferioCompanies(result.items);
    setOfferioCompanyTotal(result.total);
    setOfferioCompanyPage(result.page);
    setOfferioCompanyTotalPages(result.total_pages || 1);
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

  useEffect(() => {
    loadAgentSession().catch((error: unknown) => {
      setNotice({ kind: "warning", message: `AI 会话暂未连接：${toDisplayError(error)}` });
    });
  }, [loadAgentSession]);

  useEffect(() => {
    refreshSkills().catch((error: unknown) => {
      setNotice({ kind: "warning", message: `Skill 列表暂未连接：${toDisplayError(error)}` });
    });
  }, [refreshSkills]);

  useEffect(() => {
    refreshAgentRuntimePanel().catch((error: unknown) => {
      setNotice({ kind: "warning", message: `Agent 面板暂未连接：${toDisplayError(error)}` });
    });
  }, [refreshAgentRuntimePanel]);

  useEffect(() => {
    refreshApplications().catch((error: unknown) => {
      setNotice({ kind: "warning", message: `投递进度暂未连接：${toDisplayError(error)}` });
    });
  }, [refreshApplications]);

  useEffect(() => {
    refreshJobBoard().catch((error: unknown) => {
      setNotice({ kind: "warning", message: `公司展览暂未连接：${toDisplayError(error)}` });
    });
  }, [refreshJobBoard]);

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
      await refreshAgentRuntimePanel();
      await refreshSkills();
      await refreshApplications();
      await refreshJobBoard(jobBoardFilters);
      return "已刷新岗位信息源、公司展览与投递进度。";
    });

  const handleImportSkill = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    void runAction("agent-skill-import", async () => {
      const sourcePath = skillImportDraft.source_path.trim();
      if (!sourcePath) {
        throw new Error("请填写本地 SKILL.md 或目录路径。");
      }

      const payload: AgentSkillImportInput = {
        source_path: sourcePath,
        category: skillImportDraft.category.trim() || "content_source",
      };
      await importAgentSkill(payload);
      setSkillImportDraft(INITIAL_SKILL_IMPORT_DRAFT);
      await refreshSkills();
      return "Skill 已导入，已加入 Agent 可用能力目录。";
    });
  };

  const handlePinSkill = (skill: AgentSkill) => {
    void runAction(`agent-skill-pin-${skill.id}`, async () => {
      await pinAgentSkill(skill.id);
      await refreshSkills();
      return `${skill.title} 已置顶。`;
    });
  };

  const handleArchiveSkill = (skill: AgentSkill) => {
    if (!window.confirm(`确定归档 Skill“${skill.title}”吗？归档后不会进入默认 Skill 目录。`)) {
      return;
    }

    void runAction(`agent-skill-archive-${skill.id}`, async () => {
      await archiveAgentSkill(skill.id);
      await refreshSkills();
      return `${skill.title} 已归档。`;
    });
  };

  const handleSubmitSource = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    const actionName = editingSourceId ? `update-source-${editingSourceId}` : "create-source";
    void runAction(actionName, async () => {
      if (!sourceDraft.name.trim()) {
        throw new Error("请先填写信息源名称。");
      }

      const payload = {
        name: sourceDraft.name.trim(),
        source_type: sourceDraft.source_type,
        entry_url: sourceDraft.entry_url.trim() || null,
        enabled: true,
        sync_interval_hours: sourceDraft.sync_interval_hours,
        trust_level: sourceDraft.trust_level,
        fetch_mode: sourceDraft.fetch_mode,
        notes: sourceDraft.notes.trim() || null,
      };

      if (editingSourceId) {
        await updateJobSource(editingSourceId, payload);
        setEditingSourceId(null);
        setSourceDraft(INITIAL_SOURCE_DRAFT);
        await refreshData(buildLeadFilters(leadFilters));
        return "信息源已更新。";
      }

      await createJobSource(payload);
      setSourceDraft(INITIAL_SOURCE_DRAFT);
      await refreshData(buildLeadFilters(leadFilters));
      return "信息源已创建，后续可加入定时同步。";
    });
  };

  const handleEditSource = (source: JobSource) => {
    setEditingSourceId(source.id);
    setSourceDraft(sourceToDraft(source));
    setNotice({ kind: "info", message: `正在编辑：${source.name}` });
  };

  const handleCancelSourceEdit = () => {
    setEditingSourceId(null);
    setSourceDraft(INITIAL_SOURCE_DRAFT);
    setNotice(null);
  };

  const handleDisableSource = (source: JobSource) => {
    if (!window.confirm(`确认禁用信息源“${source.name}”？历史线索会保留，但它不会再出现在启用来源里。`)) {
      return;
    }

    void runAction(`disable-source-${source.id}`, async () => {
      await disableJobSource(source.id);
      if (editingSourceId === source.id) {
        setEditingSourceId(null);
        setSourceDraft(INITIAL_SOURCE_DRAFT);
      }
      await refreshData(buildLeadFilters(leadFilters));
      return "信息源已禁用，历史线索仍可追溯。";
    });
  };

  const handleSyncSource = (source: JobSource) => {
    void runAction(`sync-${source.id}`, async () => {
      const result = await syncJobSource(source.id, syncLimitForSource(source));
      await refreshData(buildLeadFilters(leadFilters));
      if (result.status === "failed") {
        throw new Error(`${source.name} 同步失败：${result.error || "来源页面无法抓取，请检查来源类型、入口 URL 或改用岗位线索页的粘贴链接解析。"}`);
      }
      if (source.source_type === "wechat_account") {
        return `${source.name} 同步完成：发现 ${result.article_candidates.length} 篇候选招聘文章，记录 ${result.recruiting_signals.length} 个公司校招来源。`;
      }
      return `${source.name} 同步完成：抓取 ${result.fetched_count} 条，抽取 ${result.extracted_count} 条。`;
    });
  };

  const handleSearchJobBoard = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    void runAction("offerio-company-search", async () => {
      const nextFilters = { ...jobBoardFilters, page: 1 };
      setJobBoardFilters(nextFilters);
      await refreshJobBoard(nextFilters);
      setOfferioJobs([]);
      setSelectedOfferioCompany(null);
      return "公司展览已从 OfferIO 临时接口刷新。";
    });
  };

  const handleJobBoardPageChange = (page: number) => {
    const nextFilters = { ...jobBoardFilters, page: Math.max(1, page) };
    setJobBoardFilters(nextFilters);
    void runAction("offerio-company-search", async () => {
      await refreshJobBoard(nextFilters);
      setOfferioJobs([]);
      setSelectedOfferioCompany(null);
      return "公司展览分页已刷新。";
    });
  };

  const handleSelectOfferIOCompany = (company: OfferIOCompany) => {
    void runAction(`offerio-company-jobs-${company.name}`, async () => {
      const result = await listOfferIOJobs({
        company: company.name,
        job_type: jobBoardFilters.job_type,
        page: 1,
        page_size: 80,
      });
      setSelectedOfferioCompany(company.name);
      setOfferioJobs(result.items);
      return `已加载 ${company.name} 的 ${result.items.length} 个岗位。`;
    });
  };

  const handleAddOfferIOJobToPipeline = (job: OfferIOJob) => {
    void runAction(`offerio-job-application-${job.id}`, async () => {
      await createApplicationFromJob({
        job: offerIOJobToImportDraft(job),
        status: "evaluating",
        priority: "medium",
        channel: "offerio",
        notes: "来自 OfferIO 临时接口，投递前仍需官网验证。",
      });
      await refreshApplications();
      return `${job.company} - ${job.title} 已加入投递板。`;
    });
  };

  const handleAddOfferIOOpeningToPipeline = (opening: OfferIOCompanyOpening) => {
    void runAction(`offerio-opening-application-${opening.id}`, async () => {
      await createApplicationFromJob({
        job: offerIOOpeningToImportDraft(opening),
        status: "evaluating",
        priority: "medium",
        channel: "offerio-openings",
        notes: "来自 OfferIO 开放岗位公司库，投递前仍需官网验证。",
      });
      await refreshApplications();
      return `${opening.company_name} 已加入投递板。`;
    });
  };

  const handleUpdateApplicationStatus = (application: ApplicationBoardItem, status: ApplicationStatus) => {
    void runAction(`application-status-${application.id}`, async () => {
      await updateApplication(application.id, {
        status,
        actor: "user",
        source: "manual_board",
      });
      await refreshApplications();
      return `${application.job.company.name} - ${application.job.title} 已更新为${APPLICATION_STATUS_LABELS[status]}。`;
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

  const handleImportUrl = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    void runAction("import-url", async () => {
      const url = urlImportDraft.url.trim();
      if (!url) {
        throw new Error("请先粘贴公众号、企业官网、高校就业网或公开招聘文章链接。");
      }

      setUrlImportProgress({ accepted: null, run: null });
      setVisiblePageDraft(INITIAL_VISIBLE_PAGE_DRAFT);
      const payload: UrlImportInput = {
        url,
        source_id: urlImportDraft.source_id || null,
        source_hint: urlImportDraft.source_hint || null,
        trust_level: urlImportDraft.trust_level || null,
        force_refresh: urlImportDraft.force_refresh,
      };
      const accepted = await importJobLeadsFromUrl(payload);
      setUrlImportProgress({ accepted, run: null });

      const run = await waitForUrlImportRun(accepted.run_id, (nextRun) => {
        setUrlImportProgress({ accepted, run: nextRun });
      });

      if (run.domain) {
        setDomainHealth(await listDomainHealthByDomain(run.domain));
      } else {
        setDomainHealth(await listDomainHealth());
      }

      if (["succeeded", "duplicate"].includes(run.status)) {
        await refreshData(buildLeadFilters(leadFilters));
      }

      if (run.status === "succeeded") {
        setUrlImportDraft((current) => ({ ...current, url: "", force_refresh: false }));
        return `链接解析完成，已抽取 ${run.extracted_count} 条岗位线索。`;
      }
      if (run.status === "waiting_user") {
        return "该链接需要用户可见页面确认，当前不会后台硬爬。";
      }
      if (run.status === "duplicate") {
        return "该链接已有导入记录，已跳过重复抓取。";
      }
      if (run.status === "running") {
        return "URL 导入任务已创建，后台仍在解析，可继续刷新状态。";
      }

      throw new Error(run.error_message || `URL 导入未完成：${URL_IMPORT_STATUS_LABELS[run.status]}`);
    });
  };

  const handleSubmitVisiblePageContent = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    void runAction("submit-visible-page", async () => {
      const runId = urlImportProgress.run?.id ?? urlImportProgress.accepted?.run_id;
      if (!runId) {
        throw new Error("请先创建一个 URL 解析任务。");
      }
      if (!visiblePageDraft.visible_text.trim()) {
        throw new Error("请粘贴小红书可见页面正文后再继续解析。");
      }

      const run = await submitVisiblePageContent(runId, {
        title: visiblePageDraft.title.trim() || null,
        final_url: visiblePageDraft.final_url.trim() || null,
        visible_text: visiblePageDraft.visible_text.trim(),
      });
      setUrlImportProgress((current) => ({ ...current, run }));
      setVisiblePageDraft(INITIAL_VISIBLE_PAGE_DRAFT);
      await refreshData(buildLeadFilters(leadFilters));
      return run.next_action === "enrich_recruiting_signal"
        ? "可见页面正文已保存，已记录为公司校招来源。"
        : `可见页面正文已提交，当前状态：${URL_IMPORT_STATUS_LABELS[run.status] ?? run.status}。`;
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

  const handleCreateChatSession = () => {
    void runAction("agent-chat-session-create", async () => {
      const session = await createAgentSession({
        title: "新对话",
        primary_intent: "agent_chat",
      });
      setAgentSessions((current) => [session, ...current]);
      setAgentSession(session);
      setChatMessages(INITIAL_CHAT_MESSAGES);
      setChatContextMetadata(null);
      setPendingApproval(null);
      setRuntimeEvents([]);
      setChatDraft("");
      return "已创建新对话。";
    });
  };

  const handleSelectChatSession = (sessionId: string) => {
    if (sessionId === agentSession?.id || chatLoading) {
      return;
    }
    loadAgentSession(sessionId).catch((error: unknown) => {
      setNotice({ kind: "warning", message: `AI 会话加载失败：${toDisplayError(error)}` });
    });
  };

  const handleRenameChatSession = (sessionToRename: AgentSession) => {
    const title = window.prompt("请输入新的会话名称", sessionToRename.title ?? "未命名对话");
    if (title === null) {
      return;
    }
    const trimmedTitle = title.trim();
    if (!trimmedTitle) {
      setNotice({ kind: "warning", message: "会话名称不能为空。" });
      return;
    }

    void runAction(`agent-chat-session-rename-${sessionToRename.id}`, async () => {
      const updated = await updateAgentSession(sessionToRename.id, { title: trimmedTitle });
      setAgentSessions((current) => current.map((item) => (item.id === updated.id ? updated : item)));
      setAgentSession((current) => (current?.id === updated.id ? updated : current));
      return "会话已重命名。";
    });
  };

  const handleDeleteChatSession = (sessionToDelete: AgentSession) => {
    if (!window.confirm(`确定删除会话“${sessionToDelete.title ?? "未命名对话"}”吗？`)) {
      return;
    }

    void runAction(`agent-chat-session-delete-${sessionToDelete.id}`, async () => {
      await deleteAgentSession(sessionToDelete.id);
      const remainingSessions = await listAgentSessions(50, 0);
      setAgentSessions(remainingSessions);
      if (agentSession?.id === sessionToDelete.id) {
        await loadAgentSession(remainingSessions[0]?.id);
      }
      return "会话已删除。";
    });
  };

  const handleApprovePendingApproval = () => {
    const approval = pendingApproval;
    const session = agentSession;
    if (!approval || !session) {
      return;
    }

    void runAction(`agent-approval-approve-${approval.approval_request_id}`, async () => {
      const result = await approveAgentApproval(approval.approval_request_id, { decision_reason: "approved from chat UI" });
      const messages = await getAgentMessages(session.id, 100);
      const nextMessages = appendAgentMessageIfMissing(messages, result.assistant_message);
      const lastMessage = result.assistant_message ?? nextMessages[nextMessages.length - 1] ?? null;
      const nextContextMetadata = result.context_metadata ?? (lastMessage ? extractContextMetadata(lastMessage) : null);
      setPendingApproval(null);
      setChatContextMetadata(nextContextMetadata);
      setChatMessages(toChatMessages(nextMessages));
      await refreshActiveTaskPlan(nextContextMetadata);
      setAgentSession({
        ...session,
        message_count: Math.max(session.message_count, nextMessages.length),
        last_message_at: lastMessage?.created_at ?? session.last_message_at,
        updated_at: lastMessage?.created_at ?? session.updated_at,
      });
      setAgentSessions((current) =>
        current.map((item) =>
          item.id === session.id
            ? {
                ...item,
                message_count: Math.max(item.message_count, nextMessages.length),
                last_message_at: lastMessage?.created_at ?? item.last_message_at,
                updated_at: lastMessage?.created_at ?? item.updated_at,
              }
            : item,
        ),
      );
      return `已确认工具调用：${approval.tool_name}`;
    });
  };

  const handleRejectPendingApproval = () => {
    const approval = pendingApproval;
    const session = agentSession;
    if (!approval || !session) {
      return;
    }

    void runAction(`agent-approval-reject-${approval.approval_request_id}`, async () => {
      const result = await rejectAgentApproval(approval.approval_request_id, { decision_reason: "rejected from chat UI" });
      const messages = await getAgentMessages(session.id, 100);
      const nextMessages = appendAgentMessageIfMissing(messages, result.assistant_message);
      const lastMessage = result.assistant_message ?? nextMessages[nextMessages.length - 1] ?? null;
      const nextContextMetadata = result.context_metadata ?? (lastMessage ? extractContextMetadata(lastMessage) : null);
      setPendingApproval(null);
      setChatContextMetadata(nextContextMetadata);
      setChatMessages(toChatMessages(nextMessages));
      await refreshActiveTaskPlan(nextContextMetadata);
      setAgentSession({
        ...session,
        message_count: Math.max(session.message_count, nextMessages.length),
        last_message_at: lastMessage?.created_at ?? session.last_message_at,
        updated_at: lastMessage?.created_at ?? session.updated_at,
      });
      return `已拒绝工具调用：${approval.tool_name}`;
    });
  };

  const handleSendChat = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const content = chatDraft.trim();

    if (!content) {
      return;
    }

    void runAction("agent-chat-send", async () => {
      const session =
        agentSession ??
        (await createAgentSession({
          title: "AI Career Copilot",
          primary_intent: "agent_chat",
        }));
      const tempKey = Date.now();
      const tempUserId = `stream-user-${tempKey}`;
      const tempAssistantId = `stream-assistant-${tempKey}`;

      setAgentSession(session);
      setAgentSessions((current) => (current.some((item) => item.id === session.id) ? current : [session, ...current]));
      setPendingApproval(null);
      setChatDraft("");
      setRuntimeEvents([]);
      setTaskPlan(null);
      setChatMessages((current) => [
        ...withoutWelcomeMessage(current),
        { id: tempUserId, role: "user", content, meta: "你" },
        { id: tempAssistantId, role: "assistant", content: "", meta: "流式输出中" },
      ]);

      let finalAssistantMessage: AgentMessage | null = null;
      const approvalRequiredRef: { current: AgentApprovalRequiredPayload | null } = { current: null };
      await streamAgentMessage(session.id, { content_text: content }, {
        onUserMessage: (message) => {
          setChatMessages((current) => current.map((item) => (item.id === tempUserId ? toChatMessage(message) : item)));
        },
        onToken: (chunk) => {
          setChatMessages((current) =>
            current.map((item) =>
              item.id === tempAssistantId ? { ...item, content: `${item.content}${chunk}`, meta: "流式输出中" } : item,
            ),
          );
        },
        onDone: (message) => {
          finalAssistantMessage = message;
          setChatContextMetadata(extractContextMetadata(message));
          setChatMessages((current) => current.map((item) => (item.id === tempAssistantId ? toChatMessage(message) : item)));
        },
        onApprovalRequired: (payload) => {
          approvalRequiredRef.current = payload;
          setPendingApproval(payload);
          setChatContextMetadata(payload.context_metadata);
          setChatMessages((current) =>
            current.map((item) =>
              item.id === tempAssistantId
                ? {
                    ...item,
                    content: buildApprovalChatMessage(payload),
                    meta: "Tool approval",
                  }
                : item,
            ),
          );
        },
        onOuterSessionEvent: (payload) => {
          setRuntimeEvents((current) => appendRuntimeEvent(current, toRuntimeEventFromOuterSession(payload, current.length)));
        },
        onToolEvent: (payload) => {
          setRuntimeEvents((current) => appendRuntimeEvent(current, toRuntimeEventFromTool(payload, current.length)));
        },
      });

      const messages = await getAgentMessages(session.id, 100);
      const lastMessage = finalAssistantMessage ?? messages[messages.length - 1] ?? null;
      const approvalRequiredPayload = approvalRequiredRef.current;
      setAgentSession({
        ...session,
        message_count: Math.max(session.message_count, messages.length),
        last_message_at: lastMessage?.created_at ?? session.last_message_at,
        updated_at: lastMessage?.created_at ?? session.updated_at,
      });
      setAgentSessions((current) =>
        current.map((item) =>
          item.id === session.id
            ? {
                ...item,
                message_count: Math.max(item.message_count, messages.length),
                last_message_at: lastMessage?.created_at ?? item.last_message_at,
                updated_at: lastMessage?.created_at ?? item.updated_at,
              }
            : item,
        ),
      );
      const nextChatMessages = toChatMessages(messages);
      setChatMessages(
        approvalRequiredPayload
          ? [
              ...nextChatMessages,
              {
                id: `approval-${approvalRequiredPayload.approval_request_id}`,
                role: "assistant",
                content: buildApprovalChatMessage(approvalRequiredPayload),
                meta: "Tool approval",
              },
            ]
          : nextChatMessages,
      );
      if (lastMessage?.role === "assistant") {
        const nextContextMetadata = extractContextMetadata(lastMessage);
        setChatContextMetadata(nextContextMetadata);
        await refreshActiveTaskPlan(nextContextMetadata);
      } else if (approvalRequiredPayload) {
        await refreshActiveTaskPlan(approvalRequiredPayload.context_metadata);
      }
      return "AI 对话已流式回复并写入会话记忆。";
    });
  };

  return (
    <div className="app-root">
      <a className="skip-link" href="#main-content">
        跳到主内容
      </a>
      <AsciiArt className="app-ascii-background" rows={48} density="compact" intensity={0.2} seedText="agent tool runtime" showHeader={false} aria-hidden="true" />
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
              <button className="button button-primary" type="button" onClick={() => navigate("jobs")}>
                <Search size={16} />
                查岗位
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
                  contextMetadata={chatContextMetadata}
                  isLoading={chatLoading}
                  isWorking={workingAction === "agent-chat-send"}
                  approvalWorking={workingAction?.startsWith("agent-approval-") ?? false}
                  messages={chatMessages}
                  pendingApproval={pendingApproval}
                  runtimeEvents={runtimeEvents}
                  session={agentSession}
                  sessions={agentSessions}
                  taskPlan={taskPlan}
                  onDraftChange={setChatDraft}
                  onApprovePendingApproval={handleApprovePendingApproval}
                  onPromptSelect={setChatDraft}
                  onRejectPendingApproval={handleRejectPendingApproval}
                  onSubmit={handleSendChat}
                  onCreateSession={handleCreateChatSession}
                  onDeleteSession={handleDeleteChatSession}
                  onRenameSession={handleRenameChatSession}
                  onSelectSession={handleSelectChatSession}
                  navigate={navigate}
                />
              ) : null}
              {activePage === "skills" ? (
                <SkillManagementPage
                  draft={skillImportDraft}
                  skills={agentSkills}
                  workingAction={workingAction}
                  onArchive={handleArchiveSkill}
                  onDraftChange={setSkillImportDraft}
                  onImport={handleImportSkill}
                  onPin={handlePinSkill}
                />
              ) : null}
              {activePage === "agents" ? <AgentRuntimePage panel={agentRuntimePanel} navigate={navigate} /> : null}
              {activePage === "dashboard" ? <DashboardPage summary={summary} sources={sources} leads={leads} navigate={navigate} /> : null}
              {activePage === "sources" ? (
                <SourcesPage
                  draft={sourceDraft}
                  editingSourceId={editingSourceId}
                  onDraftChange={setSourceDraft}
                  sources={sources}
                  workingAction={workingAction}
                  onCancelEdit={handleCancelSourceEdit}
                  onDisable={handleDisableSource}
                  onEdit={handleEditSource}
                  onSubmit={handleSubmitSource}
                  onSync={handleSyncSource}
                />
              ) : null}
              {activePage === "jobs" ? (
                <JobExhibitionPage
                  applications={applications}
                  companies={offerioCompanies}
                  companyPage={offerioCompanyPage}
                  companyTotal={offerioCompanyTotal}
                  companyTotalPages={offerioCompanyTotalPages}
                  filters={jobBoardFilters}
                  jobs={offerioJobs}
                  leads={leads}
                  openings={offerioOpenings}
                  openingPage={offerioOpeningPage}
                  openingTotal={offerioOpeningTotal}
                  openingTotalPages={offerioOpeningTotalPages}
                  selectedCompany={selectedOfferioCompany}
                  workingAction={workingAction}
                  onAddOpeningToPipeline={handleAddOfferIOOpeningToPipeline}
                  onAddToPipeline={handleAddOfferIOJobToPipeline}
                  onMarkLeadStatus={handleMarkLeadStatus}
                  onVerifyAndConvertLead={handleVerifyAndConvert}
                  onFiltersChange={setJobBoardFilters}
                  onPageChange={handleJobBoardPageChange}
                  onSearch={handleSearchJobBoard}
                  onSelectCompany={handleSelectOfferIOCompany}
                />
              ) : null}
              {activePage === "leads" ? (
                <LeadsPage
                  domainHealth={domainHealth}
                  extractDraft={extractDraft}
                  articleCandidates={articleCandidates}
                  sources={sources}
                  urlImportDraft={urlImportDraft}
                  urlImportProgress={urlImportProgress}
                  visiblePageDraft={visiblePageDraft}
                  workingAction={workingAction}
                  onExtract={handleExtractLeads}
                  onExtractDraftChange={setExtractDraft}
                  onImportUrl={handleImportUrl}
                  onSubmitVisiblePageContent={handleSubmitVisiblePageContent}
                  onUrlImportDraftChange={setUrlImportDraft}
                  onVisiblePageDraftChange={setVisiblePageDraft}
                />
              ) : null}
              {activePage === "pipeline" ? <PipelinePage applications={applications} navigate={navigate} onUpdateStatus={handleUpdateApplicationStatus} /> : null}
              {activePage === "guardrails" ? <GuardrailsPage /> : null}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

function ChatPage({
  approvalWorking,
  draft,
  contextMetadata,
  isLoading,
  isWorking,
  messages,
  navigate,
  pendingApproval,
  runtimeEvents,
  session,
  sessions,
  taskPlan,
  onApprovePendingApproval,
  onCreateSession,
  onDeleteSession,
  onDraftChange,
  onPromptSelect,
  onRejectPendingApproval,
  onRenameSession,
  onSelectSession,
  onSubmit,
}: {
  approvalWorking: boolean;
  draft: string;
  contextMetadata: AgentContextMetadata | null;
  isLoading: boolean;
  isWorking: boolean;
  messages: ChatMessage[];
  pendingApproval: AgentApprovalRequiredPayload | null;
  runtimeEvents: ChatRuntimeTimelineEvent[];
  session: AgentSession | null;
  sessions: AgentSession[];
  taskPlan: AgentTaskPlan | null;
  navigate: (page: PageId) => void;
  onApprovePendingApproval: () => void;
  onCreateSession: () => void;
  onDeleteSession: (session: AgentSession) => void;
  onDraftChange: (value: string) => void;
  onPromptSelect: (value: string) => void;
  onRejectPendingApproval: () => void;
  onRenameSession: (session: AgentSession) => void;
  onSelectSession: (sessionId: string) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
}) {
  const messageListRef = useRef<HTMLDivElement | null>(null);
  const shouldStickToBottomRef = useRef(true);

  useLayoutEffect(() => {
    if (!shouldStickToBottomRef.current) {
      return;
    }

    scrollChatMessagesToBottom(messageListRef.current);
  }, [messages, isLoading, isWorking]);

  const handleMessageListScroll = () => {
    const element = messageListRef.current;
    if (!element) {
      return;
    }

    shouldStickToBottomRef.current = isChatMessageListNearBottom(element);
  };

  const handleChatSubmit = (event: FormEvent<HTMLFormElement>) => {
    shouldStickToBottomRef.current = true;
    onSubmit(event);
    scrollChatMessagesToBottom(messageListRef.current);
  };

  const handleChatInputKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  };

  return (
    <section className="chat-layout">
      <aside className="glass-panel chat-session-panel" aria-label="AI 会话列表">
        <div className="chat-session-heading">
          <div>
            <p className="eyebrow">Sessions</p>
            <h3>对话列表</h3>
          </div>
          <button className="button button-primary chat-new-session" type="button" onClick={onCreateSession} disabled={isWorking || isLoading}>
            <Plus size={16} />
            新对话
          </button>
        </div>
        <div className="chat-session-list">
          {sessions.length ? (
            sessions.map((item) => (
              <article className={`chat-session-item ${session?.id === item.id ? "is-active" : ""}`} key={item.id}>
                <button className="chat-session-main" type="button" onClick={() => onSelectSession(item.id)} disabled={isLoading || isWorking}>
                  <strong>{formatSessionTitle(item)}</strong>
                  <small>
                    {formatDateTime(item.last_message_at ?? item.updated_at)} · {item.message_count} 条消息
                  </small>
                </button>
                <div className="chat-session-actions" aria-label="会话操作">
                  <button className="icon-button" type="button" aria-label="重命名会话" onClick={() => onRenameSession(item)} disabled={isWorking || isLoading}>
                    <Pencil size={14} />
                  </button>
                  <button className="icon-button icon-button-danger" type="button" aria-label="删除会话" onClick={() => onDeleteSession(item)} disabled={isWorking || isLoading}>
                    <Trash2 size={14} />
                  </button>
                </div>
              </article>
            ))
          ) : (
            <div className="chat-session-empty">
              <MessageCircle size={18} />
              <span>暂无历史会话</span>
            </div>
          )}
        </div>
      </aside>

      <div className="glass-panel chat-panel">
        <div className="chat-hero">
          <div className="chat-hero-icon" aria-hidden="true">
            <Bot size={22} />
          </div>
          <div>
            <p className="eyebrow">AI Career Copilot</p>
            <h3>AI 求职助手</h3>
            <p>已接入百炼模型、会话记忆和自动上下文压缩；回复会通过 SSE 流式输出。</p>
          </div>
        </div>

        <div ref={messageListRef} className="chat-message-list" aria-label="AI 对话消息" onScroll={handleMessageListScroll}>
          {isLoading ? (
            <article className="chat-message chat-message-assistant">
              <div className="chat-avatar" aria-hidden="true">
                <Loader2 className="spin" size={16} />
              </div>
              <div className="chat-bubble">
                <div className="chat-meta">
                  <strong>OfferMaster AI</strong>
                  <span>同步中</span>
                </div>
                <p>正在加载最近的 AI 会话记录。</p>
              </div>
            </article>
          ) : null}
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
                <ChatMessageContent content={message.content || (message.role === "assistant" && isWorking ? "正在生成回复..." : "")} />
              </div>
            </article>
          ))}
        </div>

        {pendingApproval ? (
          <ToolApprovalCard
            approval={pendingApproval}
            approvalWorking={approvalWorking}
            disabled={approvalWorking || isWorking || isLoading}
            onApprove={onApprovePendingApproval}
            onReject={onRejectPendingApproval}
          />
        ) : null}

        <div className="prompt-row" aria-label="快捷问题">
          {CHAT_PROMPTS.map((prompt) => (
            <button className="prompt-chip" key={prompt} type="button" onClick={() => onPromptSelect(prompt)}>
              {prompt}
            </button>
          ))}
        </div>

        <form className="chat-composer" onSubmit={handleChatSubmit}>
          <label htmlFor="chat-input">输入你想问 AI 的问题</label>
          <div className="chat-input-row">
            <textarea
              id="chat-input"
              value={draft}
              onChange={(event) => onDraftChange(event.target.value)}
              onKeyDown={handleChatInputKeyDown}
              placeholder="例如：帮我分析 Java 后端秋招岗位和我的项目匹配度"
              rows={3}
            />
            <button className="button button-primary chat-send" type="submit" disabled={!draft.trim() || isWorking || isLoading}>
              {isWorking ? <Loader2 className="spin" size={16} /> : <Send size={16} />}
              发送
            </button>
          </div>
          <p className="composer-hint">Enter 发送，Shift + Enter 换行。</p>
        </form>
      </div>

      <aside className="glass-panel chat-side-panel">
        <ChatTaskPlanPanel plan={taskPlan} isWorking={isWorking} />
        <ChatRuntimeTimeline events={runtimeEvents} isWorking={isWorking} />
        <button className="button button-ghost full-width" type="button" onClick={() => navigate("leads")}>
          <FileSearch size={16} />
          查看岗位线索
        </button>
      </aside>
    </section>
  );
}

function ChatTaskPlanPanel({ plan, isWorking }: { plan: AgentTaskPlan | null; isWorking: boolean }) {
  const stages = plan?.stages ?? [];
  const currentStage = stages.find((stage) => stage.capability === plan?.current_stage_id) ?? null;

  return (
    <section className="chat-task-plan-panel" aria-label="阶段任务链">
      <div className="task-plan-heading">
        <div>
          <p className="eyebrow">任务链</p>
          <h4>阶段任务链</h4>
        </div>
        <span className={`runtime-status-pill ${isWorking ? "is-running" : stages.length ? "is-finished" : "is-idle"}`}>
          {isWorking ? "运行中" : stages.length ? `${stages.length} 阶段` : "暂无"}
        </span>
      </div>

      {plan?.user_goal ? <p className="task-plan-goal">{plan.user_goal}</p> : null}

      {stages.length ? (
        <ol className="task-plan-step-list">
          {stages.map((stage) => (
            <ChatTaskPlanStageCard isCurrent={stage.step_id === currentStage?.step_id} key={stage.step_id} stage={stage} />
          ))}
        </ol>
      ) : (
        <div className="task-plan-empty">
          {isWorking ? <Loader2 className="spin" size={15} aria-hidden="true" /> : <Workflow size={15} aria-hidden="true" />}
          <span>{isWorking ? "任务阶段正在生成，稍后会显示阶段链。" : "暂无阶段任务链，发送任务后会在这里展示。"}</span>
        </div>
      )}
    </section>
  );
}

function ChatTaskPlanStageCard({ stage, isCurrent }: { stage: AgentTaskPlanStage; isCurrent: boolean }) {
  const tone = taskPlanStageTone(stage.status);
  const Icon = taskPlanStageIcon(stage.status);
  const handoffSummary = taskPlanPayloadSummary(stage.handoff_payload);
  const receivedSummary = taskPlanPayloadSummary(stage.received_context);
  const toolNames = taskPlanToolNames(stage.handoff_payload ?? stage.received_context);
  const allowedToolNames = stage.allowed_capabilities ?? [];
  const strategyLabel = taskPlanStrategyLabel(stage.tool_strategy);
  const rankingPolicy = stage.ranking_policy ?? [];

  return (
    <li className={`task-plan-step-card task-plan-step-${tone} ${isCurrent ? "is-current" : ""}`}>
      <span className="task-plan-step-icon" aria-hidden="true">
        <Icon size={14} />
      </span>
      <div className="task-plan-step-body">
        <div className="task-plan-step-title-row">
          <strong>{stage.title}</strong>
          <span>{taskPlanStatusLabel(stage.status)}</span>
        </div>
        <p>{stage.objective}</p>
        {stage.business_action ? <TaskPlanInfoBox label="业务动作" summary={stage.business_action} /> : null}
        {allowedToolNames.length ? (
          <div className="task-plan-strategy-box">
            <b>可用工具</b>
            <div className="task-plan-tool-tags" aria-label="阶段可用工具">
              {allowedToolNames.map((toolName) => (
                <span key={`${stage.step_id}-allowed-${toolName}`}>{formatRuntimeToolName(toolName)}</span>
              ))}
            </div>
          </div>
        ) : null}
        {strategyLabel ? <TaskPlanInfoBox label="工具策略" summary={strategyLabel} /> : null}
        {rankingPolicy.length ? (
          <div className="task-plan-strategy-box">
            <b>匹配排序规则</b>
            <ul className="task-plan-ranking-list">
              {rankingPolicy.slice(0, 4).map((item) => (
                <li key={`${stage.step_id}-ranking-${item}`}>{item}</li>
              ))}
            </ul>
          </div>
        ) : null}
        {toolNames.length ? (
          <div className="task-plan-tool-tags" aria-label="阶段工具">
            {toolNames.map((toolName) => (
              <span key={`${stage.step_id}-${toolName}`}>{formatRuntimeToolName(toolName)}</span>
            ))}
          </div>
        ) : null}
        {receivedSummary ? <TaskPlanInfoBox label="收到上游信息" summary={receivedSummary} /> : null}
        {handoffSummary ? <TaskPlanInfoBox label="阶段产物" summary={handoffSummary} /> : null}
      </div>
    </li>
  );
}

function TaskPlanInfoBox({ label, summary }: { label: string; summary: string }) {
  return (
    <div className="task-plan-handoff-box">
      <b>{label}</b>
      <span>{summary}</span>
    </div>
  );
}

function ChatRuntimeTimeline({ events, isWorking }: { events: ChatRuntimeTimelineEvent[]; isWorking: boolean }) {
  const runtimeTimelineRef = useRef<HTMLElement | null>(null);
  const shouldRuntimeTimelineStickToBottomRef = useRef(true);
  const [runtimeFlowExpanded, setRuntimeFlowExpanded] = useState(false);
  const visibleEvents = events.slice(-16);
  const groupedEvents = runtimeEventGroups(visibleEvents);
  const eventOrder = new Map(visibleEvents.map((event, index) => [event.id, index]));
  const latestEvent = visibleEvents[visibleEvents.length - 1] ?? null;
  const statusLabel = isWorking ? "运行中" : visibleEvents.length ? "已结束" : "待运行";

  useLayoutEffect(() => {
    if (isWorking && visibleEvents.length <= 1) {
      shouldRuntimeTimelineStickToBottomRef.current = true;
    }
  }, [isWorking, visibleEvents.length]);

  useLayoutEffect(() => {
    if (!shouldRuntimeTimelineStickToBottomRef.current) {
      return;
    }

    scrollElementToBottom(runtimeTimelineRef.current);
  }, [visibleEvents.length, latestEvent?.id, isWorking]);

  const handleRuntimeTimelineScroll = () => {
    const element = runtimeTimelineRef.current;
    if (!element) {
      return;
    }

    shouldRuntimeTimelineStickToBottomRef.current = isScrollableElementNearBottom(element);
  };

  return (
    <section ref={runtimeTimelineRef} className="chat-runtime-timeline" aria-label="Agent 执行过程" aria-live="polite" onScroll={handleRuntimeTimelineScroll}>
      <div className="chat-runtime-heading">
        <div>
          <p className="eyebrow">执行过程</p>
          <h4>本轮执行过程</h4>
        </div>
        <div className="chat-runtime-heading-actions">
          <button
            type="button"
            className="runtime-flow-toggle"
            aria-expanded={runtimeFlowExpanded}
            onClick={() => setRuntimeFlowExpanded((expanded) => !expanded)}
          >
            <Workflow size={14} aria-hidden="true" />
            {runtimeFlowExpanded ? "收起流程图" : "展开流程图"}
          </button>
          <span className={`runtime-status-pill ${isWorking ? "is-running" : visibleEvents.length ? "is-finished" : "is-idle"}`}>{statusLabel}</span>
        </div>
      </div>
      <RuntimeFlowMap events={visibleEvents} />
      {runtimeFlowExpanded ? <RuntimeExpandedFlowBoard events={visibleEvents} isWorking={isWorking} /> : null}
      {visibleEvents.length ? (
        <div className="runtime-event-groups">
          {groupedEvents.map((group) => (
            <section className="runtime-event-group" key={group.group} aria-label={runtimeEventGroupTitle(group.group)}>
              <div className="runtime-event-group-heading">
                <h5>{runtimeEventGroupTitle(group.group)}</h5>
                <span>{group.events.length} 项</span>
              </div>
              <ol className="runtime-event-list">
                {group.events.map((event, groupEventIndex) => {
                  const index = eventOrder.get(event.id) ?? groupEventIndex;
                  const Icon = runtimeEventIcon(event);
                  const details = runtimeEventDetails(event);
                  const actor = runtimeEventActor(event);
                  const chain = runtimeAgentChain(event, actor);
                  const payloadPreview = runtimePayloadPreview(event);
                  return (
                    <li
                      className={`runtime-event-item runtime-event-animated runtime-event-${event.tone}`}
                      key={event.id}
                      style={{ animationDelay: runtimeEventAnimationDelay(index) }}
                    >
                      <span className="runtime-event-icon" aria-hidden="true">
                        <Icon className={runtimeEventShouldSpin(event, latestEvent, isWorking) ? "spin-slow" : undefined} size={15} />
                      </span>
                      <div className={`runtime-event-card ${runtimeActorEventCardClass(actor.type)}`}>
                        <div className="runtime-event-card-head">
                          <div className="runtime-event-title-row">
                            <RuntimeActorBadge actor={actor} />
                            <strong>{event.label}</strong>
                            {event.toolName ? <code>{formatRuntimeToolName(event.toolName)}</code> : null}
                          </div>
                          <span className="runtime-event-state">{runtimeEventStateLabel(event)}</span>
                        </div>
                        <p>{formatRuntimeEventSummary(event)}</p>
                        {chain.length ? <RuntimeAgentChain chain={chain} /> : null}
                        <div className="runtime-event-meta-row">
                          {event.stepIndex ? <span>第 {event.stepIndex} 步</span> : null}
                          <span>{actor.detail}</span>
                        </div>
                        {event.inputHint ? <small>{event.inputHint}</small> : null}
                        {details.length ? (
                          <div className="runtime-event-detail-grid">
                            {details.map((detail) => (
                              <span key={`${event.id}-${detail.label}`}>
                                <b>{detail.label}</b>
                                {detail.value}
                              </span>
                            ))}
                          </div>
                        ) : null}
                        {payloadPreview ? <pre className="runtime-event-payload-preview">{payloadPreview}</pre> : null}
                        {event.candidateNames?.length ? (
                          <div className="runtime-event-candidates" aria-label="候选能力">
                            {event.candidateNames.map((candidate) => (
                              <span key={candidate}>{candidate}</span>
                            ))}
                          </div>
                        ) : null}
                        {event.evidence?.length ? (
                          <div className="runtime-event-evidence" aria-label="证据来源">
                            {event.evidence.map((item) =>
                              item.url ? (
                                <a href={item.url} key={`${item.title}-${item.url}`} rel="noreferrer" target="_blank">
                                  <Link2 size={12} aria-hidden="true" />
                                  {item.title}
                                </a>
                              ) : (
                                <span key={item.title}>{item.title}</span>
                              ),
                            )}
                          </div>
                        ) : null}
                      </div>
                    </li>
                  );
                })}
              </ol>
            </section>
          ))}
        </div>
      ) : (
        <div className="runtime-event-empty">
          {isWorking ? <Loader2 className="spin" size={15} aria-hidden="true" /> : <Activity size={15} aria-hidden="true" />}
          <span>{isWorking ? "正在等待后端运行事件。" : "暂无执行过程，本轮还未调用工具。"}</span>
        </div>
      )}
    </section>
  );
}

function RuntimeFlowMap({ events }: { events: ChatRuntimeTimelineEvent[] }) {
  const actorCounts = runtimeActorCounts(events);
  const mainVisual = runtimeActorVisual("main_model");
  const subAgentVisual = runtimeActorVisual("sub_agent");
  const toolVisual = runtimeActorVisual("local_tool");
  const returnVisual = runtimeActorVisual("observation");
  return (
    <div className="runtime-flow-map" aria-label="Agent 执行链路概览">
      <RuntimeFlowNode visual={mainVisual} count={actorCounts.main_model + actorCounts.runtime} />
      <ArrowRight size={14} aria-hidden="true" />
      <RuntimeFlowNode visual={subAgentVisual} count={actorCounts.sub_agent} />
      <ArrowRight size={14} aria-hidden="true" />
      <RuntimeFlowNode visual={toolVisual} count={actorCounts.local_tool} />
      <ArrowRight size={14} aria-hidden="true" />
      <RuntimeFlowNode visual={returnVisual} count={actorCounts.observation} />
    </div>
  );
}

function RuntimeFlowNode({ visual, count }: { visual: RuntimeActorVisual; count: number }) {
  const Icon = visual.Icon;
  return (
    <div className={`runtime-flow-node ${visual.className}`}>
      <Icon size={15} aria-hidden="true" />
      <span>{visual.label}</span>
      <b>{count} 步</b>
    </div>
  );
}

function RuntimeExpandedFlowBoard({ events, isWorking }: { events: ChatRuntimeTimelineEvent[]; isWorking: boolean }) {
  const flowEvents = events.length ? events : buildRuntimeEmptyFlowEvents(isWorking);

  return (
    <div className="runtime-flow-expanded-board" aria-label="横向 Agent 执行流程">
      <ol className="runtime-flow-lane-list">
        {flowEvents.map((event, index) => {
          const actor = runtimeEventActor(event);
          const visual = runtimeActorVisual(actor.type);
          const Icon = visual.Icon;
          const details = runtimeEventDetails(event).slice(0, 2);
          const chain = runtimeAgentChain(event, actor);
          return (
            <li className={`runtime-flow-lane-card ${visual.className}`} key={`${event.id}-${index}`}>
              <div className="runtime-flow-lane-topline">
                <span className="runtime-flow-lane-icon" aria-hidden="true">
                  <Icon size={15} />
                </span>
                <span className="runtime-flow-lane-step">{index + 1}</span>
              </div>
              <div className="runtime-flow-lane-body">
                <b>{visual.label}</b>
                <small>{visual.caption}</small>
                <strong>{event.label}</strong>
                <p>{formatRuntimeEventSummary(event)}</p>
              </div>
              {event.toolName ? <code>{formatRuntimeToolName(event.toolName)}</code> : null}
              {chain.length ? <RuntimeAgentChain chain={chain} /> : null}
              {details.length ? (
                <div className="runtime-flow-lane-details">
                  {details.map((detail) => (
                    <span key={`${event.id}-lane-${detail.label}`}>
                      <b>{detail.label}</b>
                      {detail.value}
                    </span>
                  ))}
                </div>
              ) : null}
            </li>
          );
        })}
      </ol>
    </div>
  );
}

function RuntimeActorBadge({ actor }: { actor: RuntimeActorInfo }) {
  const visual = runtimeActorVisual(actor.type);
  const Icon = visual.Icon;
  return (
    <span className={`runtime-actor-badge ${runtimeActorBadgeClass(actor.type)}`}>
      <Icon size={12} aria-hidden="true" />
      {actor.label}
    </span>
  );
}

function RuntimeAgentChain({ chain }: { chain: string[] }) {
  return (
    <div className="runtime-agent-chain" aria-label="执行链路">
      {chain.map((item, index) => (
        <span key={`${item}-${index}`}>
          {index > 0 ? <ChevronRight size={12} aria-hidden="true" /> : null}
          {item}
        </span>
      ))}
    </div>
  );
}

function ToolApprovalCard({
  approval,
  approvalWorking,
  disabled,
  onApprove,
  onReject,
}: {
  approval: AgentApprovalRequiredPayload;
  approvalWorking: boolean;
  disabled: boolean;
  onApprove: () => void;
  onReject: () => void;
}) {
  return (
    <section className="tool-approval-card" aria-label="工具调用确认">
      <div className="tool-approval-icon" aria-hidden="true">
        <ShieldCheck size={18} />
      </div>
      <div className="tool-approval-body">
        <p className="eyebrow">Tool Approval</p>
        <h4>需要确认工具调用</h4>
        <p>{buildApprovalChatMessage(approval)}</p>
        <div className="tool-approval-meta">
          <span>tool: {approval.tool_name}</span>
          <span>decision: {approval.permission_decision ?? "ask"}</span>
          <span>skills: {approval.skill_ids.length || 0}</span>
        </div>
      </div>
      <div className="tool-approval-actions">
        <button className="button button-ghost" type="button" onClick={onReject} disabled={disabled}>
          {approvalWorking ? <Loader2 className="spin" size={16} /> : <X size={16} />}
          {approvalWorking ? "处理中" : "拒绝"}
        </button>
        <button className="button button-primary" type="button" onClick={onApprove} disabled={disabled}>
          {approvalWorking ? <Loader2 className="spin" size={16} /> : <CheckCircle2 size={16} />}
          {approvalWorking ? "正在执行" : "确认执行"}
        </button>
      </div>
    </section>
  );
}

function SkillManagementPage({
  draft,
  skills,
  workingAction,
  onArchive,
  onDraftChange,
  onImport,
  onPin,
}: {
  draft: SkillImportDraft;
  skills: AgentSkill[];
  workingAction: string | null;
  onArchive: (skill: AgentSkill) => void;
  onDraftChange: (draft: SkillImportDraft) => void;
  onImport: (event: FormEvent<HTMLFormElement>) => void;
  onPin: (skill: AgentSkill) => void;
}) {
  const pinnedCount = skills.filter((skill) => skill.pinned).length;
  const needsAttentionCount = skills.filter((skill) => ["partial", "unavailable"].includes(getSkillAvailabilityState(skill))).length;

  return (
    <section className="skill-management-grid">
      <Panel className="span-12 skill-list-panel" title="已接入能力" eyebrow="Skill 管理">
        <div className="skill-list-toolbar">
          <div className="skill-catalog-summary" aria-label="Skill 目录概览">
          <MetricRow label="已启用" value={`${skills.length} 个`} />
          <MetricRow label="已置顶" value={`${pinnedCount} 个`} />
            <MetricRow label="需处理" value={`${needsAttentionCount} 个`} />
          </div>
          <details className="skill-import-drawer">
            <summary>
              <Plus size={16} />
              导入新 Skill
            </summary>
            <form className="form-stack" onSubmit={onImport}>
              <p className="form-hint">把本机的 Skill 登记进平台后，主 agent 才能在对话和工作流中按需加载它。</p>
              <label>
                <span>本地 Skill 文件或目录</span>
                <input
                  value={draft.source_path}
                  onChange={(event) => onDraftChange({ ...draft, source_path: event.target.value })}
                  placeholder="F:/skills/xiaohongshu-recruiting/SKILL.md"
                />
              </label>
              <label>
                <span>能力类型</span>
                <select value={draft.category} onChange={(event) => onDraftChange({ ...draft, category: event.target.value })}>
                  <option value="content_source">内容源解析</option>
                  <option value="job_discovery">岗位发现</option>
                  <option value="resume_delivery">简历投递</option>
                  <option value="tool_recovery">工具恢复经验</option>
                </select>
              </label>
              <button className="button button-primary full-width" type="submit" disabled={workingAction === "agent-skill-import"}>
                {workingAction === "agent-skill-import" ? <Loader2 className="spin" size={16} /> : <Plus size={16} />}
                导入 Skill
              </button>
            </form>
          </details>
        </div>

        {skills.length ? (
          <div className="skill-list">
            {skills.map((skill) => (
              <SkillListItem
                key={skill.id}
                skill={skill}
                workingAction={workingAction}
                onArchive={onArchive}
                onPin={onPin}
              />
            ))}
          </div>
        ) : (
          <EmptyState icon={Layers3} title="还没有导入 Skill" body="先安装或下载小红书、公众号、抖音内容解析 Skill，再把本地 Skill 文件路径登记进来。" />
        )}
      </Panel>
    </section>
  );
}

function SkillListItem({
  skill,
  workingAction,
  onArchive,
  onPin,
}: {
  skill: AgentSkill;
  workingAction: string | null;
  onArchive: (skill: AgentSkill) => void;
  onPin: (skill: AgentSkill) => void;
}) {
  const metadata = skill.metadata_json ?? {};
  const sourceTypes = normalizeMetadataList(metadata.source_types);
  const requiredTools = normalizeMetadataList(metadata.required_tools);
  const availableRequiredTools = normalizeMetadataList(metadata.available_required_tools);
  const missingRequiredTools = normalizeMetadataList(metadata.missing_required_tools);
  const missingOptionalTools = normalizeMetadataList(metadata.missing_optional_tools);
  const allowedTools = normalizeMetadataList(metadata.allowed_tools);
  const askTools = normalizeMetadataList(metadata.ask_tools);
  const disallowedTools = normalizeMetadataList(metadata.disallowed_tools);
  const descriptionQualityScore = typeof metadata.description_quality_score === "number" ? metadata.description_quality_score : null;
  const securityRiskLevel = typeof metadata.security_risk_level === "string" ? metadata.security_risk_level : "unknown";
  const availability_state = getSkillAvailabilityState(skill);
  const Icon = skillListIcon(skill);

  return (
    <article className="skill-list-item">
      <div className="skill-list-icon" aria-hidden="true">
        <Icon size={22} />
      </div>
      <div className="skill-list-main">
        <div className="skill-list-title-row">
          <div>
            <h3>{skillDisplayTitle(skill)}</h3>
            <p>{skillDisplayDescription(skill)}</p>
          </div>
          <span className={`skill-toggle skill-toggle-${availability_state}`} aria-label={`启用状态：${skillAvailabilityLabel(availability_state)}`}>
            <span />
          </span>
        </div>
        <div className="skill-list-meta-row" aria-label="Skill 简要信息">
          <span>能力类型：{skillCategoryLabel(skill.category)}</span>
          <span>使用场景：{skillUseCaseLabel(skill, sourceTypes)}</span>
          <span>启用状态：{skillAvailabilityLabel(availability_state)}</span>
          {skill.pinned ? <span>已置顶</span> : null}
        </div>
        <details className="skill-list-details">
          <summary>查看详情</summary>
          <div className="skill-detail-grid">
            <SkillDetail label="内容来源" value={formatSkillList(sourceTypes.map(skillSourceTypeLabel), "未指定来源")} />
            <SkillDetail label="关键依赖" value={formatSkillList(requiredTools.map(skillToolLabel), "无外部依赖")} />
            <SkillDetail label="依赖检查" value={skillDependencySummary(availableRequiredTools, missingRequiredTools, missingOptionalTools)} />
            <SkillDetail label="可自动使用" value={formatSkillList(allowedTools.map(skillToolLabel), "未单独声明")} />
            <SkillDetail label="需要确认" value={formatSkillList(askTools.map(skillToolLabel), "暂无")} />
            <SkillDetail label="安全边界" value={formatSkillList(disallowedTools.map(skillToolLabel), "暂无禁止项")} />
            <SkillDetail label="描述质量" value={descriptionQualityScore === null ? "未评分" : `${descriptionQualityScore}/10`} />
            <SkillDetail label="风险等级" value={skillRiskLabel(securityRiskLevel)} />
          </div>
        </details>
      </div>
      <div className="skill-list-actions">
        <button className="button button-small button-ghost" type="button" onClick={() => onPin(skill)} disabled={skill.pinned || workingAction === `agent-skill-pin-${skill.id}`}>
          <BadgeCheck size={14} />
          {skill.pinned ? "已置顶" : "置顶"}
        </button>
        <button className="button button-small button-danger" type="button" onClick={() => onArchive(skill)} disabled={workingAction === `agent-skill-archive-${skill.id}`}>
          <Trash2 size={14} />
          归档
        </button>
      </div>
    </article>
  );
}

function SkillDetail({ label, value }: { label: string; value: string }) {
  return (
    <div className="skill-detail-item">
      <strong>{label}</strong>
      <span>{value}</span>
    </div>
  );
}

function AgentRuntimePage({ panel, navigate }: { panel: AgentRuntimePanel | null; navigate: (page: PageId) => void }) {
  const [activeTab, setActiveTab] = useState<"members" | "capabilities">("members");

  if (!panel) {
    return (
      <section className="agent-console-layout">
        <div className="agent-console-main glass-panel agent-console-empty">
          <EmptyState icon={Network} title="Agent 面板暂未连接" body="后端启动后会显示主 agent、已注册子 agent 和每个能力声明。" />
        </div>
      </section>
    );
  }

  const activeAgents = panel.agents.filter((agent) => agent.status === "active").length;
  const standbyAgents = panel.agents.length - activeAgents;

  return (
    <section className="agent-console-layout" aria-label="Agent 运行时面板">
      <aside className="agent-console-profile glass-panel">
        <div className="agent-profile-mark" aria-hidden="true">
          <Bot size={28} />
        </div>
        <div className="agent-profile-heading">
          <p className="eyebrow">Main Orchestrator</p>
          <h3>{panel.main_agent.name}</h3>
          <p>{panel.main_agent.description}</p>
        </div>

        <div className="agent-console-stats" aria-label="Agent 注册概览">
          <MetricRow label="接入成员" value={`${panel.summary.agent_count} 个`} />
          <MetricRow label="能力声明" value={`${panel.summary.capability_count} 个`} />
          <MetricRow label="低风险能力" value={`${panel.summary.low_risk_count} 个`} />
          <MetricRow label="需确认能力" value={`${panel.summary.confirmation_required_count} 个`} />
        </div>

        <div className="agent-runtime-status-list">
          <RuntimeStatusRow icon={Workflow} label="统一运行时" value="已接入" />
          <RuntimeStatusRow icon={Network} label="搜索 Provider" value={panel.summary.configured_web_search_provider} />
          <RuntimeStatusRow icon={ShieldCheck} label="权限边界" value="运行时校验" />
          <RuntimeStatusRow icon={BadgeCheck} label="结果标准化" value="观察结果回传" />
        </div>

        <button className="button button-ghost full-width" type="button" onClick={() => navigate("skills")}>
          <Layers3 size={16} />
          查看 Skill 管理
        </button>
      </aside>

      <section className="agent-console-main glass-panel">
        <header className="agent-console-header">
          <div>
            <p className="eyebrow">Agent As Tool</p>
            <h3>运行时成员</h3>
            <p>
              主 agent 只负责编排；每个子 agent 或工具先声明能力，再进入统一调度。
            </p>
          </div>
          <div className="agent-console-counts" aria-label="成员状态">
            <span>{activeAgents} 个运行中</span>
            <span>{standbyAgents} 个待命</span>
          </div>
        </header>

        <div className="agent-console-tabs" role="tablist" aria-label="Agent 面板视图">
          <button className={activeTab === "members" ? "is-active" : ""} type="button" role="tab" aria-selected={activeTab === "members"} onClick={() => setActiveTab("members")}>
            <Network size={16} />
            成员
          </button>
          <button className={activeTab === "capabilities" ? "is-active" : ""} type="button" role="tab" aria-selected={activeTab === "capabilities"} onClick={() => setActiveTab("capabilities")}>
            <Sparkles size={16} />
            能力声明
          </button>
        </div>

        {activeTab === "members" ? <AgentMemberList agents={panel.agents} /> : <AgentCapabilityList capabilities={panel.capabilities} />}
      </section>
    </section>
  );
}

function RuntimeStatusRow({ icon: Icon, label, value }: { icon: LucideIcon; label: string; value: string }) {
  return (
    <div className="agent-runtime-status-row">
      <Icon size={16} aria-hidden="true" />
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function AgentMemberList({ agents }: { agents: AgentRuntimeMember[] }) {
  if (!agents.length) {
    return <EmptyState icon={Network} title="还没有注册成员" body="接入新的子 agent 后，它会在这里声明自己负责的能力。" />;
  }

  return (
    <div className="agent-member-list">
      {agents.map((agent) => (
        <article className="agent-member-card" key={agent.id}>
          <div className="agent-member-avatar" aria-hidden="true">
            {agent.kind === "external_agent" ? <Bot size={22} /> : <DatabaseZap size={22} />}
          </div>
          <div className="agent-member-body">
            <div className="agent-member-title-row">
              <div>
                <h3>{agent.name}</h3>
                <span>{agent.role}</span>
              </div>
              <span className={`agent-status-chip agent-status-${agent.status}`}>{agentStatusLabel(agent.status)}</span>
            </div>
            <p>{agent.description}</p>
            {agent.health ? <p className={`agent-health-line agent-health-${agent.health.status}`}>心跳检查：{agentHealthLabel(agent.health)}</p> : null}
            <div className="agent-chip-row" aria-label={`${agent.name} 能力列表`}>
              {agent.capabilities.length ? agent.capabilities.slice(0, 6).map((capability) => <span key={capability.id}>{capability.name}</span>) : <span>当前没有接管能力</span>}
            </div>
          </div>
        </article>
      ))}
    </div>
  );
}

function AgentCapabilityList({ capabilities }: { capabilities: AgentRuntimeCapability[] }) {
  if (!capabilities.length) {
    return <EmptyState icon={Sparkles} title="还没有能力声明" body="工具或子 agent 注册能力后，主 agent 才能把它放进候选工具池。" />;
  }

  return (
    <div className="agent-capability-grid">
      {capabilities.map((capability) => (
        <article className="agent-capability-card" key={capability.id}>
          <div className="agent-capability-header">
            <div>
              <p className="eyebrow">{capability.provider} · {capability.executor_id}</p>
              <h3>{capability.name}</h3>
            </div>
            <span className={`agent-risk-chip agent-risk-${capability.risk_level}`}>{riskLevelLabel(capability.risk_level)}</span>
          </div>
          <p>{capability.description}</p>
          <div className="agent-capability-facts">
            <span><strong>输入</strong>{formatCompactList(capability.input_fields, "未声明")}</span>
            <span><strong>输出</strong>{formatCompactList(capability.output_fields, "未声明")}</span>
            <span><strong>来源</strong>{formatCompactList(capability.allowed_source_types, "不限")}</span>
            <span><strong>确认</strong>{capability.requires_confirmation ? "需要用户确认" : "可自动执行"}</span>
          </div>
          <div className="agent-chip-row">
            {(capability.candidate_categories.length ? capability.candidate_categories : capability.candidate_keywords).slice(0, 5).map((item) => <span key={item}>{item}</span>)}
          </div>
        </article>
      ))}
    </div>
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
        <StatCard icon={ShieldCheck} label="待验证" value={summary.unverifiedLeads} helper="未验证线索暂不进公司数" tone="blue" />
        <StatCard icon={BadgeCheck} label="已验证" value={summary.verifiedLeads} helper="可转正式岗位" tone="green" />
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

        <Panel className="span-5" title="公司数据口径" eyebrow="Company Scope" actionLabel="去公司展览" onAction={() => navigate("jobs")}>
          <div className="action-list">
            <div className="action-item">
              <Building2 size={18} aria-hidden="true" />
              <div>
                <strong>只展示完整公司</strong>
                <p>文章或社媒识别出的公司信号数据不完整，先保留在线索补全流程里，暂不进入公司展览，也不计入公司数量。</p>
              </div>
            </div>
          </div>
        </Panel>
      </section>

      <section className="dashboard-grid">
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
        <Panel className="span-7" title="确认边界" eyebrow="Guardrails">
          <div className="action-list">
            <div className="action-item">
              <ShieldCheck size={18} aria-hidden="true" />
              <div>
                <strong>公司展览只放完整数据</strong>
                <p>公众号或社媒只说明某公司已开放校招时，平台先记录为待补全信号，等岗位和公司字段补齐后再进入公司展览。</p>
              </div>
            </div>
          </div>
        </Panel>
      </section>
    </>
  );
}

function JobExhibitionPage({
  applications,
  companies,
  companyPage,
  companyTotal,
  companyTotalPages,
  filters,
  jobs,
  leads,
  openings,
  openingPage,
  openingTotal,
  openingTotalPages,
  selectedCompany,
  workingAction,
  onAddOpeningToPipeline,
  onAddToPipeline,
  onMarkLeadStatus,
  onVerifyAndConvertLead,
  onFiltersChange,
  onPageChange,
  onSearch,
  onSelectCompany,
}: {
  applications: ApplicationBoardItem[];
  companies: OfferIOCompany[];
  companyPage: number;
  companyTotal: number;
  companyTotalPages: number;
  filters: JobBoardFilterDraft;
  jobs: OfferIOJob[];
  leads: JobLead[];
  openings: OfferIOCompanyOpening[];
  openingPage: number;
  openingTotal: number;
  openingTotalPages: number;
  selectedCompany: string | null;
  workingAction: string | null;
  onAddOpeningToPipeline: (opening: OfferIOCompanyOpening) => void;
  onAddToPipeline: (job: OfferIOJob) => void;
  onMarkLeadStatus: (lead: JobLead, status: JobLeadStatus) => void;
  onVerifyAndConvertLead: (lead: JobLead) => void;
  onFiltersChange: (filters: JobBoardFilterDraft) => void;
  onPageChange: (page: number) => void;
  onSearch: (event: FormEvent<HTMLFormElement>) => void;
  onSelectCompany: (company: OfferIOCompany) => void;
}) {
  const trackedSourceIds = new Set(applications.map((item) => item.job.source_job_id));
  const isOpeningMode = filters.source_mode === "company_openings";
  const activePage = isOpeningMode ? openingPage : companyPage;
  const totalPages = isOpeningMode ? openingTotalPages : companyTotalPages;
  const totalItems = isOpeningMode ? openingTotal : companyTotal;
  const importedLeads = dedupeJobLeadsForJobBoard(leads);

  return (
    <section className="job-board-layout">
      <Panel className="span-12" title="公司展览" eyebrow="Company Board">
        <form className="filter-bar job-board-filters" onSubmit={onSearch}>
          <label>
            <span>公司来源分类</span>
            <select value={filters.source_mode} onChange={(event) => onFiltersChange({ ...filters, source_mode: event.target.value as JobBoardFilterDraft["source_mode"], page: 1 })}>
              <option value="company_openings">开放岗位公司库</option>
              <option value="company_jobs">公司聚合岗位库</option>
            </select>
          </label>
          <label>
            <span>公司 / 关键词</span>
            <input value={filters.keyword} onChange={(event) => onFiltersChange({ ...filters, keyword: event.target.value })} placeholder="腾讯 / Java / Agent" />
          </label>
          <label>
            <span>行业</span>
            <input value={filters.industry} onChange={(event) => onFiltersChange({ ...filters, industry: event.target.value })} placeholder="互联网/游戏/软件" />
          </label>
          {isOpeningMode ? (
            <>
              <label>
                <span>批次</span>
                <input value={filters.batch} onChange={(event) => onFiltersChange({ ...filters, batch: event.target.value })} placeholder="秋招" />
              </label>
              <label>
                <span>届别</span>
                <input value={filters.target} onChange={(event) => onFiltersChange({ ...filters, target: event.target.value })} placeholder="2027届" />
              </label>
              <label>
                <span>企业性质</span>
                <input value={filters.company_nature} onChange={(event) => onFiltersChange({ ...filters, company_nature: event.target.value })} placeholder="民企 / 外企 / 央国企" />
              </label>
            </>
          ) : null}
          <label>
            <span>类型</span>
            <select value={filters.job_type} onChange={(event) => onFiltersChange({ ...filters, job_type: event.target.value })}>
              <option value="校招">校招</option>
              <option value="实习">实习</option>
            </select>
          </label>
          <label>
            <span>每页</span>
            <select value={filters.page_size} onChange={(event) => onFiltersChange({ ...filters, page_size: Number(event.target.value), page: 1 })}>
              <option value={20}>20</option>
              <option value={50}>50</option>
              <option value={100}>100</option>
            </select>
          </label>
          <button className="button button-primary" type="submit" disabled={workingAction === "offerio-company-search"}>
            {workingAction === "offerio-company-search" ? <Loader2 className="spin" size={16} /> : <Search size={16} />}
            筛选
          </button>
        </form>

        <div className="job-board-summary" aria-label="公司展览概览">
          <MetricRow label="当前来源" value={isOpeningMode ? "开放岗位公司库" : "公司聚合岗位库"} />
          <MetricRow label="来源库公司数" value={`${totalItems || (isOpeningMode ? openings.length : companies.length)} 个`} />
          <MetricRow label="当前筛选导入线索" value={`${importedLeads.length} 条`} />
        </div>

        {isOpeningMode ? (
          <OpeningSourceList openings={openings} trackedSourceIds={trackedSourceIds} workingAction={workingAction} onAddOpeningToPipeline={onAddOpeningToPipeline} />
        ) : (
          <CompanyJobTable companies={companies} selectedCompany={selectedCompany} onSelectCompany={onSelectCompany} />
        )}

        <JobBoardPagination page={activePage} totalPages={Math.max(1, totalPages)} onPageChange={onPageChange} />
      </Panel>

      <ImportedLeadList leads={importedLeads} workingAction={workingAction} onMarkStatus={onMarkLeadStatus} onVerifyAndConvert={onVerifyAndConvertLead} />

      {!isOpeningMode ? <Panel className="span-12" title={selectedCompany ? `${selectedCompany} 岗位` : "岗位详情"} eyebrow="Job Detail List">
        {jobs.length ? (
          <div className="job-card-grid">
            {jobs.map((job) => {
              const tracked = trackedSourceIds.has(job.id);
              return (
                <article className="job-display-card" key={job.id}>
                  <div className="job-display-card-main">
                    <div>
                      <p className="eyebrow">{job.company}</p>
                      <h3>{job.title}</h3>
                    </div>
                    <div className="tag-row">
                      <span>{job.location ?? "地点未披露"}</span>
                      <span>{job.category ?? "类别未披露"}</span>
                      <span>{job.job_type ?? filters.job_type}</span>
                      {job.department ? <span>{job.department}</span> : null}
                    </div>
                    <p>{[...(job.responsibilities ?? []), ...(job.requirements ?? [])].slice(0, 2).join("；") || "OfferIO 已给出结构化岗位，投递前仍建议官网验证。"}</p>
                  </div>
                  <div className="job-display-actions">
                    {job.apply_link ? (
                      <a className="button button-small button-ghost" href={job.apply_link} target="_blank" rel="noreferrer">
                        <ExternalLink size={14} />
                        原链接
                      </a>
                    ) : null}
                    <button className="button button-small button-primary" type="button" onClick={() => onAddToPipeline(job)} disabled={tracked || workingAction === `offerio-job-application-${job.id}`}>
                      {tracked ? <CheckCircle2 size={14} /> : <Plus size={14} />}
                      {tracked ? "已在投递板" : "加入投递板"}
                    </button>
                  </div>
                </article>
              );
            })}
          </div>
        ) : (
          <EmptyState icon={BriefcaseBusiness} title="先选择公司查看岗位" body="左侧表格点击“查看岗位”后，会从 OfferIO 临时接口拉取该公司的结构化 JD。" />
        )}
      </Panel> : null}
    </section>
  );
}

function CompanyJobTable({ companies, selectedCompany, onSelectCompany }: { companies: OfferIOCompany[]; selectedCompany: string | null; onSelectCompany: (company: OfferIOCompany) => void }) {
  return (
    <div className="table-wrap job-company-table">
      <table className="data-table">
        <thead>
          <tr>
            <th>操作</th>
            <th>公司</th>
            <th>企业性质</th>
            <th>行业</th>
            <th>工作地点</th>
            <th>岗位数</th>
            <th>更新时间</th>
          </tr>
        </thead>
        <tbody>
          {companies.map((company) => (
            <tr className={selectedCompany === company.name ? "is-selected" : ""} key={`${company.name}-${company.industry ?? ""}`}>
              <td>
                <button className="button button-small button-ghost" type="button" onClick={() => onSelectCompany(company)}>
                  <ExternalLink size={14} />
                  查看岗位
                </button>
              </td>
              <td>
                <strong>{company.name}</strong>
              </td>
              <td>
                <span className="pill company-nature">{company.company_nature ?? "未知"}</span>
              </td>
              <td>{company.industry ?? "未披露"}</td>
              <td className="muted-cell">{company.locations ?? "未披露"}</td>
              <td>{company.job_count}</td>
              <td>{company.updated_at ?? "未披露"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function OpeningSourceList({
  openings,
  trackedSourceIds,
  workingAction,
  onAddOpeningToPipeline,
}: {
  openings: OfferIOCompanyOpening[];
  trackedSourceIds: Set<string>;
  workingAction: string | null;
  onAddOpeningToPipeline: (opening: OfferIOCompanyOpening) => void;
}) {
  if (!openings.length) {
    return <EmptyState icon={BriefcaseBusiness} title="暂无开放岗位公司" body="调整关键词、行业、批次或届别后再筛选。" />;
  }

  return (
    <div className="job-card-grid opening-source-grid">
      {openings.map((opening) => {
        const sourceJobId = offerIOOpeningSourceJobId(opening);
        const tracked = trackedSourceIds.has(sourceJobId);
        return (
          <article className="job-display-card opening-source-card" key={opening.id}>
            <div className="job-display-card-main">
              <div>
                <p className="eyebrow">{[opening.batch, opening.target].filter(Boolean).join(" · ") || "OfferIO 来源"}</p>
                <h3>{opening.company_name}</h3>
              </div>
              <div className="tag-row">
                <span>{opening.company_nature ?? "性质未披露"}</span>
                <span>{opening.industry ?? "行业未披露"}</span>
                <span>{opening.location ?? "地点未披露"}</span>
                <span>{opening.has_written_test ?? "笔试未知"}</span>
              </div>
              <p>{opening.positions || "OfferIO 只给出公司开放信息，投递前建议进入原链接确认具体岗位。"}</p>
              <p className="muted-cell">截止：{opening.deadline ?? "未披露"} · 更新：{opening.update_date ?? "未披露"}</p>
            </div>
            <div className="job-display-actions">
              {opening.apply_link ? (
                <a className="button button-small button-ghost" href={opening.apply_link} target="_blank" rel="noreferrer">
                  <ExternalLink size={14} />
                  原链接
                </a>
              ) : null}
              <button className="button button-small button-primary" type="button" onClick={() => onAddOpeningToPipeline(opening)} disabled={tracked || workingAction === `offerio-opening-application-${opening.id}`}>
                {tracked ? <CheckCircle2 size={14} /> : <Plus size={14} />}
                {tracked ? "已在投递板" : "加入投递板"}
              </button>
            </div>
          </article>
        );
      })}
    </div>
  );
}

function JobBoardPagination({ page, totalPages, onPageChange }: { page: number; totalPages: number; onPageChange: (page: number) => void }) {
  return (
    <div className="pagination-row" aria-label="公司展览分页">
      <button className="button button-ghost" type="button" onClick={() => onPageChange(page - 1)} disabled={page <= 1}>
        上一页
      </button>
      <span>
        第 {page} / {totalPages} 页
      </span>
      <button className="button button-ghost" type="button" onClick={() => onPageChange(page + 1)} disabled={page >= totalPages}>
        下一页
      </button>
    </div>
  );
}

function SourcesPage({
  draft,
  editingSourceId,
  onDraftChange,
  sources,
  workingAction,
  onCancelEdit,
  onDisable,
  onEdit,
  onSubmit,
  onSync,
}: {
  draft: SourceDraft;
  editingSourceId: string | null;
  onDraftChange: (draft: SourceDraft) => void;
  sources: JobSource[];
  workingAction: string | null;
  onCancelEdit: () => void;
  onDisable: (source: JobSource) => void;
  onEdit: (source: JobSource) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  onSync: (source: JobSource) => void;
}) {
  const isEditing = editingSourceId !== null;
  const submitAction = isEditing ? `update-source-${editingSourceId}` : "create-source";

  return (
    <section className="content-grid">
      <Panel className="span-4" title={isEditing ? "编辑信息源" : "新增信息源"} eyebrow="Source Registry">
        {isEditing ? <p className="form-hint">正在修改已登记来源。保存后会立即刷新列表；禁用不会删除历史线索。</p> : null}
        <form className="form-stack" onSubmit={onSubmit}>
          <label>
            <span>名称</span>
            <input value={draft.name} onChange={(event) => onDraftChange({ ...draft, name: event.target.value })} placeholder="例如：大连海事就业网" />
          </label>
          <label>
            <span>来源类型</span>
            <select
              value={draft.source_type}
              onChange={(event) => {
                const sourceType = event.target.value as JobSourceType;
                onDraftChange({ ...draft, source_type: sourceType, fetch_mode: defaultFetchModeForSourceType(sourceType) });
              }}
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
            <span>{draft.source_type === "wechat_account" ? "账号标识 / 参考 URL" : "入口 URL"}</span>
            <input
              value={draft.entry_url}
              onChange={(event) => onDraftChange({ ...draft, entry_url: event.target.value })}
              placeholder={draft.source_type === "wechat_account" ? "可选：公众号微信号、搜狗链接或一篇参考文章" : "https://..."}
            />
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
          <button className="button button-primary full-width" type="submit" disabled={workingAction === submitAction}>
            {workingAction === submitAction ? <Loader2 className="spin" size={16} /> : isEditing ? <Save size={16} /> : <Plus size={16} />}
            {isEditing ? "保存修改" : "创建来源"}
          </button>
          {isEditing ? (
            <button className="button button-ghost full-width" type="button" onClick={onCancelEdit}>
              <X size={16} />
              取消编辑
            </button>
          ) : null}
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
                      <div className="table-actions">
                        <button
                          className="button button-small button-ghost"
                          type="button"
                          onClick={() => onEdit(source)}
                          disabled={workingAction !== null}
                        >
                          <Pencil size={14} />
                          编辑
                        </button>
                        <button
                          className="button button-small button-ghost"
                          type="button"
                          onClick={() => onSync(source)}
                          disabled={workingAction === `sync-${source.id}`}
                        >
                          {workingAction === `sync-${source.id}` ? <Loader2 className="spin" size={14} /> : <RefreshCcw size={14} />}
                          同步
                        </button>
                        <button
                          className="button button-small button-danger"
                          type="button"
                          onClick={() => onDisable(source)}
                          disabled={workingAction === `disable-source-${source.id}`}
                        >
                          {workingAction === `disable-source-${source.id}` ? <Loader2 className="spin" size={14} /> : <Trash2 size={14} />}
                          禁用
                        </button>
                      </div>
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
  domainHealth,
  extractDraft,
  articleCandidates,
  sources,
  urlImportDraft,
  urlImportProgress,
  visiblePageDraft,
  workingAction,
  onExtract,
  onExtractDraftChange,
  onImportUrl,
  onSubmitVisiblePageContent,
  onUrlImportDraftChange,
  onVisiblePageDraftChange,
}: {
  domainHealth: DomainHealth[];
  extractDraft: ExtractDraft;
  articleCandidates: ArticleCandidate[];
  sources: JobSource[];
  urlImportDraft: UrlImportDraft;
  urlImportProgress: UrlImportProgress;
  visiblePageDraft: VisiblePageDraft;
  workingAction: string | null;
  onExtract: (event: FormEvent<HTMLFormElement>) => void;
  onExtractDraftChange: (draft: ExtractDraft) => void;
  onImportUrl: (event: FormEvent<HTMLFormElement>) => void;
  onSubmitVisiblePageContent: (event: FormEvent<HTMLFormElement>) => void;
  onUrlImportDraftChange: (draft: UrlImportDraft) => void;
  onVisiblePageDraftChange: (draft: VisiblePageDraft) => void;
}) {
  return (
    <section className="content-grid">
      <Panel className="span-8" title="候选文章" eyebrow="Article Candidates">
        {articleCandidates.length ? (
          <div className="candidate-list">
            {articleCandidates.slice(0, 8).map((candidate) => (
              <article className="candidate-row" key={candidate.id}>
                <div>
                  <strong>{candidate.title}</strong>
                  <p>{candidate.source_account ?? "公众号账号"} · {ARTICLE_STATUS_LABELS[candidate.status]}</p>
                </div>
                <a className="icon-link" href={candidate.url} target="_blank" rel="noreferrer" aria-label={`打开 ${candidate.title}`}>
                  <ExternalLink size={14} />
                </a>
              </article>
            ))}
          </div>
        ) : (
          <EmptyState icon={FileSearch} title="还没有候选文章" body="添加微信公众号账号并同步后，近期招聘相关文章会先进入这里；只有公司和岗位字段完整的数据才进入“公司展览”。" />
        )}
      </Panel>

      <Panel className="span-4 utility-panel" title="临时链接导入" eyebrow="One-off Import">
        <p className="section-helper">URL 粘贴解析只是一次性兜底：用于还没登记成信息源的公众号文章、小红书笔记或临时招聘页。长期来源请先放到“信息源”，结构化公司和岗位请去“公司展览”。</p>
        <form className="form-stack" onSubmit={onImportUrl}>
          <label htmlFor="url-import-input">
            <span>招聘信息链接</span>
            <input
              id="url-import-input"
              type="url"
              value={urlImportDraft.url}
              onChange={(event) => onUrlImportDraftChange({ ...urlImportDraft, url: event.target.value })}
              placeholder="https://mp.weixin.qq.com/... 或临时招聘页"
              required
            />
          </label>
          <label>
            <span>归属来源</span>
            <select value={urlImportDraft.source_id} onChange={(event) => onUrlImportDraftChange({ ...urlImportDraft, source_id: event.target.value })}>
              <option value="">自动创建来源</option>
              {sources.map((source) => (
                <option key={source.id} value={source.id}>
                  {source.name}
                </option>
              ))}
            </select>
          </label>
          <div className="form-row">
            <label>
              <span>来源类型提示</span>
              <select value={urlImportDraft.source_hint} onChange={(event) => onUrlImportDraftChange({ ...urlImportDraft, source_hint: event.target.value as JobSourceType | "" })}>
                <option value="">自动识别</option>
                {SOURCE_TYPE_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span>可信度</span>
              <select value={urlImportDraft.trust_level} onChange={(event) => onUrlImportDraftChange({ ...urlImportDraft, trust_level: event.target.value as JobSourceTrustLevel | "" })}>
                <option value="">默认</option>
                {TRUST_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <label className="checkbox-row">
            <input type="checkbox" checked={urlImportDraft.force_refresh} onChange={(event) => onUrlImportDraftChange({ ...urlImportDraft, force_refresh: event.target.checked })} />
            <span>忽略重复 URL，强制重新解析</span>
          </label>
          <button className="button button-primary full-width" type="submit" disabled={workingAction === "import-url"}>
            {workingAction === "import-url" ? <Loader2 className="spin" size={16} /> : <Link2 size={16} />}
            导入临时链接
          </button>
        </form>

        <UrlImportStatus progress={urlImportProgress} domainHealth={domainHealth} isWorking={workingAction === "import-url"} />
        <VisiblePageFallbackForm
          draft={visiblePageDraft}
          progress={urlImportProgress}
          isWorking={workingAction === "submit-visible-page"}
          onDraftChange={onVisiblePageDraftChange}
          onSubmit={onSubmitVisiblePageContent}
        />
      </Panel>

      <Panel className="span-4" title="手动粘贴兜底" eyebrow="Manual Fallback">
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
    </section>
  );
}

function ImportedLeadList({
  leads,
  workingAction,
  onMarkStatus,
  onVerifyAndConvert,
}: {
  leads: JobLead[];
  workingAction: string | null;
  onMarkStatus: (lead: JobLead, status: JobLeadStatus) => void;
  onVerifyAndConvert: (lead: JobLead) => void;
}) {
  return (
    <Panel className="span-12" title="导入岗位线索" eyebrow="Imported Leads">
      <p className="section-helper">这里展示从公众号、小红书、临时链接和手动粘贴中抽取出来的岗位候选。相同公司+相同岗位标题会先在前端去重；投递前仍需要验证并转正式岗位。</p>
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
        <EmptyState icon={FileSearch} title="暂无导入岗位线索" body="从线索导入页解析文章或粘贴招聘文本后，具体岗位候选会出现在这里。" />
      )}
    </Panel>
  );
}

function dedupeJobLeadsForJobBoard(leads: JobLead[]): JobLead[] {
  const seen = new Set<string>();
  return leads.filter((lead) => {
    const company = normalizeCompanyNameForDedupe(lead.company_name);
    const title = normalizeCompanyNameForDedupe(lead.title);
    const key = `${company}:${title || lead.id}`;
    if (seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
}

function normalizeCompanyNameForDedupe(companyName: string | null | undefined): string {
  let normalized = (companyName ?? "")
    .replace(/[\s·・,，.。()（）【】\[\]《》<>]/g, "")
    .toLocaleLowerCase("zh-CN")
    .trim();

  const suffixPattern = /(股份有限公司|有限责任公司|有限公司|集团|控股|科技|信息技术|软件|网络|公司|校园招聘|校招|招聘)$/;
  let previous = "";
  while (normalized && normalized !== previous) {
    previous = normalized;
    normalized = normalized.replace(suffixPattern, "").trim();
  }

  return normalized;
}

function VisiblePageFallbackForm({
  draft,
  isWorking,
  onDraftChange,
  onSubmit,
  progress,
}: {
  draft: VisiblePageDraft;
  isWorking: boolean;
  onDraftChange: (draft: VisiblePageDraft) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  progress: UrlImportProgress;
}) {
  const run = progress.run;
  const accepted = progress.accepted;
  const needsVisiblePage = run?.next_action === "request_user_visible_page" || accepted?.status === "waiting_user";

  if (!needsVisiblePage) {
    return null;
  }

  return (
    <form className="visible-page-fallback form-stack" onSubmit={onSubmit}>
      <div className="url-import-advice">
        <AlertTriangle size={17} aria-hidden="true" />
        <div>
          <strong>需要可见页面正文</strong>
          <span>小红书这类页面不后台硬爬。请在可见浏览器/MCP 页面拿到正文后粘贴到这里继续解析；图片内容先记录数量，后续再做多模态解析。</span>
        </div>
      </div>
      <label>
        <span>页面标题</span>
        <input value={draft.title} onChange={(event) => onDraftChange({ ...draft, title: event.target.value })} placeholder="例如：27 届秋招 | 七月总结 - 小红书" />
      </label>
      <label>
        <span>最终 URL</span>
        <input value={draft.final_url} onChange={(event) => onDraftChange({ ...draft, final_url: event.target.value })} placeholder="可选：浏览器跳转后的 URL" />
      </label>
      <label>
        <span>可见页面正文</span>
        <textarea
          className="visible-page-textarea"
          value={draft.visible_text}
          onChange={(event) => onDraftChange({ ...draft, visible_text: event.target.value })}
          placeholder="粘贴小红书页面可见文本，系统会清理登录区、评论区和页脚。"
          rows={8}
          required
        />
      </label>
      <button className="button button-primary full-width" type="submit" disabled={isWorking}>
        {isWorking ? <Loader2 className="spin" size={16} /> : <FileSearch size={16} />}
        提交正文继续解析
      </button>
    </form>
  );
}

function UrlImportStatus({ domainHealth, isWorking, progress }: { domainHealth: DomainHealth[]; isWorking: boolean; progress: UrlImportProgress }) {
  const run = progress.run;
  const accepted = progress.accepted;
  const currentStatus = run?.status ?? accepted?.status;
  const currentStage = run?.current_stage ?? accepted?.current_stage;
  const visibleHealth = run?.domain ? domainHealth.filter((item) => item.domain === run.domain) : domainHealth;
  const flowSteps = buildUrlImportFlowSteps(run, accepted ?? null);
  const runExplanation = run ? getUrlImportRunExplanation(run) : null;

  if (!accepted && !run && !visibleHealth.length) {
    return (
      <div className="url-import-status is-idle" aria-live="polite">
        <Network size={17} aria-hidden="true" />
        <div>
          <strong>等待链接</strong>
          <p>公开网页会自动解析；小红书、Boss、登录或验证码页面会进入用户可见边界。</p>
        </div>
      </div>
    );
  }

  return (
    <div className="url-import-status" aria-live="polite">
      <div className="url-import-status-header">
        <div>
          <p className="eyebrow">Run State</p>
          <strong>{currentStatus ? URL_IMPORT_STATUS_LABELS[currentStatus] : "等待创建"}</strong>
          <span>{currentStage ? formatUrlImportStage(currentStage) : "尚未开始"}</span>
        </div>
        {isWorking || currentStatus === "running" ? <Loader2 className="spin" size={20} aria-hidden="true" /> : <Workflow size={20} aria-hidden="true" />}
      </div>

      <UrlImportFlow steps={flowSteps} />

      {runExplanation ? (
        <div className={`url-import-explanation explanation-${runExplanation.tone}`} role={runExplanation.tone === "danger" ? "alert" : "status"}>
          <strong>{runExplanation.title}</strong>
          <p>{runExplanation.body}</p>
        </div>
      ) : null}

      {run?.raw_content_preview ? <RawContentPreview run={run} /> : null}

      {run ? (
        <div className="url-import-metrics">
          <MetricRow label="抽取线索" value={`${run.extracted_count} 条`} />
          <MetricRow label="工具调用" value={`${run.tool_call_count} 次`} />
          <MetricRow label="LLM 调用" value={`${run.llm_call_count} 次`} />
          <MetricRow label="域名" value={run.domain ?? "未识别"} />
        </div>
      ) : null}

      {run?.error_code || run?.next_action ? (
        <div className="url-import-advice" role={run.error_message ? "alert" : "status"}>
          <AlertTriangle size={16} aria-hidden="true" />
          <div>
            <strong>{run.error_code ?? "需要处理"}</strong>
            <p>{run.error_message ?? NEXT_ACTION_LABELS[run.next_action ?? ""] ?? "请根据下一步动作继续。"}</p>
            {run.next_action ? <span>下一步：{NEXT_ACTION_LABELS[run.next_action] ?? run.next_action}</span> : null}
          </div>
        </div>
      ) : null}

      {visibleHealth.length ? (
        <div className="domain-health-list" aria-label="域名健康状态">
          {visibleHealth.slice(0, 3).map((item) => (
            <div className="domain-health-item" key={item.id}>
              <span className={`health-dot health-${item.state}`} aria-hidden="true" />
              <div>
                <strong>{item.domain}</strong>
                <p>
                  {item.tool_name} · {DOMAIN_HEALTH_LABELS[item.state]} · 失败 {item.failure_count} 次
                </p>
              </div>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function RawContentPreview({ run }: { run: ImportUrlRun }) {
  return (
    <div className="raw-content-preview" role="status">
      <div className="raw-content-preview-header">
        <div>
          <span className="eyebrow">Parsed Article</span>
          <strong>已解析到文章正文</strong>
        </div>
        <span className="pill">{run.raw_extraction_method ?? "raw text"}</span>
      </div>
      <p>{run.raw_content_preview}</p>
      {run.raw_image_count || run.raw_image_parse_deferred ? (
        <div className="raw-content-preview-meta">
          {run.raw_image_count ? <span>图片 {run.raw_image_count} 张</span> : null}
          {run.raw_image_parse_deferred ? <span>图片内容后续再解析</span> : null}
        </div>
      ) : null}
    </div>
  );
}

function UrlImportFlow({ steps }: { steps: UrlImportFlowStep[] }) {
  return (
    <ol className="url-import-flow" aria-label="URL 解析流程进度">
      {steps.map((step, index) => {
        const Icon = step.icon;
        return (
          <li className={`flow-step flow-${step.state}`} key={step.id}>
            <div className="flow-marker">
              <Icon size={14} aria-hidden="true" />
              <span>{String(index + 1).padStart(2, "0")}</span>
            </div>
            <div>
              <strong>{step.title}</strong>
              <p>{step.detail}</p>
            </div>
          </li>
        );
      })}
    </ol>
  );
}

function PipelinePage({
  applications,
  navigate,
  onUpdateStatus,
}: {
  applications: ApplicationBoardItem[];
  navigate: (page: PageId) => void;
  onUpdateStatus: (application: ApplicationBoardItem, status: ApplicationStatus) => void;
}) {
  return (
    <section className="application-board-section">
      <div className="application-board-toolbar glass-panel">
        <div>
          <p className="eyebrow">Application Progress</p>
          <h3>投递进度</h3>
          <p>这里是正式岗位的进度面板。你可以手动改状态，后续 agent 也会通过同一套 API 修改。</p>
        </div>
        <button className="button button-primary" type="button" onClick={() => navigate("jobs")}>
          <Plus size={16} />
          添加岗位
        </button>
      </div>
      <div className="application-stage-strip" aria-label="投递阶段统计">
        {APPLICATION_STAGES.map((stage) => (
          <button className="stage-chip" key={stage.status} type="button">
            <span className={`stage-dot stage-${stage.status}`} />
            {stage.label} {applications.filter((item) => item.status === stage.status).length}
          </button>
        ))}
      </div>
      <div className="application-board">
        {APPLICATION_STAGES.map((stage) => {
          const stageApplications = applications.filter((item) => item.status === stage.status);
          return (
            <section className="application-column" key={stage.status}>
              <div className="application-column-heading">
                <strong>{stage.label}</strong>
                <span>{stageApplications.length}</span>
              </div>
              <div className="application-card-stack">
                {stageApplications.length ? (
                  stageApplications.map((application) => (
                    <article className="application-card" key={application.id}>
                      <div className="application-card-title">
                        <strong>{application.job.title}</strong>
                        <span>{application.priority}</span>
                      </div>
                      <p>{application.job.company.name}</p>
                      <div className="tag-row">
                        <span>{application.job.city ?? "地点未披露"}</span>
                        <span>{application.channel ?? application.job.source}</span>
                      </div>
                      <select value={application.status} onChange={(event) => onUpdateStatus(application, event.target.value as ApplicationStatus)}>
                        {APPLICATION_STAGES.map((option) => (
                          <option key={option.status} value={option.status}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                      {application.notes ? <p className="application-note">{application.notes}</p> : null}
                    </article>
                  ))
                ) : (
                  <div className="application-drop-hint">拖拽或改状态后会进入这里</div>
                )}
              </div>
            </section>
          );
        })}
      </div>
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

function normalizeMetadataList(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value.map((item) => String(item).trim()).filter(Boolean);
  }
  if (typeof value === "string") {
    return value
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
  }
  return [];
}

function getSkillAvailabilityState(skill: AgentSkill): AgentSkillAvailabilityState {
  const state = skill.metadata_json?.availability_state;
  if (state === "available" || state === "partial" || state === "unavailable" || state === "disabled") {
    return state;
  }
  return normalizeMetadataList(skill.metadata_json?.required_tools).length ? "unavailable" : "available";
}

function skillAvailabilityLabel(state: AgentSkillAvailabilityState): string {
  const labels: Record<AgentSkillAvailabilityState, string> = {
    available: "可用",
    partial: "部分可用",
    unavailable: "缺少工具",
    disabled: "禁用",
  };
  return labels[state];
}

function skillDisplayTitle(skill: AgentSkill): string {
  const text = `${skill.name} ${skill.title}`.toLowerCase();
  if (text.includes("xiaohongshu")) {
    return "小红书内容抓取";
  }
  if (text.includes("wechat") || text.includes("weixin")) {
    return "微信公众号文章读取";
  }
  if (text.includes("resume")) {
    return "简历处理能力";
  }
  if (text.includes("pdf")) {
    return "PDF 文档处理";
  }
  return skill.title || humanizeIdentifier(skill.name);
}

function skillDisplayDescription(skill: AgentSkill): string {
  const text = `${skill.name} ${skill.title} ${skill.description}`.toLowerCase();
  if (text.includes("xiaohongshu")) {
    return "读取小红书笔记、账号或可见页面里的招聘信息，适合从社媒内容发现公司和岗位线索。";
  }
  if (text.includes("wechat") || text.includes("weixin")) {
    return "读取微信公众号文章内容，适合从校招汇总文章中提取公司、岗位和报名入口。";
  }
  return skill.description || "为主 agent 提供可按需加载的专项能力。";
}

function skillCategoryLabel(category: string): string {
  const labels: Record<string, string> = {
    content_source: "内容来源",
    job_discovery: "岗位发现",
    resume_delivery: "简历投递",
    tool_recovery: "工具恢复经验",
  };
  return labels[category] ?? humanizeIdentifier(category);
}

function skillUseCaseLabel(skill: AgentSkill, sourceTypes: string[]): string {
  if (sourceTypes.length) {
    return sourceTypes.map(skillSourceTypeLabel).slice(0, 2).join("、");
  }
  return skillCategoryLabel(skill.category);
}

function skillSourceTypeLabel(sourceType: string): string {
  const labels: Record<string, string> = {
    xiaohongshu_note: "小红书笔记",
    wechat_article: "公众号文章",
    wechat_account: "公众号账号",
    job_board_visible_page: "招聘页面",
    university_career_site: "高校就业网",
    official_career_site: "企业招聘官网",
    manual_clip: "手动粘贴内容",
  };
  return labels[sourceType] ?? humanizeIdentifier(sourceType);
}

function skillToolLabel(toolName: string): string {
  const labels: Record<string, string> = {
    "xiaohongshu-mcp.search_feeds": "搜索小红书笔记",
    "xiaohongshu-mcp.get_feed_detail": "读取小红书笔记详情",
    "xiaohongshu-mcp.check_login_status": "检查小红书登录状态",
    "xiaohongshu-mcp.get_login_qrcode": "获取小红书登录二维码",
    "xiaohongshu-mcp.user_profile": "读取小红书账号资料",
    "xhscrawl.search": "搜索小红书内容",
    "xhscrawl.note_detail": "读取小红书笔记详情",
    "xhscrawl.user_notes": "读取小红书用户笔记",
    "weixin-articles-mcp.read_article": "读取公众号文章",
    "weixin-articles.read_article": "读取公众号文章",
    read_article: "读取文章正文",
    "browser.open": "打开浏览器页面",
    mcp_visible_page_read: "读取用户可见页面",
    "mcp.visible_page_read": "读取用户可见页面",
    "ocr.extract_text": "图片文字识别",
    memory_search: "搜索会话记忆",
    memory_get: "读取会话记忆",
    submit_application: "提交投递动作",
  };
  return labels[toolName] ?? humanizeIdentifier(toolName);
}

function skillDependencySummary(availableTools: string[], missingTools: string[], missingOptionalTools: string[]): string {
  if (missingTools.length) {
    return `缺少 ${missingTools.length} 个必要依赖：${missingTools.map(skillToolLabel).slice(0, 2).join("、")}`;
  }
  if (missingOptionalTools.length) {
    return `核心可用，${missingOptionalTools.length} 个可选依赖未接入`;
  }
  if (availableTools.length) {
    return `必要依赖已满足：${availableTools.length} 个`;
  }
  return "无需额外依赖";
}

function skillRiskLabel(riskLevel: string): string {
  if (riskLevel === "low") {
    return "低风险";
  }
  if (riskLevel === "medium") {
    return "中风险";
  }
  if (riskLevel === "high") {
    return "高风险";
  }
  return "未标注";
}

function skillListIcon(skill: AgentSkill): LucideIcon {
  const text = `${skill.name} ${skill.title} ${skill.category}`.toLowerCase();
  if (text.includes("wechat") || text.includes("weixin") || text.includes("xiaohongshu")) {
    return MessageCircle;
  }
  if (text.includes("resume") || text.includes("pdf") || text.includes("document")) {
    return FileSearch;
  }
  if (text.includes("job") || text.includes("source")) {
    return RadioTower;
  }
  return Layers3;
}

function formatSkillList(values: string[], fallback: string): string {
  const cleanValues = values.map((value) => value.trim()).filter(Boolean);
  if (!cleanValues.length) {
    return fallback;
  }
  return cleanValues.slice(0, 4).join("、");
}

function humanizeIdentifier(value: string): string {
  return value
    .replaceAll("_", " ")
    .replaceAll("-", " ")
    .replaceAll(".", " ")
    .replace(/\s+/g, " ")
    .trim();
}

function sourceToDraft(source: JobSource): SourceDraft {
  return {
    name: source.name,
    source_type: source.source_type,
    entry_url: source.entry_url ?? "",
    sync_interval_hours: source.sync_interval_hours,
    trust_level: source.trust_level,
    fetch_mode: source.fetch_mode,
    notes: source.notes ?? "",
  };
}

function syncLimitForSource(source: JobSource): number {
  if (
    source.source_type === "official_api" &&
    source.fetch_mode === "official_api" &&
    source.entry_url?.includes("/api/recruitment/job-companies")
  ) {
    return 1000;
  }
  return 20;
}

function defaultFetchModeForSourceType(sourceType: JobSourceType): JobSourceFetchMode {
  if (["wechat_account", "xiaohongshu_note", "job_board_visible_page"].includes(sourceType)) {
    return "mcp_visible_page";
  }
  if (sourceType === "manual_clip" || sourceType === "wechat_article") {
    return "manual_clip";
  }
  if (sourceType === "official_api") {
    return "official_api";
  }
  return "public_html";
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

function offerIOJobToImportDraft(job: OfferIOJob): JobImportDraft {
  const rawIndustry = isRecord(job.raw_payload) && typeof job.raw_payload.industry === "string" ? job.raw_payload.industry : undefined;
  return {
    company_name: job.company,
    company_industry: rawIndustry,
    title: job.title,
    city: job.location,
    source: "offerio",
    source_job_id: job.id,
    source_url: job.apply_link,
    job_type: job.job_type,
    salary_text: job.salary,
    jd_text: buildOfferIOJobDescription(job),
    skills: job.category ? [job.category] : [],
    date_posted: job.publish_date,
    raw_payload: job.raw_payload,
  };
}

function offerIOOpeningToImportDraft(opening: OfferIOCompanyOpening): JobImportDraft {
  return {
    company_name: opening.company_name,
    company_industry: opening.industry,
    title: opening.positions ? `${opening.company_name} - ${opening.positions}` : `${opening.company_name} 校招开放`,
    city: opening.location,
    source: "offerio_company_openings",
    source_job_id: offerIOOpeningSourceJobId(opening),
    source_url: opening.apply_link,
    job_type: opening.batch,
    salary_text: null,
    jd_text: buildOfferIOOpeningDescription(opening),
    skills: opening.positions ? [opening.positions] : [],
    date_posted: normalizeOfferIODate(opening.update_date),
    raw_payload: opening.raw_payload,
  };
}

function offerIOOpeningSourceJobId(opening: OfferIOCompanyOpening): string {
  return `offerio_company_opening_${opening.id}`;
}

function buildOfferIOOpeningDescription(opening: OfferIOCompanyOpening): string {
  return [
    opening.positions ? `岗位方向：${opening.positions}` : null,
    opening.batch || opening.target ? `批次届别：${[opening.batch, opening.target].filter(Boolean).join(" / ")}` : null,
    opening.location ? `工作地点：${opening.location}` : null,
    opening.deadline ? `截止时间：${opening.deadline}` : null,
    opening.has_written_test ? `笔试信息：${opening.has_written_test}` : null,
  ]
    .filter(Boolean)
    .join("\n");
}

function normalizeOfferIODate(value: string | null): string | null {
  if (!value) {
    return null;
  }
  const normalized = value.replaceAll("/", "-");
  return /^\d{4}-\d{2}-\d{2}$/.test(normalized) ? normalized : null;
}

function buildOfferIOJobDescription(job: OfferIOJob): string {
  const sections: string[] = [];
  if (job.department) {
    sections.push(`部门：${job.department}`);
  }
  if (job.responsibilities.length) {
    sections.push(`岗位职责：\n${job.responsibilities.map((item) => `- ${item}`).join("\n")}`);
  }
  if (job.requirements.length) {
    sections.push(`岗位要求：\n${job.requirements.map((item) => `- ${item}`).join("\n")}`);
  }
  return sections.join("\n\n");
}

async function waitForUrlImportRun(runId: string, onProgress: (run: ImportUrlRun) => void): Promise<ImportUrlRun> {
  let latest = await pollUrlImportRun(runId);
  onProgress(latest);

  for (let attempt = 0; attempt < 12 && latest.status === "running"; attempt += 1) {
    await delay(1_000);
    latest = await pollUrlImportRun(runId);
    onProgress(latest);
  }

  return latest;
}

function isScrollableElementNearBottom(element: HTMLElement, threshold = 96): boolean {
  const { scrollTop, clientHeight, scrollHeight } = element;
  return scrollTop + clientHeight >= scrollHeight - threshold;
}

function scrollElementToBottom(element: HTMLElement | null): void {
  if (!element) {
    return;
  }

  window.requestAnimationFrame(() => {
    element.scrollTop = element.scrollHeight;
  });
}

function isChatMessageListNearBottom(element: HTMLDivElement, threshold = 96): boolean {
  return isScrollableElementNearBottom(element, threshold);
}

function scrollChatMessagesToBottom(element: HTMLDivElement | null): void {
  scrollElementToBottom(element);
}

function appendAgentMessageIfMissing(messages: AgentMessage[], message: AgentMessage | null | undefined): AgentMessage[] {
  if (!message || messages.some((item) => item.id === message.id)) {
    return messages;
  }
  return [...messages, message];
}

function parseMarkdownTables(content: string): ChatContentBlock[] {
  const lines = String(content || "").split("\n");
  const blocks: ChatContentBlock[] = [];
  let textBuffer: string[] = [];
  let index = 0;

  const flushText = () => {
    const text = textBuffer.join("\n").trim();
    if (text) {
      blocks.push({ type: "text", text });
    }
    textBuffer = [];
  };

  while (index < lines.length) {
    const current = lines[index] ?? "";
    const next = lines[index + 1] ?? "";
    if (isMarkdownTableRow(current) && isMarkdownTableSeparator(next)) {
      flushText();
      const headers = parseMarkdownTableRow(current);
      index += 2;
      const rows: string[][] = [];
      while (index < lines.length && isMarkdownTableRow(lines[index] ?? "") && !isMarkdownTableSeparator(lines[index] ?? "")) {
        rows.push(parseMarkdownTableRow(lines[index] ?? ""));
        index += 1;
      }
      if (headers.length && rows.length) {
        blocks.push({ type: "table", table: { headers, rows } });
      }
      continue;
    }
    textBuffer.push(current);
    index += 1;
  }

  flushText();
  return blocks.length ? blocks : [{ type: "text", text: content }];
}

function isMarkdownTableRow(line: string): boolean {
  return line.includes("|") && parseMarkdownTableRow(line).length >= 2;
}

function isMarkdownTableSeparator(line: string): boolean {
  const cells = parseMarkdownTableRow(line);
  return cells.length >= 2 && cells.every((cell) => /^:?-{3,}:?$/.test(cell.trim()));
}

function parseMarkdownTableRow(line: string): string[] {
  const trimmed = line.trim().replace(/^\|/, "").replace(/\|$/, "");
  return trimmed.split("|").map((cell) => cell.trim()).filter(Boolean);
}

function markdownTableToText(table: ParsedMarkdownTable): string {
  const separator = table.headers.map(() => "---");
  return [table.headers, separator, ...table.rows].map((row) => `| ${row.map(escapeMarkdownTableCell).join(" | ")} |`).join("\n");
}

function escapeMarkdownTableCell(value: string): string {
  return String(value || "").replaceAll("|", "／").replaceAll("\n", " ");
}

function toChatMessages(messages: AgentMessage[]): ChatMessage[] {
  const mapped = orderAgentMessagesForChat(messages)
    .map(toChatMessage)
    .filter((message) => message.content.trim());

  return mapped.length ? mapped : INITIAL_CHAT_MESSAGES;
}

function orderAgentMessagesForChat(messages: AgentMessage[]): AgentMessage[] {
  const visibleMessages = messages.filter(isChatAgentMessage);
  const messagesById = new Map(visibleMessages.map((message) => [message.id, message]));
  const pendingRepliesByParentId = new Map<string, AgentMessage[]>();

  visibleMessages.forEach((message) => {
    if (message.role !== "assistant" || !message.parent_message_id) {
      return;
    }

    const parent = messagesById.get(message.parent_message_id);
    if (parent?.role !== "user") {
      return;
    }

    const replies = pendingRepliesByParentId.get(parent.id) ?? [];
    replies.push(message);
    pendingRepliesByParentId.set(parent.id, replies);
  });

  const ordered: AgentMessage[] = [];
  const emittedIds = new Set<string>();
  const flushAssistantReplies = (parentMessageId: string) => {
    const replies = pendingRepliesByParentId.get(parentMessageId) ?? [];
    replies.forEach((reply) => {
      if (emittedIds.has(reply.id)) {
        return;
      }
      ordered.push(reply);
      emittedIds.add(reply.id);
    });
  };

  visibleMessages.forEach((message) => {
    if (emittedIds.has(message.id)) {
      return;
    }
    if (message.role === "assistant" && message.parent_message_id && pendingRepliesByParentId.has(message.parent_message_id)) {
      return;
    }

    ordered.push(message);
    emittedIds.add(message.id);
    flushAssistantReplies(message.id);
  });

  visibleMessages.forEach((message) => {
    if (emittedIds.has(message.id)) {
      return;
    }
    ordered.push(message);
    emittedIds.add(message.id);
    flushAssistantReplies(message.id);
  });

  return ordered;
}

function isChatAgentMessage(message: AgentMessage): boolean {
  return message.role === "user" || message.role === "assistant";
}

function toChatMessage(message: AgentMessage): ChatMessage {
  return {
    id: message.id,
    role: message.role as "assistant" | "user",
    content: message.visible_content_text ?? message.content_text ?? "",
    meta: message.role === "assistant" ? (message.metadata_json?.response_mode === "llm_stream" ? "SSE 流式" : "会话 API") : "你",
  };
}

function withoutWelcomeMessage(messages: ChatMessage[]): ChatMessage[] {
  return messages.filter((message) => message.id !== "assistant-welcome");
}

function buildApprovalChatMessage(approval: AgentApprovalRequiredPayload): string {
  const reason = approval.user_message || approval.reason || "当前 Skill 要求用户确认后才能执行该工具。";
  return `工具 ${approval.tool_name} 需要确认。${reason}`;
}

function appendRuntimeEvent(current: ChatRuntimeTimelineEvent[], nextEvent: ChatRuntimeTimelineEvent): ChatRuntimeTimelineEvent[] {
  return [...current, nextEvent].slice(-24);
}

function runtimeEventAnimationDelay(index: number): string {
  return `${Math.min(index, 8) * 35}ms`;
}

function toRuntimeEventFromOuterSession(payload: AgentStreamOuterSessionEvent, ordinal: number): ChatRuntimeTimelineEvent {
  const eventType = payload.event_type;
  const status = payload.status ?? null;
  return {
    id: runtimeEventId("outer_session", eventType, payload.run_id ?? payload.task_id ?? null, ordinal),
    kind: "outer_session",
    eventType,
    label: payload.event_label || formatRuntimeEventType(eventType),
    summary: stringOrNull(payload.waiting_message) ?? stringOrNull(payload.summary) ?? "主 agent 正在推进当前任务。",
    status,
    createdAt: Date.now(),
    tone: runtimeEventTone(eventType, status),
  };
}

function toRuntimeEventFromTool(payload: AgentStreamToolEvent, ordinal: number): ChatRuntimeTimelineEvent {
  const eventType = payload.event_type;
  const status = payload.status ?? null;
  const inputHint = runtimeInputHint(payload.suggested_input_patch ?? null);
  const toolName = stringOrNull(payload.tool_name) ?? stringOrNull(payload.capability);
  const candidateNames = runtimeCandidateNames(payload.candidate_capabilities ?? null);
  const inputPreview = recordOrNull(payload.input_preview);
  const resultSummary = recordOrNull(payload.result_summary);
  const reflection = recordOrNull(payload.reflection);
  const evidence = runtimeEvidence(payload.evidence ?? null);
  return {
    id: runtimeEventId("tool", eventType, payload.tool_call_id ?? payload.workflow_run_id ?? null, ordinal),
    kind: "tool",
    eventType,
    label: runtimeEventLabel(eventType, payload.event_label),
    summary: runtimeEventSummary(eventType, payload, toolName),
    status,
    toolName,
    stepIndex: typeof payload.step_index === "number" ? payload.step_index : null,
    inputHint,
    candidateNames,
    inputPreview,
    resultSummary,
    reflection,
    evidence,
    createdAt: Date.now(),
    tone: runtimeEventTone(eventType, status),
  };
}

function runtimeEventGroups(events: ChatRuntimeTimelineEvent[]): Array<{ group: ChatRuntimeEventGroup; events: ChatRuntimeTimelineEvent[] }> {
  const order: ChatRuntimeEventGroup[] = ["reasoning", "tooling", "observation", "evidence", "status"];
  const grouped = new Map<ChatRuntimeEventGroup, ChatRuntimeTimelineEvent[]>();
  events.forEach((event) => {
    const group = runtimeEventGroup(event.eventType);
    grouped.set(group, [...(grouped.get(group) ?? []), event]);
  });
  return order.flatMap((group) => {
    const groupEvents = grouped.get(group) ?? [];
    return groupEvents.length ? [{ group, events: groupEvents }] : [];
  });
}

function runtimeEventGroup(eventType: string): ChatRuntimeEventGroup {
  if (["reasoning_summary", "candidate_capabilities", "turn_started", "model_decision"].includes(eventType)) {
    return "reasoning";
  }
  if (["tool_input_preview", "tool_started", "tool_finished", "textual_tool_call_recovered", "textual_tool_call_blocked"].includes(eventType)) {
    return "tooling";
  }
  if (["tool_result_summary", "reflection_evaluation", "tool_reflection_retry", "observation_insufficient", "turn_finished"].includes(eventType)) {
    return "observation";
  }
  if (eventType === "evidence_selected") {
    return "evidence";
  }
  return "status";
}

function runtimeEventGroupTitle(group: ChatRuntimeEventGroup): string {
  return {
    reasoning: "思考摘要",
    tooling: "工具调用",
    observation: "观察与重试",
    evidence: "证据来源",
    status: "任务状态",
  }[group];
}

function runtimeEventId(kind: ChatRuntimeEventKind, eventType: string, runOrCallId: string | null, ordinal: number): string {
  return `${kind}-${eventType}-${runOrCallId ?? "runtime"}-${ordinal}-${Date.now()}`;
}

function runtimeEventTone(eventType: string, status?: string | null): ChatRuntimeEventTone {
  if (status === "failed" || status === "error" || status === "denied") {
    return "danger";
  }
  if (eventType === "waiting_user" || eventType === "tool_reflection_retry" || eventType === "textual_tool_call_blocked" || eventType === "observation_insufficient" || status === "retry") {
    return "warning";
  }
  if (eventType === "task_finished" || status === "succeeded" || status === "success") {
    return "success";
  }
  if (eventType === "task_started" || eventType === "turn_started" || eventType === "model_decision" || eventType === "candidate_capabilities" || eventType === "reasoning_summary" || eventType === "tool_input_preview" || eventType === "tool_started" || eventType === "textual_tool_call_recovered" || status === "running") {
    return "running";
  }
  return "muted";
}

function runtimeEventActor(event: ChatRuntimeTimelineEvent): RuntimeActorInfo {
  if (["candidate_capabilities", "reasoning_summary", "turn_started", "model_decision", "task_started"].includes(event.eventType)) {
    return {
      type: "main_model",
      label: "主模型调度",
      detail: "主 agent 在判断下一步、选择能力或规划本轮行动。",
    };
  }
  if (event.eventType === "textual_tool_call_blocked") {
    return {
      type: "runtime",
      label: "运行时守卫",
      detail: "模型输出了像工具调用的普通文字，运行时没有把它当成真实工具执行。",
    };
  }
  if (event.eventType === "textual_tool_call_recovered") {
    return {
      type: "runtime",
      label: "运行时守卫",
      detail: "模型输出了像工具调用的普通文字，运行时已转入真实工具流程。",
    };
  }
  if (["tool_result_summary", "reflection_evaluation", "tool_reflection_retry", "observation_insufficient", "evidence_selected", "turn_finished"].includes(event.eventType)) {
    return {
      type: "observation",
      label: "结果返回主 agent",
      detail: "子任务结果已回到主 agent，由主 agent 验收、重试或总结。",
    };
  }
  if (runtimeToolRunsInSubAgent(event.toolName)) {
    return {
      type: "sub_agent",
      label: "子 Agent 执行",
      detail: `${runtimeSubAgentName(event.toolName)} 正在执行主 agent 派发的任务。`,
    };
  }
  if (event.kind === "tool" || event.toolName) {
    return {
      type: "local_tool",
      label: "工具函数执行",
      detail: "主 agent 正在通过本地工具注册中心执行这个能力。",
    };
  }
  return {
    type: "runtime",
    label: "运行时守卫",
    detail: "后端运行时正在推进会话、权限或状态管理。",
  };
}

function runtimeActorVisual(actorType: ChatRuntimeActorType): RuntimeActorVisual {
  return {
    main_model: {
      Icon: Bot,
      label: "主模型调度",
      caption: "判断下一步、选择能力、汇总回答",
      className: "runtime-flow-lane-card-main",
    },
    sub_agent: {
      Icon: Network,
      label: "子 Agent 执行",
      caption: "Claude / OpenAI 等子 agent 接收任务",
      className: "runtime-flow-lane-card-subagent",
    },
    local_tool: {
      Icon: Wrench,
      label: "工具函数执行",
      caption: "搜索、文件、本地数据库等真实工具动作",
      className: "runtime-flow-lane-card-tool",
    },
    observation: {
      Icon: BadgeCheck,
      label: "结果返回主 agent",
      caption: "把观察结果交回主模型继续判断",
      className: "runtime-flow-lane-card-return",
    },
    runtime: {
      Icon: ShieldCheck,
      label: "运行时守卫",
      caption: "权限、参数、状态和纠偏检查",
      className: "runtime-flow-lane-card-runtime",
    },
  }[actorType];
}

function runtimeActorEventCardClass(actorType: ChatRuntimeActorType): string {
  return {
    main_model: "runtime-event-card-main",
    sub_agent: "runtime-event-card-subagent",
    local_tool: "runtime-event-card-tool",
    observation: "runtime-event-card-return",
    runtime: "runtime-event-card-runtime",
  }[actorType];
}

function buildRuntimeEmptyFlowEvents(isWorking: boolean): ChatRuntimeTimelineEvent[] {
  const now = Date.now();
  return [
    {
      id: "empty-main-model",
      kind: "outer_session",
      eventType: isWorking ? "turn_started" : "task_started",
      label: isWorking ? "等待主模型判断" : "主模型待命",
      summary: isWorking ? "主 agent 即将判断下一步。" : "本轮还没有真实执行事件。",
      status: isWorking ? "thinking" : "idle",
      createdAt: now,
      tone: isWorking ? "running" : "muted",
    },
    {
      id: "empty-worker",
      kind: "tool",
      eventType: "tool_started",
      label: "等待选择工具或子 Agent",
      summary: "当模型选择能力后，这里会显示被调用的一方。",
      status: "not_executed",
      createdAt: now + 1,
      tone: "muted",
    },
    {
      id: "empty-return",
      kind: "outer_session",
      eventType: "turn_finished",
      label: "等待结果返回",
      summary: "工具或子 Agent 完成后，结果会回到主 agent。",
      status: "not_executed",
      createdAt: now + 2,
      tone: "muted",
    },
  ];
}

function runtimeToolRunsInSubAgent(toolName?: string | null): boolean {
  if (!toolName) {
    return false;
  }
  return (
    toolName === "resume.tailor" ||
    toolName === "external.web_search" ||
    toolName === "applications.find_apply_entry" ||
    toolName.includes("openai") ||
    toolName.includes("claude") ||
    toolName.includes("agent")
  );
}

function runtimeSubAgentName(toolName?: string | null): string {
  if (toolName === "resume.tailor" || toolName?.includes("openai")) {
    return "OpenAI SDK Agent";
  }
  if (toolName?.includes("claude")) {
    return "Claude SDK Agent";
  }
  if (toolName === "external.web_search") {
    return "联网搜索 Agent";
  }
  if (toolName === "applications.find_apply_entry") {
    return "浏览器执行 Agent";
  }
  return "能力子 Agent";
}

function runtimeActorBadgeClass(actorType: ChatRuntimeActorType): string {
  return {
    main_model: "runtime-actor-badge-main",
    sub_agent: "runtime-actor-badge-subagent",
    local_tool: "runtime-actor-badge-tool",
    runtime: "runtime-actor-badge-runtime",
    observation: "runtime-actor-badge-return",
  }[actorType];
}

function runtimeActorCounts(events: ChatRuntimeTimelineEvent[]): Record<ChatRuntimeActorType, number> {
  return events.reduce<Record<ChatRuntimeActorType, number>>(
    (counts, event) => {
      const actorType = runtimeEventActor(event).type;
      return { ...counts, [actorType]: counts[actorType] + 1 };
    },
    { main_model: 0, sub_agent: 0, local_tool: 0, runtime: 0, observation: 0 },
  );
}

function runtimeAgentChain(event: ChatRuntimeTimelineEvent, actor: RuntimeActorInfo): string[] {
  if (actor.type === "main_model") {
    return ["主模型", "选择下一步"];
  }
  if (actor.type === "sub_agent") {
    return ["主 agent", runtimeSubAgentName(event.toolName), "返回主 agent"];
  }
  if (actor.type === "local_tool") {
    return ["主 agent", "本地工具注册中心", "返回主 agent"];
  }
  if (actor.type === "observation") {
    return ["工具结果", "主 agent 验收", event.eventType === "tool_reflection_retry" ? "准备重试" : "整理回答"];
  }
  return ["运行时", "更新状态"];
}

function runtimeEventStateLabel(event: ChatRuntimeTimelineEvent): string {
  if (event.status === "not_executed") {
    return "未执行";
  }
  if (event.status === "thinking") {
    return "思考中";
  }
  if (event.status === "retry") {
    return "重试";
  }
  if (event.status === "running") {
    return "执行中";
  }
  if (event.status === "succeeded" || event.status === "success") {
    return "完成";
  }
  if (event.status === "failed" || event.status === "error") {
    return "失败";
  }
  if (event.eventType === "task_finished") {
    return "结束";
  }
  return event.status ?? "记录";
}

function runtimePayloadPreview(event: ChatRuntimeTimelineEvent): string | null {
  if (event.inputPreview) {
    return runtimeCompactJson(event.inputPreview);
  }
  if (event.resultSummary) {
    return runtimeCompactJson(event.resultSummary);
  }
  if (event.reflection) {
    return runtimeCompactJson(event.reflection);
  }
  return null;
}

function runtimeCompactJson(value: Record<string, unknown>): string {
  const entries = Object.entries(value)
    .filter(([, item]) => item !== undefined && item !== null && item !== "")
    .slice(0, 6)
    .map(([key, item]) => `${runtimeFieldLabel(key)}：${runtimePreviewValue(item)}`);
  return entries.join("\n");
}

function runtimeFieldLabel(key: string): string {
  return {
    query: "关键词",
    max_results: "结果上限",
    path: "路径",
    encoding: "编码",
    result_count: "结果数",
    source_count: "来源数",
    source_domains: "来源域名",
    next_action: "下一步",
    reason: "原因",
  }[key] ?? key.replaceAll("_", " ");
}

function runtimePreviewValue(value: unknown): string {
  if (Array.isArray(value)) {
    return value.map((item) => String(item)).slice(0, 4).join("、");
  }
  if (isRecord(value)) {
    return Object.keys(value).slice(0, 4).join("、") || "对象";
  }
  const text = String(value);
  return text.length > 180 ? `${text.slice(0, 180)}...` : text;
}

function runtimeEventShouldSpin(
  event: ChatRuntimeTimelineEvent,
  latestEvent: ChatRuntimeTimelineEvent | null,
  isWorking: boolean,
): boolean {
  return Boolean(isWorking && event.tone === "running" && latestEvent?.id === event.id);
}

function runtimeEventIcon(event: ChatRuntimeTimelineEvent): LucideIcon {
  if (event.tone === "danger") {
    return AlertTriangle;
  }
  if (event.eventType === "textual_tool_call_blocked") {
    return AlertTriangle;
  }
  if (event.eventType === "textual_tool_call_recovered") {
    return Wrench;
  }
  if (event.eventType === "tool_reflection_retry" || event.eventType === "observation_insufficient") {
    return RefreshCcw;
  }
  if (event.eventType === "candidate_capabilities") {
    return Layers3;
  }
  if (event.eventType === "reasoning_summary") {
    return Sparkles;
  }
  if (event.eventType === "tool_input_preview") {
    return FileSearch;
  }
  if (event.eventType === "tool_result_summary") {
    return BadgeCheck;
  }
  if (event.eventType === "reflection_evaluation") {
    return Activity;
  }
  if (event.eventType === "evidence_selected") {
    return Link2;
  }
  if (event.eventType === "model_decision") {
    return Sparkles;
  }
  if (event.eventType === "waiting_user") {
    return Clock3;
  }
  if (event.tone === "success") {
    return CheckCircle2;
  }
  if (runtimeToolRunsInSubAgent(event.toolName)) {
    return Cpu;
  }
  if (event.toolName?.includes("search")) {
    return Search;
  }
  if (event.toolName?.includes("database")) {
    return DatabaseZap;
  }
  return event.kind === "tool" ? Activity : Workflow;
}

function formatRuntimeEventSummary(event: ChatRuntimeTimelineEvent): string {
  if (event.eventType === "tool_reflection_retry" && event.inputHint) {
    return `${event.summary} ${event.inputHint}`;
  }
  return event.summary;
}

function runtimeEventLabel(eventType: string, fallbackLabel?: string | null): string {
  return RUNTIME_EVENT_LABELS[eventType] ?? stringOrNull(fallbackLabel) ?? formatRuntimeEventType(eventType);
}

function runtimeEventSummary(eventType: string, payload: AgentStreamToolEvent, toolName: string | null): string {
  const explicitSummary = stringOrNull(payload.summary);
  if (explicitSummary) {
    return explicitSummary;
  }
  const toolLabel = toolName ? formatRuntimeToolName(toolName) : "候选能力";
  return RUNTIME_EVENT_SUMMARIES[eventType]?.(toolLabel) ?? "工具事件已更新。";
}

function formatRuntimeToolName(toolName: string): string {
  return RUNTIME_TOOL_LABELS[toolName] ?? toolName;
}

function formatRuntimeEventType(eventType: string): string {
  return eventType.replaceAll("_", " ");
}

function runtimeCandidateNames(capabilities: string[] | null): string[] {
  if (!Array.isArray(capabilities)) {
    return [];
  }
  return capabilities.map((capability) => formatRuntimeToolName(capability)).filter(Boolean).slice(0, 6);
}

function runtimeEventDetails(event: ChatRuntimeTimelineEvent): Array<{ label: string; value: string }> {
  const details: Array<{ label: string; value: string }> = [];
  const query = stringOrNull(event.inputPreview?.query);
  if (query) {
    details.push({ label: "关键词", value: query });
  }
  const maxResults = event.inputPreview?.max_results;
  if (typeof maxResults === "number" || typeof maxResults === "string") {
    details.push({ label: "上限", value: `${maxResults}` });
  }
  const resultCount = event.resultSummary?.result_count;
  if (typeof resultCount === "number") {
    details.push({ label: "结果数", value: `${resultCount}` });
  }
  const sourceCount = event.resultSummary?.source_count;
  if (typeof sourceCount === "number") {
    details.push({ label: "来源数", value: `${sourceCount}` });
  }
  const sourceDomains = Array.isArray(event.resultSummary?.source_domains)
    ? event.resultSummary.source_domains.map((domain) => String(domain)).filter(Boolean)
    : [];
  if (sourceDomains.length) {
    details.push({ label: "来源", value: sourceDomains.slice(0, 4).join("、") });
  }
  return details;
}

function runtimeEvidence(value: Array<Record<string, unknown>> | null): RuntimeEvidenceItem[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map((item) => {
      const title = stringOrNull(item.title) ?? stringOrNull(item.name) ?? "证据来源";
      const url = stringOrNull(item.url) ?? undefined;
      return { title, url };
    })
    .slice(0, 5);
}

function recordOrNull(value: unknown): Record<string, unknown> | null {
  return isRecord(value) ? value : null;
}

function runtimeInputHint(patch: Record<string, unknown> | null): string | null {
  if (!patch) {
    return null;
  }
  const query = stringOrNull(patch.query);
  if (query) {
    return `下一次会改用关键词：${query}`;
  }
  const keys = Object.keys(patch);
  return keys.length ? `下一次会调整输入字段：${keys.join("、")}` : null;
}

function taskPlanStageTone(status: string): ChatRuntimeEventTone {
  if (status === "succeeded") {
    return "success";
  }
  if (status === "failed" || status === "blocked") {
    return "danger";
  }
  if (status === "waiting_user") {
    return "warning";
  }
  if (status === "running") {
    return "running";
  }
  return "muted";
}

function taskPlanStageIcon(status: string): LucideIcon {
  if (status === "succeeded") {
    return CheckCircle2;
  }
  if (status === "failed" || status === "blocked") {
    return AlertTriangle;
  }
  if (status === "waiting_user") {
    return Clock3;
  }
  if (status === "running") {
    return Activity;
  }
  return CircleDot;
}

function taskPlanStatusLabel(status: string): string {
  return {
    pending: "待执行",
    running: "运行中",
    succeeded: "已完成",
    failed: "失败",
    skipped: "已跳过",
    waiting_user: "等用户",
    blocked: "已阻塞",
  }[status] ?? status;
}

function taskPlanPayloadSummary(payload: Record<string, unknown> | null | undefined): string | null {
  if (!payload) {
    return null;
  }
  const summary = stringOrNull(payload.summary) ?? stringOrNull(payload.final_answer_preview) ?? stringOrNull(payload.waiting_message);
  return summary ? summary.slice(0, 220) : null;
}

function taskPlanToolNames(payload: Record<string, unknown> | null | undefined): string[] {
  if (!payload || !Array.isArray(payload.tool_names)) {
    return [];
  }
  return payload.tool_names.map((toolName) => String(toolName)).filter(Boolean).slice(0, 3);
}

function taskPlanStrategyLabel(strategy: Record<string, unknown> | null | undefined): string | null {
  if (!strategy) {
    return null;
  }
  const description = stringOrNull(strategy.description);
  if (description) {
    return description;
  }
  const mode = stringOrNull(strategy.mode);
  if (mode === "none") {
    return "不调用工具，只基于已有信息分析或整理。";
  }
  if (mode === "allowlist") {
    return "只允许本阶段声明的工具。";
  }
  if (mode === "inherit") {
    return "继承本轮候选工具。";
  }
  return mode;
}

function stringOrNull(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function formatSessionTitle(session: AgentSession): string {
  return session.title?.trim() || "未命名对话";
}

function extractContextMetadata(message: AgentMessage): AgentContextMetadata | null {
  const metadata = message.metadata_json;
  const contextMetadata = metadata?.context_metadata;
  return isRecord(contextMetadata) ? (contextMetadata as AgentContextMetadata) : null;
}

function extractActiveTaskId(metadata: AgentContextMetadata | null): string | null {
  const outerSession = metadata?.outer_session_loop;
  if (!isRecord(outerSession)) {
    return null;
  }
  return stringOrNull(outerSession.active_task_id);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function formatUrlImportStage(stage: string): string {
  return URL_IMPORT_STAGE_LABELS[stage] ?? stage.replaceAll("_", " ");
}

function buildUrlImportFlowSteps(run: ImportUrlRun | null, accepted: ImportUrlAcceptedResponse | null): UrlImportFlowStep[] {
  const stage = run?.current_stage ?? accepted?.current_stage ?? "queued";
  const status = run?.status ?? accepted?.status;
  const nextAction = run?.next_action ?? null;
  const failed = status === "failed_recoverable" || status === "failed_terminal" || Boolean(run?.error_code);
  const blocked = status === "waiting_user" || nextAction === "request_user_visible_page" || nextAction === "request_manual_paste";
  const savedRaw = Boolean(run?.raw_job_lead_id) || stageRank(stage) >= stageRank("save_raw_job_lead");
  const extracted = (run?.extracted_count ?? 0) > 0;
  const recruitingSignalCount = Number(run?.run_metadata?.recruiting_signal_count ?? 0);
  const hasRecruitingSignal = recruitingSignalCount > 0;

  return [
    {
      id: "classify",
      title: "识别链接",
      detail: stageRank(stage) > stageRank("classify_source") || run?.source_type ? `已识别为 ${sourceTypeLabel(run?.source_type ?? null)}` : "判断 URL 类型和解析策略",
      icon: Link2,
      state: flowState(stage, ["queued", "normalize_url", "classify_source"], Boolean(run?.source_type), failed, false),
    },
    {
      id: "fetch",
      title: "抓取正文",
      detail: run?.domain ? `${run.domain} · ${run.fetch_layer ?? "自动选择工具"}` : "用 fetcher 获取页面正文",
      icon: FileSearch,
      state: flowState(stage, ["http_article_fetch", "wechat_article_fetch", "js_render_fetch", "crawl4ai_extract", "request_user_visible_page"], savedRaw, failed, blocked && !savedRaw),
    },
    {
      id: "raw",
      title: "保存原文",
      detail: savedRaw ? "raw_job_leads 已保存，可用于恢复重试" : "等待正文保存为 raw lead",
      icon: DatabaseZap,
      state: flowState(stage, ["save_raw_job_lead"], savedRaw, failed && !savedRaw, false),
    },
    {
      id: "extract",
      title: "模型抽取",
      detail: run ? `LLM ${run.llm_call_count} 次，抽出 ${run.extracted_count} 条` : "从正文里抽取公司和岗位",
      icon: Bot,
      state: flowState(stage, ["extract_job_leads", "resume_extract_job_leads"], extracted || hasRecruitingSignal, failed, blocked && savedRaw && !extracted && !hasRecruitingSignal),
    },
    {
      id: "signal",
      title: "公司来源沉淀",
      detail: hasRecruitingSignal ? `已记录 ${recruitingSignalCount} 个公司校招来源` : "没有具体岗位时，先保存公司校招来源",
      icon: Sparkles,
      state: flowState(stage, ["extract_recruiting_signals"], hasRecruitingSignal, failed, false),
    },
    {
      id: "save",
      title: "线索入库",
      detail: extracted ? "岗位线索已进入公司展览" : hasRecruitingSignal ? "等待根据公司信号补全具体岗位" : "等待有效岗位线索",
      icon: CheckCircle2,
      state: extracted || status === "succeeded" ? "done" : failed ? "failed" : blocked ? "blocked" : stage === "save_job_leads" || stage === "completed" ? "running" : "pending",
    },
  ];
}

function flowState(currentStage: string, activeStages: string[], completed: boolean, failed: boolean, blocked: boolean): UrlImportFlowState {
  if (completed) {
    return "done";
  }
  if (failed && activeStages.includes(currentStage)) {
    return "failed";
  }
  if (blocked && activeStages.includes(currentStage)) {
    return "blocked";
  }
  if (activeStages.includes(currentStage)) {
    return "running";
  }
  return stageRank(currentStage) > Math.max(...activeStages.map(stageRank)) ? "done" : "pending";
}

function stageRank(stage: string): number {
  const ranks: Record<string, number> = {
    queued: 0,
    normalize_url: 1,
    classify_source: 2,
    duplicate_url: 2,
    http_article_fetch: 3,
    wechat_article_fetch: 3,
    js_render_fetch: 3,
    crawl4ai_extract: 3,
    request_user_visible_page: 3,
    save_raw_job_lead: 4,
    extract_job_leads: 5,
    resume_extract_job_leads: 5,
    extract_recruiting_signals: 6,
    save_job_leads: 7,
    completed: 8,
  };
  return ranks[stage] ?? 0;
}

function sourceTypeLabel(sourceType: JobSourceType | string | null): string {
  if (!sourceType) {
    return "未识别来源";
  }
  return SOURCE_TYPE_LABELS[sourceType as JobSourceType] ?? sourceType;
}

function getUrlImportRunExplanation(run: ImportUrlRun): { title: string; body: string; tone: "info" | "warning" | "danger" | "success" } {
  if (run.status === "succeeded") {
    return { title: "解析完成", body: `已抽取 ${run.extracted_count} 条岗位线索，并写入线索池。`, tone: "success" };
  }
  if (run.status === "duplicate") {
    return { title: "重复链接", body: "这个 URL 之前已经解析过，本次不会重复抓取和调用模型。", tone: "info" };
  }
  if (run.next_action === "request_manual_paste") {
    return {
      title: "卡在模型抽取阶段",
      body: "系统已经尝试抓取页面并调用模型，但没有抽出有效岗位。常见原因是公众号只返回了摘要、验证码页或正文不完整。请把文章正文复制到下方“手动粘贴兜底”再抽取。",
      tone: "warning",
    };
  }
  if (run.next_action === "enrich_recruiting_signal") {
    const signalCount = Number(run.run_metadata?.recruiting_signal_count ?? 0);
    return {
      title: "已记录公司校招来源",
      body: `这篇文章没有具体岗位 JD，但识别出 ${signalCount || "若干"} 个公司校招来源。下一步会去企业官网或招聘站补全 Java/Agent/AI 相关岗位。`,
      tone: "info",
    };
  }
  if (run.next_action === "request_user_visible_page") {
    return {
      title: "需要用户可见页面",
      body: "这个来源可能需要登录、验证码或浏览器会话，不能后台静默抓取。后续会走 MCP 可见页面边界。",
      tone: "warning",
    };
  }
  if (run.error_message) {
    return { title: "解析失败", body: run.error_message, tone: "danger" };
  }
  if (run.status === "partial") {
    return { title: "部分完成", body: "流程已保存中间结果，但还没有形成可用岗位线索。请按下一步提示继续处理。", tone: "warning" };
  }
  return { title: "正在处理", body: "系统正在按流程抓取、保存和抽取岗位线索。", tone: "info" };
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

function agentStatusLabel(status: string): string {
  if (status === "active") {
    return "运行中";
  }
  if (status === "standby") {
    return "待命";
  }
  if (status === "offline") {
    return "未启动";
  }
  if (status === "disabled") {
    return "已停用";
  }
  return status;
}

function agentHealthLabel(health: AgentRuntimeHealth): string {
  if (health.status === "healthy") {
    return health.label || "已连接";
  }
  if (health.status === "unreachable") {
    return health.label || "未启动或连接失败";
  }
  return health.label || "未检测";
}

function riskLevelLabel(riskLevel: string): string {
  if (riskLevel === "low") {
    return "低风险";
  }
  if (riskLevel === "medium") {
    return "中风险";
  }
  if (riskLevel === "high") {
    return "高风险";
  }
  return riskLevel;
}

function formatCompactList(values: string[], fallback: string): string {
  const cleanValues = values.map((value) => value.trim()).filter(Boolean);
  if (!cleanValues.length) {
    return fallback;
  }
  return cleanValues.slice(0, 4).join("、");
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
  agents: "Agent 面板",
  skills: "Skill 管理",
  dashboard: "秋招发现总览",
  sources: "岗位信息源管理",
  jobs: "公司展览",
  leads: "线索导入",
  pipeline: "投递进度面板",
  guardrails: "自动化边界",
};

const PAGE_DESCRIPTIONS: Record<PageId, string> = {
  chat: "已接入 Agent 会话 API，先验证记忆、历史加载和手动压缩链路。",
  agents: "查看主 agent 当前接入了哪些子 agent，以及每个 agent 声明的能力、风险和执行边界。",
  skills: "管理内容源解析 Skill，展示依赖工具和 Agent 可加载能力。",
  dashboard: "先广撒网收集来源，再让线索进入验证与用户确认流程。",
  sources: "管理高校就业网、企业官网、公众号、小红书和可见招聘页来源。",
  jobs: "临时接入 OfferIO 结构化接口，先按公司展示，再查看岗位，投递前仍做官网验证。",
  leads: "只负责候选文章、临时链接和手动文本导入；字段不完整的文章/社媒信号暂不作为公司展示。",
  pipeline: "岗位申请阶段可人工修改，后续 agent 也会通过同一 API 更新进度。",
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

const APPLICATION_STAGES: Array<{ status: ApplicationStatus; label: string }> = [
  { status: "evaluating", label: "待投递" },
  { status: "preparing", label: "材料准备" },
  { status: "applied", label: "已投未回" },
  { status: "assessment", label: "测评/AI面试" },
  { status: "written_test", label: "笔试" },
  { status: "interview_1", label: "一面" },
  { status: "interview_2", label: "二面" },
  { status: "hr_interview", label: "三面" },
  { status: "offer", label: "已拿Offer" },
  { status: "rejected", label: "已被拒" },
  { status: "withdrawn", label: "已放弃" },
];

const APPLICATION_STATUS_LABELS: Record<ApplicationStatus, string> = Object.fromEntries(APPLICATION_STAGES.map((stage) => [stage.status, stage.label])) as Record<
  ApplicationStatus,
  string
>;

const ARTICLE_STATUS_LABELS: Record<string, string> = {
  pending: "待解析",
  parsed: "已解析",
  skipped: "已跳过",
  needs_visible_page: "需可见页面",
};

const URL_IMPORT_STATUS_LABELS: Record<string, string> = {
  running: "解析中",
  waiting_user: "等待用户确认",
  succeeded: "解析成功",
  partial: "部分完成",
  failed_recoverable: "可恢复失败",
  failed_terminal: "终止失败",
  duplicate: "重复链接",
};

const URL_IMPORT_STAGE_LABELS: Record<string, string> = {
  queued: "已创建任务，等待后台解析",
  normalize_url: "正在归一化 URL",
  classify_source: "正在识别来源类型",
  http_article_fetch: "正在抓取公开网页正文",
  wechat_article_fetch: "正在解析公众号文章",
  js_render_fetch: "正在渲染公开 JS 页面",
  crawl4ai_extract: "正在增强正文抽取",
  save_raw_job_lead: "正在保存原始线索",
  extract_job_leads: "正在用模型抽取岗位线索",
  save_job_leads: "正在保存岗位线索",
  completed: "解析完成",
  duplicate_url: "发现重复链接",
  request_user_visible_page: "需要用户可见页面",
  resume_extract_job_leads: "正在从 raw checkpoint 恢复抽取",
};

const DOMAIN_HEALTH_LABELS: Record<string, string> = {
  unknown: "未检测",
  closed: "正常",
  open: "熔断",
  half_open: "半开探测",
};

const NEXT_ACTION_LABELS: Record<string, string> = {
  continue_workflow: "继续 workflow",
  retry_same_stage: "重试当前阶段",
  retry_with_next_fetcher: "切换下一个 fetcher",
  wait_for_cooldown: "等待熔断冷却",
  request_user_visible_page: "打开用户可见页面确认",
  request_manual_paste: "手动粘贴正文兜底",
  skip_duplicate: "跳过重复链接",
  stop_terminal_failure: "停止本次解析",
  enrich_recruiting_signal: "根据公司来源补全企业官网岗位",
};

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
