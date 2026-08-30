# Long-Term Memory Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first deterministic long-term memory loop for OfferMaster: extract evidence-backed memory candidates, score and deduplicate them, promote safe candidates into `agent_memories`, and recall relevant active memories into runtime context.

**Architecture:** Keep extraction and scoring deterministic so the first version is reproducible and safe. Reuse `AgentLearningCandidate` as the review queue, add a small promotion service for approved or low-risk candidates, and make context recall an explicit policy with scope and token limits. Business facts remain in business tables; only reusable preferences, recovery lessons, and workflow rules enter long-term memory.

**Tech Stack:** Python 3.11+, SQLAlchemy 2.x, Pydantic v2, FastAPI, `unittest`, existing `AgentMemoryRepository`, `AgentLearningService`, and `MemoryContextBuilder`.

**Spec:** `C:\Users\phoenix\Documents\Obsidian Vault\秋招助手\开发\开发目标\记忆模块\记忆改进\2026-08-29-OfferMaster记忆系统OpenClaw式改进开发方案.md`

## Global Constraints

- Do not use long-term memory as the authoritative store for job, company, verification, or application status.
- Every promoted memory must retain at least one source evidence id in `metadata_json`.
- High-risk candidates require explicit approval and must never be auto-promoted.
- Candidate extraction must be deterministic and must not call external tools.
- Deduplication must merge repeated evidence into one active memory instead of creating duplicate rows.
- Runtime recall must be scope-filtered, bounded by count and characters, and must not load archived memories.
- Preserve the existing pending-review and protected/pinned Skill approval boundaries.

### Task 1: Define Evidence-Backed Memory Candidate Extraction

**Files:**
- Create: `apps/api/app/agent_runtime/memory/memory_candidate_extractor.py`
- Test: `tests/test_memory_candidate_extractor.py`

**Interfaces:**
- Consumes: `AgentMessage` rows and `ToolCallLog` rows associated with one session/workflow.
- Produces: `MemoryCandidateDraft` values with `memory_type`, `scope`, `title`, `content`, `importance`, `risk_level`, `lesson_type`, `evidence_ids`, and `metadata`.

- [ ] **Step 1: Write the failing test**

```python
def test_extracts_explicit_application_confirmation_boundary_from_user_message():
    message = AgentMessage(
        id="message-1",
        role=AgentMessageRole.USER,
        message_kind=AgentMessageKind.USER_TEXT,
        content_text="投递前一定要让我确认，不要自动提交。",
        visible_content_text="投递前一定要让我确认，不要自动提交。",
    )

    drafts = extract_memory_candidates(messages=[message], tool_logs=[])

    assert len(drafts) == 1
    assert drafts[0].memory_type == "user_preference"
    assert drafts[0].scope == "application_submission"
    assert drafts[0].importance >= 90
    assert drafts[0].risk_level == AgentLearningCandidateRiskLevel.HIGH
    assert drafts[0].evidence_ids == ("message-1",)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_memory_candidate_extractor -v`

Expected: FAIL with `ModuleNotFoundError` because `memory_candidate_extractor.py` does not exist.

- [ ] **Step 3: Write minimal implementation**

Implement:

```python
@dataclass(frozen=True)
class MemoryCandidateDraft:
    memory_type: str
    scope: str
    title: str
    content: str
    importance: int
    risk_level: AgentLearningCandidateRiskLevel
    lesson_type: AgentLearningCandidateLessonType
    evidence_ids: tuple[str, ...]
    metadata: dict[str, Any]

def extract_memory_candidates(
    *,
    messages: Sequence[AgentMessage],
    tool_logs: Sequence[ToolCallLog],
) -> list[MemoryCandidateDraft]:
    ...
```

Recognize only evidence-backed deterministic patterns in the first slice:

- User messages containing application confirmation boundaries produce `user_preference/application_submission`.
- User messages containing explicit “do not invent / leave blank when unknown” rules produce `user_preference/data_integrity`.
- Failed and later successful tool logs with `recovery_path`, `verified`, or positive extraction output produce `tool_recovery/<tool_group>`.
- Ignore assistant guesses, empty messages, transient timeout-only failures, and business facts without a reusable rule.

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_memory_candidate_extractor -v`

Expected: PASS, including tests for confirmation boundaries, data-integrity preferences, recovered tool failures, timeout exclusion, and secret redaction.

- [ ] **Step 5: Commit**

```powershell
git add tests/test_memory_candidate_extractor.py apps/api/app/agent_runtime/memory/memory_candidate_extractor.py
git commit -m "feat: extract evidence-backed memory candidates"
```

### Task 2: Add Scoring, Normalization, and Deduplication

**Files:**
- Create: `apps/api/app/agent_runtime/memory/memory_scorer.py`
- Create: `apps/api/app/agent_runtime/memory/memory_deduplicator.py`
- Test: `tests/test_memory_consolidation_rules.py`

**Interfaces:**
- Consumes: `MemoryCandidateDraft` values and active `AgentMemory` rows.
- Produces: `ScoredMemoryCandidate` values and merge decisions with normalized keys.

- [ ] **Step 1: Write the failing test**

```python
def test_same_preference_is_deduplicated_and_evidence_is_merged():
    first = _draft("message-1", "投递前必须用户确认")
    second = _draft("message-2", "投递前必须用户确认")

    result = deduplicate_memory_candidates([first, second], existing_memories=[])

    assert len(result) == 1
    assert result[0].evidence_ids == ("message-1", "message-2")
    assert result[0].normalized_key == "user_preference:application_submission:投递前必须用户确认"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_memory_consolidation_rules -v`

Expected: FAIL because the scorer and deduplicator modules do not exist.

- [ ] **Step 3: Write minimal implementation**

Implement:

```python
@dataclass(frozen=True)
class ScoredMemoryCandidate:
    draft: MemoryCandidateDraft
    score: int
    normalized_key: str
    auto_promotable: bool

def score_memory_candidate(draft: MemoryCandidateDraft) -> ScoredMemoryCandidate:
    ...

def deduplicate_memory_candidates(
    candidates: Sequence[MemoryCandidateDraft],
    *,
    existing_memories: Sequence[AgentMemory],
) -> list[ScoredMemoryCandidate]:
    ...
```

Scoring uses fixed points: explicit user statement `+40`, reusable across tasks `+25`, repeated or recovered evidence `+20`, clear structured evidence `+10`, high-risk penalty `-15`. Clamp to `0..100`. Auto-promotion requires score `>= 80`, low risk, and at least one evidence id. Normalize whitespace and case for Latin text while preserving Chinese content.

Merge candidates with the same normalized key. When an existing active memory has the key, return a merge decision carrying the existing memory id and unioned evidence ids; do not create a second active memory.

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_memory_consolidation_rules -v`

Expected: PASS for score thresholds, high-risk blocking, repeated evidence merge, and existing-memory duplicate detection.

- [ ] **Step 5: Commit**

```powershell
git add tests/test_memory_consolidation_rules.py apps/api/app/agent_runtime/memory/memory_scorer.py apps/api/app/agent_runtime/memory/memory_deduplicator.py
git commit -m "feat: score and deduplicate memory candidates"
```

### Task 3: Persist Candidates and Promote Safe Memories

**Files:**
- Modify: `apps/api/app/domains/agent_memory/repository.py`
- Modify: `apps/api/app/domains/agent_memory/service.py`
- Create: `apps/api/app/agent_runtime/memory/consolidation.py`
- Create: `apps/api/app/agent_runtime/memory/memory_promotion.py`
- Test: `tests/test_memory_consolidation_service.py`

**Interfaces:**
- Consumes: session/workflow ids, extracted drafts, `AgentMemoryRepository`, and existing `AgentLearningService`.
- Produces: `MemoryConsolidationResult` containing created candidate ids, promoted memory ids, merged memory ids, and skipped reasons.

- [ ] **Step 1: Write the failing test**

```python
def test_low_risk_high_score_candidate_is_promoted_and_high_risk_stays_pending():
    result = MemoryConsolidationService(
        session=session,
        learning_service=AgentLearningService(AgentMemoryRepository(session)),
    ).consolidate(
        MemoryConsolidationCommand(
            session_id="session-1",
            workflow_run_id=workflow_run.id,
            agent_run_id="agent-run-1",
            target_scope="job_discovery",
        )
    )

    assert result.promoted_memory_ids
    assert result.created_candidate_ids
    high_risk = session.get(AgentLearningCandidate, result.created_candidate_ids[-1])
    assert high_risk.status == AgentLearningCandidateStatus.PENDING_REVIEW
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_memory_consolidation_service -v`

Expected: FAIL because `MemoryConsolidationService` and promotion methods do not exist.

- [ ] **Step 3: Write minimal implementation**

Add repository methods:

```python
def list_memories(self, *, scope: str | None = None, status: AgentMemoryStatus | None = None, limit: int = 100) -> list[AgentMemory]:
    ...

def find_active_memory_by_key(self, normalized_key: str) -> AgentMemory | None:
    ...

def update_memory(self, memory: AgentMemory) -> AgentMemory:
    ...
```

Add promotion behavior:

- Convert each draft into an `AgentLearningCandidateCreate` with source ids and evidence JSON.
- Persist high-risk or low-score candidates as `PENDING_REVIEW`.
- Auto-promote only low-risk, score `>= 80`, and evidence-backed candidates.
- Store promoted memory metadata with `normalized_key`, `evidence_ids`, `score`, `promotion_mode`, `source_candidate_id`, and `last_observed_at`.
- On duplicate, update the existing active memory content only when the new candidate has stronger evidence; always union evidence ids and update `updated_at`.
- Never promote to a protected or pinned Skill in this task.

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_memory_consolidation_service tests.test_agent_learning_candidates -v`

Expected: PASS with pending review preserved for high-risk candidates and no duplicate active memories.

- [ ] **Step 5: Commit**

```powershell
git add tests/test_memory_consolidation_service.py apps/api/app/domains/agent_memory/repository.py apps/api/app/domains/agent_memory/service.py apps/api/app/agent_runtime/memory/consolidation.py apps/api/app/agent_runtime/memory/memory_promotion.py
git commit -m "feat: persist and promote long-term memories"
```

### Task 4: Add Explicit Long-Term Memory Recall Policy

**Files:**
- Create: `apps/api/app/agent_runtime/memory/recall_policy.py`
- Modify: `apps/api/app/agent_runtime/memory/context_builder.py`
- Test: `tests/test_agent_context_builder.py`
- Test: `tests/test_memory_recall_policy.py`

**Interfaces:**
- Consumes: current user query, active `AgentMemory` rows, and context limits.
- Produces: bounded memory system messages, loaded memory ids, and recall metadata.

- [ ] **Step 1: Write the failing test**

```python
def test_context_builder_recalls_relevant_active_memory_but_not_archived_memory():
    active = AgentMemory(
        memory_type="user_preference",
        scope="application_submission",
        title="投递前必须用户确认",
        content="任何岗位最终提交前都必须等待用户确认。",
        status=AgentMemoryStatus.ACTIVE,
        importance=95,
        metadata_json={"normalized_key": "user_preference:application_submission:投递前必须用户确认"},
    )

    built = MemoryContextBuilder(service).build(
        conversation.id,
        new_user_message="帮我投递腾讯的 Java 岗位",
        config=ContextBuildConfig(max_recent_messages=10, max_loaded_memories=3),
    )

    assert active.id in built.loaded_memory_ids
    assert any("最终提交前都必须等待用户确认" in message["content"] for message in built.llm_messages)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_memory_recall_policy tests.test_agent_context_builder.AgentContextBuilderTest.test_context_builder_recalls_relevant_active_memory_but_not_archived_memory -v`

Expected: FAIL because `max_loaded_memories` and recall policy integration do not exist.

- [ ] **Step 3: Write minimal implementation**

Extend `ContextBuildConfig` with `max_loaded_memories` and `max_memory_context_chars`. Implement:

```python
def recall_relevant_memories(
    session: Session,
    *,
    query: str,
    limit: int,
    max_chars: int,
) -> MemoryRecallResult:
    ...
```

Use deterministic keyword matching over `title`, `content`, `scope`, and `memory_type`; rank by match count and `importance`; filter to `ACTIVE`; truncate the combined context by `max_chars`. Insert one system message with source metadata and include recalled ids in `BuiltContext.loaded_memory_ids` and `context_metadata`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_memory_recall_policy tests.test_agent_context_builder -v`

Expected: PASS and existing skill/context tests remain green.

- [ ] **Step 5: Commit**

```powershell
git add tests/test_memory_recall_policy.py tests/test_agent_context_builder.py apps/api/app/agent_runtime/memory/recall_policy.py apps/api/app/agent_runtime/memory/context_builder.py
git commit -m "feat: recall scoped long-term memories in context"
```

### Task 5: Expose Consolidation as a Safe Internal/API Operation and Document Verification

**Files:**
- Create: `apps/api/app/api/v1/agent_memory_consolidation.py`
- Modify: `apps/api/app/main.py`
- Create: `tests/test_agent_memory_consolidation_api.py`
- Create: `C:\Users\phoenix\Documents\Obsidian Vault\秋招助手\开发\开发目标\记忆模块\记忆改进\开发记录\2026-08-29-长期记忆沉淀第二部分开发记录.md`

**Interfaces:**
- Consumes: a session/workflow id and optional target scope.
- Produces: JSON status with candidate, promotion, merge, and skip counts.

- [ ] **Step 1: Write the failing test**

```python
def test_consolidation_endpoint_returns_review_and_promotion_counts():
    response = client.post(
        "/api/v1/agent-memory/consolidate",
        json={"session_id": session.id, "workflow_run_id": workflow_run.id, "target_scope": "job_discovery"},
    )

    assert response.status_code == 200
    assert set(response.json()) >= {"created_candidate_count", "promoted_memory_count", "merged_memory_count"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_agent_memory_consolidation_api -v`

Expected: FAIL with HTTP 404 because the endpoint is not registered.

- [ ] **Step 3: Write minimal implementation**

Add a route that validates the session and workflow run, executes the deterministic consolidation service once, commits on success, and returns counts and ids. Do not invoke it automatically from arbitrary user chat in this task; it is an explicit internal operation until the pre-compaction flush is implemented.

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_agent_memory_consolidation_api tests.test_agent_learning_candidates tests.test_agent_memory_tools -v`

Expected: PASS.

- [ ] **Step 5: Update development record and verify the repository**

Write the implementation record with:

- The unfinished item recovered from thread `019ff91d-e382-7712-98e9-c323c4e68bdd`.
- Files changed and the data flow from evidence to candidate to memory to recall.
- Explicit high-risk approval boundary and duplicate handling.
- Exact test commands and results.

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_memory_candidate_extractor tests.test_memory_consolidation_rules tests.test_memory_consolidation_service tests.test_memory_recall_policy tests.test_agent_memory_consolidation_api tests.test_agent_learning_candidates tests.test_agent_memory_tools tests.test_agent_context_builder -v
git diff --check
```

Expected: all listed tests pass and `git diff --check` reports no whitespace errors.
