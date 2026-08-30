from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
import re
import shutil

from app.domains.agent_memory.models import (
    AgentSkill,
    AgentSkillStatus,
    AgentSkillStorageType,
    AgentSkillUsage,
    AgentSkillUsageEvent,
    utc_now,
)
from app.domains.agent_memory.repository import AgentMemoryRepository
from app.domains.agent_memory.schemas import AgentSkillCreate
from app.agent_runtime.memory.skill_package_parser import SkillPackageParser


@dataclass(frozen=True)
class SkillDocument:
    skill: AgentSkill
    content: str
    version_hash: str


@dataclass(frozen=True)
class SkillPatchResult:
    skill: AgentSkill
    content: str
    previous_version_hash: str
    applied_version_hash: str


class AgentSkillRepository:
    def __init__(self, repository: AgentMemoryRepository, *, skill_root: Path | None = None) -> None:
        self._repository = repository
        self._skill_root = skill_root or _default_skill_root()

    def create_skill(self, draft: AgentSkillCreate) -> AgentSkill:
        existing = self._repository.get_skill_by_name(draft.name)
        if existing is not None:
            raise ValueError(f"Agent skill name already exists: {draft.name}")

        skill = self._repository.add_skill(
            AgentSkill(
                name=_normalize_skill_name(draft.name),
                title=" ".join(draft.title.split()),
                description=" ".join(draft.description.split()),
                category=" ".join(draft.category.split()),
                storage_type=AgentSkillStorageType.MARKDOWN_FILE,
                status=AgentSkillStatus.ACTIVE,
                protected=draft.protected,
                pinned=draft.pinned,
                created_by=draft.created_by,
                metadata_json=draft.metadata_json,
            )
        )
        skill_dir = self._skill_root / skill.id
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(_build_skill_markdown(draft), encoding="utf-8")
        skill.file_path = str(skill_file)
        self._ensure_usage(skill.id)
        self._repository.flush()
        return skill

    def import_skill_from_path(
        self,
        source_path: str | Path,
        *,
        category: str = "content_source",
        protected: bool = False,
        pinned: bool = False,
        created_by: str = "developer",
        metadata_json: dict | None = None,
    ) -> AgentSkill:
        package = SkillPackageParser().parse(source_path)
        blocking_errors = package.import_report.get("blocking_errors") or []
        if blocking_errors:
            raise ValueError("; ".join(str(error) for error in blocking_errors))

        skill_file = package.skill_file
        name = package.name
        if self._repository.get_skill_by_name(name) is not None:
            raise ValueError(f"Agent skill name already exists: {name}")

        merged_metadata = {
            **(metadata_json or {}),
            **package.import_report,
        }
        skill = self._repository.add_skill(
            AgentSkill(
                name=name,
                title=" ".join(package.title.split()),
                description=" ".join(package.description.split()),
                category=" ".join(category.split()),
                storage_type=AgentSkillStorageType.MARKDOWN_FILE,
                status=AgentSkillStatus.ACTIVE,
                protected=protected,
                pinned=pinned,
                created_by=created_by,
                metadata_json=merged_metadata,
            )
        )
        skill_dir = self._skill_root / skill.id
        skill_dir.mkdir(parents=True, exist_ok=True)
        imported_file = skill_dir / "SKILL.md"
        shutil.copyfile(skill_file, imported_file)
        skill.file_path = str(imported_file)
        self._ensure_usage(skill.id)
        self._repository.flush()
        return skill

    def list_skills(
        self,
        *,
        status: AgentSkillStatus | None = None,
        limit: int = 100,
    ) -> list[AgentSkill]:
        return self._repository.list_skills(status=status, limit=limit)

    def ensure_builtin_content_source_skills(self) -> list[AgentSkill]:
        return self._ensure_builtin_skill_paths(_builtin_content_source_skill_paths())

    def ensure_builtin_skills(self) -> list[AgentSkill]:
        return self._ensure_builtin_skill_paths(
            [
                *_builtin_content_source_skill_paths(),
                *_builtin_database_skill_paths(),
            ]
        )

    def _ensure_builtin_skill_paths(self, source_paths: list[Path]) -> list[AgentSkill]:
        imported: list[AgentSkill] = []
        parser = SkillPackageParser()
        for source_path in source_paths:
            if not source_path.is_dir():
                continue
            package = parser.parse(source_path)
            if self._repository.get_skill_by_name(package.name) is not None:
                continue
            imported.append(
                self.import_skill_from_path(
                    source_path,
                    category="content_source",
                    protected=True,
                    pinned=True,
                    created_by="system_bootstrap",
                    metadata_json={"builtin": True},
                )
            )
        return imported

    def get_skill(self, skill_id: str) -> AgentSkill:
        return self._require_skill(skill_id)

    def read_skill(self, skill_id: str, *, record_view: bool = False) -> SkillDocument:
        skill = self._require_skill(skill_id)
        content = self._read_skill_file(skill)
        if record_view:
            self.record_usage(skill.id, AgentSkillUsageEvent.VIEW)
        return SkillDocument(skill=skill, content=content, version_hash=_hash_text(content))

    def append_section(
        self,
        skill_id: str,
        *,
        heading: str,
        body: str,
        actor: str,
    ) -> SkillPatchResult:
        document = self.read_skill(skill_id)
        skill = document.skill
        if actor == "agent_review" and (skill.protected or skill.pinned):
            raise PermissionError(f"Agent review cannot patch protected or pinned skill: {skill.id}")

        patched = _append_under_heading(document.content, heading=heading, body=body)
        self._write_skill_file(skill, patched)
        skill.updated_at = utc_now()
        skill.metadata_json = {**(skill.metadata_json or {}), "last_patch_actor": actor}
        self.record_usage(skill.id, AgentSkillUsageEvent.PATCH)
        self._repository.flush()
        return SkillPatchResult(
            skill=skill,
            content=patched,
            previous_version_hash=document.version_hash,
            applied_version_hash=_hash_text(patched),
        )

    def record_usage(self, skill_id: str, event: AgentSkillUsageEvent) -> AgentSkillUsage:
        usage = self._ensure_usage(skill_id)
        now = utc_now()
        if event == AgentSkillUsageEvent.USE:
            usage.use_count += 1
            usage.last_used_at = now
        elif event == AgentSkillUsageEvent.VIEW:
            usage.view_count += 1
            usage.last_viewed_at = now
        elif event == AgentSkillUsageEvent.PATCH:
            usage.patch_count += 1
            usage.last_patched_at = now
        elif event == AgentSkillUsageEvent.SUCCESS:
            usage.success_count += 1
            usage.last_success_at = now
        elif event == AgentSkillUsageEvent.FAILURE:
            usage.failure_count += 1
            usage.last_failure_at = now
        self._repository.flush()
        return usage

    def record_runtime_event(
        self,
        skill_id: str,
        *,
        event: str,
        evidence: dict[str, Any] | None = None,
    ) -> AgentSkillUsage:
        self._require_skill(skill_id)
        usage = self._ensure_usage(skill_id)
        now = utc_now()
        evidence = evidence or {}
        entry = {
            "event": event,
            "recorded_at": now.isoformat(),
            **_json_safe_runtime_evidence(evidence),
        }

        metadata = _merge_runtime_metadata(
            self._repository.get_usage_metadata(usage.id),
        )
        events = [*metadata["runtime_events"], entry][-50:]
        counts = _runtime_event_counts(events)

        metadata["runtime_events"] = events
        metadata["runtime_event_counts"] = counts
        metadata["last_runtime_event"] = entry
        return self._repository.update_usage_runtime_event(
            usage.id,
            metadata_json=metadata,
            recorded_at=now,
            success_delta=1 if event == "tool_succeeded" else 0,
            failure_delta=1 if event == "tool_failed" else 0,
        )

    def get_usage(self, skill_id: str) -> AgentSkillUsage:
        self._require_skill(skill_id)
        return self._ensure_usage(skill_id)

    def pin_skill(self, skill_id: str) -> AgentSkill:
        skill = self._require_skill(skill_id)
        skill.pinned = True
        skill.updated_at = utc_now()
        self._repository.flush()
        return skill

    def archive_skill(self, skill_id: str) -> AgentSkill:
        skill = self._require_skill(skill_id)
        skill.status = AgentSkillStatus.ARCHIVED
        skill.updated_at = utc_now()
        usage = self._ensure_usage(skill.id)
        usage.state = AgentSkillStatus.ARCHIVED
        usage.archived_at = utc_now()
        self._repository.flush()
        return skill

    def _require_skill(self, skill_id: str) -> AgentSkill:
        skill = self._repository.get_skill(skill_id)
        if skill is None:
            raise ValueError(f"Agent skill not found: {skill_id}")
        return skill

    def _ensure_usage(self, skill_id: str) -> AgentSkillUsage:
        usage = self._repository.get_usage(skill_id)
        if usage is not None:
            return usage
        return self._repository.add_usage(AgentSkillUsage(skill_id=skill_id, state=AgentSkillStatus.ACTIVE))

    @staticmethod
    def _read_skill_file(skill: AgentSkill) -> str:
        if not skill.file_path:
            raise ValueError(f"Agent skill has no markdown file path: {skill.id}")
        path = Path(skill.file_path)
        if not path.is_file():
            raise ValueError(f"Agent skill markdown file not found: {skill.id}")
        return path.read_text(encoding="utf-8")

    @staticmethod
    def _write_skill_file(skill: AgentSkill, content: str) -> None:
        if not skill.file_path:
            raise ValueError(f"Agent skill has no markdown file path: {skill.id}")
        path = Path(skill.file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _default_skill_root() -> Path:
    return Path(__file__).resolve().parents[5] / "docs" / "agent-skills"


def _builtin_content_source_skill_paths() -> list[Path]:
    root = _default_skill_root() / "vendor-content-sources"
    return [
        root / "wechat-article-content-fetch",
        root / "xiaohongshu-content-fetch",
    ]


def _builtin_database_skill_paths() -> list[Path]:
    return [_default_skill_root() / "builtin-database" / "database-operations"]


def _resolve_skill_file(source_path: Path) -> Path:
    path = source_path.expanduser().resolve()
    if path.is_dir():
        path = path / "SKILL.md"
    if path.name != "SKILL.md" or not path.is_file():
        raise ValueError("Skill import path must be a SKILL.md file or a directory containing SKILL.md")
    return path


def _parse_frontmatter(content: str) -> tuple[dict[str, object], str]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, content
    frontmatter_lines: list[str] = []
    end_index = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_index = index
            break
        frontmatter_lines.append(line)
    if end_index is None:
        return {}, content
    metadata: dict[str, object] = {}
    for line in frontmatter_lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        cleaned_key = key.strip()
        cleaned_value = value.strip()
        if cleaned_key:
            metadata[cleaned_key] = _parse_metadata_value(cleaned_value)
    return metadata, "\n".join(lines[end_index + 1 :])


def _parse_metadata_value(value: str) -> object:
    if value.startswith("[") and value.endswith("]"):
        return [item.strip().strip('"\'') for item in value[1:-1].split(",") if item.strip()]
    return value.strip('"\'')


def _extract_title(body: str) -> str | None:
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip() or None
    return None


def _first_non_heading_line(body: str) -> str | None:
    for line in body.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    return None


def _title_from_name(name: str) -> str:
    return " ".join(part.capitalize() for part in name.replace("_", "-").split("-") if part)


def _string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _infer_source_types(content: str) -> list[str]:
    lowered = content.lower()
    candidates: list[str] = []
    if "xiaohongshu" in lowered or "小红书" in content:
        candidates.append("xiaohongshu_note")
    if "wechat" in lowered or "weixin" in lowered or "公众号" in content or "微信" in content:
        candidates.extend(["wechat_article", "wechat_account"])
    if "douyin" in lowered or "抖音" in content:
        candidates.append("douyin_video")
    return list(dict.fromkeys(candidates))


def _infer_required_tools(content: str) -> list[str]:
    lowered = content.lower()
    candidates: list[str] = []
    for tool_name in [
        "xiaohongshu-mcp",
        "weixin-articles-mcp",
        "douyin-video-analysis",
        "ffmpeg",
        "ocr",
        "asr",
        "mcp_visible_page",
    ]:
        if tool_name in lowered:
            candidates.append(tool_name)
    return candidates


def _availability_state(required_tools: list[str]) -> str:
    return "unavailable" if required_tools else "available"


def _normalize_skill_name(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", name.strip()).strip("-").lower()
    if not normalized:
        raise ValueError("Agent skill name is required")
    return normalized


def _build_skill_markdown(draft: AgentSkillCreate) -> str:
    sections = draft.sections or {}
    return "\n".join(
        [
            f"# {draft.title}",
            "",
            f"描述：{draft.description}",
            f"分类：{draft.category}",
            "",
            "## 何时使用",
            sections.get("when_to_use", "待补充。"),
            "",
            "## 输入",
            sections.get("inputs", "待补充。"),
            "",
            "## 输出",
            sections.get("outputs", "待补充。"),
            "",
            "## 标准流程",
            sections.get("workflow", "待补充。"),
            "",
            "## 工具边界",
            sections.get("tool_boundaries", "待补充。"),
            "",
            "## 用户确认点",
            sections.get("confirmation_points", "待补充。"),
            "",
            "## 错误处理",
            sections.get("error_handling", "待补充。"),
            "",
            "## 验证方式",
            sections.get("verification", "待补充。"),
            "",
            "## 关联参考文件",
            sections.get("references", "待补充。"),
            "",
            "## 历史经验",
            "暂无。",
            "",
        ]
    )


def _append_under_heading(content: str, *, heading: str, body: str) -> str:
    normalized_heading = heading.strip().lstrip("#").strip()
    entry = f"\n\n### {utc_now().isoformat()}\n\n{body.strip()}\n"
    marker = f"## {normalized_heading}"
    if marker not in content:
        return f"{content.rstrip()}\n\n{marker}{entry}"
    return f"{content.rstrip()}{entry}"


def _hash_text(content: str) -> str:
    return sha256(content.encode("utf-8")).hexdigest()


def _json_safe_runtime_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    blocked_keys = {"cookie", "token", "authorization", "api_key", "apikey", "password", "secret"}
    safe: dict[str, Any] = {}
    for key, value in evidence.items():
        key_text = str(key)
        if key_text.lower() in blocked_keys:
            continue
        safe[key_text] = _json_safe_value(value, blocked_keys=blocked_keys)
    return safe


def _merge_runtime_metadata(*metadata_items: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    events: list[dict[str, Any]] = []
    for metadata in metadata_items:
        if not isinstance(metadata, dict):
            continue
        merged.update(metadata)
        raw_events = metadata.get("runtime_events")
        if isinstance(raw_events, list):
            events.extend(event for event in raw_events if isinstance(event, dict))
    merged["runtime_events"] = _dedupe_runtime_events(events)[-50:]
    merged["runtime_event_counts"] = _runtime_event_counts(merged["runtime_events"])
    if merged["runtime_events"]:
        merged["last_runtime_event"] = merged["runtime_events"][-1]
    return merged


def _dedupe_runtime_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str]] = set()
    result: list[dict[str, Any]] = []
    for event in events:
        key = (
            str(event.get("event") or ""),
            str(event.get("recorded_at") or ""),
            str(event.get("approval_request_id") or ""),
            str(event.get("tool_call_log_id") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(event)
    return result


def _runtime_event_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        event_name = str(event.get("event") or "").strip()
        if not event_name:
            continue
        counts[event_name] = counts.get(event_name, 0) + 1
    return counts


def _json_safe_value(value: Any, *, blocked_keys: set[str]) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _json_safe_value(item, blocked_keys=blocked_keys)
            for key, item in value.items()
            if str(key).lower() not in blocked_keys
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_value(item, blocked_keys=blocked_keys) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
