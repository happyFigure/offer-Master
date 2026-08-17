from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
from typing import Any


GENERIC_DESCRIPTION_PATTERNS = {
    "处理网页",
    "分析内容",
    "自动化",
    "爬取数据",
    "网页处理",
    "内容分析",
    "web helper",
}

HIGH_RISK_TERMS = (
    "submit",
    "apply",
    "delete",
    "login",
    "payment",
    "secret",
    "token",
    "password",
    "投递",
    "提交",
    "删除",
    "登录",
    "付款",
    "密钥",
    "密码",
)


@dataclass(frozen=True)
class SkillPackage:
    skill_file: Path
    package_root: Path
    content: str
    frontmatter: dict[str, Any]
    body: str
    name: str
    title: str
    description: str
    import_report: dict[str, Any]


class SkillPackageParser:
    def parse(self, source_path: str | Path) -> SkillPackage:
        skill_file = self.resolve_skill_file(source_path)
        package_root = skill_file.parent
        content = skill_file.read_text(encoding="utf-8")
        frontmatter, body = parse_frontmatter(content)

        name = normalize_skill_name(str(frontmatter.get("name") or package_root.name or skill_file.stem))
        title = extract_title(body) or title_from_name(name)
        raw_description = str(frontmatter.get("description") or "").strip()
        description = raw_description or first_non_heading_line(body) or title

        source_types = string_list(frontmatter_value(frontmatter, "source_types", "source_type")) or infer_source_types(content)
        required_tools = string_list(frontmatter_value(frontmatter, "required_tools", "tools")) or infer_required_tools(content)
        allowed_tools = string_list(frontmatter_value(frontmatter, "allowed-tools", "allowed_tools"))
        ask_tools = string_list(frontmatter_value(frontmatter, "ask-tools", "ask_tools"))
        disallowed_tools = string_list(frontmatter_value(frontmatter, "disallowed-tools", "disallowed_tools"))
        compatibility = string_list(frontmatter.get("compatibility"))
        resources = collect_resources(package_root)
        description_quality = evaluate_description_quality(raw_description)
        blocking_errors = blocking_errors_for(raw_description)
        import_warnings = warnings_for(raw_description, description_quality)
        security_risk_level = security_risk(frontmatter, allowed_tools, disallowed_tools, required_tools)
        auto_trigger_state = auto_trigger_state_for(description_quality, blocking_errors, security_risk_level)

        report: dict[str, Any] = {
            "import_source_path": str(skill_file),
            "version_hash": package_hash(package_root),
            "source_types": source_types,
            "required_tools": required_tools,
            "allowed_tools": allowed_tools,
            "ask_tools": ask_tools,
            "disallowed_tools": disallowed_tools,
            "disable_model_invocation": bool_value(
                frontmatter_value(frontmatter, "disable-model-invocation", "disable_model_invocation")
            ),
            "compatibility": compatibility,
            "license": str(frontmatter.get("license") or "").strip(),
            "resources": resources,
            "openai_agent_metadata": load_openai_agent_metadata(resources),
            "import_warnings": import_warnings,
            "blocking_errors": blocking_errors,
            "description_quality_score": description_quality,
            "auto_trigger_state": auto_trigger_state,
            "security_risk_level": security_risk_level,
            "availability_state": availability_state(required_tools),
            "permission_notice": "allowed_tools 只是 Skill 申请权限，最终执行仍必须经过 ToolRuntimeGuard 和用户确认边界。",
        }

        return SkillPackage(
            skill_file=skill_file,
            package_root=package_root,
            content=content,
            frontmatter=frontmatter,
            body=body,
            name=name,
            title=title,
            description=" ".join(description.split())[:512],
            import_report=report,
        )

    @staticmethod
    def resolve_skill_file(source_path: str | Path) -> Path:
        path = Path(source_path).expanduser().resolve()
        if path.is_dir():
            path = path / "SKILL.md"
        if path.name != "SKILL.md" or not path.is_file():
            raise ValueError("Skill import path must be a SKILL.md file or a directory containing SKILL.md")
        return path


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, content

    frontmatter_lines: list[str] = []
    end_index: int | None = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_index = index
            break
        frontmatter_lines.append(line)
    if end_index is None:
        return {}, content

    metadata: dict[str, Any] = {}
    index = 0
    while index < len(frontmatter_lines):
        line = frontmatter_lines[index]
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            index += 1
            continue
        key, value = line.split(":", 1)
        cleaned_key = key.strip()
        cleaned_value = value.strip()
        if not cleaned_key:
            index += 1
            continue
        if cleaned_value:
            metadata[cleaned_key] = parse_metadata_value(cleaned_value)
            index += 1
            continue

        items: list[str] = []
        index += 1
        while index < len(frontmatter_lines):
            child = frontmatter_lines[index]
            stripped = child.strip()
            if not child.startswith((" ", "\t")) or (":" in child and not stripped.startswith("-")):
                break
            if stripped.startswith("-"):
                items.append(stripped[1:].strip().strip('"\''))
            index += 1
        metadata[cleaned_key] = items

    return metadata, "\n".join(lines[end_index + 1 :])


def parse_metadata_value(value: str) -> Any:
    stripped = value.strip().strip('"\'')
    if value.startswith("[") and value.endswith("]"):
        return [item.strip().strip('"\'') for item in value[1:-1].split(",") if item.strip()]
    if stripped.lower() == "true":
        return True
    if stripped.lower() == "false":
        return False
    return stripped


def frontmatter_value(frontmatter: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in frontmatter:
            return frontmatter[key]
    return None


def string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def extract_title(body: str) -> str | None:
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip() or None
    return None


def first_non_heading_line(body: str) -> str | None:
    for line in body.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    return None


def title_from_name(name: str) -> str:
    return " ".join(part.capitalize() for part in name.replace("_", "-").split("-") if part)


def normalize_skill_name(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", name.strip()).strip("-").lower()
    if not normalized:
        raise ValueError("Agent skill name is required")
    return normalized


def infer_source_types(content: str) -> list[str]:
    lowered = content.lower()
    candidates: list[str] = []
    if "xiaohongshu" in lowered or "小红书" in content:
        candidates.append("xiaohongshu_note")
    if "wechat" in lowered or "weixin" in lowered or "公众号" in content or "微信" in content:
        candidates.extend(["wechat_article", "wechat_account"])
    if "douyin" in lowered or "抖音" in content:
        candidates.append("douyin_video")
    return list(dict.fromkeys(candidates))


def infer_required_tools(content: str) -> list[str]:
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


def availability_state(required_tools: list[str]) -> str:
    return "unavailable" if required_tools else "available"


def collect_resources(package_root: Path) -> dict[str, list[str]]:
    resources: dict[str, list[str]] = {"scripts": [], "references": [], "assets": [], "agents": []}
    for folder in resources:
        root = package_root / folder
        if not root.is_dir():
            continue
        resources[folder] = sorted(
            path.relative_to(package_root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        )
    return resources


def load_openai_agent_metadata(resources: dict[str, list[str]]) -> dict[str, Any]:
    return {"agent_files": [path for path in resources.get("agents", []) if path.endswith((".yaml", ".yml", ".json"))]}


def package_hash(package_root: Path) -> str:
    digest = sha256()
    for path in sorted(package_root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative_path = path.relative_to(package_root).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def evaluate_description_quality(description: str) -> int:
    text = description.strip().lower()
    if not text:
        return 0

    score = 0
    score += criterion_score(text, ("用户", "当", "when", "提供", "输入", "粘贴", "上传", "选择", "user", "given", "provide"))
    score += criterion_score(
        text,
        (
            "文件",
            "文档",
            "图片",
            "视频",
            "音频",
            "网页",
            "页面",
            "链接",
            "文章",
            "账号",
            "代码",
            "仓库",
            "数据",
            "表格",
            "简历",
            "pdf",
            "url",
            "file",
            "document",
            "image",
            "video",
            "audio",
            "page",
            "article",
            "code",
            "repo",
            "data",
            "table",
        ),
    )
    score += criterion_score(
        text,
        (
            "抽取",
            "解析",
            "抓取",
            "读取",
            "同步",
            "识别",
            "转换",
            "生成",
            "总结",
            "分类",
            "验证",
            "搜索",
            "查询",
            "分析",
            "extract",
            "parse",
            "fetch",
            "read",
            "convert",
            "generate",
            "summarize",
            "classify",
            "validate",
            "search",
            "analyze",
        ),
    )
    score += criterion_score(
        text,
        (
            "输出",
            "返回",
            "保存",
            "生成",
            "结果",
            "列表",
            "字段",
            "表格",
            "json",
            "csv",
            "报告",
            "摘要",
            "候选",
            "信号",
            "公司",
            "岗位",
            "return",
            "output",
            "save",
            "result",
            "list",
            "field",
            "report",
            "summary",
        ),
    )
    score += criterion_score(
        text,
        (
            "不自动",
            "不修改",
            "原始",
            "确认",
            "边界",
            "只",
            "仅",
            "不得",
            "禁止",
            "需要确认",
            "without",
            "confirm",
            "boundary",
            "only",
            "not",
            "never",
            "readonly",
        ),
    )
    return min(score, 10)


def criterion_score(text: str, keywords: tuple[str, ...]) -> int:
    matches = sum(1 for keyword in keywords if keyword in text)
    if matches >= 2:
        return 2
    if matches == 1:
        return 1
    return 0


def blocking_errors_for(description: str) -> list[str]:
    errors: list[str] = []
    stripped = description.strip()
    if not stripped:
        errors.append("description 缺失，不能自动导入为可触发 Skill")
        return errors
    if requires_confirmation(stripped) and not has_confirmation_boundary(stripped):
        errors.append("description 包含投递/提交能力，但没有写明用户确认边界")
    return errors


def warnings_for(description: str, score: int) -> list[str]:
    warnings: list[str] = []
    stripped = description.strip()
    if stripped and len(stripped) < 10:
        warnings.append("description 过短，自动触发将被禁用")
    if stripped.lower() in GENERIC_DESCRIPTION_PATTERNS or score <= 5:
        warnings.append("description 过于空泛，只允许手动选择，不允许自动触发")
    return warnings


def security_risk(frontmatter: dict[str, Any], allowed_tools: list[str], disallowed_tools: list[str], required_tools: list[str]) -> str:
    values = " ".join(
        [
            str(frontmatter.get("description") or ""),
            " ".join(allowed_tools),
            " ".join(disallowed_tools),
            " ".join(required_tools),
        ]
    ).lower()
    if any(term in values for term in HIGH_RISK_TERMS):
        return "high"
    if allowed_tools or required_tools:
        return "medium"
    return "low"


def auto_trigger_state_for(score: int, blocking_errors: list[str], security_risk_level: str) -> str:
    if blocking_errors:
        return "disabled"
    if score >= 8 and security_risk_level != "high":
        return "enabled"
    if score >= 6:
        return "manual_only"
    return "disabled"


def requires_confirmation(description: str) -> bool:
    lowered = description.lower()
    return any(term in lowered for term in ("投递", "提交", "submit", "apply"))


def has_confirmation_boundary(description: str) -> bool:
    lowered = description.lower()
    return any(term in lowered for term in ("确认", "不自动", "用户", "confirm", "without"))
