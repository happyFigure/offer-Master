from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ReflectionQuality(str, Enum):
    GOOD = "good"
    PARTIAL = "partial"
    BAD = "bad"
    UNSAFE = "unsafe"
    UNKNOWN = "unknown"


class ReflectionNextAction(str, Enum):
    CONTINUE = "continue"
    RETRY = "retry"
    REPLAN = "replan"
    ASK_USER = "ask_user"
    STOP = "stop"


@dataclass(frozen=True)
class ReflectionDecision:
    quality: ReflectionQuality
    next_action: ReflectionNextAction
    confidence: float
    reason: str
    suggested_input_patch: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "quality": self.quality.value,
            "next_action": self.next_action.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "suggested_input_patch": dict(self.suggested_input_patch),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CapabilityResultEvaluationSpec:
    evaluator_id: str
    good_result_criteria: tuple[str, ...] = field(default_factory=tuple)
    bad_result_criteria: tuple[str, ...] = field(default_factory=tuple)
    uncertain_result_criteria: tuple[str, ...] = field(default_factory=tuple)
    retry_instruction: str | None = None
    rule_evaluator_id: str | None = None
    model_guidance_examples: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "evaluator_id": self.evaluator_id,
            "good_result_criteria": list(self.good_result_criteria),
            "bad_result_criteria": list(self.bad_result_criteria),
            "uncertain_result_criteria": list(self.uncertain_result_criteria),
            "retry_instruction": self.retry_instruction,
            "rule_evaluator_id": self.rule_evaluator_id,
            "model_guidance_examples": [dict(item) for item in self.model_guidance_examples],
        }


def campus_recruiting_web_search_result_evaluation_spec() -> CapabilityResultEvaluationSpec:
    return CapabilityResultEvaluationSpec(
        evaluator_id="campus_recruiting_web_search",
        rule_evaluator_id="campus_recruiting_web_search_rules",
        good_result_criteria=(
            "结果明确匹配目标公司。",
            "结果明确包含校园招聘、校招、应届生、毕业生招聘或投递入口。",
            "优先来自官网、官方招聘系统、官方公众号或可信招聘平台。",
        ),
        bad_result_criteria=(
            "结果是百科、词典、汉字解释或其他明显偏题内容。",
            "结果不匹配目标公司。",
            "结果只是公司介绍、新闻、公关稿、股票财报或产品介绍，无法支持校招查询。",
        ),
        uncertain_result_criteria=(
            "结果命中公司名，但招聘入口或校招意图不明确。",
            "结果信息过短，无法确认是否满足用户目标。",
        ),
        retry_instruction="用“公司名 校园招聘 官网 年份”重新搜索。",
        model_guidance_examples=(
            {
                "target": "查中科曙光校招官网",
                "result": "中（汉语汉字）_百度百科",
                "quality": "bad",
                "next_action": "retry",
                "reason": "没有目标公司，也不是校招信息。",
            },
            {
                "target": "查腾讯校招",
                "result": "腾讯校园招聘官网 join.qq.com",
                "quality": "good",
                "next_action": "continue",
                "reason": "目标公司匹配，校招入口明确。",
            },
        ),
    )


def resume_tailoring_result_evaluation_spec() -> CapabilityResultEvaluationSpec:
    return CapabilityResultEvaluationSpec(
        evaluator_id="resume_tailoring",
        good_result_criteria=(
            "保留用户原始经历，不编造公司、学历、项目、时间或成果。",
            "明确结合目标岗位描述优化表达。",
            "直接输出改写后的简历内容，而不是只给修改建议。",
            "说明关键修改点，方便主 agent 汇总给用户。",
        ),
        bad_result_criteria=(
            "编造或夸大用户没有提供的经历。",
            "只输出泛泛建议，没有直接输出改写后的简历。",
            "删除用户原始简历里的关键项目、教育或工作经历。",
            "改写内容和目标岗位不相关。",
        ),
        uncertain_result_criteria=(
            "改写了一部分内容，但看不出是否贴合目标岗位。",
            "输出过短，无法确认是否保留了用户真实经历。",
            "目标岗位描述不足，无法判断优化方向是否正确。",
        ),
        retry_instruction="保留真实经历，按目标岗位重新输出 revised_resume，并说明修改摘要。",
        model_guidance_examples=(
            {
                "target": "根据 Java 后端 JD 优化简历",
                "result": "建议突出 Spring Boot 项目经验。",
                "quality": "bad",
                "next_action": "retry",
                "reason": "只给建议，没有直接输出改写后的简历。",
            },
            {
                "target": "根据 Java 后端 JD 优化简历",
                "result": "输出 revised_resume，并说明保留原项目、突出接口性能优化。",
                "quality": "good",
                "next_action": "continue",
                "reason": "保留真实经历，且直接完成了改写。",
            },
        ),
    )


def job_match_analysis_result_evaluation_spec() -> CapabilityResultEvaluationSpec:
    return CapabilityResultEvaluationSpec(
        evaluator_id="job_match_analysis",
        good_result_criteria=(
            "明确比较候选人经历和岗位要求。",
            "输出匹配理由、差距风险和可行动建议。",
            "给出可解释的匹配结论或匹配分数。",
        ),
        bad_result_criteria=(
            "只复述岗位描述，没有分析候选人是否匹配。",
            "没有说明差距、风险或下一步建议。",
            "在缺少简历或岗位信息时仍然武断给出结论。",
        ),
        uncertain_result_criteria=(
            "只分析了部分岗位要求。",
            "候选人信息或岗位描述不足，结论依据不充分。",
        ),
        retry_instruction="补齐匹配理由、风险点、可行动建议；缺少关键材料时改为 ask_user。",
    )


def application_entry_discovery_result_evaluation_spec() -> CapabilityResultEvaluationSpec:
    return CapabilityResultEvaluationSpec(
        evaluator_id="application_entry_discovery",
        good_result_criteria=(
            "找到目标岗位或目标公司的官方申请入口。",
            "结果包含可追溯证据，例如官网链接、招聘系统链接或页面标题。",
            "执行过程明确停在最终提交前，没有替用户提交申请。",
        ),
        bad_result_criteria=(
            "找到的是公司首页、泛招聘页或无关页面，无法投递目标岗位。",
            "缺少申请入口链接或证据。",
            "已经执行最终提交、上传未确认简历或做了高风险动作。",
        ),
        uncertain_result_criteria=(
            "找到疑似入口，但无法确认是否对应目标岗位。",
            "被登录、验证码、地区跳转或页面权限阻塞。",
        ),
        retry_instruction="优先用公司名、岗位名、job_id、source_url、apply_url_candidate 重新定位官方入口；若被登录或验证码阻塞则 ask_user。",
    )


def offerio_company_jobs_sync_result_evaluation_spec() -> CapabilityResultEvaluationSpec:
    return CapabilityResultEvaluationSpec(
        evaluator_id="offerio_company_jobs_sync",
        good_result_criteria=(
            "同步任务成功完成。",
            "返回抓取数量、写入或更新数量、失败数量。",
            "结果包含同步任务编号或可追踪记录。",
        ),
        bad_result_criteria=(
            "同步失败或返回错误。",
            "没有返回抓取、写入或失败数量，无法判断同步效果。",
            "返回的数据源不是 OfferIO 公司聚合岗位库。",
        ),
        uncertain_result_criteria=(
            "任务状态不是成功也不是失败，例如仍在排队或部分完成。",
            "写入数量为 0，但没有解释原因。",
        ),
        retry_instruction="如果是临时失败可重试；如果缺少 source_id、权限或数据源异常则 ask_user 或 replan。",
    )


_RESULT_EVALUATION_BY_CAPABILITY_ID = {
    "external.web_search": campus_recruiting_web_search_result_evaluation_spec,
    "resume.tailor": resume_tailoring_result_evaluation_spec,
    "job.match": job_match_analysis_result_evaluation_spec,
    "applications.find_apply_entry": application_entry_discovery_result_evaluation_spec,
    "offerio.sync_company_jobs": offerio_company_jobs_sync_result_evaluation_spec,
}

_RESULT_EVALUATION_BY_INTENT = {
    "campus_recruiting_search": campus_recruiting_web_search_result_evaluation_spec,
    "resume_tailoring": resume_tailoring_result_evaluation_spec,
    "job_match_analysis": job_match_analysis_result_evaluation_spec,
    "application_entry_discovery": application_entry_discovery_result_evaluation_spec,
    "offerio_company_jobs_sync": offerio_company_jobs_sync_result_evaluation_spec,
}


def result_evaluation_spec_for_capability(
    capability_id: str,
    *,
    supported_intents: Iterable[str] = (),
) -> CapabilityResultEvaluationSpec | None:
    factory = _RESULT_EVALUATION_BY_CAPABILITY_ID.get(str(capability_id or "").strip())
    if factory is not None:
        return factory()
    for intent in supported_intents:
        factory = _RESULT_EVALUATION_BY_INTENT.get(str(intent or "").strip())
        if factory is not None:
            return factory()
    return None
