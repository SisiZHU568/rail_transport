"""阶段 E 教师候选的完整安全搜索、生命周期预演与解码往返审计。"""

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Callable, Mapping

from src.failure_process import FailureSnapshot
from src.instance_lifecycle import (
    DeploymentTarget,
    InstanceLifecycleManager,
    LifecycleCommitResult,
    LifecycleDeploymentPlan,
)
from src.phase_e_teachers import TeacherCandidate, TeacherLabel, select_teacher_labels
from src.safe_deployment_decoder import (
    DecoderInput,
    DeploymentPlan,
    SafeDeploymentDecoder,
)


@dataclass(frozen=True)
class TeacherMetrics:
    predicted_deficit: float
    predicted_cost: float
    structural_reliability_margin: float
    balance_maximum: float
    balance_variance: float

    def __post_init__(self) -> None:
        values = (
            self.predicted_deficit,
            self.predicted_cost,
            self.structural_reliability_margin,
            self.balance_maximum,
            self.balance_variance,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("教师代理指标必须全部为有限数。")
        if self.predicted_deficit < 0.0 or self.predicted_cost < 0.0:
            raise ValueError("教师缺口和成本不能为负。")
        if self.balance_maximum < 0.0 or self.balance_variance < 0.0:
            raise ValueError("教师利用率均衡指标不能为负。")


@dataclass(frozen=True)
class TeacherCandidateBuildResult:
    code: str
    candidate: TeacherCandidate | None
    preview: LifecycleCommitResult | None


@dataclass(frozen=True)
class ExactTeacherGenerationResult:
    code: str
    labels: tuple[TeacherLabel, ...]
    candidate_count: int
    generated_patterns: int


def _canonical_plan(plan: DeploymentPlan) -> tuple[str, str]:
    counts = tuple(
        (function_id, node_id, int(value))
        for (function_id, node_id), value in sorted(plan.instance_counts.items())
    )
    retention = tuple(
        (function_id, node_id, int(value))
        for (function_id, node_id), value in sorted(plan.retention_slots.items())
    )
    serialized = json.dumps(
        {"instance_counts": counts, "retention_slots": retention},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    # 规范键先比较实例数、再比较保留档，哈希只负责最终消歧。
    key = json.dumps((counts, retention, digest), separators=(",", ":"))
    return digest, key


def _lifecycle_plan(
    plan: DeploymentPlan,
    lifecycle_manager: InstanceLifecycleManager,
    failure_snapshot: FailureSnapshot,
) -> LifecycleDeploymentPlan:
    snapshot = lifecycle_manager.snapshot()
    return LifecycleDeploymentPlan(
        expected_lifecycle_version=snapshot.version,
        expected_failure_version=failure_snapshot.version,
        current_slot=snapshot.current_slot,
        targets=tuple(
            DeploymentTarget(
                function_id,
                node_id,
                plan.instance_counts[(function_id, node_id)],
                plan.retention_slots[(function_id, node_id)],
            )
            for function_id, node_id in sorted(plan.instance_counts)
        ),
    )


def build_teacher_candidate(
    *,
    plan: DeploymentPlan,
    eligible_teacher_types: tuple[str, ...],
    decoder: SafeDeploymentDecoder,
    decoder_input: DecoderInput,
    lifecycle_manager: InstanceLifecycleManager,
    failure_snapshot: FailureSnapshot,
    metrics: TeacherMetrics,
) -> TeacherCandidateBuildResult:
    """只在生命周期预演和同解码器往返都通过后建立候选。"""

    allowed_types = {"COST", "RELIABILITY", "BALANCE"}
    if not eligible_teacher_types or not set(eligible_teacher_types) <= allowed_types:
        raise ValueError("教师类型必须来自 COST/RELIABILITY/BALANCE。")
    preview = lifecycle_manager.preview_deployment(
        _lifecycle_plan(plan, lifecycle_manager, failure_snapshot),
        failure_snapshot,
    )
    if not preview.accepted:
        return TeacherCandidateBuildResult(preview.code, None, preview)
    encoded = decoder.encode_plan(plan, decoder_input)
    if encoded.code != "OK":
        return TeacherCandidateBuildResult(encoded.code, None, preview)
    plan_hash, canonical_key = _canonical_plan(plan)
    mask = tuple(
        value
        for pair in decoder.action_spec.pairs
        for value in (True, plan.instance_counts[pair] > 0)
    )
    candidate = TeacherCandidate(
        plan_hash=plan_hash,
        scores=encoded.scores,
        predicted_deficit=metrics.predicted_deficit,
        predicted_cost=metrics.predicted_cost,
        structural_reliability_margin=metrics.structural_reliability_margin,
        balance_score=metrics.balance_maximum,
        canonical_key=canonical_key,
        plan=plan,
        effective_action_mask=mask,
        balance_variance=metrics.balance_variance,
        eligible_teacher_types=eligible_teacher_types,
    )
    return TeacherCandidateBuildResult("OK", candidate, preview)


def generate_exact_teacher_labels(
    *,
    decoder: SafeDeploymentDecoder,
    decoder_input: DecoderInput,
    lifecycle_manager: InstanceLifecycleManager,
    failure_snapshot: FailureSnapshot,
    retention_profiles: Mapping[str, Mapping[tuple[int, int], int]],
    evaluator: Callable[[DeploymentPlan, LifecycleCommitResult], TeacherMetrics],
    max_generated_patterns: int,
) -> ExactTeacherGenerationResult:
    """完整枚举每类教师的固定保留规则；任一搜索被截断就不输出部分最优。"""

    candidates: list[TeacherCandidate] = []
    generated_patterns = 0
    for teacher_type in ("COST", "RELIABILITY", "BALANCE"):
        if teacher_type not in retention_profiles:
            continue
        enumeration = decoder.enumerate_safe_plans(
            decoder_input,
            max_generated_patterns=max_generated_patterns,
            retention_slots_by_pair=retention_profiles[teacher_type],
        )
        generated_patterns += enumeration.generated_patterns
        if enumeration.code == "DECODER_SEARCH_LIMIT":
            return ExactTeacherGenerationResult(
                "TEACHER_SEARCH_LIMIT", (), 0, generated_patterns
            )
        if enumeration.code == "NO_SAFE_FEASIBLE_DEPLOYMENT":
            continue
        try:
            for plan in enumeration.plans:
                preview = lifecycle_manager.preview_deployment(
                    _lifecycle_plan(plan, lifecycle_manager, failure_snapshot),
                    failure_snapshot,
                )
                if not preview.accepted:
                    continue
                built = build_teacher_candidate(
                    plan=plan,
                    eligible_teacher_types=(teacher_type,),
                    decoder=decoder,
                    decoder_input=decoder_input,
                    lifecycle_manager=lifecycle_manager,
                    failure_snapshot=failure_snapshot,
                    metrics=evaluator(plan, preview),
                )
                if built.code == "OK" and built.candidate is not None:
                    candidates.append(built.candidate)
        except (ArithmeticError, ValueError, RuntimeError):
            return ExactTeacherGenerationResult(
                "TEACHER_INTERNAL_FAILURE", (), 0, generated_patterns
            )
    if not candidates:
        return ExactTeacherGenerationResult(
            "NO_SAFE_FEASIBLE_DEPLOYMENT", (), 0, generated_patterns
        )
    labels = select_teacher_labels(tuple(candidates))
    return ExactTeacherGenerationResult(
        "OK", labels, len(candidates), generated_patterns
    )
