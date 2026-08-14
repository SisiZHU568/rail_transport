"""从因果安全候选中生成三类服务优先教师标签。"""

from dataclasses import dataclass
import numpy as np

from src.safe_deployment_decoder import DeploymentPlan


@dataclass(frozen=True)
class TeacherCandidate:
    plan_hash: str
    scores: np.ndarray
    predicted_deficit: float
    predicted_cost: float
    structural_reliability_margin: float
    balance_score: float
    canonical_key: str
    plan: DeploymentPlan | None = None
    effective_action_mask: tuple[bool, ...] = ()
    balance_variance: float = 0.0
    eligible_teacher_types: tuple[str, ...] = ("COST", "RELIABILITY", "BALANCE")


@dataclass(frozen=True)
class TeacherLabel:
    plan_hash: str
    scores: np.ndarray
    unbounded_action: np.ndarray
    teacher_types: tuple[str, ...]
    plan: DeploymentPlan | None = None
    effective_action_mask: tuple[bool, ...] = ()
    predicted_deficit: float = 0.0
    predicted_cost: float = 0.0
    structural_reliability_margin: float = 0.0
    balance_score: float = 0.0
    balance_variance: float = 0.0


def select_teacher_labels(candidates: tuple[TeacherCandidate, ...]) -> tuple[TeacherLabel, ...]:
    """三个教师都先最小化逐时隙服务缺口，再使用各自次级目标。"""
    if not candidates:
        return ()
    for candidate in candidates:
        if np.asarray(candidate.scores).ndim != 1 or not np.all(
            (candidate.scores > 0.0) & (candidate.scores < 1.0)
        ):
            raise ValueError("教师评分必须严格位于 (0,1)，以便保存无界变量 v。")
    pools = {
        teacher_type: tuple(
            item for item in candidates
            if teacher_type in item.eligible_teacher_types
        )
        for teacher_type in ("COST", "RELIABILITY", "BALANCE")
    }
    selected: dict[str, TeacherCandidate] = {}
    if pools["COST"]:
        selected["COST"] = min(pools["COST"], key=lambda item: (
            item.predicted_deficit, item.predicted_cost,
            -item.structural_reliability_margin, item.balance_score,
            item.balance_variance, item.canonical_key))
    if pools["RELIABILITY"]:
        selected["RELIABILITY"] = min(pools["RELIABILITY"], key=lambda item: (
            item.predicted_deficit, -item.structural_reliability_margin,
            item.predicted_cost, item.balance_score, item.balance_variance,
            item.canonical_key))
    if pools["BALANCE"]:
        selected["BALANCE"] = min(pools["BALANCE"], key=lambda item: (
            item.predicted_deficit, item.balance_score, item.balance_variance,
            item.predicted_cost, -item.structural_reliability_margin,
            item.canonical_key))
    grouped: dict[str, list[str]] = {}
    by_hash: dict[str, TeacherCandidate] = {}
    for teacher_type, candidate in selected.items():
        grouped.setdefault(candidate.plan_hash, []).append(teacher_type)
        by_hash[candidate.plan_hash] = candidate
    order = {"COST": 0, "RELIABILITY": 1, "BALANCE": 2}
    labels = []
    for plan_hash in sorted(grouped):
        candidate = by_hash[plan_hash]
        scores = np.asarray(candidate.scores, dtype=np.float64).copy()
        scores.setflags(write=False)
        unbounded = np.log(scores / (1.0 - scores))
        unbounded.setflags(write=False)
        labels.append(TeacherLabel(
            plan_hash, scores, unbounded,
            tuple(sorted(grouped[plan_hash], key=order.__getitem__)),
            candidate.plan,
            candidate.effective_action_mask,
            candidate.predicted_deficit,
            candidate.predicted_cost,
            candidate.structural_reliability_margin,
            candidate.balance_score,
            candidate.balance_variance,
        ))
    return tuple(labels)
