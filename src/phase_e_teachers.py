"""从因果安全候选中生成三类服务优先教师标签。"""

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np


@dataclass(frozen=True)
class TeacherCandidate:
    plan_hash: str
    scores: np.ndarray
    predicted_deficit: float
    predicted_cost: float
    structural_reliability_margin: float
    balance_score: float
    canonical_key: str


@dataclass(frozen=True)
class TeacherLabel:
    plan_hash: str
    scores: np.ndarray
    unbounded_action: np.ndarray
    teacher_types: tuple[str, ...]


def select_teacher_labels(candidates: tuple[TeacherCandidate, ...]) -> tuple[TeacherLabel, ...]:
    """三个教师都先最小化逐时隙服务缺口，再使用各自次级目标。"""
    if not candidates:
        return ()
    for candidate in candidates:
        if np.asarray(candidate.scores).ndim != 1 or not np.all(
            (candidate.scores > 0.0) & (candidate.scores < 1.0)
        ):
            raise ValueError("教师评分必须严格位于 (0,1)，以便保存无界变量 v。")
    selected = {
        "COST": min(candidates, key=lambda item: (
            item.predicted_deficit, item.predicted_cost,
            -item.structural_reliability_margin, item.balance_score, item.canonical_key)),
        "RELIABILITY": min(candidates, key=lambda item: (
            item.predicted_deficit, -item.structural_reliability_margin,
            item.predicted_cost, item.balance_score, item.canonical_key)),
        "BALANCE": min(candidates, key=lambda item: (
            item.predicted_deficit, item.balance_score, item.predicted_cost,
            -item.structural_reliability_margin, item.canonical_key)),
    }
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
        ))
    return tuple(labels)
