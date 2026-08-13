"""汇总三类仿真教师标签的在线执行效果。"""

from dataclasses import asdict, dataclass, fields
import csv
from io import StringIO
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

from src.dppo_dataset import ExpertTransitionRecord


TEACHER_NAMES = ("cost", "reliability", "balanced")


@dataclass(frozen=True)
class TeacherTransitionMetric:
    """保存一条 held-out 专家动作通过完整执行链后的可观察指标。"""

    partition: str
    teacher_name: str
    episode_seed: int
    slow_step: int
    reward: float
    raw_feasible: bool
    projection_change_ratio: float
    projection_rejected: bool
    repair_attempts: int
    repair_successes: int
    repair_failures: int
    fast_solver_ran: bool
    fast_solver_time_seconds: float
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.partition not in {"validation", "test"}:
            raise ValueError("partition 只能是 validation 或 test。")
        if self.teacher_name not in TEACHER_NAMES:
            raise ValueError("teacher_name 不受支持。")
        for name in ("episode_seed", "slow_step", "repair_attempts", "repair_successes", "repair_failures"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} 必须是非负整数。")
        for name in ("reward", "projection_change_ratio", "fast_solver_time_seconds"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} 必须是有限数。")
            object.__setattr__(self, name, value)
        if not 0.0 <= self.projection_change_ratio <= 1.0:
            raise ValueError("projection_change_ratio 必须位于 [0, 1]。")
        if self.fast_solver_time_seconds < 0.0:
            raise ValueError("fast_solver_time_seconds 必须非负。")
        for name in ("raw_feasible", "projection_rejected", "fast_solver_ran"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} 必须是布尔值。")
        reasons = tuple(self.rejection_reasons)
        if any(not isinstance(reason, str) or not reason for reason in reasons):
            raise ValueError("rejection_reasons 必须由非空字符串组成。")
        object.__setattr__(self, "rejection_reasons", reasons)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_teacher_metrics(
    rows: Sequence[TeacherTransitionMetric],
    *,
    minimum_sample_count: int = 3,
) -> dict[str, object]:
    """按教师汇总 held-out 执行指标，并对小样本组显式降级结论。"""

    if isinstance(minimum_sample_count, bool) or minimum_sample_count <= 0:
        raise ValueError("minimum_sample_count 必须是正整数。")
    normalized = tuple(rows)
    if any(not isinstance(row, TeacherTransitionMetric) for row in normalized):
        raise TypeError("rows 必须包含 TeacherTransitionMetric。")
    teachers: dict[str, object] = {}
    for teacher_name in TEACHER_NAMES:
        selected = tuple(row for row in normalized if row.teacher_name == teacher_name)
        solver_times = tuple(
            row.fast_solver_time_seconds for row in selected if row.fast_solver_ran
        )
        repair_attempts = sum(row.repair_attempts for row in selected)
        repair_successes = sum(row.repair_successes for row in selected)
        reasons: dict[str, int] = {}
        for row in selected:
            for reason in row.rejection_reasons:
                reasons[reason] = reasons.get(reason, 0) + 1
        teachers[teacher_name] = {
            "sample_count": len(selected),
            "sufficient_samples": len(selected) >= minimum_sample_count,
            "mean_reward": _mean(tuple(row.reward for row in selected)),
            "raw_feasibility_rate": _mean(
                tuple(float(row.raw_feasible) for row in selected)
            ),
            "mean_projection_change_ratio": _mean(
                tuple(row.projection_change_ratio for row in selected)
            ),
            "projection_rejection_rate": _mean(
                tuple(float(row.projection_rejected) for row in selected)
            ),
            "repair_success_rate": (
                repair_successes / repair_attempts if repair_attempts else None
            ),
            "fast_solver_run_count": len(solver_times),
            "fast_solver_mean_time_seconds": _mean(solver_times),
            "fast_solver_max_time_seconds": max(solver_times, default=None),
            "fast_solver_total_time_seconds": sum(solver_times),
            "rejection_reason_counts": dict(sorted(reasons.items())),
        }
    return {
        "schema_version": "dppo-teacher-execution-diagnostics-v1",
        "minimum_sample_count": minimum_sample_count,
        "record_count": len(normalized),
        "teachers": teachers,
    }


def summarize_teacher_candidate_coverage(
    accepted_records: Sequence[ExpertTransitionRecord],
    rejected_records: Sequence[ExpertTransitionRecord],
) -> dict[str, object]:
    """汇总 held-out 候选进入行为克隆数据的比例及过滤原因。"""

    accepted = tuple(accepted_records)
    rejected = tuple(rejected_records)
    if any(not isinstance(record, ExpertTransitionRecord) for record in (*accepted, *rejected)):
        raise TypeError("候选覆盖记录必须是 ExpertTransitionRecord。")
    coverage: dict[str, object] = {}
    for teacher_name in TEACHER_NAMES:
        teacher_accepted = tuple(
            record for record in accepted if record.teacher_name == teacher_name
        )
        teacher_rejected = tuple(
            record for record in rejected if record.teacher_name == teacher_name
        )
        candidate_count = len(teacher_accepted) + len(teacher_rejected)
        reason_counts: dict[str, int] = {}
        for record in teacher_rejected:
            for reason in record.rejection_reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        coverage[teacher_name] = {
            "candidate_count": candidate_count,
            "accepted_count": len(teacher_accepted),
            "rejected_count": len(teacher_rejected),
            "acceptance_rate": (
                len(teacher_accepted) / candidate_count if candidate_count else None
            ),
            "rejection_reason_counts": dict(sorted(reason_counts.items())),
        }
    return coverage


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_teacher_diagnostic_outputs(
    output_root: str | Path,
    rows: Sequence[TeacherTransitionMetric],
    summary: Mapping[str, object],
) -> None:
    """原子写出逐条 CSV 和不含 NaN/Infinity 的严格 JSON。"""

    normalized = tuple(rows)
    stream = StringIO(newline="")
    field_names = tuple(field.name for field in fields(TeacherTransitionMetric))
    writer = csv.DictWriter(stream, fieldnames=field_names, lineterminator="\n")
    writer.writeheader()
    for row in normalized:
        payload = asdict(row)
        payload["rejection_reasons"] = json.dumps(
            payload["rejection_reasons"], ensure_ascii=False, separators=(",", ":")
        )
        writer.writerow(payload)
    json_text = json.dumps(
        dict(summary), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    root = Path(output_root)
    _atomic_write_text(root / "teacher_transition_metrics.csv", stream.getvalue())
    _atomic_write_text(root / "teacher_summary.json", json_text)
