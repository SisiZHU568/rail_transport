"""测试三类教师标签的独立执行诊断与稳定输出。"""

import csv
import json

import pytest
import numpy as np

import run_dppo_teacher_diagnostics as teacher_command
from src.dppo_dataset import ExpertTransitionRecord
from src.dppo_teacher_diagnostics import (
    TeacherTransitionMetric,
    summarize_teacher_candidate_coverage,
    summarize_teacher_metrics,
    write_teacher_diagnostic_outputs,
)


def _metric(
    teacher_name: str,
    *,
    reward: float,
    repair_successes: int = 1,
    solver_time: float = 0.02,
) -> TeacherTransitionMetric:
    return TeacherTransitionMetric(
        partition="validation",
        teacher_name=teacher_name,
        episode_seed=100,
        slow_step=0,
        reward=reward,
        raw_feasible=True,
        projection_change_ratio=0.0,
        projection_rejected=False,
        repair_attempts=1,
        repair_successes=repair_successes,
        repair_failures=1 - repair_successes,
        fast_solver_ran=True,
        fast_solver_time_seconds=solver_time,
        rejection_reasons=(),
    )


def test_summary_always_emits_all_teachers_and_marks_small_groups() -> None:
    """均衡教师没有样本时也要显示，少于 3 条不能据此调整训练比例。"""

    rows = tuple(
        _metric("cost", reward=-0.2 - index * 0.01)
        for index in range(3)
    ) + (_metric("reliability", reward=-0.5),)

    summary = summarize_teacher_metrics(rows, minimum_sample_count=3)

    assert tuple(summary["teachers"]) == ("cost", "reliability", "balanced")
    assert summary["teachers"]["cost"]["sample_count"] == 3
    assert summary["teachers"]["cost"]["sufficient_samples"] is True
    assert summary["teachers"]["reliability"]["sufficient_samples"] is False
    assert summary["teachers"]["balanced"]["sample_count"] == 0
    assert summary["teachers"]["balanced"]["mean_reward"] is None


def test_summary_aggregates_repairs_and_solver_time_by_teacher() -> None:
    rows = (
        _metric("cost", reward=-0.2, repair_successes=1, solver_time=0.02),
        _metric("cost", reward=-0.4, repair_successes=0, solver_time=0.04),
        _metric("cost", reward=-0.3, repair_successes=1, solver_time=0.03),
    )

    summary = summarize_teacher_metrics(rows, minimum_sample_count=3)
    cost = summary["teachers"]["cost"]

    assert cost["mean_reward"] == pytest.approx(-0.3)
    assert cost["repair_success_rate"] == pytest.approx(2 / 3)
    assert cost["fast_solver_run_count"] == 3
    assert cost["fast_solver_mean_time_seconds"] == pytest.approx(0.03)
    assert cost["fast_solver_max_time_seconds"] == pytest.approx(0.04)
    assert cost["fast_solver_total_time_seconds"] == pytest.approx(0.09)


def test_metric_rejects_training_partition_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="partition"):
        TeacherTransitionMetric(
            **{**_metric("cost", reward=-0.2).__dict__, "partition": "train"}
        )
    with pytest.raises(ValueError, match="reward"):
        TeacherTransitionMetric(
            **{**_metric("cost", reward=-0.2).__dict__, "reward": float("nan")}
        )


def test_outputs_have_reviewable_csv_and_strict_json(tmp_path) -> None:
    rows = tuple(_metric("cost", reward=-0.2) for _ in range(3))
    summary = summarize_teacher_metrics(rows, minimum_sample_count=3)

    write_teacher_diagnostic_outputs(tmp_path, rows, summary)

    with (tmp_path / "teacher_transition_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        csv_rows = list(csv.DictReader(stream))
    loaded_summary = json.loads(
        (tmp_path / "teacher_summary.json").read_text(encoding="utf-8")
    )
    assert len(csv_rows) == 3
    assert {
        "partition",
        "teacher_name",
        "episode_seed",
        "slow_step",
        "reward",
        "fast_solver_time_seconds",
        "rejection_reasons",
    } <= csv_rows[0].keys()
    assert loaded_summary == summary


def test_candidate_coverage_exposes_filtered_balanced_teacher() -> None:
    """最终标签为零时，仍要报告候选数量和被过滤原因。"""

    accepted = (
        ExpertTransitionRecord(
            state=np.array([0.0], dtype=np.float32),
            expert_action=np.array([0.0], dtype=np.float32),
            teacher_name="cost",
            episode_seed=1,
            slow_step=0,
            teacher_proposal_raw_feasible=True,
            expert_action_feasible=True,
            projection_change_ratio=0.0,
            final_feasible=True,
            run_cost=0.0,
            route_cost=0.0,
            cold_start_cost=0.0,
        ),
    )
    rejected = (
        ExpertTransitionRecord(
            state=np.array([0.0], dtype=np.float32),
            expert_action=np.array([0.0], dtype=np.float32),
            teacher_name="balanced",
            episode_seed=2,
            slow_step=0,
            teacher_proposal_raw_feasible=False,
            expert_action_feasible=False,
            projection_change_ratio=1.0,
            final_feasible=False,
            run_cost=0.0,
            route_cost=0.0,
            cold_start_cost=0.0,
            rejection_reasons=("expert_action_not_raw_feasible",),
        ),
    )

    coverage = summarize_teacher_candidate_coverage(accepted, rejected)

    assert coverage["cost"]["candidate_count"] == 1
    assert coverage["cost"]["acceptance_rate"] == 1.0
    assert coverage["balanced"]["candidate_count"] == 1
    assert coverage["balanced"]["accepted_count"] == 0
    assert coverage["balanced"]["acceptance_rate"] == 0.0
    assert coverage["balanced"]["rejection_reason_counts"] == {
        "expert_action_not_raw_feasible": 1
    }


def test_replay_executes_only_held_out_target_action(monkeypatch) -> None:
    """目标标签必须从原教师轨迹状态独立执行，且训练分区不得进入诊断。"""

    class FakeTeacher:
        def propose(self, _snapshot):
            return type("Proposal", (), {"relaxed_action": np.array([0.1])})()

    class FakeScenario:
        def __init__(self) -> None:
            self.step_index = 0

        def reset(self, *, seed: int):
            self.step_index = 0
            return np.array([float(seed), 0.0], dtype=np.float32)

        def current_public_snapshot(self):
            return object()

        def step(self, action):
            self.step_index += 1
            state = np.array([100.0, float(self.step_index)], dtype=np.float32)
            target = bool(np.isclose(float(np.asarray(action)[0]), 0.9))
            info = {
                "raw_feasible": target,
                "projection_change_ratio": 0.0 if target else 0.5,
                "projection_success": target,
                "fast_repair_attempts": 2,
                "fast_repair_successes": 1,
                "fast_repair_failures": 1,
                "fast_solver_status": "optimal",
                "fast_solver_time_seconds": 0.03,
                "rejection_reasons": () if target else ("rejected",),
            }
            return state, -0.2 if target else -0.8, False, False, info

    monkeypatch.setattr(teacher_command, "build_dppo_scenario", lambda _config: FakeScenario())
    monkeypatch.setattr(
        teacher_command,
        "build_simulation_teacher",
        lambda _name, _scenario: FakeTeacher(),
    )
    record = ExpertTransitionRecord(
        state=np.array([100.0, 1.0], dtype=np.float32),
        expert_action=np.array([0.9], dtype=np.float32),
        teacher_name="cost",
        episode_seed=100,
        slow_step=1,
        teacher_proposal_raw_feasible=False,
        expert_action_feasible=True,
        projection_change_ratio=0.0,
        final_feasible=True,
        run_cost=1.0,
        route_cost=1.0,
        cold_start_cost=0.0,
    )

    row = teacher_command.replay_expert_transition({}, "test", record)

    assert row.partition == "test"
    assert row.reward == pytest.approx(-0.2)
    assert row.raw_feasible is True
    assert row.fast_solver_time_seconds == pytest.approx(0.03)


def test_replay_rejects_saved_state_mismatch(monkeypatch) -> None:
    class FakeScenario:
        def reset(self, *, seed: int):
            return np.array([0.0], dtype=np.float32)

    monkeypatch.setattr(teacher_command, "build_dppo_scenario", lambda _config: FakeScenario())
    monkeypatch.setattr(
        teacher_command,
        "build_simulation_teacher",
        lambda _name, _scenario: object(),
    )
    record = ExpertTransitionRecord(
        state=np.array([1.0], dtype=np.float32),
        expert_action=np.array([0.0], dtype=np.float32),
        teacher_name="cost",
        episode_seed=1,
        slow_step=0,
        teacher_proposal_raw_feasible=True,
        expert_action_feasible=True,
        projection_change_ratio=0.0,
        final_feasible=True,
        run_cost=0.0,
        route_cost=0.0,
        cold_start_cost=0.0,
    )

    with pytest.raises(ValueError, match="重放状态不一致"):
        teacher_command.replay_expert_transition({}, "validation", record)
