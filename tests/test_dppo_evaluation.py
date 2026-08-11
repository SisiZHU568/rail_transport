"""测试 DDPO 独立评估、统计汇总和结构化报告。"""

import copy
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from run_dppo_evaluation import write_evaluation_reports
from src.config import load_config
from src.dppo import DPPOAgent, DPPOConfig
from src.dppo_diffusion import ConditionalDiffusionMLP, CosineNoiseSchedule
from src.dppo_evaluation import (
    REQUIRED_EVALUATION_COLUMNS,
    DPPODecisionActionRecord,
    DPPOEpisodeEvaluation,
    delay_statistics,
    evaluate_dppo_episode,
    summarize_evaluation,
)
from src.dppo_projection import ProjectionResult
from src.dppo_scenario import build_dppo_environment


class AlwaysFailProjector:
    """让评估快速走完完整线路，同时保留真实环境状态推进。"""

    def project(self, **_: object) -> ProjectionResult:
        return ProjectionResult(
            function_intents=None,
            raw_feasible=False,
            success=False,
            reasons=("评估测试注入的投影失败",),
            changed_assignment_count=0,
            requested_assignment_count=1,
            change_ratio=0.0,
            fault_domains={},
        )


def _environment(*, reject: bool = True):
    environment = build_dppo_environment(load_config("configs/debug.yaml"))
    if reject:
        environment.projector = AlwaysFailProjector()
    return environment


def _agent(environment) -> DPPOAgent:
    torch.manual_seed(23)
    model = ConditionalDiffusionMLP(
        environment.dimensions.state_dim,
        environment.dimensions.action_dim,
        (16, 16),
    )
    return DPPOAgent(
        model,
        CosineNoiseSchedule(steps=4),
        DPPOConfig(
            diffusion_steps=4,
            fine_tuned_steps=2,
            value_hidden_dims=(16, 16),
            update_epochs=1,
            seed=37,
        ),
        device="cpu",
    )


def _metrics(**overrides: float) -> dict[str, float]:
    values = {key: 0.0 for key in REQUIRED_EVALUATION_COLUMNS}
    values.update(
        {
            "success_rate": 0.8,
            "sla_success_rate": 0.7,
            "mean_delay_ms": 5.0,
            "peak_delay_ms": 9.0,
            "p95_delay_ms": 8.0,
            "p99_delay_ms": 8.8,
            "mean_reliability": 0.999,
            "mean_replica_count_vnf_0": 2.5,
        }
    )
    values.update(overrides)
    return values


def _evaluation(seed: int, delay_shift: float = 0.0) -> DPPOEpisodeEvaluation:
    actions = (
        DPPODecisionActionRecord(
            episode_seed=seed,
            decision_index=0,
            function_id=0,
            replica_count=2,
            cloud_selected=False,
            primary_retention_seconds=10.0,
            backup_retention_seconds=5.0,
        ),
        DPPODecisionActionRecord(
            episode_seed=seed,
            decision_index=1,
            function_id=0,
            replica_count=3,
            cloud_selected=True,
            primary_retention_seconds=14.0,
            backup_retention_seconds=8.0,
        ),
    )
    return DPPOEpisodeEvaluation(
        episode_seed=seed,
        metrics=_metrics(mean_delay_ms=5.0 + delay_shift),
        fast_slot_delay_samples_ms=(1.0 + delay_shift, 9.0 + delay_shift),
        replica_counts_by_function={0: 2.5},
        action_records=actions,
    )


def _assert_same_state_dict(
    first: dict[str, torch.Tensor],
    second: dict[str, torch.Tensor],
) -> None:
    assert first.keys() == second.keys()
    assert all(torch.equal(first[key], second[key]) for key in first)


def test_evaluation_schema_contains_required_metrics_and_summary_uncertainty() -> None:
    """逐 Episode 表和跨 Episode 汇总必须覆盖论文指标与不确定性。"""

    evaluation_rows = [_metrics(), _metrics(success_rate=0.6)]

    assert REQUIRED_EVALUATION_COLUMNS <= evaluation_rows[0].keys()
    summary = summarize_evaluation(evaluation_rows)
    success_summary = next(row for row in summary if row["metric"] == "success_rate")

    assert success_summary["sample_count"] == 2
    assert success_summary["mean"] == pytest.approx(0.7)
    assert success_summary["standard_deviation"] > 0.0
    assert success_summary["confidence_interval_95"] > 0.0


def test_delay_percentiles_use_all_raw_fast_slot_samples() -> None:
    """P95/P99 必须直接来自快时隙原始样本，而不是窗口平均值。"""

    samples = (1.0, 2.0, 3.0, 100.0)
    statistics = delay_statistics(samples)

    assert statistics["mean_delay_ms"] == pytest.approx(np.mean(samples))
    assert statistics["peak_delay_ms"] == 100.0
    assert statistics["p95_delay_ms"] == pytest.approx(np.percentile(samples, 95))
    assert statistics["p99_delay_ms"] == pytest.approx(np.percentile(samples, 99))


def test_slow_environment_exposes_raw_delay_and_reliability_samples() -> None:
    """通用环境必须把窗口内原始样本交给评估层，不能只留下均值。"""

    environment = _environment(reject=False)
    environment.reset(seed=123)
    _, _, _, _, info = environment.step(
        np.zeros(environment.dimensions.action_dim, dtype=np.float32)
    )

    delays = info["fast_slot_delay_samples_ms"]
    reliabilities = info["exact_sfc_reliability_samples"]
    assert isinstance(delays, tuple)
    assert isinstance(reliabilities, tuple)
    assert all(math.isfinite(value) and value >= 0.0 for value in delays)
    assert all(0.0 <= value <= 1.0 for value in reliabilities)
    if delays:
        assert info["final_metrics"].average_successful_delay_ms == pytest.approx(
            np.mean(delays)
        )


def test_fixed_seed_evaluation_is_deterministic_and_never_updates_agent() -> None:
    """独立评估必须关闭梯度，同一 Episode 种子得到同一结果。"""

    environment = _environment()
    agent = _agent(environment)
    frozen_before = copy.deepcopy(agent.frozen_policy.state_dict())
    trainable_before = copy.deepcopy(agent.trainable_policy.state_dict())
    value_before = copy.deepcopy(agent.value_network.state_dict())

    first = evaluate_dppo_episode(environment, agent, episode_seed=51)
    second = evaluate_dppo_episode(environment, agent, episode_seed=51)

    assert first.metrics == second.metrics
    assert first.action_records == second.action_records
    assert REQUIRED_EVALUATION_COLUMNS <= first.metrics.keys()
    assert {
        f"mean_replica_count_vnf_{function_id}"
        for function_id in environment.dimensions.function_ids
    } <= first.metrics.keys()
    assert all(parameter.grad is None for parameter in agent.trainable_policy.parameters())
    _assert_same_state_dict(frozen_before, agent.frozen_policy.state_dict())
    _assert_same_state_dict(trainable_before, agent.trainable_policy.state_dict())
    _assert_same_state_dict(value_before, agent.value_network.state_dict())


def test_report_writer_separates_performance_replica_and_retention_outputs(
    tmp_path: Path,
) -> None:
    """论文性能、动作副本和保留时间必须分别输出到显式临时目录。"""

    output_root = tmp_path / "evaluation"
    paths = write_evaluation_reports(
        (_evaluation(70), _evaluation(71, delay_shift=2.0)),
        output_root,
    )

    expected_names = {
        "episode_metrics.csv",
        "summary_metrics.csv",
        "delay_samples.csv",
        "pooled_delay_metrics.csv",
        "replica_distribution.csv",
        "retention_distribution.csv",
        "evaluation_overview.png",
    }
    assert expected_names == {path.name for path in paths.values()}
    assert all(path.parent == output_root and path.exists() for path in paths.values())
    assert all(path.stat().st_size > 0 for path in paths.values())
    for path in paths.values():
        if path.suffix == ".csv":
            assert path.read_bytes().startswith(b"\xef\xbb\xbf")
            assert "action_ratio" not in path.read_text(encoding="utf-8-sig")
