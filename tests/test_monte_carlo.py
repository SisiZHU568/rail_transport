"""
test_monte_carlo.py

测试蒙特卡洛统计计算和配对随机种子机制。
"""

from types import SimpleNamespace

import pytest

from src.monte_carlo import (
    run_paired_monte_carlo,
    summarize_metric,
)


class FakeSimulator:
    """
    用于测试的简化仿真器。

    不执行真实轨道仿真，
    只返回由随机种子决定的指标。
    """

    def __init__(
        self,
        random_seed: int,
        offset: float,
    ) -> None:
        self.random_seed = random_seed
        self.offset = offset

    def run(self):
        """
        返回模拟的仿真汇总结果。
        """

        value = (
            float(self.random_seed)
            + self.offset
        )

        summary = SimpleNamespace(
            request_success_rate=(
                value / 10000.0
            ),
            sla_violation_rate=(
                1.0 - value / 10000.0
            ),
            average_successful_batch_delay_ms=(
                value
            ),
            p95_successful_batch_delay_ms=(
                value + 10.0
            ),
            average_active_memory_mb=(
                value + 20.0
            ),
            peak_active_memory_mb=(
                value + 30.0
            ),
            failover_batches=int(
                value % 5
            ),
            hot_failover_function_stages=int(
                value % 7
            ),
            cold_failover_function_stages=int(
                value % 9
            ),
            total_cold_backup_startup_delay_ms=(
                value + 40.0
            ),
        )

        return SimpleNamespace(
            summary=summary
        )


def test_metric_mean_and_sample_std() -> None:
    """
    数值 [1, 2, 3]：

        均值 = 2
        样本标准差 = 1
    """

    result = summarize_metric(
        [1.0, 2.0, 3.0],
        confidence_level=0.95,
    )

    assert result.count == 3
    assert result.mean == pytest.approx(2.0)
    assert result.std == pytest.approx(1.0)
    assert result.minimum == 1.0
    assert result.maximum == 3.0
    assert result.ci_low < result.mean
    assert result.ci_high > result.mean


def test_single_value_has_zero_width_interval() -> None:
    """
    只有一个样本时，标准差为0，
    置信区间上下界均等于该样本。
    """

    result = summarize_metric(
        [5.0]
    )

    assert result.mean == 5.0
    assert result.std == 0.0
    assert result.ci_low == 5.0
    assert result.ci_high == 5.0


def test_empty_metric_is_rejected() -> None:
    """
    空指标不能计算统计量。
    """

    with pytest.raises(ValueError):
        summarize_metric([])


def test_invalid_confidence_level_is_rejected() -> None:
    """
    置信水平必须位于 (0, 1)。
    """

    with pytest.raises(ValueError):
        summarize_metric(
            [1.0, 2.0],
            confidence_level=1.0,
        )


def test_all_scenarios_use_same_random_seeds() -> None:
    """
    配对实验中，每个方案都应使用：

        100、101、102
    """

    received_seeds: dict[
        str,
        list[int],
    ] = {
        "scenario_a": [],
        "scenario_b": [],
    }

    def build_a(seed: int) -> FakeSimulator:
        received_seeds["scenario_a"].append(
            seed
        )

        return FakeSimulator(
            random_seed=seed,
            offset=0.0,
        )

    def build_b(seed: int) -> FakeSimulator:
        received_seeds["scenario_b"].append(
            seed
        )

        return FakeSimulator(
            random_seed=seed,
            offset=1.0,
        )

    result = run_paired_monte_carlo(
        scenario_builders={
            "scenario_a": build_a,
            "scenario_b": build_b,
        },
        num_runs=3,
        base_seed=100,
    )

    assert received_seeds["scenario_a"] == [
        100,
        101,
        102,
    ]

    assert received_seeds["scenario_b"] == [
        100,
        101,
        102,
    ]

    assert len(result.run_records) == 6


def test_runner_generates_scenario_summaries() -> None:
    """
    每个方案都应生成独立统计汇总。
    """

    result = run_paired_monte_carlo(
        scenario_builders={
            "scenario_a": (
                lambda seed: FakeSimulator(
                    random_seed=seed,
                    offset=0.0,
                )
            ),
            "scenario_b": (
                lambda seed: FakeSimulator(
                    random_seed=seed,
                    offset=10.0,
                )
            ),
        },
        num_runs=3,
        base_seed=100,
    )

    assert len(
        result.scenario_summaries
    ) == 2

    summary_a = result.scenario_summaries[0]
    summary_b = result.scenario_summaries[1]

    assert summary_a.scenario_name == "scenario_a"
    assert summary_b.scenario_name == "scenario_b"

    assert (
        summary_b
        .average_successful_batch_delay_ms
        .mean
        >
        summary_a
        .average_successful_batch_delay_ms
        .mean
    )