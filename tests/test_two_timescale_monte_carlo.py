"""
test_two_timescale_monte_carlo.py

测试双时间尺度配对蒙特卡洛实验。
"""

from types import SimpleNamespace

import pytest

from src.two_timescale_monte_carlo import (
    run_two_timescale_monte_carlo,
)


class FakeTwoTimescaleSimulator:
    """
    不执行真实仿真的测试替身。
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
        返回由随机种子决定的模拟指标。
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
            cold_start_function_stages=int(
                value % 7
            ),
            total_cold_start_delay_ms=(
                value + 40.0
            ),
            slow_decision_updates=int(
                value % 5
            ),
            slow_mode_switches=int(
                value % 3
            ),
            total_request_delay_cost=(
                value + 50.0
            ),
            total_memory_cost=(
                value + 60.0
            ),
            total_cold_start_cost=(
                value + 70.0
            ),
            total_sla_penalty=(
                value + 80.0
            ),
            total_slow_control_cost=(
                value + 90.0
            ),
            total_system_cost=(
                value + 350.0
            ),
        )

        return SimpleNamespace(
            summary=summary
        )


def test_invalid_run_count_is_rejected() -> None:
    """
    运行次数必须大于0。
    """

    with pytest.raises(ValueError):
        run_two_timescale_monte_carlo(
            scenario_builders={
                "a": lambda seed:
                FakeTwoTimescaleSimulator(
                    seed,
                    0.0,
                )
            },
            num_runs=0,
            base_seed=100,
            confidence_level=0.95,
        )


def test_all_scenarios_receive_same_seeds() -> None:
    """
    两个方案都应使用100、101、102。
    """

    received_seeds = {
        "a": [],
        "b": [],
    }

    def build_a(
        seed: int,
    ) -> FakeTwoTimescaleSimulator:
        received_seeds["a"].append(seed)

        return FakeTwoTimescaleSimulator(
            random_seed=seed,
            offset=0.0,
        )

    def build_b(
        seed: int,
    ) -> FakeTwoTimescaleSimulator:
        received_seeds["b"].append(seed)

        return FakeTwoTimescaleSimulator(
            random_seed=seed,
            offset=1.0,
        )

    result = run_two_timescale_monte_carlo(
        scenario_builders={
            "a": build_a,
            "b": build_b,
        },
        num_runs=3,
        base_seed=100,
        confidence_level=0.95,
    )

    assert received_seeds["a"] == [
        100,
        101,
        102,
    ]

    assert received_seeds["b"] == [
        100,
        101,
        102,
    ]

    assert len(result.run_records) == 6


def test_scenario_summaries_are_created() -> None:
    """
    每种方案应生成独立统计汇总。
    """

    result = run_two_timescale_monte_carlo(
        scenario_builders={
            "a": lambda seed:
            FakeTwoTimescaleSimulator(
                seed,
                0.0,
            ),
            "b": lambda seed:
            FakeTwoTimescaleSimulator(
                seed,
                10.0,
            ),
        },
        num_runs=3,
        base_seed=100,
        confidence_level=0.95,
    )

    assert len(
        result.scenario_summaries
    ) == 2

    summary_a = result.scenario_summaries[0]
    summary_b = result.scenario_summaries[1]

    assert summary_a.scenario_name == "a"
    assert summary_b.scenario_name == "b"

    assert (
        summary_b.total_system_cost.mean
        >
        summary_a.total_system_cost.mean
    )


def test_summary_contains_confidence_interval() -> None:
    """
    多个不同样本应产生非零宽度置信区间。
    """

    result = run_two_timescale_monte_carlo(
        scenario_builders={
            "a": lambda seed:
            FakeTwoTimescaleSimulator(
                seed,
                0.0,
            ),
        },
        num_runs=3,
        base_seed=100,
        confidence_level=0.95,
    )

    metric = (
        result.scenario_summaries[0]
        .total_system_cost
    )

    assert metric.count == 3
    assert metric.std > 0
    assert metric.ci_low < metric.mean
    assert metric.ci_high > metric.mean