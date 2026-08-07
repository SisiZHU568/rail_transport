"""
test_two_timescale_monte_carlo.py

测试双时间尺度配对蒙特卡洛实验。
"""

from types import SimpleNamespace

import pytest

from run_two_timescale_monte_carlo import (
    build_simulator as build_monte_carlo_simulator,
)
from src.config import load_config
from src.fast_slot_executor import FastSlotExecutor
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
            fast_repair_attempts=4,
            fast_repair_successes=3,
            fast_repair_failures=1,
            fast_repair_success_rate=0.75,
            constraint_rejected_batches=int(
                value % 3
            ),
            cloud_used_slots=2,
            cloud_usage_rate=0.5,
            total_run_cost=(
                value + 45.0
            ),
            total_route_cost=(
                value + 46.0
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


def test_monte_carlo_script_builder_uses_shared_executor() -> None:
    """真实多种子实验入口也必须完成共享执行器迁移。"""

    simulator = build_monte_carlo_simulator(
        config=load_config("configs/debug.yaml"),
        scenario_name="fixed_single",
        random_seed=123,
    )

    assert isinstance(
        simulator.fast_slot_executor,
        FastSlotExecutor,
    )
    result = simulator.run()
    assert len(result.records) > 0
    assert result.summary.total_system_cost == pytest.approx(
        result.summary.total_run_cost
        + result.summary.total_route_cost
        + result.summary.total_cold_start_cost
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

    # 修复统计必须保留在每次实验的原始记录中。
    assert result.run_records[0].fast_repair_attempts == 4
    assert (
        result.run_records[0]
        .fast_repair_success_rate
        == pytest.approx(0.75)
    )
    assert result.run_records[0].cloud_usage_rate == pytest.approx(
        0.5
    )
    assert result.run_records[0].total_run_cost == pytest.approx(
        145.0
    )
    assert result.run_records[0].total_route_cost == pytest.approx(
        146.0
    )


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

    # 两个关键修复指标还必须进入每种方案的统计对象。
    assert summary_a.fast_repair_success_rate.count == 3
    assert summary_a.fast_repair_success_rate.mean == pytest.approx(
        0.75
    )
    assert summary_a.constraint_rejected_batches.count == 3
    assert summary_a.cloud_usage_rate.mean == pytest.approx(0.5)
    assert summary_a.total_run_cost.count == 3
    assert summary_a.total_route_cost.count == 3


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
