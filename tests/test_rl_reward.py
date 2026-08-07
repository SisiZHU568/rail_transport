"""
test_rl_reward.py

测试强化学习奖励函数。
"""

import math

import pytest

from src.rl_reward import (
    RLCostRewardBreakdown,
    RLRewardWeights,
    RLWindowCostMetrics,
    RLWindowMetrics,
    calculate_cost_reward,
    calculate_rl_reward,
)


def build_weights() -> RLRewardWeights:
    return RLRewardWeights(
        delay=0.20,
        memory=0.20,
        cold_start=0.15,
        sla_violation=0.40,
        reconfiguration=0.05,
    )


def build_good_metrics() -> RLWindowMetrics:
    """
    构建表现较好的窗口。
    """

    return RLWindowMetrics(
        total_requests=10,
        successful_requests=10,
        request_batches=5,
        successful_batches=5,
        failed_batches=0,
        deadline_violations=0,
        sla_violations=0,
        request_success_rate=1.0,
        sla_violation_rate=0.0,
        average_successful_delay_ms=100.0,
        total_cold_start_delay_ms=0.0,
        average_active_memory_mb=100.0,
        total_active_memory_mb_seconds=1000.0,
        failover_function_stages=0,
        cold_start_function_stages=0,
        reconfigured_function_stages=0,
    )


def build_bad_metrics() -> RLWindowMetrics:
    """
    构建时延、内存和SLA均较差的窗口。
    """

    return RLWindowMetrics(
        total_requests=10,
        successful_requests=5,
        request_batches=5,
        successful_batches=2,
        failed_batches=3,
        deadline_violations=1,
        sla_violations=4,
        request_success_rate=0.5,
        sla_violation_rate=0.8,
        average_successful_delay_ms=1400.0,
        total_cold_start_delay_ms=3000.0,
        average_active_memory_mb=3000.0,
        total_active_memory_mb_seconds=30000.0,
        failover_function_stages=6,
        cold_start_function_stages=6,
        reconfigured_function_stages=3,
    )


def test_negative_weight_is_rejected() -> None:
    """
    奖励权重不能为负。
    """

    with pytest.raises(ValueError):
        RLRewardWeights(
            delay=-1.0,
            memory=0.0,
            cold_start=0.0,
            sla_violation=0.0,
            reconfiguration=0.0,
        )


def test_better_window_has_higher_reward() -> None:
    """
    好窗口的奖励应更接近0。
    """

    good = calculate_rl_reward(
        metrics=build_good_metrics(),
        deadline_ms=1500.0,
        maximum_active_memory_mb=3072.0,
        maximum_cold_start_delay_ms_per_batch=1600.0,
        function_count=3,
        weights=build_weights(),
    )

    bad = calculate_rl_reward(
        metrics=build_bad_metrics(),
        deadline_ms=1500.0,
        maximum_active_memory_mb=3072.0,
        maximum_cold_start_delay_ms_per_batch=1600.0,
        function_count=3,
        weights=build_weights(),
    )

    assert good.reward > bad.reward


def test_reward_is_negative_normalized_cost() -> None:
    """
    奖励应等于归一化成本的负数。
    """

    result = calculate_rl_reward(
        metrics=build_good_metrics(),
        deadline_ms=1500.0,
        maximum_active_memory_mb=3072.0,
        maximum_cold_start_delay_ms_per_batch=1600.0,
        function_count=3,
        weights=build_weights(),
    )

    assert result.reward == pytest.approx(
        -result.weighted_cost
    )

    assert -1.0 <= result.reward <= 0.0


def test_reward_is_normalized_cost_plus_violation() -> None:
    """新奖励只包含归一化总成本和一个违约项。"""

    result = calculate_cost_reward(
        metrics=RLWindowCostMetrics(
            run_cost=20.0,
            route_cost=10.0,
            cold_start_cost=5.0,
            has_violation=True,
        ),
        maximum_window_cost=100.0,
    )

    assert isinstance(
        result,
        RLCostRewardBreakdown,
    )
    assert result.run_cost == 20.0
    assert result.route_cost == 10.0
    assert result.cold_start_cost == 5.0
    assert result.total_cost == 35.0
    assert result.normalized_total_cost == (
        pytest.approx(0.35)
    )
    assert result.violation_penalty == 1.0
    assert result.reward == pytest.approx(-1.35)


def test_no_request_warm_cost_is_still_charged() -> None:
    """没有请求时，主动保温产生的运行成本仍需计入奖励。"""

    result = calculate_cost_reward(
        metrics=RLWindowCostMetrics(
            run_cost=12.0,
            route_cost=0.0,
            cold_start_cost=0.0,
            has_violation=False,
        ),
        maximum_window_cost=100.0,
    )

    assert result.reward == pytest.approx(-0.12)


def test_total_cost_is_clipped_before_penalty() -> None:
    """极端成本截断到 1 后，再叠加独立的违约惩罚。"""

    result = calculate_cost_reward(
        metrics=RLWindowCostMetrics(
            run_cost=200.0,
            route_cost=0.0,
            cold_start_cost=0.0,
            has_violation=True,
        ),
        maximum_window_cost=100.0,
    )

    assert result.total_cost == 200.0
    assert result.normalized_total_cost == 1.0
    assert result.reward == -2.0


@pytest.mark.parametrize(
    "invalid_cost",
    [-1.0, math.inf, math.nan],
)
def test_invalid_window_cost_is_rejected(
    invalid_cost: float,
) -> None:
    """任何成本分量都必须是非负有限值。"""

    with pytest.raises(ValueError, match="窗口成本"):
        RLWindowCostMetrics(
            run_cost=invalid_cost,
            route_cost=0.0,
            cold_start_cost=0.0,
            has_violation=False,
        )


@pytest.mark.parametrize(
    "maximum_window_cost",
    [0.0, -1.0, math.inf, math.nan],
)
def test_invalid_maximum_window_cost_is_rejected(
    maximum_window_cost: float,
) -> None:
    """归一化分母必须是正有限值。"""

    with pytest.raises(
        ValueError,
        match="窗口最大成本",
    ):
        calculate_cost_reward(
            metrics=RLWindowCostMetrics(
                run_cost=1.0,
                route_cost=0.0,
                cold_start_cost=0.0,
                has_violation=False,
            ),
            maximum_window_cost=(
                maximum_window_cost
            ),
        )
