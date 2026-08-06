"""
test_rl_reward.py

测试强化学习奖励函数。
"""

import pytest

from src.rl_reward import (
    RLRewardWeights,
    RLWindowMetrics,
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