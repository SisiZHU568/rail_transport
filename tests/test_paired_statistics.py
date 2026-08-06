"""
test_paired_statistics.py

测试配对统计分析模块。
"""

import numpy as np
import pytest

from src.paired_statistics import (
    bootstrap_mean_confidence_interval,
    build_paired_comparison_rows,
    calculate_improvements,
    holm_adjust_p_values,
    paired_effect_size_dz,
    paired_sign_flip_p_value,
)


def test_higher_is_better_improvement() -> None:
    """
    Reward等越大越好的指标。
    """

    improvements = calculate_improvements(
        reference_values=[
            3.0,
            5.0,
        ],
        baseline_values=[
            2.0,
            4.0,
        ],
        direction="higher",
    )

    assert np.allclose(
        improvements,
        [1.0, 1.0],
    )


def test_lower_is_better_improvement() -> None:
    """
    时延等越小越好的指标。
    """

    improvements = calculate_improvements(
        reference_values=[
            80.0,
            90.0,
        ],
        baseline_values=[
            100.0,
            120.0,
        ],
        direction="lower",
    )

    assert np.allclose(
        improvements,
        [20.0, 30.0],
    )


def test_bootstrap_constant_difference() -> None:
    """
    所有差值相同时，置信区间应等于该常数。
    """

    random_generator = (
        np.random.default_rng(42)
    )

    result = (
        bootstrap_mean_confidence_interval(
            values=[
                2.0,
                2.0,
                2.0,
                2.0,
            ],
            confidence_level=0.95,
            num_samples=1000,
            random_generator=(
                random_generator
            ),
        )
    )

    assert result.mean == pytest.approx(
        2.0
    )

    assert result.low == pytest.approx(
        2.0
    )

    assert result.high == pytest.approx(
        2.0
    )


def test_exact_sign_flip_p_value() -> None:
    """
    三个完全相同的正差值：

    8种符号组合中，只有全正和全负的绝对均值
    不小于观测值，因此双侧p值为2/8=0.25。
    """

    random_generator = (
        np.random.default_rng(42)
    )

    p_value = paired_sign_flip_p_value(
        paired_differences=[
            1.0,
            1.0,
            1.0,
        ],
        num_samples=1000,
        random_generator=(
            random_generator
        ),
    )

    assert p_value == pytest.approx(
        0.25
    )


def test_effect_size() -> None:
    effect_size = paired_effect_size_dz(
        [
            1.0,
            2.0,
            3.0,
        ]
    )

    assert effect_size == pytest.approx(
        2.0
    )


def test_holm_adjustment() -> None:
    adjusted = holm_adjust_p_values(
        [
            0.01,
            0.03,
            0.04,
        ]
    )

    assert adjusted[0] == pytest.approx(
        0.03
    )

    assert adjusted[1] == pytest.approx(
        0.06
    )

    assert adjusted[2] == pytest.approx(
        0.06
    )


def test_build_paired_rows() -> None:
    records = []

    for seed in [
        100,
        101,
        102,
    ]:
        records.append(
            {
                "policy_name": (
                    "Fixed-COLD"
                ),
                "episode_seed": seed,
                "episode_reward": -3.0,
                "request_success_rate": 0.98,
                "sla_violation_rate": 0.10,
                "average_delay_ms": 200.0,
                "average_memory_mb": 1500.0,
                "total_cold_start_delay_ms": 1000.0,
                "reconfiguration_count": 2.0,
            }
        )

        records.append(
            {
                "policy_name": (
                    "Double-DQN"
                ),
                "episode_seed": seed,
                "episode_reward": -2.0,
                "request_success_rate": 0.99,
                "sla_violation_rate": 0.05,
                "average_delay_ms": 150.0,
                "average_memory_mb": 1600.0,
                "total_cold_start_delay_ms": 500.0,
                "reconfiguration_count": 1.0,
            }
        )

    rows = build_paired_comparison_rows(
        records=records,
        reference_policy="Double-DQN",
        baseline_policies=[
            "Fixed-COLD"
        ],
        metric_names=[
            "episode_reward",
            "average_delay_ms",
        ],
        bootstrap_samples=1000,
        permutation_samples=1000,
        random_seed=42,
    )

    assert len(rows) == 2

    reward_row = rows[0]
    delay_row = rows[1]

    assert reward_row[
        "improvement_mean"
    ] == pytest.approx(1.0)

    assert delay_row[
        "improvement_mean"
    ] == pytest.approx(50.0)

    assert reward_row[
        "win_count"
    ] == 3

    assert reward_row[
        "loss_count"
    ] == 0