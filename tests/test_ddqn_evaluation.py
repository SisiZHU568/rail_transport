"""
test_ddqn_evaluation.py

测试Double DQN独立评估工具。
"""

from dataclasses import dataclass

import numpy as np
import pytest

from src.ddqn_evaluation import (
    build_action_distribution_rows,
    build_fixed_policy,
    build_rule_based_policy,
    build_summary_rows,
    evaluate_policy,
    summarize_values,
)


@dataclass
class DummyMetrics:
    total_requests: int = 10
    successful_request_count: int = 8

    request_batch_count: int = 5
    sla_violation_batch_count: int = 1

    successful_batch_count: int = 4
    total_successful_delay_ms: float = 400.0

    average_active_memory_mb: float = 1536.0
    total_cold_start_delay_ms: float = 300.0
    reconfiguration_count: int = 2

    @property
    def request_success_rate(self) -> float:
        return (
            self.successful_request_count
            / self.total_requests
        )

    @property
    def sla_violation_rate(self) -> float:
        return (
            self.sla_violation_batch_count
            / self.request_batch_count
        )

    @property
    def average_successful_delay_ms(
        self,
    ) -> float:
        return (
            self.total_successful_delay_ms
            / self.successful_batch_count
        )


class DummyEnvironment:
    """
    每个Episode只有一个慢决策。
    """

    state_dim = 15
    action_count = 4

    def reset(
        self,
        seed: int,
    ):
        self.seed = seed

        state = np.zeros(
            self.state_dim,
            dtype=np.float32,
        )

        info = {
            "predicted_failure_risk": 0.20,
        }

        return state, info

    def step(
        self,
        action: int,
    ):
        next_state = np.ones(
            self.state_dim,
            dtype=np.float32,
        )

        reward = -0.25
        terminated = True
        truncated = False

        info = {
            "metrics": DummyMetrics(),
            "window_length": 10,
            "selected_action": action,
        }

        return (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        )


def test_confidence_summary_mean() -> None:
    summary = summarize_values(
        [1.0, 2.0, 3.0],
        confidence_level=0.95,
    )

    assert summary.sample_count == 3
    assert summary.mean == pytest.approx(
        2.0
    )

    assert (
        summary.confidence_interval_low
        < summary.mean
    )

    assert (
        summary.confidence_interval_high
        > summary.mean
    )


def test_fixed_policy_returns_action() -> None:
    policy = build_fixed_policy(
        action=2
    )

    action = policy(
        np.zeros(
            15,
            dtype=np.float32,
        ),
        {},
    )

    assert action == 2


def test_rule_based_policy() -> None:
    policy = build_rule_based_policy(
        high_risk_threshold=0.10
    )

    state = np.zeros(
        15,
        dtype=np.float32,
    )

    high_risk_action = policy(
        state,
        {
            "predicted_failure_risk": 0.20,
        },
    )

    low_risk_action = policy(
        state,
        {
            "predicted_failure_risk": 0.02,
        },
    )

    assert high_risk_action == 2
    assert low_risk_action == 3


def test_evaluate_policy_collects_metrics() -> None:
    environment = DummyEnvironment()

    records = evaluate_policy(
        environment=environment,
        policy_name="Fixed-HOT",
        policy=build_fixed_policy(2),
        episode_seeds=[100, 101],
    )

    assert len(records) == 2

    first = records[0]

    assert first.episode_reward == pytest.approx(
        -0.25
    )

    assert first.request_success_rate == pytest.approx(
        0.8
    )

    assert first.sla_violation_rate == pytest.approx(
        0.2
    )

    assert first.average_delay_ms == pytest.approx(
        100.0
    )

    assert first.average_memory_mb == pytest.approx(
        1536.0
    )

    assert first.hot_count == 1


def test_summary_and_action_rows() -> None:
    environment = DummyEnvironment()

    records = evaluate_policy(
        environment=environment,
        policy_name="Fixed-HOT",
        policy=build_fixed_policy(2),
        episode_seeds=[100, 101],
    )

    summary_rows = build_summary_rows(
        records=records,
        confidence_level=0.95,
    )

    action_rows = (
        build_action_distribution_rows(
            records
        )
    )

    assert len(summary_rows) == 7
    assert len(action_rows) == 1

    assert action_rows[0][
        "hot_ratio"
    ] == pytest.approx(1.0)

    assert action_rows[0][
        "single_ratio"
    ] == pytest.approx(0.0)