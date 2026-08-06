"""
test_risk_aware_failure_process.py

测试高风险窗口随机故障过程。
"""

from src.config import load_config
from src.risk_aware_failure_process import (
    FailureRiskMultiplierWindow,
    WindowedMarkovFailureProcess,
    build_windowed_markov_failure_process,
)
from src.topology import build_linear_topology


def build_test_topology():
    """
    创建测试拓扑。
    """

    config = load_config(
        "configs/debug.yaml"
    )

    return build_linear_topology(config)


def test_probability_window_contains_boundaries() -> None:
    """
    起始和结束时隙均属于窗口。
    """

    window = FailureRiskMultiplierWindow(
        start_slot=10,
        end_slot=20,
        multiplier=5.0,
    )

    assert window.contains(9) is False
    assert window.contains(10) is True
    assert window.contains(20) is True
    assert window.contains(21) is False


def test_high_risk_multiplier_can_force_failure() -> None:
    """
    节点基础失效概率为0.2。

    时隙1的放大倍数为5，
    因而有效失效概率变成1.0。

    所有节点在时隙1均应发生局部失效。
    """

    topology = build_test_topology()

    domain_ids = {
        site.node.fault_domain
        for site in topology.sites
    }

    node_ids = {
        site.node.node_id
        for site in topology.sites
    }

    process = WindowedMarkovFailureProcess(
        topology=topology,
        domain_failure_probabilities={
            domain_id: 0.0
            for domain_id in domain_ids
        },
        domain_recovery_probabilities={
            domain_id: 1.0
            for domain_id in domain_ids
        },
        node_failure_probabilities={
            node_id: 0.2
            for node_id in node_ids
        },
        node_recovery_probabilities={
            node_id: 1.0
            for node_id in node_ids
        },
        risk_windows=[
            FailureRiskMultiplierWindow(
                start_slot=1,
                end_slot=1,
                multiplier=5.0,
            )
        ],
        random_seed=42,
    )

    state0 = process.state_for_slot(0)
    state1 = process.state_for_slot(1)

    assert all(
        state0.node_local_up.values()
    )

    assert not any(
        state1.node_local_up.values()
    )


def test_reset_reproduces_same_sequence() -> None:
    """
    reset后应复现完全相同的故障序列。
    """

    topology = build_test_topology()

    domain_ids = {
        site.node.fault_domain
        for site in topology.sites
    }

    node_ids = {
        site.node.node_id
        for site in topology.sites
    }

    process = WindowedMarkovFailureProcess(
        topology=topology,
        domain_failure_probabilities={
            domain_id: 0.1
            for domain_id in domain_ids
        },
        domain_recovery_probabilities={
            domain_id: 0.5
            for domain_id in domain_ids
        },
        node_failure_probabilities={
            node_id: 0.1
            for node_id in node_ids
        },
        node_recovery_probabilities={
            node_id: 0.5
            for node_id in node_ids
        },
        risk_windows=[
            FailureRiskMultiplierWindow(
                start_slot=2,
                end_slot=4,
                multiplier=3.0,
            )
        ],
        random_seed=123,
    )

    first_sequence = [
        process.state_for_slot(time_slot)
        for time_slot in range(6)
    ]

    process.reset()

    second_sequence = [
        process.state_for_slot(time_slot)
        for time_slot in range(6)
    ]

    assert first_sequence == second_sequence


def test_builder_accepts_seed_override() -> None:
    """
    构建函数应允许显式覆盖随机种子。
    """

    config = load_config(
        "configs/debug.yaml"
    )

    topology = build_linear_topology(
        config
    )

    process = (
        build_windowed_markov_failure_process(
            config=config,
            topology=topology,
            random_seed=9876,
        )
    )

    assert process.random_seed == 9876