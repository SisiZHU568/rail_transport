"""
test_risk_aware_failure_process.py

测试高风险窗口随机故障过程。
"""

from src.config import load_config
from src.orchestration_config import CTMCRates
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
    """高风险窗口只放大连续时间失效率，并重新精确离散。"""

    topology = build_test_topology()

    domain_ids = {
        node.fault_domain
        for node in topology.compute_nodes
    }

    node_ids = {
        node.node_id
        for node in topology.compute_nodes
    }

    process = WindowedMarkovFailureProcess(
        topology=topology,
        domain_rates={
            domain_id: CTMCRates(0.1, 0.5)
            for domain_id in domain_ids
        },
        node_rates={
            node_id: CTMCRates(0.2, 0.5)
            for node_id in node_ids
        },
        slot_seconds=1.0,
        risk_windows=[
            FailureRiskMultiplierWindow(
                start_slot=1,
                end_slot=1,
                multiplier=5.0,
            )
        ],
        random_seed=42,
    )

    base = process.transition_for_slot("node", 0, time_slot=0)
    high = process.transition_for_slot("node", 0, time_slot=1)

    assert high.failure_probability > base.failure_probability
    assert high.recovery_probability < base.recovery_probability


def test_reset_reproduces_same_sequence() -> None:
    """
    reset后应复现完全相同的故障序列。
    """

    topology = build_test_topology()

    domain_ids = {
        node.fault_domain
        for node in topology.compute_nodes
    }

    node_ids = {
        node.node_id
        for node in topology.compute_nodes
    }

    process = WindowedMarkovFailureProcess(
        topology=topology,
        domain_rates={
            domain_id: CTMCRates(0.1, 0.5)
            for domain_id in domain_ids
        },
        node_rates={
            node_id: CTMCRates(0.1, 0.5)
            for node_id in node_ids
        },
        slot_seconds=1.0,
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


def test_failure_state_contains_every_compute_node() -> None:
    """高风险随机故障状态必须同时描述轨旁 MEC 和中心云。"""

    config = load_config("configs/debug.yaml")
    topology = build_linear_topology(config)
    process = build_windowed_markov_failure_process(
        config=config,
        topology=topology,
        random_seed=123,
    )

    process.reset()
    state = process.state_for_slot(0)

    assert set(state.node_local_up) == {
        node.node_id
        for node in topology.compute_nodes
    }
    assert set(state.domain_up) == {
        node.fault_domain
        for node in topology.compute_nodes
    }
    assert topology.cloud_node is not None
    assert isinstance(
        state.is_node_operational(
            topology.cloud_node.node_id,
            topology,
        ),
        bool,
    )
