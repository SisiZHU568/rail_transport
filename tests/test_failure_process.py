"""
test_failure_process.py

测试确定性故障过程和马尔可夫随机故障过程。
"""

import math

import pytest

from src.config import load_config
from src.failure_process import (
    CTMCTransition,
    MarkovFailureProcess,
    ScriptedFailureProcess,
    build_markov_failure_process,
)
from src.orchestration_config import CTMCRates
from src.topology import build_linear_topology


def build_test_topology():
    """
    创建测试铁路拓扑。
    """

    config = load_config("configs/debug.yaml")
    return build_linear_topology(config)


def test_scripted_node_failure() -> None:
    """
    时隙3令 MEC-2 局部失效。
    """

    topology = build_test_topology()

    process = ScriptedFailureProcess(
        topology=topology,
        down_nodes_by_slot={
            3: {1},
        },
    )

    state2 = process.state_for_slot(2)
    state3 = process.state_for_slot(3)

    assert state2.is_node_operational(
        node_id=1,
        topology=topology,
    ) is True

    assert state3.is_node_operational(
        node_id=1,
        topology=topology,
    ) is False


def test_domain_failure_disables_all_domain_nodes() -> None:
    """
    MEC-1和MEC-2都属于故障域0。

    故障域0失效时，两者都不可用。
    """

    topology = build_test_topology()

    process = ScriptedFailureProcess(
        topology=topology,
        down_domains_by_slot={
            1: {0},
        },
    )

    state = process.state_for_slot(1)

    assert state.is_node_operational(
        node_id=0,
        topology=topology,
    ) is False

    assert state.is_node_operational(
        node_id=1,
        topology=topology,
    ) is False

    assert state.is_node_operational(
        node_id=2,
        topology=topology,
    ) is True


def test_markov_failure_and_recovery() -> None:
    """零失效率恒正常，零恢复率恒失效。"""

    topology = build_test_topology()

    domain_ids = {
        node.fault_domain
        for node in topology.compute_nodes
    }

    node_ids = {
        node.node_id
        for node in topology.compute_nodes
    }

    always_up = MarkovFailureProcess(
        topology=topology,
        domain_rates={
            domain_id: CTMCRates(0.0, 1.0)
            for domain_id in domain_ids
        },
        node_rates={
            node_id: CTMCRates(0.0, 1.0)
            for node_id in node_ids
        },
        slot_seconds=1.0,
        random_seed=42,
    )
    always_down = MarkovFailureProcess(
        topology=topology,
        domain_rates={
            domain_id: CTMCRates(1.0, 0.0)
            for domain_id in domain_ids
        },
        node_rates={
            node_id: CTMCRates(1.0, 0.0)
            for node_id in node_ids
        },
        slot_seconds=1.0,
        random_seed=42,
    )

    assert all(always_up.state_for_slot(0).domain_up.values())
    assert not any(always_down.state_for_slot(0).domain_up.values())


def test_markov_reset_reproduces_same_sequence() -> None:
    """
    使用相同随机种子并调用 reset 后，
    应得到完全相同的故障序列。
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

    process = MarkovFailureProcess(
        topology=topology,
        domain_rates={
            domain_id: CTMCRates(0.2, 0.5)
            for domain_id in domain_ids
        },
        node_rates={
            node_id: CTMCRates(0.2, 0.5)
            for node_id in node_ids
        },
        slot_seconds=1.0,
        random_seed=123,
    )

    first_sequence = [
        process.state_for_slot(time_slot)
        for time_slot in range(5)
    ]

    process.reset()

    second_sequence = [
        process.state_for_slot(time_slot)
        for time_slot in range(5)
    ]

    assert first_sequence == second_sequence


def test_markov_builder_state_contains_every_compute_node() -> None:
    """配置构建的随机故障状态必须包含轨旁 MEC 和中心云。"""

    config = load_config("configs/debug.yaml")
    topology = build_linear_topology(config)
    process = build_markov_failure_process(
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


def test_scripted_failure_process_can_fail_cloud_node() -> None:
    """测试场景可以显式注入云节点故障，便于验证故障切换。"""

    topology = build_test_topology()
    cloud_node = topology.cloud_node
    assert cloud_node is not None

    process = ScriptedFailureProcess(
        topology=topology,
        down_nodes_by_slot={
            1: {cloud_node.node_id},
        },
    )

    state = process.state_for_slot(1)

    assert state.is_node_operational(
        node_id=cloud_node.node_id,
        topology=topology,
    ) is False


def test_exact_ctmc_discretization_preserves_stationary_availability() -> None:
    """矩阵指数离散后必须保留连续时间链的稳态可用度。"""

    rates = CTMCRates(0.2, 0.8)
    transition = CTMCTransition.from_rates(rates, slot_seconds=3.0)
    expected_change = 1.0 - math.exp(-3.0)

    assert transition.failure_probability == pytest.approx(0.2 * expected_change)
    assert transition.recovery_probability == pytest.approx(0.8 * expected_change)
    assert transition.recovery_probability / (
        transition.failure_probability + transition.recovery_probability
    ) == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("rates", "expected_up"),
    ((CTMCRates(0.0, 1.0), True), (CTMCRates(1.0, 0.0), False)),
)
def test_stationary_initialization_handles_zero_rate_boundaries(
    rates: CTMCRates,
    expected_up: bool,
) -> None:
    """λ=0 与 μ=0 的稳态初始状态没有随机歧义。"""

    assert MarkovFailureProcess.stationary_initial_state(rates, draw=0.5) is expected_up


def test_effective_events_follow_domain_and_node_combined_state() -> None:
    """故障域恢复时，局部仍故障的节点不能出现在恢复集合。"""

    topology = build_test_topology()
    process = ScriptedFailureProcess(
        topology=topology,
        down_domains_by_slot={0: {0}},
        down_nodes_by_slot={0: {0}, 1: {0}},
    )

    first = process.state_for_slot(0)
    second = process.state_for_slot(1)

    assert first.version + 1 == second.version
    assert 0 not in second.newly_available_node_ids
    assert 1 in second.newly_available_node_ids
    assert second.is_node_operational(0, topology) is False


def test_failure_snapshot_mappings_are_immutable() -> None:
    """调用方不能修改故障层已经发布的只读快照。"""

    snapshot = ScriptedFailureProcess(build_test_topology()).state_for_slot(0)

    with pytest.raises(TypeError):
        snapshot.effective_node_up[0] = False  # type: ignore[index]
