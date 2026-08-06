"""
test_failure_process.py

测试确定性故障过程和马尔可夫随机故障过程。
"""

from src.config import load_config
from src.failure_process import (
    MarkovFailureProcess,
    ScriptedFailureProcess,
)
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
    """
    失效概率和恢复概率都设为1。

    时隙0：初始正常；
    时隙1：全部失效；
    时隙2：全部恢复。
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

    process = MarkovFailureProcess(
        topology=topology,
        domain_failure_probabilities={
            domain_id: 1.0
            for domain_id in domain_ids
        },
        domain_recovery_probabilities={
            domain_id: 1.0
            for domain_id in domain_ids
        },
        node_failure_probabilities={
            node_id: 1.0
            for node_id in node_ids
        },
        node_recovery_probabilities={
            node_id: 1.0
            for node_id in node_ids
        },
        random_seed=42,
    )

    state0 = process.state_for_slot(0)
    state1 = process.state_for_slot(1)
    state2 = process.state_for_slot(2)

    assert all(state0.domain_up.values())
    assert all(state0.node_local_up.values())

    assert not any(state1.domain_up.values())
    assert not any(
        state1.node_local_up.values()
    )

    assert all(state2.domain_up.values())
    assert all(state2.node_local_up.values())


def test_markov_reset_reproduces_same_sequence() -> None:
    """
    使用相同随机种子并调用 reset 后，
    应得到完全相同的故障序列。
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

    process = MarkovFailureProcess(
        topology=topology,
        domain_failure_probabilities={
            domain_id: 0.2
            for domain_id in domain_ids
        },
        domain_recovery_probabilities={
            domain_id: 0.5
            for domain_id in domain_ids
        },
        node_failure_probabilities={
            node_id: 0.2
            for node_id in node_ids
        },
        node_recovery_probabilities={
            node_id: 0.5
            for node_id in node_ids
        },
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