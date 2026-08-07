"""
test_network.py

测试轨旁 MEC 网络的跳数、路径和传输时延。
"""

import pytest

from src.config import load_config
from src.network import (
    build_hybrid_rail_network,
    build_linear_mec_network,
)
from src.topology import build_linear_topology


def build_test_network():
    """
    创建供测试使用的线性 MEC 网络。
    """

    config = load_config("configs/debug.yaml")
    topology = build_linear_topology(config)

    return build_linear_mec_network(
        config=config,
        topology=topology,
    )


def build_test_hybrid_network():
    """创建包含轨旁 MEC 和中心云的测试网络。"""

    config = load_config("configs/debug.yaml")
    topology = build_linear_topology(config)
    return build_hybrid_rail_network(
        config=config,
        topology=topology,
    )


def test_network_node_count_and_path() -> None:
    """
    测试网络节点数量和路径查询。
    """

    network = build_test_network()

    assert network.node_count == 5

    assert network.path_node_ids(
        source_node_id=0,
        destination_node_id=2,
    ) == [0, 1, 2]

    assert network.path_node_ids(
        source_node_id=3,
        destination_node_id=1,
    ) == [3, 2, 1]


def test_same_node_has_zero_transfer_delay() -> None:
    """
    数据不跨 MEC 时，传输时延应为0。
    """

    network = build_test_network()

    delay_ms = network.transfer_delay_ms(
        data_size_mb=2.0,
        source_node_id=0,
        destination_node_id=0,
    )

    assert delay_ms == 0.0


def test_adjacent_mec_transfer_delay() -> None:
    """
    2 MB数据经过1跳：

        发送时延 = 2 × 8 / 1000 × 1000 = 16 ms
        传播时延 = 1 × 2 = 2 ms
        总时延 = 18 ms
    """

    network = build_test_network()

    delay_ms = network.transfer_delay_ms(
        data_size_mb=2.0,
        source_node_id=0,
        destination_node_id=1,
    )

    assert delay_ms == pytest.approx(18.0)


def test_two_hop_transfer_delay() -> None:
    """
    2 MB数据经过2跳：

        发送时延 = 16 ms
        传播时延 = 2 × 2 = 4 ms
        总时延 = 20 ms
    """

    network = build_test_network()

    delay_ms = network.transfer_delay_ms(
        data_size_mb=2.0,
        source_node_id=0,
        destination_node_id=2,
    )

    assert delay_ms == pytest.approx(20.0)


def test_edge_transfer_cost_uses_hop_count() -> None:
    """轨旁传输成本等于数据量、跳数和每跳单价的乘积。"""

    network = build_test_network()

    cost = network.transfer_cost(
        data_size_mb=2.0,
        source_node_id=0,
        destination_node_id=2,
    )

    assert cost == pytest.approx(0.04)


def test_edge_to_cloud_uses_backhaul_parameters() -> None:
    """轨旁到云端应使用云回传带宽、传播时延和流量单价。"""

    network = build_test_hybrid_network()
    delay = network.transfer_delay_ms(
        data_size_mb=2.0,
        source_node_id=0,
        destination_node_id=5,
    )
    cost = network.transfer_cost(
        data_size_mb=2.0,
        source_node_id=0,
        destination_node_id=5,
    )

    assert delay == pytest.approx(72.0)
    assert cost == pytest.approx(0.4)


def test_cloud_same_node_transfer_has_zero_delay_and_cost() -> None:
    """数据已经位于中心云时，不应重复产生回传开销。"""

    network = build_test_hybrid_network()

    assert network.transfer_delay_ms(
        data_size_mb=2.0,
        source_node_id=5,
        destination_node_id=5,
    ) == 0.0
    assert network.transfer_cost(
        data_size_mb=2.0,
        source_node_id=5,
        destination_node_id=5,
    ) == 0.0


def test_hybrid_network_rejects_unknown_node() -> None:
    """未知节点不能被误当作中心云或轨旁节点。"""

    network = build_test_hybrid_network()

    with pytest.raises(KeyError):
        network.transfer_delay_ms(
            data_size_mb=1.0,
            source_node_id=0,
            destination_node_id=999,
        )
