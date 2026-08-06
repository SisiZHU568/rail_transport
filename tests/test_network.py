"""
test_network.py

测试轨旁 MEC 网络的跳数、路径和传输时延。
"""

import pytest

from src.config import load_config
from src.network import build_linear_mec_network
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