"""
test_topology.py

测试直线铁路拓扑是否正确构建，
以及 MEC 接入和切换逻辑是否正确。
"""

import pytest

from src.config import load_config
from src.entities import (
    EdgeNode,
    NodeType,
)
from src.topology import (
    LinearRailTopology,
    build_linear_topology,
)


def test_build_linear_topology() -> None:
    """
    测试能否根据 debug.yaml 创建 5 个轨旁 MEC。
    """

    config = load_config("configs/debug.yaml")
    topology = build_linear_topology(config)

    assert topology.mec_count == 5
    assert topology.route_start_m == 0.0
    assert topology.route_end_m == 8000.0


def test_mec_positions() -> None:
    """
    测试各 MEC 的位置是否符合 2000 米间距。
    """

    config = load_config("configs/debug.yaml")
    topology = build_linear_topology(config)

    positions = [
        site.position_m
        for site in topology.sites
    ]

    assert positions == [
        0.0,
        2000.0,
        4000.0,
        6000.0,
        8000.0,
    ]


def test_serving_mec_before_and_after_handover() -> None:
    """
    测试 MEC-1 与 MEC-2 之间的接入切换。

    二者切换边界是 1000 米：

    999 米应连接 MEC-1；
    1001 米应连接 MEC-2。
    """

    config = load_config("configs/debug.yaml")
    topology = build_linear_topology(config)

    assert topology.serving_mec(999.0) == 0
    assert topology.serving_mec(1001.0) == 1


def test_next_mec() -> None:
    """
    测试下一个 MEC 的查询结果。
    """

    config = load_config("configs/debug.yaml")
    topology = build_linear_topology(config)

    assert topology.next_mec(0) == 1
    assert topology.next_mec(1) == 2
    assert topology.next_mec(3) == 4

    # MEC-5 已经是最后一个 MEC。
    assert topology.next_mec(4) is None


def test_remaining_dwell_time() -> None:
    """
    测试剩余驻留时间计算。

    列车位于 500 米；
    MEC-1 与 MEC-2 的切换边界为 1000 米；
    剩余距离为 500 米；
    速度为 100 m/s。

    剩余驻留时间应为：

        500 / 100 = 5 秒。
    """

    config = load_config("configs/debug.yaml")
    topology = build_linear_topology(config)

    dwell_time = topology.remaining_dwell_time_s(
        train_position_m=500.0,
        train_speed_mps=100.0,
        current_node_id=0,
    )

    assert dwell_time == pytest.approx(5.0)


def test_cloud_is_compute_node_not_trackside_site() -> None:
    """中心云参与计算部署，但不能参与列车接入和线路排序。"""

    config = load_config("configs/debug.yaml")
    topology = build_linear_topology(config)

    assert topology.mec_count == 5
    assert len(topology.sites) == 5
    assert len(topology.compute_nodes) == 6
    assert topology.cloud_node is not None
    assert (
        topology.cloud_node.node_type
        is NodeType.CLOUD
    )
    assert (
        topology.get_node(
            topology.cloud_node.node_id
        )
        is topology.cloud_node
    )

    with pytest.raises(KeyError):
        topology.get_site(
            topology.cloud_node.node_id
        )


def test_cloud_can_be_disabled_without_changing_trackside_route() -> None:
    """关闭云节点时仍保留原来的五 MEC 线路端点。"""

    config = load_config("configs/debug.yaml")
    config["topology"]["include_cloud"] = False

    topology = build_linear_topology(config)

    assert topology.cloud_node is None
    assert len(topology.compute_nodes) == 5
    assert topology.route_start_m == 0.0
    assert topology.route_end_m == 8000.0


def test_cloud_node_must_have_cloud_type() -> None:
    """防止把普通边缘节点误注册为中心云。"""

    config = load_config("configs/debug.yaml")
    trackside_only = build_linear_topology(
        {
            **config,
            "topology": {
                **config["topology"],
                "include_cloud": False,
            },
        }
    )
    invalid_cloud = EdgeNode(
        node_id=5,
        name="错误云节点",
        node_type=NodeType.TRACKSIDE,
        cpu_capacity=100.0,
        memory_capacity_mb=1000.0,
        reliability=0.99,
        fault_domain=3,
    )

    with pytest.raises(ValueError, match="中心云"):
        LinearRailTopology(
            sites=list(trackside_only.sites),
            cloud_node=invalid_cloud,
        )


def test_cloud_node_id_cannot_duplicate_trackside_id() -> None:
    """中心云和轨旁 MEC 必须能通过节点编号唯一查询。"""

    config = load_config("configs/debug.yaml")
    config["topology"]["include_cloud"] = False
    trackside_only = build_linear_topology(config)
    duplicate_id_cloud = EdgeNode(
        node_id=0,
        name="重复编号的中心云",
        node_type=NodeType.CLOUD,
        cpu_capacity=1000.0,
        memory_capacity_mb=5000.0,
        reliability=0.999,
        fault_domain=3,
    )

    with pytest.raises(ValueError, match="节点编号"):
        LinearRailTopology(
            sites=list(trackside_only.sites),
            cloud_node=duplicate_id_cloud,
        )
