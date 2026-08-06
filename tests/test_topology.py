"""
test_topology.py

测试直线铁路拓扑是否正确构建，
以及 MEC 接入和切换逻辑是否正确。
"""

import pytest

from src.config import load_config
from src.topology import build_linear_topology


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