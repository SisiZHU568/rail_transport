"""
test_placement.py

测试当前接入 MEC 集中部署策略。
"""

import pytest

from src.config import load_config
from src.entities import (
    ServicePriority,
    SFCType,
    TrainState,
)
from src.placement import ServingMECPlacementPolicy
from src.topology import build_linear_topology


def build_test_sfc() -> SFCType:
    """
    创建测试 SFC。
    """

    return SFCType(
        sfc_id=0,
        name="测试SFC",
        function_ids=[0, 1, 2],
        deadline_ms=1500.0,
        reliability_target=0.99,
        priority=ServicePriority.CRITICAL,
    )


def test_all_functions_are_placed_on_serving_mec() -> None:
    """
    当前接入 MEC 为 node_id=2 时，
    三个函数都应部署在 node_id=2。
    """

    config = load_config("configs/debug.yaml")
    topology = build_linear_topology(config)

    train_state = TrainState(
        time_slot=0,
        position_m=4000.0,
        speed_mps=69.44,
        serving_mec=2,
        next_mec=3,
        remaining_dwell_time_s=10.0,
    )

    policy = ServingMECPlacementPolicy()

    placement = policy.place_functions(
        sfc=build_test_sfc(),
        train_state=train_state,
        topology=topology,
    )

    assert placement == [2, 2, 2]


def test_unknown_serving_mec_is_rejected() -> None:
    """
    拓扑中不存在 node_id=99 时，
    部署策略应主动报错。
    """

    config = load_config("configs/debug.yaml")
    topology = build_linear_topology(config)

    train_state = TrainState(
        time_slot=0,
        position_m=0.0,
        speed_mps=69.44,
        serving_mec=99,
        next_mec=99,
        remaining_dwell_time_s=0.0,
    )

    policy = ServingMECPlacementPolicy()

    with pytest.raises(KeyError):
        policy.place_functions(
            sfc=build_test_sfc(),
            train_state=train_state,
            topology=topology,
        )