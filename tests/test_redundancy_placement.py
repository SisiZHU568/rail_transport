"""
test_redundancy_placement.py

测试故障域感知的主备副本规划器。
"""

import pytest

from src.config import load_config
from src.entities import (
    ServicePriority,
    SFCType,
    TrainState,
)
from src.redundancy_placement import (
    build_reliability_aware_replica_planner,
)
from src.topology import (
    LinearRailTopology,
    build_linear_topology,
)


def build_test_sfc() -> SFCType:
    """
    创建测试 SFC。
    """

    return SFCType(
        sfc_id=0,
        name="高可靠故障诊断SFC",
        function_ids=[0, 1, 2],
        deadline_ms=1500.0,
        reliability_target=0.99,
        priority=ServicePriority.CRITICAL,
    )


def build_train_state(
    serving_mec: int,
    next_mec: int,
) -> TrainState:
    """
    创建测试列车状态。
    """

    return TrainState(
        time_slot=0,
        position_m=0.0,
        speed_mps=69.44,
        serving_mec=serving_mec,
        next_mec=next_mec,
        remaining_dwell_time_s=10.0,
    )


def test_mec1_backup_skips_same_fault_domain() -> None:
    """
    MEC-1 和 MEC-2 都属于故障域0。

    主节点为 MEC-1 时，
    备用节点不能选择 MEC-2，
    应选择最近的不同故障域节点 MEC-3。
    """

    config = load_config("configs/debug.yaml")
    topology = build_linear_topology(config)

    planner = (
        build_reliability_aware_replica_planner(
            config
        )
    )

    plan = planner.plan(
        sfc=build_test_sfc(),
        train_state=build_train_state(
            serving_mec=0,
            next_mec=1,
        ),
        topology=topology,
    )

    assert plan.primary_node_id == 0
    assert plan.shared_replica_node_ids == (0, 2)


def test_mec2_uses_nearest_different_domain_backup() -> None:
    """
    主节点为 MEC-2，属于故障域0。

    最近的不同故障域节点为 MEC-3。
    """

    config = load_config("configs/debug.yaml")
    topology = build_linear_topology(config)

    planner = (
        build_reliability_aware_replica_planner(
            config
        )
    )

    plan = planner.plan(
        sfc=build_test_sfc(),
        train_state=build_train_state(
            serving_mec=1,
            next_mec=2,
        ),
        topology=topology,
    )

    assert plan.shared_replica_node_ids == (1, 2)


def test_all_functions_receive_same_primary_backup_pair() -> None:
    """
    当前第一版策略让整条 SFC
    使用同一组主备节点。
    """

    config = load_config("configs/debug.yaml")
    topology = build_linear_topology(config)

    planner = (
        build_reliability_aware_replica_planner(
            config
        )
    )

    plan = planner.plan(
        sfc=build_test_sfc(),
        train_state=build_train_state(
            serving_mec=0,
            next_mec=1,
        ),
        topology=topology,
    )

    assert plan.function_replica_node_ids == {
        0: (0, 2),
        1: (0, 2),
        2: (0, 2),
    }


def test_impossible_fault_domain_diversity_is_rejected() -> None:
    """
    只保留 MEC-1 和 MEC-2 时，
    两者都属于故障域0。

    当前拓扑无法提供两个不同故障域，
    规划器应主动报错。
    """

    config = load_config("configs/debug.yaml")
    full_topology = build_linear_topology(config)

    # MEC-1 和 MEC-2 都在故障域0。
    same_domain_topology = LinearRailTopology(
        full_topology.sites[:2]
    )

    planner = (
        build_reliability_aware_replica_planner(
            config
        )
    )

    with pytest.raises(RuntimeError):
        planner.plan(
            sfc=build_test_sfc(),
            train_state=build_train_state(
                serving_mec=0,
                next_mec=1,
            ),
            topology=same_domain_topology,
        )


def test_builder_can_override_replica_count_to_three() -> None:
    """慢层选择三副本时，每个 VNF 都必须得到三个不同节点。"""

    config = load_config("configs/debug.yaml")
    topology = build_linear_topology(config)
    planner = build_reliability_aware_replica_planner(
        config,
        replica_count=3,
    )

    plan = planner.plan(
        sfc=build_test_sfc(),
        train_state=build_train_state(
            serving_mec=0,
            next_mec=1,
        ),
        topology=topology,
    )

    assert all(
        len(node_ids) == 3
        and len(set(node_ids)) == 3
        for node_ids
        in plan.function_replica_node_ids.values()
    )
