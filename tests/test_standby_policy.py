"""
test_standby_policy.py

测试三种主备温状态控制策略。
"""

from src.config import load_config
from src.entities import TrainState
from src.standby_policy import (
    AllHotStandbyPolicy,
    PrimaryOnlyStandbyPolicy,
    TrajectoryAwareStandbyPolicy,
)
from src.topology import build_linear_topology


def build_test_topology():
    """
    创建测试铁路拓扑。
    """

    config = load_config("configs/debug.yaml")
    return build_linear_topology(config)


def build_train_state(
    remaining_dwell_time_s: float,
) -> TrainState:
    """
    创建即将从MEC-1切换到MEC-2的状态。
    """

    return TrainState(
        time_slot=0,
        position_m=500.0,
        speed_mps=69.44,
        serving_mec=0,
        next_mec=1,
        remaining_dwell_time_s=(
            remaining_dwell_time_s
        ),
    )


def test_primary_only_keeps_backup_cold() -> None:
    """
    冷备策略只保持主实例为温状态。
    """

    policy = PrimaryOnlyStandbyPolicy()

    result = policy.hot_replica_node_ids(
        function_id=0,
        candidate_node_ids=(0, 2),
        train_state=build_train_state(10.0),
        topology=build_test_topology(),
    )

    assert result == (0,)


def test_all_hot_keeps_all_replicas_warm() -> None:
    """
    全热备策略保持主备实例全部为温状态。
    """

    policy = AllHotStandbyPolicy()

    result = policy.hot_replica_node_ids(
        function_id=0,
        candidate_node_ids=(0, 2),
        train_state=build_train_state(10.0),
        topology=build_test_topology(),
    )

    assert result == (0, 2)


def test_trajectory_policy_keeps_backup_cold_early() -> None:
    """
    距离切换还有10秒时，只保持主实例。
    """

    policy = TrajectoryAwareStandbyPolicy(
        lead_time_s=3.0
    )

    result = policy.hot_replica_node_ids(
        function_id=0,
        candidate_node_ids=(0, 2),
        train_state=build_train_state(10.0),
        topology=build_test_topology(),
    )

    assert result == (0,)


def test_trajectory_policy_heats_backup_near_handover() -> None:
    """
    距离切换只剩2秒时，主备实例都应为温状态。
    """

    policy = TrajectoryAwareStandbyPolicy(
        lead_time_s=3.0
    )

    result = policy.hot_replica_node_ids(
        function_id=0,
        candidate_node_ids=(0, 2),
        train_state=build_train_state(2.0),
        topology=build_test_topology(),
    )

    assert result == (0, 2)