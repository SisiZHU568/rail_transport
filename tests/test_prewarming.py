
"""
test_prewarming.py

测试轨迹感知提前预热策略的基本决策。
"""

from src.config import load_config
from src.entities import (
    ServicePriority,
    SFCType,
    TrainState,
)
from src.prewarming import (
    NoPrewarmingPolicy,
    TrajectoryAwarePrewarmingPolicy,
)
from src.topology import build_linear_topology


def build_test_sfc() -> SFCType:
    """
    创建测试使用的三函数 SFC。
    """

    return SFCType(
        sfc_id=0,
        name="测试SFC",
        function_ids=[0, 1, 2],
        deadline_ms=1500.0,
        reliability_target=0.99,
        priority=ServicePriority.CRITICAL,
    )


def build_test_topology():
    """
    创建测试铁路拓扑。
    """

    config = load_config("configs/debug.yaml")
    return build_linear_topology(config)


def test_no_prewarming_policy_returns_empty_list() -> None:
    """
    被动响应式策略永远不预热。
    """

    policy = NoPrewarmingPolicy()

    train_state = TrainState(
        time_slot=0,
        position_m=900.0,
        speed_mps=69.44,
        serving_mec=0,
        next_mec=1,
        remaining_dwell_time_s=1.5,
    )

    targets = policy.target_node_ids(
        train_state=train_state,
        sfc=build_test_sfc(),
        topology=build_test_topology(),
    )

    assert targets == []


def test_trajectory_policy_does_not_prewarm_too_early() -> None:
    """
    剩余驻留时间大于预热阈值时，不应预热。
    """

    policy = TrajectoryAwarePrewarmingPolicy(
        lead_time_s=3.0
    )

    train_state = TrainState(
        time_slot=0,
        position_m=0.0,
        speed_mps=69.44,
        serving_mec=0,
        next_mec=1,
        remaining_dwell_time_s=14.4,
    )

    targets = policy.target_node_ids(
        train_state=train_state,
        sfc=build_test_sfc(),
        topology=build_test_topology(),
    )

    assert targets == []


def test_trajectory_policy_prewarms_next_mec() -> None:
    """
    剩余驻留时间小于3秒时，
    应预热下一 MEC。
    """

    policy = TrajectoryAwarePrewarmingPolicy(
        lead_time_s=3.0
    )

    train_state = TrainState(
        time_slot=12,
        position_m=833.28,
        speed_mps=69.44,
        serving_mec=0,
        next_mec=1,
        remaining_dwell_time_s=2.4,
    )

    targets = policy.target_node_ids(
        train_state=train_state,
        sfc=build_test_sfc(),
        topology=build_test_topology(),
    )

    assert targets == [1]


def test_last_mec_does_not_trigger_prewarming() -> None:
    """
    列车已经到达最后一个 MEC 时，
    不应继续预热。
    """

    policy = TrajectoryAwarePrewarmingPolicy(
        lead_time_s=3.0
    )

    train_state = TrainState(
        time_slot=116,
        position_m=8000.0,
        speed_mps=69.44,
        serving_mec=4,
        next_mec=4,
        remaining_dwell_time_s=0.0,
    )

    targets = policy.target_node_ids(
        train_state=train_state,
        sfc=build_test_sfc(),
        topology=build_test_topology(),
    )

    assert targets == []