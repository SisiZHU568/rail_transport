"""
test_trajectory_prediction.py

测试轨迹预测器以及预测误差对预热决策的影响。
"""

import pytest

from src.config import load_config
from src.entities import (
    ServicePriority,
    SFCType,
    TrainState,
)
from src.prewarming import (
    TrajectoryAwarePrewarmingPolicy,
)
from src.topology import build_linear_topology
from src.trajectory_prediction import (
    BiasedTrajectoryPredictor,
    PerfectTrajectoryPredictor,
)


def build_test_topology():
    """
    创建测试铁路拓扑。
    """

    config = load_config("configs/debug.yaml")
    return build_linear_topology(config)


def build_test_sfc() -> SFCType:
    """
    创建测试用三函数 SFC。
    """

    return SFCType(
        sfc_id=0,
        name="测试SFC",
        function_ids=[0, 1, 2],
        deadline_ms=1500.0,
        reliability_target=0.99,
        priority=ServicePriority.CRITICAL,
    )


def build_test_train_state() -> TrainState:
    """
    创建一个即将从 MEC-1 切换到 MEC-2 的列车状态。

    当前真实剩余驻留时间为 2.4 秒。
    """

    return TrainState(
        time_slot=12,
        position_m=833.28,
        speed_mps=69.44,
        serving_mec=0,
        next_mec=1,
        remaining_dwell_time_s=2.4,
    )


def test_perfect_predictor_matches_real_state() -> None:
    """
    理想预测器应返回真实剩余时间和真实下一 MEC。
    """

    predictor = PerfectTrajectoryPredictor()

    prediction = predictor.predict(
        train_state=build_test_train_state(),
        topology=build_test_topology(),
    )

    assert (
        prediction.predicted_remaining_dwell_time_s
        == pytest.approx(2.4)
    )

    assert prediction.predicted_next_mec == 1


def test_negative_time_bias_predicts_earlier_handover() -> None:
    """
    时间偏差为 -4 秒时：

        真实剩余时间 = 6 秒
        预测剩余时间 = 2 秒

    系统认为切换会更早发生。
    """

    predictor = BiasedTrajectoryPredictor(
        time_bias_s=-4.0,
    )

    train_state = TrainState(
        time_slot=0,
        position_m=0.0,
        speed_mps=69.44,
        serving_mec=0,
        next_mec=1,
        remaining_dwell_time_s=6.0,
    )

    prediction = predictor.predict(
        train_state=train_state,
        topology=build_test_topology(),
    )

    assert (
        prediction.predicted_remaining_dwell_time_s
        == pytest.approx(2.0)
    )

    assert prediction.predicted_next_mec == 1


def test_positive_time_bias_predicts_later_handover() -> None:
    """
    时间偏差为 +4 秒时：

        真实剩余时间 = 2.4 秒
        预测剩余时间 = 6.4 秒

    系统认为切换距离现在还很远。
    """

    predictor = BiasedTrajectoryPredictor(
        time_bias_s=4.0,
    )

    prediction = predictor.predict(
        train_state=build_test_train_state(),
        topology=build_test_topology(),
    )

    assert (
        prediction.predicted_remaining_dwell_time_s
        == pytest.approx(6.4)
    )

    assert prediction.predicted_next_mec == 1


def test_hop_offset_predicts_wrong_next_mec() -> None:
    """
    真实下一节点是 MEC-2，即 node_id=1。

    当节点偏差为 +1 跳时，
    预测结果应变成 MEC-3，即 node_id=2。
    """

    predictor = BiasedTrajectoryPredictor(
        time_bias_s=0.0,
        next_mec_hop_offset=1,
    )

    prediction = predictor.predict(
        train_state=build_test_train_state(),
        topology=build_test_topology(),
    )

    assert prediction.predicted_next_mec == 2


def test_prewarming_policy_uses_predicted_time() -> None:
    """
    当前真实剩余驻留时间为2.4秒，
    预热阈值为3秒。

    理想预测：
        2.4 <= 3，应预热MEC-2。

    晚4秒预测：
        6.4 > 3，不应预热。
    """

    topology = build_test_topology()
    sfc = build_test_sfc()
    train_state = build_test_train_state()

    perfect_policy = (
        TrajectoryAwarePrewarmingPolicy(
            lead_time_s=3.0,
            predictor=PerfectTrajectoryPredictor(),
        )
    )

    late_policy = (
        TrajectoryAwarePrewarmingPolicy(
            lead_time_s=3.0,
            predictor=BiasedTrajectoryPredictor(
                time_bias_s=4.0
            ),
        )
    )

    perfect_targets = (
        perfect_policy.target_node_ids(
            train_state=train_state,
            sfc=sfc,
            topology=topology,
        )
    )

    late_targets = late_policy.target_node_ids(
        train_state=train_state,
        sfc=sfc,
        topology=topology,
    )

    assert perfect_targets == [1]
    assert late_targets == []