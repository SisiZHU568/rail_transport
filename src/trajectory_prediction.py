"""
trajectory_prediction.py

本文件负责预测列车未来的移动状态。

提前预热策略需要知道两个关键信息：

1. 列车预计还要多久离开当前 MEC；
2. 列车预计会切换到哪个 MEC。

真实系统中的预测不可能完全准确，因此本文件同时支持：

1. 时间预测误差；
2. 下一 MEC 预测误差。

当前阶段使用确定性偏差进行调试。
后续可以替换为神经网络、卡尔曼滤波或轨迹数据驱动模型。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.entities import TrainState
from src.topology import LinearRailTopology


@dataclass(frozen=True)
class TrajectoryPrediction:
    """
    一次轨迹预测结果。

    Attributes
    ----------
    predicted_remaining_dwell_time_s:
        预测的当前 MEC 剩余驻留时间，单位为秒。

    predicted_next_mec:
        预测的下一接入 MEC 节点编号。

        如果已经没有下一 MEC，
        或预测目标超出当前线路范围，则为 None。
    """

    predicted_remaining_dwell_time_s: float
    predicted_next_mec: int | None

    def __post_init__(self) -> None:
        """
        预测的剩余驻留时间不能小于 0。
        """

        if self.predicted_remaining_dwell_time_s < 0:
            raise ValueError(
                "预测剩余驻留时间不能小于 0。"
            )


class TrajectoryPredictor(ABC):
    """
    轨迹预测器抽象基类。

    所有轨迹预测模型都必须实现 predict() 方法。
    """

    def __init__(
        self,
        predictor_id: str,
        display_name: str,
    ) -> None:
        """
        Parameters
        ----------
        predictor_id:
            预测器英文编号。

        display_name:
            预测器显示名称。
        """

        if not predictor_id:
            raise ValueError("predictor_id 不能为空。")

        if not display_name:
            raise ValueError("display_name 不能为空。")

        self.predictor_id = predictor_id
        self.display_name = display_name

    @abstractmethod
    def predict(
        self,
        train_state: TrainState,
        topology: LinearRailTopology,
    ) -> TrajectoryPrediction:
        """
        根据当前列车状态预测未来移动信息。
        """

        raise NotImplementedError


class PerfectTrajectoryPredictor(TrajectoryPredictor):
    """
    理想轨迹预测器。

    它直接使用当前仿真器中的真实：

    1. 剩余驻留时间；
    2. 下一 MEC。

    该预测器没有任何误差，
    主要作为理想上界和调试基准。
    """

    def __init__(self) -> None:
        super().__init__(
            predictor_id="perfect",
            display_name="理想无误差预测",
        )

    def predict(
        self,
        train_state: TrainState,
        topology: LinearRailTopology,
    ) -> TrajectoryPrediction:
        """
        返回无误差预测结果。
        """

        # 当前 serving_mec 必须存在于拓扑中。
        topology.get_site(train_state.serving_mec)

        # 最后一个 MEC 中，
        # next_mec 会等于 serving_mec。
        if train_state.next_mec == train_state.serving_mec:
            predicted_next_mec = None
        else:
            topology.get_site(train_state.next_mec)
            predicted_next_mec = train_state.next_mec

        return TrajectoryPrediction(
            predicted_remaining_dwell_time_s=(
                train_state.remaining_dwell_time_s
            ),
            predicted_next_mec=predicted_next_mec,
        )


class BiasedTrajectoryPredictor(TrajectoryPredictor):
    """
    带固定预测偏差的轨迹预测器。

    它用于模拟两类预测错误：

    1. 时间偏差 time_bias_s；
    2. 节点偏差 next_mec_hop_offset。

    时间偏差示例
    ------------
    time_bias_s = -4：

        真实剩余时间为 8 秒，
        预测值为 4 秒。

        系统认为列车更早切换，
        因此会过早预热。

    time_bias_s = 4：

        真实剩余时间为 2 秒，
        预测值为 6 秒。

        系统认为切换还很远，
        因此可能错过预热时机。

    节点偏差示例
    ------------
    next_mec_hop_offset = 1：

        真实下一节点为 MEC-2，
        预测下一节点为 MEC-3。

        系统会在错误节点上预热。
    """

    def __init__(
        self,
        time_bias_s: float = 0.0,
        next_mec_hop_offset: int = 0,
    ) -> None:
        """
        Parameters
        ----------
        time_bias_s:
            预测剩余驻留时间上的固定偏差。

            负数：
                预测切换更早。

            正数：
                预测切换更晚。

        next_mec_hop_offset:
            对预测目标 MEC 的跳数偏差。

            0：
                预测正确的下一 MEC。

            1：
                向前多预测一个 MEC。

            -1：
                向后偏移一个 MEC。
        """

        if not isinstance(next_mec_hop_offset, int):
            raise TypeError(
                "next_mec_hop_offset 必须是整数。"
            )

        predictor_id = (
            f"biased_time_{time_bias_s:g}"
            f"_hop_{next_mec_hop_offset}"
        )

        display_name = (
            f"偏差预测("
            f"时间{time_bias_s:+g}s，"
            f"节点{next_mec_hop_offset:+d}跳)"
        )

        super().__init__(
            predictor_id=predictor_id,
            display_name=display_name,
        )

        self.time_bias_s = time_bias_s
        self.next_mec_hop_offset = (
            next_mec_hop_offset
        )

    def predict(
        self,
        train_state: TrainState,
        topology: LinearRailTopology,
    ) -> TrajectoryPrediction:
        """
        生成带固定偏差的预测结果。
        """

        # ----------------------------------------------------
        # 1. 预测剩余驻留时间
        # ----------------------------------------------------
        #
        # 使用 max(..., 0.0) 防止偏差导致负时间。
        predicted_remaining_dwell_time_s = max(
            train_state.remaining_dwell_time_s
            + self.time_bias_s,
            0.0,
        )

        # ----------------------------------------------------
        # 2. 判断是否还有下一 MEC
        # ----------------------------------------------------

        if train_state.next_mec == train_state.serving_mec:
            return TrajectoryPrediction(
                predicted_remaining_dwell_time_s=(
                    predicted_remaining_dwell_time_s
                ),
                predicted_next_mec=None,
            )

        # 所有 MEC 已经按照线路位置排序。
        ordered_node_ids = [
            site.node.node_id
            for site in topology.sites
        ]

        if train_state.next_mec not in ordered_node_ids:
            raise KeyError(
                f"拓扑中不存在 next_mec="
                f"{train_state.next_mec}。"
            )

        actual_next_index = ordered_node_ids.index(
            train_state.next_mec
        )

        # 在真实下一 MEC 的索引上加入节点预测偏差。
        predicted_next_index = (
            actual_next_index
            + self.next_mec_hop_offset
        )

        # 预测结果超出线路范围时，
        # 表示没有得到有效目标 MEC。
        if (
            predicted_next_index < 0
            or predicted_next_index >= len(ordered_node_ids)
        ):
            predicted_next_mec = None
        else:
            predicted_next_mec = ordered_node_ids[
                predicted_next_index
            ]

        # 不能将当前正在服务的 MEC
        # 当作“下一 MEC”进行提前预热。
        if predicted_next_mec == train_state.serving_mec:
            predicted_next_mec = None

        return TrajectoryPrediction(
            predicted_remaining_dwell_time_s=(
                predicted_remaining_dwell_time_s
            ),
            predicted_next_mec=predicted_next_mec,
        )