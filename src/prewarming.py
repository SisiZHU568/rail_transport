"""
prewarming.py

本文件定义 Serverless 函数提前预热策略。

提前预热策略不再直接使用真实未来状态，
而是使用 TrajectoryPredictor 给出的预测结果。

支持：

1. 无预热基线；
2. 理想无误差轨迹感知预热；
3. 带时间偏差的轨迹感知预热；
4. 带目标 MEC 偏差的轨迹感知预热。
"""

from abc import ABC, abstractmethod

from src.entities import SFCType, TrainState
from src.topology import LinearRailTopology
from src.trajectory_prediction import (
    PerfectTrajectoryPredictor,
    TrajectoryPredictor,
)


class PrewarmingPolicy(ABC):
    """
    提前预热策略的抽象基类。
    """

    def __init__(
        self,
        policy_id: str,
        display_name: str,
    ) -> None:
        """
        Parameters
        ----------
        policy_id:
            策略英文编号。

        display_name:
            策略显示名称。
        """

        if not policy_id:
            raise ValueError("policy_id 不能为空。")

        if not display_name:
            raise ValueError("display_name 不能为空。")

        self.policy_id = policy_id
        self.display_name = display_name

    @abstractmethod
    def target_node_ids(
        self,
        train_state: TrainState,
        sfc: SFCType,
        topology: LinearRailTopology,
    ) -> list[int]:
        """
        返回当前时隙需要预热的 MEC 节点编号。
        """

        raise NotImplementedError


class NoPrewarmingPolicy(PrewarmingPolicy):
    """
    不进行提前预热的基线策略。
    """

    def __init__(self) -> None:
        super().__init__(
            policy_id="no_prewarming",
            display_name="被动响应式",
        )

    def target_node_ids(
        self,
        train_state: TrainState,
        sfc: SFCType,
        topology: LinearRailTopology,
    ) -> list[int]:
        """
        被动策略永远不进行主动预热。
        """

        return []


class TrajectoryAwarePrewarmingPolicy(PrewarmingPolicy):
    """
    基于轨迹预测结果的提前预热策略。

    当前决策条件为：

        预测剩余驻留时间
        <=
        提前预热时间阈值

    当条件满足时，
    在预测的下一 MEC 上预热整条 SFC。
    """

    def __init__(
        self,
        lead_time_s: float,
        predictor: TrajectoryPredictor | None = None,
    ) -> None:
        """
        Parameters
        ----------
        lead_time_s:
            提前预热阈值，单位为秒。

        predictor:
            轨迹预测器。

            None：
                默认使用理想无误差预测器。

            这样原有代码：

                TrajectoryAwarePrewarmingPolicy(
                    lead_time_s=3.0
                )

            仍然可以正常工作。
        """

        if lead_time_s <= 0:
            raise ValueError(
                "提前预热时间阈值必须大于 0。"
            )

        # 没有显式指定预测器时，
        # 默认使用理想无误差预测器。
        if predictor is None:
            predictor = PerfectTrajectoryPredictor()

        super().__init__(
            policy_id=(
                f"trajectory_aware_"
                f"{predictor.predictor_id}"
            ),
            display_name=(
                f"轨迹感知预热("
                f"{lead_time_s:g}s，"
                f"{predictor.display_name})"
            ),
        )

        self.lead_time_s = lead_time_s
        self.predictor = predictor

    def target_node_ids(
        self,
        train_state: TrainState,
        sfc: SFCType,
        topology: LinearRailTopology,
    ) -> list[int]:
        """
        根据轨迹预测结果决定预轨迹预测结果决定预热目标。
        """

        prediction = self.predictor.predict(
            train_state=train_state,
            topology=topology,
        )

        # 预测器没有给出有效下一 MEC。
        if prediction.predicted_next_mec is None:
            return []

        # 检查预测目标确实位于铁路拓扑中。
        topology.get_site(
            prediction.predicted_next_mec
        )

        # 预测切换时间进入预热窗口。
        if (
            prediction.predicted_remaining_dwell_time_s
            <= self.lead_time_s
        ):
            return [
                prediction.predicted_next_mec
            ]

        return []