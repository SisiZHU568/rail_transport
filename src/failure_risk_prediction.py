"""
failure_risk_prediction.py

本文件为慢时间尺度控制器提供预测故障风险。

当前实现两种简单预测器：

1. ConstantFailureRiskProvider
   整个仿真过程使用固定风险。

2. WindowedFailureRiskProvider
   在指定时间窗口内使用高风险，
   其他时隙使用默认风险。

这些预测器用于构建可解释、可复现的调试实验。
后续可以替换为基于历史故障数据的预测模型。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.entities import TrainState


class FailureRiskProvider(ABC):
    """
    故障风险预测器抽象基类。
    """

    @abstractmethod
    def predict(
        self,
        time_slot: int,
        train_state: TrainState,
    ) -> float:
        """
        返回当前慢时间尺度使用的预测故障风险。
        """

        raise NotImplementedError


class ConstantFailureRiskProvider(
    FailureRiskProvider
):
    """
    始终返回固定故障风险。
    """

    def __init__(
        self,
        risk: float,
    ) -> None:
        if not 0 <= risk <= 1:
            raise ValueError(
                "故障风险必须位于[0,1]。"
            )

        self.risk = float(risk)

    def predict(
        self,
        time_slot: int,
        train_state: TrainState,
    ) -> float:
        """
        返回固定风险。
        """

        if time_slot < 0:
            raise ValueError(
                "time_slot不能小于0。"
            )

        return self.risk


@dataclass(frozen=True)
class FailureRiskWindow:
    """
    一个高故障风险时间窗口。

    起始和结束时隙均包含在窗口内。
    """

    start_slot: int
    end_slot: int
    risk: float

    def __post_init__(self) -> None:
        if self.start_slot < 0:
            raise ValueError(
                "风险窗口起始时隙不能小于0。"
            )

        if self.end_slot < self.start_slot:
            raise ValueError(
                "风险窗口结束时隙不能早于起始时隙。"
            )

        if not 0 <= self.risk <= 1:
            raise ValueError(
                "风险值必须位于[0,1]。"
            )

    def contains(
        self,
        time_slot: int,
    ) -> bool:
        """
        判断一个时隙是否位于当前窗口中。
        """

        return (
            self.start_slot
            <= time_slot
            <= self.end_slot
        )


class WindowedFailureRiskProvider(
    FailureRiskProvider
):
    """
    使用时间窗口描述预测风险变化。
    """

    def __init__(
        self,
        default_risk: float,
        windows: list[FailureRiskWindow],
    ) -> None:
        if not 0 <= default_risk <= 1:
            raise ValueError(
                "默认故障风险必须位于[0,1]。"
            )

        self.default_risk = float(
            default_risk
        )

        self.windows = tuple(windows)

    def predict(
        self,
        time_slot: int,
        train_state: TrainState,
    ) -> float:
        """
        时隙位于某个窗口内时返回窗口风险，
        否则返回默认风险。
        """

        if time_slot < 0:
            raise ValueError(
                "time_slot不能小于0。"
            )

        for window in self.windows:
            if window.contains(time_slot):
                return window.risk

        return self.default_risk


def build_windowed_failure_risk_provider(
    config: dict,
) -> WindowedFailureRiskProvider:
    """
    根据debug.yaml创建分时段风险预测器。
    """

    experiment_config = (
        config["two_timescale_experiment"]
    )

    high_risk = float(
        experiment_config[
            "high_failure_risk"
        ]
    )

    windows = [
        FailureRiskWindow(
            start_slot=int(
                item["start_slot"]
            ),
            end_slot=int(
                item["end_slot"]
            ),
            risk=high_risk,
        )
        for item in experiment_config[
            "high_risk_windows"
        ]
    ]

    return WindowedFailureRiskProvider(
        default_risk=float(
            experiment_config[
                "default_failure_risk"
            ]
        ),
        windows=windows,
    )