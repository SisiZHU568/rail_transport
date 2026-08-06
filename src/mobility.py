"""
mobility.py

本文件负责模拟列车沿直线铁路移动。

当前采用最简单的匀速运动模型：

    新位置 = 旧位置 + 速度 × 时隙长度

列车移动后，程序会自动更新：

1. 当前列车位置；
2. 当前接入 MEC；
3. 下一个 MEC；
4. 当前覆盖区剩余驻留时间。
"""

from src.entities import TrainState
from src.topology import LinearRailTopology


class TrainMobilityModel:
    """
    单列车匀速移动模型。

    当前只考虑列车从线路起点向终点正向移动。
    """

    def __init__(
        self,
        topology: LinearRailTopology,
        initial_position_m: float,
        speed_mps: float,
        slot_seconds: float,
    ) -> None:
        """
        创建列车移动模型。

        Parameters
        ----------
        topology:
            铁路拓扑。

        initial_position_m:
            列车初始位置，单位为米。

        speed_mps:
            列车速度，单位为米/秒。

        slot_seconds:
            一个快时隙持续时间，单位为秒。
        """

        if initial_position_m < topology.route_start_m:
            raise ValueError("列车初始位置不能位于线路起点之前。")

        if initial_position_m > topology.route_end_m:
            raise ValueError("列车初始位置不能超过线路终点。")

        if speed_mps < 0:
            raise ValueError("当前移动模型不支持负速度。")

        if slot_seconds <= 0:
            raise ValueError("时隙长度必须大于 0。")

        self.topology = topology
        self.initial_position_m = initial_position_m
        self.speed_mps = speed_mps
        self.slot_seconds = slot_seconds

        # 当前仿真时隙。
        self.time_slot = 0

        # 当前列车位置。
        self.position_m = initial_position_m

    @property
    def finished(self) -> bool:
        """
        判断列车是否到达线路终点。
        """

        return self.position_m >= self.topology.route_end_m

    def reset(self) -> TrainState:
        """
        将列车恢复到初始状态。

        Returns
        -------
        TrainState:
            重置后的列车状态。
        """

        self.time_slot = 0
        self.position_m = self.initial_position_m

        return self._build_state()

    def step(self) -> TrainState:
        """
        让列车向前移动一个快时隙。

        运动公式：

            position(t+1)
            =
            position(t) + speed × slot_seconds

        Returns
        -------
        TrainState:
            移动后的列车状态。
        """

        # 先计算下一个时隙的位置。
        new_position_m = (
            self.position_m
            + self.speed_mps * self.slot_seconds
        )

        # 列车不能越过仿真线路终点。
        self.position_m = min(
            new_position_m,
            self.topology.route_end_m,
        )

        # 时隙编号加 1。
        self.time_slot += 1

        return self._build_state()

    def _build_state(self) -> TrainState:
        """
        根据当前位置构造 TrainState。

        该方法以下划线开头，
        表示它主要供类内部使用。
        """

        serving_mec = self.topology.serving_mec(
            self.position_m
        )

        # 当前配置中相邻 MEC 覆盖区域存在重叠，
        # 正常情况下列车不会失去覆盖。
        if serving_mec is None:
            raise RuntimeError(
                f"列车位于 {self.position_m:.2f} 米，"
                "但没有任何 MEC 可以提供覆盖。"
            )

        next_mec = self.topology.next_mec(serving_mec)

        # 到达最后一个 MEC 后已经没有下一 MEC。
        # 为了满足 TrainState 中 next_mec 为整数的要求，
        # 这里让 next_mec 等于当前 MEC。
        if next_mec is None:
            next_mec = serving_mec

        remaining_dwell_time_s = (
            self.topology.remaining_dwell_time_s(
                train_position_m=self.position_m,
                train_speed_mps=self.speed_mps,
                current_node_id=serving_mec,
            )
        )

        return TrainState(
            time_slot=self.time_slot,
            position_m=self.position_m,
            speed_mps=self.speed_mps,
            serving_mec=serving_mec,
            next_mec=next_mec,
            remaining_dwell_time_s=remaining_dwell_time_s,
        )