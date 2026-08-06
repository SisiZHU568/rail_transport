"""
placement.py

本文件负责确定 SFC 中每个函数部署在哪个 MEC。

当前首先实现最简单的“当前接入 MEC 集中部署策略”：

列车当前连接 MEC-2 时：

    函数0 → MEC-2
    函数1 → MEC-2
    函数2 → MEC-2

当列车切换到 MEC-3 后：

    函数0 → MEC-3
    函数1 → MEC-3
    函数2 → MEC-3

该策略实现简单，但在 MEC 切换后可能产生新的冷启动。
"""

from abc import ABC, abstractmethod

from src.entities import SFCType, TrainState
from src.topology import LinearRailTopology


class PlacementPolicy(ABC):
    """
    SFC 函数部署策略的抽象基类。

    后续实现其他部署策略时，都需要继承本类。
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
    def place_functions(
        self,
        sfc: SFCType,
        train_state: TrainState,
        topology: LinearRailTopology,
    ) -> list[int]:
        """
        决定 SFC 中每个函数的部署节点。

        Returns
        -------
        list[int]:
            与 sfc.function_ids 一一对应的 MEC 节点编号。
        """

        raise NotImplementedError


class ServingMECPlacementPolicy(PlacementPolicy):
    """
    将所有 SFC 函数部署在当前接入 MEC。

    优点：

    1. 不产生函数之间的跨 MEC 传输；
    2. 处理结果距离列车较近；
    3. 实现简单。

    缺点：

    1. 列车切换 MEC 后可能发生冷启动；
    2. 容易在当前接入 MEC 上形成资源集中；
    3. 没有利用沿线轨旁 MEC 的协同能力。
    """

    def __init__(self) -> None:
        super().__init__(
            policy_id="serving_mec",
            display_name="当前接入MEC集中部署",
        )

    def place_functions(
        self,
        sfc: SFCType,
        train_state: TrainState,
        topology: LinearRailTopology,
    ) -> list[int]:
        """
        将所有函数放在列车当前接入 MEC。
        """

        # 检查当前 serving_mec 是否确实存在于拓扑中。
        topology.get_site(train_state.serving_mec)

        return [
            train_state.serving_mec
            for _ in sfc.function_ids
        ]