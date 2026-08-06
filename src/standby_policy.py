"""
standby_policy.py

本文件决定主备副本中的哪些实例处于温状态。

副本规划器解决的是：

    每个函数有哪些候选副本？

本文件解决的是：

    哪些候选副本现在已经加载并处于 WARM 状态？

实现三种策略：

1. PrimaryOnlyStandbyPolicy
   只有主实例处于温状态，备用实例为冷备。

2. AllHotStandbyPolicy
   主实例和所有备用实例全部保持温状态。

3. TrajectoryAwareStandbyPolicy
   主实例始终为温实例；
   临近 MEC 切换时，将备用实例转为温实例。
"""

from abc import ABC, abstractmethod

from src.entities import TrainState
from src.topology import LinearRailTopology


class StandbyActivationPolicy(ABC):
    """
    主备副本温状态控制策略的抽象基类。
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
    def hot_replica_node_ids(
        self,
        function_id: int,
        candidate_node_ids: tuple[int, ...],
        train_state: TrainState,
        topology: LinearRailTopology,
    ) -> tuple[int, ...]:
        """
        返回当前时隙处于 WARM 状态的副本节点。

        candidate_node_ids 中的第一个节点为主节点。
        """

        raise NotImplementedError

    @staticmethod
    def _validate_candidates(
        candidate_node_ids: tuple[int, ...],
        topology: LinearRailTopology,
    ) -> None:
        """
        检查候选副本节点是否合法。
        """

        if len(candidate_node_ids) == 0:
            raise ValueError(
                "一个函数至少需要一个候选副本。"
            )

        if len(candidate_node_ids) != len(
            set(candidate_node_ids)
        ):
            raise ValueError(
                "候选副本节点不能重复。"
            )

        for node_id in candidate_node_ids:
            topology.get_site(node_id)


class PrimaryOnlyStandbyPolicy(
    StandbyActivationPolicy
):
    """
    仅主实例保持温状态。

    当副本规划中只有一个节点时，
    它表示单副本方案。

    当副本规划中包含主备节点时，
    它表示冷备用方案。
    """

    def __init__(self) -> None:
        super().__init__(
            policy_id="primary_only",
            display_name="仅主实例常驻",
        )

    def hot_replica_node_ids(
        self,
        function_id: int,
        candidate_node_ids: tuple[int, ...],
        train_state: TrainState,
        topology: LinearRailTopology,
    ) -> tuple[int, ...]:
        """
        只返回候选副本列表中的第一个节点。
        """

        self._validate_candidates(
            candidate_node_ids=candidate_node_ids,
            topology=topology,
        )

        return (candidate_node_ids[0],)


class AllHotStandbyPolicy(
    StandbyActivationPolicy
):
    """
    所有主备副本始终保持温状态。
    """

    def __init__(self) -> None:
        super().__init__(
            policy_id="all_hot",
            display_name="全热备",
        )

    def hot_replica_node_ids(
        self,
        function_id: int,
        candidate_node_ids: tuple[int, ...],
        train_state: TrainState,
        topology: LinearRailTopology,
    ) -> tuple[int, ...]:
        """
        返回全部候选副本节点。
        """

        self._validate_candidates(
            candidate_node_ids=candidate_node_ids,
            topology=topology,
        )

        return candidate_node_ids


class TrajectoryAwareStandbyPolicy(
    StandbyActivationPolicy
):
    """
    轨迹感知动态主备策略。

    正常阶段：

        主实例 WARM
        备用实例 COLD

    当列车剩余驻留时间不超过 lead_time_s 时：

        主实例 WARM
        备用实例 WARM

    进入最后一个 MEC 后不再存在下一次切换，
    因此只保持主实例为温状态。
    """

    def __init__(
        self,
        lead_time_s: float,
    ) -> None:
        """
        Parameters
        ----------
        lead_time_s:
            动态激活备用实例的提前时间，单位为秒。
        """

        if lead_time_s <= 0:
            raise ValueError(
                "动态主备提前时间必须大于0。"
            )

        super().__init__(
            policy_id="trajectory_aware_standby",
            display_name=(
                f"轨迹感知动态主备({lead_time_s:g}s)"
            ),
        )

        self.lead_time_s = lead_time_s

    def hot_replica_node_ids(
        self,
        function_id: int,
        candidate_node_ids: tuple[int, ...],
        train_state: TrainState,
        topology: LinearRailTopology,
    ) -> tuple[int, ...]:
        """
        根据列车剩余驻留时间决定备用实例状态。
        """

        self._validate_candidates(
            candidate_node_ids=candidate_node_ids,
            topology=topology,
        )

        # 只有一个副本时不存在备用实例。
        if len(candidate_node_ids) == 1:
            return candidate_node_ids

        # 最后一个 MEC 中不再提前激活备用实例。
        if train_state.next_mec == train_state.serving_mec:
            return (candidate_node_ids[0],)

        # 临近切换时，所有候选副本转为温状态。
        if (
            train_state.remaining_dwell_time_s
            <= self.lead_time_s
        ):
            return candidate_node_ids

        # 其他时间仅主实例为温状态。
        return (candidate_node_ids[0],)