
"""
failure_process.py

本文件负责生成轨旁 MEC 与中心云在每个离散时隙中的运行状态。

当前考虑两层故障：

1. 故障域故障
   可能同时使同一故障域中的多个 MEC 不可用。

2. 计算节点局部故障
   即使所属故障域正常，单个轨旁 MEC 或中心云仍可能失效。

节点最终可用需要同时满足：

    所属故障域正常
    且
    节点局部状态正常

本文件实现：

1. ScriptedFailureProcess：
   使用人工指定的故障时隙，便于自动测试。

2. MarkovFailureProcess：
   使用二状态马尔可夫过程产生随机故障和恢复。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import math
import random
from typing import Any

from src.topology import LinearRailTopology
from src.orchestration_config import CTMCRates, load_phase_a_config


@dataclass(frozen=True)
class CTMCTransition:
    """连续时间两状态链在相邻时隙边界之间的精确转移概率。"""

    failure_probability: float
    recovery_probability: float

    @classmethod
    def from_rates(
        cls,
        rates: CTMCRates,
        slot_seconds: float,
    ) -> "CTMCTransition":
        """使用生成矩阵指数闭式解离散连续时间率。"""

        if not math.isfinite(slot_seconds) or slot_seconds <= 0.0:
            raise ValueError("slot_seconds 必须是正有限数。")
        total_rate = (
            rates.failure_rate_per_second
            + rates.recovery_rate_per_second
        )
        change_probability = -math.expm1(-total_rate * slot_seconds)
        return cls(
            failure_probability=(
                rates.failure_rate_per_second
                / total_rate
                * change_probability
            ),
            recovery_probability=(
                rates.recovery_rate_per_second
                / total_rate
                * change_probability
            ),
        )


def _stream_seed(base_seed: int, entity_kind: str, entity_id: int) -> int:
    """为每个域和节点派生稳定、互不共享状态的随机流种子。"""

    payload = f"{base_seed}:{entity_kind}:{entity_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@dataclass(frozen=True)
class InfrastructureState:
    """
    一个时隙中的基础设施状态。
    """

    time_slot: int

    # 故障域是否正常。
    # 对轨旁 MEC，它表示区域供电、汇聚网络等共享条件；
    # 对中心云，它表示云数据中心或云回传通道等共享条件。
    domain_up: dict[int, bool]

    # 节点自身是否正常
    #
    # 对中心云，它表示云计算服务自身状态，与上面的共享
    # 故障域状态含义不同，因此两层状态不是重复记录。
    # 注意，这里还没有结合所属故障域状态。
    node_local_up: dict[int, bool]

    # 快照版本在同一故障过程内单调递增；脚本过程直接使用时隙号。
    version: int = 0

    # 事件集合由“故障域 AND 节点局部状态”的有效状态边沿生成。
    newly_unavailable_node_ids: tuple[int, ...] = ()
    newly_available_node_ids: tuple[int, ...] = ()

    def is_node_operational(
        self,
        node_id: int,
        topology: LinearRailTopology,
    ) -> bool:
        """
        判断某个计算节点最终是否可以工作。
        """

        # 中心云没有轨道位置，不能通过 get_site 查询；
        # get_node 同时支持轨旁 MEC 和中心云。
        node = topology.get_node(node_id)
        domain_id = node.fault_domain

        if domain_id not in self.domain_up:
            raise KeyError(
                f"基础设施状态中缺少故障域 {domain_id}。"
            )

        if node_id not in self.node_local_up:
            raise KeyError(
                f"基础设施状态中缺少节点 {node_id}。"
            )

        return (
            self.domain_up[domain_id]
            and self.node_local_up[node_id]
        )

    @property
    def down_domain_ids(self) -> tuple[int, ...]:
        """
        返回当前失效的故障域编号。
        """

        return tuple(
            sorted(
                domain_id
                for domain_id, is_up
                in self.domain_up.items()
                if not is_up
            )
        )

    @property
    def down_node_ids(self) -> tuple[int, ...]:
        """
        返回当前发生局部失效的节点编号。

        这里不包含单纯由故障域失效导致不可用的节点。
        """

        return tuple(
            sorted(
                node_id
                for node_id, is_up
                in self.node_local_up.items()
                if not is_up
            )
        )


class FailureProcess(ABC):
    """
    故障过程抽象基类。
    """

    @abstractmethod
    def reset(self) -> None:
        """
        恢复到仿真开始状态。
        """

        raise NotImplementedError

    @abstractmethod
    def state_for_slot(
        self,
        time_slot: int,
    ) -> InfrastructureState:
        """
        返回指定时隙的基础设施状态。
        """

        raise NotImplementedError


class ScriptedFailureProcess(FailureProcess):
    """
    人工指定故障时隙的确定性故障过程。

    它主要用于自动测试。

    例如：

        down_nodes_by_slot={
            3: {1}
        }

    表示 node_id=1 的 MEC 只在时隙3发生局部故障。
    """

    def __init__(
        self,
        topology: LinearRailTopology,
        down_domains_by_slot: dict[
            int,
            set[int],
        ] | None = None,
        down_nodes_by_slot: dict[
            int,
            set[int],
        ] | None = None,
    ) -> None:
        """
        创建确定性故障过程。
        """

        self.topology = topology

        # 故障状态覆盖全部可部署节点。未在脚本中列出的中心云
        # 默认保持正常，因此旧的轨旁故障脚本无需增加云配置。
        self.domain_ids = {
            node.fault_domain
            for node in topology.compute_nodes
        }

        self.node_ids = {
            node.node_id
            for node in topology.compute_nodes
        }

        self.down_domains_by_slot = {
            time_slot: set(domain_ids)
            for time_slot, domain_ids
            in (down_domains_by_slot or {}).items()
        }

        self.down_nodes_by_slot = {
            time_slot: set(node_ids)
            for time_slot, node_ids
            in (down_nodes_by_slot or {}).items()
        }

        for time_slot in self.down_domains_by_slot:
            if time_slot < 0:
                raise ValueError(
                    "故障时隙不能小于 0。"
                )

        for time_slot in self.down_nodes_by_slot:
            if time_slot < 0:
                raise ValueError(
                    "故障时隙不能小于 0。"
                )

        self._previous_effective_up: dict[int, bool] | None = None

        configured_domain_ids = {
            domain_id
            for domain_ids
            in self.down_domains_by_slot.values()
            for domain_id in domain_ids
        }

        unknown_domain_ids = (
            configured_domain_ids
            - self.domain_ids
        )

        if unknown_domain_ids:
            raise ValueError(
                "存在未知故障域："
                f"{sorted(unknown_domain_ids)}"
            )

        configured_node_ids = {
            node_id
            for node_ids
            in self.down_nodes_by_slot.values()
            for node_id in node_ids
        }

        unknown_node_ids = (
            configured_node_ids
            - self.node_ids
        )

        if unknown_node_ids:
            raise ValueError(
                "存在未知计算节点："
                f"{sorted(unknown_node_ids)}"
            )

    def reset(self) -> None:
        """
        重置有效状态历史，使再次读取同一脚本得到相同事件。
        """

        self._previous_effective_up = None

    def state_for_slot(
        self,
        time_slot: int,
    ) -> InfrastructureState:
        """
        返回人工指定的故障状态。
        """

        if time_slot < 0:
            raise ValueError(
                "time_slot 不能小于 0。"
            )

        down_domains = (
            self.down_domains_by_slot.get(
                time_slot,
                set(),
            )
        )

        down_nodes = (
            self.down_nodes_by_slot.get(
                time_slot,
                set(),
            )
        )

        domain_up = {
            domain_id: domain_id not in down_domains
            for domain_id in self.domain_ids
        }

        node_local_up = {
            node_id: node_id not in down_nodes
            for node_id in self.node_ids
        }

        effective_up = {
            node.node_id: (
                domain_up[node.fault_domain]
                and node_local_up[node.node_id]
            )
            for node in self.topology.compute_nodes
        }
        previous = self._previous_effective_up or {
            node_id: True for node_id in effective_up
        }
        newly_unavailable = tuple(
            sorted(
                node_id
                for node_id, is_up in effective_up.items()
                if previous[node_id] and not is_up
            )
        )
        newly_available = tuple(
            sorted(
                node_id
                for node_id, is_up in effective_up.items()
                if not previous[node_id] and is_up
            )
        )
        self._previous_effective_up = effective_up

        return InfrastructureState(
            time_slot=time_slot,
            domain_up=domain_up,
            node_local_up=node_local_up,
            version=time_slot,
            newly_unavailable_node_ids=newly_unavailable,
            newly_available_node_ids=newly_available,
        )


class MarkovFailureProcess(FailureProcess):
    """
    二状态马尔可夫随机故障过程。

    每个故障域和节点都有两种状态：

        UP：正常
        DOWN：失效

    状态转移规则：

        UP --失效概率--> DOWN
        DOWN --恢复概率--> UP

    初始状态按各自稳态可用度采样，避免短仿真从全正常状态开始的偏差。
    """

    def __init__(
        self,
        topology: LinearRailTopology,
        domain_rates: dict[int, CTMCRates],
        node_rates: dict[int, CTMCRates],
        slot_seconds: float,
        random_seed: int,
    ) -> None:
        """
        创建随机故障过程。
        """

        self.topology = topology
        self.random_seed = random_seed
        self.slot_seconds = slot_seconds

        self.domain_ids = tuple(
            sorted(
                {
                    node.fault_domain
                    for node in topology.compute_nodes
                }
            )
        )

        self.node_ids = tuple(
            sorted(
                node.node_id
                for node in topology.compute_nodes
            )
        )

        self.domain_rates = dict(domain_rates)
        self.node_rates = dict(node_rates)
        if set(self.domain_rates) != set(self.domain_ids):
            raise ValueError("故障域 CTMC 率的编号集合不完整。")
        if set(self.node_rates) != set(self.node_ids):
            raise ValueError("节点 CTMC 率的编号集合不完整。")
        self.domain_transitions = {
            entity_id: CTMCTransition.from_rates(rates, slot_seconds)
            for entity_id, rates in self.domain_rates.items()
        }
        self.node_transitions = {
            entity_id: CTMCTransition.from_rates(rates, slot_seconds)
            for entity_id, rates in self.node_rates.items()
        }

        self._domain_up: dict[int, bool] = {}
        self._node_local_up: dict[int, bool] = {}
        self._domain_random: dict[int, random.Random] = {}
        self._node_random: dict[int, random.Random] = {}
        self._previous_effective_up: dict[int, bool] = {}
        self._next_time_slot = 0

        self.reset()

    @staticmethod
    def stationary_initial_state(rates: CTMCRates, draw: float) -> bool:
        """用稳态可用度和给定随机数确定初始 UP/DOWN。"""

        if not 0.0 <= draw < 1.0:
            raise ValueError("稳态初始化随机数必须位于 [0, 1)。")
        return draw < rates.steady_availability

    def reset(self) -> None:
        """
        重建所有独立随机流并按稳态分布初始化。
        """

        self._domain_random = {
            domain_id: random.Random(
                _stream_seed(self.random_seed, "domain", domain_id)
            )
            for domain_id in self.domain_ids
        }
        self._node_random = {
            node_id: random.Random(
                _stream_seed(self.random_seed, "node", node_id)
            )
            for node_id in self.node_ids
        }
        self._domain_up = {
            domain_id: self.stationary_initial_state(
                self.domain_rates[domain_id],
                self._domain_random[domain_id].random(),
            )
            for domain_id in self.domain_ids
        }
        self._node_local_up = {
            node_id: self.stationary_initial_state(
                self.node_rates[node_id],
                self._node_random[node_id].random(),
            )
            for node_id in self.node_ids
        }
        self._previous_effective_up = self._effective_node_states()

        self._next_time_slot = 0

    def _update_states(self) -> None:
        """
        将所有故障域和节点向前更新一个时隙。
        """

        for domain_id in self.domain_ids:
            currently_up = self._domain_up[
                domain_id
            ]

            if currently_up:
                transition_probability = (
                    self.domain_transitions[domain_id].failure_probability
                )

                if (
                    self._domain_random[domain_id].random()
                    < transition_probability
                ):
                    self._domain_up[domain_id] = False
            else:
                transition_probability = (
                    self.domain_transitions[domain_id].recovery_probability
                )

                if (
                    self._domain_random[domain_id].random()
                    < transition_probability
                ):
                    self._domain_up[domain_id] = True

        for node_id in self.node_ids:
            currently_up = self._node_local_up[
                node_id
            ]

            if currently_up:
                transition_probability = (
                    self.node_transitions[node_id].failure_probability
                )

                if (
                    self._node_random[node_id].random()
                    < transition_probability
                ):
                    self._node_local_up[node_id] = False
            else:
                transition_probability = (
                    self.node_transitions[node_id].recovery_probability
                )

                if (
                    self._node_random[node_id].random()
                    < transition_probability
                ):
                    self._node_local_up[node_id] = True

    def _effective_node_states(self) -> dict[int, bool]:
        """结合域和节点隐藏状态生成实际可用状态。"""

        return {
            node.node_id: (
                self._domain_up[node.fault_domain]
                and self._node_local_up[node.node_id]
            )
            for node in self.topology.compute_nodes
        }

    def state_for_slot(
        self,
        time_slot: int,
    ) -> InfrastructureState:
        """
        返回下一个连续时隙的基础设施状态。

        为避免同一随机过程被跳跃调用，
        本方法要求从时隙0开始顺序访问。
        """

        if time_slot != self._next_time_slot:
            raise ValueError(
                "MarkovFailureProcess 必须按照"
                "0、1、2、...的顺序访问时隙。"
            )

        # 时隙0使用稳态初始化结果；后续时隙先进行一次边界状态转移。
        if time_slot > 0:
            self._update_states()

        effective_up = self._effective_node_states()
        newly_unavailable = tuple(
            sorted(
                node_id
                for node_id, is_up in effective_up.items()
                if self._previous_effective_up[node_id] and not is_up
            )
        )
        newly_available = tuple(
            sorted(
                node_id
                for node_id, is_up in effective_up.items()
                if not self._previous_effective_up[node_id] and is_up
            )
        )

        snapshot = InfrastructureState(
            time_slot=time_slot,
            domain_up=dict(self._domain_up),
            node_local_up=dict(
                self._node_local_up
            ),
            version=time_slot,
            newly_unavailable_node_ids=newly_unavailable,
            newly_available_node_ids=newly_available,
        )

        self._previous_effective_up = effective_up
        self._next_time_slot += 1

        return snapshot


def build_markov_failure_process(
    config: dict[str, Any],
    topology: LinearRailTopology,
    random_seed: int | None = None,
) -> MarkovFailureProcess:
    """
    根据配置文件创建随机故障过程。
    """

    phase_a = load_phase_a_config(config)
    selected_random_seed = (
        phase_a.failure_base_seed
        if random_seed is None
        else random_seed
    )

    if not isinstance(selected_random_seed, int):
        raise TypeError(
            "随机种子必须是整数。"
        )

    topology_domain_ids = {
        node.fault_domain
        for node in topology.compute_nodes
    }

    missing_domain_ids = (
        topology_domain_ids
        - set(phase_a.domain_rates)
    )

    if missing_domain_ids:
        raise ValueError(
            "以下计算节点故障域缺少可靠性配置："
            f"{sorted(missing_domain_ids)}。"
        )

    return MarkovFailureProcess(
        topology=topology,
        domain_rates={
            domain_id: phase_a.domain_rates[domain_id]
            for domain_id in topology_domain_ids
        },
        node_rates={
            node.node_id: phase_a.node_rates[node.node_id]
            for node in topology.compute_nodes
        },
        slot_seconds=phase_a.fast_slot_seconds,
        random_seed=selected_random_seed,
    )

