
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
import random
from typing import Any

from src.topology import LinearRailTopology


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
        确定性故障过程没有内部随机状态。
        """

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

        return InfrastructureState(
            time_slot=time_slot,
            domain_up=domain_up,
            node_local_up=node_local_up,
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

    时隙0默认所有故障域和节点都正常。
    """

    def __init__(
        self,
        topology: LinearRailTopology,
        domain_failure_probabilities: dict[
            int,
            float,
        ],
        domain_recovery_probabilities: dict[
            int,
            float,
        ],
        node_failure_probabilities: dict[
            int,
            float,
        ],
        node_recovery_probabilities: dict[
            int,
            float,
        ],
        random_seed: int,
    ) -> None:
        """
        创建随机故障过程。
        """

        self.topology = topology
        self.random_seed = random_seed

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

        self.domain_failure_probabilities = dict(
            domain_failure_probabilities
        )

        self.domain_recovery_probabilities = dict(
            domain_recovery_probabilities
        )

        self.node_failure_probabilities = dict(
            node_failure_probabilities
        )

        self.node_recovery_probabilities = dict(
            node_recovery_probabilities
        )

        self._validate_probability_dictionary(
            values=self.domain_failure_probabilities,
            expected_ids=set(self.domain_ids),
            parameter_name="故障域失效概率",
        )

        self._validate_probability_dictionary(
            values=self.domain_recovery_probabilities,
            expected_ids=set(self.domain_ids),
            parameter_name="故障域恢复概率",
        )

        self._validate_probability_dictionary(
            values=self.node_failure_probabilities,
            expected_ids=set(self.node_ids),
            parameter_name="节点失效概率",
        )

        self._validate_probability_dictionary(
            values=self.node_recovery_probabilities,
            expected_ids=set(self.node_ids),
            parameter_name="节点恢复概率",
        )

        self._random = random.Random(
            self.random_seed
        )

        self._domain_up: dict[int, bool] = {}
        self._node_local_up: dict[int, bool] = {}
        self._next_time_slot = 0

        self.reset()

    @staticmethod
    def _validate_probability_dictionary(
        values: dict[int, float],
        expected_ids: set[int],
        parameter_name: str,
    ) -> None:
        """
        检查概率字典是否合法。
        """

        if set(values) != expected_ids:
            raise ValueError(
                f"{parameter_name}的编号集合不完整。"
            )

        for entity_id, probability in values.items():
            if not 0 <= probability <= 1:
                raise ValueError(
                    f"{parameter_name}中编号"
                    f"{entity_id}的概率不在[0,1]。"
                )

    def reset(self) -> None:
        """
        恢复随机种子和初始正常状态。
        """

        self._random = random.Random(
            self.random_seed
        )

        self._domain_up = {
            domain_id: True
            for domain_id in self.domain_ids
        }

        self._node_local_up = {
            node_id: True
            for node_id in self.node_ids
        }

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
                    self.domain_failure_probabilities[
                        domain_id
                    ]
                )

                if (
                    self._random.random()
                    < transition_probability
                ):
                    self._domain_up[domain_id] = False
            else:
                transition_probability = (
                    self.domain_recovery_probabilities[
                        domain_id
                    ]
                )

                if (
                    self._random.random()
                    < transition_probability
                ):
                    self._domain_up[domain_id] = True

        for node_id in self.node_ids:
            currently_up = self._node_local_up[
                node_id
            ]

            if currently_up:
                transition_probability = (
                    self.node_failure_probabilities[
                        node_id
                    ]
                )

                if (
                    self._random.random()
                    < transition_probability
                ):
                    self._node_local_up[node_id] = False
            else:
                transition_probability = (
                    self.node_recovery_probabilities[
                        node_id
                    ]
                )

                if (
                    self._random.random()
                    < transition_probability
                ):
                    self._node_local_up[node_id] = True

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

        # 时隙0使用初始全正常状态。
        #
        # 从时隙1开始，每个时隙先进行一次状态转移。
        if time_slot > 0:
            self._update_states()

        snapshot = InfrastructureState(
            time_slot=time_slot,
            domain_up=dict(self._domain_up),
            node_local_up=dict(
                self._node_local_up
            ),
        )

        self._next_time_slot += 1

        return snapshot


def _failure_probability_from_availability(
    availability: float,
    recovery_probability: float,
) -> float:
    """
    根据长期可用率和恢复概率计算失效概率。

    二状态马尔可夫稳态关系：

        A = μ / (λ + μ)

    因此：

        λ = μ(1-A) / A
    """

    if not 0 < availability <= 1:
        raise ValueError(
            "长期可用率必须位于 (0, 1]。"
        )

    if not 0 <= recovery_probability <= 1:
        raise ValueError(
            "恢复概率必须位于 [0, 1]。"
        )

    failure_probability = (
        recovery_probability
        * (1.0 - availability)
        / availability
    )

    if failure_probability > 1:
        raise ValueError(
            "根据当前可用率和恢复概率得到的"
            "失效概率大于1。"
        )

    return failure_probability


def build_markov_failure_process(
    config: dict[str, Any],
    topology: LinearRailTopology,
    random_seed: int | None = None,
) -> MarkovFailureProcess:
    """
    根据配置文件创建随机故障过程。
    """

    runtime_config = config["runtime_failure"]
    selected_random_seed = (
        runtime_config["random_seed"]
        if random_seed is None
        else random_seed
    )

    if not isinstance(selected_random_seed, int):
        raise TypeError(
            "随机种子必须是整数。"
        )

    domain_recovery_probability = (
        runtime_config[
            "domain_recovery_probability_per_slot"
        ]
    )

    node_recovery_probability = (
        runtime_config[
            "node_recovery_probability_per_slot"
        ]
    )

    fault_domain_availability = {
        item["domain_id"]: item["availability"]
        for item in config["reliability"][
            "fault_domains"
        ]
    }

    topology_domain_ids = {
        node.fault_domain
        for node in topology.compute_nodes
    }

    missing_domain_ids = (
        topology_domain_ids
        - set(fault_domain_availability)
    )

    if missing_domain_ids:
        raise ValueError(
            "以下计算节点故障域缺少可靠性配置："
            f"{sorted(missing_domain_ids)}。"
        )

    domain_failure_probabilities = {
        domain_id: (
            _failure_probability_from_availability(
                availability=(
                    fault_domain_availability[
                        domain_id
                    ]
                ),
                recovery_probability=(
                    domain_recovery_probability
                ),
            )
        )
        for domain_id in topology_domain_ids
    }

    domain_recovery_probabilities = {
        domain_id: domain_recovery_probability
        for domain_id in topology_domain_ids
    }

    node_failure_probabilities = {
        node.node_id: (
            _failure_probability_from_availability(
                availability=(
                    node.reliability
                ),
                recovery_probability=(
                    node_recovery_probability
                ),
            )
        )
        for node in topology.compute_nodes
    }

    node_recovery_probabilities = {
        node.node_id: node_recovery_probability
        for node in topology.compute_nodes
    }

    return MarkovFailureProcess(
        topology=topology,
        domain_failure_probabilities=(
            domain_failure_probabilities
        ),
        domain_recovery_probabilities=(
            domain_recovery_probabilities
        ),
        node_failure_probabilities=(
            node_failure_probabilities
        ),
        node_recovery_probabilities=(
            node_recovery_probabilities
        ),
        random_seed=selected_random_seed,
    )

