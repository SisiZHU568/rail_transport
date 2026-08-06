"""
reliability.py

本文件实现故障域感知的 Serverless SFC 可靠性计算。

可靠性模型包含两层故障：

1. 故障域故障
   例如公共供电、汇聚交换机或回传链路故障。

2. 节点局部故障
   在故障域正常的条件下，单个 MEC 自身仍可能失效。

对于同一函数的多个副本：

- 位于同一故障域的副本会受到共同故障影响；
- 位于不同故障域的副本能够提供更强的故障隔离。

本文件同时计算两种 SFC 可靠性：

1. 函数阶段独立近似值；
2. 考虑共享节点和共享故障域后的精确枚举值。

后续进行可靠性约束判断时，优先使用精确枚举值。
"""

from dataclasses import dataclass
from itertools import product
from math import prod
from typing import Any

from src.entities import SFCType
from src.topology import LinearRailTopology


@dataclass(frozen=True)
class FunctionReliabilityResult:
    """
    一个 SFC 函数的可靠性计算结果。

    Attributes
    ----------
    function_id:
        函数编号。

    replica_node_ids:
        该函数所有副本所在的 MEC 节点。

    fault_domain_ids:
        这些副本涉及的故障域编号。

    distinct_fault_domain_count:
        副本分布到的不同故障域数量。

    fault_domain_diversity_met:
        是否满足最小故障域数量要求。

    availability:
        至少有一个函数副本可用的概率。
    """

    function_id: int
    replica_node_ids: tuple[int, ...]
    fault_domain_ids: tuple[int, ...]
    distinct_fault_domain_count: int
    fault_domain_diversity_met: bool
    availability: float


@dataclass(frozen=True)
class SFCReliabilityResult:
    """
    一条完整 SFC 的可靠性结果。

    approximate_stage_product_availability:
        将各函数阶段看作相互独立时，
        通过函数可靠性连乘得到的近似值。

    exact_shared_failure_availability:
        显式考虑共享节点和共享故障域后，
        通过状态枚举得到的精确值。

    target_met:
        同时满足以下两个条件时才为 True：

        1. 精确可靠性达到 SFC 可靠性阈值；
        2. 所有函数满足故障域隔离要求。
    """

    sfc_id: int
    sfc_name: str
    function_results: tuple[FunctionReliabilityResult, ...]

    approximate_stage_product_availability: float
    exact_shared_failure_availability: float

    reliability_target: float
    fault_domain_diversity_met: bool
    target_met: bool


class FaultDomainReliabilityModel:
    """
    故障域感知可靠性模型。

    当前基本假设：

    1. 不同故障域之间相互独立；
    2. 同一故障域中的节点共享故障域状态；
    3. 在故障域正常的条件下，
       节点局部故障相互独立；
    4. 暂时不单独模拟函数容器自身故障；
    5. 一个节点正常时，部署在该节点上的函数可用。
    """

    def __init__(
        self,
        topology: LinearRailTopology,
        fault_domain_availability: dict[int, float],
        minimum_distinct_fault_domains: int = 1,
    ) -> None:
        """
        创建可靠性模型。

        Parameters
        ----------
        topology:
            轨旁 MEC 拓扑。

        fault_domain_availability:
            故障域编号到故障域可用率的映射。

            例如：

                {
                    0: 0.995,
                    1: 0.995,
                    2: 0.999,
                }

        minimum_distinct_fault_domains:
            同一个函数的副本至少需要分布在多少个故障域。
        """

        if minimum_distinct_fault_domains <= 0:
            raise ValueError(
                "最小故障域数量必须大于 0。"
            )

        if len(fault_domain_availability) == 0:
            raise ValueError(
                "至少需要配置一个故障域。"
            )

        for domain_id, availability in (
            fault_domain_availability.items()
        ):
            if domain_id < 0:
                raise ValueError(
                    "故障域编号不能小于 0。"
                )

            if not 0 < availability <= 1:
                raise ValueError(
                    "故障域可用率必须位于 (0, 1]。"
                )

        # 检查拓扑中所有故障域是否都配置了可用率。
        topology_domain_ids = {
            site.node.fault_domain
            for site in topology.sites
        }

        missing_domain_ids = (
            topology_domain_ids
            - set(fault_domain_availability)
        )

        if missing_domain_ids:
            raise ValueError(
                "以下故障域缺少可用率配置："
                f"{sorted(missing_domain_ids)}"
            )

        self.topology = topology

        # 创建副本，避免外部修改原字典。
        self.fault_domain_availability = dict(
            fault_domain_availability
        )

        self.minimum_distinct_fault_domains = (
            minimum_distinct_fault_domains
        )

    def evaluate_function(
        self,
        function_id: int,
        replica_node_ids: list[int] | tuple[int, ...],
    ) -> FunctionReliabilityResult:
        """
        计算一个函数的多副本可靠性。

        对于故障域 g：

            A(f, g)
            =
            q_g × [1 - Π(1-r_i)]

        其中：

        q_g:
            故障域 g 的可用率。

        r_i:
            节点 i 在故障域正常时的局部可用率。

        函数总可用率：

            A_f
            =
            1 - Π_g [1-A(f,g)]

        该公式能够避免错误地将同故障域副本
        当作完全独立副本。
        """

        if function_id < 0:
            raise ValueError(
                "function_id 不能小于 0。"
            )

        replica_node_ids = tuple(replica_node_ids)

        if len(replica_node_ids) == 0:
            raise ValueError(
                "一个函数至少需要一个部署副本。"
            )

        if len(replica_node_ids) != len(
            set(replica_node_ids)
        ):
            raise ValueError(
                "同一函数的副本节点不能重复。"
            )

        # 按故障域保存各节点局部可用率。
        domain_node_availabilities: dict[
            int,
            list[float],
        ] = {}

        for node_id in replica_node_ids:
            site = self.topology.get_site(node_id)

            domain_id = site.node.fault_domain
            node_local_availability = (
                site.node.reliability
            )

            domain_node_availabilities.setdefault(
                domain_id,
                [],
            ).append(node_local_availability)

        domain_service_availabilities: list[float] = []

        for domain_id, node_availabilities in (
            domain_node_availabilities.items()
        ):
            domain_availability = (
                self.fault_domain_availability[
                    domain_id
                ]
            )

            # 在故障域正常时，
            # 至少一个节点局部可用的概率。
            at_least_one_node_available = (
                1.0
                - prod(
                    1.0 - availability
                    for availability
                    in node_availabilities
                )
            )

            # 故障域正常且至少一个节点正常。
            domain_can_serve_function = (
                domain_availability
                * at_least_one_node_available
            )

            domain_service_availabilities.append(
                domain_can_serve_function
            )

        # 所有故障域都无法提供该函数时，
        # 函数才会完全不可用。
        function_availability = (
            1.0
            - prod(
                1.0 - availability
                for availability
                in domain_service_availabilities
            )
        )

        fault_domain_ids = tuple(
            sorted(domain_node_availabilities)
        )

        distinct_fault_domain_count = len(
            fault_domain_ids
        )

        fault_domain_diversity_met = (
            distinct_fault_domain_count
            >= self.minimum_distinct_fault_domains
        )

        return FunctionReliabilityResult(
            function_id=function_id,
            replica_node_ids=replica_node_ids,
            fault_domain_ids=fault_domain_ids,
            distinct_fault_domain_count=(
                distinct_fault_domain_count
            ),
            fault_domain_diversity_met=(
                fault_domain_diversity_met
            ),
            availability=function_availability,
        )

    def evaluate_sfc(
        self,
        sfc: SFCType,
        function_replica_node_ids: dict[
            int,
            list[int] | tuple[int, ...],
        ],
    ) -> SFCReliabilityResult:
        """
        计算完整 SFC 的可靠性。

        Parameters
        ----------
        sfc:
            当前服务功能链。

        function_replica_node_ids:
            每个函数编号对应的副本节点。

            例如：

                {
                    0: (0, 2),
                    1: (0, 2),
                    2: (0, 2),
                }
        """

        required_function_ids = set(
            sfc.function_ids
        )

        supplied_function_ids = set(
            function_replica_node_ids
        )

        if supplied_function_ids != required_function_ids:
            missing = (
                required_function_ids
                - supplied_function_ids
            )

            extra = (
                supplied_function_ids
                - required_function_ids
            )

            raise ValueError(
                "副本计划与 SFC 函数不一致。"
                f"缺少函数：{sorted(missing)}；"
                f"多余函数：{sorted(extra)}。"
            )

        function_results = tuple(
            self.evaluate_function(
                function_id=function_id,
                replica_node_ids=(
                    function_replica_node_ids[
                        function_id
                    ]
                ),
            )
            for function_id in sfc.function_ids
        )

        # 常见的函数阶段独立近似。
        approximate_stage_product_availability = (
            prod(
                result.availability
                for result in function_results
            )
        )

        # 精确考虑共享节点和共享故障域。
        exact_shared_failure_availability = (
            self._calculate_exact_sfc_availability(
                sfc=sfc,
                function_replica_node_ids=(
                    function_replica_node_ids
                ),
            )
        )

        fault_domain_diversity_met = all(
            result.fault_domain_diversity_met
            for result in function_results
        )

        # 加入极小容差，避免浮点精度影响边界判断。
        reliability_met = (
            exact_shared_failure_availability
            + 1e-12
            >= sfc.reliability_target
        )

        target_met = (
            reliability_met
            and fault_domain_diversity_met
        )

        return SFCReliabilityResult(
            sfc_id=sfc.sfc_id,
            sfc_name=sfc.name,
            function_results=function_results,
            approximate_stage_product_availability=(
                approximate_stage_product_availability
            ),
            exact_shared_failure_availability=(
                exact_shared_failure_availability
            ),
            reliability_target=(
                sfc.reliability_target
            ),
            fault_domain_diversity_met=(
                fault_domain_diversity_met
            ),
            target_met=target_met,
        )

    def _calculate_exact_sfc_availability(
        self,
        sfc: SFCType,
        function_replica_node_ids: dict[
            int,
            list[int] | tuple[int, ...],
        ],
    ) -> float:
        """
        通过状态枚举计算完整 SFC 的精确可用率。

        精确枚举会考虑：

        1. 一个故障域可能同时影响多个节点；
        2. 一个节点可能同时承载多个函数；
        3. 不同函数之间可能共享相同副本节点。

        对当前 5 个 MEC 的小规模调试拓扑，
        完全枚举的计算量很小。
        """

        used_node_ids = sorted(
            {
                node_id
                for replica_nodes
                in function_replica_node_ids.values()
                for node_id in replica_nodes
            }
        )

        used_domain_ids = sorted(
            {
                self.topology
                .get_site(node_id)
                .node
                .fault_domain
                for node_id in used_node_ids
            }
        )

        state_variable_count = (
            len(used_node_ids)
            + len(used_domain_ids)
        )

        # 防止未来拓扑过大时错误使用指数级枚举。
        if state_variable_count > 20:
            raise RuntimeError(
                "精确可靠性枚举的状态变量超过20个。"
                "大规模拓扑应改用蒙特卡洛估计。"
            )

        exact_availability = 0.0

        # 枚举所有故障域状态。
        #
        # False：故障域失效
        # True：故障域正常
        for domain_state_values in product(
            [False, True],
            repeat=len(used_domain_ids),
        ):
            domain_states = dict(
                zip(
                    used_domain_ids,
                    domain_state_values,
                )
            )

            domain_state_probability = 1.0

            for domain_id, is_available in (
                domain_states.items()
            ):
                availability = (
                    self.fault_domain_availability[
                        domain_id
                    ]
                )

                if is_available:
                    domain_state_probability *= (
                        availability
                    )
                else:
                    domain_state_probability *= (
                        1.0 - availability
                    )

            # 枚举所有节点局部状态。
            #
            # 节点最终是否可用还要同时满足：
            # 所属故障域正常。
            for node_state_values in product(
                [False, True],
                repeat=len(used_node_ids),
            ):
                node_local_states = dict(
                    zip(
                        used_node_ids,
                        node_state_values,
                    )
                )

                node_state_probability = 1.0

                for node_id, is_locally_available in (
                    node_local_states.items()
                ):
                    node_availability = (
                        self.topology
                        .get_site(node_id)
                        .node
                        .reliability
                    )

                    if is_locally_available:
                        node_state_probability *= (
                            node_availability
                        )
                    else:
                        node_state_probability *= (
                            1.0 - node_availability
                        )

                state_probability = (
                    domain_state_probability
                    * node_state_probability
                )

                operational_nodes: set[int] = set()

                for node_id in used_node_ids:
                    site = self.topology.get_site(
                        node_id
                    )

                    domain_id = (
                        site.node.fault_domain
                    )

                    node_is_operational = (
                        domain_states[domain_id]
                        and node_local_states[node_id]
                    )

                    if node_is_operational:
                        operational_nodes.add(
                            node_id
                        )

                # 每个 SFC 函数都至少需要一个可用副本。
                sfc_is_available = all(
                    any(
                        node_id in operational_nodes
                        for node_id
                        in function_replica_node_ids[
                            function_id
                        ]
                    )
                    for function_id
                    in sfc.function_ids
                )

                if sfc_is_available:
                    exact_availability += (
                        state_probability
                    )

        return exact_availability


def build_fault_domain_reliability_model(
    config: dict[str, Any],
    topology: LinearRailTopology,
) -> FaultDomainReliabilityModel:
    """
    根据 debug.yaml 创建可靠性模型。
    """

    reliability_config = config["reliability"]

    fault_domain_availability: dict[
        int,
        float,
    ] = {}

    for domain_config in (
        reliability_config["fault_domains"]
    ):
        domain_id = domain_config["domain_id"]
        availability = domain_config["availability"]

        if domain_id in fault_domain_availability:
            raise ValueError(
                f"故障域 {domain_id} 重复配置。"
            )

        fault_domain_availability[domain_id] = (
            availability
        )

    return FaultDomainReliabilityModel(
        topology=topology,
        fault_domain_availability=(
            fault_domain_availability
        ),
        minimum_distinct_fault_domains=(
            reliability_config[
                "minimum_distinct_fault_domains"
            ]
        ),
    )