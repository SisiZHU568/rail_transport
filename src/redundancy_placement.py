"""
redundancy_placement.py

本文件负责为 SFC 函数规划主实例和备用实例。

当前策略：

1. 主实例部署在列车当前接入 MEC；
2. 备用实例优先选择距离主实例最近的其他 MEC；
3. 备用实例必须属于不同故障域；
4. 当前整条 SFC 的函数使用相同主备节点。

例如：

当前接入 MEC-1，属于故障域0。

候选节点：

MEC-2：故障域0，距离最近，但故障域相同，不选；
MEC-3：故障域1，选择为备用节点。

最终：

主实例：MEC-1
备用实例：MEC-3
"""

from dataclasses import dataclass
from typing import Any

from src.entities import SFCType, TrainState
from src.topology import LinearRailTopology


@dataclass
class ReplicaPlacementPlan:
    """
    一条 SFC 的副本部署计划。

    function_replica_node_ids:
        函数编号到副本节点列表的映射。

        例如：

            {
                0: (0, 2),
                1: (0, 2),
                2: (0, 2),
            }
    """

    policy_id: str
    display_name: str

    primary_node_id: int
    shared_replica_node_ids: tuple[int, ...]

    function_replica_node_ids: dict[
        int,
        tuple[int, ...],
    ]


class ReliabilityAwareReplicaPlanner:
    """
    故障域感知主备副本规划器。
    """

    def __init__(
        self,
        replica_count: int,
        minimum_distinct_fault_domains: int,
    ) -> None:
        """
        Parameters
        ----------
        replica_count:
            每个函数的副本数量。

            2 表示：
                1个主实例 + 1个备用实例。

        minimum_distinct_fault_domains:
            副本至少覆盖多少个不同故障域。
        """

        if replica_count <= 0:
            raise ValueError(
                "副本数量必须大于 0。"
            )

        if minimum_distinct_fault_domains <= 0:
            raise ValueError(
                "最小故障域数量必须大于 0。"
            )

        if minimum_distinct_fault_domains > replica_count:
            raise ValueError(
                "最小故障域数量不能大于副本数量。"
            )

        self.replica_count = replica_count

        self.minimum_distinct_fault_domains = (
            minimum_distinct_fault_domains
        )

    def plan(
        self,
        sfc: SFCType,
        train_state: TrainState,
        topology: LinearRailTopology,
    ) -> ReplicaPlacementPlan:
        """
        为当前 SFC 规划主备副本。

        主节点固定为当前接入 MEC。

        备用节点按照以下顺序选择：

        1. 优先满足不同故障域；
        2. 在满足隔离条件的节点中，
           选择距离主节点最近的节点；
        3. 距离相同时，选择 node_id 较小的节点。
        """

        primary_site = topology.get_site(
            train_state.serving_mec
        )

        selected_node_ids = [
            primary_site.node.node_id
        ]

        selected_fault_domains = {
            primary_site.node.fault_domain
        }

        # 除主节点外的所有候选 MEC。
        candidate_sites = [
            site
            for site in topology.sites
            if site.node.node_id
            != primary_site.node.node_id
        ]

        # 按照与主节点的地理距离排序。
        candidate_sites.sort(
            key=lambda site: (
                abs(
                    site.position_m
                    - primary_site.position_m
                ),
                site.node.node_id,
            )
        )

        # ----------------------------------------------------
        # 第一轮：
        # 优先选择新的故障域。
        # ----------------------------------------------------

        for site in candidate_sites:
            if len(selected_node_ids) >= self.replica_count:
                break

            candidate_domain = (
                site.node.fault_domain
            )

            if (
                candidate_domain
                in selected_fault_domains
            ):
                continue

            selected_node_ids.append(
                site.node.node_id
            )

            selected_fault_domains.add(
                candidate_domain
            )

        if (
            len(selected_fault_domains)
            < self.minimum_distinct_fault_domains
        ):
            raise RuntimeError(
                "当前拓扑无法满足最小故障域隔离要求。"
            )

        # ----------------------------------------------------
        # 第二轮：
        # 如果副本数量仍不足，
        # 再从剩余的最近节点中补充。
        # ----------------------------------------------------

        if len(selected_node_ids) < self.replica_count:
            for site in candidate_sites:
                if (
                    len(selected_node_ids)
                    >= self.replica_count
                ):
                    break

                if (
                    site.node.node_id
                    in selected_node_ids
                ):
                    continue

                selected_node_ids.append(
                    site.node.node_id
                )

        if len(selected_node_ids) < self.replica_count:
            raise RuntimeError(
                "当前拓扑中的 MEC 数量不足，"
                "无法满足副本数量要求。"
            )

        shared_replica_node_ids = tuple(
            selected_node_ids
        )

        function_replica_node_ids = {
            function_id: shared_replica_node_ids
            for function_id in sfc.function_ids
        }

        return ReplicaPlacementPlan(
            policy_id="fault_domain_aware",
            display_name="故障域感知主备部署",
            primary_node_id=(
                primary_site.node.node_id
            ),
            shared_replica_node_ids=(
                shared_replica_node_ids
            ),
            function_replica_node_ids=(
                function_replica_node_ids
            ),
        )


def build_reliability_aware_replica_planner(
    config: dict[str, Any],
    replica_count: int | None = None,
) -> ReliabilityAwareReplicaPlanner:
    """
    根据配置文件创建主备副本规划器。

    ``replica_count`` 用于接收慢层本周期选出的副本数；
    未传入时继续使用配置文件中的默认值，兼容已有实验脚本。
    """

    reliability_config = config["reliability"]
    selected_replica_count = (
        int(reliability_config["replica_count"])
        if replica_count is None
        else int(replica_count)
    )

    return ReliabilityAwareReplicaPlanner(
        replica_count=selected_replica_count,
        minimum_distinct_fault_domains=(
            int(
                reliability_config[
                    "minimum_distinct_fault_domains"
                ]
            )
        ),
    )
