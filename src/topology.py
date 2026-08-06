"""
topology.py

本文件负责构建铁路沿线的轨旁 MEC 拓扑。

当前阶段采用最简单的“直线铁路”模型：

MEC-1 ---- MEC-2 ---- MEC-3 ---- MEC-4 ---- MEC-5

每个轨旁 MEC 都有：

1. 在线路上的位置；
2. 无线覆盖半径；
3. CPU 和内存资源；
4. 节点可靠性；
5. 所属故障域。

后续的列车移动、函数预热和跨边缘路由，
都需要使用本文件中的拓扑信息。
"""

from dataclasses import dataclass
from typing import Any

from src.entities import EdgeNode, NodeType


@dataclass(frozen=True)
class TracksideSite:
    """
    一个轨旁 MEC 站点。

    TracksideSite 与 EdgeNode 的区别：

    EdgeNode:
        描述节点的计算资源、内存和可靠性。

    TracksideSite:
        在 EdgeNode 的基础上增加地理位置和覆盖半径。

    Parameters
    ----------
    node:
        该站点对应的轨旁 MEC 节点。

    position_m:
        MEC 在线路上的位置，单位为米。

    coverage_radius_m:
        MEC 的无线覆盖半径，单位为米。
    """

    node: EdgeNode
    position_m: float
    coverage_radius_m: float

    def __post_init__(self) -> None:
        """
        对象创建后自动检查参数。
        """

        # 本类只能用于轨旁 MEC。
        if self.node.node_type != NodeType.TRACKSIDE:
            raise ValueError("TracksideSite 只能绑定轨旁 MEC 节点。")

        if self.position_m < 0:
            raise ValueError("轨旁 MEC 的位置不能小于 0。")

        if self.coverage_radius_m <= 0:
            raise ValueError("MEC 覆盖半径必须大于 0。")


class LinearRailTopology:
    """
    直线铁路拓扑。

    本类保存多个按位置排列的轨旁 MEC，
    并提供以下功能：

    1. 查询某个 MEC；
    2. 判断列车当前连接哪个 MEC；
    3. 查询下一个 MEC；
    4. 计算 MEC 切换边界；
    5. 估算当前覆盖区剩余驻留时间。
    """

    def __init__(self, sites: list[TracksideSite]) -> None:
        """
        创建直线铁路拓扑。

        Parameters
        ----------
        sites:
            所有轨旁 MEC 站点。
        """

        if len(sites) == 0:
            raise ValueError("铁路拓扑至少需要一个轨旁 MEC。")

        # 按 MEC 在线路上的位置从小到大排序。
        self.sites = sorted(
            sites,
            key=lambda site: site.position_m,
        )

        # 检查节点编号是否重复。
        node_ids = [site.node.node_id for site in self.sites]

        if len(node_ids) != len(set(node_ids)):
            raise ValueError("轨旁 MEC 的 node_id 不能重复。")

        # 检查位置是否重复。
        positions = [site.position_m for site in self.sites]

        if len(positions) != len(set(positions)):
            raise ValueError("两个轨旁 MEC 不能位于完全相同的位置。")

    @property
    def mec_count(self) -> int:
        """
        返回轨旁 MEC 数量。
        """

        return len(self.sites)

    @property
    def route_start_m(self) -> float:
        """
        返回仿真线路的起点。
        """

        return self.sites[0].position_m

    @property
    def route_end_m(self) -> float:
        """
        返回仿真线路的终点。

        当前将最后一个 MEC 所在位置视为线路终点。
        """

        return self.sites[-1].position_m

    def get_site(self, node_id: int) -> TracksideSite:
        """
        根据节点编号查询 MEC 站点。

        Parameters
        ----------
        node_id:
            MEC 节点编号。

        Returns
        -------
        TracksideSite:
            对应的轨旁 MEC 站点。

        Raises
        ------
        KeyError:
            没有找到指定节点。
        """

        for site in self.sites:
            if site.node.node_id == node_id:
                return site

        raise KeyError(f"没有找到 node_id={node_id} 的轨旁 MEC。")

    def serving_mec(self, train_position_m: float) -> int | None:
        """
        判断列车当前位置应连接哪个轨旁 MEC。

        处理规则：

        1. 首先找出覆盖列车的所有 MEC；
        2. 如果有多个 MEC 同时覆盖列车，则选择距离最近的 MEC；
        3. 如果没有任何 MEC 覆盖列车，则返回 None。

        Parameters
        ----------
        train_position_m:
            列车在线路上的当前位置，单位为米。

        Returns
        -------
        int | None:
            当前接入 MEC 的 node_id。
            如果列车不在任何 MEC 覆盖范围内，则返回 None。
        """

        if train_position_m < 0:
            raise ValueError("列车位置不能小于 0。")

        covered_sites: list[TracksideSite] = []

        for site in self.sites:
            distance = abs(train_position_m - site.position_m)

            if distance <= site.coverage_radius_m:
                covered_sites.append(site)

        if len(covered_sites) == 0:
            return None

        # min() 返回距离列车最近的站点。
        # 第二个排序条件 position_m 用于解决距离完全相同的情况。
        nearest_site = min(
            covered_sites,
            key=lambda site: (
                abs(train_position_m - site.position_m),
                site.position_m,
            ),
        )

        return nearest_site.node.node_id

    def next_mec(self, current_node_id: int) -> int | None:
        """
        查询当前 MEC 后面的下一个 MEC。

        当前模型只考虑列车沿线路正方向移动：

        MEC-1 → MEC-2 → MEC-3 → MEC-4 → MEC-5

        Parameters
        ----------
        current_node_id:
            当前 MEC 编号。

        Returns
        -------
        int | None:
            下一个 MEC 的编号。

            如果当前已经是最后一个 MEC，则返回 None。
        """

        for index, site in enumerate(self.sites):
            if site.node.node_id == current_node_id:
                # 已经是最后一个 MEC。
                if index == len(self.sites) - 1:
                    return None

                return self.sites[index + 1].node.node_id

        raise KeyError(f"没有找到 node_id={current_node_id} 的轨旁 MEC。")

    def handover_boundary_m(self, current_node_id: int) -> float:
        """
        计算当前 MEC 与下一 MEC 之间的切换边界。

        当前使用两个 MEC 位置的中点作为切换边界。

        例如：

        MEC-1 位于 0 米；
        MEC-2 位于 2000 米；

        则二者之间的切换边界是：

        (0 + 2000) / 2 = 1000 米。

        Parameters
        ----------
        current_node_id:
            当前 MEC 编号。

        Returns
        -------
        float:
            切换位置，单位为米。
        """

        current_site = self.get_site(current_node_id)
        next_node_id = self.next_mec(current_node_id)

        # 如果已经是最后一个 MEC，
        # 则把线路终点视为最后的边界。
        if next_node_id is None:
            return self.route_end_m

        next_site = self.get_site(next_node_id)

        return (current_site.position_m + next_site.position_m) / 2.0

    def remaining_dwell_time_s(
        self,
        train_position_m: float,
        train_speed_mps: float,
        current_node_id: int,
    ) -> float:
        """
        估算列车在当前 MEC 服务范围中的剩余驻留时间。

        计算公式：

            剩余驻留时间
            =
            距离切换边界的距离 / 列车速度

        Parameters
        ----------
        train_position_m:
            列车当前位置，单位为米。

        train_speed_mps:
            列车速度，单位为米/秒。

        current_node_id:
            当前接入 MEC 编号。

        Returns
        -------
        float:
            剩余驻留时间，单位为秒。
        """

        if train_speed_mps < 0:
            raise ValueError("当前模型不支持负速度。")

        # 速度为 0 表示列车停止，
        # 此时理论上不会发生切换。
        if train_speed_mps == 0:
            return float("inf")

        boundary_m = self.handover_boundary_m(current_node_id)

        remaining_distance_m = max(
            boundary_m - train_position_m,
            0.0,
        )

        return remaining_distance_m / train_speed_mps


def build_linear_topology(
    config: dict[str, Any],
) -> LinearRailTopology:
    """
    根据配置文件创建直线铁路拓扑。

    Parameters
    ----------
    config:
        从 debug.yaml 中读取的完整配置字典。

    Returns
    -------
    LinearRailTopology:
        构建完成的铁路拓扑。
    """

    mec_count = config["topology"]["mec_count"]
    spacing_m = config["topology"]["mec_spacing_m"]
    coverage_radius_m = config["topology"]["mec_coverage_radius_m"]

    cpu_capacity = config["node_resources"]["mec_cpu_capacity"]
    memory_capacity_mb = config["node_resources"]["mec_memory_mb"]
    reliability = config["node_resources"]["mec_reliability"]

    if mec_count <= 0:
        raise ValueError("mec_count 必须大于 0。")

    if spacing_m <= 0:
        raise ValueError("MEC 间距必须大于 0。")

    sites: list[TracksideSite] = []

    for index in range(mec_count):
        # 创建轨旁 MEC 的计算节点。
        node = EdgeNode(
            node_id=index,
            name=f"MEC-{index + 1}",
            node_type=NodeType.TRACKSIDE,
            cpu_capacity=cpu_capacity,
            memory_capacity_mb=memory_capacity_mb,
            reliability=reliability,

            # 当前规定每两个相邻 MEC 属于同一故障域。
            fault_domain=index // 2,
        )

        # MEC 在线路上的位置：
        # MEC-1 = 0 米
        # MEC-2 = 2000 米
        # MEC-3 = 4000 米
        # ...
        position_m = index * spacing_m

        site = TracksideSite(
            node=node,
            position_m=position_m,
            coverage_radius_m=coverage_radius_m,
        )

        sites.append(site)

    return LinearRailTopology(sites)