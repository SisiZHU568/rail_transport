"""
network.py

本文件负责模拟轨旁 MEC 之间的数据传输。

当前采用线性轨旁网络：

MEC-1 ---- MEC-2 ---- MEC-3 ---- MEC-4 ---- MEC-5

例如：

MEC-1 到 MEC-2：1 跳
MEC-1 到 MEC-3：2 跳
MEC-2 到 MEC-5：3 跳

当前传输时延由两部分组成：

1. 数据发送时延；
2. 多跳传播与转发时延。
"""

from typing import Any

from src.topology import LinearRailTopology


class LinearMECNetwork:
    """
    线性轨旁 MEC 网络模型。

    当前假设：

    1. 相邻 MEC 之间使用相同带宽；
    2. 链路是双向的；
    3. 每经过一跳都会产生固定传播与转发时延；
    4. 暂时不考虑排队、丢包和带宽竞争。
    """

    def __init__(
        self,
        node_ids: list[int],
        adjacent_bandwidth_mbps: float,
        propagation_delay_per_hop_ms: float,
    ) -> None:
        """
        创建线性 MEC 网络。

        Parameters
        ----------
        node_ids:
            按照铁路位置顺序排列的 MEC 节点编号。

            例如：
                [0, 1, 2, 3, 4]

        adjacent_bandwidth_mbps:
            相邻 MEC 之间的可用带宽，单位为 Mbps。

        propagation_delay_per_hop_ms:
            每经过一个网络跳数产生的固定时延，单位为毫秒。
        """

        if len(node_ids) == 0:
            raise ValueError("MEC 网络至少需要包含一个节点。")

        if len(node_ids) != len(set(node_ids)):
            raise ValueError("MEC 网络中的节点编号不能重复。")

        if adjacent_bandwidth_mbps <= 0:
            raise ValueError("相邻 MEC 带宽必须大于 0。")

        if propagation_delay_per_hop_ms < 0:
            raise ValueError("每跳传播时延不能小于 0。")

        # 使用 list() 创建副本，避免外部代码修改原始列表后
        # 影响已经创建好的网络对象。
        self.node_ids = list(node_ids)

        self.adjacent_bandwidth_mbps = (
            adjacent_bandwidth_mbps
        )

        self.propagation_delay_per_hop_ms = (
            propagation_delay_per_hop_ms
        )

        # 建立“节点编号 → 在线性网络中的位置”的索引。
        #
        # 例如：
        # node_id 0 位于索引0
        # node_id 1 位于索引1
        # node_id 2 位于索引2
        self._node_index = {
            node_id: index
            for index, node_id in enumerate(self.node_ids)
        }

    @property
    def node_count(self) -> int:
        """
        返回网络中的 MEC 数量。
        """

        return len(self.node_ids)

    def _get_node_index(self, node_id: int) -> int:
        """
        获取节点在线性网络中的位置索引。

        该方法以下划线开头，表示它主要供类内部使用。
        """

        if node_id not in self._node_index:
            raise KeyError(
                f"网络中不存在 node_id={node_id} 的 MEC。"
            )

        return self._node_index[node_id]

    def hop_count(
        self,
        source_node_id: int,
        destination_node_id: int,
    ) -> int:
        """
        计算两个 MEC 之间的跳数。

        Examples
        --------
        MEC-1 到 MEC-2：
            abs(0 - 1) = 1 跳

        MEC-1 到 MEC-3：
            abs(0 - 2) = 2 跳

        MEC-3 到 MEC-3：
            abs(2 - 2) = 0 跳
        """

        source_index = self._get_node_index(
            source_node_id
        )

        destination_index = self._get_node_index(
            destination_node_id
        )

        return abs(source_index - destination_index)

    def path_node_ids(
        self,
        source_node_id: int,
        destination_node_id: int,
    ) -> list[int]:
        """
        返回数据传输路径经过的所有 MEC 节点。

        例如：

        source=0，destination=2：

            [0, 1, 2]

        source=3，destination=1：

            [3, 2, 1]
        """

        source_index = self._get_node_index(
            source_node_id
        )

        destination_index = self._get_node_index(
            destination_node_id
        )

        if source_index <= destination_index:
            return self.node_ids[
                source_index:destination_index + 1
            ]

        reverse_path = self.node_ids[
            destination_index:source_index + 1
        ]

        return list(reversed(reverse_path))

    def transfer_delay_ms(
        self,
        data_size_mb: float,
        source_node_id: int,
        destination_node_id: int,
    ) -> float:
        """
        计算一批数据在两个 MEC 之间的传输时延。

        Parameters
        ----------
        data_size_mb:
            需要传输的数据量，单位为 MB。

        source_node_id:
            数据当前所在 MEC。

        destination_node_id:
            数据需要传输到的 MEC。

        Returns
        -------
        float:
            数据传输时延，单位为毫秒。

        Notes
        -----
        当前计算公式：

            传输时延
            =
            数据发送时延
            +
            每跳传播时延 × 跳数

        因为：

            1 Byte = 8 bit

        所以：

            数据量(Mb) = 数据量(MB) × 8

        数据发送时间：

            秒 = 数据量(Mb) / 带宽(Mbps)

        再乘以1000，将秒转换成毫秒。
        """

        if data_size_mb < 0:
            raise ValueError("传输数据量不能小于 0。")

        # 即使数据量为0，也先检查节点是否存在。
        hop_count = self.hop_count(
            source_node_id=source_node_id,
            destination_node_id=destination_node_id,
        )

        # 数据已经位于目标 MEC，或者没有数据需要传输。
        if hop_count == 0 or data_size_mb == 0:
            return 0.0

        # 将 MB 转换成 Mb。
        data_size_megabits = data_size_mb * 8.0

        # Mbps 表示每秒可以传输多少兆比特。
        serialization_delay_seconds = (
            data_size_megabits
            / self.adjacent_bandwidth_mbps
        )

        serialization_delay_ms = (
            serialization_delay_seconds * 1000.0
        )

        propagation_delay_ms = (
            hop_count
            * self.propagation_delay_per_hop_ms
        )

        return (
            serialization_delay_ms
            + propagation_delay_ms
        )


def build_linear_mec_network(
    config: dict[str, Any],
    topology: LinearRailTopology,
) -> LinearMECNetwork:
    """
    根据配置文件和铁路拓扑创建 MEC 网络。

    Parameters
    ----------
    config:
        debug.yaml 的完整配置字典。

    topology:
        已经创建好的线性铁路拓扑。

    Returns
    -------
    LinearMECNetwork:
        线性轨旁 MEC 网络。
    """

    # topology.sites 已经按照铁路位置从小到大排列。
    node_ids = [
        site.node.node_id
        for site in topology.sites
    ]

    return LinearMECNetwork(
        node_ids=node_ids,
        adjacent_bandwidth_mbps=(
            config["network"]["adjacent_bandwidth_mbps"]
        ),
        propagation_delay_per_hop_ms=(
            config["network"][
                "propagation_delay_per_hop_ms"
            ]
        ),
    )