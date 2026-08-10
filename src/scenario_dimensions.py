"""根据场景配置统一计算 DPPO 的节点顺序、VNF 顺序和向量维度。"""

from collections.abc import Sequence
from dataclasses import dataclass

from src.entities import ServerlessFunction
from src.topology import LinearRailTopology


@dataclass(frozen=True)
class ScenarioDimensions:
    """保存一个训练规模的唯一维度定义。

    其他模块只读取本对象，不再分别写死 ``5`` 个 MEC、``3`` 个 VNF、
    ``78`` 维状态或 ``27`` 维动作。改变实验规模时，只需重建本对象。
    """

    mec_node_ids: tuple[int, ...]
    cloud_node_id: int
    function_ids: tuple[int, ...]

    @classmethod
    def from_scenario(
        cls,
        topology: LinearRailTopology,
        functions: Sequence[ServerlessFunction],
    ) -> "ScenarioDimensions":
        """按真实拓扑和配置化 VNF 的既有顺序生成唯一维度对象。"""

        if topology.cloud_node is None:
            raise ValueError("创建 DPPO 维度时必须配置中心云节点。")
        return cls(
            mec_node_ids=tuple(site.node.node_id for site in topology.sites),
            cloud_node_id=topology.cloud_node.node_id,
            function_ids=tuple(function.function_id for function in functions),
        )

    def __post_init__(self) -> None:
        """尽早拒绝会让向量切片含义不唯一的场景配置。"""

        compute_ids = self.mec_node_ids + (self.cloud_node_id,)
        if len(set(compute_ids)) != len(compute_ids):
            raise ValueError("计算节点 ID 必须唯一。")
        if any(
            isinstance(node_id, bool)
            or not isinstance(node_id, int)
            or node_id < 0
            for node_id in compute_ids
        ):
            raise ValueError("计算节点 ID 必须是非负整数。")
        if len(compute_ids) < 3:
            raise ValueError("支持三副本时至少需要三个计算节点。")

        if not self.function_ids:
            raise ValueError("VNF ID 列表不能为空。")
        if len(set(self.function_ids)) != len(self.function_ids):
            raise ValueError("VNF ID 必须唯一。")
        if any(
            isinstance(function_id, bool)
            or not isinstance(function_id, int)
            or function_id < 0
            for function_id in self.function_ids
        ):
            raise ValueError("VNF ID 必须是非负整数。")

    @property
    def compute_node_ids(self) -> tuple[int, ...]:
        """按“全部轨旁 MEC，最后是中心云”的稳定顺序返回计算节点。"""

        return self.mec_node_ids + (self.cloud_node_id,)

    @property
    def mec_count(self) -> int:
        """返回轨旁 MEC 数量。"""

        return len(self.mec_node_ids)

    @property
    def compute_node_count(self) -> int:
        """返回可部署 VNF 的 MEC 与中心云节点总数。"""

        return len(self.compute_node_ids)

    @property
    def function_count(self) -> int:
        """返回 SFC 中声明的 VNF 数量。"""

        return len(self.function_ids)

    @property
    def state_dim(self) -> int:
        """使用研究方案中的平铺状态公式计算维度。"""

        return 8 * self.mec_count + 4 * self.function_count + 26

    @property
    def action_dim(self) -> int:
        """计算联合动作：节点评分、每 VNF 副本数和两种保留时间。"""

        return self.function_count * (self.compute_node_count + 3)
