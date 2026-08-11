"""生成不读取未来信息的确定性 DPPO 仿真教师动作。"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.dppo_action_space import (
    DPPOActionSpace,
    DecodedDPPOAction,
    DecodedFunctionAction,
)
from src.dppo_slow_timescale_env import DPPOSlowTimescaleEnvironment
from src.dppo_state_encoder import DPPONodeObservation, DPPOStateSnapshot
from src.entities import NodeType
from src.scenario_dimensions import ScenarioDimensions


@dataclass(frozen=True)
class TeacherProposal:
    """同时保存可训练连续向量和便于 review 的解码动作。"""

    teacher_name: str
    relaxed_action: np.ndarray
    decoded_action: DecodedDPPOAction


@dataclass(frozen=True)
class SimulationTeacherContext:
    """保存教师允许使用的公开、静态场景信息。"""

    dimensions: ScenarioDimensions
    action_space: DPPOActionSpace
    fault_domain_pairs: tuple[tuple[int, int], ...]
    cloud_node_ids: frozenset[int]

    @property
    def fault_domains(self) -> dict[int, int]:
        """返回节点到故障域的副本，防止教师修改场景拓扑。"""

        return dict(self.fault_domain_pairs)


def ranking_scores(node_ids: Sequence[int]) -> dict[int, float]:
    """把完整节点名次均匀映射到 ``[1, -1]`` 连续评分。"""

    ordered_ids = tuple(node_ids)
    if len(ordered_ids) != len(set(ordered_ids)):
        raise ValueError("节点排名不能包含重复节点。")
    denominator = max(1, len(ordered_ids) - 1)
    return {
        node_id: 1.0 - 2.0 * rank / denominator
        for rank, node_id in enumerate(ordered_ids)
    }


class SimulationTeacher:
    """仿真教师基类：只把当前公开快照转换成确定性联合动作。"""

    teacher_name = "base"

    def __init__(self, context: SimulationTeacherContext) -> None:
        self.context = context

    def propose(self, snapshot: DPPOStateSnapshot) -> TeacherProposal:
        """对整条 SFC 一次性生成节点排名、副本数和保留时间。"""

        self._validate_snapshot(snapshot)
        ranking = self._rank_nodes(snapshot.node_observations)
        primary_retention, backup_retention = self._retention_seconds(snapshot)
        replica_count = self._replica_count()
        function_actions = tuple(
            DecodedFunctionAction(
                function_id=function_id,
                ranked_node_ids=ranking,
                replica_count=replica_count,
                primary_retention_seconds=primary_retention,
                backup_retention_seconds=backup_retention,
            )
            for function_id in self.context.dimensions.function_ids
        )
        relaxed_action = self.context.action_space.encode_teacher_action(
            function_actions
        )
        # 重新解码可同时验证连续向量与教师声明的离散语义完全一致。
        decoded_action = self.context.action_space.decode(relaxed_action)
        return TeacherProposal(
            teacher_name=self.teacher_name,
            relaxed_action=relaxed_action,
            decoded_action=decoded_action,
        )

    def _replica_count(self) -> int:
        """成本型基准使用配置允许的最少副本。"""

        return self.context.action_space.minimum_replicas

    def _validate_snapshot(self, snapshot: DPPOStateSnapshot) -> None:
        """拒绝错规模或乱序快照，避免训练标签与状态模式错位。"""

        if not isinstance(snapshot, DPPOStateSnapshot):
            raise TypeError("snapshot 必须是 DPPOStateSnapshot。")
        node_ids = tuple(item.node_id for item in snapshot.node_observations)
        if node_ids != self.context.dimensions.compute_node_ids:
            raise ValueError("教师快照必须包含按配置顺序排列的全部计算节点。")
        function_ids = tuple(
            item.function_id for item in snapshot.function_observations
        )
        if function_ids != self.context.dimensions.function_ids:
            raise ValueError("教师快照必须包含按配置顺序排列的全部 VNF。")

    def _rank_nodes(
        self,
        observations: tuple[DPPONodeObservation, ...],
    ) -> tuple[int, ...]:
        raise NotImplementedError

    def _retention_seconds(
        self,
        snapshot: DPPOStateSnapshot,
    ) -> tuple[float, float]:
        raise NotImplementedError

    def _cost_key(self, item: DPPONodeObservation) -> tuple[object, ...]:
        """固定成本字典序：可运行、边缘、时延、资源、风险、可靠性。"""

        free_resource = min(item.free_cpu_ratio, item.free_memory_ratio)
        return (
            not item.operational,
            item.node_id in self.context.cloud_node_ids,
            item.normalized_delay_from_serving,
            -free_resource,
            item.predicted_failure_probability,
            -item.base_availability,
            item.node_id,
        )

    def _reliability_key(
        self,
        item: DPPONodeObservation,
    ) -> tuple[object, ...]:
        """固定可靠性字典序：可运行、风险、可用率、资源、时延。"""

        free_resource = min(item.free_cpu_ratio, item.free_memory_ratio)
        return (
            not item.operational,
            item.predicted_failure_probability,
            -item.base_availability,
            -free_resource,
            item.normalized_delay_from_serving,
            item.node_id,
        )

    def _diverse_prefix(
        self,
        ordered_observations: Sequence[DPPONodeObservation],
        replica_count: int,
    ) -> tuple[int, ...]:
        """在保留基础优先级的同时，让副本前缀尽量跨故障域。"""

        fault_domains = self.context.fault_domains
        selected: list[DPPONodeObservation] = []
        used_domains: set[int] = set()
        # 第一轮只选不同故障域，保证可靠性约束优先于后续成本排序。
        for item in ordered_observations:
            if len(selected) >= replica_count:
                break
            domain_id = fault_domains[item.node_id]
            if item.operational and domain_id not in used_domains:
                selected.append(item)
                used_domains.add(domain_id)
        # 极端故障下故障域可能不足；此时补齐排名，实际可行性仍交给投影器。
        selected_ids = {item.node_id for item in selected}
        remaining = [
            item for item in ordered_observations if item.node_id not in selected_ids
        ]
        return tuple(item.node_id for item in (*selected, *remaining))


class CostTeacher(SimulationTeacher):
    """偏向低时延、低云成本和较少副本的启发式教师。"""

    teacher_name = "cost"

    def _rank_nodes(
        self,
        observations: tuple[DPPONodeObservation, ...],
    ) -> tuple[int, ...]:
        return tuple(item.node_id for item in sorted(observations, key=self._cost_key))

    def _retention_seconds(
        self,
        snapshot: DPPOStateSnapshot,
    ) -> tuple[float, float]:
        maximum = self.context.action_space.maximum_retention_seconds
        # 低成本策略只根据已观测平均负载保留短时间，备用保留再减半。
        primary_ratio = max(0.10, min(0.50, snapshot.normalized_mean_requests))
        return maximum * primary_ratio, maximum * primary_ratio / 2.0


class ReliabilityTeacher(SimulationTeacher):
    """偏向低故障风险、跨故障域和长保留时间的启发式教师。"""

    teacher_name = "reliability"

    def _replica_count(self) -> int:
        """可靠性教师使用配置允许的最多副本。"""

        return self.context.action_space.maximum_replicas

    def _rank_nodes(
        self,
        observations: tuple[DPPONodeObservation, ...],
    ) -> tuple[int, ...]:
        ordered = sorted(observations, key=self._reliability_key)
        return self._diverse_prefix(ordered, self._replica_count())

    def _retention_seconds(
        self,
        snapshot: DPPOStateSnapshot,
    ) -> tuple[float, float]:
        maximum = self.context.action_space.maximum_retention_seconds
        return maximum, maximum


class BalancedTeacher(SimulationTeacher):
    """在最低副本数上增加一档，再按边缘成本排序的启发式教师。"""

    teacher_name = "balanced"

    def _replica_count(self) -> int:
        """平衡教师采用低成本与高可靠之间的下一档副本数。"""

        action_space = self.context.action_space
        return min(
            action_space.minimum_replicas + 1,
            action_space.maximum_replicas,
        )

    def _rank_nodes(
        self,
        observations: tuple[DPPONodeObservation, ...],
    ) -> tuple[int, ...]:
        ordered = sorted(observations, key=self._cost_key)
        return self._diverse_prefix(ordered, self._replica_count())

    def _retention_seconds(
        self,
        snapshot: DPPOStateSnapshot,
    ) -> tuple[float, float]:
        maximum = self.context.action_space.maximum_retention_seconds
        # 峰值负载或故障风险升高时延长主备保留，但不引入可调奖励权重。
        primary_ratio = max(0.50, snapshot.normalized_peak_requests)
        backup_ratio = max(0.25, snapshot.global_failure_risk)
        return maximum * primary_ratio, maximum * backup_ratio


def _teacher_context(
    scenario: DPPOSlowTimescaleEnvironment,
) -> SimulationTeacherContext:
    """从统一 DPPO 场景提取教师可读取的静态公开信息。"""

    if not isinstance(scenario, DPPOSlowTimescaleEnvironment):
        raise TypeError("scenario 必须由 build_dppo_scenario 创建。")
    nodes = scenario.execution_core.topology.compute_nodes
    return SimulationTeacherContext(
        dimensions=scenario.dimensions,
        action_space=scenario.action_space,
        fault_domain_pairs=tuple(
            (node.node_id, node.fault_domain) for node in nodes
        ),
        cloud_node_ids=frozenset(
            node.node_id for node in nodes if node.node_type is NodeType.CLOUD
        ),
    )


def build_simulation_teacher(
    name: str,
    scenario: DPPOSlowTimescaleEnvironment,
) -> SimulationTeacher:
    """按显式名称构造教师；名称错误时立即给出可定位提示。"""

    teacher_types: dict[str, type[SimulationTeacher]] = {
        "cost": CostTeacher,
        "reliability": ReliabilityTeacher,
        "balanced": BalancedTeacher,
    }
    try:
        teacher_type = teacher_types[name]
    except KeyError as error:
        allowed = "、".join(teacher_types)
        raise ValueError(f"未知的仿真教师 {name!r}；可选值：{allowed}。") from error
    return teacher_type(_teacher_context(scenario))
