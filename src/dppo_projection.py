"""把 DPPO 的宽松连续动作投影为满足必要条件的部署意图。"""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType

from src.dppo_action_space import DecodedDPPOAction, DecodedFunctionAction
from src.scenario_dimensions import ScenarioDimensions
from src.sfc_deployment_intent import FunctionDeploymentIntent


@dataclass(frozen=True)
class ProjectionResourceDemand:
    """一个 VNF 副本在投影阶段需要检查的 CPU 和内存需求。"""

    cpu: float
    memory_mb: float

    def __post_init__(self) -> None:
        """资源需求必须是具有物理意义的正有限数。"""

        for field_name, value in (
            ("cpu", self.cpu),
            ("memory_mb", self.memory_mb),
        ):
            if isinstance(value, bool):
                raise ValueError(f"{field_name} must be a positive finite number.")
            try:
                number = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{field_name} must be a positive finite number."
                ) from error
            if not math.isfinite(number) or number <= 0.0:
                raise ValueError(
                    f"{field_name} must be a positive finite number."
                )


@dataclass(frozen=True)
class ProjectionResult:
    """保存投影后的意图以及论文实验需要的可行性诊断。"""

    function_intents: tuple[FunctionDeploymentIntent, ...] | None
    raw_feasible: bool
    success: bool
    reasons: tuple[str, ...]
    changed_assignment_count: int
    requested_assignment_count: int
    change_ratio: float
    fault_domains: Mapping[int, int]

    def __post_init__(self) -> None:
        """防止成功标志、意图和统计量之间出现互相矛盾的数据。"""

        if self.success != (self.function_intents is not None):
            raise ValueError("success must agree with function_intents.")
        if self.raw_feasible and not self.success:
            raise ValueError("A raw-feasible action cannot have failed projection.")
        if self.success and self.reasons:
            raise ValueError("A successful projection cannot contain failure reasons.")
        if not self.success and not self.reasons:
            raise ValueError("A failed projection must contain a reason.")
        if (
            isinstance(self.changed_assignment_count, bool)
            or not isinstance(self.changed_assignment_count, int)
            or self.changed_assignment_count < 0
        ):
            raise ValueError("changed_assignment_count must be non-negative.")
        if (
            isinstance(self.requested_assignment_count, bool)
            or not isinstance(self.requested_assignment_count, int)
            or self.requested_assignment_count <= 0
        ):
            raise ValueError("requested_assignment_count must be positive.")
        if self.changed_assignment_count > self.requested_assignment_count:
            raise ValueError("changed assignments cannot exceed requested assignments.")
        expected_ratio = (
            self.changed_assignment_count / self.requested_assignment_count
        )
        if not math.isclose(
            float(self.change_ratio),
            expected_ratio,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("change_ratio does not match assignment counts.")

        # 冻结字典副本，避免调用方在结果生成后悄悄改变故障域解释。
        object.__setattr__(
            self,
            "fault_domains",
            MappingProxyType(dict(self.fault_domains)),
        )


class DPPOProjector:
    """按演员网络完整排名执行确定性的必要条件投影。

    投影器只处理故障节点、唯一节点、累计资源容量和故障域分散。
    路由、时延、精确可靠性等耦合约束仍交给现有快层执行器处理。
    """

    _TOLERANCE = 1e-12

    def __init__(
        self,
        dimensions: ScenarioDimensions,
        resource_demands: Mapping[int, ProjectionResourceDemand],
        minimum_distinct_fault_domains: int,
    ) -> None:
        """固定本场景的 VNF 顺序、每副本需求和最低故障域数量。"""

        if not isinstance(dimensions, ScenarioDimensions):
            raise TypeError("dimensions must be a ScenarioDimensions instance.")
        if (
            isinstance(minimum_distinct_fault_domains, bool)
            or not isinstance(minimum_distinct_fault_domains, int)
            or not 1 <= minimum_distinct_fault_domains <= 3
        ):
            raise ValueError(
                "minimum_distinct_fault_domains must be an integer from 1 to 3."
            )

        demands = dict(resource_demands)
        expected_ids = set(dimensions.function_ids)
        if set(demands) != expected_ids:
            raise ValueError(
                "resource_demands must cover exactly the configured function IDs."
            )
        if any(
            not isinstance(demand, ProjectionResourceDemand)
            for demand in demands.values()
        ):
            raise ValueError(
                "resource_demands values must be ProjectionResourceDemand objects."
            )

        self.dimensions = dimensions
        self.resource_demands = MappingProxyType(demands)
        self.minimum_distinct_fault_domains = minimum_distinct_fault_domains

    def project(
        self,
        *,
        decoded_action: DecodedDPPOAction,
        operational_node_ids: frozenset[int],
        free_cpu: Mapping[int, float],
        free_memory_mb: Mapping[int, float],
        fault_domains: Mapping[int, int],
    ) -> ProjectionResult:
        """投影一次整条 SFC 动作，并累计扣减所有 VNF 的节点余量。"""

        actions = self._validate_decoded_action(decoded_action)
        operational = self._validate_operational_nodes(operational_node_ids)
        remaining_cpu = self._validate_capacity_mapping(free_cpu, "free_cpu")
        remaining_memory = self._validate_capacity_mapping(
            free_memory_mb,
            "free_memory_mb",
        )
        domains = self._validate_fault_domains(fault_domains)
        requested_count = sum(action.replica_count for action in actions)
        raw_feasible = self._raw_action_is_feasible(
            actions,
            operational,
            remaining_cpu,
            remaining_memory,
            domains,
        )

        projected_intents: list[FunctionDeploymentIntent] = []
        changed_count = 0
        for action in actions:
            demand = self.resource_demands[action.function_id]
            raw_nodes = action.ranked_node_ids[: action.replica_count]

            # 如果当前 VNF 的原始选择已经可行，就原样保留，减少不必要投影。
            if self._nodes_are_feasible(
                raw_nodes,
                action.replica_count,
                demand,
                operational,
                remaining_cpu,
                remaining_memory,
                domains,
            ):
                selected_nodes = raw_nodes
            else:
                selected_nodes, failure_reason = self._select_nodes(
                    action,
                    demand,
                    operational,
                    remaining_cpu,
                    remaining_memory,
                    domains,
                )
                if selected_nodes is None:
                    return self._failure_result(
                        raw_feasible=raw_feasible,
                        reason=failure_reason,
                        requested_count=requested_count,
                        fault_domains=domains,
                    )

            for node_id in selected_nodes:
                remaining_cpu[node_id] -= float(demand.cpu)
                remaining_memory[node_id] -= float(demand.memory_mb)

            changed_count += sum(
                original != projected
                for original, projected in zip(
                    raw_nodes,
                    selected_nodes,
                    strict=True,
                )
            )
            projected_intents.append(
                FunctionDeploymentIntent(
                    function_id=action.function_id,
                    preferred_node_ids=selected_nodes,
                    replica_count=action.replica_count,
                    primary_retention_seconds=action.primary_retention_seconds,
                    backup_retention_seconds=action.backup_retention_seconds,
                )
            )

        return ProjectionResult(
            function_intents=tuple(projected_intents),
            raw_feasible=raw_feasible,
            success=True,
            reasons=(),
            changed_assignment_count=changed_count,
            requested_assignment_count=requested_count,
            change_ratio=changed_count / requested_count,
            fault_domains=domains,
        )

    def _select_nodes(
        self,
        action: DecodedFunctionAction,
        demand: ProjectionResourceDemand,
        operational: frozenset[int],
        remaining_cpu: Mapping[int, float],
        remaining_memory: Mapping[int, float],
        domains: Mapping[int, int],
    ) -> tuple[tuple[int, ...] | None, str]:
        """先诊断不可行原因，再按照完整排名选择分散且有容量的节点。"""

        function_id = action.function_id
        replica_count = action.replica_count
        operational_candidates = tuple(
            node_id
            for node_id in action.ranked_node_ids
            if node_id in operational
        )
        if len(operational_candidates) < replica_count:
            return None, (
                f"insufficient_operational_nodes:function_id={function_id}"
            )

        cpu_candidates = tuple(
            node_id
            for node_id in operational_candidates
            if remaining_cpu[node_id] + self._TOLERANCE >= float(demand.cpu)
        )
        if len(cpu_candidates) < replica_count:
            return None, f"insufficient_cpu:function_id={function_id}"

        memory_candidates = tuple(
            node_id
            for node_id in operational_candidates
            if remaining_memory[node_id] + self._TOLERANCE
            >= float(demand.memory_mb)
        )
        if len(memory_candidates) < replica_count:
            return None, f"insufficient_memory:function_id={function_id}"

        eligible_candidates = tuple(
            node_id
            for node_id in operational_candidates
            if node_id in cpu_candidates and node_id in memory_candidates
        )
        if len(eligible_candidates) < replica_count:
            return None, f"insufficient_joint_resources:function_id={function_id}"
        if (
            len({domains[node_id] for node_id in eligible_candidates})
            < self.minimum_distinct_fault_domains
            or self.minimum_distinct_fault_domains > replica_count
        ):
            return None, f"insufficient_fault_domains:function_id={function_id}"

        selected: list[int] = []
        selected_domains: set[int] = set()
        # 第一遍只选新故障域，先达到研究方案规定的最低分散数量。
        for node_id in eligible_candidates:
            domain_id = domains[node_id]
            if domain_id in selected_domains:
                continue
            selected.append(node_id)
            selected_domains.add(domain_id)
            if len(selected_domains) >= self.minimum_distinct_fault_domains:
                break

        # 第二遍按演员网络原排名补满，其余副本可以复用已出现的故障域。
        for node_id in eligible_candidates:
            if len(selected) >= replica_count:
                break
            if node_id not in selected:
                selected.append(node_id)

        if len(selected) != replica_count:
            # 前面的计数检查理论上已排除此分支，保留诊断以防未来选择规则变化。
            return None, f"insufficient_joint_resources:function_id={function_id}"
        return tuple(selected), ""

    def _raw_action_is_feasible(
        self,
        actions: tuple[DecodedFunctionAction, ...],
        operational: frozenset[int],
        free_cpu: Mapping[int, float],
        free_memory: Mapping[int, float],
        domains: Mapping[int, int],
    ) -> bool:
        """在独立资源副本上检查未经投影的整条 SFC 动作。"""

        remaining_cpu = dict(free_cpu)
        remaining_memory = dict(free_memory)
        for action in actions:
            demand = self.resource_demands[action.function_id]
            raw_nodes = action.ranked_node_ids[: action.replica_count]
            if not self._nodes_are_feasible(
                raw_nodes,
                action.replica_count,
                demand,
                operational,
                remaining_cpu,
                remaining_memory,
                domains,
            ):
                return False
            for node_id in raw_nodes:
                remaining_cpu[node_id] -= float(demand.cpu)
                remaining_memory[node_id] -= float(demand.memory_mb)
        return True

    def _nodes_are_feasible(
        self,
        node_ids: tuple[int, ...],
        replica_count: int,
        demand: ProjectionResourceDemand,
        operational: frozenset[int],
        remaining_cpu: Mapping[int, float],
        remaining_memory: Mapping[int, float],
        domains: Mapping[int, int],
    ) -> bool:
        """检查一组明确节点是否满足投影器负责的全部必要条件。"""

        return (
            len(node_ids) == replica_count
            and len(set(node_ids)) == replica_count
            and all(node_id in operational for node_id in node_ids)
            and all(
                remaining_cpu[node_id] + self._TOLERANCE >= float(demand.cpu)
                and remaining_memory[node_id] + self._TOLERANCE
                >= float(demand.memory_mb)
                for node_id in node_ids
            )
            and len({domains[node_id] for node_id in node_ids})
            >= self.minimum_distinct_fault_domains
        )

    def _validate_decoded_action(
        self,
        decoded_action: DecodedDPPOAction,
    ) -> tuple[DecodedFunctionAction, ...]:
        """确认动作沿用统一场景中的 VNF 和节点顺序定义。"""

        if not isinstance(decoded_action, DecodedDPPOAction):
            raise TypeError("decoded_action must be a DecodedDPPOAction.")
        actions = tuple(decoded_action.function_actions)
        if tuple(action.function_id for action in actions) != self.dimensions.function_ids:
            raise ValueError(
                "decoded action function order must match ScenarioDimensions."
            )
        expected_nodes = set(self.dimensions.compute_node_ids)
        for action in actions:
            if not isinstance(action, DecodedFunctionAction):
                raise ValueError(
                    "decoded action must contain DecodedFunctionAction values."
                )
            if (
                len(action.ranked_node_ids) != len(expected_nodes)
                or set(action.ranked_node_ids) != expected_nodes
            ):
                raise ValueError(
                    "each node ranking must contain every compute node exactly once."
                )
            if action.replica_count not in (2, 3):
                raise ValueError("replica_count must be either 2 or 3.")
            for retention in (
                action.primary_retention_seconds,
                action.backup_retention_seconds,
            ):
                if (
                    isinstance(retention, bool)
                    or not math.isfinite(float(retention))
                    or float(retention) < 0.0
                ):
                    raise ValueError(
                        "decoded retention values must be finite and non-negative."
                    )
        return actions

    def _validate_operational_nodes(
        self,
        operational_node_ids: frozenset[int],
    ) -> frozenset[int]:
        """拒绝未知运行节点，防止拓扑和状态快照混用。"""

        operational = frozenset(operational_node_ids)
        if any(
            isinstance(node_id, bool) or not isinstance(node_id, int)
            for node_id in operational
        ):
            raise ValueError("operational node IDs must be integers.")
        unknown = operational - set(self.dimensions.compute_node_ids)
        if unknown:
            raise ValueError(f"unknown operational node IDs: {sorted(unknown)}")
        return operational

    def _validate_capacity_mapping(
        self,
        values: Mapping[int, float],
        name: str,
    ) -> dict[int, float]:
        """复制容量快照，使投影时的累计扣减不会修改调用方数据。"""

        copied = dict(values)
        expected_nodes = set(self.dimensions.compute_node_ids)
        if set(copied) != expected_nodes:
            raise ValueError(f"{name} must cover exactly all compute nodes.")
        normalized: dict[int, float] = {}
        for node_id, value in copied.items():
            if isinstance(value, bool):
                raise ValueError(f"{name} values must be finite and non-negative.")
            try:
                number = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{name} values must be finite and non-negative."
                ) from error
            if not math.isfinite(number) or number < 0.0:
                raise ValueError(
                    f"{name} values must be finite and non-negative."
                )
            normalized[node_id] = number
        return normalized

    def _validate_fault_domains(
        self,
        fault_domains: Mapping[int, int],
    ) -> Mapping[int, int]:
        """复制并冻结节点到故障域的映射，保持诊断结果可复现。"""

        copied = dict(fault_domains)
        expected_nodes = set(self.dimensions.compute_node_ids)
        if set(copied) != expected_nodes:
            raise ValueError("fault_domains must cover exactly all compute nodes.")
        if any(
            isinstance(domain_id, bool)
            or not isinstance(domain_id, int)
            or domain_id < 0
            for domain_id in copied.values()
        ):
            raise ValueError("fault domain IDs must be non-negative integers.")
        return MappingProxyType(copied)

    @staticmethod
    def _failure_result(
        *,
        raw_feasible: bool,
        reason: str,
        requested_count: int,
        fault_domains: Mapping[int, int],
    ) -> ProjectionResult:
        """统一失败结果；失败动作没有完整投影，因此修改比例记为零。"""

        return ProjectionResult(
            function_intents=None,
            raw_feasible=raw_feasible,
            success=False,
            reasons=(reason,),
            changed_assignment_count=0,
            requested_assignment_count=requested_count,
            change_ratio=0.0,
            fault_domains=fault_domains,
        )
