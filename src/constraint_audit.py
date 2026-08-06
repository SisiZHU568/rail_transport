"""快时隙资源、副本计划和可靠性硬约束审计。"""

from src.entities import (
    ServerlessFunction,
    SFCType,
    SlotConstraintAudit,
)
from src.reliability import FaultDomainReliabilityModel
from src.topology import LinearRailTopology


class SlotConstraintAuditor:
    """只计算约束结果，不修改部署计划或请求状态。"""

    def __init__(
        self,
        functions: list[ServerlessFunction],
        sfc: SFCType,
        topology: LinearRailTopology,
        reliability_model: FaultDomainReliabilityModel,
    ) -> None:
        """保存审计所需的函数、拓扑和精确可靠性模型。"""

        self.functions = list(functions)
        self.function_map = {
            function.function_id: function
            for function in functions
        }

        if len(self.function_map) != len(functions):
            raise ValueError("Serverless函数编号不能重复。")

        if any(
            function_id not in self.function_map
            for function_id in sfc.function_ids
        ):
            raise KeyError("SFC引用了未提供的Serverless函数。")

        self.sfc = sfc
        self.topology = topology
        self.reliability_model = reliability_model
        self.node_map = {
            site.node.node_id: site.node
            for site in topology.sites
        }

    def _calculate_resource_demands(
        self,
        request_count: int,
        selected_execution_node_ids: tuple[int, ...],
        request_success: bool | None,
        function_hot_node_ids: dict[int, tuple[int, ...]],
        cold_activated_pairs: set[tuple[int, int]],
    ) -> tuple[dict[int, float], dict[int, float]]:
        """用唯一公式累计本时隙各节点的CPU和内存需求。"""

        if request_count < 0:
            raise ValueError("请求数量不能小于0。")

        # 只有已经温热或本时隙实际冷启动的实例才占用活动内存；
        # 使用集合可以防止同一实例同时出现在两类记录中而重复计费。
        active_pairs = {
            (function_id, node_id)
            for function_id, node_ids
            in function_hot_node_ids.items()
            for node_id in node_ids
        }
        active_pairs.update(cold_activated_pairs)

        node_memory_demand_mb: dict[int, float] = {}
        for function_id, node_id in active_pairs:
            node_memory_demand_mb[node_id] = (
                node_memory_demand_mb.get(node_id, 0.0)
                + self.function_map[function_id].memory_mb
            )

        node_cpu_demand: dict[int, float] = {}

        # 没有请求或请求没有可执行完整路径时，不产生函数执行CPU需求。
        if request_success is True:
            for function_id, node_id in zip(
                self.sfc.function_ids,
                selected_execution_node_ids,
            ):
                node_cpu_demand[node_id] = (
                    node_cpu_demand.get(node_id, 0.0)
                    + self.function_map[function_id].cpu_demand(
                        request_count
                    )
                )

        return node_cpu_demand, node_memory_demand_mb

    def resource_constraints_met(
        self,
        request_count: int,
        selected_execution_node_ids: tuple[int, ...],
        request_success: bool | None,
        function_hot_node_ids: dict[int, tuple[int, ...]],
        cold_activated_pairs: set[tuple[int, int]],
    ) -> bool:
        """快速判断资源是否可行，供后续修复搜索提前剪枝。"""

        cpu, memory = self._calculate_resource_demands(
            request_count=request_count,
            selected_execution_node_ids=selected_execution_node_ids,
            request_success=request_success,
            function_hot_node_ids=function_hot_node_ids,
            cold_activated_pairs=cold_activated_pairs,
        )
        referenced_node_ids = set(cpu) | set(memory)

        # 未知节点没有容量信息，必须直接判为不可行。
        if any(
            node_id not in self.node_map
            for node_id in referenced_node_ids
        ):
            return False

        return all(
            self.node_map[node_id].has_sufficient_capacity(
                cpu_demand=cpu.get(node_id, 0.0),
                memory_demand_mb=memory.get(node_id, 0.0),
            )
            for node_id in referenced_node_ids
        )

    def audit(
        self,
        request_count: int,
        expected_replica_count: int,
        candidate_map: dict[int, tuple[int, ...]],
        selected_execution_node_ids: tuple[int, ...],
        request_success: bool | None,
        function_hot_node_ids: dict[int, tuple[int, ...]],
        cold_activated_pairs: set[tuple[int, int]],
    ) -> SlotConstraintAudit:
        """返回本时隙的完整硬约束审计结果。"""

        if expected_replica_count <= 0:
            raise ValueError("期望副本数量必须大于0。")

        required_ids = set(self.sfc.function_ids)
        supplied_ids = set(candidate_map)
        missing_ids = tuple(sorted(required_ids - supplied_ids))
        extra_ids = tuple(sorted(supplied_ids - required_ids))
        invalid_node_ids: set[int] = set()
        count_violation_ids: list[int] = []
        reasons: list[str] = []
        replica_plan_valid = not (missing_ids or extra_ids)

        if missing_ids:
            reasons.append(
                f"副本计划缺少函数{list(missing_ids)}。"
            )
        if extra_ids:
            reasons.append(
                f"副本计划包含额外函数{list(extra_ids)}。"
            )

        for function_id in self.sfc.function_ids:
            node_ids = candidate_map.get(function_id, ())

            # 快层只能移动副本，不能改变慢层许可的副本数量。
            if len(node_ids) != expected_replica_count:
                count_violation_ids.append(function_id)
                replica_plan_valid = False
                reasons.append(
                    f"函数{function_id}的副本数量{len(node_ids)}"
                    f"不等于慢层要求{expected_replica_count}。"
                )

            if len(node_ids) != len(set(node_ids)):
                replica_plan_valid = False
                reasons.append(
                    f"函数{function_id}存在重复副本节点。"
                )

            for node_id in node_ids:
                if node_id not in self.node_map:
                    invalid_node_ids.add(node_id)
                    replica_plan_valid = False

        if invalid_node_ids:
            reasons.append(
                "副本计划引用未知节点"
                f"{sorted(invalid_node_ids)}。"
            )

        cpu, memory = self._calculate_resource_demands(
            request_count=request_count,
            selected_execution_node_ids=selected_execution_node_ids,
            request_success=request_success,
            function_hot_node_ids=function_hot_node_ids,
            cold_activated_pairs=cold_activated_pairs,
        )
        cpu_violations = tuple(
            sorted(
                node_id
                for node_id, demand in cpu.items()
                if node_id not in self.node_map
                or demand > self.node_map[node_id].cpu_capacity
            )
        )
        memory_violations = tuple(
            sorted(
                node_id
                for node_id, demand in memory.items()
                if node_id not in self.node_map
                or demand
                > self.node_map[node_id].memory_capacity_mb
            )
        )

        for node_id in cpu_violations:
            if node_id in self.node_map:
                reasons.append(
                    f"节点{node_id}的CPU需求{cpu[node_id]:.3f}"
                    "超过容量"
                    f"{self.node_map[node_id].cpu_capacity:.3f}。"
                )

        for node_id in memory_violations:
            if node_id in self.node_map:
                reasons.append(
                    f"节点{node_id}的内存需求"
                    f"{memory[node_id]:.3f}MB超过容量"
                    f"{self.node_map[node_id].memory_capacity_mb:.3f}MB。"
                )

        exact_reliability: float | None = None
        reliability_target_met = False

        # 只有副本计划结构完整、节点有效且数量正确时，
        # 精确可靠性结果才具有实际意义。
        if replica_plan_valid:
            reliability = self.reliability_model.evaluate_sfc(
                sfc=self.sfc,
                function_replica_node_ids=candidate_map,
            )
            exact_reliability = (
                reliability.exact_shared_failure_availability
            )
            reliability_target_met = reliability.target_met

            if not reliability_target_met:
                reasons.append(
                    f"精确SFC可靠性{exact_reliability:.6f}"
                    f"未达到目标{self.sfc.reliability_target:.6f}，"
                    "或副本未满足故障域隔离要求。"
                )

        resource_met = not (
            cpu_violations or memory_violations
        )

        return SlotConstraintAudit(
            node_cpu_demand=dict(sorted(cpu.items())),
            node_memory_demand_mb=dict(sorted(memory.items())),
            cpu_violation_node_ids=cpu_violations,
            memory_violation_node_ids=memory_violations,
            invalid_replica_node_ids=tuple(
                sorted(invalid_node_ids)
            ),
            missing_function_ids=missing_ids,
            exact_sfc_reliability=exact_reliability,
            reliability_target=self.sfc.reliability_target,
            reliability_target_met=reliability_target_met,
            resource_constraints_met=resource_met,
            replica_plan_valid=replica_plan_valid,
            all_constraints_met=(
                resource_met
                and replica_plan_valid
                and reliability_target_met
            ),
            violation_reasons=tuple(reasons),
            replica_count_violation_function_ids=tuple(
                sorted(count_violation_ids)
            ),
        )
