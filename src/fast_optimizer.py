"""慢层模板约束下的确定性快层可行性修复。"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import permutations, product

from src.constraint_audit import SlotConstraintAuditor
from src.entities import (
    NodeType,
    ServerlessFunction,
    SFCType,
    SlotConstraintAudit,
)
from src.network import TransferNetworkProtocol
from src.rl_agent_action_space import CloudPolicy
from src.topology import LinearRailTopology
from src.two_timescale_control import (
    FastTimescaleDecision,
    FastTimescaleState,
    SlowTimescaleDecision,
    build_fast_decision_for_hot_nodes,
    build_fast_decision_for_plan,
)


@dataclass(frozen=True)
class FastOptimizationResult:
    """保存一次快层修复的输入审计、最终方案和搜索统计。"""

    # attempted=False表示初始方案可直接执行，没有启动搜索。
    attempted: bool

    # None表示无需搜索；True/False分别表示搜索成功或失败。
    succeeded: bool | None

    function_replica_node_ids: dict[
        int,
        tuple[int, ...],
    ]
    decision: FastTimescaleDecision
    initial_audit: SlotConstraintAudit
    final_audit: SlotConstraintAudit
    reason: str
    evaluated_candidate_count: int


class FastFeasibilityOptimizer:
    """搜索改动最少且满足全部硬约束的快层方案。"""

    def __init__(
        self,
        functions: list[ServerlessFunction],
        sfc: SFCType,
        topology: LinearRailTopology,
        auditor: SlotConstraintAuditor,
        return_result_to_source: bool = True,
        network: TransferNetworkProtocol | None = None,
        edge_cpu_cost_per_unit: float = 0.0,
        edge_memory_cost_per_mb_second: float = 0.0,
        cloud_cpu_cost_per_unit: float = 0.0,
        cloud_memory_cost_per_mb_second: float = 0.0,
        cold_start_cost_per_ms: float = 0.0,
        input_size_mb_per_request: float = 0.0,
        slot_seconds: float = 1.0,
    ) -> None:
        """保存搜索、传输和候选成本估计所需的模型。"""

        nonnegative_values = (
            edge_cpu_cost_per_unit,
            edge_memory_cost_per_mb_second,
            cloud_cpu_cost_per_unit,
            cloud_memory_cost_per_mb_second,
            cold_start_cost_per_ms,
            input_size_mb_per_request,
        )
        if any(value < 0 for value in nonnegative_values):
            raise ValueError("快层成本参数不能小于0。")
        if slot_seconds <= 0:
            raise ValueError("快时隙长度必须大于0。")

        self.functions = list(functions)
        self.function_map = {
            function.function_id: function
            for function in self.functions
        }
        self.sfc = sfc
        self.topology = topology
        self.auditor = auditor
        self.return_result_to_source = return_result_to_source
        self.network = network
        self.edge_cpu_cost_per_unit = edge_cpu_cost_per_unit
        self.edge_memory_cost_per_mb_second = (
            edge_memory_cost_per_mb_second
        )
        self.cloud_cpu_cost_per_unit = cloud_cpu_cost_per_unit
        self.cloud_memory_cost_per_mb_second = (
            cloud_memory_cost_per_mb_second
        )
        self.cold_start_cost_per_ms = cold_start_cost_per_ms
        self.input_size_mb_per_request = input_size_mb_per_request
        self.slot_seconds = slot_seconds
        self.edge_node_ids = tuple(
            sorted(
                site.node.node_id
                for site in topology.sites
            )
        )
        self.cloud_node_id = (
            None
            if topology.cloud_node is None
            else topology.cloud_node.node_id
        )
        self.node_map = {
            node.node_id: node
            for node in topology.compute_nodes
        }
        if network is None:
            self.network_node_ids = set(self.node_map)
        elif hasattr(network, "node_ids"):
            self.network_node_ids = set(network.node_ids)
        elif hasattr(network, "edge_network") and hasattr(network, "cloud_node_id"):
            self.network_node_ids = {
                *network.edge_network.node_ids,
                network.cloud_node_id,
            }
        else:
            # 自定义协议实现若不公开节点表，就沿用拓扑并由其传输方法最终校验。
            self.network_node_ids = set(self.node_map)

    def _cold_activated_pairs(
        self,
        decision: FastTimescaleDecision,
    ) -> set[tuple[int, int]]:
        """把冷启动函数编号还原为具体的函数—节点实例。"""

        if decision.request_success is not True:
            return set()

        selected_by_function = dict(
            zip(
                self.sfc.function_ids,
                decision.selected_execution_node_ids,
            )
        )
        return {
            (
                function_id,
                selected_by_function[function_id],
            )
            for function_id
            in decision.cold_start_function_ids
        }

    def _rejected_decision(
        self,
        state: FastTimescaleState,
        initial_decision: FastTimescaleDecision,
    ) -> FastTimescaleDecision:
        """构造不可修复时的最终快层决定。"""

        # 无请求时没有成功或失败事件，必须保留原来的None语义。
        if state.request_count == 0:
            return initial_decision

        return replace(
            initial_decision,
            selected_execution_node_ids=(),
            failover_function_ids=(),
            cold_start_function_ids=(),
            request_success=False,
        )

    def _failure_result(
        self,
        state: FastTimescaleState,
        initial_decision: FastTimescaleDecision,
        initial_audit: SlotConstraintAudit,
        reason: str,
        evaluated_candidate_count: int,
    ) -> FastOptimizationResult:
        """统一生成可解释的无解结果。"""

        return FastOptimizationResult(
            attempted=True,
            succeeded=False,
            function_replica_node_ids=dict(
                state.candidate_node_ids
            ),
            decision=self._rejected_decision(
                state,
                initial_decision,
            ),
            initial_audit=initial_audit,
            final_audit=initial_audit,
            reason=reason,
            evaluated_candidate_count=(
                evaluated_candidate_count
            ),
        )

    def _score(
        self,
        state: FastTimescaleState,
        candidate_map: dict[int, tuple[int, ...]],
        decision: FastTimescaleDecision,
    ) -> tuple[
        float,
        int,
        tuple[int, ...],
    ]:
        """按预计总成本、修改函数数和节点编号进行稳定排序。"""

        initial_map = state.candidate_node_ids
        changed_function_count = sum(
            candidate_map[function_id]
            != initial_map[function_id]
            for function_id in state.function_ids
        )
        flattened_ids = tuple(
            node_id
            for function_id in state.function_ids
            for node_id in candidate_map[function_id]
        )

        # 成本优先体现论文目标；后两项只在同成本时减少改动并
        # 保证相同输入重复运行得到相同结果。
        return (
            self._estimated_total_cost(
                state=state,
                decision=decision,
            ),
            int(changed_function_count),
            flattened_ids,
        )

    def _node_cost_rates(
        self,
        node_id: int,
    ) -> tuple[float, float]:
        """返回指定节点的CPU单价和内存单价。"""

        node = self.node_map[node_id]
        if node.node_type is NodeType.CLOUD:
            return (
                self.cloud_cpu_cost_per_unit,
                self.cloud_memory_cost_per_mb_second,
            )
        return (
            self.edge_cpu_cost_per_unit,
            self.edge_memory_cost_per_mb_second,
        )

    def _estimated_total_cost(
        self,
        state: FastTimescaleState,
        decision: FastTimescaleDecision,
    ) -> float:
        """估计候选方案的运行、传输和冷启动总成本。"""

        selected_by_function = dict(
            zip(
                state.function_ids,
                decision.selected_execution_node_ids,
            )
        )
        active_pairs = {
            (function_id, node_id)
            for function_id, node_ids
            in decision.function_hot_node_ids.items()
            for node_id in node_ids
        }

        # 被选中但尚未保温的容器在本时隙冷启动后同样占用内存。
        for function_id in decision.cold_start_function_ids:
            if function_id in selected_by_function:
                active_pairs.add(
                    (
                        function_id,
                        selected_by_function[function_id],
                    )
                )

        run_cost = 0.0
        for function_id, node_id in active_pairs:
            function = self.function_map[function_id]
            _, memory_rate = self._node_cost_rates(
                node_id
            )
            run_cost += (
                function.memory_mb
                * self.slot_seconds
                * memory_rate
            )

        if decision.request_success is True:
            for function_id, node_id in selected_by_function.items():
                function = self.function_map[function_id]
                cpu_rate, _ = self._node_cost_rates(node_id)
                run_cost += (
                    function.cpu_demand(state.request_count)
                    * cpu_rate
                )

        route_cost = 0.0
        if (
            self.network is not None
            and decision.request_success is True
        ):
            current_node_id = state.serving_mec
            current_data_mb = (
                self.input_size_mb_per_request
                * state.request_count
            )
            for function_id, destination_node_id in (
                selected_by_function.items()
            ):
                route_cost += self.network.transfer_cost(
                    data_size_mb=current_data_mb,
                    source_node_id=current_node_id,
                    destination_node_id=(
                        destination_node_id
                    ),
                )
                current_node_id = destination_node_id
                current_data_mb *= (
                    self.function_map[function_id]
                    .output_ratio
                )

            if self.return_result_to_source:
                route_cost += self.network.transfer_cost(
                    data_size_mb=current_data_mb,
                    source_node_id=current_node_id,
                    destination_node_id=state.serving_mec,
                )

        cold_start_cost = sum(
            self.function_map[function_id]
            .cold_start_time_ms
            * self.cold_start_cost_per_ms
            for function_id
            in decision.cold_start_function_ids
        )
        return run_cost + route_cost + cold_start_cost

    def _preserve_initial_failure_failovers(
        self,
        state: FastTimescaleState,
        decision: FastTimescaleDecision,
    ) -> FastTimescaleDecision:
        """保留相对修复前故障主节点发生的真实接管事件。"""

        if decision.request_success is not True:
            return decision

        selected_by_function = dict(
            zip(
                state.function_ids,
                decision.selected_execution_node_ids,
            )
        )
        failover_ids = set(
            decision.failover_function_ids
        )

        for function_id in state.function_ids:
            initial_primary = (
                state.candidate_node_ids[function_id][0]
            )

            # 修复器可能把原备用节点提升为新主节点。若原主节点
            # 本时隙已经故障，这次执行在运行语义上仍属于主备接管。
            if (
                initial_primary
                not in state.operational_node_ids
                and selected_by_function[function_id]
                != initial_primary
            ):
                failover_ids.add(function_id)

        return replace(
            decision,
            failover_function_ids=tuple(
                function_id
                for function_id in state.function_ids
                if function_id in failover_ids
            ),
        )

    def optimize(
        self,
        state: FastTimescaleState,
        slow_decision: SlowTimescaleDecision | None,
        initial_decision: FastTimescaleDecision,
        initial_audit: SlotConstraintAudit,
        *,
        expected_replica_counts: Mapping[int, int] | None = None,
        allow_cloud: bool | None = None,
        retained_hot_node_ids: dict[int, tuple[int, ...]] | None = None,
        retention_role_flags: Mapping[int, tuple[bool, ...]] | None = None,
    ) -> FastOptimizationResult:
        """在慢层副本数、保留策略和云权限不变时修复部署。"""

        if expected_replica_counts is None:
            if slow_decision is None:
                raise ValueError(
                    "Explicit execution requires per-function replica counts."
                )
            required_counts = {
                function_id: slow_decision.replica_count
                for function_id in state.function_ids
            }
        else:
            required_counts = dict(expected_replica_counts)
            if set(required_counts) != set(state.function_ids):
                raise ValueError(
                    "Per-function replica counts must cover the complete SFC."
                )

        if retention_role_flags is not None:
            if retained_hot_node_ids is None:
                raise ValueError(
                    "Retention role flags require explicit retained hot nodes."
                )
            if set(retention_role_flags) != set(state.function_ids):
                raise ValueError(
                    "Retention role flags must cover the complete SFC."
                )
            for function_id in state.function_ids:
                if len(retention_role_flags[function_id]) != required_counts[
                    function_id
                ]:
                    raise ValueError(
                        "Retention role flags must match each function's replicas."
                    )

        if any(
            isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            for count in required_counts.values()
        ):
            raise ValueError("慢层副本数量必须大于0。")

        if allow_cloud is None:
            if slow_decision is None:
                raise ValueError("Explicit execution must define cloud eligibility.")
            cloud_is_allowed = (
                slow_decision.cloud_policy is CloudPolicy.CLOUD_ALLOWED
            )
        else:
            cloud_is_allowed = bool(allow_cloud)

        allowed_node_ids = set(self.edge_node_ids)
        if (
            cloud_is_allowed
            and self.cloud_node_id is not None
        ):
            allowed_node_ids.add(self.cloud_node_id)
        # 修复器不能选择网络模型无法路由的计算节点。
        allowed_node_ids.intersection_update(self.network_node_ids)
        initial_plan_policy_compliant = all(
            node_id in allowed_node_ids
            for function_id in state.function_ids
            for node_id in state.candidate_node_ids[
                function_id
            ]
        )

        # 静态约束满足还不够：活动请求还必须拥有完整可运行路径。
        execution_ready = (
            state.request_count == 0
            or initial_decision.request_success is True
        )

        # 静态审计不读取本时隙故障状态。即使主节点仍可执行，
        # 只要任一计划副本已经故障，原冗余模板就不能视为完整可行。
        initial_plan_operational = all(
            node_id in state.operational_node_ids
            for function_id in state.function_ids
            for node_id in state.candidate_node_ids[
                function_id
            ]
        )
        if (
            initial_audit.all_constraints_met
            and execution_ready
            and initial_plan_operational
            and initial_plan_policy_compliant
        ):
            return FastOptimizationResult(
                attempted=False,
                succeeded=None,
                function_replica_node_ids=dict(
                    state.candidate_node_ids
                ),
                decision=initial_decision,
                initial_audit=initial_audit,
                final_audit=initial_audit,
                reason="初始方案满足全部硬约束，无需修复。",
                evaluated_candidate_count=0,
            )

        maximum_required_count = max(required_counts.values())
        minimum_domains = (
            self.auditor.reliability_model
            .minimum_distinct_fault_domains
        )

        # 快层只能搬迁副本，不能突破慢层给出的副本数量上限。
        insufficient_domain_functions = tuple(
            function_id
            for function_id in state.function_ids
            if required_counts[function_id] < minimum_domains
        )
        if insufficient_domain_functions:
            required_count = min(
                required_counts[function_id]
                for function_id in insufficient_domain_functions
            )
            # 旧错误文本仍使用一个数值，这里取最小值仅用于兼容诊断展示。
            return self._failure_result(
                state=state,
                initial_decision=initial_decision,
                initial_audit=initial_audit,
                reason=(
                    f"慢层只允许{required_count}个副本，"
                    f"但可靠性模型至少要求{minimum_domains}个故障域，"
                    "快层不能擅自增加副本数量。"
                ),
                evaluated_candidate_count=0,
            )

        # 云策略是慢层硬边界。EDGE_ONLY时即使云节点正常，
        # 快层也不能把它加入搜索空间。
        operational_edge_ids = tuple(
            node_id
            for node_id in self.edge_node_ids
            if node_id in state.operational_node_ids
        )
        operational_ids = operational_edge_ids
        if (
            self.cloud_node_id in allowed_node_ids
            and self.cloud_node_id
            in state.operational_node_ids
        ):
            operational_ids = (
                *operational_ids,
                self.cloud_node_id,
            )

        required_count = maximum_required_count
        if len(operational_ids) < maximum_required_count:
            return self._failure_result(
                state=state,
                initial_decision=initial_decision,
                initial_audit=initial_audit,
                reason=(
                    f"当前只有{len(operational_ids)}个正常MEC/云候选节点，"
                    f"少于慢层要求的{required_count}个副本。"
                ),
                evaluated_candidate_count=0,
            )

        # 排列而非组合：元组首项是主节点，因此(0,1)和(1,0)
        # 对执行路径、接管和冷启动具有不同含义。
        per_function_options = tuple(
            tuple(
                permutations(
                    operational_ids,
                    required_counts[function_id],
                )
            )
            for function_id in state.function_ids
        )
        ranked_candidates: list[
            tuple[
                tuple[
                    float,
                    int,
                    tuple[int, ...],
                ],
                dict[int, tuple[int, ...]],
                FastTimescaleDecision,
            ]
        ] = []

        for node_tuples in product(*per_function_options):
            candidate_map = {
                function_id: tuple(node_ids)
                for function_id, node_ids in zip(
                    state.function_ids,
                    node_tuples,
                )
            }

            # 初始方案已经审计过，不重复计入候选评估数量。
            if candidate_map == state.candidate_node_ids:
                continue

            repaired_state = replace(
                state,
                candidate_node_ids=candidate_map,
            )
            if retained_hot_node_ids is None:
                if slow_decision is None:
                    raise ValueError(
                        "Legacy repair requires a slow-timescale decision."
                    )
                decision = build_fast_decision_for_plan(
                    state=repaired_state,
                    retention_policy=slow_decision.retention_policy,
                    backup_activation_triggered=(
                        initial_decision.backup_activation_triggered
                    ),
                    previously_hot_node_ids=(
                        initial_decision.function_hot_node_ids
                    ),
                )
            else:
                # 历史保留节点不能随修复迁移；但“本次动作要求变热”的
                # 主/备角色必须绑定到当前候选方案，否则修复器会错误地把
                # 已迁出的节点计入内存，并把新节点一律当作冷副本。
                candidate_hot_node_ids = {
                    function_id: tuple(
                        sorted(
                            {
                                node_id
                                for node_id in retained_hot_node_ids[
                                    function_id
                                ]
                                if node_id in state.operational_node_ids
                            }
                            | {
                                node_id
                                for node_id, should_be_hot in zip(
                                    candidate_map[function_id],
                                    (
                                        retention_role_flags[function_id]
                                        if retention_role_flags is not None
                                        else ()
                                    ),
                                )
                                if should_be_hot
                                and node_id in state.operational_node_ids
                            }
                        )
                    )
                    for function_id in state.function_ids
                }
                decision = build_fast_decision_for_hot_nodes(
                    state=repaired_state,
                    function_hot_node_ids=candidate_hot_node_ids,
                    previously_hot_node_ids=(
                        initial_decision.function_hot_node_ids
                    ),
                    backup_activation_triggered=False,
                )
            decision = (
                self._preserve_initial_failure_failovers(
                    state=state,
                    decision=decision,
                )
            )
            ranked_candidates.append(
                (
                    self._score(
                        state,
                        candidate_map,
                        decision,
                    ),
                    candidate_map,
                    decision,
                )
            )

        evaluated_count = 0
        for _, candidate_map, decision in sorted(
            ranked_candidates,
            key=lambda item: item[0],
        ):
            evaluated_count += 1
            cold_pairs = self._cold_activated_pairs(
                decision
            )

            # 资源判断比精确共享故障可靠性计算更便宜，先剪枝。
            if not self.auditor.resource_constraints_met(
                request_count=state.request_count,
                selected_execution_node_ids=(
                    decision.selected_execution_node_ids
                ),
                request_success=decision.request_success,
                function_hot_node_ids=(
                    decision.function_hot_node_ids
                ),
                cold_activated_pairs=cold_pairs,
            ):
                continue

            audit = self.auditor.audit(
                request_count=state.request_count,
                expected_replica_count=required_counts,
                candidate_map=candidate_map,
                selected_execution_node_ids=(
                    decision.selected_execution_node_ids
                ),
                request_success=decision.request_success,
                function_hot_node_ids=(
                    decision.function_hot_node_ids
                ),
                cold_activated_pairs=cold_pairs,
            )
            execution_ready = (
                state.request_count == 0
                or decision.request_success is True
            )

            if audit.all_constraints_met and execution_ready:
                changed_ids = [
                    function_id
                    for function_id in state.function_ids
                    if candidate_map[function_id]
                    != state.candidate_node_ids[
                        function_id
                    ]
                ]
                return FastOptimizationResult(
                    attempted=True,
                    succeeded=True,
                    function_replica_node_ids=dict(
                        candidate_map
                    ),
                    decision=decision,
                    initial_audit=initial_audit,
                    final_audit=audit,
                    reason=(
                        f"已修复函数{changed_ids}的副本位置。"
                    ),
                    evaluated_candidate_count=(
                        evaluated_count
                    ),
                )

        return self._failure_result(
            state=state,
            initial_decision=initial_decision,
            initial_audit=initial_audit,
            reason=(
                "已检查全部候选方案，"
                "没有找到满足硬约束的方案。"
            ),
            evaluated_candidate_count=evaluated_count,
        )
