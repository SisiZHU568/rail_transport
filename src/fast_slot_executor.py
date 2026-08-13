"""训练环境与规则仿真器共用的单快时隙执行闭环。"""

from dataclasses import dataclass, replace
import math

from src.continuous_retention import ContinuousRetentionTracker
from src.constraint_audit import SlotConstraintAuditor
from src.entities import (
    NodeType,
    ServerlessFunction,
    SFCType,
    SlotConstraintAudit,
    TrainState,
)
from src.failure_process import FailureSnapshot
from src.fast_convex_scheduler import (
    FastConvexScheduler,
    FastConvexSchedulingResult,
    FastScheduledBatch,
)
from src.fast_optimizer import FastFeasibilityOptimizer
from src.network import TransferNetworkProtocol
from src.deployment_policies import RetentionPolicy
from src.runtime_reliability import (
    ReplicaPlannerProtocol,
    SingleReplicaPlanner,
)
from src.sfc_deployment_intent import SFCDeploymentIntent
from src.sfc_execution import execute_sfc_batch
from src.topology import LinearRailTopology
from src.two_timescale_control import (
    FastTimescaleDecision,
    FastTimescaleState,
    SlowTimescaleDecision,
    build_fast_decision_for_hot_nodes,
    build_fast_decision_for_plan,
    build_rule_based_deployment_intent,
)


@dataclass(frozen=True)
class RuntimeCostRates:
    """保存快时隙三类原始成本使用的单位价格。"""

    edge_cpu_cost_per_unit: float
    edge_memory_cost_per_mb_second: float
    cloud_cpu_cost_per_unit: float
    cloud_memory_cost_per_mb_second: float
    cold_start_cost_per_ms: float

    def __post_init__(self) -> None:
        """成本单价必须是可用于稳定训练的非负有限数。"""

        values = (
            self.edge_cpu_cost_per_unit,
            self.edge_memory_cost_per_mb_second,
            self.cloud_cpu_cost_per_unit,
            self.cloud_memory_cost_per_mb_second,
            self.cold_start_cost_per_ms,
        )
        if any(
            not math.isfinite(value) or value < 0
            for value in values
        ):
            raise ValueError("快时隙成本单价必须是非负有限值。")


@dataclass(frozen=True)
class FastSlotInput:
    """保存执行一个快时隙所需的全部动态输入。"""

    train_state: TrainState
    request_count: int
    infrastructure_state: FailureSnapshot
    slow_decision: SlowTimescaleDecision | None
    previous_candidate_map: (
        dict[int, tuple[int, ...]] | None
    )
    deployment_intent: SFCDeploymentIntent | None = None

    def __post_init__(self) -> None:
        """在闭环入口拒绝没有物理意义的负请求数量。"""

        if self.request_count < 0:
            raise ValueError("快时隙请求数量不能小于0。")
        if (self.slow_decision is None) == (self.deployment_intent is None):
            raise ValueError(
                "Exactly one of slow_decision or deployment_intent is required."
            )


@dataclass(frozen=True)
class FastSlotExecutionResult:
    """保存快层规划、修复、执行和计费的完整结果。"""

    function_replica_node_ids: dict[int, tuple[int, ...]]
    function_hot_node_ids: dict[int, tuple[int, ...]]
    initial_candidate_map: dict[int, tuple[int, ...]]
    retained_hot_node_ids_by_function: dict[int, frozenset[int]]
    selected_execution_node_ids: tuple[int, ...]
    initial_audit: SlotConstraintAudit
    final_audit: SlotConstraintAudit
    fast_repair_attempted: bool
    fast_repair_succeeded: bool | None
    fast_repair_reason: str
    request_success: bool | None
    deadline_met: bool | None
    end_to_end_delay_ms: float | None
    cold_start_delay_ms: float
    active_memory_mb: float
    plan_change_count: int
    used_cloud: bool
    constraint_rejected: bool
    run_cost: float
    route_cost: float
    cold_start_cost: float

    # 以下明细供规则仿真器原样记录，避免调用方根据汇总值二次推断。
    backup_activation_triggered: bool = False
    failover_function_ids: tuple[int, ...] = ()
    cold_start_function_ids: tuple[int, ...] = ()
    unavailable_function_ids: tuple[int, ...] = ()
    transmission_delay_ms: float | None = None
    execution_delay_ms: float | None = None
    failover_delay_ms: float = 0.0
    active_instance_count: int = 0
    fast_repair_evaluated_candidate_count: int = 0
    # 多路径字段是论文实验的正式输出；上面的单路径字段仅保留兼容性。
    fast_solver_status: str = "not_used"
    fast_solver_objective_value: float | None = None
    fast_solver_time_seconds: float = 0.0
    scheduled_request_counts: tuple[int, ...] = ()
    scheduled_execution_node_ids: tuple[tuple[int, ...], ...] = ()
    cold_start_function_node_pairs: tuple[tuple[int, int], ...] = ()

    @property
    def total_cost(self) -> float:
        """返回奖励函数使用的总原始成本，不在这里添加权重。"""

        return (
            self.run_cost
            + self.route_cost
            + self.cold_start_cost
        )


class FastSlotExecutor:
    """执行一次“规划—审计—修复—可行后执行”的快层闭环。"""

    def __init__(
        self,
        topology: LinearRailTopology,
        network: TransferNetworkProtocol,
        functions: list[ServerlessFunction],
        sfc: SFCType,
        replica_planners: dict[int, ReplicaPlannerProtocol],
        constraint_auditor: SlotConstraintAuditor,
        fast_optimizer: FastFeasibilityOptimizer,
        input_size_mb_per_request: float,
        slot_seconds: float,
        handover_hot_window_s: float,
        failover_delay_ms_per_function: float,
        cost_rates: RuntimeCostRates,
        return_result_to_source: bool = True,
        fast_convex_scheduler: FastConvexScheduler | None = None,
    ) -> None:
        """保存快时隙闭环所需的唯一一组依赖。"""

        if set(replica_planners) != {1, 2, 3}:
            raise ValueError(
                "副本规划器必须且只能提供1、2、3副本三个入口。"
            )

        nonnegative_values = (
            input_size_mb_per_request,
            handover_hot_window_s,
            failover_delay_ms_per_function,
        )
        if any(
            not math.isfinite(value) or value < 0
            for value in nonnegative_values
        ):
            raise ValueError(
                "输入数据量、预热窗口和故障切换时延必须是非负有限值。"
            )
        if not math.isfinite(slot_seconds) or slot_seconds <= 0:
            raise ValueError("快时隙长度必须是正有限值。")

        self.topology = topology
        self.network = network
        self.functions = list(functions)
        self.function_map = {
            function.function_id: function
            for function in self.functions
        }
        if len(self.function_map) != len(self.functions):
            raise ValueError("Serverless函数编号不能重复。")
        if set(self.function_map) != set(sfc.function_ids):
            raise ValueError("函数列表必须与SFC函数集合完全一致。")

        self.sfc = sfc
        self.replica_planners = dict(replica_planners)
        self.constraint_auditor = constraint_auditor
        self.fast_optimizer = fast_optimizer
        self.fast_convex_scheduler = fast_convex_scheduler
        self.input_size_mb_per_request = (
            input_size_mb_per_request
        )
        self.slot_seconds = slot_seconds
        self.handover_hot_window_s = handover_hot_window_s
        self.failover_delay_ms_per_function = (
            failover_delay_ms_per_function
        )
        self.cost_rates = cost_rates
        self.return_result_to_source = return_result_to_source
        self.retention_tracker = ContinuousRetentionTracker(
            slot_seconds=slot_seconds
        )
        # 规则基线的旧语义是“每次慢决策重新生成温热模板”。记录最近一次
        # 规则决策时隙，才能只在新决策到来时清空旧模板，而不影响通用
        # DPPO 意图跨慢决策延续尚未到期的保留状态。
        self._last_rule_decision_slot: int | None = None
        self.node_map = {
            node.node_id: node
            for node in topology.compute_nodes
        }
        self.cloud_node_id = (
            None
            if topology.cloud_node is None
            else topology.cloud_node.node_id
        )

    def reset(self) -> None:
        """清空跨时隙保留状态，供新 Episode 开始时调用。"""

        self.retention_tracker.reset()
        self._last_rule_decision_slot = None

    def build_rule_based_intent(
        self,
        *,
        train_state: TrainState,
        slow_decision: SlowTimescaleDecision,
    ) -> SFCDeploymentIntent:
        """为过渡期规则仿真生成通用意图；``execute`` 本身不再选择规划器。"""

        if self._last_rule_decision_slot != slow_decision.decision_slot:
            self.retention_tracker.reset()
            self._last_rule_decision_slot = slow_decision.decision_slot

        if slow_decision.replica_count not in self.replica_planners:
            raise ValueError("No planner exists for the rule-based replica count.")
        plan = self.replica_planners[slow_decision.replica_count].plan(
            sfc=self.sfc,
            train_state=train_state,
            topology=self.topology,
        )
        candidate_map = {
            function_id: tuple(node_ids)
            for function_id, node_ids in plan.function_replica_node_ids.items()
        }
        backup_activation_triggered = (
            slow_decision.retention_policy is RetentionPolicy.PRIMARY_WARM
            and slow_decision.use_redundancy
            and train_state.remaining_dwell_time_s <= self.handover_hot_window_s
        )
        return build_rule_based_deployment_intent(
            slow_decision=slow_decision,
            candidate_map=candidate_map,
            slot_seconds=self.slot_seconds,
            backup_activation_triggered=backup_activation_triggered,
        )

    def _candidate_map_from_intent(
        self,
        intent: SFCDeploymentIntent,
        *,
        time_slot: int,
    ) -> dict[int, tuple[int, ...]]:
        """直接读取逐 VNF 节点意图，不在快层重新选择副本规划器。"""

        intent_function_ids = tuple(
            function_intent.function_id
            for function_intent in intent.function_intents
        )
        if intent_function_ids != tuple(self.sfc.function_ids):
            raise ValueError(
                "Deployment intent function order must match the configured SFC."
            )
        if not intent.decision_slot <= time_slot < intent.valid_until_slot:
            raise ValueError("Deployment intent is not valid for this fast slot.")
        return {
            function_intent.function_id: tuple(
                function_intent.preferred_node_ids[
                    : function_intent.replica_count
                ]
            )
            for function_intent in intent.function_intents
        }

    def _retained_hot_nodes(
        self,
        *,
        time_slot: int,
        operational_node_ids: frozenset[int],
    ) -> dict[int, frozenset[int]]:
        """读取决策前的连续保留状态，并立即移除已经故障的节点。"""

        failed_node_ids = set(self.node_map) - set(operational_node_ids)
        self.retention_tracker.remove_failed_nodes(failed_node_ids)
        return {
            function_id: self.retention_tracker.hot_node_ids(
                function_id,
                slot=time_slot,
            )
            for function_id in self.sfc.function_ids
        }

    @staticmethod
    def _retention_role_flags(
        intent: SFCDeploymentIntent,
        *,
        time_slot: int,
    ) -> dict[int, tuple[bool, ...]]:
        """标记本次意图中哪些主/备副本需要在当前决策时隙变热。"""

        is_decision_slot = time_slot == intent.decision_slot
        return {
            function_intent.function_id: (
                is_decision_slot
                and function_intent.primary_retention_seconds > 0.0,
                *(
                    is_decision_slot
                    and function_intent.backup_retention_seconds > 0.0
                    for _ in function_intent.preferred_node_ids[1:]
                ),
            )
            for function_intent in intent.function_intents
        }

    @staticmethod
    def _effective_hot_nodes(
        *,
        retained_hot_nodes: dict[int, frozenset[int]],
        candidate_map: dict[int, tuple[int, ...]],
        retention_role_flags: dict[int, tuple[bool, ...]],
        operational_node_ids: frozenset[int],
    ) -> dict[int, frozenset[int]]:
        """合并历史保留与当前候选角色，且绝不把故障节点重新标热。"""

        effective_hot_nodes: dict[int, frozenset[int]] = {}
        for function_id, node_ids in candidate_map.items():
            flags = retention_role_flags[function_id]
            if len(flags) != len(node_ids):
                raise ValueError(
                    "Retention role flags must match each function's replicas."
                )
            current_hot_nodes = {
                node_id
                for node_id, should_be_hot in zip(node_ids, flags)
                if should_be_hot and node_id in operational_node_ids
            }
            current_hot_nodes.update(
                node_id
                for node_id in retained_hot_nodes[function_id]
                if node_id in operational_node_ids
            )
            effective_hot_nodes[function_id] = frozenset(current_hot_nodes)
        return effective_hot_nodes

    def _apply_final_intent_retention(
        self,
        *,
        intent: SFCDeploymentIntent,
        final_candidate_map: dict[int, tuple[int, ...]],
        time_slot: int,
    ) -> None:
        """修复结束后把保留时间绑定到最终副本，避免保留已被迁走的节点。"""

        if time_slot != intent.decision_slot:
            return
        intent_map = {
            function_intent.function_id: function_intent
            for function_intent in intent.function_intents
        }
        for function_id, node_ids in final_candidate_map.items():
            function_intent = intent_map[function_id]
            self.retention_tracker.apply_intent(
                slot=time_slot,
                function_id=function_id,
                primary_node_id=node_ids[0],
                backup_node_ids=tuple(node_ids[1:]),
                primary_seconds=function_intent.primary_retention_seconds,
                backup_seconds=function_intent.backup_retention_seconds,
            )

    def _build_candidate_map(
        self,
        slot_input: FastSlotInput,
    ) -> dict[int, tuple[int, ...]]:
        """按慢动作的副本数选择唯一对应的副本规划器。"""

        if slot_input.slow_decision is None:
            raise ValueError("Legacy planning requires a slow-timescale decision.")
        replica_count = slot_input.slow_decision.replica_count
        if replica_count not in self.replica_planners:
            raise ValueError(
                f"共享执行器只支持1至3副本，收到{replica_count}。"
            )

        plan = self.replica_planners[replica_count].plan(
            sfc=self.sfc,
            train_state=slot_input.train_state,
            topology=self.topology,
        )
        candidate_map = {
            function_id: tuple(node_ids)
            for function_id, node_ids
            in plan.function_replica_node_ids.items()
        }
        if set(candidate_map) != set(self.sfc.function_ids):
            raise ValueError("副本计划与SFC函数集合不一致。")

        return candidate_map

    def _operational_node_ids(
        self,
        infrastructure_state: FailureSnapshot,
    ) -> frozenset[int]:
        """同时检查轨旁MEC和中心云的局部、故障域两层状态。"""

        return frozenset(
            node.node_id
            for node in self.topology.compute_nodes
            if infrastructure_state.is_node_operational(
                node_id=node.node_id,
                topology=self.topology,
            )
        )

    def _cold_activated_pairs(
        self,
        decision: FastTimescaleDecision,
    ) -> set[tuple[int, int]]:
        """把发生冷启动的函数编号还原为函数—执行节点对。"""

        if decision.request_success is not True:
            return set()

        selected_by_function = dict(
            zip(
                self.sfc.function_ids,
                decision.selected_execution_node_ids,
            )
        )
        return {
            (function_id, selected_by_function[function_id])
            for function_id in decision.cold_start_function_ids
        }

    def _audit(
        self,
        *,
        request_count: int,
        replica_count: int | dict[int, int],
        candidate_map: dict[int, tuple[int, ...]],
        decision: FastTimescaleDecision,
    ) -> SlotConstraintAudit:
        """用同一参数口径审计初始方案和修复候选方案。"""

        return self.constraint_auditor.audit(
            request_count=request_count,
            expected_replica_count=replica_count,
            candidate_map=candidate_map,
            selected_execution_node_ids=(
                decision.selected_execution_node_ids
            ),
            request_success=decision.request_success,
            function_hot_node_ids=(
                decision.function_hot_node_ids
            ),
            cold_activated_pairs=(
                self._cold_activated_pairs(decision)
            ),
        )

    def _count_plan_changes(
        self,
        previous_map: dict[int, tuple[int, ...]] | None,
        current_map: dict[int, tuple[int, ...]],
    ) -> int:
        """统计部署发生变化的函数数，首时隙视为初始部署。"""

        if previous_map is None:
            return len(self.sfc.function_ids)
        if set(previous_map) != set(self.sfc.function_ids):
            raise ValueError("上一时隙副本计划与SFC函数集合不一致。")

        return sum(
            previous_map[function_id]
            != current_map[function_id]
            for function_id in self.sfc.function_ids
        )

    def _node_cost_rates(
        self,
        node_id: int,
    ) -> tuple[float, float]:
        """根据节点类型返回CPU和内存单价。"""

        node = self.node_map[node_id]
        if node.node_type is NodeType.CLOUD:
            return (
                self.cost_rates.cloud_cpu_cost_per_unit,
                self.cost_rates.cloud_memory_cost_per_mb_second,
            )
        return (
            self.cost_rates.edge_cpu_cost_per_unit,
            self.cost_rates.edge_memory_cost_per_mb_second,
        )

    def _calculate_run_cost(
        self,
        audit: SlotConstraintAudit,
    ) -> float:
        """用最终审计中的真实CPU、内存需求计算运行成本。"""

        run_cost = 0.0
        demand_node_ids = (
            set(audit.node_cpu_demand)
            | set(audit.node_memory_demand_mb)
        )
        for node_id in demand_node_ids:
            # 未知节点已经由审计器判为违规；不可行方案仍需正常
            # 返回拒绝结果，因此这里跳过无法计价的未知节点。
            if node_id not in self.node_map:
                continue
            cpu_rate, memory_rate = self._node_cost_rates(node_id)
            run_cost += (
                audit.node_cpu_demand.get(node_id, 0.0)
                * cpu_rate
            )
            run_cost += (
                audit.node_memory_demand_mb.get(node_id, 0.0)
                * self.slot_seconds
                * memory_rate
            )

        return run_cost

    def execute(
        self,
        slot_input: FastSlotInput,
    ) -> FastSlotExecutionResult:
        """执行一个快时隙；最终约束不满足时只拒绝、不运行SFC。"""

        train_state = slot_input.train_state
        if (
            slot_input.infrastructure_state.time_slot
            != train_state.time_slot
        ):
            raise ValueError("基础设施状态与列车状态不属于同一时隙。")

        operational_node_ids = self._operational_node_ids(
            slot_input.infrastructure_state
        )
        if slot_input.deployment_intent is None:
            candidate_map = self._build_candidate_map(slot_input)
            expected_replica_counts: int | dict[int, int] = (
                slot_input.slow_decision.replica_count
            )
            retained_hot_sets: dict[int, frozenset[int]] | None = None
            retention_role_flags: dict[int, tuple[bool, ...]] | None = None
        else:
            candidate_map = self._candidate_map_from_intent(
                slot_input.deployment_intent,
                time_slot=train_state.time_slot,
            )
            expected_replica_counts = {
                function_id: len(node_ids)
                for function_id, node_ids in candidate_map.items()
            }
            # 先读取历史状态，当前意图只作为候选角色标记参与审计；真正的
            # 到期时间必须等快层修复结束后再绑定到最终部署节点。
            retained_hot_sets = self._retained_hot_nodes(
                time_slot=train_state.time_slot,
                operational_node_ids=operational_node_ids,
            )
            retention_role_flags = self._retention_role_flags(
                slot_input.deployment_intent,
                time_slot=train_state.time_slot,
            )
        fast_state = FastTimescaleState(
            time_slot=train_state.time_slot,
            serving_mec=train_state.serving_mec,
            remaining_dwell_time_s=(
                train_state.remaining_dwell_time_s
            ),
            request_count=slot_input.request_count,
            function_ids=tuple(self.sfc.function_ids),
            candidate_node_ids=candidate_map,
            operational_node_ids=operational_node_ids,
        )

        # PRIMARY_WARM只在临近切换且确实存在备用副本时临时预热。
        if slot_input.deployment_intent is None:
            if slot_input.slow_decision is None:
                raise ValueError("Legacy execution requires a slow decision.")
            backup_activation_triggered = (
                slot_input.slow_decision.retention_policy
                is RetentionPolicy.PRIMARY_WARM
                and slot_input.slow_decision.use_redundancy
                and train_state.remaining_dwell_time_s
                <= self.handover_hot_window_s
            )
            initial_decision = build_fast_decision_for_plan(
                state=fast_state,
                retention_policy=(
                    slot_input.slow_decision.retention_policy
                ),
                backup_activation_triggered=(
                    backup_activation_triggered
                ),
            )
            retained_hot_sets = {
                function_id: frozenset(node_ids)
                for function_id, node_ids
                in initial_decision.function_hot_node_ids.items()
            }
        else:
            if retained_hot_sets is None or retention_role_flags is None:
                raise RuntimeError("Explicit retention state was not initialized.")
            effective_hot_sets = self._effective_hot_nodes(
                retained_hot_nodes=retained_hot_sets,
                candidate_map=candidate_map,
                retention_role_flags=retention_role_flags,
                operational_node_ids=operational_node_ids,
            )
            initial_decision = build_fast_decision_for_hot_nodes(
                state=fast_state,
                function_hot_node_ids={
                    function_id: tuple(sorted(node_ids))
                    for function_id, node_ids in effective_hot_sets.items()
                },
                backup_activation_triggered=False,
            )
        initial_audit = self._audit(
            request_count=slot_input.request_count,
            replica_count=expected_replica_counts,
            candidate_map=candidate_map,
            decision=initial_decision,
        )
        convex_result: FastConvexSchedulingResult | None = None
        scheduled_batches = ()

        if slot_input.deployment_intent is None:
            # 历史规则入口继续使用旧修复器；这条分支绝不会作为 DPPO
            # 数学求解失败后的回退路径。
            optimization = self.fast_optimizer.optimize(
                state=fast_state,
                slow_decision=slot_input.slow_decision,
                initial_decision=initial_decision,
                initial_audit=initial_audit,
            )
            final_candidate_map = dict(optimization.function_replica_node_ids)
            final_decision = optimization.decision
            final_audit = optimization.final_audit
            repair_attempted = optimization.attempted
            repair_succeeded = optimization.succeeded
            repair_reason = optimization.reason
            evaluated_candidate_count = optimization.evaluated_candidate_count
        else:
            if self.fast_convex_scheduler is None:
                raise RuntimeError(
                    "DPPO explicit intent requires FastConvexScheduler."
                )
            deployment_audit = self.constraint_auditor.audit_deployment(
                expected_replica_count=expected_replica_counts,
                candidate_map=candidate_map,
                function_hot_node_ids=initial_decision.function_hot_node_ids,
                operational_node_ids=operational_node_ids,
            )
            initial_audit = deployment_audit
            if deployment_audit.all_constraints_met:
                convex_result = self.fast_convex_scheduler.schedule(
                    state=fast_state,
                    function_hot_node_ids=initial_decision.function_hot_node_ids,
                )
            else:
                convex_result = FastConvexSchedulingResult(
                    succeeded=False,
                    solver_status="not_run",
                    objective_value=None,
                    solve_time_seconds=0.0,
                    path_node_ids=(),
                    path_fractions=(),
                    scheduled_batches=(),
                    reason="慢层部署审计未通过，未调用 CLARABEL。",
                )
            scheduled_batches = convex_result.scheduled_batches
            final_candidate_map = dict(candidate_map)
            final_audit = self.constraint_auditor.audit_scheduled_batches(
                request_count=slot_input.request_count,
                expected_replica_count=expected_replica_counts,
                candidate_map=final_candidate_map,
                function_hot_node_ids=initial_decision.function_hot_node_ids,
                scheduled_batches=scheduled_batches,
            )
            representative_path = (
                scheduled_batches[0].execution_node_ids
                if scheduled_batches
                else ()
            )
            if slot_input.request_count == 0 and convex_result.succeeded:
                request_success: bool | None = None
            else:
                request_success = bool(
                    convex_result.succeeded and final_audit.all_constraints_met
                )
            final_decision = replace(
                initial_decision,
                selected_execution_node_ids=representative_path,
                failover_function_ids=tuple(
                    function_id
                    for function_id, node_id in zip(
                        self.sfc.function_ids,
                        representative_path,
                    )
                    if node_id != candidate_map[function_id][0]
                ),
                cold_start_function_ids=tuple(
                    function_id
                    for function_id, node_id in zip(
                        self.sfc.function_ids,
                        representative_path,
                    )
                    if node_id
                    not in initial_decision.function_hot_node_ids[function_id]
                ),
                unavailable_function_ids=(),
                request_success=request_success,
            )
            repair_attempted = slot_input.request_count > 0
            repair_succeeded = convex_result.succeeded
            repair_reason = convex_result.reason
            evaluated_candidate_count = len(convex_result.path_node_ids)

        constraint_rejected = (
            repair_succeeded is False
            or not final_audit.all_constraints_met
            or (
                slot_input.request_count > 0
                and final_decision.request_success is not True
            )
        )

        # 只有最终方案可执行时才写入跨时隙状态。否则故障或不可行的部署
        # 不能污染后续时隙的温热记录。
        if (
            slot_input.deployment_intent is not None
            and not constraint_rejected
        ):
            self._apply_final_intent_retention(
                intent=slot_input.deployment_intent,
                final_candidate_map=final_candidate_map,
                time_slot=train_state.time_slot,
            )
            retained_hot_sets = self._retained_hot_nodes(
                time_slot=train_state.time_slot,
                operational_node_ids=operational_node_ids,
            )

        end_to_end_delay_ms: float | None = None
        deadline_met: bool | None = None
        transmission_delay_ms: float | None = None
        execution_delay_ms: float | None = None
        cold_start_delay_ms = 0.0
        failover_delay_ms = 0.0
        route_cost = 0.0
        cold_start_pairs: set[tuple[int, int]] = set()
        all_failover_function_ids: set[int] = set(
            final_decision.failover_function_ids
        )

        # 这是安全边界：只有最终审计可行且执行路径完整，才调用
        # 真实SFC执行器，杜绝“先执行、后发现约束违规”。
        if (
            not constraint_rejected
            and final_decision.request_success is True
        ):
            batches_to_execute = (
                scheduled_batches
                if convex_result is not None
                else (
                    # 历史单路径执行也转换为统一批次循环，避免维护两套计费公式。
                    FastScheduledBatch(
                        slot_input.request_count,
                        final_decision.selected_execution_node_ids,
                    ),
                )
            )
            weighted_delay = 0.0
            weighted_transmission = 0.0
            weighted_execution = 0.0
            for batch in batches_to_execute:
                batch_cold_ids: set[int] = set()
                for function_id, node_id in zip(
                    self.sfc.function_ids,
                    batch.execution_node_ids,
                ):
                    if node_id != candidate_map[function_id][0]:
                        all_failover_function_ids.add(function_id)
                    pair = (function_id, node_id)
                    if (
                        node_id not in initial_decision.function_hot_node_ids[function_id]
                        and pair not in cold_start_pairs
                    ):
                        batch_cold_ids.add(function_id)
                        cold_start_pairs.add(pair)

                sfc_result = execute_sfc_batch(
                    functions=self.functions,
                    sfc=self.sfc,
                    placement_node_ids=list(batch.execution_node_ids),
                    source_node_id=train_state.serving_mec,
                    input_size_mb_per_request=self.input_size_mb_per_request,
                    request_count=batch.request_count,
                    network=self.network,
                    cold_start_function_ids=batch_cold_ids,
                    return_result_to_source=self.return_result_to_source,
                )
                batch_failover_delay = (
                    sum(
                        node_id != candidate_map[function_id][0]
                        for function_id, node_id in zip(
                            self.sfc.function_ids,
                            batch.execution_node_ids,
                        )
                    )
                    * self.failover_delay_ms_per_function
                )
                batch_delay = (
                    sfc_result.total_end_to_end_delay_ms
                    + batch_failover_delay
                )
                weighted_delay += batch_delay * batch.request_count
                weighted_transmission += (
                    sfc_result.total_transmission_delay_ms
                    * batch.request_count
                )
                weighted_execution += (
                    sfc_result.total_execution_delay_ms
                    * batch.request_count
                )
                failover_delay_ms += batch_failover_delay
                cold_start_delay_ms += sfc_result.total_cold_start_delay_ms
                route_cost += sfc_result.total_routing_cost

            end_to_end_delay_ms = weighted_delay / slot_input.request_count
            transmission_delay_ms = (
                weighted_transmission / slot_input.request_count
            )
            execution_delay_ms = weighted_execution / slot_input.request_count
            deadline_met = end_to_end_delay_ms <= self.sfc.deadline_ms

        active_pairs = {
            (function_id, node_id)
            for function_id, node_ids
            in final_decision.function_hot_node_ids.items()
            for node_id in node_ids
        }
        active_pairs.update(
            self._cold_activated_pairs(final_decision)
        )
        active_pairs.update(cold_start_pairs)
        active_memory_mb = sum(
            self.function_map[function_id].memory_mb
            for function_id, _ in active_pairs
        )
        run_cost = self._calculate_run_cost(final_audit)
        cold_start_cost = (
            cold_start_delay_ms
            * self.cost_rates.cold_start_cost_per_ms
        )
        used_cloud = (
            self.cloud_node_id is not None
            and any(
                self.cloud_node_id in node_ids
                for node_ids in final_candidate_map.values()
            )
        )

        return FastSlotExecutionResult(
            function_replica_node_ids=final_candidate_map,
            function_hot_node_ids=dict(
                final_decision.function_hot_node_ids
            ),
            initial_candidate_map=dict(candidate_map),
            retained_hot_node_ids_by_function={
                function_id: frozenset(node_ids)
                for function_id, node_ids in retained_hot_sets.items()
            },
            selected_execution_node_ids=(
                final_decision.selected_execution_node_ids
            ),
            initial_audit=initial_audit,
            final_audit=final_audit,
            fast_repair_attempted=repair_attempted,
            fast_repair_succeeded=repair_succeeded,
            fast_repair_reason=repair_reason,
            request_success=final_decision.request_success,
            deadline_met=deadline_met,
            end_to_end_delay_ms=end_to_end_delay_ms,
            cold_start_delay_ms=cold_start_delay_ms,
            active_memory_mb=active_memory_mb,
            plan_change_count=self._count_plan_changes(
                previous_map=slot_input.previous_candidate_map,
                current_map=final_candidate_map,
            ),
            used_cloud=used_cloud,
            constraint_rejected=constraint_rejected,
            run_cost=run_cost,
            route_cost=route_cost,
            cold_start_cost=cold_start_cost,
            backup_activation_triggered=(
                final_decision.backup_activation_triggered
            ),
            failover_function_ids=(
                tuple(
                    function_id
                    for function_id in self.sfc.function_ids
                    if function_id in all_failover_function_ids
                )
            ),
            cold_start_function_ids=(
                tuple(
                    function_id
                    for function_id in self.sfc.function_ids
                    if any(pair[0] == function_id for pair in cold_start_pairs)
                )
            ),
            unavailable_function_ids=(
                final_decision.unavailable_function_ids
            ),
            transmission_delay_ms=transmission_delay_ms,
            execution_delay_ms=execution_delay_ms,
            failover_delay_ms=failover_delay_ms,
            active_instance_count=len(active_pairs),
            fast_repair_evaluated_candidate_count=(
                evaluated_candidate_count
            ),
            fast_solver_status=(
                "not_used" if convex_result is None else convex_result.solver_status
            ),
            fast_solver_objective_value=(
                None if convex_result is None else convex_result.objective_value
            ),
            fast_solver_time_seconds=(
                0.0 if convex_result is None else convex_result.solve_time_seconds
            ),
            scheduled_request_counts=tuple(
                batch.request_count for batch in scheduled_batches
            ),
            scheduled_execution_node_ids=tuple(
                batch.execution_node_ids for batch in scheduled_batches
            ),
            cold_start_function_node_pairs=tuple(sorted(cold_start_pairs)),
        )


# 在本模块保留统一来源的单副本规划器符号，便于两个调用方
# 使用同一个定义；这里没有重新声明新的规划器或Protocol。
__all__ = [
    "FastSlotExecutionResult",
    "FastSlotExecutor",
    "FastSlotInput",
    "RuntimeCostRates",
    "SingleReplicaPlanner",
]
