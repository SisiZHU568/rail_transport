"""慢层模板约束下的确定性快层可行性修复。"""

from dataclasses import dataclass, replace
from itertools import permutations, product

from src.constraint_audit import SlotConstraintAuditor
from src.entities import (
    ServerlessFunction,
    SFCType,
    SlotConstraintAudit,
)
from src.topology import LinearRailTopology
from src.two_timescale_control import (
    FastTimescaleDecision,
    FastTimescaleState,
    SlowTimescaleDecision,
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
    ) -> None:
        """保存搜索所需模型，并建立稳定的节点顺序和位置索引。"""

        self.functions = list(functions)
        self.sfc = sfc
        self.topology = topology
        self.auditor = auditor
        self.return_result_to_source = return_result_to_source
        self.node_ids = tuple(
            sorted(
                site.node.node_id
                for site in topology.sites
            )
        )
        self.node_positions = {
            site.node.node_id: site.position_m
            for site in topology.sites
        }

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
        int,
        int,
        int,
        float,
        tuple[int, ...],
    ]:
        """计算确定性字典序评分，元组越小越优。"""

        initial_map = state.candidate_node_ids
        changed_function_count = sum(
            candidate_map[function_id]
            != initial_map[function_id]
            for function_id in state.function_ids
        )
        replaced_replica_count = sum(
            len(
                set(initial_map[function_id])
                - set(candidate_map[function_id])
            )
            for function_id in state.function_ids
        )

        # 路径从当前接入MEC出发，按SFC顺序访问执行节点；
        # 若配置要求返回结果，再把返回接入MEC的距离计入评分。
        path = [state.serving_mec]
        path.extend(decision.selected_execution_node_ids)
        if (
            self.return_result_to_source
            and decision.selected_execution_node_ids
        ):
            path.append(state.serving_mec)

        distance = sum(
            abs(
                self.node_positions[left]
                - self.node_positions[right]
            )
            for left, right in zip(path, path[1:])
        )
        flattened_ids = tuple(
            node_id
            for function_id in state.function_ids
            for node_id in candidate_map[function_id]
        )

        # 五级优先级依次为：修改函数数、副本替换数、冷启动数、
        # 地理路径距离和节点编号。最后一项保证同分时结果仍稳定。
        return (
            int(changed_function_count),
            int(replaced_replica_count),
            len(decision.cold_start_function_ids),
            float(distance),
            flattened_ids,
        )

    def optimize(
        self,
        state: FastTimescaleState,
        slow_decision: SlowTimescaleDecision,
        initial_decision: FastTimescaleDecision,
        initial_audit: SlotConstraintAudit,
    ) -> FastOptimizationResult:
        """在慢层副本数和主备模式不变的前提下修复部署。"""

        if slow_decision.replica_count <= 0:
            raise ValueError("慢层副本数量必须大于0。")

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

        required_count = slow_decision.replica_count
        minimum_domains = (
            self.auditor.reliability_model
            .minimum_distinct_fault_domains
        )

        # 快层只能搬迁副本，不能突破慢层给出的副本数量上限。
        if required_count < minimum_domains:
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

        # 搜索空间只包含拓扑中存在且本时隙正常运行的轨旁MEC。
        operational_ids = tuple(
            sorted(
                node_id
                for node_id in state.operational_node_ids
                if node_id in self.node_ids
            )
        )
        if len(operational_ids) < required_count:
            return self._failure_result(
                state=state,
                initial_decision=initial_decision,
                initial_audit=initial_audit,
                reason=(
                    f"当前只有{len(operational_ids)}个正常MEC，"
                    f"少于慢层要求的{required_count}个副本。"
                ),
                evaluated_candidate_count=0,
            )

        # 排列而非组合：元组首项是主节点，因此(0,1)和(1,0)
        # 对执行路径、接管和冷启动具有不同含义。
        per_function_options = tuple(
            permutations(
                operational_ids,
                required_count,
            )
        )
        ranked_candidates: list[
            tuple[
                tuple[
                    int,
                    int,
                    int,
                    float,
                    tuple[int, ...],
                ],
                dict[int, tuple[int, ...]],
                FastTimescaleDecision,
            ]
        ] = []

        for node_tuples in product(
            per_function_options,
            repeat=len(state.function_ids),
        ):
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
            decision = build_fast_decision_for_plan(
                state=repaired_state,
                standby_mode=slow_decision.standby_mode,
                backup_activation_triggered=(
                    initial_decision
                    .backup_activation_triggered
                ),
                previously_hot_node_ids=(
                    initial_decision.function_hot_node_ids
                ),
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
                expected_replica_count=required_count,
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
