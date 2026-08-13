"""两阶段词典序 CLARABEL 快层：服务优先，资源成本次优。"""

from dataclasses import dataclass
import math
from time import perf_counter

import cvxpy as cp

from src.failure_process import FailureSnapshot
from src.fast_resource_model import FastResourceConfig, NetworkSnapshot
from src.instance_lifecycle import LifecycleSnapshot, LifecycleStatus
from src.queue_manager import AllocationOperation, FastAllocationPlan, QueueKey
from src.queue_state import QueueSnapshot, StageFlowConfig


@dataclass(frozen=True)
class FastResourceOptimizationResult:
    succeeded: bool
    code: str
    plan: FastAllocationPlan | None
    is_dcp: bool
    primary_status: str
    secondary_status: str
    primary_solve_seconds: float
    secondary_solve_seconds: float
    primary_optimum: float
    secondary_primary_value: float
    lex_tolerance: float
    service_shortfall_equivalent_bits: float
    uplink_service_bits: float
    executed_physical_bits: float
    energy_cost: float
    cpu_cost: float
    maximum_residual: float


@dataclass
class _QueueVariable:
    key: QueueKey
    queue_mbit: float
    gamma: float
    weight: float
    service: cp.Variable | None
    shortfall: cp.Variable | cp.Expression
    bandwidth: cp.Variable | None = None
    power: cp.Variable | None = None
    work_gcycle: cp.Expression | None = None
    active_time: cp.Variable | None = None
    energy_joule: cp.Variable | None = None


class FastResourceOptimizer:
    """无状态纯求解器；不访问环境，也不直接修改任何快照。"""

    _MBIT = 1e6

    def __init__(
        self,
        config: FastResourceConfig,
        flow_config: StageFlowConfig,
    ) -> None:
        self.config = config
        self.flow_config = flow_config

    @staticmethod
    def _warm_counts(snapshot: LifecycleSnapshot) -> dict[tuple[int, int], int]:
        counts: dict[tuple[int, int], int] = {}
        for batch in snapshot.batches:
            if batch.status is LifecycleStatus.WARM:
                key = (batch.function_id, batch.node_id)
                counts[key] = counts.get(key, 0) + batch.count
        return counts

    @staticmethod
    def _queue_weight(
        queue_snapshot: QueueSnapshot,
        batch_ids: set[str],
        slot_seconds: float,
    ) -> float:
        batches = {
            item.batch_id: item for item in queue_snapshot.batches
            if item.batch_id in batch_ids
        }
        if not batches:
            return 1.0
        current_time = queue_snapshot.current_slot * slot_seconds
        slack = min(
            item.absolute_deadline_time - current_time
            for item in batches.values()
        )
        return slot_seconds / max(slack, slot_seconds)

    def _failure_result(
        self,
        *,
        is_dcp: bool = False,
        primary_status: str = "error",
        primary_seconds: float = 0.0,
    ) -> FastResourceOptimizationResult:
        return FastResourceOptimizationResult(
            False, "FAST_SOLVER_FAILURE", None, is_dcp,
            primary_status, "not_run", primary_seconds, 0.0,
            math.inf, math.inf, 0.0, math.inf, 0.0, 0.0,
            math.inf, math.inf, math.inf,
        )

    @staticmethod
    def _expression_value(value: cp.Expression | float | int) -> float:
        """CVXPY 空求和可能退化为 Python 数值，统一读取结果。"""

        if isinstance(value, cp.Expression):
            return 0.0 if value.value is None else float(value.value)
        return float(value)

    def solve(
        self,
        queue_snapshot: QueueSnapshot,
        lifecycle_snapshot: LifecycleSnapshot,
        failure_snapshot: FailureSnapshot,
        network_snapshot: NetworkSnapshot,
    ) -> FastResourceOptimizationResult:
        """连续求解一级服务能力和二级最低成本，并输出可提交计划。"""

        if (
            queue_snapshot.current_slot != lifecycle_snapshot.current_slot
            or queue_snapshot.current_slot != failure_snapshot.time_slot
            or queue_snapshot.current_slot != network_snapshot.current_slot
        ):
            return self._failure_result(primary_status="stale_input")
        available_uplink = [
            item for item in queue_snapshot.uplink_fragments
            if item.available_slot <= queue_snapshot.current_slot
        ]
        available_stage = [
            item for item in queue_snapshot.stage_fragments
            if item.available_slot <= queue_snapshot.current_slot
        ]
        if not available_uplink and not available_stage:
            empty_plan = FastAllocationPlan(
                queue_snapshot.version,
                lifecycle_snapshot.version,
                failure_snapshot.version,
                network_snapshot.version,
                queue_snapshot.current_slot,
                (),
            )
            return FastResourceOptimizationResult(
                True, "OK", empty_plan, True, "not_run", "not_run",
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
            )

        constraints: list[cp.Constraint] = []
        variables: list[_QueueVariable] = []
        warm_counts = self._warm_counts(lifecycle_snapshot)
        slot = self.config.slot_seconds

        uplink_groups: dict[int, list[object]] = {}
        for fragment in available_uplink:
            uplink_groups.setdefault(fragment.service_id, []).append(fragment)
        for service_id, fragments in sorted(uplink_groups.items()):
            queue_mbit = sum(item.input_equivalent_bits for item in fragments) / self._MBIT
            service = cp.Variable(nonneg=True, name=f"ul_q_{service_id}")
            shortfall = cp.Variable(nonneg=True, name=f"ul_xi_{service_id}")
            bandwidth = cp.Variable(nonneg=True, name=f"ul_b_{service_id}")
            power = cp.Variable(nonneg=True, name=f"ul_p_{service_id}")
            constraints.append(service + shortfall == queue_mbit)
            if network_snapshot.channel_gain <= 0.0:
                constraints.append(service == 0.0)
            else:
                noise_mhz = self.config.noise_psd_watt_per_hz * self._MBIT
                rate_mbps = -cp.rel_entr(
                    bandwidth,
                    bandwidth
                    + network_snapshot.channel_gain * power / noise_mhz,
                ) / math.log(2.0)
                constraints.append(service <= slot * rate_mbps)
            variables.append(
                _QueueVariable(
                    QueueKey.uplink(service_id), queue_mbit, 1.0,
                    self._queue_weight(
                        queue_snapshot,
                        {item.batch_id for item in fragments},
                        slot,
                    ),
                    service, shortfall, bandwidth, power,
                )
            )
        if uplink_groups:
            constraints.extend(
                [
                    cp.sum([item.bandwidth for item in variables if item.bandwidth is not None])
                    <= network_snapshot.uplink_bandwidth_hz / self._MBIT,
                    cp.sum([item.power for item in variables if item.power is not None])
                    <= network_snapshot.maximum_uplink_power_watt,
                ]
            )

        stage_groups: dict[QueueKey, list[object]] = {}
        for fragment in available_stage:
            stage_groups.setdefault(
                QueueKey.stage(
                    fragment.stage_id,
                    fragment.location,
                    fragment.routing_target_node,
                ),
                [],
            ).append(fragment)
        node_times: dict[int, list[cp.Expression]] = {}
        node_work: dict[int, list[cp.Expression]] = {}
        for key, fragments in sorted(stage_groups.items()):
            gamma = self.flow_config.gamma(key.stage_id)
            physical_mbit = (
                sum(item.input_equivalent_bits for item in fragments)
                * gamma / self._MBIT
            )
            can_execute = (
                failure_snapshot.effective_node_up.get(key.location, False)
                and warm_counts.get((key.stage_id, key.location), 0) > 0
                and (key.optional_target is None or key.optional_target == key.location)
                and (key.stage_id, key.location) in self.config.vnfs
            )
            if not can_execute:
                shortfall = cp.Constant(physical_mbit)
                variables.append(
                    _QueueVariable(
                        key, physical_mbit, gamma,
                        self._queue_weight(
                            queue_snapshot,
                            {item.batch_id for item in fragments}, slot,
                        ),
                        None, shortfall,
                    )
                )
                continue
            service = cp.Variable(nonneg=True, name=f"z_{key.stage_id}_{key.location}_{key.routing_target_node}")
            shortfall = cp.Variable(nonneg=True, name=f"xi_{key.stage_id}_{key.location}_{key.routing_target_node}")
            tau = cp.Variable(nonneg=True, name=f"tau_{key.stage_id}_{key.location}_{key.routing_target_node}")
            energy = cp.Variable(nonneg=True, name=f"energy_{key.stage_id}_{key.location}_{key.routing_target_node}")
            resource = self.config.vnfs[(key.stage_id, key.location)]
            work = resource.cpu_cycles_per_physical_bit * service / 1000.0
            warm = warm_counts[(key.stage_id, key.location)]
            constraints.extend(
                [
                    service + shortfall == physical_mbit,
                    work <= resource.single_instance_max_cpu_cycles_per_second / 1e9 * tau,
                    tau <= warm * slot,
                    cp.PowCone3D(
                        energy / (self.config.nodes[key.location].dvfs_kappa * 1e27),
                        tau,
                        work,
                        alpha=1.0 / 3.0,
                    ),
                ]
            )
            node_times.setdefault(key.location, []).append(tau)
            node_work.setdefault(key.location, []).append(work)
            variables.append(
                _QueueVariable(
                    key, physical_mbit, gamma,
                    self._queue_weight(
                        queue_snapshot,
                        {item.batch_id for item in fragments}, slot,
                    ),
                    service, shortfall,
                    work_gcycle=work, active_time=tau, energy_joule=energy,
                )
            )
        for node_id, times in node_times.items():
            node = self.config.nodes[node_id]
            constraints.extend(
                [
                    cp.sum(times) <= node.core_count * slot,
                    cp.sum(node_work[node_id])
                    <= node.maximum_cpu_cycles_per_second / 1e9 * slot,
                ]
            )

        primary = cp.sum(
            [item.weight * item.shortfall / item.gamma for item in variables]
        )
        primary_problem = cp.Problem(cp.Minimize(primary), constraints)
        if not primary_problem.is_dcp():
            return self._failure_result(is_dcp=False, primary_status="not_dcp")
        start = perf_counter()
        try:
            primary_problem.solve(
                solver=self.config.solver_name,
                max_iter=self.config.max_iterations,
            )
        except (cp.SolverError, ArithmeticError, ValueError):
            return self._failure_result(
                is_dcp=True,
                primary_status="solver_error",
                primary_seconds=perf_counter() - start,
            )
        primary_seconds = perf_counter() - start
        acceptable = {cp.OPTIMAL}
        if self.config.allow_optimal_inaccurate:
            acceptable.add(cp.OPTIMAL_INACCURATE)
        if primary_problem.status not in acceptable or primary.value is None:
            return self._failure_result(
                is_dcp=True,
                primary_status=str(primary_problem.status),
                primary_seconds=primary_seconds,
            )
        primary_optimum = float(primary.value)
        lex_tolerance = max(
            self.config.absolute_lex_tolerance,
            self.config.relative_lex_tolerance * max(1.0, abs(primary_optimum)),
        )
        constraints_secondary = [*constraints, primary <= primary_optimum + lex_tolerance]
        uplink_energy = slot * cp.sum(
            [item.power for item in variables if item.power is not None]
        )
        computation_energy = cp.sum(
            [item.energy_joule for item in variables if item.energy_joule is not None]
        )
        cpu_cost_expr = cp.sum(
            [
                self.config.nodes[item.key.location].cpu_price_per_second
                * item.active_time
                for item in variables if item.active_time is not None
            ]
        )
        energy_cost_expr = self.config.energy_price_per_joule * (
            uplink_energy + computation_energy
        )
        secondary_problem = cp.Problem(
            cp.Minimize(energy_cost_expr + cpu_cost_expr),
            constraints_secondary,
        )
        start = perf_counter()
        try:
            secondary_problem.solve(
                solver=self.config.solver_name,
                max_iter=self.config.max_iterations,
            )
        except (cp.SolverError, ArithmeticError, ValueError):
            return self._failure_result(
                is_dcp=True,
                primary_status=str(primary_problem.status),
                primary_seconds=primary_seconds,
            )
        secondary_seconds = perf_counter() - start
        if secondary_problem.status not in acceptable or primary.value is None:
            return self._failure_result(
                is_dcp=True,
                primary_status=str(primary_problem.status),
                primary_seconds=primary_seconds,
            )

        operations: list[AllocationOperation] = []
        total_shortfall_mbit = 0.0
        uplink_service_mbit = 0.0
        executed_mbit = 0.0
        maximum_residual = max(0.0, float(primary.value) - primary_optimum - lex_tolerance)
        for item in variables:
            shortfall = float(item.shortfall.value) if item.shortfall.value is not None else item.queue_mbit
            service = 0.0 if item.service is None or item.service.value is None else max(0.0, float(item.service.value))
            maximum_residual = max(
                maximum_residual,
                abs(service + shortfall - item.queue_mbit),
            )
            total_shortfall_mbit += shortfall / item.gamma
            if service <= self.config.residual_tolerance:
                continue
            if item.key.queue_type == "uplink":
                uplink_service_mbit += service
                operations.append(
                    AllocationOperation(
                        item.key, "uplink", service * self._MBIT,
                        destination_node_id=network_snapshot.serving_mec,
                    )
                )
            else:
                executed_mbit += service
                operations.append(
                    AllocationOperation(
                        item.key, "execute", service * self._MBIT,
                    )
                )
        if maximum_residual > self.config.residual_tolerance:
            return self._failure_result(
                is_dcp=True,
                primary_status=str(primary_problem.status),
                primary_seconds=primary_seconds,
            )
        plan = FastAllocationPlan(
            queue_snapshot.version,
            lifecycle_snapshot.version,
            failure_snapshot.version,
            network_snapshot.version,
            queue_snapshot.current_slot,
            tuple(operations),
        )
        return FastResourceOptimizationResult(
            True, "OK", plan, True,
            str(primary_problem.status), str(secondary_problem.status),
            primary_seconds, secondary_seconds, primary_optimum,
            float(primary.value), lex_tolerance,
            total_shortfall_mbit * self._MBIT,
            uplink_service_mbit * self._MBIT,
            executed_mbit * self._MBIT,
            self._expression_value(energy_cost_expr),
            self._expression_value(cpu_cost_expr),
            maximum_residual,
        )
