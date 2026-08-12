"""DPPO 固定部署下的快时间尺度凸优化请求调度。"""

from dataclasses import dataclass
from itertools import product
import math
from time import perf_counter

import cvxpy as cp
import numpy as np

from src.entities import NodeType, ServerlessFunction, SFCType
from src.network import TransferNetworkProtocol
from src.sfc_execution import execute_sfc_request
from src.topology import LinearRailTopology
from src.two_timescale_control import FastTimescaleState


@dataclass(frozen=True)
class FastScheduledBatch:
    """一组沿同一条完整 SFC 路径执行的整数请求。"""

    request_count: int
    execution_node_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if isinstance(self.request_count, bool) or self.request_count <= 0:
            raise ValueError("调度批次的请求数必须是正整数。")
        if not self.execution_node_ids:
            raise ValueError("调度批次必须包含完整执行路径。")


@dataclass(frozen=True)
class FastConvexSchedulingResult:
    """保存连续松弛解及其可执行整数批次。"""

    succeeded: bool
    solver_status: str
    objective_value: float | None
    solve_time_seconds: float
    path_node_ids: tuple[tuple[int, ...], ...]
    path_fractions: tuple[float, ...]
    scheduled_batches: tuple[FastScheduledBatch, ...]
    reason: str


@dataclass(frozen=True)
class _PathEstimate:
    """保存一条候选路径进入凸模型所需的固定系数。"""

    node_ids: tuple[int, ...]
    cost: float
    end_to_end_delay_ms: float
    cpu_per_request_by_node: dict[int, float]
    cold_pairs: frozenset[tuple[int, int]]


class FastConvexScheduler:
    """用 CLARABEL 在慢层已部署副本之间分配当前请求。"""

    def __init__(
        self,
        *,
        topology: LinearRailTopology,
        network: TransferNetworkProtocol,
        functions: list[ServerlessFunction],
        sfc: SFCType,
        input_size_mb_per_request: float,
        slot_seconds: float,
        edge_cpu_cost_per_unit: float,
        edge_memory_cost_per_mb_second: float,
        cloud_cpu_cost_per_unit: float,
        cloud_memory_cost_per_mb_second: float,
        cold_start_cost_per_ms: float,
        solver_name: str,
        max_iterations: int,
        feasibility_tolerance: float,
        return_result_to_source: bool = True,
    ) -> None:
        if solver_name != "CLARABEL":
            raise ValueError("快层求解器当前必须为 CLARABEL。")
        if (
            isinstance(max_iterations, bool)
            or not isinstance(max_iterations, int)
            or max_iterations <= 0
        ):
            raise ValueError("快层最大迭代次数必须是正整数。")
        nonnegative_values = (
            input_size_mb_per_request,
            edge_cpu_cost_per_unit,
            edge_memory_cost_per_mb_second,
            cloud_cpu_cost_per_unit,
            cloud_memory_cost_per_mb_second,
            cold_start_cost_per_ms,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in nonnegative_values):
            raise ValueError("快层数据量和成本单价必须是非负有限数。")
        if not math.isfinite(slot_seconds) or slot_seconds <= 0.0:
            raise ValueError("快时隙长度必须是正有限数。")
        if (
            not math.isfinite(feasibility_tolerance)
            or feasibility_tolerance <= 0.0
        ):
            raise ValueError("快层可行性容差必须是正有限数。")

        self.topology = topology
        self.network = network
        self.functions = list(functions)
        self.function_map = {
            function.function_id: function for function in self.functions
        }
        if len(self.function_map) != len(self.functions):
            raise ValueError("Serverless 函数编号不能重复。")
        if set(self.function_map) != set(sfc.function_ids):
            raise ValueError("函数列表必须与 SFC 函数集合完全一致。")

        self.sfc = sfc
        self.node_map = {node.node_id: node for node in topology.compute_nodes}
        self.input_size_mb_per_request = input_size_mb_per_request
        self.slot_seconds = slot_seconds
        self.edge_cpu_cost_per_unit = edge_cpu_cost_per_unit
        self.edge_memory_cost_per_mb_second = edge_memory_cost_per_mb_second
        self.cloud_cpu_cost_per_unit = cloud_cpu_cost_per_unit
        self.cloud_memory_cost_per_mb_second = cloud_memory_cost_per_mb_second
        self.cold_start_cost_per_ms = cold_start_cost_per_ms
        self.solver_name = solver_name
        self.max_iterations = max_iterations
        self.feasibility_tolerance = feasibility_tolerance
        self.return_result_to_source = return_result_to_source

    def _node_cost_rates(self, node_id: int) -> tuple[float, float]:
        node = self.node_map[node_id]
        if node.node_type is NodeType.CLOUD:
            return self.cloud_cpu_cost_per_unit, self.cloud_memory_cost_per_mb_second
        return self.edge_cpu_cost_per_unit, self.edge_memory_cost_per_mb_second

    def _base_hot_memory_cost(
        self,
        function_hot_node_ids: dict[int, tuple[int, ...]],
    ) -> float:
        cost = 0.0
        for function_id, node_ids in function_hot_node_ids.items():
            for node_id in node_ids:
                _, memory_rate = self._node_cost_rates(node_id)
                cost += (
                    self.function_map[function_id].memory_mb
                    * self.slot_seconds
                    * memory_rate
                )
        return cost

    def _estimate_path(
        self,
        *,
        state: FastTimescaleState,
        path: tuple[int, ...],
        hot_pairs: set[tuple[int, int]],
    ) -> _PathEstimate | None:
        cold_pairs = frozenset(
            (function_id, node_id)
            for function_id, node_id in zip(state.function_ids, path)
            if (function_id, node_id) not in hot_pairs
        )
        cold_function_ids = {function_id for function_id, _ in cold_pairs}

        try:
            # 单请求预计时延用于排除明显违反 SLA 的路径；整数批次执行后
            # 仍会用真实批量时延做最终检查。
            execution = execute_sfc_request(
                functions=self.functions,
                sfc=self.sfc,
                placement_node_ids=list(path),
                source_node_id=state.serving_mec,
                input_size_mb=self.input_size_mb_per_request,
                network=self.network,
                cold_start_function_ids=cold_function_ids,
                return_result_to_source=self.return_result_to_source,
            )
        except (KeyError, ValueError):
            return None
        if execution.total_end_to_end_delay_ms > self.sfc.deadline_ms:
            return None

        cpu_by_node: dict[int, float] = {}
        run_cost = 0.0
        for function_id, node_id in zip(state.function_ids, path):
            per_request_cpu = self.function_map[function_id].cpu_cycles_per_request
            cpu_by_node[node_id] = cpu_by_node.get(node_id, 0.0) + per_request_cpu
            cpu_rate, _ = self._node_cost_rates(node_id)
            run_cost += per_request_cpu * state.request_count * cpu_rate

        cold_cost = 0.0
        for function_id, node_id in cold_pairs:
            function = self.function_map[function_id]
            _, memory_rate = self._node_cost_rates(node_id)
            cold_cost += function.cold_start_time_ms * self.cold_start_cost_per_ms
            cold_cost += function.memory_mb * self.slot_seconds * memory_rate

        return _PathEstimate(
            node_ids=path,
            cost=(
                run_cost
                + execution.total_routing_cost * state.request_count
                + cold_cost
            ),
            end_to_end_delay_ms=execution.total_end_to_end_delay_ms,
            cpu_per_request_by_node=cpu_by_node,
            cold_pairs=cold_pairs,
        )

    def _build_eligible_paths(
        self,
        *,
        state: FastTimescaleState,
        function_hot_node_ids: dict[int, tuple[int, ...]],
    ) -> tuple[_PathEstimate, ...]:
        if tuple(state.function_ids) != tuple(self.sfc.function_ids):
            raise ValueError("快层状态中的函数顺序必须与 SFC 完全一致。")
        if set(function_hot_node_ids) != set(state.function_ids):
            raise ValueError("温实例映射必须覆盖完整 SFC。")

        hot_pairs = {
            (function_id, node_id)
            for function_id, node_ids in function_hot_node_ids.items()
            for node_id in node_ids
        }
        available_by_function: list[tuple[int, ...]] = []
        for function_id in state.function_ids:
            available = tuple(
                sorted(
                    node_id
                    for node_id in state.candidate_node_ids[function_id]
                    if node_id in state.operational_node_ids
                    and node_id in self.node_map
                )
            )
            if not available:
                return ()
            available_by_function.append(available)

        estimates = (
            estimate
            for path in sorted(product(*available_by_function))
            if (estimate := self._estimate_path(
                state=state,
                path=tuple(path),
                hot_pairs=hot_pairs,
            )) is not None
        )
        return tuple(estimates)

    def _memory_is_feasible(
        self,
        *,
        function_hot_node_ids: dict[int, tuple[int, ...]],
        activated_cold_pairs: set[tuple[int, int]],
    ) -> bool:
        active_pairs = {
            (function_id, node_id)
            for function_id, node_ids in function_hot_node_ids.items()
            for node_id in node_ids
        }
        active_pairs.update(activated_cold_pairs)
        memory_by_node: dict[int, float] = {}
        for function_id, node_id in active_pairs:
            if node_id not in self.node_map:
                return False
            memory_by_node[node_id] = (
                memory_by_node.get(node_id, 0.0)
                + self.function_map[function_id].memory_mb
            )
        return all(
            demand <= self.node_map[node_id].memory_capacity_mb + self.feasibility_tolerance
            for node_id, demand in memory_by_node.items()
        )

    def _allocation_is_feasible(
        self,
        *,
        paths: tuple[_PathEstimate, ...],
        counts: list[int],
        function_hot_node_ids: dict[int, tuple[int, ...]],
    ) -> bool:
        cpu_by_node: dict[int, float] = {}
        cold_pairs: set[tuple[int, int]] = set()
        for path, count in zip(paths, counts):
            if count <= 0:
                continue
            cold_pairs.update(path.cold_pairs)
            for node_id, per_request_cpu in path.cpu_per_request_by_node.items():
                cpu_by_node[node_id] = (
                    cpu_by_node.get(node_id, 0.0) + count * per_request_cpu
                )
        cpu_ok = all(
            demand <= self.node_map[node_id].cpu_capacity + self.feasibility_tolerance
            for node_id, demand in cpu_by_node.items()
        )
        return cpu_ok and self._memory_is_feasible(
            function_hot_node_ids=function_hot_node_ids,
            activated_cold_pairs=cold_pairs,
        )

    def _round_request_counts(
        self,
        *,
        request_count: int,
        fractions: tuple[float, ...],
        paths: tuple[_PathEstimate, ...],
        function_hot_node_ids: dict[int, tuple[int, ...]],
    ) -> tuple[FastScheduledBatch, ...] | None:
        raw_counts = [request_count * fraction for fraction in fractions]
        counts = [math.floor(value) for value in raw_counts]
        if not self._allocation_is_feasible(
            paths=paths,
            counts=counts,
            function_hot_node_ids=function_hot_node_ids,
        ):
            return None

        remaining = request_count - sum(counts)
        order = sorted(
            range(len(paths)),
            key=lambda index: (
                -(raw_counts[index] - counts[index]),
                paths[index].node_ids,
            ),
        )
        for index in order:
            if remaining == 0:
                break
            candidate_counts = list(counts)
            candidate_counts[index] += 1
            if self._allocation_is_feasible(
                paths=paths,
                counts=candidate_counts,
                function_hot_node_ids=function_hot_node_ids,
            ):
                counts = candidate_counts
                remaining -= 1
        if remaining != 0:
            return None

        return tuple(
            FastScheduledBatch(count, path.node_ids)
            for path, count in zip(paths, counts)
            if count > 0
        )

    def _failure(
        self,
        *,
        status: str,
        reason: str,
        solve_time_seconds: float,
        paths: tuple[_PathEstimate, ...] = (),
        fractions: tuple[float, ...] = (),
    ) -> FastConvexSchedulingResult:
        return FastConvexSchedulingResult(
            succeeded=False,
            solver_status=status,
            objective_value=None,
            solve_time_seconds=solve_time_seconds,
            path_node_ids=tuple(path.node_ids for path in paths),
            path_fractions=fractions,
            scheduled_batches=(),
            reason=reason,
        )

    def schedule(
        self,
        *,
        state: FastTimescaleState,
        function_hot_node_ids: dict[int, tuple[int, ...]],
    ) -> FastConvexSchedulingResult:
        """求解连续分流比例，并转换为容量可行的整数请求批次。"""

        if state.request_count == 0:
            return FastConvexSchedulingResult(
                succeeded=True,
                solver_status="not_run",
                objective_value=0.0,
                solve_time_seconds=0.0,
                path_node_ids=(),
                path_fractions=(),
                scheduled_batches=(),
                reason="当前时隙没有请求，无需调用求解器。",
            )

        paths = self._build_eligible_paths(
            state=state,
            function_hot_node_ids=function_hot_node_ids,
        )
        if not paths:
            return self._failure(
                status="not_run",
                reason="慢层部署中没有满足故障和 SLA 约束的完整 SFC 路径。",
                solve_time_seconds=0.0,
            )
        if not self._memory_is_feasible(
            function_hot_node_ids=function_hot_node_ids,
            activated_cold_pairs=set(),
        ):
            return self._failure(
                status="not_run",
                reason="慢层温实例内存需求超过节点容量。",
                solve_time_seconds=0.0,
                paths=paths,
            )

        # x[p] 是分到路径 p 的请求比例，不是新的副本部署动作。
        fractions_variable = cp.Variable(len(paths), nonneg=True)
        constraints = [cp.sum(fractions_variable) == 1.0, fractions_variable <= 1.0]
        for node_id, node in sorted(self.node_map.items()):
            coefficients = np.array(
                [path.cpu_per_request_by_node.get(node_id, 0.0) for path in paths],
                dtype=np.float64,
            )
            # 每请求 CPU 系数乘当前请求总数，才是本快时隙的总资源需求。
            constraints.append(
                state.request_count * coefficients @ fractions_variable
                <= node.cpu_capacity
            )
        path_costs = np.array([path.cost for path in paths], dtype=np.float64)
        base_memory_cost = self._base_hot_memory_cost(function_hot_node_ids)
        problem = cp.Problem(
            cp.Minimize(path_costs @ fractions_variable + base_memory_cost),
            constraints,
        )

        started_at = perf_counter()
        try:
            problem.solve(
                solver=self.solver_name,
                max_iter=self.max_iterations,
                tol_feas=self.feasibility_tolerance,
                tol_gap_abs=self.feasibility_tolerance,
                verbose=False,
            )
        except Exception as error:  # CVXPY 会用不同异常包装底层求解失败。
            return self._failure(
                status="solver_error",
                reason=f"CLARABEL 求解失败：{error}",
                solve_time_seconds=perf_counter() - started_at,
                paths=paths,
            )
        elapsed = perf_counter() - started_at
        status = str(problem.status or "unknown")
        if status != cp.OPTIMAL:
            return self._failure(
                status=status,
                reason=f"CLARABEL 未返回严格 optimal 状态：{status}。",
                solve_time_seconds=elapsed,
                paths=paths,
            )

        raw_fractions = fractions_variable.value
        if raw_fractions is None:
            return self._failure(
                status=status,
                reason="CLARABEL 没有返回路径分流比例。",
                solve_time_seconds=elapsed,
                paths=paths,
            )
        values = np.asarray(raw_fractions, dtype=np.float64).reshape(-1)
        tolerance = self.feasibility_tolerance
        if (
            values.size != len(paths)
            or not np.all(np.isfinite(values))
            or np.any(values < -tolerance)
            or np.any(values > 1.0 + tolerance)
            or not math.isclose(float(values.sum()), 1.0, abs_tol=tolerance)
        ):
            return self._failure(
                status=status,
                reason="CLARABEL 返回了非有限或超出可行性容差的比例。",
                solve_time_seconds=elapsed,
                paths=paths,
            )
        clipped = np.clip(values, 0.0, 1.0)
        fractions = tuple(float(value / clipped.sum()) for value in clipped)
        batches = self._round_request_counts(
            request_count=state.request_count,
            fractions=fractions,
            paths=paths,
            function_hot_node_ids=function_hot_node_ids,
        )
        if batches is None:
            return self._failure(
                status=status,
                reason="连续比例无法转换为满足 CPU 和内存容量的整数请求批次。",
                solve_time_seconds=elapsed,
                paths=paths,
                fractions=fractions,
            )

        objective = problem.value
        return FastConvexSchedulingResult(
            succeeded=True,
            solver_status=status,
            objective_value=(
                float(objective)
                if objective is not None and math.isfinite(float(objective))
                else None
            ),
            solve_time_seconds=elapsed,
            path_node_ids=tuple(path.node_ids for path in paths),
            path_fractions=fractions,
            scheduled_batches=batches,
            reason="CLARABEL 已生成容量可行的整数请求调度。",
        )


__all__ = [
    "FastConvexScheduler",
    "FastConvexSchedulingResult",
    "FastScheduledBatch",
]
