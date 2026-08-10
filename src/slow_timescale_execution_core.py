"""提供算法无关的慢窗口轨迹、快层执行、指标和奖励闭环。"""

from collections.abc import Callable
from dataclasses import dataclass, field
import math

import numpy as np

from src.entities import ServerlessFunction, SFCType, SlotConstraintAudit, TrainState
from src.failure_process import FailureProcess, InfrastructureState
from src.failure_risk_prediction import FailureRiskProvider
from src.fast_slot_executor import FastSlotExecutionResult, FastSlotExecutor, FastSlotInput
from src.mobility import TrainMobilityModel
from src.network import TransferNetworkProtocol
from src.rl_reward import (
    RLCostRewardBreakdown,
    RLWindowCostMetrics,
    RLWindowMetrics,
    calculate_cost_reward,
)
from src.sfc_deployment_intent import SFCDeploymentIntent
from src.topology import LinearRailTopology
from src.workload import DeterministicWorkload
from src.workload_prediction import HistoricalWorkloadPredictor


@dataclass(frozen=True)
class EpisodeTraceSlot:
    """保存一个预生成快时隙的真实外生条件。"""

    train_state: TrainState
    request_count: int
    infrastructure_state: InfrastructureState
    predicted_failure_risk: float


@dataclass(frozen=True)
class SlowNodeObservation:
    """保存一个计算节点在当前决策点的归一化可观测状态。"""

    node_id: int
    free_cpu_ratio: float
    free_memory_ratio: float
    base_availability: float
    predicted_failure_probability: float
    operational: bool
    normalized_delay_from_serving: float


@dataclass(frozen=True)
class SlowFunctionObservation:
    """保存一个 VNF 的配置化归一化资源特征。"""

    function_id: int
    normalized_cpu: float
    normalized_memory: float
    normalized_execution_time: float
    normalized_cold_start_time: float


@dataclass(frozen=True)
class SlowSFCObservation:
    """保存整条 SFC 的归一化约束特征。"""

    reliability_target: float
    normalized_deadline: float
    normalized_input_size: float


@dataclass(frozen=True)
class SlowTimescaleObservation:
    """保存不依赖具体学习算法的当前慢决策观测。"""

    train_state: TrainState
    route_progress: float
    normalized_remaining_dwell: float
    normalized_mean_requests: float
    normalized_peak_requests: float
    normalized_load_trend: float
    global_failure_risk: float
    node_observations: tuple[SlowNodeObservation, ...]
    function_observations: tuple[SlowFunctionObservation, ...]
    sfc_observation: SlowSFCObservation


@dataclass(frozen=True)
class ProjectionInputs:
    """保存投影器在当前时隙允许读取的基础设施快照。"""

    operational_node_ids: frozenset[int]
    free_cpu: dict[int, float]
    free_memory_mb: dict[int, float]
    fault_domains: dict[int, int]


@dataclass(frozen=True)
class SlowWindowExecution:
    """保存一个已执行或被拒绝慢窗口的全部最终结果。"""

    metrics: RLWindowMetrics
    reward_breakdown: RLCostRewardBreakdown
    window_start_slot: int
    window_end_slot: int
    window_length: int
    boundary_reason: str
    terminated: bool
    rejected: bool
    rejection_reasons: tuple[str, ...]
    final_candidate_map: dict[int, tuple[int, ...]] | None
    last_fast_result: FastSlotExecutionResult | None


@dataclass
class _WindowAccumulator:
    """逐快时隙累加共享执行器的原始结果，不在此处重新推断部署。"""

    total_requests: int = 0
    successful_requests: int = 0
    request_batches: int = 0
    successful_batches: int = 0
    failed_batches: int = 0
    deadline_violations: int = 0
    successful_delays_ms: list[float] = field(default_factory=list)
    total_cold_start_delay_ms: float = 0.0
    total_active_memory_mb_seconds: float = 0.0
    failover_function_stages: int = 0
    cold_start_function_stages: int = 0
    reconfigured_function_stages: int = 0
    total_run_cost: float = 0.0
    total_route_cost: float = 0.0
    total_cold_start_cost: float = 0.0
    fast_repair_attempts: int = 0
    fast_repair_successes: int = 0
    fast_repair_failures: int = 0
    constraint_rejected_batches: int = 0
    cloud_used_slots: int = 0
    exact_reliabilities: list[float] = field(default_factory=list)

    def add(
        self,
        result: FastSlotExecutionResult,
        *,
        request_count: int,
        slot_seconds: float,
    ) -> None:
        """把一个快时隙结果合并到当前慢窗口。"""

        self.total_requests += request_count
        if request_count > 0:
            self.request_batches += 1
            if result.request_success is True:
                self.successful_batches += 1
                self.successful_requests += request_count
                if result.deadline_met is False:
                    self.deadline_violations += 1
            else:
                self.failed_batches += 1
            if result.constraint_rejected:
                self.constraint_rejected_batches += 1
        if result.request_success is True and result.end_to_end_delay_ms is not None:
            self.successful_delays_ms.append(result.end_to_end_delay_ms)
        self.total_cold_start_delay_ms += result.cold_start_delay_ms
        self.total_active_memory_mb_seconds += result.active_memory_mb * slot_seconds
        self.failover_function_stages += len(result.failover_function_ids)
        self.cold_start_function_stages += len(result.cold_start_function_ids)
        self.reconfigured_function_stages += result.plan_change_count
        self.total_run_cost += result.run_cost
        self.total_route_cost += result.route_cost
        self.total_cold_start_cost += result.cold_start_cost
        if result.fast_repair_attempted:
            self.fast_repair_attempts += 1
            if result.fast_repair_succeeded is True:
                self.fast_repair_successes += 1
            elif result.fast_repair_succeeded is False:
                self.fast_repair_failures += 1
        if result.used_cloud:
            self.cloud_used_slots += 1
        reliability = result.final_audit.exact_sfc_reliability
        if reliability is not None:
            self.exact_reliabilities.append(float(reliability))

    def build_metrics(self, *, window_length: int, slot_seconds: float) -> RLWindowMetrics:
        """把累加值转换为奖励和论文评估共用的窗口指标。"""

        if window_length <= 0:
            raise ValueError("慢窗口长度必须大于 0。")
        sla_violations = self.failed_batches + self.deadline_violations
        duration_seconds = window_length * slot_seconds
        return RLWindowMetrics(
            total_requests=self.total_requests,
            successful_requests=self.successful_requests,
            request_batches=self.request_batches,
            successful_batches=self.successful_batches,
            failed_batches=self.failed_batches,
            deadline_violations=self.deadline_violations,
            sla_violations=sla_violations,
            request_success_rate=(
                self.successful_requests / self.total_requests
                if self.total_requests > 0
                else 1.0
            ),
            sla_violation_rate=(
                sla_violations / self.request_batches
                if self.request_batches > 0
                else 0.0
            ),
            average_successful_delay_ms=(
                float(np.mean(self.successful_delays_ms))
                if self.successful_delays_ms
                else 0.0
            ),
            total_cold_start_delay_ms=self.total_cold_start_delay_ms,
            average_active_memory_mb=(
                self.total_active_memory_mb_seconds / duration_seconds
            ),
            total_active_memory_mb_seconds=self.total_active_memory_mb_seconds,
            failover_function_stages=self.failover_function_stages,
            cold_start_function_stages=self.cold_start_function_stages,
            reconfigured_function_stages=self.reconfigured_function_stages,
            total_run_cost=self.total_run_cost,
            total_route_cost=self.total_route_cost,
            total_cold_start_cost=self.total_cold_start_cost,
            fast_repair_attempts=self.fast_repair_attempts,
            fast_repair_successes=self.fast_repair_successes,
            fast_repair_failures=self.fast_repair_failures,
            constraint_rejected_batches=self.constraint_rejected_batches,
            cloud_used_slots=self.cloud_used_slots,
            cloud_usage_rate=self.cloud_used_slots / window_length,
            minimum_exact_sfc_reliability=(
                min(self.exact_reliabilities) if self.exact_reliabilities else 0.0
            ),
            mean_exact_sfc_reliability=(
                float(np.mean(self.exact_reliabilities))
                if self.exact_reliabilities
                else 0.0
            ),
        )


class SlowTimescaleExecutionCore:
    """管理因果外生轨迹，并把通用部署意图逐时隙交给共享快层。"""

    def __init__(
        self,
        *,
        topology: LinearRailTopology,
        network: TransferNetworkProtocol,
        mobility_model: TrainMobilityModel,
        workload: DeterministicWorkload,
        functions: list[ServerlessFunction],
        sfc: SFCType,
        failure_process_builder: Callable[[int], FailureProcess],
        failure_risk_provider: FailureRiskProvider,
        slow_period_slots: int,
        slot_seconds: float,
        workload_predictor: HistoricalWorkloadPredictor,
        fast_slot_executor: FastSlotExecutor,
        maximum_request_rate: float,
        maximum_network_delay_ms: float,
        maximum_input_size_mb: float,
        input_size_mb_per_request: float,
        maximum_window_cost: float,
        default_seed: int,
    ) -> None:
        """保存共享依赖，并提前校验所有归一化分母。"""

        if not functions or tuple(f.function_id for f in functions) != tuple(
            sfc.function_ids
        ):
            raise ValueError("函数必须非空并严格遵循 SFC 顺序。")
        if topology.cloud_node is None:
            raise ValueError("慢尺度执行核心要求配置中心云。")
        if slow_period_slots <= 0:
            raise ValueError("慢时间尺度周期必须大于 0。")
        positive_values = (
            slot_seconds,
            maximum_request_rate,
            maximum_network_delay_ms,
            maximum_input_size_mb,
            maximum_window_cost,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive_values):
            raise ValueError("时隙和归一化上限必须是正有限值。")
        if not isinstance(default_seed, int):
            raise TypeError("默认随机种子必须是整数。")

        self.topology = topology
        self.network = network
        self.mobility_model = mobility_model
        self.workload = workload
        self.functions = list(functions)
        self.sfc = sfc
        self.failure_process_builder = failure_process_builder
        self.failure_risk_provider = failure_risk_provider
        self.slow_period_slots = slow_period_slots
        self.slot_seconds = slot_seconds
        self.workload_predictor = workload_predictor
        self.fast_slot_executor = fast_slot_executor
        self.maximum_request_rate = maximum_request_rate
        self.maximum_network_delay_ms = maximum_network_delay_ms
        self.maximum_input_size_mb = maximum_input_size_mb
        self.input_size_mb_per_request = input_size_mb_per_request
        self.maximum_window_cost = maximum_window_cost
        self.default_seed = default_seed

        self._trace: tuple[EpisodeTraceSlot, ...] = ()
        self._cursor = 0
        self._terminated = False
        self._observed_request_counts: list[int] = []
        self._maximum_dwell_time_s = 1.0
        self._previous_candidate_map: dict[int, tuple[int, ...]] | None = None
        self._last_final_audit: SlotConstraintAudit | None = None
        self._last_execution_result: FastSlotExecutionResult | None = None

    @property
    def total_fast_slots(self) -> int:
        """返回预生成轨迹长度，但不公开未来请求值。"""

        return len(self._trace)

    @property
    def observed_request_counts(self) -> tuple[int, ...]:
        """只返回已经执行结束的请求前缀。"""

        return tuple(self._observed_request_counts)

    @property
    def terminated(self) -> bool:
        """返回当前 Episode 是否已经结束。"""

        return self._terminated

    @property
    def current_slot(self) -> int:
        """返回当前慢决策起始时隙。"""

        self._require_active_episode()
        return self._trace[self._cursor].train_state.time_slot

    @property
    def next_window_end_slot(self) -> int:
        """返回当前慢窗口右开边界，直接用作意图有效期。"""

        self._require_active_episode()
        end_index, _ = self._next_decision_boundary(self._cursor)
        return self._trace[end_index - 1].train_state.time_slot + 1

    def _require_active_episode(self) -> None:
        """统一拒绝尚未 reset 或已经结束后的访问。"""

        if not self._trace:
            raise RuntimeError("请先调用 reset() 生成 Episode。")
        if self._terminated:
            raise RuntimeError("Episode 已经结束，请先调用 reset()。")

    def _build_episode_trace(self, seed: int) -> tuple[EpisodeTraceSlot, ...]:
        """一次性预生成请求、移动与故障，保证不同动作面对相同外生条件。"""

        failure_process = self.failure_process_builder(seed)
        failure_process.reset()
        train_state = self.mobility_model.reset()
        trace: list[EpisodeTraceSlot] = []
        while True:
            trace.append(
                EpisodeTraceSlot(
                    train_state=train_state,
                    request_count=self.workload.request_count(train_state.time_slot),
                    infrastructure_state=failure_process.state_for_slot(
                        train_state.time_slot
                    ),
                    predicted_failure_risk=self.failure_risk_provider.predict(
                        time_slot=train_state.time_slot,
                        train_state=train_state,
                    ),
                )
            )
            if self.mobility_model.finished:
                break
            train_state = self.mobility_model.step()
        return tuple(trace)

    def reset(self, seed: int | None = None) -> SlowTimescaleObservation:
        """重置 Episode、快层保留状态和全部已执行历史。"""

        selected_seed = self.default_seed if seed is None else seed
        if not isinstance(selected_seed, int):
            raise TypeError("随机种子必须是整数。")
        self._trace = self._build_episode_trace(selected_seed)
        if not self._trace:
            raise RuntimeError("Episode 轨迹不能为空。")
        self.fast_slot_executor.reset()
        self._cursor = 0
        self._terminated = False
        self._observed_request_counts = []
        self._previous_candidate_map = None
        self._last_final_audit = None
        self._last_execution_result = None
        self._maximum_dwell_time_s = max(
            1.0,
            max(slot.train_state.remaining_dwell_time_s for slot in self._trace),
        )
        return self.current_observation()

    def _current_forecast(self):
        """只把已执行请求前缀交给预测器，禁止读取内部未来轨迹。"""

        return self.workload_predictor.predict(self._observed_request_counts)

    def _free_resources(self, node_id: int) -> tuple[float, float]:
        """根据上个快时隙最终审计计算当前空闲 CPU 和内存。"""

        node = self.topology.get_node(node_id)
        if self._last_final_audit is None:
            return node.cpu_capacity, node.memory_capacity_mb
        return (
            max(
                0.0,
                node.cpu_capacity
                - self._last_final_audit.node_cpu_demand.get(node_id, 0.0),
            ),
            max(
                0.0,
                node.memory_capacity_mb
                - self._last_final_audit.node_memory_demand_mb.get(node_id, 0.0),
            ),
        )

    def current_projection_inputs(self) -> ProjectionInputs:
        """构造当前动作投影唯一允许读取的节点快照。"""

        self._require_active_episode()
        slot = self._trace[self._cursor]
        free_resources = {
            node.node_id: self._free_resources(node.node_id)
            for node in self.topology.compute_nodes
        }
        return ProjectionInputs(
            operational_node_ids=frozenset(
                node.node_id
                for node in self.topology.compute_nodes
                if slot.infrastructure_state.is_node_operational(
                    node.node_id,
                    self.topology,
                )
            ),
            free_cpu={node_id: values[0] for node_id, values in free_resources.items()},
            free_memory_mb={
                node_id: values[1] for node_id, values in free_resources.items()
            },
            fault_domains={
                node.node_id: node.fault_domain for node in self.topology.compute_nodes
            },
        )

    def current_observation(self) -> SlowTimescaleObservation:
        """构造当前因果观测，不包含任何未来请求或未来执行结果。"""

        self._require_active_episode()
        slot = self._trace[self._cursor]
        forecast = self._current_forecast()
        route_denominator = max(
            self.topology.route_end_m - self.topology.route_start_m,
            1.0,
        )
        max_cpu = max(function.cpu_cycles_per_request for function in self.functions)
        max_memory = max(function.memory_mb for function in self.functions)
        max_execution = max(function.warm_exec_time_ms for function in self.functions)
        max_cold_start = max(
            function.cold_start_time_ms for function in self.functions
        )
        serving_node_id = slot.train_state.serving_mec
        node_observations: list[SlowNodeObservation] = []
        for node in self.topology.compute_nodes:
            free_cpu, free_memory = self._free_resources(node.node_id)
            node_observations.append(
                SlowNodeObservation(
                    node_id=node.node_id,
                    free_cpu_ratio=free_cpu / node.cpu_capacity,
                    free_memory_ratio=free_memory / node.memory_capacity_mb,
                    base_availability=node.reliability,
                    predicted_failure_probability=min(
                        1.0,
                        max(
                            0.0,
                            1.0
                            - node.reliability
                            * (1.0 - slot.predicted_failure_risk),
                        ),
                    ),
                    operational=slot.infrastructure_state.is_node_operational(
                        node.node_id,
                        self.topology,
                    ),
                    normalized_delay_from_serving=min(
                        1.0,
                        self.network.transfer_delay_ms(
                            1.0,
                            serving_node_id,
                            node.node_id,
                        )
                        / self.maximum_network_delay_ms,
                    ),
                )
            )
        return SlowTimescaleObservation(
            train_state=slot.train_state,
            route_progress=min(
                1.0,
                max(
                    0.0,
                    (slot.train_state.position_m - self.topology.route_start_m)
                    / route_denominator,
                ),
            ),
            normalized_remaining_dwell=min(
                slot.train_state.remaining_dwell_time_s / self._maximum_dwell_time_s,
                1.0,
            ),
            normalized_mean_requests=min(
                forecast.mean_requests / self.maximum_request_rate,
                1.0,
            ),
            normalized_peak_requests=min(
                forecast.peak_requests / self.maximum_request_rate,
                1.0,
            ),
            normalized_load_trend=forecast.normalized_trend,
            global_failure_risk=slot.predicted_failure_risk,
            node_observations=tuple(node_observations),
            function_observations=tuple(
                SlowFunctionObservation(
                    function_id=function.function_id,
                    normalized_cpu=function.cpu_cycles_per_request / max_cpu,
                    normalized_memory=function.memory_mb / max_memory,
                    normalized_execution_time=function.warm_exec_time_ms / max_execution,
                    normalized_cold_start_time=(
                        function.cold_start_time_ms / max_cold_start
                    ),
                )
                for function in self.functions
            ),
            sfc_observation=SlowSFCObservation(
                reliability_target=self.sfc.reliability_target,
                normalized_deadline=1.0,
                normalized_input_size=min(
                    self.input_size_mb_per_request / self.maximum_input_size_mb,
                    1.0,
                ),
            ),
        )

    def current_observation_info(self) -> dict[str, object]:
        """返回适合日志和 review 的当前决策点原始信息。"""

        if self._terminated:
            return {"terminated": True, "decision_slot": None}
        observation = self.current_observation()
        forecast = self._current_forecast()
        return {
            "terminated": False,
            "decision_slot": observation.train_state.time_slot,
            "serving_mec": observation.train_state.serving_mec,
            "next_mec": observation.train_state.next_mec,
            "remaining_dwell_time_s": (
                observation.train_state.remaining_dwell_time_s
            ),
            "predicted_request_rate": forecast.mean_requests,
            "predicted_peak_requests": forecast.peak_requests,
            "predicted_load_trend": forecast.normalized_trend,
            "predicted_failure_risk": observation.global_failure_risk,
        }

    def current_replica_ratios(self) -> tuple[float, float]:
        """返回上一最终部署的热副本比例和中心云副本比例。"""

        result = self._last_execution_result
        if result is None:
            return 0.0, 0.0
        replica_pairs = {
            (function_id, node_id)
            for function_id, node_ids in result.function_replica_node_ids.items()
            for node_id in node_ids
        }
        if not replica_pairs:
            return 0.0, 0.0
        hot_pairs = {
            (function_id, node_id)
            for function_id, node_ids in result.function_hot_node_ids.items()
            for node_id in node_ids
        }
        cloud_node_id = self.topology.cloud_node.node_id  # type: ignore[union-attr]
        return (
            len(hot_pairs & replica_pairs) / len(replica_pairs),
            sum(node_id == cloud_node_id for _, node_id in replica_pairs)
            / len(replica_pairs),
        )

    def _next_decision_boundary(self, start_index: int) -> tuple[int, str]:
        """返回当前慢窗口尾部索引（不包含）和边界原因。"""

        natural_end = min(start_index + self.slow_period_slots, len(self._trace))
        serving_mec = self._trace[start_index].train_state.serving_mec
        for index in range(start_index + 1, natural_end):
            if self._trace[index].train_state.serving_mec != serving_mec:
                return index, "handover"
        if natural_end >= len(self._trace):
            return natural_end, "episode_end"
        return natural_end, "period_end"

    def _reward_for_metrics(self, metrics: RLWindowMetrics) -> RLCostRewardBreakdown:
        """使用三项原始成本和一个二值硬违约项计算奖励。"""

        return calculate_cost_reward(
            RLWindowCostMetrics(
                run_cost=metrics.total_run_cost,
                route_cost=metrics.total_route_cost,
                cold_start_cost=metrics.total_cold_start_cost,
                has_violation=(
                    metrics.sla_violations > 0
                    or metrics.constraint_rejected_batches > 0
                ),
            ),
            maximum_window_cost=self.maximum_window_cost,
        )

    def _advance_cursor(self, end_index: int) -> None:
        """推进到下一慢决策点并更新终止标志。"""

        self._cursor = end_index
        self._terminated = self._cursor >= len(self._trace)

    def execute_window(self, intent: SFCDeploymentIntent) -> SlowWindowExecution:
        """在当前慢窗口逐快时隙执行一个成功投影后的通用意图。"""

        self._require_active_episode()
        if not isinstance(intent, SFCDeploymentIntent):
            raise TypeError("intent 必须是 SFCDeploymentIntent。")
        start_index = self._cursor
        end_index, boundary_reason = self._next_decision_boundary(start_index)
        if intent.decision_slot != self.current_slot:
            raise ValueError("意图决策时隙必须等于当前慢窗口起点。")
        if intent.valid_until_slot != self.next_window_end_slot:
            raise ValueError("意图有效期必须等于当前慢窗口右开边界。")

        accumulator = _WindowAccumulator()
        previous_candidate_map = self._previous_candidate_map
        last_result: FastSlotExecutionResult | None = None
        for index in range(start_index, end_index):
            trace_slot = self._trace[index]
            last_result = self.fast_slot_executor.execute(
                FastSlotInput(
                    train_state=trace_slot.train_state,
                    request_count=trace_slot.request_count,
                    infrastructure_state=trace_slot.infrastructure_state,
                    slow_decision=None,
                    previous_candidate_map=previous_candidate_map,
                    deployment_intent=intent,
                )
            )
            accumulator.add(
                last_result,
                request_count=trace_slot.request_count,
                slot_seconds=self.slot_seconds,
            )
            previous_candidate_map = dict(last_result.function_replica_node_ids)
            self._previous_candidate_map = previous_candidate_map
            self._last_final_audit = last_result.final_audit
            self._last_execution_result = last_result
            self._observed_request_counts.append(trace_slot.request_count)

        metrics = accumulator.build_metrics(
            window_length=end_index - start_index,
            slot_seconds=self.slot_seconds,
        )
        reward_breakdown = self._reward_for_metrics(metrics)
        start_slot = self._trace[start_index].train_state.time_slot
        end_slot = self._trace[end_index - 1].train_state.time_slot
        self._advance_cursor(end_index)
        return SlowWindowExecution(
            metrics=metrics,
            reward_breakdown=reward_breakdown,
            window_start_slot=start_slot,
            window_end_slot=end_slot,
            window_length=end_index - start_index,
            boundary_reason=boundary_reason,
            terminated=self._terminated,
            rejected=False,
            rejection_reasons=(),
            final_candidate_map=(
                None if last_result is None else dict(last_result.function_replica_node_ids)
            ),
            last_fast_result=last_result,
        )

    def advance_rejected_window(
        self,
        reasons: tuple[str, ...],
    ) -> SlowWindowExecution:
        """不调用快层地推进投影失败窗口，并固定记录一次硬违约。"""

        self._require_active_episode()
        if not reasons:
            raise ValueError("拒绝窗口必须提供至少一个投影失败原因。")
        start_index = self._cursor
        end_index, boundary_reason = self._next_decision_boundary(start_index)
        request_counts = [
            self._trace[index].request_count for index in range(start_index, end_index)
        ]
        self._observed_request_counts.extend(request_counts)
        total_requests = sum(request_counts)
        request_batches = sum(count > 0 for count in request_counts)
        metrics = RLWindowMetrics(
            total_requests=total_requests,
            successful_requests=0,
            request_batches=request_batches,
            successful_batches=0,
            failed_batches=request_batches,
            deadline_violations=0,
            sla_violations=max(1, request_batches),
            request_success_rate=0.0 if total_requests > 0 else 1.0,
            sla_violation_rate=1.0,
            average_successful_delay_ms=0.0,
            total_cold_start_delay_ms=0.0,
            average_active_memory_mb=0.0,
            total_active_memory_mb_seconds=0.0,
            failover_function_stages=0,
            cold_start_function_stages=0,
            reconfigured_function_stages=0,
            constraint_rejected_batches=max(1, request_batches),
        )
        reward_breakdown = self._reward_for_metrics(metrics)
        start_slot = self._trace[start_index].train_state.time_slot
        end_slot = self._trace[end_index - 1].train_state.time_slot
        self._advance_cursor(end_index)
        return SlowWindowExecution(
            metrics=metrics,
            reward_breakdown=reward_breakdown,
            window_start_slot=start_slot,
            window_end_slot=end_slot,
            window_length=end_index - start_index,
            boundary_reason=boundary_reason,
            terminated=self._terminated,
            rejected=True,
            rejection_reasons=tuple(reasons),
            final_candidate_map=None,
            last_fast_result=None,
        )
