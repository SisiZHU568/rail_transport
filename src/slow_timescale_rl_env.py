"""双时间尺度 Serverless SFC 的 DDQN 慢层环境。

一个 ``step`` 产生一项结构化慢层决策。该决策持续到慢周期结束或列车
发生 MEC 切换；窗口内每个快时隙统一交给 :class:`FastSlotExecutor` 执行。
"""

from collections.abc import Callable
from dataclasses import dataclass, field
import math

import numpy as np

from src.entities import (
    ServerlessFunction,
    SFCType,
    SlotConstraintAudit,
    TrainState,
)
from src.failure_process import FailureProcess, InfrastructureState
from src.failure_risk_prediction import FailureRiskProvider
from src.fast_slot_executor import (
    FastSlotExecutionResult,
    FastSlotExecutor,
    FastSlotInput,
)
from src.mobility import TrainMobilityModel
from src.network import TransferNetworkProtocol
from src.rl_agent_action_space import (
    ActionFeasibilityContext,
    DDQN_ACTION_NAMES,
    StructuredSlowAction,
    build_valid_action_mask,
    decode_ddqn_action,
    get_ddqn_action_count,
)
from src.rl_reward import (
    RLCostRewardBreakdown,
    RLWindowCostMetrics,
    RLWindowMetrics,
    calculate_cost_reward,
)
from src.rl_state_encoder import (
    NodeStateObservation,
    RLStateEncoder,
    RLStateSnapshot,
)
from src.topology import LinearRailTopology
from src.two_timescale_control import SlowTimescaleDecision
from src.workload import DeterministicWorkload
from src.workload_prediction import HistoricalWorkloadPredictor, WorkloadForecast


@dataclass(frozen=True)
class EpisodeTraceSlot:
    """保存一个预生成快时隙的真实外生条件。"""

    train_state: TrainState
    request_count: int
    infrastructure_state: InfrastructureState
    predicted_failure_risk: float


@dataclass
class _WindowAccumulator:
    """在环境内部逐时隙累加共享执行器的原始结果。"""

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
        request_count: int,
        slot_seconds: float,
    ) -> None:
        """把一个已执行快时隙合并到当前慢窗口。"""

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

        if (
            result.request_success is True
            and result.end_to_end_delay_ms is not None
        ):
            self.successful_delays_ms.append(result.end_to_end_delay_ms)

        self.total_cold_start_delay_ms += result.cold_start_delay_ms
        self.total_active_memory_mb_seconds += (
            result.active_memory_mb * slot_seconds
        )
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

    def build_metrics(
        self,
        window_length: int,
        slot_seconds: float,
    ) -> RLWindowMetrics:
        """把累加值转换成供奖励和评估统一读取的窗口指标。"""

        if window_length <= 0:
            raise ValueError("慢窗口长度必须大于 0。")
        sla_violations = self.failed_batches + self.deadline_violations
        request_success_rate = (
            self.successful_requests / self.total_requests
            if self.total_requests > 0
            else 1.0
        )
        sla_violation_rate = (
            sla_violations / self.request_batches
            if self.request_batches > 0
            else 0.0
        )
        average_delay = (
            float(np.mean(self.successful_delays_ms))
            if self.successful_delays_ms
            else 0.0
        )
        window_duration_seconds = window_length * slot_seconds
        minimum_reliability = (
            min(self.exact_reliabilities)
            if self.exact_reliabilities
            else 0.0
        )
        mean_reliability = (
            float(np.mean(self.exact_reliabilities))
            if self.exact_reliabilities
            else 0.0
        )

        return RLWindowMetrics(
            total_requests=self.total_requests,
            successful_requests=self.successful_requests,
            request_batches=self.request_batches,
            successful_batches=self.successful_batches,
            failed_batches=self.failed_batches,
            deadline_violations=self.deadline_violations,
            sla_violations=sla_violations,
            request_success_rate=request_success_rate,
            sla_violation_rate=sla_violation_rate,
            average_successful_delay_ms=average_delay,
            total_cold_start_delay_ms=self.total_cold_start_delay_ms,
            average_active_memory_mb=(
                self.total_active_memory_mb_seconds / window_duration_seconds
            ),
            total_active_memory_mb_seconds=(
                self.total_active_memory_mb_seconds
            ),
            failover_function_stages=self.failover_function_stages,
            cold_start_function_stages=self.cold_start_function_stages,
            reconfigured_function_stages=(
                self.reconfigured_function_stages
            ),
            total_run_cost=self.total_run_cost,
            total_route_cost=self.total_route_cost,
            total_cold_start_cost=self.total_cold_start_cost,
            fast_repair_attempts=self.fast_repair_attempts,
            fast_repair_successes=self.fast_repair_successes,
            fast_repair_failures=self.fast_repair_failures,
            constraint_rejected_batches=self.constraint_rejected_batches,
            cloud_used_slots=self.cloud_used_slots,
            cloud_usage_rate=self.cloud_used_slots / window_length,
            minimum_exact_sfc_reliability=minimum_reliability,
            mean_exact_sfc_reliability=mean_reliability,
        )


class SlowTimescaleRLEnvironment:
    """把结构化慢决策和共享快时隙闭环封装为强化学习环境。"""

    def __init__(
        self,
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
        state_encoder: RLStateEncoder,
        workload_predictor: HistoricalWorkloadPredictor,
        fast_slot_executor: FastSlotExecutor,
        maximum_request_rate: float,
        maximum_network_delay_ms: float,
        maximum_window_cost: float,
        minimum_distinct_fault_domains: int,
        default_seed: int,
    ) -> None:
        """保存环境依赖；副本规划和精确修复只由共享执行器负责。"""

        if not functions:
            raise ValueError("至少需要配置一个 Serverless 函数。")
        if set(function.function_id for function in functions) != set(
            sfc.function_ids
        ):
            raise ValueError("函数列表必须与 SFC 函数集合完全一致。")
        if topology.cloud_node is None:
            raise ValueError("当前 78 维状态模式要求配置中心云节点。")
        if slow_period_slots <= 0:
            raise ValueError("慢时间尺度周期必须大于 0。")
        positive_values = (
            slot_seconds,
            maximum_request_rate,
            maximum_network_delay_ms,
            maximum_window_cost,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive_values):
            raise ValueError("时隙和归一化上限必须是正有限值。")
        if minimum_distinct_fault_domains <= 0:
            raise ValueError("最小故障域数量必须大于 0。")
        if not isinstance(default_seed, int):
            raise TypeError("默认随机种子必须是整数。")
        if state_encoder.state_dim != 78:
            raise ValueError("当前慢层环境必须使用 78 维状态编码器。")

        self.topology = topology
        self.network = network
        self.mobility_model = mobility_model
        self.workload = workload
        self.functions = list(functions)
        self.function_map = {
            function.function_id: function for function in self.functions
        }
        self.sfc = sfc
        self.failure_process_builder = failure_process_builder
        self.failure_risk_provider = failure_risk_provider
        self.slow_period_slots = slow_period_slots
        self.slot_seconds = slot_seconds
        self.state_encoder = state_encoder
        self.workload_predictor = workload_predictor
        self.fast_slot_executor = fast_slot_executor
        self.maximum_request_rate = maximum_request_rate
        self.maximum_network_delay_ms = maximum_network_delay_ms
        self.maximum_window_cost = maximum_window_cost
        self.minimum_distinct_fault_domains = minimum_distinct_fault_domains
        self.default_seed = default_seed
        self.state_dim = state_encoder.state_dim
        self.action_count = get_ddqn_action_count()

        self._trace: tuple[EpisodeTraceSlot, ...] = ()
        self._cursor = 0
        self._terminated = False
        self._episode_seed = default_seed
        self._maximum_dwell_time_s = 1.0
        self._observed_request_counts: list[int] = []
        self._previous_action: StructuredSlowAction | None = None
        self._previous_metrics: RLWindowMetrics | None = None
        self._previous_reward_breakdown: RLCostRewardBreakdown | None = None
        self._previous_candidate_map: dict[int, tuple[int, ...]] | None = None
        self._last_final_audit: SlotConstraintAudit | None = None
        self._last_execution_result: FastSlotExecutionResult | None = None

    @property
    def observed_request_counts(self) -> tuple[int, ...]:
        """返回预测器当前允许看到的已执行请求前缀。"""

        return tuple(self._observed_request_counts)

    def _build_episode_trace(self, seed: int) -> tuple[EpisodeTraceSlot, ...]:
        """预生成外生轨迹，使不同动作面对完全相同的请求和故障。"""

        failure_process = self.failure_process_builder(seed)
        failure_process.reset()
        train_state = self.mobility_model.reset()
        trace: list[EpisodeTraceSlot] = []
        while True:
            request_count = self.workload.request_count(train_state.time_slot)
            infrastructure_state = failure_process.state_for_slot(
                train_state.time_slot
            )
            predicted_failure_risk = self.failure_risk_provider.predict(
                time_slot=train_state.time_slot,
                train_state=train_state,
            )
            trace.append(
                EpisodeTraceSlot(
                    train_state=train_state,
                    request_count=request_count,
                    infrastructure_state=infrastructure_state,
                    predicted_failure_risk=predicted_failure_risk,
                )
            )
            if self.mobility_model.finished:
                break
            train_state = self.mobility_model.step()
        return tuple(trace)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        """重置 Episode，并返回 78 维状态和当前 12 维动作掩码。"""

        selected_seed = self.default_seed if seed is None else seed
        if not isinstance(selected_seed, int):
            raise TypeError("随机种子必须是整数。")
        self._episode_seed = selected_seed
        self._trace = self._build_episode_trace(selected_seed)
        if not self._trace:
            raise RuntimeError("Episode 轨迹不能为空。")

        self._cursor = 0
        self._terminated = False
        self._observed_request_counts = []
        self._previous_action = None
        self._previous_metrics = None
        self._previous_reward_breakdown = None
        self._previous_candidate_map = None
        self._last_final_audit = None
        self._last_execution_result = None
        self._maximum_dwell_time_s = max(
            1.0,
            max(
                slot.train_state.remaining_dwell_time_s for slot in self._trace
            ),
        )

        state = self._build_state()
        info = self._build_observation_info()
        info.update(
            {
                "episode_seed": self._episode_seed,
                "total_fast_slots": len(self._trace),
                "state_feature_names": self.state_encoder.feature_names,
            }
        )
        return state, info

    def _current_forecast(self) -> WorkloadForecast:
        """只把已经执行完成的请求前缀交给历史预测器。"""

        # 禁止把 self._trace 或当前游标之后的请求传入预测器。
        return self.workload_predictor.predict(self._observed_request_counts)

    def _free_resources(self, node_id: int) -> tuple[float, float]:
        """根据上一快时隙的最终审计计算当前空闲 CPU 和内存。"""

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
                - self._last_final_audit.node_memory_demand_mb.get(
                    node_id, 0.0
                ),
            ),
        )

    def _build_node_observations(
        self,
        slot: EpisodeTraceSlot,
    ) -> tuple[NodeStateObservation, ...]:
        """按五个 MEC 加中心云的固定顺序生成六项节点观测。"""

        serving_node_id = slot.train_state.serving_mec
        global_failure_risk = slot.predicted_failure_risk
        observations: list[NodeStateObservation] = []
        for node in self.topology.compute_nodes:
            free_cpu, free_memory = self._free_resources(node.node_id)
            node_failure_probability = min(
                1.0,
                max(
                    0.0,
                    1.0
                    - node.reliability * (1.0 - global_failure_risk),
                ),
            )
            normalized_node_delay = min(
                1.0,
                self.network.transfer_delay_ms(
                    1.0,
                    serving_node_id,
                    node.node_id,
                )
                / self.maximum_network_delay_ms,
            )
            observations.append(
                NodeStateObservation(
                    node_id=node.node_id,
                    free_cpu_ratio=free_cpu / node.cpu_capacity,
                    free_memory_ratio=(
                        free_memory / node.memory_capacity_mb
                    ),
                    base_availability=node.reliability,
                    predicted_failure_probability=(
                        node_failure_probability
                    ),
                    operational=slot.infrastructure_state.is_node_operational(
                        node.node_id,
                        self.topology,
                    ),
                    normalized_delay_from_serving=normalized_node_delay,
                )
            )
        return tuple(observations)

    def _last_replica_ratios(self) -> tuple[float, float]:
        """返回上一最终方案的热副本比例和云副本比例。"""

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
        cloud_pairs = {
            pair for pair in replica_pairs if pair[1] == cloud_node_id
        }
        return (
            len(hot_pairs & replica_pairs) / len(replica_pairs),
            len(cloud_pairs) / len(replica_pairs),
        )

    def _build_state(self) -> np.ndarray:
        """构造当前决策点状态；终止状态不再参与动作选择。"""

        if self._terminated:
            return np.zeros(self.state_dim, dtype=np.float32)
        slot = self._trace[self._cursor]
        forecast = self._current_forecast()
        route_denominator = max(
            self.topology.route_end_m - self.topology.route_start_m,
            1.0,
        )
        route_progress = min(
            1.0,
            max(
                0.0,
                (slot.train_state.position_m - self.topology.route_start_m)
                / route_denominator,
            ),
        )
        hot_ratio, cloud_ratio = self._last_replica_ratios()
        has_history = (
            self._previous_action is not None
            and self._previous_metrics is not None
            and self._previous_reward_breakdown is not None
        )
        metrics = self._previous_metrics
        reward = self._previous_reward_breakdown
        repair_failure_rate = 0.0
        if has_history and metrics is not None and metrics.fast_repair_attempts > 0:
            repair_failure_rate = (
                metrics.fast_repair_failures / metrics.fast_repair_attempts
            )

        snapshot = RLStateSnapshot(
            serving_mec=slot.train_state.serving_mec,
            next_mec=slot.train_state.next_mec,
            route_progress=route_progress,
            normalized_remaining_dwell=min(
                slot.train_state.remaining_dwell_time_s
                / self._maximum_dwell_time_s,
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
            node_observations=self._build_node_observations(slot),
            current_action=self._previous_action,
            hot_replica_ratio=hot_ratio,
            cloud_replica_ratio=cloud_ratio,
            has_history=has_history,
            previous_success_rate=(
                metrics.request_success_rate if has_history and metrics else 0.0
            ),
            previous_sla_violation_rate=(
                metrics.sla_violation_rate if has_history and metrics else 0.0
            ),
            previous_normalized_total_cost=(
                reward.normalized_total_cost if has_history and reward else 0.0
            ),
            previous_repair_failure_rate=repair_failure_rate,
        )
        return self.state_encoder.encode(snapshot)

    def get_valid_action_mask(self) -> np.ndarray:
        """根据当前故障、资源和故障域条件生成 12 维必要条件掩码。"""

        if self._terminated:
            return np.zeros(self.action_count, dtype=np.bool_)
        infrastructure = self._trace[self._cursor].infrastructure_state
        operational_edge_node_ids = frozenset(
            site.node.node_id
            for site in self.topology.sites
            if infrastructure.is_node_operational(
                site.node.node_id,
                self.topology,
            )
        )
        cloud_node = self.topology.cloud_node
        operational_cloud_node_id = (
            cloud_node.node_id
            if cloud_node is not None
            and infrastructure.is_node_operational(
                cloud_node.node_id,
                self.topology,
            )
            else None
        )
        node_free_memory_mb = {
            node.node_id: self._free_resources(node.node_id)[1]
            for node in self.topology.compute_nodes
        }
        context = ActionFeasibilityContext(
            operational_edge_node_ids=operational_edge_node_ids,
            operational_cloud_node_id=operational_cloud_node_id,
            node_free_memory_mb=node_free_memory_mb,
            node_fault_domains={
                node.node_id: node.fault_domain
                for node in self.topology.compute_nodes
            },
            total_function_memory_mb=sum(
                self.function_map[function_id].memory_mb
                for function_id in self.sfc.function_ids
            ),
            minimum_distinct_fault_domains=(
                self.minimum_distinct_fault_domains
            ),
        )
        return build_valid_action_mask(context)

    def _build_observation_info(self) -> dict:
        """返回当前决策点的可解释原始信息和合法动作掩码。"""

        action_mask = self.get_valid_action_mask()
        if self._terminated:
            return {
                "terminated": True,
                "decision_slot": None,
                "serving_mec": None,
                "next_mec": None,
                "remaining_dwell_time_s": 0.0,
                "predicted_request_rate": 0.0,
                "predicted_peak_requests": 0.0,
                "predicted_load_trend": 0.0,
                "predicted_failure_risk": 0.0,
                "action_mask": action_mask,
            }
        slot = self._trace[self._cursor]
        forecast = self._current_forecast()
        return {
            "terminated": False,
            "decision_slot": slot.train_state.time_slot,
            "serving_mec": slot.train_state.serving_mec,
            "next_mec": slot.train_state.next_mec,
            "remaining_dwell_time_s": (
                slot.train_state.remaining_dwell_time_s
            ),
            "predicted_request_rate": forecast.mean_requests,
            "predicted_peak_requests": forecast.peak_requests,
            "predicted_load_trend": forecast.normalized_trend,
            "predicted_failure_risk": slot.predicted_failure_risk,
            "action_mask": action_mask,
        }

    def _next_decision_boundary(self, start_index: int) -> tuple[int, str]:
        """返回当前慢决策窗口的尾部索引（不包含）及结束原因。"""

        natural_end = min(
            start_index + self.slow_period_slots,
            len(self._trace),
        )
        current_serving_mec = self._trace[
            start_index
        ].train_state.serving_mec
        for index in range(start_index + 1, natural_end):
            if self._trace[index].train_state.serving_mec != current_serving_mec:
                return index, "handover"
        if natural_end >= len(self._trace):
            return natural_end, "episode_end"
        return natural_end, "period_end"

    def _build_slow_decision(
        self,
        action: StructuredSlowAction,
        start_index: int,
        end_index: int,
    ) -> SlowTimescaleDecision:
        """把 DDQN 组合动作转换成共享执行器理解的慢层决策。"""

        return SlowTimescaleDecision(
            decision_slot=self._trace[start_index].train_state.time_slot,
            valid_until_slot=self._trace[end_index - 1].train_state.time_slot,
            replica_count=action.replica_count,
            retention_policy=action.retention_policy,
            cloud_policy=action.cloud_policy,
            reason=f"DDQN 结构化动作 {action.action_id}",
        )

    def _reward_for_metrics(
        self,
        metrics: RLWindowMetrics,
    ) -> RLCostRewardBreakdown:
        """只使用运行、路由、冷启动三项成本和单一违约项。"""

        return calculate_cost_reward(
            RLWindowCostMetrics(
                run_cost=metrics.total_run_cost,
                route_cost=metrics.total_route_cost,
                cold_start_cost=metrics.total_cold_start_cost,
                has_violation=metrics.sla_violations > 0,
            ),
            maximum_window_cost=self.maximum_window_cost,
        )

    def _advance_cursor(self, end_index: int) -> None:
        """推进到下一个决策点，并更新 Episode 终止标志。"""

        self._cursor = end_index
        self._terminated = self._cursor >= len(self._trace)

    def _build_step_info(
        self,
        *,
        action: StructuredSlowAction | None,
        metrics: RLWindowMetrics,
        reward_breakdown: RLCostRewardBreakdown,
        start_index: int,
        end_index: int,
        boundary_reason: str,
    ) -> dict:
        """统一构造普通动作和无动作推进的窗口结果。"""

        info = {
            "episode_seed": self._episode_seed,
            "action": None if action is None else action.action_id,
            "action_name": (
                None if action is None else DDQN_ACTION_NAMES[action.action_id]
            ),
            "structured_action": action,
            "action_recorded": action is not None,
            "window_start_index": start_index,
            "window_end_index": end_index,
            "window_start_slot": self._trace[start_index].train_state.time_slot,
            "window_end_slot": self._trace[end_index - 1].train_state.time_slot,
            "window_length": end_index - start_index,
            "boundary_reason": boundary_reason,
            "handover_boundary_reached": boundary_reason == "handover",
            "metrics": metrics,
            "reward_breakdown": reward_breakdown,
            "next_observation": self._build_observation_info(),
        }
        if action is not None:
            info.update(
                {
                    "replica_count": action.replica_count,
                    "retention_policy": action.retention_policy.name,
                    "cloud_policy": action.cloud_policy.name,
                }
            )
        return info

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """执行一个合法结构化慢动作及其包含的全部快时隙。"""

        if self._terminated:
            raise RuntimeError("Episode 已经结束，请先调用 reset()。")
        structured_action = decode_ddqn_action(action)
        action_mask = self.get_valid_action_mask()
        if not bool(action_mask[structured_action.action_id]):
            raise ValueError(
                f"动作 {structured_action.action_id} 被当前合法动作掩码拒绝。"
            )

        start_index = self._cursor
        end_index, boundary_reason = self._next_decision_boundary(start_index)
        slow_decision = self._build_slow_decision(
            structured_action,
            start_index,
            end_index,
        )
        accumulator = _WindowAccumulator()
        previous_candidate_map = self._previous_candidate_map

        for index in range(start_index, end_index):
            trace_slot = self._trace[index]
            result = self.fast_slot_executor.execute(
                FastSlotInput(
                    train_state=trace_slot.train_state,
                    request_count=trace_slot.request_count,
                    infrastructure_state=trace_slot.infrastructure_state,
                    slow_decision=slow_decision,
                    previous_candidate_map=previous_candidate_map,
                )
            )
            accumulator.add(result, trace_slot.request_count, self.slot_seconds)
            previous_candidate_map = dict(result.function_replica_node_ids)
            self._previous_candidate_map = previous_candidate_map
            self._last_final_audit = result.final_audit
            self._last_execution_result = result

            # 请求只有在该快时隙已经执行完后才变成“已观测历史”。
            self._observed_request_counts.append(trace_slot.request_count)

        metrics = accumulator.build_metrics(
            end_index - start_index,
            self.slot_seconds,
        )
        reward_breakdown = self._reward_for_metrics(metrics)
        self._previous_action = structured_action
        self._previous_metrics = metrics
        self._previous_reward_breakdown = reward_breakdown
        self._advance_cursor(end_index)
        next_state = self._build_state()
        info = self._build_step_info(
            action=structured_action,
            metrics=metrics,
            reward_breakdown=reward_breakdown,
            start_index=start_index,
            end_index=end_index,
            boundary_reason=boundary_reason,
        )
        return (
            next_state,
            reward_breakdown.reward,
            self._terminated,
            False,
            info,
        )

    def advance_without_action(
        self,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """无合法动作时推进一个窗口并记录硬违约，但不伪造 DDQN 动作。"""

        if self._terminated:
            raise RuntimeError("Episode 已经结束，请先调用 reset()。")
        if self.get_valid_action_mask().any():
            raise ValueError("当前仍存在合法动作，不能执行无动作推进。")

        start_index = self._cursor
        end_index, boundary_reason = self._next_decision_boundary(start_index)
        request_counts = [
            self._trace[index].request_count
            for index in range(start_index, end_index)
        ]
        self._observed_request_counts.extend(request_counts)
        total_requests = sum(request_counts)
        request_batches = sum(count > 0 for count in request_counts)
        # 即使该窗口恰好没有请求，“系统无可行动作”本身仍是一项硬约束违约。
        sla_violations = max(1, request_batches)
        metrics = RLWindowMetrics(
            total_requests=total_requests,
            successful_requests=0,
            request_batches=request_batches,
            successful_batches=0,
            failed_batches=request_batches,
            deadline_violations=0,
            sla_violations=sla_violations,
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
        self._previous_metrics = metrics
        self._previous_reward_breakdown = reward_breakdown
        self._advance_cursor(end_index)
        next_state = self._build_state()
        info = self._build_step_info(
            action=None,
            metrics=metrics,
            reward_breakdown=reward_breakdown,
            start_index=start_index,
            end_index=end_index,
            boundary_reason=boundary_reason,
        )
        return (
            next_state,
            reward_breakdown.reward,
            self._terminated,
            False,
            info,
        )
