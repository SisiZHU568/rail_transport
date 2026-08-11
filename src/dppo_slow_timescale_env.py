"""定义 DPPO 连续联合动作使用的唯一慢时间尺度学习环境。"""

import numpy as np

from src.dppo_action_space import DPPOActionSpace, DecodedDPPOAction
from src.dppo_intent_adapter import DPPOIntentAdapter
from src.dppo_projection import DPPOProjector, ProjectionResult
from src.dppo_state_encoder import (
    DPPOFunctionObservation,
    DPPOHistoryObservation,
    DPPONodeObservation,
    DPPOSFCObservation,
    DPPOStateEncoder,
    DPPOStateSnapshot,
)
from src.scenario_dimensions import ScenarioDimensions
from src.slow_timescale_execution_core import (
    SlowTimescaleExecutionCore,
    SlowTimescaleObservation,
    SlowWindowExecution,
)


class DPPOSlowTimescaleEnvironment:
    """把连续动作解码、投影和共享快层执行封装为因果环境。"""

    def __init__(
        self,
        *,
        dimensions: ScenarioDimensions,
        action_space: DPPOActionSpace,
        projector: DPPOProjector,
        intent_adapter: DPPOIntentAdapter,
        state_encoder: DPPOStateEncoder,
        execution_core: SlowTimescaleExecutionCore,
    ) -> None:
        """保存唯一一组由场景工厂创建的配置化依赖。"""

        if action_space.dimensions != dimensions:
            raise ValueError("动作空间必须使用环境的 ScenarioDimensions。")
        if state_encoder.dimensions != dimensions:
            raise ValueError("状态编码器必须使用环境的 ScenarioDimensions。")
        self.dimensions = dimensions
        self.action_space = action_space
        self.projector = projector
        self.intent_adapter = intent_adapter
        self.state_encoder = state_encoder
        self.execution_core = execution_core
        self._history = DPPOHistoryObservation.zeros()

    def _to_snapshot(
        self,
        observation: SlowTimescaleObservation,
    ) -> DPPOStateSnapshot:
        """把算法无关观测适配为公开的 DPPO 状态模式。"""

        return DPPOStateSnapshot(
            serving_mec=observation.train_state.serving_mec,
            next_mec=observation.train_state.next_mec,
            route_progress=observation.route_progress,
            normalized_remaining_dwell=observation.normalized_remaining_dwell,
            normalized_mean_requests=observation.normalized_mean_requests,
            normalized_peak_requests=observation.normalized_peak_requests,
            normalized_load_trend=observation.normalized_load_trend,
            global_failure_risk=observation.global_failure_risk,
            node_observations=tuple(
                DPPONodeObservation(
                    node_id=node.node_id,
                    free_cpu_ratio=node.free_cpu_ratio,
                    free_memory_ratio=node.free_memory_ratio,
                    base_availability=node.base_availability,
                    predicted_failure_probability=(
                        node.predicted_failure_probability
                    ),
                    operational=node.operational,
                    normalized_delay_from_serving=(
                        node.normalized_delay_from_serving
                    ),
                )
                for node in observation.node_observations
            ),
            function_observations=tuple(
                DPPOFunctionObservation(
                    function_id=function.function_id,
                    normalized_cpu=function.normalized_cpu,
                    normalized_memory=function.normalized_memory,
                    normalized_execution_time=function.normalized_execution_time,
                    normalized_cold_start_time=(
                        function.normalized_cold_start_time
                    ),
                )
                for function in observation.function_observations
            ),
            sfc_observation=DPPOSFCObservation(
                reliability_target=observation.sfc_observation.reliability_target,
                normalized_deadline=observation.sfc_observation.normalized_deadline,
                normalized_input_size=observation.sfc_observation.normalized_input_size,
            ),
        )

    def _encoded_current_state(self) -> np.ndarray:
        """终止后返回固定零状态，否则编码当前因果观测和上一窗口历史。"""

        if self.execution_core.terminated:
            return np.zeros(self.dimensions.state_dim, dtype=np.float32)
        return self.state_encoder.encode(
            self._to_snapshot(self.execution_core.current_observation()),
            self._history,
        )

    def current_public_snapshot(self) -> DPPOStateSnapshot:
        """返回当前决策点可公开给策略和仿真教师的因果快照。

        教师通过这个入口读取与在线 DPPO 相同的信息，不能接触执行核心中
        尚未发生的请求、故障状态或未来窗口执行结果。
        """

        return self._to_snapshot(self.execution_core.current_observation())

    def reset(self, seed: int | None = None) -> np.ndarray:
        """重置外生轨迹和历史，并返回当前规模对应的状态向量。"""

        observation = self.execution_core.reset(seed=seed)
        self._history = DPPOHistoryObservation.zeros()
        return self.state_encoder.encode(self._to_snapshot(observation), self._history)

    def _build_history(
        self,
        *,
        decoded_action: DecodedDPPOAction,
        projection: ProjectionResult,
        execution: SlowWindowExecution,
    ) -> DPPOHistoryObservation:
        """把上一动作质量与最终执行结果压缩为公开的 11 项历史特征。"""

        actions = decoded_action.function_actions
        function_count = len(actions)
        mean_replica_ratio = (
            sum(action.replica_count for action in actions) / function_count / 3.0
        )
        three_replica_ratio = (
            sum(action.replica_count == 3 for action in actions) / function_count
        )
        maximum_retention = self.action_space.maximum_retention_seconds
        primary_retention_ratio = (
            sum(action.primary_retention_seconds for action in actions)
            / function_count
            / maximum_retention
        )
        backup_retention_ratio = (
            sum(action.backup_retention_seconds for action in actions)
            / function_count
            / maximum_retention
        )
        hot_ratio, cloud_ratio = self.execution_core.current_replica_ratios()
        metrics = execution.metrics
        repair_failure_rate = (
            metrics.fast_repair_failures / metrics.fast_repair_attempts
            if metrics.fast_repair_attempts > 0
            else 0.0
        )
        return DPPOHistoryObservation(
            previous_mean_replica_count=mean_replica_ratio,
            previous_three_replica_ratio=three_replica_ratio,
            previous_primary_retention_ratio=primary_retention_ratio,
            previous_backup_retention_ratio=backup_retention_ratio,
            previous_projection_change_ratio=projection.change_ratio,
            current_hot_replica_ratio=hot_ratio,
            current_cloud_replica_ratio=cloud_ratio,
            previous_success_rate=metrics.request_success_rate,
            previous_sla_violation_rate=metrics.sla_violation_rate,
            previous_normalized_cost=(
                execution.reward_breakdown.normalized_total_cost
            ),
            previous_repair_failure_rate=repair_failure_rate,
        )

    def step(
        self,
        raw_action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        """执行一次连续联合动作，并分开记录原始、投影和最终部署。"""

        clipped_action = self.action_space.clip(raw_action)
        raw_values = np.asarray(raw_action, dtype=np.float32).copy()
        decoded_action = self.action_space.decode(clipped_action)
        projection_inputs = self.execution_core.current_projection_inputs()
        projection = self.projector.project(
            decoded_action=decoded_action,
            operational_node_ids=projection_inputs.operational_node_ids,
            free_cpu=projection_inputs.free_cpu,
            free_memory_mb=projection_inputs.free_memory_mb,
            fault_domains=projection_inputs.fault_domains,
        )
        projected_intent = None
        if projection.success:
            projected_intent = self.intent_adapter.to_intent(
                projection,
                decision_slot=self.execution_core.current_slot,
                valid_until_slot=self.execution_core.next_window_end_slot,
            )
            execution = self.execution_core.execute_window(projected_intent)
        else:
            execution = self.execution_core.advance_rejected_window(
                projection.reasons
            )

        self._history = self._build_history(
            decoded_action=decoded_action,
            projection=projection,
            execution=execution,
        )
        next_state = self._encoded_current_state()
        info: dict[str, object] = {
            "raw_action": raw_values,
            "clipped_action": clipped_action,
            "decoded_action": decoded_action,
            "projection_result": projection,
            # 下列扁平字段供训练历史直接汇总，避免脚本依赖内部数据类结构。
            "raw_feasible": projection.raw_feasible,
            "projection_success": projection.success,
            "projection_change_ratio": projection.change_ratio,
            "projected_intent": projected_intent,
            "final_candidate_map": execution.final_candidate_map,
            "final_metrics": execution.metrics,
            "fast_repair_attempts": execution.metrics.fast_repair_attempts,
            "fast_repair_successes": execution.metrics.fast_repair_successes,
            "fast_repair_failures": execution.metrics.fast_repair_failures,
            "reward_breakdown": execution.reward_breakdown,
            "next_observation": next_state.copy(),
            "window_start_slot": execution.window_start_slot,
            "window_end_slot": execution.window_end_slot,
            "window_length": execution.window_length,
            "boundary_reason": execution.boundary_reason,
            "projection_rejected": execution.rejected,
            "rejection_reasons": execution.rejection_reasons,
            "next_observation_info": (
                self.execution_core.current_observation_info()
            ),
        }
        return (
            next_state,
            execution.reward_breakdown.reward,
            execution.terminated,
            False,
            info,
        )
