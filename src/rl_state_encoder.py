"""把慢层语义快照编码为顺序稳定的 DDQN 状态向量。"""

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np

from src.entities import (
    ServerlessFunction,
    SFCType,
)
from src.rl_agent_action_space import (
    CloudPolicy,
    RetentionPolicy,
    StructuredSlowAction,
    encode_ddqn_action,
)


@dataclass(frozen=True)
class NodeStateObservation:
    """保存一个轨旁 MEC 或中心云节点的六项观测。"""

    node_id: int
    free_cpu_ratio: float
    free_memory_ratio: float
    base_availability: float
    predicted_failure_probability: float
    operational: bool
    normalized_delay_from_serving: float


@dataclass(frozen=True)
class RLStateSnapshot:
    """保存一次慢层决策时刻需要编码的全部语义数据。"""

    serving_mec: int
    next_mec: int
    route_progress: float
    normalized_remaining_dwell: float
    normalized_mean_requests: float
    normalized_peak_requests: float
    normalized_load_trend: float
    global_failure_risk: float
    node_observations: tuple[
        NodeStateObservation,
        ...,
    ]
    current_action: StructuredSlowAction | None
    hot_replica_ratio: float
    cloud_replica_ratio: float
    has_history: bool
    previous_success_rate: float
    previous_sla_violation_rate: float
    previous_normalized_total_cost: float
    previous_repair_failure_rate: float


class RLStateEncoder:
    """按照公开特征名和固定顺序生成 DDQN 状态。"""

    def __init__(
        self,
        trackside_node_ids: Sequence[int],
        cloud_node_id: int,
        functions: Sequence[
            ServerlessFunction
        ],
        sfc: SFCType,
        input_size_mb_per_request: float,
        maximum_cpu_cycles_per_request: float,
        maximum_function_memory_mb: float,
        maximum_execution_time_ms: float,
        maximum_cold_start_time_ms: float,
        maximum_deadline_ms: float,
        maximum_input_size_mb: float,
        maximum_replica_count: int,
    ) -> None:
        """保存静态 SFC 特征及其归一化上限。"""

        self.trackside_node_ids = tuple(
            int(node_id)
            for node_id in trackside_node_ids
        )
        if len(self.trackside_node_ids) == 0:
            raise ValueError(
                "至少需要一个轨旁MEC节点。"
            )
        if len(set(self.trackside_node_ids)) != len(
            self.trackside_node_ids
        ):
            raise ValueError(
                "轨旁MEC节点编号不能重复。"
            )

        self.cloud_node_id = int(cloud_node_id)
        if self.cloud_node_id in (
            self.trackside_node_ids
        ):
            raise ValueError(
                "中心云节点编号不能与轨旁MEC重复。"
            )
        self.expected_node_ids = (
            self.trackside_node_ids
            + (self.cloud_node_id,)
        )

        function_map = {
            function.function_id: function
            for function in functions
        }
        if len(function_map) != len(functions):
            raise ValueError(
                "Serverless函数编号不能重复。"
            )
        if len(set(sfc.function_ids)) != len(
            sfc.function_ids
        ):
            raise ValueError(
                "SFC函数编号不能重复。"
            )
        if any(
            function_id not in function_map
            for function_id in sfc.function_ids
        ):
            raise KeyError(
                "SFC引用了未提供的Serverless函数。"
            )

        # 始终按照 SFC 执行顺序编码函数，不依赖传入列表的排列。
        self.functions = tuple(
            function_map[function_id]
            for function_id in sfc.function_ids
        )
        self.sfc = sfc

        self.input_size_mb_per_request = (
            self._validate_nonnegative_finite(
                input_size_mb_per_request,
                "单请求输入数据量",
            )
        )
        self.maximum_cpu_cycles_per_request = (
            self._validate_positive_finite(
                maximum_cpu_cycles_per_request,
                "最大单请求CPU需求",
            )
        )
        self.maximum_function_memory_mb = (
            self._validate_positive_finite(
                maximum_function_memory_mb,
                "最大函数内存",
            )
        )
        self.maximum_execution_time_ms = (
            self._validate_positive_finite(
                maximum_execution_time_ms,
                "最大温执行时延",
            )
        )
        self.maximum_cold_start_time_ms = (
            self._validate_positive_finite(
                maximum_cold_start_time_ms,
                "最大冷启动时延",
            )
        )
        self.maximum_deadline_ms = (
            self._validate_positive_finite(
                maximum_deadline_ms,
                "最大SFC时延上限",
            )
        )
        self.maximum_input_size_mb = (
            self._validate_positive_finite(
                maximum_input_size_mb,
                "最大单请求输入数据量",
            )
        )
        if maximum_replica_count < 3:
            raise ValueError(
                "最大副本数必须至少为3。"
            )
        self.maximum_replica_count = int(
            maximum_replica_count
        )

        self.feature_names = (
            self._build_feature_names()
        )
        self.state_dim = len(self.feature_names)

    @staticmethod
    def _validate_positive_finite(
        value: float,
        name: str,
    ) -> float:
        """检查归一化分母，防止除零或产生非有限状态。"""

        numeric_value = float(value)
        if (
            not math.isfinite(numeric_value)
            or numeric_value <= 0
        ):
            raise ValueError(
                f"{name}必须是正有限值。"
            )
        return numeric_value

    @staticmethod
    def _validate_nonnegative_finite(
        value: float,
        name: str,
    ) -> float:
        """检查只能为非负数的静态 SFC 参数。"""

        numeric_value = float(value)
        if (
            not math.isfinite(numeric_value)
            or numeric_value < 0
        ):
            raise ValueError(
                f"{name}必须是非负有限值。"
            )
        return numeric_value

    def _build_feature_names(
        self,
    ) -> tuple[str, ...]:
        """生成与编码值完全同序的公开特征名称。"""

        current_mec_names = tuple(
            f"current_mec_{node_id}"
            for node_id in self.trackside_node_ids
        )
        next_mec_names = tuple(
            f"next_mec_{node_id}"
            for node_id in self.trackside_node_ids
        )
        mobility_and_demand_names = (
            "route_progress",
            "normalized_remaining_dwell",
            "normalized_mean_requests",
            "normalized_peak_requests",
            "normalized_load_trend",
            "global_failure_risk",
        )
        node_names = tuple(
            f"node_{node_id}_{suffix}"
            for node_id in self.expected_node_ids
            for suffix in (
                "free_cpu_ratio",
                "free_memory_ratio",
                "base_availability",
                "predicted_failure_probability",
                "operational",
                "normalized_delay_from_serving",
            )
        )
        function_names = tuple(
            f"function_{function.function_id}_{suffix}"
            for function in self.functions
            for suffix in (
                "normalized_cpu",
                "normalized_memory",
                "normalized_execution_time",
                "normalized_cold_start_time",
            )
        )
        sfc_names = (
            "sfc_reliability_target",
            "sfc_normalized_deadline",
            "sfc_normalized_input_size",
        )
        history_names = (
            "current_normalized_replica_count",
            "current_retention_on_demand",
            "current_retention_primary_warm",
            "current_retention_all_warm",
            "current_cloud_allowed",
            "current_hot_replica_ratio",
            "current_cloud_replica_ratio",
            "previous_success_rate",
            "previous_sla_violation_rate",
            "previous_normalized_total_cost",
            "previous_repair_failure_rate",
        )

        return (
            current_mec_names
            + next_mec_names
            + mobility_and_demand_names
            + node_names
            + function_names
            + sfc_names
            + history_names
        )

    @staticmethod
    def _ensure_unit_interval(
        value: float,
        name: str,
    ) -> float:
        """检查已经归一化的动态特征是否位于 [0, 1]。"""

        numeric_value = float(value)
        if (
            not math.isfinite(numeric_value)
            or not 0.0 <= numeric_value <= 1.0
        ):
            raise ValueError(
                f"状态特征{name}必须是[0,1]内的有限值。"
            )
        return numeric_value

    @classmethod
    def _normalize_nonnegative(
        cls,
        value: float,
        maximum: float,
        name: str,
    ) -> float:
        """用固定上限缩放静态需求，并把极端值截断到 1。"""

        numeric_value = (
            cls._validate_nonnegative_finite(
                value,
                name,
            )
        )
        return min(
            numeric_value / maximum,
            1.0,
        )

    def _encode_one_hot(
        self,
        node_id: int,
        name: str,
    ) -> list[float]:
        """按轨旁节点固定顺序编码当前或下一服务 MEC。"""

        if node_id not in self.trackside_node_ids:
            raise ValueError(
                f"{name}必须是已配置的轨旁MEC。"
            )
        return [
            float(candidate_id == node_id)
            for candidate_id
            in self.trackside_node_ids
        ]

    def _encode_node_observations(
        self,
        observations: tuple[
            NodeStateObservation,
            ...,
        ],
    ) -> list[float]:
        """严格按五个 MEC 后接中心云的顺序编码节点观测。"""

        observed_ids = tuple(
            observation.node_id
            for observation in observations
        )
        if observed_ids != self.expected_node_ids:
            raise ValueError(
                "节点观测必须完整、无重复并按配置顺序提供。"
            )

        values: list[float] = []
        for observation in observations:
            prefix = f"node_{observation.node_id}"
            values.extend(
                [
                    self._ensure_unit_interval(
                        observation.free_cpu_ratio,
                        f"{prefix}_free_cpu_ratio",
                    ),
                    self._ensure_unit_interval(
                        observation.free_memory_ratio,
                        f"{prefix}_free_memory_ratio",
                    ),
                    self._ensure_unit_interval(
                        observation.base_availability,
                        f"{prefix}_base_availability",
                    ),
                    self._ensure_unit_interval(
                        observation.predicted_failure_probability,
                        f"{prefix}_predicted_failure_probability",
                    ),
                    float(observation.operational),
                    self._ensure_unit_interval(
                        observation.normalized_delay_from_serving,
                        f"{prefix}_normalized_delay_from_serving",
                    ),
                ]
            )
        return values

    def _encode_static_sfc(self) -> list[float]:
        """编码 VNF 需求和整条 SFC 的三个静态约束。"""

        values: list[float] = []
        for function in self.functions:
            prefix = (
                f"function_{function.function_id}"
            )
            values.extend(
                [
                    self._normalize_nonnegative(
                        function.cpu_cycles_per_request,
                        self.maximum_cpu_cycles_per_request,
                        f"{prefix}_cpu",
                    ),
                    self._normalize_nonnegative(
                        function.memory_mb,
                        self.maximum_function_memory_mb,
                        f"{prefix}_memory",
                    ),
                    self._normalize_nonnegative(
                        function.warm_exec_time_ms,
                        self.maximum_execution_time_ms,
                        f"{prefix}_execution_time",
                    ),
                    self._normalize_nonnegative(
                        function.cold_start_time_ms,
                        self.maximum_cold_start_time_ms,
                        f"{prefix}_cold_start_time",
                    ),
                ]
            )

        values.extend(
            [
                self._ensure_unit_interval(
                    self.sfc.reliability_target,
                    "sfc_reliability_target",
                ),
                self._normalize_nonnegative(
                    self.sfc.deadline_ms,
                    self.maximum_deadline_ms,
                    "sfc_deadline",
                ),
                self._normalize_nonnegative(
                    self.input_size_mb_per_request,
                    self.maximum_input_size_mb,
                    "sfc_input_size",
                ),
            ]
        )
        return values

    def _encode_history(
        self,
        snapshot: RLStateSnapshot,
    ) -> list[float]:
        """编码当前部署动作及上一慢窗口的四项结果。"""

        # reset 时还没有真实慢层动作，全部置零而不是伪造默认动作。
        if not snapshot.has_history:
            return [0.0] * 11

        if snapshot.current_action is None:
            raise ValueError(
                "已有历史窗口时必须提供历史动作。"
            )

        action = snapshot.current_action
        encoded_action_id = encode_ddqn_action(
            replica_count=action.replica_count,
            retention_policy=(
                action.retention_policy
            ),
            cloud_policy=action.cloud_policy,
        )
        if encoded_action_id != action.action_id:
            raise ValueError(
                "历史动作编号与动作分量不一致。"
            )

        retention_one_hot = [
            float(
                action.retention_policy
                is retention_policy
            )
            for retention_policy in RetentionPolicy
        ]
        return [
            self._ensure_unit_interval(
                action.replica_count
                / self.maximum_replica_count,
                "current_normalized_replica_count",
            ),
            *retention_one_hot,
            float(
                action.cloud_policy
                is CloudPolicy.CLOUD_ALLOWED
            ),
            self._ensure_unit_interval(
                snapshot.hot_replica_ratio,
                "current_hot_replica_ratio",
            ),
            self._ensure_unit_interval(
                snapshot.cloud_replica_ratio,
                "current_cloud_replica_ratio",
            ),
            self._ensure_unit_interval(
                snapshot.previous_success_rate,
                "previous_success_rate",
            ),
            self._ensure_unit_interval(
                snapshot.previous_sla_violation_rate,
                "previous_sla_violation_rate",
            ),
            self._ensure_unit_interval(
                snapshot.previous_normalized_total_cost,
                "previous_normalized_total_cost",
            ),
            self._ensure_unit_interval(
                snapshot.previous_repair_failure_rate,
                "previous_repair_failure_rate",
            ),
        ]

    def encode(
        self,
        snapshot: RLStateSnapshot,
    ) -> np.ndarray:
        """按移动性、节点、SFC、历史四组顺序生成状态向量。"""

        normalized_trend = float(
            snapshot.normalized_load_trend
        )
        if (
            not math.isfinite(normalized_trend)
            or not -1.0
            <= normalized_trend
            <= 1.0
        ):
            raise ValueError(
                "状态特征normalized_load_trend"
                "必须是[-1,1]内的有限值。"
            )

        values = (
            self._encode_one_hot(
                snapshot.serving_mec,
                "当前服务MEC",
            )
            + self._encode_one_hot(
                snapshot.next_mec,
                "下一服务MEC",
            )
            + [
                self._ensure_unit_interval(
                    snapshot.route_progress,
                    "route_progress",
                ),
                self._ensure_unit_interval(
                    snapshot.normalized_remaining_dwell,
                    "normalized_remaining_dwell",
                ),
                self._ensure_unit_interval(
                    snapshot.normalized_mean_requests,
                    "normalized_mean_requests",
                ),
                self._ensure_unit_interval(
                    snapshot.normalized_peak_requests,
                    "normalized_peak_requests",
                ),
                # 预测器输出 [-1,1]，神经网络状态统一映射到 [0,1]。
                (normalized_trend + 1.0) / 2.0,
                self._ensure_unit_interval(
                    snapshot.global_failure_risk,
                    "global_failure_risk",
                ),
            ]
            + self._encode_node_observations(
                snapshot.node_observations
            )
            + self._encode_static_sfc()
            + self._encode_history(snapshot)
        )
        state = np.asarray(
            values,
            dtype=np.float32,
        )

        if state.shape != (self.state_dim,):
            raise ValueError(
                "状态特征维度与公开模式不一致："
                f"期望{self.state_dim}，"
                f"实际{state.shape}。"
            )
        if not np.isfinite(state).all():
            raise ValueError(
                "状态特征必须全部是有限值。"
            )
        if not (
            (0.0 <= state) & (state <= 1.0)
        ).all():
            raise ValueError(
                "状态特征必须全部位于[0,1]。"
            )

        return state
