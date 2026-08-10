"""把配置驱动的轨道场景观测编码为 DPPO 使用的平铺状态。"""

from dataclasses import dataclass
import math

import numpy as np

from src.scenario_dimensions import ScenarioDimensions


# 状态维度相同并不代表语义相同，数据集和检查点还必须核对这个版本。
DPPO_STATE_SCHEMA_VERSION = "dppo-v1-flat"


# 最后 11 维的顺序属于公开状态模式 ``dppo-v1-flat``，不能随意调整。
DPPO_HISTORY_FEATURE_NAMES = (
    "previous_mean_replica_count",
    "previous_three_replica_ratio",
    "previous_primary_retention_ratio",
    "previous_backup_retention_ratio",
    "previous_projection_change_ratio",
    "current_hot_replica_ratio",
    "current_cloud_replica_ratio",
    "previous_success_rate",
    "previous_sla_violation_rate",
    "previous_normalized_cost",
    "previous_repair_failure_rate",
)


@dataclass(frozen=True)
class DPPONodeObservation:
    """保存一个 MEC 或中心云节点的六项归一化观测。"""

    node_id: int
    free_cpu_ratio: float
    free_memory_ratio: float
    base_availability: float
    predicted_failure_probability: float
    operational: bool
    normalized_delay_from_serving: float


@dataclass(frozen=True)
class DPPOFunctionObservation:
    """保存一个 VNF 的四项归一化资源与执行特征。"""

    function_id: int
    normalized_cpu: float
    normalized_memory: float
    normalized_execution_time: float
    normalized_cold_start_time: float


@dataclass(frozen=True)
class DPPOSFCObservation:
    """保存整条 SFC 的可靠性、截止时间和输入规模特征。"""

    reliability_target: float
    normalized_deadline: float
    normalized_input_size: float


@dataclass(frozen=True)
class DPPOStateSnapshot:
    """保存一个慢尺度决策点可观测的当前场景状态。

    除 ``normalized_load_trend`` 使用 ``[-1, 1]`` 外，其余名称中含比例或
    ``normalized`` 的浮点特征都必须在进入编码器前归一化到 ``[0, 1]``。
    """

    serving_mec: int
    next_mec: int
    route_progress: float
    normalized_remaining_dwell: float
    normalized_mean_requests: float
    normalized_peak_requests: float
    normalized_load_trend: float
    global_failure_risk: float
    node_observations: tuple[DPPONodeObservation, ...]
    function_observations: tuple[DPPOFunctionObservation, ...]
    sfc_observation: DPPOSFCObservation

    @classmethod
    def zeros(cls, dimensions: ScenarioDimensions) -> "DPPOStateSnapshot":
        """构造 Episode reset 使用的零观测，不伪造历史执行结果。"""

        if not isinstance(dimensions, ScenarioDimensions):
            raise TypeError("dimensions must be a ScenarioDimensions instance.")
        default_mec = dimensions.mec_node_ids[0]
        return cls(
            serving_mec=default_mec,
            next_mec=default_mec,
            route_progress=0.0,
            normalized_remaining_dwell=0.0,
            normalized_mean_requests=0.0,
            normalized_peak_requests=0.0,
            normalized_load_trend=0.0,
            global_failure_risk=0.0,
            node_observations=tuple(
                DPPONodeObservation(
                    node_id=node_id,
                    free_cpu_ratio=0.0,
                    free_memory_ratio=0.0,
                    base_availability=0.0,
                    predicted_failure_probability=0.0,
                    operational=False,
                    normalized_delay_from_serving=0.0,
                )
                for node_id in dimensions.compute_node_ids
            ),
            function_observations=tuple(
                DPPOFunctionObservation(
                    function_id=function_id,
                    normalized_cpu=0.0,
                    normalized_memory=0.0,
                    normalized_execution_time=0.0,
                    normalized_cold_start_time=0.0,
                )
                for function_id in dimensions.function_ids
            ),
            sfc_observation=DPPOSFCObservation(
                reliability_target=0.0,
                normalized_deadline=0.0,
                normalized_input_size=0.0,
            ),
        )


@dataclass(frozen=True)
class DPPOHistoryObservation:
    """保存上一慢窗口动作质量和执行结果对应的 11 项历史特征。

    ``previous_mean_replica_count`` 保存“平均副本数除以最大副本数 3”的结果，
    因而与其余比例一样位于 ``[0, 1]``。
    """

    previous_mean_replica_count: float
    previous_three_replica_ratio: float
    previous_primary_retention_ratio: float
    previous_backup_retention_ratio: float
    previous_projection_change_ratio: float
    current_hot_replica_ratio: float
    current_cloud_replica_ratio: float
    previous_success_rate: float
    previous_sla_violation_rate: float
    previous_normalized_cost: float
    previous_repair_failure_rate: float

    @classmethod
    def zeros(cls) -> "DPPOHistoryObservation":
        """返回没有任何上一窗口信息的 Episode 初始历史。"""

        return cls(**{name: 0.0 for name in DPPO_HISTORY_FEATURE_NAMES})


def _unit_interval(value: object, name: str, *, group: str) -> float:
    """把归一化特征转换为浮点数，并拒绝非有限值和越界值。"""

    if isinstance(value, bool):
        raise ValueError(f"{group} {name} must be finite and in [0, 1].")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{group} {name} must be finite and in [0, 1]."
        ) from error
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{group} {name} must be finite and in [0, 1].")
    return number


class DPPOHistoryEncoder:
    """按照公开的 DPPO 历史特征顺序生成最后 11 维。"""

    def encode(self, history: DPPOHistoryObservation) -> list[float]:
        """验证并编码上一窗口历史，避免异常比例进入神经网络。"""

        if not isinstance(history, DPPOHistoryObservation):
            raise TypeError("history must be a DPPOHistoryObservation.")
        return [
            _unit_interval(
                getattr(history, name),
                name,
                group="history feature",
            )
            for name in DPPO_HISTORY_FEATURE_NAMES
        ]


class DPPOStateEncoder:
    """按 ``ScenarioDimensions`` 的唯一节点和 VNF 顺序编码状态。"""

    def __init__(self, dimensions: ScenarioDimensions) -> None:
        """保存统一维度对象，并一次性生成与数值完全同序的特征名。"""

        if not isinstance(dimensions, ScenarioDimensions):
            raise TypeError("dimensions must be a ScenarioDimensions instance.")
        self.dimensions = dimensions
        self.history_encoder = DPPOHistoryEncoder()
        self.feature_names = self._build_feature_names()
        if len(self.feature_names) != dimensions.state_dim:
            raise ValueError("Feature-name count does not match configured state_dim.")

    def _build_feature_names(self) -> tuple[str, ...]:
        """根据场景顺序动态构建状态模式，不依赖默认规模常数。"""

        current_mec_names = tuple(
            f"current_mec_{node_id}" for node_id in self.dimensions.mec_node_ids
        )
        next_mec_names = tuple(
            f"next_mec_{node_id}" for node_id in self.dimensions.mec_node_ids
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
            for node_id in self.dimensions.compute_node_ids
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
            f"function_{function_id}_{suffix}"
            for function_id in self.dimensions.function_ids
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
        return (
            current_mec_names
            + next_mec_names
            + mobility_and_demand_names
            + node_names
            + function_names
            + sfc_names
            + DPPO_HISTORY_FEATURE_NAMES
        )

    def encode(
        self,
        snapshot: DPPOStateSnapshot,
        history: DPPOHistoryObservation,
    ) -> np.ndarray:
        """编码当前快照和历史，并最终复核形状、有限性与数值范围。"""

        if not isinstance(snapshot, DPPOStateSnapshot):
            raise TypeError("snapshot must be a DPPOStateSnapshot.")
        values = self._encode_current_snapshot(snapshot)
        values.extend(self.history_encoder.encode(history))
        encoded = np.asarray(values, dtype=np.float32)
        if encoded.shape != (self.dimensions.state_dim,):
            raise ValueError("Encoded state dimension does not match configuration.")
        if not np.isfinite(encoded).all():
            raise ValueError("Encoded state must contain only finite values.")
        if not ((0.0 <= encoded) & (encoded <= 1.0)).all():
            raise ValueError("Encoded state values must be in [0, 1].")
        return encoded

    def _encode_current_snapshot(self, snapshot: DPPOStateSnapshot) -> list[float]:
        """按移动性、节点、VNF 和 SFC 四组顺序编码当前观测。"""

        trend = self._normalized_load_trend(snapshot.normalized_load_trend)
        values = self._one_hot_mec(snapshot.serving_mec, "serving_mec")
        values.extend(self._one_hot_mec(snapshot.next_mec, "next_mec"))
        for name, value in (
            ("route_progress", snapshot.route_progress),
            ("normalized_remaining_dwell", snapshot.normalized_remaining_dwell),
            ("normalized_mean_requests", snapshot.normalized_mean_requests),
            ("normalized_peak_requests", snapshot.normalized_peak_requests),
        ):
            values.append(_unit_interval(value, name, group="state feature"))
        values.extend(
            [
                trend,
                _unit_interval(
                    snapshot.global_failure_risk,
                    "global_failure_risk",
                    group="state feature",
                ),
            ]
        )
        values.extend(self._encode_nodes(snapshot.node_observations))
        values.extend(self._encode_functions(snapshot.function_observations))
        values.extend(self._encode_sfc(snapshot.sfc_observation))
        return values

    def _one_hot_mec(self, node_id: int, name: str) -> list[float]:
        """把当前和下一服务 MEC 作为类别 one-hot，而不是连续编号。"""

        if node_id not in self.dimensions.mec_node_ids:
            raise ValueError(f"{name} must be a configured MEC node ID.")
        return [
            float(candidate_id == node_id)
            for candidate_id in self.dimensions.mec_node_ids
        ]

    @staticmethod
    def _normalized_load_trend(value: object) -> float:
        """把预测器的 ``[-1, 1]`` 负载趋势映射到统一的 ``[0, 1]``。"""

        if isinstance(value, bool):
            raise ValueError(
                "state feature normalized_load_trend must be finite and in [-1, 1]."
            )
        try:
            trend = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "state feature normalized_load_trend must be finite and in [-1, 1]."
            ) from error
        if not math.isfinite(trend) or not -1.0 <= trend <= 1.0:
            raise ValueError(
                "state feature normalized_load_trend must be finite and in [-1, 1]."
            )
        return (trend + 1.0) / 2.0

    def _encode_nodes(
        self,
        observations: tuple[DPPONodeObservation, ...],
    ) -> list[float]:
        """严格按照所有 MEC 后接中心云的顺序编码节点观测。"""

        observed_ids = tuple(observation.node_id for observation in observations)
        if observed_ids != self.dimensions.compute_node_ids:
            raise ValueError(
                "node observations must be complete and follow configured order."
            )
        values: list[float] = []
        for observation in observations:
            if not isinstance(observation, DPPONodeObservation):
                raise ValueError(
                    "node observations must contain DPPONodeObservation values."
                )
            if not isinstance(observation.operational, bool):
                raise ValueError("node operational flag must be boolean.")
            prefix = f"node_{observation.node_id}"
            values.extend(
                [
                    _unit_interval(
                        observation.free_cpu_ratio,
                        f"{prefix}_free_cpu_ratio",
                        group="state feature",
                    ),
                    _unit_interval(
                        observation.free_memory_ratio,
                        f"{prefix}_free_memory_ratio",
                        group="state feature",
                    ),
                    _unit_interval(
                        observation.base_availability,
                        f"{prefix}_base_availability",
                        group="state feature",
                    ),
                    _unit_interval(
                        observation.predicted_failure_probability,
                        f"{prefix}_predicted_failure_probability",
                        group="state feature",
                    ),
                    float(observation.operational),
                    _unit_interval(
                        observation.normalized_delay_from_serving,
                        f"{prefix}_normalized_delay_from_serving",
                        group="state feature",
                    ),
                ]
            )
        return values

    def _encode_functions(
        self,
        observations: tuple[DPPOFunctionObservation, ...],
    ) -> list[float]:
        """严格按照联合动作相同的 VNF 顺序编码四项静态特征。"""

        observed_ids = tuple(observation.function_id for observation in observations)
        if observed_ids != self.dimensions.function_ids:
            raise ValueError(
                "function observations must be complete and follow configured order."
            )
        values: list[float] = []
        for observation in observations:
            if not isinstance(observation, DPPOFunctionObservation):
                raise ValueError(
                    "function observations must contain DPPOFunctionObservation values."
                )
            prefix = f"function_{observation.function_id}"
            for suffix, value in (
                ("normalized_cpu", observation.normalized_cpu),
                ("normalized_memory", observation.normalized_memory),
                ("normalized_execution_time", observation.normalized_execution_time),
                (
                    "normalized_cold_start_time",
                    observation.normalized_cold_start_time,
                ),
            ):
                values.append(
                    _unit_interval(
                        value,
                        f"{prefix}_{suffix}",
                        group="state feature",
                    )
                )
        return values

    @staticmethod
    def _encode_sfc(observation: DPPOSFCObservation) -> list[float]:
        """编码整条 SFC 的三项归一化约束。"""

        if not isinstance(observation, DPPOSFCObservation):
            raise ValueError("sfc_observation must be a DPPOSFCObservation.")
        return [
            _unit_interval(
                observation.reliability_target,
                "sfc_reliability_target",
                group="state feature",
            ),
            _unit_interval(
                observation.normalized_deadline,
                "sfc_normalized_deadline",
                group="state feature",
            ),
            _unit_interval(
                observation.normalized_input_size,
                "sfc_normalized_input_size",
                group="state feature",
            ),
        ]
