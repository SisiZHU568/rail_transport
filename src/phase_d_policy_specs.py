"""阶段 D 的观察/动作规格、归一化状态和策略输出适配器。"""

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from src.safe_deployment_decoder import ActionSpec


@dataclass(frozen=True)
class ObservationFeature:
    name: str
    dimension: int
    normalization: str
    physical_scale: float

    def __post_init__(self) -> None:
        if not self.name or self.dimension <= 0:
            raise ValueError("观察特征名称不能为空且维度必须为正。")
        if self.normalization not in {"none", "fixed_scale", "running"}:
            raise ValueError("未知观察归一化方法。")
        if not math.isfinite(self.physical_scale) or self.physical_scale <= 0.0:
            raise ValueError("观察特征的物理尺度必须是正有限数。")


@dataclass(frozen=True)
class ObservationSpec:
    features: tuple[ObservationFeature, ...]
    ordered_function_ids: tuple[int, ...]
    ordered_node_ids: tuple[int, ...]
    schema_version: str

    def __post_init__(self) -> None:
        if not self.features or not self.schema_version:
            raise ValueError("ObservationSpec 不能为空。")
        if len({item.name for item in self.features}) != len(self.features):
            raise ValueError("观察特征名称必须唯一。")

    @property
    def dimension(self) -> int:
        return sum(item.dimension for item in self.features)

    @property
    def sha256(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "features": [item.__dict__ for item in self.features],
            "ordered_function_ids": self.ordered_function_ids,
            "ordered_node_ids": self.ordered_node_ids,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True)
class NormalizationState:
    version: int
    means: tuple[float, ...]
    variances: tuple[float, ...]
    sample_count: int

    def __post_init__(self) -> None:
        if self.version <= 0 or self.sample_count < 0:
            raise ValueError("normalization version 必须为正且样本数不能为负。")
        if len(self.means) != len(self.variances):
            raise ValueError("normalization 均值和方差维度不一致。")
        if any(not math.isfinite(value) for value in (*self.means, *self.variances)):
            raise ValueError("normalization 统计必须有限。")
        if any(value < 0.0 for value in self.variances):
            raise ValueError("normalization 方差不能为负。")


@dataclass(frozen=True)
class PolicySpecification:
    observation_spec: ObservationSpec
    action_spec: ActionSpec
    normalization_state: NormalizationState

    def __post_init__(self) -> None:
        if len(self.normalization_state.means) != self.observation_spec.dimension:
            raise ValueError("normalization 维度与 ObservationSpec 不一致。")

    @property
    def observation_spec_hash(self) -> str:
        return self.observation_spec.sha256

    @property
    def action_spec_hash(self) -> str:
        return self.action_spec.sha256


class PolicyAdapter:
    """把扩散策略的无界变量 v 平滑映射为安全解码评分 u。"""

    def __init__(self, action_dim: int) -> None:
        if action_dim <= 0:
            raise ValueError("action_dim 必须为正。")
        self.action_dim = action_dim

    def to_scores(self, unbounded_action: np.ndarray) -> np.ndarray:
        values = np.asarray(unbounded_action, dtype=np.float64)
        if values.shape != (self.action_dim,) or not np.isfinite(values).all():
            raise ValueError("INVALID_POLICY_SCORE")
        # 分支写法避免 exp 在极端负值处溢出；这里不是动作裁剪。
        result = np.empty_like(values)
        positive = values >= 0.0
        result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
        exponent = np.exp(values[~positive])
        result[~positive] = exponent / (1.0 + exponent)
        return result
