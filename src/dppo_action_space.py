"""定义配置驱动的 DPPO 连续联合 SFC 动作。"""

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np

from src.scenario_dimensions import ScenarioDimensions


@dataclass(frozen=True)
class DecodedFunctionAction:
    """保存一个 VNF 从连续向量中解码出的部署偏好。"""

    function_id: int
    ranked_node_ids: tuple[int, ...]
    replica_count: int
    primary_retention_seconds: float
    backup_retention_seconds: float


@dataclass(frozen=True)
class DecodedDPPOAction:
    """保存一次慢时间尺度决策中整条 SFC 的联合动作。"""

    function_actions: tuple[DecodedFunctionAction, ...]


class DPPOActionSpace:
    """负责连续动作的布局、截断、解码和教师动作编码。

    动作向量严格按以下顺序排列：

    1. ``F * N`` 个节点评分，重排为 ``(F, N)``；
    2. ``F`` 个副本评分；
    3. ``2 * F`` 个保留值，每个 VNF 依次保存“主、备”两个值。

    其中 ``F`` 是 VNF 数量，``N`` 是全部 MEC 加一个中心云的节点数量。
    """

    def __init__(
        self,
        dimensions: ScenarioDimensions,
        maximum_retention_seconds: float,
        minimum_replicas: int,
        maximum_replicas: int,
    ) -> None:
        """保存动作参数，并预先计算所有动态切片。"""

        maximum_retention = float(maximum_retention_seconds)
        if (
            isinstance(maximum_retention_seconds, bool)
            or not math.isfinite(maximum_retention)
            or maximum_retention <= 0.0
        ):
            raise ValueError("maximum_retention_seconds 必须是正有限数。")

        if (
            isinstance(minimum_replicas, bool)
            or not isinstance(minimum_replicas, int)
            or minimum_replicas <= 0
        ):
            raise ValueError("minimum_replicas 必须是正整数。")
        if (
            isinstance(maximum_replicas, bool)
            or not isinstance(maximum_replicas, int)
            or maximum_replicas <= 0
        ):
            raise ValueError("maximum_replicas 必须是正整数。")
        if minimum_replicas > maximum_replicas:
            raise ValueError("minimum_replicas 不能大于 maximum_replicas。")
        if maximum_replicas > dimensions.compute_node_count:
            raise ValueError("maximum_replicas 不能大于 compute_node_count。")

        self.dimensions = dimensions
        self.maximum_retention_seconds = maximum_retention
        self.minimum_replicas = minimum_replicas
        self.maximum_replicas = maximum_replicas

        self.node_score_count = (
            dimensions.function_count * dimensions.compute_node_count
        )
        self.replica_score_count = dimensions.function_count
        self.retention_value_count = 2 * dimensions.function_count

        replica_start = self.node_score_count
        retention_start = replica_start + self.replica_score_count
        self.node_score_slice = slice(0, replica_start)
        self.replica_score_slice = slice(replica_start, retention_start)
        self.retention_slice = slice(retention_start, dimensions.action_dim)

    @property
    def action_dim(self) -> int:
        """返回当前场景对应的连续动作维度。"""

        return self.dimensions.action_dim

    def _validated_action(self, action: np.ndarray) -> np.ndarray:
        """复制并检查动作，保证调用方数组不会被本类意外修改。"""

        array = np.asarray(action)
        expected_shape = (self.action_dim,)
        if array.shape != expected_shape:
            raise ValueError(
                f"动作形状必须是 {expected_shape}，实际为 {array.shape}。"
            )
        try:
            values = array.astype(np.float32, copy=True)
        except (TypeError, ValueError) as error:
            raise ValueError("动作必须由数值组成。") from error
        if not np.isfinite(values).all():
            raise ValueError("动作的每个分量都必须是有限值。")
        return values

    def clip(self, action: np.ndarray) -> np.ndarray:
        """把用于环境执行的动作逐分量截断到 ``[-1, 1]``。"""

        values = self._validated_action(action)
        return np.clip(values, -1.0, 1.0).astype(np.float32, copy=False)

    def decode(self, action: np.ndarray) -> DecodedDPPOAction:
        """将连续联合动作解码成每个 VNF 的可解释部署偏好。"""

        clipped = self.clip(action)
        node_scores = clipped[self.node_score_slice].reshape(
            self.dimensions.function_count,
            self.dimensions.compute_node_count,
        )
        replica_scores = clipped[self.replica_score_slice]
        # reshape(F, 2) 明确规定每行依次为主副本、备用副本保留值。
        retention_values = clipped[self.retention_slice].reshape(
            self.dimensions.function_count,
            2,
        )

        function_actions: list[DecodedFunctionAction] = []
        node_ids = self.dimensions.compute_node_ids
        for function_index, function_id in enumerate(self.dimensions.function_ids):
            score_by_node = {
                node_id: float(node_scores[function_index, node_index])
                for node_index, node_id in enumerate(node_ids)
            }
            ranked_node_ids = tuple(
                sorted(
                    node_ids,
                    key=lambda node_id: (-score_by_node[node_id], node_id),
                )
            )
            replica_count = self._decode_replica_count(
                float(replica_scores[function_index])
            )
            primary_seconds, backup_seconds = (
                self._retention_seconds(value)
                for value in retention_values[function_index]
            )
            function_actions.append(
                DecodedFunctionAction(
                    function_id=function_id,
                    ranked_node_ids=ranked_node_ids,
                    replica_count=replica_count,
                    primary_retention_seconds=primary_seconds,
                    backup_retention_seconds=backup_seconds,
                )
            )

        return DecodedDPPOAction(function_actions=tuple(function_actions))

    def _decode_replica_count(self, score: float) -> int:
        """把一个连续分量均匀量化为配置区间内的整数副本数。"""

        candidate_count = self.maximum_replicas - self.minimum_replicas + 1
        normalized = (float(score) + 1.0) / 2.0
        index = min(
            int(math.floor(normalized * candidate_count)),
            candidate_count - 1,
        )
        return self.minimum_replicas + index

    def _encode_replica_count(self, replica_count: int) -> float:
        """把教师副本数放在量化区间中心，确保编码后能稳定解码。"""

        if isinstance(replica_count, bool) or not isinstance(replica_count, int):
            raise ValueError("教师动作的副本数必须是整数。")
        if not self.minimum_replicas <= replica_count <= self.maximum_replicas:
            raise ValueError("教师动作的副本数必须位于配置上下限内。")
        candidate_count = self.maximum_replicas - self.minimum_replicas + 1
        index = replica_count - self.minimum_replicas
        return -1.0 + 2.0 * (index + 0.5) / candidate_count

    def _retention_seconds(self, normalized_value: float) -> float:
        """把 ``[-1, 1]`` 中的归一化值线性转换为秒。"""

        return (
            (float(normalized_value) + 1.0)
            / 2.0
            * self.maximum_retention_seconds
        )

    def encode_teacher_action(
        self,
        function_actions: Sequence[DecodedFunctionAction],
    ) -> np.ndarray:
        """把仿真教师给出的可解释动作编码为同一连续向量。

        节点排名被均匀映射到 ``[1, -1]``。副本数使用量化区间中心编码，
        因此增加可选档位时仍然只占用每个 VNF 的一个连续动作分量。
        """

        actions = tuple(function_actions)
        actual_ids = tuple(action.function_id for action in actions)
        if actual_ids != self.dimensions.function_ids:
            raise ValueError("教师动作必须按 ScenarioDimensions 的 VNF 顺序提供。")

        encoded = np.zeros(self.action_dim, dtype=np.float32)
        node_score_matrix = encoded[self.node_score_slice].reshape(
            self.dimensions.function_count,
            self.dimensions.compute_node_count,
        )
        replica_scores = encoded[self.replica_score_slice]
        retention_values = encoded[self.retention_slice].reshape(
            self.dimensions.function_count,
            2,
        )
        expected_nodes = self.dimensions.compute_node_ids
        expected_node_set = set(expected_nodes)
        ranking_denominator = self.dimensions.compute_node_count - 1

        for function_index, action in enumerate(actions):
            if (
                len(action.ranked_node_ids) != len(expected_nodes)
                or set(action.ranked_node_ids) != expected_node_set
            ):
                raise ValueError("教师节点排名必须包含全部计算节点且不能重复。")
            score_by_node = {
                node_id: 1.0 - 2.0 * rank / ranking_denominator
                for rank, node_id in enumerate(action.ranked_node_ids)
            }
            node_score_matrix[function_index] = np.asarray(
                [score_by_node[node_id] for node_id in expected_nodes],
                dtype=np.float32,
            )
            replica_scores[function_index] = self._encode_replica_count(
                action.replica_count
            )
            retention_values[function_index, 0] = self._encode_retention(
                action.primary_retention_seconds
            )
            retention_values[function_index, 1] = self._encode_retention(
                action.backup_retention_seconds
            )

        return encoded

    def _encode_retention(self, seconds: float) -> float:
        """把教师给出的秒数转换回 ``[-1, 1]`` 动作分量。"""

        value = float(seconds)
        if (
            isinstance(seconds, bool)
            or not math.isfinite(value)
            or not 0.0 <= value <= self.maximum_retention_seconds
        ):
            raise ValueError(
                "教师保留时间必须位于 [0, maximum_retention_seconds]。"
            )
        return 2.0 * value / self.maximum_retention_seconds - 1.0
