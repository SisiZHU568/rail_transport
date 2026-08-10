"""定义与具体学习算法无关的 SFC 部署意图。"""

from dataclasses import dataclass
import math


def _is_nonnegative_integer(value: object) -> bool:
    """布尔值在 Python 中属于整数子类，这里显式排除它。"""

    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _is_nonnegative_finite(value: object) -> bool:
    """检查保留时间是否可以安全用于后续的时隙换算。"""

    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number >= 0.0


@dataclass(frozen=True)
class FunctionDeploymentIntent:
    """描述一个 VNF 在一个慢尺度窗口内希望采用的副本部署。

    ``preferred_node_ids`` 的第一个节点是主副本，其余节点是备用副本。
    这里只表达慢层意图；精确路由、SLA 和可靠性仍由快层审计与修复。
    """

    function_id: int
    preferred_node_ids: tuple[int, ...]
    replica_count: int
    primary_retention_seconds: float
    backup_retention_seconds: float

    def __post_init__(self) -> None:
        """在错误意图进入快层之前，先检查最基本的数据不变量。"""

        if not _is_nonnegative_integer(self.function_id):
            raise ValueError("function_id must be a non-negative integer.")
        if (
            isinstance(self.replica_count, bool)
            or not isinstance(self.replica_count, int)
            or self.replica_count not in (1, 2, 3)
        ):
            raise ValueError("replica_count must be 1, 2, or 3.")
        if len(self.preferred_node_ids) != self.replica_count:
            raise ValueError(
                "preferred_node_ids length must match replica_count."
            )
        if any(
            not _is_nonnegative_integer(node_id)
            for node_id in self.preferred_node_ids
        ):
            raise ValueError("preferred node IDs must be non-negative integers.")
        if len(set(self.preferred_node_ids)) != len(self.preferred_node_ids):
            raise ValueError("preferred node IDs must be unique.")
        if not _is_nonnegative_finite(self.primary_retention_seconds):
            raise ValueError(
                "primary_retention_seconds must be finite and non-negative."
            )
        if not _is_nonnegative_finite(self.backup_retention_seconds):
            raise ValueError(
                "backup_retention_seconds must be finite and non-negative."
            )


@dataclass(frozen=True)
class SFCDeploymentIntent:
    """描述整条 SFC 在一段慢尺度有效期内的部署意图。"""

    decision_slot: int
    valid_until_slot: int
    function_intents: tuple[FunctionDeploymentIntent, ...]
    source_algorithm: str

    def __post_init__(self) -> None:
        """确保执行窗口和 VNF 列表含义明确且不会互相冲突。"""

        if not _is_nonnegative_integer(self.decision_slot):
            raise ValueError("decision_slot must be a non-negative integer.")
        if (
            not _is_nonnegative_integer(self.valid_until_slot)
            or self.valid_until_slot <= self.decision_slot
        ):
            raise ValueError(
                "valid_until_slot must be greater than decision_slot."
            )
        if not self.function_intents:
            raise ValueError("function_intents cannot be empty.")
        if any(
            not isinstance(intent, FunctionDeploymentIntent)
            for intent in self.function_intents
        ):
            raise ValueError(
                "function_intents must contain FunctionDeploymentIntent values."
            )
        function_ids = tuple(
            intent.function_id for intent in self.function_intents
        )
        if len(set(function_ids)) != len(function_ids):
            raise ValueError("function IDs in an SFC intent must be unique.")
        if not isinstance(self.source_algorithm, str) or not self.source_algorithm:
            raise ValueError("source_algorithm cannot be empty.")
