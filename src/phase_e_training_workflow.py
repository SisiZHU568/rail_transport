"""阶段 E 的奖励、事务 rollout 与固定验证检查点选择。"""

from dataclasses import dataclass
import math
from typing import Generic, TypeVar


@dataclass(frozen=True)
class FrameRewardInput:
    at_risk_batches: int
    new_violations: int
    queue_equivalent_bits: float
    deficit_equivalent_bits: float
    raw_cost: float
    reference_cost: float


@dataclass(frozen=True)
class FrameReward:
    violation_rate: float
    deficit_rate: float
    service_loss: float
    normalized_cost: float
    reward: float


def compute_frame_reward(data: FrameRewardInput, *, alpha: float = 0.8) -> FrameReward:
    """仅使用服务损失与总成本；不重复奖励可靠性、冷启动或投影。"""
    values = (data.queue_equivalent_bits, data.deficit_equivalent_bits,
              data.raw_cost, data.reference_cost, alpha)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("奖励输入必须有限。")
    if not 0.0 <= alpha <= 1.0 or data.reference_cost <= 0.0:
        raise ValueError("alpha 必须位于 [0,1] 且 reference_cost 必须为正。")
    if min(data.at_risk_batches, data.new_violations) < 0:
        raise ValueError("批次数不能为负。")
    violation = data.new_violations / data.at_risk_batches if data.at_risk_batches else 0.0
    deficit = data.deficit_equivalent_bits / data.queue_equivalent_bits if data.queue_equivalent_bits > 0.0 else 0.0
    violation = min(1.0, max(0.0, violation))
    deficit = min(1.0, max(0.0, deficit))
    service = violation + deficit - violation * deficit
    cost = min(1.0, max(0.0, data.raw_cost / data.reference_cost))
    reward = -(alpha * service + (1.0 - alpha) * cost)
    return FrameReward(violation, deficit, service, cost, reward)


T = TypeVar("T")


class TransactionalRollout(Generic[T]):
    """内部失败只丢弃当前尚未用于 PPO 更新的固定长度轨迹段。"""
    def __init__(self, required_slow_frames: int) -> None:
        if required_slow_frames <= 0:
            raise ValueError("required_slow_frames 必须为正。")
        self.required_slow_frames = required_slow_frames
        self._staged: list[T] = []
        self._committed: list[T] = []
        self.last_failure_code: str | None = None

    @property
    def committed(self) -> tuple[T, ...]:
        return tuple(self._committed)

    def stage(self, transition: T) -> None:
        self._staged.append(transition)

    def commit_if_ready(self) -> tuple[T, ...]:
        if len(self._staged) < self.required_slow_frames:
            return ()
        committed = tuple(self._staged)
        self._committed.extend(committed)
        self._staged.clear()
        return committed

    def abort(self, failure_code: str) -> tuple[T, ...]:
        if not failure_code:
            raise ValueError("failure_code 不能为空。")
        discarded = tuple(self._staged)
        self._staged.clear()
        self.last_failure_code = failure_code
        return discarded


@dataclass(frozen=True)
class ValidationResult:
    checkpoint_id: str
    update_index: int
    violation_rate: float
    deficit_rate: float
    raw_cost: float
    internal_failure_count: int


def _quantize(value: float, precision: float) -> int:
    if not math.isfinite(value) or not math.isfinite(precision) or precision <= 0.0:
        raise ValueError("验证指标和精度必须是有效有限数。")
    return math.floor(value / precision + 0.5)


def select_validation_checkpoint(candidates: tuple[ValidationResult, ...], *,
                                 violation_precision: float,
                                 deficit_precision: float,
                                 cost_precision: float) -> ValidationResult:
    """先硬过滤内部失败，再按 SLA、缺口、成本和较早更新词典序选择。"""
    valid = tuple(item for item in candidates if item.internal_failure_count == 0)
    if not valid:
        raise ValueError("NO_VALID_CHECKPOINT")
    return min(valid, key=lambda item: (
        _quantize(item.violation_rate, violation_precision),
        _quantize(item.deficit_rate, deficit_precision),
        _quantize(item.raw_cost, cost_precision), item.update_index))
