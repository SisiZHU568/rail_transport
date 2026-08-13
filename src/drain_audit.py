"""排空期停止边界、截断批次和成功率口径。"""

from dataclasses import dataclass
import math

from src.queue_state import BatchRecord


@dataclass(frozen=True)
class CensoredBatch:
    batch_id: str
    remaining_input_equivalent_bits: float
    violation_recorded: bool


@dataclass(frozen=True)
class DrainReport:
    arrived_batch_count: int
    on_time_completed_count: int
    violated_batch_count: int
    censored_batch_count: int
    success_rate_all_arrivals: float
    success_rate_resolved: float
    censored_batches: tuple[CensoredBatch, ...]


@dataclass(frozen=True)
class DrainPolicy:
    """排空上限统一补齐到下一个完整慢帧边界。"""

    normal_end_slot: int
    maximum_drain_slots: int
    slow_frame_slots: int

    def __post_init__(self) -> None:
        if self.normal_end_slot < 0 or self.maximum_drain_slots < 0:
            raise ValueError("正常期和排空期时隙数不能为负。")
        if self.slow_frame_slots <= 0:
            raise ValueError("slow_frame_slots 必须为正。")

    @property
    def effective_drain_end_slot(self) -> int:
        raw_end = self.normal_end_slot + self.maximum_drain_slots
        return (
            math.ceil(raw_end / self.slow_frame_slots)
            * self.slow_frame_slots
        )

    def should_stop(self, *, current_slot: int, unfinished_batch_count: int) -> bool:
        if current_slot < self.normal_end_slot:
            return False
        # 即使提前排空，也运行到当前慢帧末，保证慢层转移等长。
        at_slow_boundary = current_slot % self.slow_frame_slots == 0
        return current_slot >= self.effective_drain_end_slot or (
            unfinished_batch_count == 0 and at_slow_boundary
        )


def build_drain_report(
    batches: tuple[BatchRecord, ...],
    *,
    current_slot: int,
    slot_seconds: float,
) -> DrainReport:
    """按全部到达批次和已解决批次分别计算成功率。"""

    if current_slot < 0 or not math.isfinite(slot_seconds) or slot_seconds <= 0.0:
        raise ValueError("排空审计时间参数无效。")
    on_time = 0
    violated = 0
    censored: list[CensoredBatch] = []
    for batch in batches:
        if batch.violation_recorded:
            violated += 1
        if batch.completion_slot is not None:
            completion_time = batch.completion_slot * slot_seconds
            if completion_time <= batch.absolute_deadline_time:
                on_time += 1
        else:
            censored.append(
                CensoredBatch(
                    batch.batch_id,
                    max(
                        0.0,
                        batch.total_input_equivalent_bits
                        - batch.completed_input_equivalent_bits,
                    ),
                    batch.violation_recorded,
                )
            )
    arrived = len(batches)
    resolved = on_time + violated
    return DrainReport(
        arrived_batch_count=arrived,
        on_time_completed_count=on_time,
        violated_batch_count=violated,
        censored_batch_count=len(censored),
        success_rate_all_arrivals=(on_time / arrived if arrived else 0.0),
        success_rate_resolved=(on_time / resolved if resolved else 0.0),
        censored_batches=tuple(sorted(censored, key=lambda item: item.batch_id)),
    )
