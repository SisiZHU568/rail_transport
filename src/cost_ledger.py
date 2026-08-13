"""阶段 A 唯一成本账本；保留承诺只通过运行时间产生费用。"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CostLedgerEntry:
    """一笔可追溯的实例事件费或实例运行时间费。"""

    slot: int
    slow_frame_index: int
    cost_type: str
    function_id: int
    node_id: int
    instance_batch_id: str
    count: int
    physical_quantity: float
    unit_price: float
    amount: float
    price_version: str

    def __post_init__(self) -> None:
        if (
            self.slot < 0
            or self.slow_frame_index < 0
            or self.function_id < 0
            or self.node_id < 0
        ):
            raise ValueError("成本记录的时隙和实体 ID 不能为负。")
        if self.count <= 0 or not self.price_version:
            raise ValueError("成本记录的数量必须为正且价格版本不能为空。")
        values = (self.physical_quantity, self.unit_price, self.amount)
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("成本记录的物理量、单价和金额必须非负有限。")
        if not math.isclose(
            self.physical_quantity * self.unit_price,
            self.amount,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("成本金额必须等于物理量乘以单价。")


class CostLedger:
    """先校验整批记录，再一次性追加到不可变记录序列。"""

    def __init__(self) -> None:
        self._entries: tuple[CostLedgerEntry, ...] = ()

    @property
    def entries(self) -> tuple[CostLedgerEntry, ...]:
        return self._entries

    def append_all(self, entries: tuple[CostLedgerEntry, ...]) -> None:
        if any(entry.cost_type == "retention" for entry in entries):
            raise ValueError("retention 不能单独计费，避免与 running 重复。")
        if any(
            entry.cost_type not in {"deployment", "cold", "running"}
            for entry in entries
        ):
            raise ValueError("未知的实例成本类型。")
        self._entries = (*self._entries, *entries)

    @staticmethod
    def running_entries(
        *,
        slot: int,
        slow_frame_index: int,
        slot_seconds: float,
        active_counts: dict[tuple[int, int], int],
        running_prices: dict[tuple[int, int], float],
        price_version: str,
    ) -> tuple[CostLedgerEntry, ...]:
        """用活动实例数乘时隙秒数，生成货币/实例/秒计费记录。"""

        if not math.isfinite(slot_seconds) or slot_seconds <= 0.0:
            raise ValueError("slot_seconds 必须是正有限数。")
        return tuple(
            CostLedgerEntry(
                slot=slot,
                slow_frame_index=slow_frame_index,
                cost_type="running",
                function_id=function_id,
                node_id=node_id,
                instance_batch_id="",
                count=count,
                physical_quantity=count * slot_seconds,
                unit_price=running_prices[(function_id, node_id)],
                amount=(
                    count
                    * slot_seconds
                    * running_prices[(function_id, node_id)]
                ),
                price_version=price_version,
            )
            for (function_id, node_id), count in sorted(active_counts.items())
            if count > 0
        )
