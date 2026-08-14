"""测试阶段 A 唯一成本账本和实例—秒计费。"""

import pytest

from src.cost_ledger import CostLedger, CostLedgerEntry


def test_running_cost_uses_instance_seconds_not_slot_count() -> None:
    one_second = CostLedger.running_entries(
        slot=0,
        slow_frame_index=0,
        slot_seconds=1.0,
        active_counts={(0, 0): 2},
        running_prices={(0, 0): 0.5},
        price_version="debug-v1",
    )
    half_second_twice = (
        CostLedger.running_entries(
            slot=0,
            slow_frame_index=0,
            slot_seconds=0.5,
            active_counts={(0, 0): 2},
            running_prices={(0, 0): 0.5},
            price_version="debug-v1",
        )
        + CostLedger.running_entries(
            slot=1,
            slow_frame_index=0,
            slot_seconds=0.5,
            active_counts={(0, 0): 2},
            running_prices={(0, 0): 0.5},
            price_version="debug-v1",
        )
    )

    assert sum(item.amount for item in one_second) == pytest.approx(1.0)
    assert sum(item.amount for item in half_second_twice) == pytest.approx(1.0)


def test_ledger_append_is_atomic_and_rejects_retention_charge() -> None:
    ledger = CostLedger()
    deployment = CostLedgerEntry(
        slot=3,
        slow_frame_index=0,
        cost_type="deployment",
        function_id=0,
        node_id=0,
        instance_batch_id="batch-1",
        count=2,
        physical_quantity=2.0,
        unit_price=1.5,
        amount=3.0,
        price_version="debug-v1",
    )
    ledger.append_all((deployment,))
    before = ledger.entries
    retention = CostLedgerEntry(
        slot=3,
        slow_frame_index=0,
        cost_type="retention",
        function_id=0,
        node_id=0,
        instance_batch_id="batch-1",
        count=2,
        physical_quantity=10.0,
        unit_price=1.0,
        amount=10.0,
        price_version="debug-v1",
    )

    with pytest.raises(ValueError, match="retention"):
        ledger.append_all((retention,))

    assert ledger.entries == before


def test_cost_entries_store_slow_frame_index() -> None:
    entries = CostLedger.running_entries(
        slot=12,
        slow_frame_index=1,
        slot_seconds=1.0,
        active_counts={(0, 0): 1},
        running_prices={(0, 0): 0.5},
        price_version="debug-v1",
    )

    assert entries[0].slot == 12
    assert entries[0].slow_frame_index == 1
