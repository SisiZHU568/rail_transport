"""测试批次级 SLA、迟到完成和排空期统计。"""

import pytest

from src.drain_audit import DrainPolicy, build_drain_report
from src.queue_manager import (
    AllocationOperation,
    FastAllocationPlan,
    QueueCommitContext,
    QueueKey,
    QueueStateManager,
)
from src.queue_state import (
    BatchRecord,
    CompletionEvent,
    QueueFragment,
    StageFlowConfig,
)


def execute_final(
    manager: QueueStateManager,
    amount: float,
) -> None:
    snapshot = manager.snapshot()
    plan = FastAllocationPlan(
        snapshot.version,
        1,
        2,
        3,
        snapshot.current_slot,
        (
            AllocationOperation(
                QueueKey.stage(0, 0, None),
                "execute",
                amount,
            ),
        ),
    )
    context = QueueCommitContext(
        snapshot.version,
        1,
        2,
        3,
        snapshot.current_slot,
        {0: True},
        {(0, 0): 1},
    )
    assert manager.commit_allocation(plan, context).accepted is True


def test_completion_exactly_at_deadline_is_on_time() -> None:
    batch = BatchRecord("b", 0, 0.0, 1.0, total_input_equivalent_bits=2.0)
    fragment = QueueFragment("f", "b", 0, 0, 0, None, 2.0, 0)
    manager = QueueStateManager(
        StageFlowConfig((1.0,)),
        slot_seconds=1.0,
        initial_batches=(batch,),
        initial_stage_fragments=(fragment,),
    )

    execute_final(manager, 2.0)
    manager.begin_slot(1)
    report = manager.audit_deadlines()

    assert report.new_violation_count == 0
    assert report.on_time_completed_count == 1
    assert manager.snapshot().batches[0].violation_recorded is False


def test_violation_is_recorded_once_and_late_batch_still_completes() -> None:
    batch = BatchRecord("b", 0, 0.0, 0.5, total_input_equivalent_bits=2.0)
    fragment = QueueFragment("f", "b", 0, 0, 0, None, 2.0, 0)
    manager = QueueStateManager(
        StageFlowConfig((1.0,)),
        slot_seconds=1.0,
        initial_batches=(batch,),
        initial_stage_fragments=(fragment,),
    )

    manager.begin_slot(1)
    first = manager.audit_deadlines()
    second = manager.audit_deadlines()
    execute_final(manager, 2.0)
    manager.begin_slot(2)
    completed = manager.audit_deadlines()

    assert first.new_violation_count == 1
    assert second.new_violation_count == 0
    assert completed.late_completed_count == 1
    assert manager.snapshot().batches[0].completion_slot == 2


def test_late_completion_records_violation_when_no_earlier_audit_occurred() -> None:
    """完成事件与截止审计同边界发生时，也不能漏记首次违约。"""

    batch = BatchRecord("b", 0, 0.0, 0.5, total_input_equivalent_bits=1.0)
    fragment = QueueFragment("f", "b", 0, 0, 0, None, 1.0, 0)
    manager = QueueStateManager(
        StageFlowConfig((1.0,)),
        slot_seconds=1.0,
        initial_batches=(batch,),
        initial_stage_fragments=(fragment,),
    )

    execute_final(manager, 1.0)
    manager.begin_slot(1)
    report = manager.audit_deadlines()

    assert report.new_violation_count == 1
    assert report.late_completed_count == 1


def test_drain_limit_rounds_up_to_slow_frame_boundary() -> None:
    policy = DrainPolicy(
        normal_end_slot=20,
        maximum_drain_slots=7,
        slow_frame_slots=10,
    )

    assert policy.effective_drain_end_slot == 30
    # 正常期恰好结束在慢帧边界且没有遗留批次时，可以直接结束。
    assert policy.should_stop(current_slot=20, unfinished_batch_count=0) is True
    # 若在慢帧中途才排空，则必须继续到下一个慢帧边界。
    assert policy.should_stop(current_slot=23, unfinished_batch_count=0) is False
    assert policy.should_stop(current_slot=30, unfinished_batch_count=1) is True
    assert policy.should_stop(current_slot=30, unfinished_batch_count=0) is True


def test_drain_report_uses_all_arrivals_and_resolved_denominators() -> None:
    batches = (
        BatchRecord(
            "on-time",
            0,
            0.0,
            2.0,
            total_input_equivalent_bits=1.0,
            completed_input_equivalent_bits=1.0,
            completion_slot=2,
        ),
        BatchRecord(
            "late",
            0,
            0.0,
            1.0,
            total_input_equivalent_bits=1.0,
            completed_input_equivalent_bits=1.0,
            completion_slot=3,
            violation_recorded=True,
        ),
        BatchRecord(
            "censored",
            0,
            0.0,
            10.0,
            total_input_equivalent_bits=4.0,
            completed_input_equivalent_bits=1.0,
        ),
    )

    report = build_drain_report(batches, current_slot=4, slot_seconds=1.0)

    assert report.arrived_batch_count == 3
    assert report.on_time_completed_count == 1
    assert report.violated_batch_count == 1
    assert report.censored_batch_count == 1
    assert report.success_rate_all_arrivals == pytest.approx(1.0 / 3.0)
    assert report.success_rate_resolved == pytest.approx(0.5)
    assert report.censored_batches[0].remaining_input_equivalent_bits == pytest.approx(3.0)


def test_drain_report_derives_late_completion_even_without_prior_audit() -> None:
    batch = BatchRecord(
        "late",
        0,
        0.0,
        1.0,
        total_input_equivalent_bits=1.0,
        completed_input_equivalent_bits=1.0,
        completion_slot=2,
    )

    report = build_drain_report((batch,), current_slot=2, slot_seconds=1.0)

    assert report.violated_batch_count == 1
    assert report.success_rate_resolved == 0.0


def test_delayed_boundary_processing_preserves_event_completion_slot() -> None:
    batch = BatchRecord("b", 0, 0.0, 2.0, total_input_equivalent_bits=1.0)
    fragment = QueueFragment("f", "b", 0, 0, 0, None, 1.0, 0)
    manager = QueueStateManager(
        StageFlowConfig((1.0,)),
        slot_seconds=1.0,
        initial_batches=(batch,),
        initial_stage_fragments=(fragment,),
    )

    execute_final(manager, 1.0)
    manager.begin_slot(5)

    assert manager.snapshot().batches[0].completion_slot == 1


def test_boundary_event_overcompletion_is_rejected_atomically() -> None:
    batch = BatchRecord("b", 0, 0.0, 2.0, total_input_equivalent_bits=1.0)
    event = CompletionEvent("c", "b", 1, 2.0)
    manager = QueueStateManager(
        StageFlowConfig((1.0,)),
        slot_seconds=1.0,
        initial_batches=(batch,),
        initial_completion_events=(event,),
    )
    before = manager.snapshot()

    with pytest.raises(ValueError, match="completion flow"):
        manager.begin_slot(1)

    assert manager.snapshot() == before


def test_drain_report_derives_overdue_unfinished_violation() -> None:
    batch = BatchRecord("b", 0, 0.0, 1.0, total_input_equivalent_bits=1.0)

    report = build_drain_report((batch,), current_slot=2, slot_seconds=1.0)

    assert report.violated_batch_count == 1
    assert report.censored_batches[0].violation_recorded is True
