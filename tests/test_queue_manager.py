"""测试跨时隙流水线、确定性 EDF 和原子队列提交。"""

from dataclasses import replace

import pytest

from src.queue_manager import (
    AllocationOperation,
    FastAllocationPlan,
    QueueCommitContext,
    QueueKey,
    QueueStateManager,
)
from src.queue_state import BatchRecord, QueueFragment, StageFlowConfig


def context(*, queue_version: int, current_slot: int) -> QueueCommitContext:
    return QueueCommitContext(
        queue_version=queue_version,
        lifecycle_version=3,
        failure_version=4,
        network_version=5,
        current_slot=current_slot,
        effective_node_up={0: True, 1: True, 2: True},
        warm_instance_counts={(0, 0): 1, (0, 1): 1, (1, 0): 1},
    )


def plan(
    manager: QueueStateManager,
    operations: tuple[AllocationOperation, ...],
) -> FastAllocationPlan:
    snapshot = manager.snapshot()
    return FastAllocationPlan(
        expected_queue_version=snapshot.version,
        expected_lifecycle_version=3,
        expected_failure_version=4,
        expected_network_version=5,
        current_slot=snapshot.current_slot,
        operations=operations,
    )


def test_uplink_and_vnf_outputs_are_available_only_next_slot() -> None:
    manager = QueueStateManager(StageFlowConfig((0.5, 1.0)), slot_seconds=1.0)
    manager.admit_batch("b", 0, 0.0, 10.0, 4.0)
    uplink = AllocationOperation(
        QueueKey.uplink(0), "uplink", 4.0, destination_node_id=0
    )

    uploaded = manager.commit_allocation(
        plan(manager, (uplink,)),
        context(queue_version=manager.snapshot().version, current_slot=0),
    )

    assert uploaded.accepted is True
    stage0 = uploaded.snapshot.stage_fragments[0]
    assert (stage0.stage_id, stage0.location, stage0.available_slot) == (0, 0, 1)

    same_slot_execute = AllocationOperation(
        QueueKey.stage(0, 0, None), "execute", 4.0
    )
    rejected = manager.commit_allocation(
        plan(manager, (same_slot_execute,)),
        context(queue_version=manager.snapshot().version, current_slot=0),
    )
    assert rejected.code == "FLOW_EXCEEDS_AVAILABLE_QUEUE"

    manager.begin_slot(1)
    executed = manager.commit_allocation(
        plan(manager, (same_slot_execute,)),
        context(queue_version=manager.snapshot().version, current_slot=1),
    )
    assert executed.accepted is True
    stage1 = executed.snapshot.stage_fragments[0]
    assert (stage1.stage_id, stage1.available_slot) == (1, 2)


def test_final_vnf_creates_next_slot_completion_event() -> None:
    batch = BatchRecord("b", 0, 0.0, 10.0, total_input_equivalent_bits=4.0)
    fragment = QueueFragment("f", "b", 0, 1, 0, None, 4.0, 0)
    manager = QueueStateManager(
        StageFlowConfig((0.5, 1.0)),
        slot_seconds=1.0,
        initial_batches=(batch,),
        initial_stage_fragments=(fragment,),
    )
    execute = AllocationOperation(
        QueueKey.stage(1, 0, None), "execute", 2.0
    )

    result = manager.commit_allocation(
        plan(manager, (execute,)),
        context(queue_version=0, current_slot=0),
    )

    assert result.accepted is True
    assert result.snapshot.batches[0].completion_slot is None
    assert result.snapshot.completion_events[0].completion_slot == 1
    completed = manager.begin_slot(1)
    assert completed.batches[0].completion_slot == 1


def test_queue_commit_uses_configured_physical_solver_tolerance() -> None:
    batch = BatchRecord("b", 0, 0.0, 10.0, total_input_equivalent_bits=4.0)
    fragment = QueueFragment("f", "b", 0, 0, 0, None, 4.0, 0)
    manager = QueueStateManager(
        StageFlowConfig((1.0,)),
        slot_seconds=1.0,
        initial_batches=(batch,),
        initial_stage_fragments=(fragment,),
        flow_absolute_tolerance_bits=1e-3,
    )
    within_tolerance = AllocationOperation(
        QueueKey.stage(0, 0, None), "execute", 4.0005
    )

    accepted = manager.commit_allocation(
        plan(manager, (within_tolerance,)),
        context(queue_version=0, current_slot=0),
    )

    assert accepted.accepted is True


def test_wired_forwarding_has_at_least_one_slot_propagation() -> None:
    batch = BatchRecord("b", 0, 0.0, 10.0, total_input_equivalent_bits=3.0)
    fragment = QueueFragment("f", "b", 0, 0, 0, None, 3.0, 0)
    manager = QueueStateManager(
        StageFlowConfig((1.0,)),
        slot_seconds=1.0,
        initial_batches=(batch,),
        initial_stage_fragments=(fragment,),
    )
    forward = AllocationOperation(
        QueueKey.stage(0, 0, None),
        "forward",
        3.0,
        routing_target_node=2,
        destination_node_id=1,
        link_id=7,
        propagation_slots=0,
    )

    result = manager.commit_allocation(
        plan(manager, (forward,)),
        context(queue_version=0, current_slot=0),
    )

    assert result.accepted is True
    assert result.snapshot.stage_fragments == ()
    transit = result.snapshot.in_transit[0]
    assert transit.arrival_slot == 1
    assert transit.fragment.routing_target_node == 2
    arrived = manager.begin_slot(1)
    assert arrived.stage_fragments[0].location == 1


def test_edf_uses_fragment_first_then_fixed_outlet_order() -> None:
    manager = QueueStateManager(StageFlowConfig((1.0,)), slot_seconds=1.0)
    manager.admit_batch("urgent", 0, 0.0, 5.0, 6.0)
    manager.admit_batch("late", 0, 0.0, 9.0, 6.0)
    operations = (
        AllocationOperation(QueueKey.uplink(0), "uplink", 4.0, destination_node_id=1),
        AllocationOperation(QueueKey.uplink(0), "uplink", 4.0, destination_node_id=0),
    )

    result = manager.commit_allocation(
        plan(manager, operations),
        context(queue_version=manager.snapshot().version, current_slot=0),
    )

    assert result.accepted is True
    served = sorted(
        (item.batch_id, item.location, item.input_equivalent_bits)
        for item in result.snapshot.stage_fragments
    )
    assert served == [
        ("late", 1, 2.0),
        ("urgent", 0, 4.0),
        ("urgent", 1, 2.0),
    ]


def test_bound_fragment_cannot_be_redirected() -> None:
    batch = BatchRecord("b", 0, 0.0, 10.0, total_input_equivalent_bits=2.0)
    fragment = QueueFragment("f", "b", 0, 0, 0, 2, 2.0, 0)
    manager = QueueStateManager(
        StageFlowConfig((1.0,)),
        slot_seconds=1.0,
        initial_batches=(batch,),
        initial_stage_fragments=(fragment,),
    )
    redirect = AllocationOperation(
        QueueKey.stage(0, 0, 2),
        "forward",
        2.0,
        routing_target_node=1,
        destination_node_id=1,
        link_id=4,
    )

    before = manager.snapshot()
    result = manager.commit_allocation(
        plan(manager, (redirect,)),
        context(queue_version=0, current_slot=0),
    )

    assert result.code == "INVALID_STAGE_TRANSITION"
    assert manager.snapshot() == before


def test_failed_node_cannot_execute_but_can_forward() -> None:
    batch = BatchRecord("b", 0, 0.0, 10.0, total_input_equivalent_bits=2.0)
    fragment = QueueFragment("f", "b", 0, 0, 0, None, 2.0, 0)
    manager = QueueStateManager(
        StageFlowConfig((1.0,)),
        slot_seconds=1.0,
        initial_batches=(batch,),
        initial_stage_fragments=(fragment,),
    )
    down = replace(
        context(queue_version=0, current_slot=0),
        effective_node_up={0: False, 1: True, 2: True},
    )

    execute_result = manager.commit_allocation(
        plan(
            manager,
            (AllocationOperation(QueueKey.stage(0, 0, None), "execute", 2.0),),
        ),
        down,
    )
    assert execute_result.code == "INVALID_ALLOCATION_PLAN"

    forward_result = manager.commit_allocation(
        plan(
            manager,
            (
                AllocationOperation(
                    QueueKey.stage(0, 0, None),
                    "forward",
                    2.0,
                    routing_target_node=1,
                    destination_node_id=1,
                    link_id=1,
                ),
            ),
        ),
        down,
    )
    assert forward_result.accepted is True


@pytest.mark.parametrize("mode", ["stale", "overflow"])
def test_invalid_plan_is_rejected_atomically(mode: str) -> None:
    manager = QueueStateManager(StageFlowConfig((1.0,)), slot_seconds=1.0)
    manager.admit_batch("b", 0, 0.0, 10.0, 1.0)
    operation = AllocationOperation(
        QueueKey.uplink(0), "uplink", 2.0, destination_node_id=0
    )
    current_plan = plan(manager, (operation,))
    current_context = context(
        queue_version=manager.snapshot().version,
        current_slot=0,
    )
    if mode == "stale":
        current_plan = replace(current_plan, expected_queue_version=999)

    before = manager.snapshot()
    result = manager.commit_allocation(current_plan, current_context)

    assert result.code == (
        "STALE_SNAPSHOT" if mode == "stale" else "FLOW_EXCEEDS_AVAILABLE_QUEUE"
    )
    assert manager.snapshot() == before


def test_queue_manager_clone_from_snapshot_is_exact_and_independent() -> None:
    manager = QueueStateManager(StageFlowConfig((1.0,)), slot_seconds=1.0)
    manager.admit_batch("batch-clone", 0, 0.0, 10.0, 100.0)
    source = manager.snapshot()

    clone = QueueStateManager.from_snapshot(
        manager.flow_config,
        source,
        slot_seconds=manager.slot_seconds,
    )

    assert clone.snapshot() == source
    clone.admit_batch("batch-only-in-clone", 0, 0.0, 10.0, 10.0)
    assert clone.snapshot() != manager.snapshot()
    assert len(manager.snapshot().batches) == 1


def test_commit_absorbs_numeric_tail_into_last_successful_operation() -> None:
    manager = QueueStateManager(
        StageFlowConfig((1.0,)),
        slot_seconds=1.0,
        flow_absolute_tolerance_bits=10.0,
    )
    manager.admit_batch("batch", 0, 0.0, 20.0, 100.0)

    result = manager.commit_allocation(
        plan(
            manager,
            (
                AllocationOperation(
                    QueueKey.uplink(0),
                    "uplink",
                    95.0,
                    destination_node_id=0,
                ),
            ),
        ),
        context(queue_version=manager.snapshot().version, current_slot=0),
    )

    assert result.accepted
    assert result.snapshot.uplink_fragments == ()
    assert sum(
        item.input_equivalent_bits for item in result.snapshot.stage_fragments
    ) == 100.0


def test_final_stage_tail_completes_batch_with_shared_tolerance() -> None:
    manager = QueueStateManager(
        StageFlowConfig((1.0,)),
        slot_seconds=1.0,
        initial_batches=(BatchRecord("batch", 0, 0.0, 20.0, 100.0),),
        initial_stage_fragments=(
            QueueFragment("final", "batch", 0, 0, 0, None, 100.0, 0),
        ),
        flow_absolute_tolerance_bits=10.0,
    )

    result = manager.commit_allocation(
        plan(
            manager,
            (
                AllocationOperation(
                    QueueKey.stage(0, 0, None),
                    "execute",
                    95.0,
                ),
            ),
        ),
        context(queue_version=manager.snapshot().version, current_slot=0),
    )
    completed = manager.begin_slot(1).batches[0]

    assert result.accepted
    assert completed.completed_input_equivalent_bits == 100.0
    assert completed.completion_slot == 1


def test_numeric_tail_absorption_never_exceeds_group_tolerance() -> None:
    manager = QueueStateManager(
        StageFlowConfig((1.0,)),
        slot_seconds=1.0,
        flow_absolute_tolerance_bits=10.0,
    )
    for index in range(3):
        manager.admit_batch(f"batch-{index}", 0, 0.0, 20.0 + index, 100.0)
    operation = AllocationOperation(
        QueueKey.uplink(0),
        "uplink",
        95.0,
        destination_node_id=0,
    )

    result = manager.commit_allocation(
        plan(manager, (operation, operation, operation)),
        context(queue_version=manager.snapshot().version, current_slot=0),
    )

    assert result.accepted
    assert sum(
        item.input_equivalent_bits for item in result.snapshot.stage_fragments
    ) == 295.0
    assert sum(
        item.input_equivalent_bits for item in result.snapshot.uplink_fragments
    ) == 5.0
