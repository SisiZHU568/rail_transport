"""测试阶段 B 固定边界顺序和跨模块版本提交。"""

from src.config import load_config
from src.cost_ledger import CostLedger
from src.failure_process import ScriptedFailureProcess
from src.instance_lifecycle import (
    InstanceBatch,
    InstanceLifecycleManager,
    LifecycleStatus,
)
from src.orchestration_config import load_phase_a_config
from src.orchestration_core import PhaseASlotCoordinator, PhaseBSlotCoordinator
from src.queue_manager import AllocationOperation, FastAllocationPlan, QueueKey, QueueStateManager
from src.queue_state import BatchRecord, InTransitRecord, QueueFragment, StageFlowConfig
from src.topology import build_linear_topology


def build_phase_a(
    initial_batches: tuple[InstanceBatch, ...] = (),
) -> PhaseASlotCoordinator:
    config = load_config("configs/debug.yaml")
    phase_a_config = load_phase_a_config(config)
    lifecycle = InstanceLifecycleManager(
        config=phase_a_config,
        function_memory_mb={
            item["function_id"]: float(item["memory_mb"])
            for item in config["rl_scenario"]["functions"]
        },
        initial_batches=initial_batches,
    )
    return PhaseASlotCoordinator(
        ScriptedFailureProcess(build_linear_topology(config)),
        lifecycle,
        CostLedger(),
    )


def test_arriving_transit_is_unbound_only_after_arrival() -> None:
    batch = BatchRecord("b", 0, 0.0, 10.0, total_input_equivalent_bits=2.0)
    fragment = QueueFragment("f", "b", 0, 0, 1, 2, 2.0, 1)
    transit = InTransitRecord("t", fragment, 0, 1, 7, 0, 1, 2.0)
    queue = QueueStateManager(
        StageFlowConfig((1.0,)),
        slot_seconds=1.0,
        initial_batches=(batch,),
        initial_in_transit=(transit,),
    )
    coordinator = PhaseBSlotCoordinator(build_phase_a(), queue)
    original = queue.snapshot().in_transit[0]

    result = coordinator.begin_slot(1, network_version=8)

    assert original == transit
    assert result.queue_snapshot.in_transit == ()
    assert result.queue_snapshot.stage_fragments[0].routing_target_node is None


def test_completion_event_is_committed_before_same_slot_sla_audit() -> None:
    batch = BatchRecord("b", 0, 0.0, 1.0, total_input_equivalent_bits=1.0)
    fragment = QueueFragment("f", "b", 0, 0, 0, None, 1.0, 0)
    queue = QueueStateManager(
        StageFlowConfig((1.0,)),
        slot_seconds=1.0,
        initial_batches=(batch,),
        initial_stage_fragments=(fragment,),
    )
    coordinator = PhaseBSlotCoordinator(
        build_phase_a(
            (InstanceBatch("warm", 0, 0, LifecycleStatus.WARM, 1, 0, 5),)
        ),
        queue,
    )
    coordinator.begin_slot(0, network_version=8)
    current = queue.snapshot()
    plan = FastAllocationPlan(
        current.version,
        coordinator.lifecycle_snapshot.version,
        coordinator.failure_snapshot.version,
        8,
        0,
        (AllocationOperation(QueueKey.stage(0, 0, None), "execute", 1.0),),
    )
    assert coordinator.commit_allocation(plan).accepted is True

    result = coordinator.begin_slot(1, network_version=9)

    assert result.queue_snapshot.batches[0].completion_slot == 1
    assert result.sla_report.new_violation_count == 0
    assert result.sla_report.on_time_completed_count == 1


def test_commit_rejects_network_version_change_without_queue_mutation() -> None:
    queue = QueueStateManager(StageFlowConfig((1.0,)), slot_seconds=1.0)
    queue.admit_batch("b", 0, 0.0, 10.0, 1.0)
    coordinator = PhaseBSlotCoordinator(build_phase_a(), queue)
    coordinator.begin_slot(0, network_version=8)
    snapshot = queue.snapshot()
    stale = FastAllocationPlan(
        snapshot.version,
        coordinator.lifecycle_snapshot.version,
        coordinator.failure_snapshot.version,
        7,
        0,
        (AllocationOperation(QueueKey.uplink(0), "uplink", 1.0, destination_node_id=0),),
    )

    result = coordinator.commit_allocation(stale)

    assert result.code == "STALE_SNAPSHOT"
    assert queue.snapshot() == snapshot
