"""测试阶段 A 单时隙事件顺序。"""

from src.config import load_config
from src.cost_ledger import CostLedger
from src.failure_process import ScriptedFailureProcess
from src.instance_lifecycle import (
    DeploymentTarget,
    InstanceBatch,
    InstanceLifecycleManager,
    LifecycleStatus,
    LifecycleDeploymentPlan,
)
from src.orchestration_config import load_phase_a_config
from src.orchestration_core import PhaseASlotCoordinator
from src.topology import build_linear_topology


def build_coordinator() -> PhaseASlotCoordinator:
    config = load_config("configs/debug.yaml")
    topology = build_linear_topology(config)
    failure = ScriptedFailureProcess(
        topology,
        down_nodes_by_slot={3: {0}},
    )
    lifecycle = InstanceLifecycleManager(
        config=load_phase_a_config(config),
        function_memory_mb={
            item["function_id"]: float(item["memory_mb"])
            for item in config["rl_scenario"]["functions"]
        },
        initial_batches=(
            InstanceBatch(
                "starting",
                0,
                0,
                LifecycleStatus.STARTING,
                2,
                3,
                9,
            ),
        ),
    )
    return PhaseASlotCoordinator(failure, lifecycle, CostLedger())


def test_slot_order_completes_start_then_applies_failure() -> None:
    coordinator = build_coordinator()

    result = coordinator.begin_slot(3)

    assert 0 in result.failure_snapshot.newly_unavailable_node_ids
    assert result.lifecycle_snapshot.warm_count(0, 0) == 0
    assert result.lifecycle_snapshot.starting_count(0, 0) == 0


def test_recovered_node_stays_empty_until_a_new_plan() -> None:
    coordinator = build_coordinator()
    coordinator.begin_slot(3)

    recovered = coordinator.begin_slot(4)

    assert 0 in recovered.failure_snapshot.newly_available_node_ids
    assert all(batch.node_id != 0 for batch in recovered.lifecycle_snapshot.batches)


def test_coordinator_bills_creation_cold_start_and_running_once() -> None:
    config = load_config("configs/debug.yaml")
    topology = build_linear_topology(config)
    lifecycle = InstanceLifecycleManager(
        config=load_phase_a_config(config),
        function_memory_mb={
            item["function_id"]: float(item["memory_mb"])
            for item in config["rl_scenario"]["functions"]
        },
    )
    coordinator = PhaseASlotCoordinator(
        ScriptedFailureProcess(topology),
        lifecycle,
        CostLedger(),
    )
    boundary = coordinator.begin_slot(0)
    plan = LifecycleDeploymentPlan(
        expected_lifecycle_version=boundary.lifecycle_snapshot.version,
        expected_failure_version=boundary.failure_snapshot.version,
        current_slot=0,
        targets=(DeploymentTarget(0, 0, 2, 5),),
    )

    commit = coordinator.commit_deployment(plan)
    coordinator.finalize_slot_costing()

    assert commit.accepted is True
    assert [entry.cost_type for entry in coordinator.cost_ledger.entries] == [
        "deployment",
        "cold",
        "running",
    ]
    assert [entry.amount for entry in coordinator.cost_ledger.entries] == [
        2.0,
        0.6,
        0.002,
    ]
    assert all(
        entry.slow_frame_index == 0
        for entry in coordinator.cost_ledger.entries
    )


def test_slot_costing_cannot_be_applied_twice() -> None:
    coordinator = build_coordinator()
    coordinator.begin_slot(0)
    coordinator.finalize_slot_costing()

    import pytest

    with pytest.raises(RuntimeError, match="已经完成计费"):
        coordinator.finalize_slot_costing()
