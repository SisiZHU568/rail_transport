"""测试实例批次生命周期、边界时隙和原子部署提交。"""

from dataclasses import replace

from src.config import load_config
from src.failure_process import ScriptedFailureProcess
from src.instance_lifecycle import (
    DeploymentTarget,
    InstanceBatch,
    InstanceLifecycleManager,
    LifecycleDeploymentPlan,
    LifecycleStatus,
)
from src.orchestration_config import load_phase_a_config
from src.topology import build_linear_topology


def build_manager(
    *,
    batches: tuple[InstanceBatch, ...] = (),
) -> tuple[InstanceLifecycleManager, object]:
    config = load_config("configs/debug.yaml")
    topology = build_linear_topology(config)
    phase_a = load_phase_a_config(config)
    manager = InstanceLifecycleManager(
        config=phase_a,
        function_memory_mb={
            item["function_id"]: float(item["memory_mb"])
            for item in config["rl_scenario"]["functions"]
        },
        initial_batches=batches,
    )
    failure = ScriptedFailureProcess(topology).state_for_slot(0)
    return manager, failure


def test_zero_cold_start_is_warm_immediately() -> None:
    manager, failure = build_manager()
    zero_cold = replace(
        manager.config.deployment_pairs[(0, 0)],
        cold_start_seconds=0.0,
    )
    manager.config = replace(
        manager.config,
        deployment_pairs={
            **manager.config.deployment_pairs,
            (0, 0): zero_cold,
        },
    )
    plan = LifecycleDeploymentPlan(
        expected_lifecycle_version=manager.snapshot().version,
        expected_failure_version=failure.version,
        current_slot=0,
        targets=(DeploymentTarget(0, 0, 2, 5),),
    )

    result = manager.commit_deployment(plan, failure)

    assert result.accepted is True
    assert result.snapshot.warm_count(0, 0) == 2
    assert result.snapshot.starting_count(0, 0) == 0
    assert result.snapshot.batches[0].retention_deadline_slot == 5


def test_ready_and_retention_boundaries_use_greater_or_equal() -> None:
    batch = InstanceBatch(
        batch_id="starting-1",
        function_id=0,
        node_id=0,
        status=LifecycleStatus.STARTING,
        count=2,
        ready_slot=5,
        retention_deadline_slot=8,
    )
    manager, _ = build_manager(batches=(batch,))

    before = manager.advance_to_slot(4)
    at_ready = manager.advance_to_slot(5)

    assert before.starting_count(0, 0) == 2
    assert at_ready.warm_count(0, 0) == 2
    assert at_ready.locked_count(0, 0, current_slot=7) == 2
    assert at_ready.locked_count(0, 0, current_slot=8) == 0


def test_partial_refresh_splits_batch_without_extending_excess() -> None:
    batch = InstanceBatch(
        batch_id="warm-5",
        function_id=0,
        node_id=0,
        status=LifecycleStatus.WARM,
        count=5,
        ready_slot=0,
        retention_deadline_slot=10,
    )
    manager, failure = build_manager(batches=(batch,))
    manager.advance_to_slot(5)
    failure = replace(failure, version=5, time_slot=5)
    plan = LifecycleDeploymentPlan(
        expected_lifecycle_version=manager.snapshot().version,
        expected_failure_version=failure.version,
        current_slot=5,
        targets=(DeploymentTarget(0, 0, 3, 20),),
    )

    result = manager.commit_deployment(plan, failure)

    assert result.accepted is True
    assert sorted(
        (batch.count, batch.retention_deadline_slot)
        for batch in result.snapshot.batches
    ) == [(2, 10), (3, 25)]


def test_starting_instances_count_toward_target_and_memory() -> None:
    manager, failure = build_manager()
    first = manager.commit_deployment(
        LifecycleDeploymentPlan(
            expected_lifecycle_version=0,
            expected_failure_version=failure.version,
            current_slot=0,
            targets=(DeploymentTarget(0, 0, 2, 5),),
        ),
        failure,
    )
    second = manager.commit_deployment(
        LifecycleDeploymentPlan(
            expected_lifecycle_version=first.snapshot.version,
            expected_failure_version=failure.version,
            current_slot=0,
            targets=(DeploymentTarget(0, 0, 2, 5),),
        ),
        failure,
    )

    assert first.snapshot.starting_count(0, 0) == 2
    assert second.snapshot.active_count(0, 0) == 2
    assert second.snapshot.memory_used_mb_by_node[0] == 512.0


def test_stale_plan_has_no_partial_effect() -> None:
    manager, failure = build_manager()
    before = manager.snapshot()
    stale = LifecycleDeploymentPlan(
        expected_lifecycle_version=before.version + 1,
        expected_failure_version=failure.version,
        current_slot=0,
        targets=(DeploymentTarget(0, 0, 2, 1),),
    )

    result = manager.commit_deployment(stale, failure)

    assert result.code == "STALE_SNAPSHOT"
    assert manager.snapshot() == before


def test_invalid_count_and_memory_overflow_are_atomic() -> None:
    manager, failure = build_manager()
    before = manager.snapshot()
    invalid = LifecycleDeploymentPlan(
        expected_lifecycle_version=before.version,
        expected_failure_version=failure.version,
        current_slot=0,
        targets=(DeploymentTarget(0, 0, 99, 1),),
    )

    result = manager.commit_deployment(invalid, failure)

    assert result.code == "INVALID_DEPLOYMENT_PLAN"
    assert manager.snapshot() == before


def test_failure_destroys_all_batches_and_recovery_stays_empty() -> None:
    batch = InstanceBatch("warm", 0, 0, LifecycleStatus.WARM, 2, 0, 9)
    manager, _ = build_manager(batches=(batch,))
    config = load_config("configs/debug.yaml")
    topology = build_linear_topology(config)
    process = ScriptedFailureProcess(
        topology,
        down_nodes_by_slot={1: {0}},
    )
    process.state_for_slot(0)

    failed = manager.apply_failure_snapshot(process.state_for_slot(1))
    recovered = manager.apply_failure_snapshot(process.state_for_slot(2))

    assert failed.active_count(0, 0) == 0
    assert recovered.active_count(0, 0) == 0


def test_locked_instances_cannot_be_deleted_or_shortened() -> None:
    batch = InstanceBatch("locked", 0, 0, LifecycleStatus.WARM, 3, 0, 10)
    manager, failure = build_manager(batches=(batch,))
    manager.advance_to_slot(5)
    failure = replace(failure, version=5, time_slot=5)

    result = manager.commit_deployment(
        LifecycleDeploymentPlan(
            expected_lifecycle_version=manager.snapshot().version,
            expected_failure_version=5,
            current_slot=5,
            targets=(DeploymentTarget(0, 0, 0, 0),),
        ),
        failure,
    )

    assert result.accepted is True
    assert result.snapshot.active_count(0, 0) == 3
    assert result.snapshot.batches[0].retention_deadline_slot == 10


def test_starting_refresh_uses_ready_slot_as_retention_origin() -> None:
    batch = InstanceBatch("starting", 0, 0, LifecycleStatus.STARTING, 1, 8, 9)
    manager, failure = build_manager(batches=(batch,))
    manager.advance_to_slot(5)
    failure = replace(failure, version=5, time_slot=5)

    result = manager.commit_deployment(
        LifecycleDeploymentPlan(
            expected_lifecycle_version=manager.snapshot().version,
            expected_failure_version=5,
            current_slot=5,
            targets=(DeploymentTarget(0, 0, 1, 5),),
        ),
        failure,
    )

    assert result.snapshot.batches[0].retention_deadline_slot == 13


def test_unhealthy_target_and_memory_overflow_are_atomic() -> None:
    manager, failure = build_manager()
    before = manager.snapshot()
    unhealthy = replace(
        failure,
        effective_node_up={**failure.effective_node_up, 0: False},
    )
    unavailable = manager.commit_deployment(
        LifecycleDeploymentPlan(0, unhealthy.version, 0, (DeploymentTarget(0, 0, 1, 5),)),
        unhealthy,
    )
    assert unavailable.code == "UNHEALTHY_TARGET_NODE"
    assert manager.snapshot() == before

    tiny_node = replace(manager.config.node_resources[0], memory_capacity_mb=100.0)
    manager.config = replace(
        manager.config,
        node_resources={**manager.config.node_resources, 0: tiny_node},
    )
    overflow = manager.commit_deployment(
        LifecycleDeploymentPlan(0, failure.version, 0, (DeploymentTarget(0, 0, 1, 5),)),
        failure,
    )
    assert overflow.code == "MEMORY_CAPACITY_EXCEEDED"
    assert manager.snapshot() == before


def test_preview_deployment_does_not_modify_manager() -> None:
    manager, failure = build_manager()
    before = manager.snapshot()
    plan = LifecycleDeploymentPlan(
        expected_lifecycle_version=before.version,
        expected_failure_version=failure.version,
        current_slot=0,
        targets=(DeploymentTarget(0, 0, 2, 5),),
    )

    preview = manager.preview_deployment(plan, failure)

    assert preview.accepted is True
    assert preview.snapshot.active_count(0, 0) == 2
    assert manager.snapshot() == before
