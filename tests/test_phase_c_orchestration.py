"""测试新快层求解器与阶段 B 原子提交的闭环。"""

from src.failure_process import FailureSnapshot
from src.fast_resource_model import (
    FastResourceConfig,
    NetworkSnapshot,
    NodeFastResource,
    VNFComputeResource,
)
from src.fast_resource_optimizer import FastResourceOptimizer
from src.instance_lifecycle import InstanceBatch, LifecycleSnapshot, LifecycleStatus
from src.orchestration_core import solve_and_commit_fast_resources
from src.queue_manager import QueueStateManager
from src.queue_state import BatchRecord, QueueFragment, StageFlowConfig


def test_solver_plan_is_committed_without_fallback() -> None:
    flow = StageFlowConfig((1.0,))
    batch = BatchRecord("b", 0, 0.0, 5.0, total_input_equivalent_bits=1e5)
    fragment = QueueFragment("f", "b", 0, 0, 0, None, 1e5, 0)
    queue = QueueStateManager(
        flow,
        slot_seconds=1.0,
        initial_batches=(batch,),
        initial_stage_fragments=(fragment,),
    )
    lifecycle = LifecycleSnapshot(
        2, 0, (InstanceBatch("w", 0, 0, LifecycleStatus.WARM, 1, 0, 10),), {0: 1.0}
    )
    failure = FailureSnapshot(0, {0: True}, {0: True}, {0: True}, 3)
    network = NetworkSnapshot(4, 0, 0, 1e-6, 1e6, 1.0, ())
    config = FastResourceConfig(
        1.0, 1e-12, 0.01, 1e-7, 1e-7, 1e-5, 1e-9,
        "CLARABEL", 200, False,
        {0: NodeFastResource(0, 1e9, 1, 1e-27, 0.1, 0.0)},
        {(0, 0): VNFComputeResource(0, 0, 1000.0, 1e9)},
    )

    result = solve_and_commit_fast_resources(
        FastResourceOptimizer(config, flow), queue, lifecycle, failure, network
    )

    assert result.optimization.succeeded is True
    assert result.commit is not None and result.commit.accepted is True
    assert queue.snapshot().completion_events[0].completion_slot == 1


def test_solver_failure_does_not_commit_any_service(monkeypatch) -> None:
    import cvxpy as cp

    flow = StageFlowConfig((1.0,))
    batch = BatchRecord("b", 0, 0.0, 5.0, total_input_equivalent_bits=1.0)
    fragment = QueueFragment("f", "b", 0, 0, 0, None, 1.0, 0)
    queue = QueueStateManager(flow, slot_seconds=1.0, initial_batches=(batch,), initial_stage_fragments=(fragment,))
    lifecycle = LifecycleSnapshot(2, 0, (InstanceBatch("w", 0, 0, LifecycleStatus.WARM, 1, 0, 10),), {0: 1.0})
    failure = FailureSnapshot(0, {0: True}, {0: True}, {0: True}, 3)
    network = NetworkSnapshot(4, 0, 0, 1e-6, 1e6, 1.0, ())
    config = FastResourceConfig(1.0, 1e-12, 0.01, 1e-7, 1e-7, 1e-5, 1e-9, "CLARABEL", 200, False, {0: NodeFastResource(0, 1e9, 1, 1e-27, 0.1, 0.0)}, {(0, 0): VNFComputeResource(0, 0, 1000.0, 1e9)})
    before = queue.snapshot()
    monkeypatch.setattr(cp.Problem, "solve", lambda *args, **kwargs: (_ for _ in ()).throw(cp.SolverError("forced")))

    result = solve_and_commit_fast_resources(FastResourceOptimizer(config, flow), queue, lifecycle, failure, network)

    assert result.optimization.code == "FAST_SOLVER_FAILURE"
    assert result.commit is None
    assert queue.snapshot() == before
