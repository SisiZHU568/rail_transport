"""测试两阶段 CLARABEL 快层的无线、计算和词典序行为。"""

import cvxpy as cp
import pytest

from src.failure_process import FailureSnapshot
from src.fast_resource_model import (
    FastResourceConfig,
    NetworkSnapshot,
    NodeFastResource,
    VNFComputeResource,
)
from src.fast_resource_optimizer import FastResourceOptimizer
from src.instance_lifecycle import InstanceBatch, LifecycleSnapshot, LifecycleStatus
from src.queue_state import BatchRecord, QueueFragment, QueueSnapshot, StageFlowConfig


def config(*, max_cycles: float = 1e9) -> FastResourceConfig:
    return FastResourceConfig(
        slot_seconds=1.0,
        noise_psd_watt_per_hz=1e-12,
        energy_price_per_joule=0.01,
        absolute_lex_tolerance=1e-7,
        relative_lex_tolerance=1e-7,
        residual_tolerance=1e-5,
        active_time_tolerance_seconds=1e-9,
        solver_name="CLARABEL",
        max_iterations=200,
        allow_optimal_inaccurate=False,
        nodes={0: NodeFastResource(0, max_cycles, 1, 1e-27, 0.1, 0.0)},
        vnfs={(0, 0): VNFComputeResource(0, 0, 1000.0, max_cycles)},
    )


def lifecycle(*, warm: bool = True) -> LifecycleSnapshot:
    batches = (
        (InstanceBatch("warm", 0, 0, LifecycleStatus.WARM, 1, 0, 10),)
        if warm
        else (InstanceBatch("cold", 0, 0, LifecycleStatus.STARTING, 1, 2, 10),)
    )
    return LifecycleSnapshot(2, 0, batches, {0: 1.0})


def failure() -> FailureSnapshot:
    return FailureSnapshot(0, {0: True}, {0: True}, {0: True}, 3)


def network(*, channel_gain: float = 1e-6) -> NetworkSnapshot:
    return NetworkSnapshot(4, 0, 0, channel_gain, 1e6, 1.0, ())


def queue(
    *,
    uplink_bits: float = 0.0,
    stage_bits: float = 0.0,
) -> QueueSnapshot:
    total = uplink_bits + stage_bits
    batches = (
        (BatchRecord("b", 0, 0.0, 5.0, total_input_equivalent_bits=total),)
        if total > 0.0
        else ()
    )
    uplink = (
        (QueueFragment("u", "b", 0, -1, None, None, uplink_bits, 0),)
        if uplink_bits > 0.0
        else ()
    )
    stage = (
        (QueueFragment("s", "b", 0, 0, 0, None, stage_bits, 0),)
        if stage_bits > 0.0
        else ()
    )
    return QueueSnapshot(1, 0, batches, uplink, stage, (), ())


def test_problem_is_dcp_and_records_two_solver_stages() -> None:
    optimizer = FastResourceOptimizer(config(), StageFlowConfig((1.0,)))

    result = optimizer.solve(queue(uplink_bits=1e5, stage_bits=1e5), lifecycle(), failure(), network())

    assert result.succeeded is True
    assert result.is_dcp is True
    assert result.primary_status == cp.OPTIMAL
    assert result.secondary_status == cp.OPTIMAL
    assert result.primary_solve_seconds >= 0.0
    assert result.secondary_solve_seconds >= 0.0
    assert result.maximum_residual <= 1e-5


def test_zero_queue_returns_empty_plan_without_solver() -> None:
    result = FastResourceOptimizer(config(), StageFlowConfig((1.0,))).solve(
        queue(), lifecycle(), failure(), network()
    )

    assert result.succeeded is True
    assert result.primary_status == "not_run"
    assert result.plan.operations == ()


def test_zero_channel_leaves_finite_uplink_shortfall() -> None:
    result = FastResourceOptimizer(config(), StageFlowConfig((1.0,))).solve(
        queue(uplink_bits=1e5), lifecycle(), failure(), network(channel_gain=0.0)
    )

    assert result.succeeded is True
    assert result.uplink_service_bits == pytest.approx(0.0, abs=1e-3)
    assert result.service_shortfall_equivalent_bits == pytest.approx(1e5, rel=1e-5)


def test_starting_instance_cannot_execute() -> None:
    result = FastResourceOptimizer(config(), StageFlowConfig((1.0,))).solve(
        queue(stage_bits=1e5), lifecycle(warm=False), failure(), network()
    )

    assert result.succeeded is True
    assert result.executed_physical_bits == pytest.approx(0.0, abs=1e-3)
    assert result.service_shortfall_equivalent_bits == pytest.approx(1e5, rel=1e-5)


def test_capacity_shortage_remains_feasible_and_primary_is_preserved() -> None:
    result = FastResourceOptimizer(
        config(max_cycles=5e7), StageFlowConfig((1.0,))
    ).solve(queue(stage_bits=1e5), lifecycle(), failure(), network())

    assert result.succeeded is True
    assert result.executed_physical_bits == pytest.approx(5e4, rel=1e-3)
    assert result.service_shortfall_equivalent_bits == pytest.approx(5e4, rel=1e-3)
    assert result.secondary_primary_value <= result.primary_optimum + result.lex_tolerance + 1e-5


def test_solver_failure_returns_no_plan_and_no_fallback(monkeypatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise cp.SolverError("forced")

    monkeypatch.setattr(cp.Problem, "solve", fail)
    result = FastResourceOptimizer(config(), StageFlowConfig((1.0,))).solve(
        queue(stage_bits=1.0), lifecycle(), failure(), network()
    )

    assert result.succeeded is False
    assert result.code == "FAST_SOLVER_FAILURE"
    assert result.plan is None
