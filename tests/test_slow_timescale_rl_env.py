"""测试重建后的双时间尺度 DDQN 慢层环境。"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from src.config import load_config
from src.failure_process import ScriptedFailureProcess
from src.failure_risk_prediction import ConstantFailureRiskProvider
from src.fast_slot_executor import (
    FastSlotExecutionResult,
    FastSlotExecutor,
    FastSlotInput,
)
from src.rl_scenario import build_rl_environment
from src.slow_timescale_rl_env import SlowTimescaleRLEnvironment


class SpyFastSlotExecutor:
    """记录环境传给共享快层执行器的输入，同时保留真实执行结果。"""

    def __init__(self, delegate: FastSlotExecutor) -> None:
        self.delegate = delegate
        self.inputs: list[FastSlotInput] = []
        self.results: list[FastSlotExecutionResult] = []

    @property
    def call_count(self) -> int:
        return len(self.inputs)

    def execute(self, slot_input: FastSlotInput) -> FastSlotExecutionResult:
        self.inputs.append(slot_input)
        result = self.delegate.execute(slot_input)
        self.results.append(result)
        return result


def build_test_environment(
    *,
    slow_period_slots: int = 3,
    request_trace: list[int] | None = None,
    down_nodes_by_slot: dict[int, set[int]] | None = None,
    with_spy: bool = False,
) -> tuple[SlowTimescaleRLEnvironment, SpyFastSlotExecutor | None]:
    """使用五个 MEC、一个中心云和三个 VNF 创建可复现测试环境。"""

    config = deepcopy(load_config("configs/debug.yaml"))
    config["train"]["speed_mps"] = 500.0
    config["two_timescale"]["slow_period_slots"] = slow_period_slots
    config["integrated_simulation"]["request_trace"] = (
        request_trace
        if request_trace is not None
        else [1, 1, 1, 1]
    )
    config["integrated_simulation"]["repeat_request_trace"] = False

    # 第 14 步会把这些参数写入正式配置文件；本测试提前锁定接口。
    config["rl_environment"].update(
        {
            "workload_prediction": {
                "lookback_slots": 10,
                "baseline_request_rate": 1.0,
            },
            "state_normalization": {
                "maximum_request_rate": 5.0,
                "maximum_network_delay_ms": 200.0,
                "maximum_input_size_mb": 10.0,
            },
            "cost_rates": {
                "edge_cpu_cost_per_unit": 0.01,
                "edge_memory_cost_per_mb_second": 0.001,
                "cloud_cpu_cost_per_unit": 0.05,
                "cloud_memory_cost_per_mb_second": 0.005,
                "cold_start_cost_per_ms": 0.10,
                "maximum_window_cost": 10000.0,
            },
        }
    )

    environment = build_rl_environment(config)

    # 用脚本化故障过程替换随机过程，使每个断言都完全可复现。
    def failure_process_builder(seed: int) -> ScriptedFailureProcess:
        _ = seed
        return ScriptedFailureProcess(
            topology=environment.topology,
            down_nodes_by_slot=down_nodes_by_slot,
        )

    environment.failure_process_builder = failure_process_builder
    environment.failure_risk_provider = ConstantFailureRiskProvider(risk=0.01)

    spy: SpyFastSlotExecutor | None = None
    if with_spy:
        spy = SpyFastSlotExecutor(environment.fast_slot_executor)
        environment.fast_slot_executor = spy  # type: ignore[assignment]

    return environment, spy


def first_valid_action(action_mask: np.ndarray) -> int:
    """返回掩码中的第一个合法动作，便于终止性测试稳定运行。"""

    valid_ids = np.flatnonzero(action_mask)
    if len(valid_ids) == 0:
        raise AssertionError("测试场景没有任何合法动作。")
    return int(valid_ids[0])


def test_reset_returns_v2_state_and_action_mask() -> None:
    """reset 必须返回 78 维状态和 12 维合法动作掩码。"""

    environment, _ = build_test_environment()

    state, info = environment.reset(seed=123)

    assert state.shape == (78,)
    assert state.dtype == np.float32
    assert environment.state_dim == 78
    assert environment.action_count == 12
    assert info["action_mask"].shape == (12,)
    assert info["action_mask"].dtype == np.bool_
    assert info["action_mask"].any()
    assert info["episode_seed"] == 123
    assert info["decision_slot"] == 0


def test_future_trace_change_does_not_change_current_or_next_state() -> None:
    """预测器只能使用已经执行的请求，不能偷看未来请求轨迹。"""

    first, _ = build_test_environment(
        slow_period_slots=2,
        request_trace=[0, 1, 9, 9],
    )
    second, _ = build_test_environment(
        slow_period_slots=2,
        request_trace=[0, 1, 0, 0],
    )

    first_state, first_info = first.reset(seed=1)
    second_state, second_info = second.reset(seed=1)
    np.testing.assert_array_equal(first_state, second_state)

    action = first_valid_action(first_info["action_mask"])
    assert bool(second_info["action_mask"][action])
    first_next_state, *_ = first.step(action)
    second_next_state, *_ = second.step(action)

    # 两个环境此时都只观察过请求 [0, 1]，未来的 [9, 9] 不得进入状态。
    np.testing.assert_array_equal(first_next_state, second_next_state)
    assert first.observed_request_counts == (0, 1)
    assert second.observed_request_counts == (0, 1)


def test_step_uses_shared_executor_and_simplified_reward() -> None:
    """每个快时隙只经过共享执行器，并使用三项原始成本奖励。"""

    environment, spy = build_test_environment(with_spy=True)
    assert spy is not None
    _, reset_info = environment.reset(seed=1)
    action = first_valid_action(reset_info["action_mask"])

    next_state, reward, terminated, truncated, info = environment.step(action)

    assert spy.call_count == info["window_length"]
    assert reward == info["reward_breakdown"].reward
    assert info["metrics"].total_run_cost == pytest.approx(
        info["reward_breakdown"].run_cost
    )
    assert info["metrics"].total_route_cost == pytest.approx(
        info["reward_breakdown"].route_cost
    )
    assert info["metrics"].total_cold_start_cost == pytest.approx(
        info["reward_breakdown"].cold_start_cost
    )
    assert "fast_repair_attempts" in info["metrics"].__dict__
    assert "constraint_rejected_batches" in info["metrics"].__dict__
    assert "cloud_usage_rate" in info["metrics"].__dict__
    assert next_state.shape == (78,)
    assert truncated is False
    assert terminated is False
    assert info["next_observation"]["action_mask"].shape == (12,)


def test_three_replica_action_reaches_shared_executor() -> None:
    """3 副本动作必须使用共享执行器中对应的规划器，不能退化为 2 副本。"""

    environment, spy = build_test_environment(with_spy=True)
    assert spy is not None
    _, reset_info = environment.reset(seed=11)

    # 动作 6 是固定编码表中的 R3_ON_DEMAND_EDGE_ONLY。
    three_replica_action_id = 6
    assert bool(reset_info["action_mask"][three_replica_action_id])
    assert set(spy.delegate.replica_planners) == {1, 2, 3}

    environment.step(three_replica_action_id)

    assert spy.inputs
    assert all(
        slot_input.slow_decision.replica_count == 3
        for slot_input in spy.inputs
    )


def test_next_state_resources_come_from_last_final_audit() -> None:
    """下一决策点的空闲资源必须来自最后一次最终审计，而不是初始方案。"""

    environment, spy = build_test_environment(
        slow_period_slots=1,
        with_spy=True,
    )
    assert spy is not None
    _, info = environment.reset(seed=2)
    action = first_valid_action(info["action_mask"])

    next_state, *_ = environment.step(action)

    final_audit = spy.results[-1].final_audit
    node = environment.topology.compute_nodes[0]
    expected_free_cpu_ratio = max(
        0.0,
        node.cpu_capacity - final_audit.node_cpu_demand.get(node.node_id, 0.0),
    ) / node.cpu_capacity
    feature_index = environment.state_encoder.feature_names.index(
        f"node_{node.node_id}_free_cpu_ratio"
    )
    assert next_state[feature_index] == pytest.approx(expected_free_cpu_ratio)


def test_masked_action_raises_clear_error_without_advancing() -> None:
    """掩码判为不可行的动作不能进入共享执行器，也不能推进游标。"""

    down_nodes = {1, 2, 3, 4}
    environment, spy = build_test_environment(
        down_nodes_by_slot={slot: down_nodes for slot in range(20)},
        with_spy=True,
    )
    assert spy is not None
    _, info = environment.reset(seed=3)
    assert not bool(info["action_mask"][0])

    with pytest.raises(ValueError, match="掩码"):
        environment.step(0)

    assert spy.call_count == 0
    assert environment.observed_request_counts == ()


def test_advance_without_action_records_violation_without_action() -> None:
    """无合法动作时环境应推进并记违约，但不得伪造一个回放动作。"""

    down_nodes = {1, 2, 3, 4, 5}
    environment, spy = build_test_environment(
        request_trace=[1, 1, 1, 1],
        down_nodes_by_slot={slot: down_nodes for slot in range(20)},
        with_spy=True,
    )
    assert spy is not None
    _, info = environment.reset(seed=4)
    assert not info["action_mask"].any()

    _, reward, _, _, step_info = environment.advance_without_action()

    assert spy.call_count == 0
    assert step_info["action"] is None
    assert step_info["action_recorded"] is False
    assert step_info["metrics"].sla_violations > 0
    assert step_info["reward_breakdown"].violation_penalty == 1.0
    assert reward <= -1.0
    assert len(environment.observed_request_counts) == step_info["window_length"]


def test_handover_creates_early_decision_boundary() -> None:
    """即使慢周期未结束，服务 MEC 切换也应提前结束当前动作窗口。"""

    environment, _ = build_test_environment(slow_period_slots=10)
    _, info = environment.reset(seed=5)

    _, _, terminated, _, step_info = environment.step(
        first_valid_action(info["action_mask"])
    )

    assert terminated is False
    assert step_info["handover_boundary_reached"] is True
    assert step_info["boundary_reason"] == "handover"
    assert step_info["window_length"] == 3
    assert step_info["window_start_slot"] == 0
    assert step_info["window_end_slot"] == 2
    assert step_info["next_observation"]["serving_mec"] == 1


def test_episode_eventually_terminates() -> None:
    """持续执行合法结构化动作后，Episode 应到达线路终点。"""

    environment, _ = build_test_environment(slow_period_slots=10)
    _, info = environment.reset(seed=6)
    terminated = False
    step_count = 0

    while not terminated:
        action_mask = info["action_mask"]
        if action_mask.any():
            _, _, terminated, truncated, step_info = environment.step(
                first_valid_action(action_mask)
            )
        else:
            _, _, terminated, truncated, step_info = (
                environment.advance_without_action()
            )
        assert truncated is False
        info = step_info["next_observation"]
        step_count += 1
        if step_count > 100:
            pytest.fail("Episode 未能正常结束。")

    assert terminated is True
