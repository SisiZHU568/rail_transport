"""测试 DPPO 连续联合动作到共享快层闭环的因果环境。"""

from copy import deepcopy

import numpy as np
import pytest

from src.config import load_config
from src.dppo_projection import ProjectionResult
from src.dppo_scenario import build_dppo_environment


class CountingFastExecutor:
    """只统计共享执行器调用次数，其余行为全部委托给真实对象。"""

    def __init__(self, delegate: object) -> None:
        self.delegate = delegate
        self.call_count = 0

    def reset(self) -> None:
        self.delegate.reset()

    def execute(self, slot_input: object):
        self.call_count += 1
        return self.delegate.execute(slot_input)


class AlwaysFailProjector:
    """稳定制造投影失败，用于验证环境的拒绝窗口分支。"""

    def project(self, **_: object) -> ProjectionResult:
        return ProjectionResult(
            function_intents=None,
            raw_feasible=False,
            success=False,
            reasons=("测试注入的投影失败",),
            changed_assignment_count=0,
            requested_assignment_count=1,
            change_ratio=0.0,
            fault_domains={},
        )


@pytest.fixture
def dppo_environment():
    """从同一调试配置创建真实 DPPO 环境。"""

    return build_dppo_environment(load_config("configs/debug.yaml"))


def test_reset_and_step_expose_projected_and_final_results(
    dppo_environment,
) -> None:
    """一次慢动作必须同时保留原始、截断、投影和最终执行结果。"""

    state = dppo_environment.reset(seed=123)
    raw_action = np.zeros(dppo_environment.dimensions.action_dim, dtype=np.float32)
    raw_action[0] = 2.0

    next_state, reward, terminated, truncated, info = (
        dppo_environment.step(raw_action)
    )

    expected_shape = (dppo_environment.dimensions.state_dim,)
    assert state.shape == next_state.shape == expected_shape
    assert np.isfinite(reward)
    assert not (terminated and truncated)
    assert {
        "raw_action",
        "clipped_action",
        "decoded_action",
        "projection_result",
        "projected_intent",
        "final_candidate_map",
        "final_metrics",
        "reward_breakdown",
        "next_observation",
    } <= info.keys()
    assert info["raw_action"][0] == pytest.approx(2.0)
    assert info["clipped_action"][0] == pytest.approx(1.0)
    assert info["projection_result"].success is True


def test_step_calls_fast_executor_once_per_fast_slot(dppo_environment) -> None:
    """成功投影后，慢窗口内每个快时隙都只能调用一次共享执行器。"""

    spy = CountingFastExecutor(dppo_environment.execution_core.fast_slot_executor)
    dppo_environment.execution_core.fast_slot_executor = spy
    dppo_environment.reset(seed=123)

    _, _, _, _, info = dppo_environment.step(
        np.zeros(dppo_environment.dimensions.action_dim, dtype=np.float32)
    )

    assert spy.call_count == info["window_length"]


def test_projection_failure_advances_with_violation(dppo_environment) -> None:
    """无可行投影不能调用快层，但必须保留动作轨迹并返回负奖励。"""

    spy = CountingFastExecutor(dppo_environment.execution_core.fast_slot_executor)
    dppo_environment.execution_core.fast_slot_executor = spy
    dppo_environment.projector = AlwaysFailProjector()
    dppo_environment.reset(seed=123)

    _, reward, _, _, info = dppo_environment.step(
        np.zeros(dppo_environment.dimensions.action_dim, dtype=np.float32)
    )

    assert spy.call_count == 0
    assert info["projection_result"].success is False
    assert info["projected_intent"] is None
    assert info["final_metrics"].sla_violations >= 1
    assert info["reward_breakdown"].violation_penalty == 1.0
    assert reward <= -1.0


def test_action_shape_must_match_configured_dimension(dppo_environment) -> None:
    """环境不能静默接受其他场景规模的连续动作。"""

    dppo_environment.reset(seed=123)

    with pytest.raises(ValueError, match="动作形状"):
        dppo_environment.step(
            np.zeros(dppo_environment.dimensions.action_dim - 1)
        )


def test_future_request_changes_do_not_change_current_state() -> None:
    """尚未执行的未来请求不能泄漏进当前状态或负载预测。"""

    first_config = load_config("configs/debug.yaml")
    second_config = deepcopy(first_config)
    second_config["integrated_simulation"]["request_trace"][-1] += 100
    first_environment = build_dppo_environment(first_config)
    second_environment = build_dppo_environment(second_config)

    first_state = first_environment.reset(seed=123)
    second_state = second_environment.reset(seed=123)

    np.testing.assert_array_equal(first_state, second_state)
