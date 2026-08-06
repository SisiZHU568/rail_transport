"""
test_slow_timescale_rl_env.py

测试强化学习环境：

1. reset状态维度；
2. 非法动作；
3. MEC切换决策边界；
4. 冷备与热备内存；
5. 冷备用接管；
6. 热备用接管；
7. 单副本故障失败；
8. Episode终止。
"""

import numpy as np
import pytest

from src.config import load_config
from src.entities import (
    ServerlessFunction,
    ServicePriority,
    SFCType,
)
from src.failure_process import (
    ScriptedFailureProcess,
)
from src.failure_risk_prediction import (
    ConstantFailureRiskProvider,
)
from src.mobility import TrainMobilityModel
from src.network import build_linear_mec_network
from src.redundancy_placement import (
    build_reliability_aware_replica_planner,
)
from src.rl_reward import (
    RLRewardWeights,
)
from src.runtime_reliability import (
    SingleReplicaPlanner,
)
from src.slow_timescale_rl_env import (
    SlowControlAction,
    SlowTimescaleRLEnvironment,
)
from src.topology import build_linear_topology
from src.workload import DeterministicWorkload


def build_test_function() -> ServerlessFunction:
    return ServerlessFunction(
        function_id=0,
        name="测试函数",
        memory_mb=100.0,
        cpu_cycles_per_request=10.0,
        image_size_mb=50.0,
        warm_exec_time_ms=10.0,
        cold_start_time_ms=100.0,
        output_ratio=0.5,
    )


def build_test_sfc() -> SFCType:
    return SFCType(
        sfc_id=0,
        name="强化学习测试SFC",
        function_ids=[0],
        deadline_ms=500.0,
        reliability_target=0.99,
        priority=ServicePriority.CRITICAL,
    )


def build_test_environment(
    slow_period_slots: int = 3,
    request_trace: list[int] | None = None,
    down_nodes_by_slot: (
        dict[int, set[int]] | None
    ) = None,
) -> SlowTimescaleRLEnvironment:
    """
    创建测试强化学习环境。
    """

    config = load_config(
        "configs/debug.yaml"
    )

    topology = build_linear_topology(
        config
    )

    network = build_linear_mec_network(
        config=config,
        topology=topology,
    )

    mobility_model = TrainMobilityModel(
        topology=topology,
        initial_position_m=0.0,
        speed_mps=500.0,
        slot_seconds=1.0,
    )

    workload = DeterministicWorkload(
        request_trace=(
            request_trace
            if request_trace is not None
            else [1, 1, 1, 1]
        ),
        repeat=False,
    )

    def failure_builder(
        seed: int,
    ) -> ScriptedFailureProcess:
        # seed在确定性故障过程中不参与随机计算，
        # 但保留统一构建接口。
        _ = seed

        return ScriptedFailureProcess(
            topology=topology,
            down_nodes_by_slot=(
                down_nodes_by_slot
            ),
        )

    return SlowTimescaleRLEnvironment(
        topology=topology,
        network=network,
        mobility_model=mobility_model,
        workload=workload,
        functions=[build_test_function()],
        sfc=build_test_sfc(),
        single_replica_planner=(
            SingleReplicaPlanner()
        ),
        redundant_replica_planner=(
            build_reliability_aware_replica_planner(
                config
            )
        ),
        failure_process_builder=(
            failure_builder
        ),
        failure_risk_provider=(
            ConstantFailureRiskProvider(
                risk=0.01
            )
        ),
        slow_period_slots=(
            slow_period_slots
        ),
        handover_hot_window_s=1.0,
        input_size_mb_per_request=1.0,
        slot_seconds=1.0,
        failover_delay_ms_per_function=20.0,
        reward_weights=RLRewardWeights(
            delay=0.20,
            memory=0.20,
            cold_start=0.15,
            sla_violation=0.40,
            reconfiguration=0.05,
        ),
        default_seed=42,
        return_result_to_source=True,
    )


def test_reset_returns_correct_state_shape() -> None:
    """
    reset应返回15维float32状态。
    """

    environment = build_test_environment()

    state, info = environment.reset(
        seed=123
    )

    assert state.shape == (
        environment.state_dim,
    )

    assert state.dtype == np.float32
    assert environment.state_dim == 15
    assert environment.action_count == 4

    assert info["episode_seed"] == 123
    assert info["decision_slot"] == 0


def test_invalid_action_is_rejected() -> None:
    """
    当前动作空间只能取0、1、2、3。
    """

    environment = build_test_environment()

    environment.reset()

    with pytest.raises(ValueError):
        environment.step(4)


def test_handover_creates_early_decision_boundary() -> None:
    """
    速度500m/s时：

    时隙0：MEC-1
    时隙1：MEC-1
    时隙2：MEC-1
    时隙3：MEC-2

    即使慢周期为10，
    第一个动作也只执行时隙0～2。
    """

    environment = build_test_environment(
        slow_period_slots=10
    )

    environment.reset()

    _, _, terminated, _, info = (
        environment.step(
            SlowControlAction.COLD
        )
    )

    assert terminated is False

    assert info[
        "handover_boundary_reached"
    ] is True

    assert info["window_length"] == 3
    assert info["window_start_slot"] == 0
    assert info["window_end_slot"] == 2

    assert (
        info["next_observation"][
            "serving_mec"
        ]
        == 1
    )


def test_hot_standby_uses_more_memory_than_cold() -> None:
    """
    无故障情况下：

    冷备只保持主实例温热；
    热备保持主备两个实例温热。
    """

    cold_environment = (
        build_test_environment(
            slow_period_slots=1,
            request_trace=[0],
        )
    )

    hot_environment = (
        build_test_environment(
            slow_period_slots=1,
            request_trace=[0],
        )
    )

    cold_environment.reset(seed=42)
    hot_environment.reset(seed=42)

    _, _, _, _, cold_info = (
        cold_environment.step(
            SlowControlAction.COLD
        )
    )

    _, _, _, _, hot_info = (
        hot_environment.step(
            SlowControlAction.HOT
        )
    )

    cold_memory = (
        cold_info["metrics"]
        .average_active_memory_mb
    )

    hot_memory = (
        hot_info["metrics"]
        .average_active_memory_mb
    )

    assert cold_memory == pytest.approx(
        100.0
    )

    assert hot_memory == pytest.approx(
        200.0
    )


def test_cold_failover_requires_cold_start() -> None:
    """
    时隙0令MEC-1主节点失效。

    冷备用MEC-3接管时需要100ms冷启动。
    """

    environment = build_test_environment(
        slow_period_slots=1,
        request_trace=[1],
        down_nodes_by_slot={
            0: {0},
        },
    )

    environment.reset()

    _, _, _, _, info = environment.step(
        SlowControlAction.COLD
    )

    metrics = info["metrics"]

    assert metrics.successful_requests == 1

    assert (
        metrics.cold_start_function_stages
        == 1
    )

    assert (
        metrics.total_cold_start_delay_ms
        == pytest.approx(100.0)
    )


def test_hot_failover_has_no_cold_start() -> None:
    """
    同样的主节点故障下，
    全热备接管不需要冷启动。
    """

    environment = build_test_environment(
        slow_period_slots=1,
        request_trace=[1],
        down_nodes_by_slot={
            0: {0},
        },
    )

    environment.reset()

    _, _, _, _, info = environment.step(
        SlowControlAction.HOT
    )

    metrics = info["metrics"]

    assert metrics.successful_requests == 1

    assert (
        metrics.cold_start_function_stages
        == 0
    )

    assert (
        metrics.total_cold_start_delay_ms
        == 0.0
    )


def test_single_replica_fails_when_primary_is_down() -> None:
    """
    单副本主节点失效时请求失败。
    """

    environment = build_test_environment(
        slow_period_slots=1,
        request_trace=[1],
        down_nodes_by_slot={
            0: {0},
        },
    )

    environment.reset()

    _, _, _, _, info = environment.step(
        SlowControlAction.SINGLE
    )

    metrics = info["metrics"]

    assert metrics.successful_requests == 0
    assert metrics.failed_batches == 1
    assert metrics.sla_violations == 1
    assert metrics.request_success_rate == 0.0


def test_episode_eventually_terminates() -> None:
    """
    连续调用step后应到达线路终点。
    """

    environment = build_test_environment(
        slow_period_slots=10
    )

    environment.reset()

    terminated = False
    step_count = 0

    while not terminated:
        (
            _,
            _,
            terminated,
            truncated,
            _,
        ) = environment.step(
            SlowControlAction.DYNAMIC
        )

        assert truncated is False

        step_count += 1

        if step_count > 100:
            pytest.fail(
                "Episode未能正常结束。"
            )

    assert terminated is True