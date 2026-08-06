"""
test_retention_experiment.py

测试完整的容器保留策略实验流程。
"""

from src.cost_model import CostWeights
from src.entities import ServerlessFunction
from src.retention_experiment import run_retention_policy
from src.retention_policies import (
    AlwaysWarmPolicy,
    FixedWindowPolicy,
)


def build_test_function() -> ServerlessFunction:
    """
    创建测试函数。
    """

    return ServerlessFunction(
        function_id=0,
        name="测试函数",
        memory_mb=256.0,
        cpu_cycles_per_request=20.0,
        image_size_mb=80.0,
        warm_exec_time_ms=15.0,
        cold_start_time_ms=300.0,
        output_ratio=0.5,
    )


def build_weights() -> CostWeights:
    """
    创建测试权重。
    """

    return CostWeights(
        delay_cost_per_ms=1.0,
        warm_memory_cost_per_mb_second=0.1,
        prewarm_start_cost_per_ms=4.0,
    )


def test_fixed_window_causes_two_cold_starts() -> None:
    """
    请求轨迹：

        [0, 1, 0, 0, 1]

    生存窗口为2：

    时隙1：第一次请求，冷启动；
    时隙2：空闲1；
    时隙3：空闲2，容器销毁；
    时隙4：再次请求，第二次冷启动。
    """

    result = run_retention_policy(
        function=build_test_function(),
        request_trace=[0, 1, 0, 0, 1],
        policy=FixedWindowPolicy(window_slots=2),
        slot_seconds=1.0,
        cost_weights=build_weights(),
    )

    assert result.summary.total_requests == 2
    assert result.summary.user_cold_starts == 2
    assert result.summary.expiration_count == 1


def test_always_warm_avoids_user_cold_start() -> None:
    """
    始终保温策略会在时隙0主动预热，
    因此后续真实请求不再产生用户冷启动。
    """

    result = run_retention_policy(
        function=build_test_function(),
        request_trace=[0, 1, 0, 0, 1],
        policy=AlwaysWarmPolicy(),
        slot_seconds=1.0,
        cost_weights=build_weights(),
    )

    assert result.summary.prewarm_starts == 1
    assert result.summary.user_cold_starts == 0
    assert result.summary.warm_hit_batches == 2