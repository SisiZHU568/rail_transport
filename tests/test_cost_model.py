"""
test_cost_model.py

测试容器时隙成本模型。
"""

import pytest

from src.cold_start import ContainerLifecycleManager
from src.cost_model import CostWeights, calculate_slot_cost
from src.entities import (
    FunctionInstance,
    InstanceStatus,
    ServerlessFunction,
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


def build_cost_weights() -> CostWeights:
    """
    创建测试成本权重。
    """

    return CostWeights(
        delay_cost_per_ms=1.0,
        warm_memory_cost_per_mb_second=0.1,
        prewarm_start_cost_per_ms=4.0,
    )


def test_idle_warm_container_has_retention_cost() -> None:
    """
    一个256 MB的温容器空闲1秒时：

        保留成本 = 256 × 1 × 0.1 = 25.6
    """

    function = build_test_function()

    instance = FunctionInstance(
        node_id=0,
        function_id=0,
        status=InstanceStatus.WARM,
    )

    manager = ContainerLifecycleManager(
        survival_slots=3
    )

    lifecycle_result = manager.process_slot(
        instance=instance,
        function=function,
        request_count=0,
        keep_warm=False,
    )

    cost = calculate_slot_cost(
        function=function,
        lifecycle_result=lifecycle_result,
        slot_seconds=1.0,
        weights=build_cost_weights(),
    )

    assert cost.request_delay_ms == 0.0
    assert cost.retention_cost == pytest.approx(25.6)
    assert cost.prewarm_cost == 0.0


def test_prewarm_is_not_user_request_delay() -> None:
    """
    主动预热会产生资源成本，
    但没有真实请求，因此用户请求时延应为0。
    """

    function = build_test_function()
    instance = FunctionInstance(node_id=0, function_id=0)
    manager = ContainerLifecycleManager(survival_slots=3)

    lifecycle_result = manager.process_slot(
        instance=instance,
        function=function,
        request_count=0,
        keep_warm=True,
    )

    cost = calculate_slot_cost(
        function=function,
        lifecycle_result=lifecycle_result,
        slot_seconds=1.0,
        weights=build_cost_weights(),
    )

    assert lifecycle_result.prewarm_occurred is True
    assert cost.request_delay_ms == 0.0

    # 预热成本 = 300 ms × 4.0
    assert cost.prewarm_cost == pytest.approx(1200.0)