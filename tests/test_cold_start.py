"""
test_cold_start.py

测试 Serverless 容器生命周期模型。
"""

import pytest

from src.cold_start import ContainerLifecycleManager
from src.entities import (
    FunctionInstance,
    InstanceStatus,
    ServerlessFunction,
)


def build_test_function() -> ServerlessFunction:
    """
    创建供测试使用的简单 Serverless 函数。
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


def test_first_request_causes_cold_start() -> None:
    """
    容器不存在时，第一个请求应触发冷启动。
    """

    function = build_test_function()

    instance = FunctionInstance(
        node_id=0,
        function_id=0,
    )

    manager = ContainerLifecycleManager(
        survival_slots=3
    )

    result = manager.process_slot(
        instance=instance,
        function=function,
        request_count=2,
    )

    assert result.cold_start_occurred is True
    assert result.warm_hit is False
    assert result.cold_start_delay_ms == 300.0
    assert result.execution_delay_ms == 30.0
    assert result.total_delay_ms == 330.0
    assert instance.status == InstanceStatus.WARM


def test_second_request_hits_warm_instance() -> None:
    """
    第一个请求完成后，紧接着到达的请求应命中温实例。
    """

    function = build_test_function()
    instance = FunctionInstance(node_id=0, function_id=0)
    manager = ContainerLifecycleManager(survival_slots=3)

    # 第一个时隙触发冷启动。
    manager.process_slot(
        instance=instance,
        function=function,
        request_count=1,
    )

    # 第二个时隙复用温实例。
    result = manager.process_slot(
        instance=instance,
        function=function,
        request_count=1,
    )

    assert result.cold_start_occurred is False
    assert result.warm_hit is True
    assert result.cold_start_delay_ms == 0.0
    assert instance.status == InstanceStatus.WARM


def test_container_expires_after_survival_window() -> None:
    """
    容器连续空闲达到生存窗口后，应被销毁。
    """

    function = build_test_function()
    instance = FunctionInstance(node_id=0, function_id=0)
    manager = ContainerLifecycleManager(survival_slots=3)

    # 首先创建温实例。
    manager.process_slot(
        instance=instance,
        function=function,
        request_count=1,
    )

    # 第一个空闲时隙。
    result_1 = manager.process_slot(
        instance=instance,
        function=function,
        request_count=0,
    )

    # 第二个空闲时隙。
    result_2 = manager.process_slot(
        instance=instance,
        function=function,
        request_count=0,
    )

    # 第三个空闲时隙，达到生存窗口。
    result_3 = manager.process_slot(
        instance=instance,
        function=function,
        request_count=0,
    )

    assert result_1.expired is False
    assert result_2.expired is False
    assert result_3.expired is True
    assert instance.status == InstanceStatus.ABSENT


def test_keep_warm_prevents_expiration() -> None:
    """
    主动保温应阻止容器因空闲而被回收。
    """

    function = build_test_function()
    instance = FunctionInstance(node_id=0, function_id=0)
    manager = ContainerLifecycleManager(survival_slots=3)

    # 在没有真实请求的情况下主动预热。
    first_result = manager.process_slot(
        instance=instance,
        function=function,
        request_count=0,
        keep_warm=True,
    )

    assert first_result.prewarm_occurred is True
    assert instance.status == InstanceStatus.WARM

    # 连续多个时隙执行保活。
    for _ in range(10):
        manager.process_slot(
            instance=instance,
            function=function,
            request_count=0,
            keep_warm=True,
        )

    assert instance.status == InstanceStatus.WARM
    assert instance.idle_slots == 0


def test_failed_instance_rejects_request() -> None:
    """
    FAILED 状态的实例不能处理请求。
    """

    function = build_test_function()

    instance = FunctionInstance(
        node_id=0,
        function_id=0,
        status=InstanceStatus.FAILED,
    )

    manager = ContainerLifecycleManager(
        survival_slots=3
    )

    with pytest.raises(RuntimeError):
        manager.process_slot(
            instance=instance,
            function=function,
            request_count=1,
        )