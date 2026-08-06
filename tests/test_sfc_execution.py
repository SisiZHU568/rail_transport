"""
test_sfc_execution.py

测试完整 SFC 的顺序执行、数据量变化和端到端时延。
"""

import pytest

from src.config import load_config
from src.entities import (
    ServerlessFunction,
    ServicePriority,
    SFCType,
)
from src.network import build_linear_mec_network
from src.sfc_execution import (
    execute_sfc_request, 
    execute_sfc_batch,
    )
from src.topology import build_linear_topology


def build_test_functions() -> list[ServerlessFunction]:
    """
    创建三个测试函数。
    """

    return [
        ServerlessFunction(
            function_id=0,
            name="数据清洗",
            memory_mb=256.0,
            cpu_cycles_per_request=20.0,
            image_size_mb=80.0,
            warm_exec_time_ms=15.0,
            cold_start_time_ms=300.0,
            output_ratio=0.7,
        ),
        ServerlessFunction(
            function_id=1,
            name="特征提取",
            memory_mb=512.0,
            cpu_cycles_per_request=40.0,
            image_size_mb=150.0,
            warm_exec_time_ms=30.0,
            cold_start_time_ms=500.0,
            output_ratio=0.4,
        ),
        ServerlessFunction(
            function_id=2,
            name="异常检测",
            memory_mb=768.0,
            cpu_cycles_per_request=60.0,
            image_size_mb=300.0,
            warm_exec_time_ms=50.0,
            cold_start_time_ms=800.0,
            output_ratio=0.1,
        ),
    ]


def build_test_sfc() -> SFCType:
    """
    创建测试 SFC。
    """

    return SFCType(
        sfc_id=0,
        name="列车设备故障诊断",
        function_ids=[0, 1, 2],
        deadline_ms=1500.0,
        reliability_target=0.99,
        priority=ServicePriority.CRITICAL,
    )


def build_test_network():
    """
    创建测试网络。
    """

    config = load_config("configs/debug.yaml")
    topology = build_linear_topology(config)

    return build_linear_mec_network(
        config=config,
        topology=topology,
    )


def test_local_warm_sfc_execution() -> None:
    """
    三个函数都位于 MEC-1，且都是温实例。

    没有跨 MEC 传输时延。

    总执行时延：

        15 + 30 + 50 = 95 ms
    """

    result = execute_sfc_request(
        functions=build_test_functions(),
        sfc=build_test_sfc(),
        placement_node_ids=[0, 0, 0],
        source_node_id=0,
        input_size_mb=2.0,
        network=build_test_network(),
    )

    assert result.total_transmission_delay_ms == 0.0
    assert result.total_cold_start_delay_ms == 0.0
    assert result.total_execution_delay_ms == 95.0
    assert result.total_end_to_end_delay_ms == 95.0
    assert result.final_output_size_mb == pytest.approx(0.056)
    assert result.deadline_met is True


def test_distributed_warm_sfc_execution() -> None:
    """
    三个函数分别部署在 MEC-1、MEC-2、MEC-3。

    传输过程：

    函数0在源节点：
        传输时延 = 0

    函数0输出1.4 MB，从MEC-1传到MEC-2：
        1.4 × 8 / 1000 × 1000 + 2
        = 13.2 ms

    函数1输出0.56 MB，从MEC-2传到MEC-3：
        0.56 × 8 / 1000 × 1000 + 2
        = 6.48 ms

    最终结果0.056 MB，从MEC-3返回MEC-1：
        0.056 × 8 / 1000 × 1000 + 4
        = 4.448 ms

    总传输时延：
        13.2 + 6.48 + 4.448
        = 24.128 ms

    总端到端时延：
        24.128 + 95
        = 119.128 ms
    """

    result = execute_sfc_request(
        functions=build_test_functions(),
        sfc=build_test_sfc(),
        placement_node_ids=[0, 1, 2],
        source_node_id=0,
        input_size_mb=2.0,
        network=build_test_network(),
    )

    assert result.total_transmission_delay_ms == pytest.approx(
        24.128
    )

    assert result.total_execution_delay_ms == 95.0

    assert result.total_end_to_end_delay_ms == pytest.approx(
        119.128
    )

    assert result.deadline_met is True


def test_distributed_all_cold_sfc_misses_deadline() -> None:
    """
    分布式部署，三个函数全部冷启动。

    冷启动总时延：

        300 + 500 + 800 = 1600 ms

    总端到端时延：

        119.128 + 1600
        = 1719.128 ms

    超过1500 ms时延约束。
    """

    result = execute_sfc_request(
        functions=build_test_functions(),
        sfc=build_test_sfc(),
        placement_node_ids=[0, 1, 2],
        source_node_id=0,
        input_size_mb=2.0,
        network=build_test_network(),
        cold_start_function_ids={0, 1, 2},
    )

    assert result.total_cold_start_delay_ms == 1600.0

    assert result.total_end_to_end_delay_ms == pytest.approx(
        1719.128
    )

    assert result.deadline_met is False


def test_invalid_placement_length() -> None:
    """
    三个函数必须提供三个部署节点。

    只提供两个节点时，程序应主动报错。
    """

    with pytest.raises(ValueError):
        execute_sfc_request(
            functions=build_test_functions(),
            sfc=build_test_sfc(),
            placement_node_ids=[0, 1],
            source_node_id=0,
            input_size_mb=2.0,
            network=build_test_network(),
        )

def test_local_warm_batch_execution() -> None:
    """
    测试2个请求的批量执行。

    三个函数都在本地 MEC，且都为温实例。

    单请求执行时延：

        15 + 30 + 50 = 95 ms

    2个请求串行执行：

        95 × 2 = 190 ms

    最终批量数据量：

        2 MB/请求 × 2个请求
        × 0.7 × 0.4 × 0.1
        = 0.112 MB
    """

    result = execute_sfc_batch(
        functions=build_test_functions(),
        sfc=build_test_sfc(),
        placement_node_ids=[0, 0, 0],
        source_node_id=0,
        input_size_mb_per_request=2.0,
        request_count=2,
        network=build_test_network(),
        cold_start_function_ids=set(),
    )

    assert result.total_transmission_delay_ms == 0.0
    assert result.total_cold_start_delay_ms == 0.0

    assert result.total_execution_delay_ms == pytest.approx(
        190.0
    )

    assert result.final_output_size_mb == pytest.approx(
        0.112
    )

    assert result.total_end_to_end_delay_ms == pytest.approx(
        190.0
    )