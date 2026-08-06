"""
test_simulator.py

测试完整轨道边缘 Serverless SFC 离散时隙仿真器。
"""

from src.config import load_config
from src.entities import (
    ServerlessFunction,
    ServicePriority,
    SFCType,
)
from src.mobility import TrainMobilityModel
from src.network import build_linear_mec_network
from src.placement import ServingMECPlacementPolicy
from src.simulator import RailServerlessSFCSimulator
from src.topology import build_linear_topology
from src.workload import DeterministicWorkload


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


def build_test_simulator(
    request_trace: list[int],
    repeat: bool,
) -> RailServerlessSFCSimulator:
    """
    创建完整测试仿真器。

    测试中将列车速度提高到500 m/s，
    使测试能够更快完成。

    该速度仅用于自动测试，
    不代表真实高速列车速度。
    """

    config = load_config("configs/debug.yaml")

    topology = build_linear_topology(config)

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
        request_trace=request_trace,
        repeat=repeat,
    )

    return RailServerlessSFCSimulator(
        topology=topology,
        network=network,
        mobility_model=mobility_model,
        workload=workload,
        placement_policy=(
            ServingMECPlacementPolicy()
        ),
        functions=build_test_functions(),
        sfc=build_test_sfc(),
        survival_slots=2,
        input_size_mb_per_request=2.0,
        return_result_to_source=True,
    )


def test_simulator_reaches_route_end() -> None:
    """
    列车应从0米运行到8000米。

    速度为500 m/s，
    因此记录的时隙为0到16，共17个状态。
    """

    simulator = build_test_simulator(
        request_trace=[0],
        repeat=True,
    )

    result = simulator.run()

    assert result.summary.total_slots == 17
    assert result.records[0].position_m == 0.0
    assert result.records[-1].position_m == 8000.0

    # MEC-1到MEC-5共发生4次切换。
    assert result.summary.handover_count == 4


def test_first_request_cold_starts_all_functions() -> None:
    """
    仿真开始时所有函数容器都不存在。

    第一个请求应触发三个函数冷启动。
    """

    simulator = build_test_simulator(
        request_trace=[1],
        repeat=True,
    )

    result = simulator.run()

    first_record = result.records[0]

    assert first_record.request_count == 1
    assert first_record.cold_start_count == 3
    assert first_record.warm_hit_count == 0

    # 300 + 500 + 800 = 1600 ms
    assert first_record.cold_start_delay_ms == 1600.0

    # 1600 + 95 = 1695 ms，超过1500 ms。
    assert first_record.deadline_met is False


def test_second_request_hits_warm_instances() -> None:
    """
    列车在前两个测试时隙仍位于 MEC-1 服务区。

    第一个时隙完成冷启动后，
    第二个时隙应命中三个温实例。
    """

    simulator = build_test_simulator(
        request_trace=[1],
        repeat=True,
    )

    result = simulator.run()

    second_record = result.records[1]

    assert second_record.serving_mec == 0
    assert second_record.request_count == 1
    assert second_record.cold_start_count == 0
    assert second_record.warm_hit_count == 3
    assert second_record.end_to_end_delay_ms == 95.0
    assert second_record.deadline_met is True


def test_no_request_slot_has_zero_delay() -> None:
    """
    没有请求的时隙不执行 SFC，
    所有时延都应为0。
    """

    simulator = build_test_simulator(
        request_trace=[0, 1],
        repeat=False,
    )

    result = simulator.run()

    first_record = result.records[0]

    assert first_record.request_count == 0
    assert first_record.transmission_delay_ms == 0.0
    assert first_record.cold_start_delay_ms == 0.0
    assert first_record.execution_delay_ms == 0.0
    assert first_record.end_to_end_delay_ms == 0.0
    assert first_record.deadline_met is None


def test_idle_containers_are_eventually_expired() -> None:
    """
    第0时隙创建容器后，
    后续没有请求。

    生存窗口为2，因此旧容器最终会被销毁。
    """

    simulator = build_test_simulator(
        request_trace=[1, 0, 0, 0],
        repeat=False,
    )

    result = simulator.run()

    assert result.summary.user_cold_starts == 3

    # 三个函数实例都应在空闲后被回收。
    assert result.summary.expiration_count >= 3