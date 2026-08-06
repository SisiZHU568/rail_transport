"""
test_prewarming_simulation.py

测试轨迹感知预热是否能在完整仿真中
减少用户冷启动和时延违反。
"""

import pytest

from src.config import load_config
from src.entities import (
    ServerlessFunction,
    ServicePriority,
    SFCType,
)
from src.mobility import TrainMobilityModel
from src.network import build_linear_mec_network
from src.placement import ServingMECPlacementPolicy
from src.prewarming import (
    NoPrewarmingPolicy,
    PrewarmingPolicy,
    TrajectoryAwarePrewarmingPolicy,
)
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
    prewarming_policy: PrewarmingPolicy,
) -> RailServerlessSFCSimulator:
    """
    创建测试仿真器。

    列车速度设置为500 m/s，
    用于缩短自动测试时间。

    请求轨迹 [0, 0, 0, 1] 表示：
    列车切换到MEC-2后才出现第一个请求。
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
        request_trace=[0, 0, 0, 1],
        repeat=False,
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
        prewarming_policy=prewarming_policy,
    )


def test_reactive_policy_cold_starts_after_handover() -> None:
    """
    不进行提前预热时，
    MEC-2上的第一个请求会触发三个函数冷启动。
    """

    simulator = build_test_simulator(
        prewarming_policy=NoPrewarmingPolicy()
    )

    result = simulator.run()

    request_record = result.records[3]

    assert request_record.serving_mec == 1
    assert request_record.request_count == 1
    assert request_record.cold_start_count == 3
    assert request_record.warm_hit_count == 0
    assert request_record.deadline_met is False


def test_trajectory_prewarming_avoids_handover_cold_start() -> None:
    """
    时隙1时，列车距离切换边界还剩1秒，
    因此提前在MEC-2预热三个函数。

    时隙2仍处于预热条件内，
    但容器已经为温实例，不会重复启动。

    时隙3列车进入MEC-2后，
    请求应命中三个温实例。
    """

    simulator = build_test_simulator(
        prewarming_policy=(
            TrajectoryAwarePrewarmingPolicy(
                lead_time_s=1.0
            )
        )
    )

    result = simulator.run()

    # 时隙1：
    # 列车位于500米，距离1000米切换边界还剩1秒。
    # 此时第一次在MEC-2启动三个预热容器。
    first_prewarm_record = result.records[1]

    assert first_prewarm_record.position_m == pytest.approx(
        500.0
    )
    assert first_prewarm_record.serving_mec == 0
    assert first_prewarm_record.prewarm_target_node_ids == (1,)
    assert first_prewarm_record.prewarm_start_count == 3
    assert (
        first_prewarm_record.prewarm_startup_overhead_ms
        == pytest.approx(1600.0)
    )

    # 时隙2：
    # 仍然需要维持MEC-2上的预热状态，
    # 但三个容器已经存在，因此不会再次启动。
    keep_warm_record = result.records[2]

    assert keep_warm_record.position_m == pytest.approx(
        1000.0
    )
    assert keep_warm_record.prewarm_target_node_ids == (1,)
    assert keep_warm_record.prewarm_start_count == 0
    assert (
        keep_warm_record.prewarm_startup_overhead_ms
        == 0.0
    )

    # 时隙3：
    # 列车已经切换到MEC-2，
    # 请求直接命中三个温实例。
    request_record = result.records[3]

    assert request_record.serving_mec == 1
    assert request_record.request_count == 1
    assert request_record.cold_start_count == 0
    assert request_record.warm_hit_count == 3
    assert request_record.end_to_end_delay_ms == pytest.approx(
        95.0
    )
    assert request_record.deadline_met is True


def test_prewarming_overhead_is_recorded() -> None:
    """
    经过四次 MEC 切换时，
    每次预热三个函数。

    预热容器数量：

        4 × 3 = 12

    每次整条 SFC 的启动开销：

        300 + 500 + 800 = 1600 ms

    总后台预热启动开销：

        4 × 1600 = 6400 ms
    """

    simulator = build_test_simulator(
        prewarming_policy=(
            TrajectoryAwarePrewarmingPolicy(
                lead_time_s=1.0
            )
        )
    )

    result = simulator.run()

    assert result.summary.prewarm_starts == 12

    assert (
        result.summary
        .total_prewarm_startup_overhead_ms
        == pytest.approx(6400.0)
    )