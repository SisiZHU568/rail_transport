"""
test_runtime_reliability.py

测试运行态主备接管、服务失败和热备内存开销。
"""

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
from src.mobility import TrainMobilityModel
from src.network import build_linear_mec_network
from src.redundancy_placement import (
    build_reliability_aware_replica_planner,
)
from src.runtime_reliability import (
    HotStandbyRuntimeSimulator,
    SingleReplicaPlanner,
)
from src.topology import build_linear_topology
from src.workload import DeterministicWorkload


def build_test_function() -> ServerlessFunction:
    """
    创建单函数测试 SFC。
    """

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
    """
    创建单函数测试 SFC。
    """

    return SFCType(
        sfc_id=0,
        name="高可靠测试SFC",
        function_ids=[0],
        deadline_ms=500.0,
        reliability_target=0.99,
        priority=ServicePriority.CRITICAL,
    )


def build_test_simulator(
    replica_planner,
    down_nodes_by_slot: dict[
        int,
        set[int],
    ] | None = None,
) -> HotStandbyRuntimeSimulator:
    """
    创建测试仿真器。

    速度500 m/s时：

    时隙0：0 m，MEC-1
    时隙1：500 m，MEC-1
    时隙2：1000 m，MEC-1
    时隙3：1500 m，MEC-2

    请求仅在时隙3到达。
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

    failure_process = ScriptedFailureProcess(
        topology=topology,
        down_nodes_by_slot=(
            down_nodes_by_slot
        ),
    )

    return HotStandbyRuntimeSimulator(
        topology=topology,
        network=network,
        mobility_model=mobility_model,
        workload=workload,
        functions=[build_test_function()],
        sfc=build_test_sfc(),
        replica_planner=replica_planner,
        failure_process=failure_process,
        input_size_mb_per_request=1.0,
        slot_seconds=1.0,
        failover_delay_ms_per_function=20.0,
        return_result_to_source=True,
    )


def test_primary_is_used_when_healthy() -> None:
    """
    主节点正常时应直接使用主实例。
    """

    simulator = build_test_simulator(
        replica_planner=SingleReplicaPlanner(),
    )

    result = simulator.run()

    request_record = result.records[3]

    assert request_record.serving_mec == 1
    assert request_record.request_success is True

    assert (
        request_record
        .selected_execution_node_ids
        == (1,)
    )

    assert request_record.batch_failover is False
    assert request_record.failover_delay_ms == 0.0


def test_single_replica_fails_when_primary_is_down() -> None:
    """
    时隙3的当前接入 MEC-2 失效。

    单副本方案没有备用实例，因此请求失败。
    """

    simulator = build_test_simulator(
        replica_planner=SingleReplicaPlanner(),
        down_nodes_by_slot={
            3: {1},
        },
    )

    result = simulator.run()

    request_record = result.records[3]

    assert request_record.request_success is False

    assert (
        request_record.unavailable_function_ids
        == (0,)
    )

    assert request_record.end_to_end_delay_ms is None

    assert result.summary.failed_requests == 1
    assert result.summary.request_success_rate == 0.0


def test_backup_takes_over_when_primary_is_down() -> None:
    """
    MEC-2为主节点，MEC-3为跨故障域备用节点。

    主节点失效时，备用节点应成功接管。
    """

    config = load_config("configs/debug.yaml")

    planner = (
        build_reliability_aware_replica_planner(
            config
        )
    )

    simulator = build_test_simulator(
        replica_planner=planner,
        down_nodes_by_slot={
            3: {1},
        },
    )

    result = simulator.run()

    request_record = result.records[3]

    assert request_record.request_success is True

    # MEC-2的 node_id 为1，
    # MEC-3的 node_id 为2。
    assert (
        request_record
        .selected_execution_node_ids
        == (2,)
    )

    assert request_record.batch_failover is True
    assert request_record.failover_function_count == 1

    assert request_record.failover_delay_ms == pytest.approx(
        20.0
    )

    assert result.summary.failover_batches == 1
    assert result.summary.failed_requests == 0


def test_request_fails_when_primary_and_backup_are_down() -> None:
    """
    主节点和备用节点同时失效时，
    即使有双副本，请求仍然失败。
    """

    config = load_config("configs/debug.yaml")

    planner = (
        build_reliability_aware_replica_planner(
            config
        )
    )

    simulator = build_test_simulator(
        replica_planner=planner,
        down_nodes_by_slot={
            3: {1, 2},
        },
    )

    result = simulator.run()

    request_record = result.records[3]

    assert request_record.request_success is False
    assert request_record.unavailable_function_ids == (0,)

    assert result.summary.failed_requests == 1
    assert result.summary.failed_batches == 1


def test_hot_standby_uses_twice_the_memory() -> None:
    """
    单函数内存为100 MB。

    单副本：
        100 MB

    主备双副本：
        200 MB
    """

    config = load_config("configs/debug.yaml")

    single_simulator = build_test_simulator(
        replica_planner=SingleReplicaPlanner(),
    )

    redundant_simulator = build_test_simulator(
        replica_planner=(
            build_reliability_aware_replica_planner(
                config
            )
        ),
    )

    single_result = single_simulator.run()
    redundant_result = redundant_simulator.run()

    assert (
        single_result.records[0].hot_memory_mb
        == pytest.approx(100.0)
    )

    assert (
        redundant_result.records[0].hot_memory_mb
        == pytest.approx(200.0)
    )

    assert (
        redundant_result.summary
        .total_hot_memory_mb_seconds
        == pytest.approx(
            2.0
            * single_result.summary
            .total_hot_memory_mb_seconds
        )
    )