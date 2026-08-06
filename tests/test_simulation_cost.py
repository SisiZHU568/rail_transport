"""
test_simulation_cost.py

测试：

1. 预热命中统计；
2. 预热浪费统计；
3. 温容器内存成本；
4. 综合成本计算；
5. 过早预测带来的额外内存成本。
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
from src.simulation_cost import (
    SimulationCostWeights,
    analyze_simulation_cost,
)
from src.simulator import RailServerlessSFCSimulator
from src.topology import build_linear_topology
from src.trajectory_prediction import (
    BiasedTrajectoryPredictor,
    PerfectTrajectoryPredictor,
)
from src.workload import DeterministicWorkload


def build_test_function() -> ServerlessFunction:
    """
    使用单函数 SFC 简化成本测试。
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
        name="测试SFC",
        function_ids=[0],
        deadline_ms=500.0,
        reliability_target=0.99,
        priority=ServicePriority.CRITICAL,
    )


def build_test_simulator(
    policy: PrewarmingPolicy,
    request_trace: list[int],
) -> RailServerlessSFCSimulator:
    """
    创建测试仿真器。

    列车速度为 500 m/s，
    MEC-1 到 MEC-2 的切换边界为 1000 m。
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
        functions=[build_test_function()],
        sfc=build_test_sfc(),
        survival_slots=2,
        input_size_mb_per_request=1.0,
        return_result_to_source=True,
        prewarming_policy=policy,
    )


def build_cost_weights() -> SimulationCostWeights:
    """
    创建测试成本权重。
    """

    return SimulationCostWeights(
        delay_cost_per_ms=1.0,
        warm_memory_cost_per_mb_second=0.1,
        prewarm_start_cost_per_ms=2.0,
    )


def test_negative_cost_weight_is_rejected() -> None:
    """
    成本权重不能为负数。
    """

    with pytest.raises(ValueError):
        SimulationCostWeights(
            delay_cost_per_ms=-1.0,
            warm_memory_cost_per_mb_second=0.1,
            prewarm_start_cost_per_ms=1.0,
        )


def test_successful_prewarm_hit_is_recorded() -> None:
    """
    时隙1在 MEC-2 预热；
    时隙3请求到达 MEC-2。

    该预热应被记为一次命中。
    """

    simulator = build_test_simulator(
        policy=TrajectoryAwarePrewarmingPolicy(
            lead_time_s=1.0,
            predictor=PerfectTrajectoryPredictor(),
        ),
        request_trace=[0, 0, 0, 1],
    )

    result = simulator.run()

    assert result.records[1].prewarm_start_count == 1
    assert result.records[3].prewarm_hit_count == 1


def test_wrong_target_prewarm_becomes_waste() -> None:
    """
    将下一 MEC 错误向前预测一跳。

    接近 MEC-2 时，系统错误预热 MEC-3。
    该实例没有被请求使用，并最终过期。
    """

    simulator = build_test_simulator(
        policy=TrajectoryAwarePrewarmingPolicy(
            lead_time_s=1.0,
            predictor=BiasedTrajectoryPredictor(
                next_mec_hop_offset=1
            ),
        ),
        request_trace=[0, 0, 0, 1],
    )

    result = simulator.run()

    # 时隙1错误预热MEC-3。
    assert result.records[1].prewarm_start_count == 1

    # 时隙4达到生存窗口后被销毁。
    assert result.records[4].prewarm_waste_count == 1


def test_early_prediction_has_higher_retention_cost() -> None:
    """
    提前预测会更早创建温容器，
    因此内存占用成本应高于理想预测。
    """

    perfect_simulator = build_test_simulator(
        policy=TrajectoryAwarePrewarmingPolicy(
            lead_time_s=1.0,
            predictor=PerfectTrajectoryPredictor(),
        ),
        request_trace=[0],
    )

    early_simulator = build_test_simulator(
        policy=TrajectoryAwarePrewarmingPolicy(
            lead_time_s=1.0,
            predictor=BiasedTrajectoryPredictor(
                time_bias_s=-2.0
            ),
        ),
        request_trace=[0],
    )

    perfect_result = perfect_simulator.run()
    early_result = early_simulator.run()

    perfect_cost = analyze_simulation_cost(
        result=perfect_result,
        slot_seconds=1.0,
        weights=build_cost_weights(),
    )

    early_cost = analyze_simulation_cost(
        result=early_result,
        slot_seconds=1.0,
        weights=build_cost_weights(),
    )

    assert (
        early_cost.total_retention_cost
        > perfect_cost.total_retention_cost
    )


def test_total_cost_equals_three_cost_components() -> None:
    """
    综合成本应等于：

        时延成本
        + 内存保留成本
        + 预热启动成本
    """

    simulator = build_test_simulator(
        policy=NoPrewarmingPolicy(),
        request_trace=[1, 0, 0],
    )

    result = simulator.run()

    analysis = analyze_simulation_cost(
        result=result,
        slot_seconds=1.0,
        weights=build_cost_weights(),
    )

    expected_total = (
        analysis.total_delay_cost
        + analysis.total_retention_cost
        + analysis.total_prewarm_cost
    )

    assert analysis.total_system_cost == pytest.approx(
        expected_total
    )