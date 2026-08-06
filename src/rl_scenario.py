"""
rl_scenario.py

集中创建强化学习训练和评估使用的轨道边缘场景。
"""

from typing import Any

from src.entities import (
    ServerlessFunction,
    ServicePriority,
    SFCType,
)
from src.failure_risk_prediction import (
    build_windowed_failure_risk_provider,
)
from src.mobility import TrainMobilityModel
from src.network import build_linear_mec_network
from src.redundancy_placement import (
    build_reliability_aware_replica_planner,
)
from src.risk_aware_failure_process import (
    build_windowed_markov_failure_process,
)
from src.rl_reward import RLRewardWeights
from src.runtime_reliability import (
    SingleReplicaPlanner,
)
from src.slow_timescale_rl_env import (
    SlowTimescaleRLEnvironment,
)
from src.topology import build_linear_topology
from src.workload import DeterministicWorkload


def build_rl_functions() -> list[
    ServerlessFunction
]:
    """
    创建三函数故障诊断SFC。
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


def build_rl_sfc() -> SFCType:
    """
    创建高可靠列车故障诊断SFC。
    """

    return SFCType(
        sfc_id=0,
        name="高可靠列车设备故障诊断",
        function_ids=[0, 1, 2],
        deadline_ms=1500.0,
        reliability_target=0.99,
        priority=ServicePriority.CRITICAL,
    )


def build_rl_environment(
    config: dict[str, Any],
) -> SlowTimescaleRLEnvironment:
    """
    创建一个完整强化学习环境。
    """

    topology = build_linear_topology(
        config
    )

    network = build_linear_mec_network(
        config=config,
        topology=topology,
    )

    mobility_model = TrainMobilityModel(
        topology=topology,
        initial_position_m=float(
            config["train"][
                "initial_position_m"
            ]
        ),
        speed_mps=float(
            config["train"]["speed_mps"]
        ),
        slot_seconds=float(
            config["simulation"][
                "fast_slot_seconds"
            ]
        ),
    )

    workload = DeterministicWorkload(
        request_trace=(
            config[
                "integrated_simulation"
            ]["request_trace"]
        ),
        repeat=bool(
            config[
                "integrated_simulation"
            ]["repeat_request_trace"]
        ),
    )

    reward_config = config[
        "rl_environment"
    ]["reward_weights"]

    def failure_process_builder(
        seed: int,
    ):
        return (
            build_windowed_markov_failure_process(
                config=config,
                topology=topology,
                random_seed=seed,
            )
        )

    return SlowTimescaleRLEnvironment(
        topology=topology,
        network=network,
        mobility_model=mobility_model,
        workload=workload,
        functions=build_rl_functions(),
        sfc=build_rl_sfc(),
        single_replica_planner=(
            SingleReplicaPlanner()
        ),
        redundant_replica_planner=(
            build_reliability_aware_replica_planner(
                config
            )
        ),
        failure_process_builder=(
            failure_process_builder
        ),
        failure_risk_provider=(
            build_windowed_failure_risk_provider(
                config
            )
        ),
        slow_period_slots=int(
            config["two_timescale"][
                "slow_period_slots"
            ]
        ),
        handover_hot_window_s=float(
            config["two_timescale"][
                "handover_hot_window_s"
            ]
        ),
        input_size_mb_per_request=float(
            config[
                "integrated_simulation"
            ][
                "input_size_mb_per_request"
            ]
        ),
        slot_seconds=float(
            config["simulation"][
                "fast_slot_seconds"
            ]
        ),
        failover_delay_ms_per_function=float(
            config["runtime_failure"][
                "failover_delay_ms_per_function"
            ]
        ),
        reward_weights=RLRewardWeights(
            delay=float(
                reward_config["delay"]
            ),
            memory=float(
                reward_config["memory"]
            ),
            cold_start=float(
                reward_config[
                    "cold_start"
                ]
            ),
            sla_violation=float(
                reward_config[
                    "sla_violation"
                ]
            ),
            reconfiguration=float(
                reward_config[
                    "reconfiguration"
                ]
            ),
        ),
        default_seed=int(
            config["rl_environment"][
                "default_seed"
            ]
        ),
        return_result_to_source=True,
    )