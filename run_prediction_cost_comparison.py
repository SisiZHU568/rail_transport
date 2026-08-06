"""
run_prediction_cost_comparison.py

比较不同轨迹预测质量下的：

1. 用户冷启动；
2. 预热命中；
3. 预热浪费；
4. 用户请求时延成本；
5. 温容器内存成本；
6. 后台预热启动成本；
7. 系统综合成本。

比较场景：

1. 被动响应式；
2. 理想无误差预测；
3. 提前4秒预测；
4. 滞后4秒预测；
5. 下一MEC预测错误1跳。
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
    SimulationCostAnalysis,
    SimulationCostWeights,
    analyze_simulation_cost,
)
from src.simulator import (
    RailServerlessSFCSimulator,
    SimulationRunResult,
)
from src.topology import build_linear_topology
from src.trajectory_prediction import (
    BiasedTrajectoryPredictor,
    PerfectTrajectoryPredictor,
)
from src.workload import DeterministicWorkload


def build_demo_functions() -> list[ServerlessFunction]:
    """
    创建故障诊断 SFC 的三个函数。
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


def build_demo_sfc() -> SFCType:
    """
    创建故障诊断 SFC。
    """

    return SFCType(
        sfc_id=0,
        name="列车设备故障诊断",
        function_ids=[0, 1, 2],
        deadline_ms=1500.0,
        reliability_target=0.99,
        priority=ServicePriority.CRITICAL,
    )


def build_simulator(
    config: dict,
    policy: PrewarmingPolicy,
) -> RailServerlessSFCSimulator:
    """
    为每种策略创建独立仿真器。
    """

    topology = build_linear_topology(config)

    network = build_linear_mec_network(
        config=config,
        topology=topology,
    )

    mobility_model = TrainMobilityModel(
        topology=topology,
        initial_position_m=(
            config["train"]["initial_position_m"]
        ),
        speed_mps=config["train"]["speed_mps"],
        slot_seconds=(
            config["simulation"]["fast_slot_seconds"]
        ),
    )

    workload = DeterministicWorkload(
        request_trace=(
            config["integrated_simulation"][
                "request_trace"
            ]
        ),
        repeat=(
            config["integrated_simulation"][
                "repeat_request_trace"
            ]
        ),
    )

    return RailServerlessSFCSimulator(
        topology=topology,
        network=network,
        mobility_model=mobility_model,
        workload=workload,
        placement_policy=(
            ServingMECPlacementPolicy()
        ),
        functions=build_demo_functions(),
        sfc=build_demo_sfc(),
        survival_slots=(
            config["integrated_simulation"][
                "survival_slots"
            ]
        ),
        input_size_mb_per_request=(
            config["integrated_simulation"][
                "input_size_mb_per_request"
            ]
        ),
        return_result_to_source=True,
        prewarming_policy=policy,
    )


def print_result(
    scenario_name: str,
    policy: PrewarmingPolicy,
    result: SimulationRunResult,
    cost: SimulationCostAnalysis,
) -> None:
    """
    分块打印一种预测场景的结果。
    """

    summary = result.summary

    print("\n" + "=" * 72)
    print(f"场景：{scenario_name}")
    print(f"策略：{policy.display_name}")
    print("=" * 72)

    print(f"  用户冷启动次数：{summary.user_cold_starts}")
    print(f"  主动预热次数：{cost.prewarm_starts}")
    print(f"  预热命中次数：{cost.prewarm_hits}")
    print(f"  预热浪费次数：{cost.prewarm_wastes}")
    print(f"  未完成验证的预热：{cost.unresolved_prewarms}")

    print(
        f"  预热命中率："
        f"{cost.prewarm_hit_rate:.4f}"
    )

    print(
        f"  预热浪费率："
        f"{cost.prewarm_waste_rate:.4f}"
    )

    print(
        f"  时延违反次数："
        f"{summary.deadline_violations}"
    )

    print(
        f"  平均端到端时延："
        f"{summary.average_batch_delay_ms:.3f} ms"
    )

    print(
        f"  P95端到端时延："
        f"{summary.p95_batch_delay_ms:.3f} ms"
    )

    print(
        f"  平均温容器内存："
        f"{cost.average_warm_memory_mb:.3f} MB"
    )

    print(
        f"  峰值温容器内存："
        f"{cost.peak_warm_memory_mb:.3f} MB"
    )

    print(
        f"  用户时延成本："
        f"{cost.total_delay_cost:.3f}"
    )

    print(
        f"  温容器内存成本："
        f"{cost.total_retention_cost:.3f}"
    )

    print(
        f"  后台预热成本："
        f"{cost.total_prewarm_cost:.3f}"
    )

    print(
        f"  系统综合成本："
        f"{cost.total_system_cost:.3f}"
    )


def save_summary_csv(
    scenario_results: list[
        tuple[
            str,
            PrewarmingPolicy,
            SimulationRunResult,
            SimulationCostAnalysis,
        ]
    ],
    output_path: Path,
) -> None:
    """
    保存所有场景的汇总结果。
    """

    fieldnames = [
        "scenario",
        "policy_id",
        "user_cold_starts",
        "prewarm_starts",
        "prewarm_hits",
        "prewarm_wastes",
        "unresolved_prewarms",
        "prewarm_hit_rate",
        "prewarm_waste_rate",
        "deadline_violations",
        "average_batch_delay_ms",
        "p95_batch_delay_ms",
        "average_warm_memory_mb",
        "peak_warm_memory_mb",
        "total_delay_cost",
        "total_retention_cost",
        "total_prewarm_cost",
        "total_system_cost",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for (
            scenario_name,
            policy,
            result,
            cost,
        ) in scenario_results:
            writer.writerow(
                {
                    "scenario": scenario_name,
                    "policy_id": policy.policy_id,
                    "user_cold_starts": (
                        result.summary.user_cold_starts
                    ),
                    "prewarm_starts": cost.prewarm_starts,
                    "prewarm_hits": cost.prewarm_hits,
                    "prewarm_wastes": (
                        cost.prewarm_wastes
                    ),
                    "unresolved_prewarms": (
                        cost.unresolved_prewarms
                    ),
                    "prewarm_hit_rate": (
                        cost.prewarm_hit_rate
                    ),
                    "prewarm_waste_rate": (
                        cost.prewarm_waste_rate
                    ),
                    "deadline_violations": (
                        result.summary.deadline_violations
                    ),
                    "average_batch_delay_ms": (
                        result.summary
                        .average_batch_delay_ms
                    ),
                    "p95_batch_delay_ms": (
                        result.summary.p95_batch_delay_ms
                    ),
                    "average_warm_memory_mb": (
                        cost.average_warm_memory_mb
                    ),
                    "peak_warm_memory_mb": (
                        cost.peak_warm_memory_mb
                    ),
                    "total_delay_cost": (
                        cost.total_delay_cost
                    ),
                    "total_retention_cost": (
                        cost.total_retention_cost
                    ),
                    "total_prewarm_cost": (
                        cost.total_prewarm_cost
                    ),
                    "total_system_cost": (
                        cost.total_system_cost
                    ),
                }
            )


def plot_total_cost(
    scenario_results: list[
        tuple[
            str,
            PrewarmingPolicy,
            SimulationRunResult,
            SimulationCostAnalysis,
        ]
    ],
    output_path: Path,
) -> None:
    """
    绘制不同预测场景的系统综合成本。
    """

    scenario_names = [
        scenario_name
        for scenario_name, _, _, _
        in scenario_results
    ]

    total_costs = [
        cost.total_system_cost
        for _, _, _, cost
        in scenario_results
    ]

    figure, axis = plt.subplots(figsize=(10, 5))

    axis.bar(
        scenario_names,
        total_costs,
    )

    axis.set_title(
        "Prediction Error: Total System Cost"
    )
    axis.set_xlabel("Prediction Scenario")
    axis.set_ylabel("Normalized Total Cost")
    axis.tick_params(
        axis="x",
        rotation=20,
    )
    axis.grid(axis="y", alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_prewarm_quality(
    scenario_results: list[
        tuple[
            str,
            PrewarmingPolicy,
            SimulationRunResult,
            SimulationCostAnalysis,
        ]
    ],
    output_path: Path,
) -> None:
    """
    绘制预热命中次数和浪费次数。
    """

    scenario_names = [
        scenario_name
        for scenario_name, _, _, _
        in scenario_results
    ]

    hits = [
        cost.prewarm_hits
        for _, _, _, cost
        in scenario_results
    ]

    wastes = [
        cost.prewarm_wastes
        for _, _, _, cost
        in scenario_results
    ]

    x_positions = np.arange(
        len(scenario_names)
    )

    bar_width = 0.35

    figure, axis = plt.subplots(figsize=(10, 5))

    axis.bar(
        x_positions - bar_width / 2,
        hits,
        width=bar_width,
        label="Prewarm hits",
    )

    axis.bar(
        x_positions + bar_width / 2,
        wastes,
        width=bar_width,
        label="Prewarm wastes",
    )

    axis.set_title(
        "Prediction Error: Prewarm Quality"
    )
    axis.set_xlabel("Prediction Scenario")
    axis.set_ylabel("Function Instances")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(
        scenario_names,
        rotation=20,
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def main() -> None:
    """
    程序主入口。
    """

    project_root = Path(__file__).resolve().parent

    config = load_config(
        project_root / "configs" / "debug.yaml"
    )

    lead_time_s = (
        config["prewarming_demo"]["lead_time_s"]
    )

    scenarios: list[
        tuple[str, PrewarmingPolicy]
    ] = [
        (
            "reactive",
            NoPrewarmingPolicy(),
        ),
        (
            "perfect",
            TrajectoryAwarePrewarmingPolicy(
                lead_time_s=lead_time_s,
                predictor=(
                    PerfectTrajectoryPredictor()
                ),
            ),
        ),
        (
            "early",
            TrajectoryAwarePrewarmingPolicy(
                lead_time_s=lead_time_s,
                predictor=BiasedTrajectoryPredictor(
                    time_bias_s=(
                        config["prediction_cost_demo"][
                            "early_time_bias_s"
                        ]
                    )
                ),
            ),
        ),
        (
            "late",
            TrajectoryAwarePrewarmingPolicy(
                lead_time_s=lead_time_s,
                predictor=BiasedTrajectoryPredictor(
                    time_bias_s=(
                        config["prediction_cost_demo"][
                            "late_time_bias_s"
                        ]
                    )
                ),
            ),
        ),
        (
            "wrong_target",
            TrajectoryAwarePrewarmingPolicy(
                lead_time_s=lead_time_s,
                predictor=BiasedTrajectoryPredictor(
                    next_mec_hop_offset=(
                        config["prediction_cost_demo"][
                            "wrong_mec_hop_offset"
                        ]
                    )
                ),
            ),
        ),
    ]

    weights = SimulationCostWeights(
        delay_cost_per_ms=(
            config["simulation_cost"][
                "delay_cost_per_ms"
            ]
        ),
        warm_memory_cost_per_mb_second=(
            config["simulation_cost"][
                "warm_memory_cost_per_mb_second"
            ]
        ),
        prewarm_start_cost_per_ms=(
            config["simulation_cost"][
                "prewarm_start_cost_per_ms"
            ]
        ),
    )

    slot_seconds = (
        config["simulation"]["fast_slot_seconds"]
    )

    scenario_results: list[
        tuple[
            str,
            PrewarmingPolicy,
            SimulationRunResult,
            SimulationCostAnalysis,
        ]
    ] = []

    for scenario_name, policy in scenarios:
        simulator = build_simulator(
            config=config,
            policy=policy,
        )

        result = simulator.run()

        cost = analyze_simulation_cost(
            result=result,
            slot_seconds=slot_seconds,
            weights=weights,
        )

        scenario_results.append(
            (
                scenario_name,
                policy,
                result,
                cost,
            )
        )

        print_result(
            scenario_name=scenario_name,
            policy=policy,
            result=result,
            cost=cost,
        )

    tables_dir = project_root / "results" / "tables"
    figures_dir = project_root / "results" / "figures"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_path = (
        tables_dir
        / "prediction_cost_summary.csv"
    )

    total_cost_figure_path = (
        figures_dir
        / "prediction_total_cost.png"
    )

    quality_figure_path = (
        figures_dir
        / "prediction_prewarm_quality.png"
    )

    save_summary_csv(
        scenario_results=scenario_results,
        output_path=summary_path,
    )

    plot_total_cost(
        scenario_results=scenario_results,
        output_path=total_cost_figure_path,
    )

    plot_prewarm_quality(
        scenario_results=scenario_results,
        output_path=quality_figure_path,
    )

    print("\n" + "=" * 72)
    print("实验结果已保存")
    print("=" * 72)
    print(f"汇总表：{summary_path}")
    print(f"综合成本图：{total_cost_figure_path}")
    print(f"预热质量图：{quality_figure_path}")


if __name__ == "__main__":
    main()