"""
run_two_timescale_monte_carlo.py

使用多个随机种子比较五种控制方案：

1. fixed_single
2. fixed_cold
3. fixed_hot
4. fixed_dynamic
5. two_timescale

高风险窗口中：

1. 风险预测器向双时间尺度控制器报告高风险；
2. 随机故障过程同步提高真实失效概率。

所有方案在相同种子下使用完全相同的随机故障轨迹。
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from src.config import load_config
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
from src.reliability import (
    build_fault_domain_reliability_model,
)
from src.risk_aware_failure_process import (
    build_windowed_markov_failure_process,
)
from src.runtime_reliability import (
    SingleReplicaPlanner,
)
from src.topology import build_linear_topology
from src.two_timescale_control import (
    FixedModeTwoTimescaleController,
    StandbyMode,
    build_rule_based_two_timescale_controller,
)
from src.two_timescale_monte_carlo import (
    TwoTimescaleMonteCarloResult,
    TwoTimescaleMonteCarloScenarioSummary,
    run_two_timescale_monte_carlo,
)
from src.two_timescale_simulator import (
    TwoTimescaleCostWeights,
    TwoTimescaleRuntimeSimulator,
)
from src.workload import DeterministicWorkload


def build_demo_functions() -> list[
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


def build_demo_sfc() -> SFCType:
    """
    创建高可靠故障诊断SFC。
    """

    return SFCType(
        sfc_id=0,
        name="高可靠列车设备故障诊断",
        function_ids=[0, 1, 2],
        deadline_ms=1500.0,
        reliability_target=0.99,
        priority=ServicePriority.CRITICAL,
    )


def build_cost_weights(
    config: dict,
) -> TwoTimescaleCostWeights:
    """
    创建综合成本权重。
    """

    cost_config = config[
        "two_timescale_cost"
    ]

    return TwoTimescaleCostWeights(
        delay_cost_per_ms=float(
            cost_config[
                "delay_cost_per_ms"
            ]
        ),
        memory_cost_per_mb_second=float(
            cost_config[
                "memory_cost_per_mb_second"
            ]
        ),
        cold_start_cost_per_ms=float(
            cost_config[
                "cold_start_cost_per_ms"
            ]
        ),
        sla_violation_penalty_per_batch=float(
            cost_config[
                "sla_violation_penalty_per_batch"
            ]
        ),
        slow_decision_cost=float(
            cost_config[
                "slow_decision_cost"
            ]
        ),
        replica_reconfiguration_cost_per_function=float(
            cost_config[
                "replica_reconfiguration_cost_per_function"
            ]
        ),
    )


def build_controller(
    config: dict,
    scenario_name: str,
):
    """
    根据方案名称创建全新的控制器。
    """

    handover_window = float(
        config["two_timescale"][
            "handover_hot_window_s"
        ]
    )

    if scenario_name == "fixed_single":
        return FixedModeTwoTimescaleController(
            standby_mode=StandbyMode.SINGLE,
            handover_hot_window_s=(
                handover_window
            ),
        )

    if scenario_name == "fixed_cold":
        return FixedModeTwoTimescaleController(
            standby_mode=StandbyMode.COLD,
            handover_hot_window_s=(
                handover_window
            ),
            enable_handover_activation=False,
        )

    if scenario_name == "fixed_hot":
        return FixedModeTwoTimescaleController(
            standby_mode=StandbyMode.HOT,
            handover_hot_window_s=(
                handover_window
            ),
        )

    if scenario_name == "fixed_dynamic":
        return FixedModeTwoTimescaleController(
            standby_mode=StandbyMode.COLD,
            handover_hot_window_s=(
                handover_window
            ),
            enable_handover_activation=True,
        )

    if scenario_name == "two_timescale":
        return (
            build_rule_based_two_timescale_controller(
                config
            )
        )

    raise ValueError(
        f"未知方案：{scenario_name}"
    )


def build_simulator(
    config: dict,
    scenario_name: str,
    random_seed: int,
) -> TwoTimescaleRuntimeSimulator:
    """
    创建一个方案在指定随机种子下的仿真器。
    """

    topology = build_linear_topology(config)

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

    failure_process = (
        build_windowed_markov_failure_process(
            config=config,
            topology=topology,
            random_seed=random_seed,
        )
    )

    return TwoTimescaleRuntimeSimulator(
        topology=topology,
        network=network,
        mobility_model=mobility_model,
        workload=workload,
        functions=build_demo_functions(),
        sfc=build_demo_sfc(),
        controller=build_controller(
            config=config,
            scenario_name=scenario_name,
        ),
        single_replica_planner=(
            SingleReplicaPlanner()
        ),
        redundant_replica_planner=(
            build_reliability_aware_replica_planner(
                config
            )
        ),
        failure_process=failure_process,
        failure_risk_provider=(
            build_windowed_failure_risk_provider(
                config
            )
        ),
        reliability_model=(
            build_fault_domain_reliability_model(
                config=config,
                topology=topology,
            )
        ),
        prediction_horizon_slots=int(
            config["two_timescale"][
                "slow_period_slots"
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
        cost_weights=build_cost_weights(
            config
        ),
        return_result_to_source=True,
    )


def print_metric(
    metric_name: str,
    metric,
    unit: str = "",
) -> None:
    """
    打印均值、标准差和置信区间。
    """

    unit_text = (
        f" {unit}" if unit else ""
    )

    print(
        f"  {metric_name}均值："
        f"{metric.mean:.6f}{unit_text}"
    )

    print(
        f"    标准差："
        f"{metric.std:.6f}{unit_text}"
    )

    print(
        f"    95%置信区间："
        f"[{metric.ci_low:.6f}, "
        f"{metric.ci_high:.6f}]"
        f"{unit_text}"
    )


def print_scenario_summary(
    summary: TwoTimescaleMonteCarloScenarioSummary,
) -> None:
    """
    打印一种方案的统计结果。
    """

    print("\n" + "=" * 72)
    print(f"方案：{summary.scenario_name}")
    print(f"随机故障轨迹数：{summary.run_count}")
    print("=" * 72)

    print_metric(
        "请求成功率",
        summary.request_success_rate,
    )

    print_metric(
        "SLA违反率",
        summary.sla_violation_rate,
    )

    print_metric(
        "成功批次平均时延",
        summary
        .average_successful_batch_delay_ms,
        "ms",
    )

    print_metric(
        "成功批次P95时延",
        summary
        .p95_successful_batch_delay_ms,
        "ms",
    )

    print_metric(
        "平均活动内存",
        summary.average_active_memory_mb,
        "MB",
    )

    print_metric(
        "冷启动函数阶段数",
        summary.cold_start_function_stages,
    )

    print_metric(
        "系统综合成本",
        summary.total_system_cost,
    )


def save_raw_csv(
    result: TwoTimescaleMonteCarloResult,
    output_path: Path,
) -> None:
    """
    保存所有逐次实验结果。
    """

    if not result.run_records:
        raise ValueError(
            "没有可保存的原始结果。"
        )

    fieldnames = list(
        result.run_records[0]
        .__dataclass_fields__
        .keys()
    )

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

        for record in result.run_records:
            writer.writerow(
                {
                    field_name: getattr(
                        record,
                        field_name,
                    )
                    for field_name in fieldnames
                }
            )


def metric_columns(
    prefix: str,
    metric,
) -> dict[str, float | int]:
    """
    将一个统计指标转换为CSV列。
    """

    return {
        f"{prefix}_count": metric.count,
        f"{prefix}_mean": metric.mean,
        f"{prefix}_std": metric.std,
        f"{prefix}_ci_low": metric.ci_low,
        f"{prefix}_ci_high": metric.ci_high,
        f"{prefix}_min": metric.minimum,
        f"{prefix}_max": metric.maximum,
    }


def save_summary_csv(
    result: TwoTimescaleMonteCarloResult,
    output_path: Path,
) -> None:
    """
    保存方案统计汇总。
    """

    rows: list[
        dict[str, str | int | float]
    ] = []

    for summary in result.scenario_summaries:
        row: dict[
            str,
            str | int | float,
        ] = {
            "scenario_name": (
                summary.scenario_name
            ),
            "run_count": summary.run_count,
            "confidence_level": (
                result.confidence_level
            ),
        }

        metric_mapping = {
            "request_success_rate": (
                summary.request_success_rate
            ),
            "sla_violation_rate": (
                summary.sla_violation_rate
            ),
            "average_delay_ms": (
                summary
                .average_successful_batch_delay_ms
            ),
            "p95_delay_ms": (
                summary
                .p95_successful_batch_delay_ms
            ),
            "average_memory_mb": (
                summary.average_active_memory_mb
            ),
            "cold_start_stages": (
                summary.cold_start_function_stages
            ),
            "request_delay_cost": (
                summary.total_request_delay_cost
            ),
            "memory_cost": (
                summary.total_memory_cost
            ),
            "cold_start_cost": (
                summary.total_cold_start_cost
            ),
            "sla_penalty": (
                summary.total_sla_penalty
            ),
            "slow_control_cost": (
                summary.total_slow_control_cost
            ),
            "total_system_cost": (
                summary.total_system_cost
            ),
        }

        for prefix, metric in (
            metric_mapping.items()
        ):
            row.update(
                metric_columns(
                    prefix=prefix,
                    metric=metric,
                )
            )

        rows.append(row)

    fieldnames = list(
        rows[0].keys()
    )

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
        writer.writerows(rows)


def plot_metric(
    summaries: tuple[
        TwoTimescaleMonteCarloScenarioSummary,
        ...,
    ],
    metric_getter,
    title: str,
    y_label: str,
    output_path: Path,
    rate_metric: bool = False,
) -> None:
    """
    绘制带95%置信区间误差棒的柱状图。
    """

    scenario_names = [
        summary.scenario_name
        for summary in summaries
    ]

    metrics = [
        metric_getter(summary)
        for summary in summaries
    ]

    means = [
        metric.mean
        for metric in metrics
    ]

    error_sizes = [
        max(
            metric.mean - metric.ci_low,
            metric.ci_high - metric.mean,
        )
        for metric in metrics
    ]

    figure, axis = plt.subplots(
        figsize=(11, 5)
    )

    axis.bar(
        scenario_names,
        means,
        yerr=error_sizes,
        capsize=5,
    )

    axis.set_title(title)
    axis.set_xlabel("Control Strategy")
    axis.set_ylabel(y_label)

    axis.tick_params(
        axis="x",
        rotation=15,
    )

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    if rate_metric:
        axis.set_ylim(0.0, 1.05)

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
    )

    plt.close(figure)


def main() -> None:
    """
    程序入口。
    """

    project_root = (
        Path(__file__).resolve().parent
    )

    config = load_config(
        project_root
        / "configs"
        / "debug.yaml"
    )

    monte_carlo_config = config[
        "two_timescale_monte_carlo"
    ]

    num_runs = int(
        monte_carlo_config["num_runs"]
    )

    base_seed = int(
        monte_carlo_config["base_seed"]
    )

    confidence_level = float(
        monte_carlo_config[
            "confidence_level"
        ]
    )

    scenario_names = [
        "fixed_single",
        "fixed_cold",
        "fixed_hot",
        "fixed_dynamic",
        "two_timescale",
    ]

    scenario_builders = {
        scenario_name: (
            lambda seed, name=scenario_name:
            build_simulator(
                config=config,
                scenario_name=name,
                random_seed=seed,
            )
        )
        for scenario_name in scenario_names
    }

    print("=" * 72)
    print("双时间尺度多随机种子蒙特卡洛实验")
    print("=" * 72)

    print(f"每种方案运行次数：{num_runs}")
    print(f"起始随机种子：{base_seed}")

    print(
        f"置信水平："
        f"{confidence_level:.2%}"
    )

    print(
        f"总完整仿真次数："
        f"{num_runs * len(scenario_names)}"
    )

    result = run_two_timescale_monte_carlo(
        scenario_builders=(
            scenario_builders
        ),
        num_runs=num_runs,
        base_seed=base_seed,
        confidence_level=(
            confidence_level
        ),
    )

    for summary in result.scenario_summaries:
        print_scenario_summary(summary)

    tables_dir = (
        project_root / "results" / "tables"
    )

    figures_dir = (
        project_root / "results" / "figures"
    )

    tables_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        tables_dir
        / "two_timescale_monte_carlo_raw.csv"
    )

    summary_path = (
        tables_dir
        / "two_timescale_monte_carlo_summary.csv"
    )

    success_path = (
        figures_dir
        / "two_timescale_mc_success_rate.png"
    )

    sla_path = (
        figures_dir
        / "two_timescale_mc_sla_violation.png"
    )

    delay_path = (
        figures_dir
        / "two_timescale_mc_average_delay.png"
    )

    memory_path = (
        figures_dir
        / "two_timescale_mc_average_memory.png"
    )

    cost_path = (
        figures_dir
        / "two_timescale_mc_total_cost.png"
    )

    save_raw_csv(
        result=result,
        output_path=raw_path,
    )

    save_summary_csv(
        result=result,
        output_path=summary_path,
    )

    summaries = result.scenario_summaries

    plot_metric(
        summaries=summaries,
        metric_getter=(
            lambda summary:
            summary.request_success_rate
        ),
        title=(
            "Two-Timescale Monte Carlo: "
            "Request Success Rate"
        ),
        y_label="Request Success Rate",
        output_path=success_path,
        rate_metric=True,
    )

    plot_metric(
        summaries=summaries,
        metric_getter=(
            lambda summary:
            summary.sla_violation_rate
        ),
        title=(
            "Two-Timescale Monte Carlo: "
            "SLA Violation Rate"
        ),
        y_label="SLA Violation Rate",
        output_path=sla_path,
        rate_metric=True,
    )

    plot_metric(
        summaries=summaries,
        metric_getter=(
            lambda summary:
            summary
            .average_successful_batch_delay_ms
        ),
        title=(
            "Two-Timescale Monte Carlo: "
            "Average Successful Delay"
        ),
        y_label="Average Delay (ms)",
        output_path=delay_path,
    )

    plot_metric(
        summaries=summaries,
        metric_getter=(
            lambda summary:
            summary.average_active_memory_mb
        ),
        title=(
            "Two-Timescale Monte Carlo: "
            "Average Active Memory"
        ),
        y_label="Average Active Memory (MB)",
        output_path=memory_path,
    )

    plot_metric(
        summaries=summaries,
        metric_getter=(
            lambda summary:
            summary.total_system_cost
        ),
        title=(
            "Two-Timescale Monte Carlo: "
            "Total System Cost"
        ),
        y_label="Normalized Total Cost",
        output_path=cost_path,
    )

    print("\n" + "=" * 72)
    print("实验结果已保存")
    print("=" * 72)

    print(f"逐次结果：{raw_path}")
    print(f"统计汇总：{summary_path}")
    print(f"请求成功率图：{success_path}")
    print(f"SLA违反率图：{sla_path}")
    print(f"平均时延图：{delay_path}")
    print(f"平均内存图：{memory_path}")
    print(f"综合成本图：{cost_path}")


if __name__ == "__main__":
    main()
