"""
run_monte_carlo_experiment.py

使用多个随机种子比较四种可靠性保障方案：

1. 单副本；
2. 跨故障域冷备用；
3. 跨故障域全热备；
4. 轨迹感知动态主备。

所有方案在同一个随机种子下，
使用完全相同的基础设施故障轨迹。

输出：

1. 每次实验的原始CSV；
2. 均值、标准差和95%置信区间；
3. 请求成功率对比图；
4. SLA违反率对比图；
5. 平均时延对比图；
6. 平均活动内存对比图。
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from src.adaptive_runtime_reliability import (
    AdaptiveStandbyRuntimeSimulator,
)
from src.config import load_config
from src.entities import (
    ServerlessFunction,
    ServicePriority,
    SFCType,
)
from src.failure_process import (
    build_markov_failure_process,
)
from src.mobility import TrainMobilityModel
from src.monte_carlo import (
    MetricSummary,
    MonteCarloExperimentResult,
    MonteCarloScenarioSummary,
    run_paired_monte_carlo,
)
from src.network import build_linear_mec_network
from src.redundancy_placement import (
    build_reliability_aware_replica_planner,
)
from src.runtime_reliability import (
    SingleReplicaPlanner,
)
from src.standby_policy import (
    AllHotStandbyPolicy,
    PrimaryOnlyStandbyPolicy,
    TrajectoryAwareStandbyPolicy,
)
from src.topology import build_linear_topology
from src.workload import DeterministicWorkload


def build_demo_functions() -> list[ServerlessFunction]:
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


def build_simulator(
    config: dict,
    scenario_name: str,
    random_seed: int,
) -> AdaptiveStandbyRuntimeSimulator:
    """
    根据方案名称和随机种子创建独立仿真器。
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
            config["simulation"][
                "fast_slot_seconds"
            ]
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

    failure_process = (
        build_markov_failure_process(
            config=config,
            topology=topology,
            random_seed=random_seed,
        )
    )

    if scenario_name == "single_replica":
        replica_planner = SingleReplicaPlanner()

        standby_policy = (
            PrimaryOnlyStandbyPolicy()
        )

    elif scenario_name == "cold_standby":
        replica_planner = (
            build_reliability_aware_replica_planner(
                config
            )
        )

        standby_policy = (
            PrimaryOnlyStandbyPolicy()
        )

    elif scenario_name == "hot_standby":
        replica_planner = (
            build_reliability_aware_replica_planner(
                config
            )
        )

        standby_policy = (
            AllHotStandbyPolicy()
        )

    elif scenario_name == "trajectory_dynamic":
        replica_planner = (
            build_reliability_aware_replica_planner(
                config
            )
        )

        standby_policy = (
            TrajectoryAwareStandbyPolicy(
                lead_time_s=(
                    config["standby_mode_demo"][
                        "trajectory_lead_time_s"
                    ]
                )
            )
        )

    else:
        raise ValueError(
            f"未知方案：{scenario_name}"
        )

    return AdaptiveStandbyRuntimeSimulator(
        topology=topology,
        network=network,
        mobility_model=mobility_model,
        workload=workload,
        functions=build_demo_functions(),
        sfc=build_demo_sfc(),
        replica_planner=replica_planner,
        standby_policy=standby_policy,
        failure_process=failure_process,
        input_size_mb_per_request=(
            config["integrated_simulation"][
                "input_size_mb_per_request"
            ]
        ),
        slot_seconds=(
            config["simulation"][
                "fast_slot_seconds"
            ]
        ),
        failover_delay_ms_per_function=(
            config["runtime_failure"][
                "failover_delay_ms_per_function"
            ]
        ),
        return_result_to_source=True,
    )


def print_metric(
    name: str,
    metric: MetricSummary,
    unit: str = "",
) -> None:
    """
    打印一个指标的均值和置信区间。
    """

    unit_text = (
        f" {unit}" if unit else ""
    )

    print(
        f"  {name}："
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

    print(
        f"    最小值/最大值："
        f"{metric.minimum:.6f} / "
        f"{metric.maximum:.6f}"
        f"{unit_text}"
    )


def print_summary(
    summary: MonteCarloScenarioSummary,
) -> None:
    """
    分块打印一个方案的蒙特卡洛结果。
    """

    print("\n" + "=" * 72)
    print(f"方案：{summary.scenario_name}")
    print(f"独立随机故障轨迹数：{summary.run_count}")
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
        "平均活动实例内存",
        summary.average_active_memory_mb,
        "MB",
    )

    print_metric(
        "故障接管批次数",
        summary.failover_batches,
    )

    print_metric(
        "冷备用启动总时延",
        summary
        .total_cold_backup_startup_delay_ms,
        "ms",
    )


def save_raw_runs_csv(
    result: MonteCarloExperimentResult,
    output_path: Path,
) -> None:
    """
    保存每个方案、每个随机种子的原始结果。
    """

    fieldnames = [
        "scenario_name",
        "run_index",
        "random_seed",
        "request_success_rate",
        "sla_violation_rate",
        "average_successful_batch_delay_ms",
        "p95_successful_batch_delay_ms",
        "average_active_memory_mb",
        "peak_active_memory_mb",
        "failover_batches",
        "hot_failover_function_stages",
        "cold_failover_function_stages",
        "total_cold_backup_startup_delay_ms",
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


def metric_to_columns(
    prefix: str,
    metric: MetricSummary,
) -> dict[str, float | int]:
    """
    将一个指标转换成适合CSV保存的列。
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
    result: MonteCarloExperimentResult,
    output_path: Path,
) -> None:
    """
    保存统计汇总结果。
    """

    rows: list[
        dict[str, float | int | str]
    ] = []

    for summary in result.scenario_summaries:
        row: dict[
            str,
            float | int | str,
        ] = {
            "scenario_name": (
                summary.scenario_name
            ),
            "run_count": summary.run_count,
            "confidence_level": (
                result.confidence_level
            ),
        }

        row.update(
            metric_to_columns(
                "request_success_rate",
                summary.request_success_rate,
            )
        )

        row.update(
            metric_to_columns(
                "sla_violation_rate",
                summary.sla_violation_rate,
            )
        )

        row.update(
            metric_to_columns(
                "average_delay_ms",
                summary
                .average_successful_batch_delay_ms,
            )
        )

        row.update(
            metric_to_columns(
                "p95_delay_ms",
                summary
                .p95_successful_batch_delay_ms,
            )
        )

        row.update(
            metric_to_columns(
                "average_memory_mb",
                summary.average_active_memory_mb,
            )
        )

        row.update(
            metric_to_columns(
                "failover_batches",
                summary.failover_batches,
            )
        )

        row.update(
            metric_to_columns(
                "cold_backup_startup_delay_ms",
                summary
                .total_cold_backup_startup_delay_ms,
            )
        )

        rows.append(row)

    fieldnames = list(rows[0].keys())

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


def plot_metric_with_confidence_interval(
    summaries: tuple[
        MonteCarloScenarioSummary,
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

    metric_summaries = [
        metric_getter(summary)
        for summary in summaries
    ]

    means = [
        metric.mean
        for metric in metric_summaries
    ]

    error_sizes = [
        max(
            metric.ci_high - metric.mean,
            metric.mean - metric.ci_low,
        )
        for metric in metric_summaries
    ]

    figure, axis = plt.subplots(
        figsize=(10, 5)
    )

    axis.bar(
        scenario_names,
        means,
        yerr=error_sizes,
        capsize=5,
    )

    axis.set_title(title)
    axis.set_xlabel("Reliability Strategy")
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
    程序主入口。
    """

    project_root = Path(__file__).resolve().parent

    config = load_config(
        project_root
        / "configs"
        / "debug.yaml"
    )

    num_runs = int(
        config["monte_carlo"]["num_runs"]
    )

    base_seed = int(
        config["monte_carlo"]["base_seed"]
    )

    confidence_level = float(
        config["monte_carlo"][
            "confidence_level"
        ]
    )

    scenario_names = [
        "single_replica",
        "cold_standby",
        "hot_standby",
        "trajectory_dynamic",
    ]

    scenario_builders = {
        scenario_name: (
            lambda random_seed, name=scenario_name:
            build_simulator(
                config=config,
                scenario_name=name,
                random_seed=random_seed,
            )
        )
        for scenario_name in scenario_names
    }

    print("=" * 72)
    print("多随机种子蒙特卡洛可靠性实验")
    print("=" * 72)
    print(f"每个方案运行次数：{num_runs}")
    print(f"起始随机种子：{base_seed}")
    print(
        f"置信水平："
        f"{confidence_level:.2%}"
    )
    print(
        f"总仿真次数："
        f"{num_runs * len(scenario_names)}"
    )

    result = run_paired_monte_carlo(
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
        print_summary(summary)

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
        / "monte_carlo_raw_runs.csv"
    )

    summary_path = (
        tables_dir
        / "monte_carlo_summary.csv"
    )

    success_path = (
        figures_dir
        / "monte_carlo_success_rate.png"
    )

    sla_path = (
        figures_dir
        / "monte_carlo_sla_violation.png"
    )

    delay_path = (
        figures_dir
        / "monte_carlo_average_delay.png"
    )

    memory_path = (
        figures_dir
        / "monte_carlo_average_memory.png"
    )

    save_raw_runs_csv(
        result=result,
        output_path=raw_path,
    )

    save_summary_csv(
        result=result,
        output_path=summary_path,
    )

    summaries = result.scenario_summaries

    plot_metric_with_confidence_interval(
        summaries=summaries,
        metric_getter=(
            lambda summary:
            summary.request_success_rate
        ),
        title=(
            "Monte Carlo: Request Success Rate"
        ),
        y_label="Request Success Rate",
        output_path=success_path,
        rate_metric=True,
    )

    plot_metric_with_confidence_interval(
        summaries=summaries,
        metric_getter=(
            lambda summary:
            summary.sla_violation_rate
        ),
        title=(
            "Monte Carlo: SLA Violation Rate"
        ),
        y_label="SLA Violation Rate",
        output_path=sla_path,
        rate_metric=True,
    )

    plot_metric_with_confidence_interval(
        summaries=summaries,
        metric_getter=(
            lambda summary:
            summary
            .average_successful_batch_delay_ms
        ),
        title=(
            "Monte Carlo: Average Successful Delay"
        ),
        y_label="Average Delay (ms)",
        output_path=delay_path,
    )

    plot_metric_with_confidence_interval(
        summaries=summaries,
        metric_getter=(
            lambda summary:
            summary.average_active_memory_mb
        ),
        title=(
            "Monte Carlo: Average Active Memory"
        ),
        y_label="Average Active Memory (MB)",
        output_path=memory_path,
    )

    print("\n" + "=" * 72)
    print("蒙特卡洛实验结果已保存")
    print("=" * 72)
    print(f"逐次原始结果：{raw_path}")
    print(f"统计汇总结果：{summary_path}")
    print(f"请求成功率图：{success_path}")
    print(f"SLA违反率图：{sla_path}")
    print(f"平均时延图：{delay_path}")
    print(f"平均内存图：{memory_path}")


if __name__ == "__main__":
    main()