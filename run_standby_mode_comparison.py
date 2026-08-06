"""
run_standby_mode_comparison.py

在完全相同的可控故障轨迹下比较：

1. 单副本；
2. 跨故障域冷备用；
3. 跨故障域全热备；
4. 轨迹感知动态主备。

比较指标：

1. 请求成功率；
2. 热备用和冷备用接管次数；
3. 冷备用启动时延；
4. SLA违反率；
5. 平均成功请求时延；
6. 平均活动实例内存。
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from src.adaptive_runtime_reliability import (
    AdaptiveReliabilityResult,
    AdaptiveStandbyRuntimeSimulator,
)
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
    SingleReplicaPlanner,
)
from src.standby_policy import (
    AllHotStandbyPolicy,
    PrimaryOnlyStandbyPolicy,
    StandbyActivationPolicy,
    TrajectoryAwareStandbyPolicy,
)
from src.topology import build_linear_topology
from src.workload import DeterministicWorkload


def build_demo_functions() -> list[ServerlessFunction]:
    """
    创建故障诊断SFC的三个函数。
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


def build_scripted_failure_process(
    config: dict,
    topology,
) -> ScriptedFailureProcess:
    """
    根据配置创建可复现的故障过程。
    """

    down_nodes_by_slot: dict[
        int,
        set[int],
    ] = {}

    for item in config["standby_mode_demo"][
        "scripted_node_failures"
    ]:
        time_slot = int(item["time_slot"])
        node_id = int(item["node_id"])

        down_nodes_by_slot.setdefault(
            time_slot,
            set(),
        ).add(node_id)

    down_domains_by_slot: dict[
        int,
        set[int],
    ] = {}

    for item in config["standby_mode_demo"][
        "scripted_domain_failures"
    ]:
        time_slot = int(item["time_slot"])
        domain_id = int(item["domain_id"])

        down_domains_by_slot.setdefault(
            time_slot,
            set(),
        ).add(domain_id)

    return ScriptedFailureProcess(
        topology=topology,
        down_domains_by_slot=(
            down_domains_by_slot
        ),
        down_nodes_by_slot=down_nodes_by_slot,
    )


def build_simulator(
    config: dict,
    replica_planner,
    standby_policy: StandbyActivationPolicy,
) -> AdaptiveStandbyRuntimeSimulator:
    """
    创建一种主备模式的独立仿真器。
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

    failure_process = (
        build_scripted_failure_process(
            config=config,
            topology=topology,
        )
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


def print_result(
    scenario_name: str,
    result: AdaptiveReliabilityResult,
) -> None:
    """
    分块打印一个方案的汇总结果。
    """

    summary = result.summary

    print("\n" + "=" * 72)
    print(f"方案：{scenario_name}")
    print("=" * 72)

    print(f"  总请求数量：{summary.total_requests}")
    print(f"  成功请求数量：{summary.successful_requests}")
    print(f"  失败请求数量：{summary.failed_requests}")

    print(
        f"  请求成功率："
        f"{summary.request_success_rate:.6f}"
    )

    print(
        f"  故障接管批次数："
        f"{summary.failover_batches}"
    )

    print(
        f"  温备用接管函数数："
        f"{summary.hot_failover_function_stages}"
    )

    print(
        f"  冷备用接管函数数："
        f"{summary.cold_failover_function_stages}"
    )

    print(
        f"  故障切换总时延："
        f"{summary.total_failover_delay_ms:.3f} ms"
    )

    print(
        f"  冷备用启动总时延："
        f"{summary.total_cold_backup_startup_delay_ms:.3f} ms"
    )

    print(
        f"  SLA违反率："
        f"{summary.sla_violation_rate:.6f}"
    )

    print(
        f"  成功批次平均时延："
        f"{summary.average_successful_batch_delay_ms:.3f} ms"
    )

    print(
        f"  成功批次P95时延："
        f"{summary.p95_successful_batch_delay_ms:.3f} ms"
    )

    print(
        f"  平均活动实例内存："
        f"{summary.average_active_memory_mb:.3f} MB"
    )

    print(
        f"  峰值活动实例内存："
        f"{summary.peak_active_memory_mb:.3f} MB"
    )


def print_failure_events(
    scenario_name: str,
    result: AdaptiveReliabilityResult,
) -> None:
    """
    打印有请求且出现故障的关键时隙。
    """

    print("\n" + "=" * 72)
    print(f"{scenario_name}：关键故障事件")
    print("=" * 72)

    event_count = 0

    for record in result.records:
        important = (
            record.request_count > 0
            and (
                record.down_domain_ids
                or record.down_node_ids
                or record.failover_function_count > 0
                or record.request_success is False
            )
        )

        if not important:
            continue

        event_count += 1

        if record.request_success is True:
            state_text = "成功"
        elif record.request_success is False:
            state_text = "失败"
        else:
            state_text = "无请求"

        print(f"\n事件 {event_count}")
        print("-" * 72)

        print(
            f"  时隙与位置："
            f"{record.time_slot}，"
            f"{record.position_m:.2f} m"
        )

        print(
            f"  当前接入："
            f"MEC-{record.serving_mec + 1}"
        )

        print(
            f"  剩余驻留时间："
            f"{record.remaining_dwell_time_s:.3f} s"
        )

        print(
            f"  请求数量："
            f"{record.request_count}"
        )

        print(
            f"  失效故障域："
            f"{list(record.down_domain_ids)}"
        )

        print(
            f"  局部失效节点："
            f"{[node_id + 1 for node_id in record.down_node_ids]}"
        )

        print(
            f"  温备用接管函数数："
            f"{record.hot_failover_function_count}"
        )

        print(
            f"  冷备用接管函数数："
            f"{record.cold_failover_function_count}"
        )

        print(
            f"  冷备用启动时延："
            f"{record.cold_backup_startup_delay_ms:.3f} ms"
        )

        print(f"  请求状态：{state_text}")

        if record.end_to_end_delay_ms is not None:
            print(
                f"  端到端时延："
                f"{record.end_to_end_delay_ms:.3f} ms"
            )

    if event_count == 0:
        print("\n没有检测到关键故障事件。")


def save_summary_csv(
    results: list[
        tuple[str, AdaptiveReliabilityResult]
    ],
    output_path: Path,
) -> None:
    """
    保存四种方案的汇总结果。
    """

    fieldnames = [
        "scenario",
        "total_requests",
        "successful_requests",
        "failed_requests",
        "request_success_rate",
        "failover_batches",
        "hot_failover_function_stages",
        "cold_failover_function_stages",
        "total_failover_delay_ms",
        "total_cold_backup_startup_delay_ms",
        "deadline_violations",
        "sla_violations",
        "sla_violation_rate",
        "average_successful_batch_delay_ms",
        "p95_successful_batch_delay_ms",
        "average_active_memory_mb",
        "peak_active_memory_mb",
        "total_active_memory_mb_seconds",
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

        for scenario_name, result in results:
            summary = result.summary

            writer.writerow(
                {
                    "scenario": scenario_name,
                    "total_requests": summary.total_requests,
                    "successful_requests": (
                        summary.successful_requests
                    ),
                    "failed_requests": (
                        summary.failed_requests
                    ),
                    "request_success_rate": (
                        summary.request_success_rate
                    ),
                    "failover_batches": (
                        summary.failover_batches
                    ),
                    "hot_failover_function_stages": (
                        summary
                        .hot_failover_function_stages
                    ),
                    "cold_failover_function_stages": (
                        summary
                        .cold_failover_function_stages
                    ),
                    "total_failover_delay_ms": (
                        summary.total_failover_delay_ms
                    ),
                    "total_cold_backup_startup_delay_ms": (
                        summary
                        .total_cold_backup_startup_delay_ms
                    ),
                    "deadline_violations": (
                        summary.deadline_violations
                    ),
                    "sla_violations": (
                        summary.sla_violations
                    ),
                    "sla_violation_rate": (
                        summary.sla_violation_rate
                    ),
                    "average_successful_batch_delay_ms": (
                        summary
                        .average_successful_batch_delay_ms
                    ),
                    "p95_successful_batch_delay_ms": (
                        summary
                        .p95_successful_batch_delay_ms
                    ),
                    "average_active_memory_mb": (
                        summary.average_active_memory_mb
                    ),
                    "peak_active_memory_mb": (
                        summary.peak_active_memory_mb
                    ),
                    "total_active_memory_mb_seconds": (
                        summary
                        .total_active_memory_mb_seconds
                    ),
                }
            )


def plot_metric(
    results: list[
        tuple[str, AdaptiveReliabilityResult]
    ],
    metric_name: str,
    values: list[float],
    y_label: str,
    output_path: Path,
) -> None:
    """
    绘制一个独立的方案对比柱状图。
    """

    scenario_names = [
        scenario_name
        for scenario_name, _ in results
    ]

    figure, axis = plt.subplots(figsize=(10, 5))

    axis.bar(
        scenario_names,
        values,
    )

    axis.set_title(metric_name)
    axis.set_xlabel("Standby Mode")
    axis.set_ylabel(y_label)
    axis.tick_params(axis="x", rotation=15)
    axis.grid(axis="y", alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def main() -> None:
    """
    主程序入口。
    """

    project_root = Path(__file__).resolve().parent

    config = load_config(
        project_root / "configs" / "debug.yaml"
    )

    redundant_planner = (
        build_reliability_aware_replica_planner(
            config
        )
    )

    scenarios = [
        (
            "single_replica",
            SingleReplicaPlanner(),
            PrimaryOnlyStandbyPolicy(),
        ),
        (
            "cross_domain_cold_standby",
            redundant_planner,
            PrimaryOnlyStandbyPolicy(),
        ),
        (
            "cross_domain_hot_standby",
            redundant_planner,
            AllHotStandbyPolicy(),
        ),
        (
            "trajectory_aware_dynamic",
            redundant_planner,
            TrajectoryAwareStandbyPolicy(
                lead_time_s=(
                    config["standby_mode_demo"][
                        "trajectory_lead_time_s"
                    ]
                )
            ),
        ),
    ]

    results: list[
        tuple[str, AdaptiveReliabilityResult]
    ] = []

    for (
        scenario_name,
        planner,
        standby_policy,
    ) in scenarios:
        simulator = build_simulator(
            config=config,
            replica_planner=planner,
            standby_policy=standby_policy,
        )

        result = simulator.run()

        results.append(
            (scenario_name, result)
        )

        print_result(
            scenario_name=scenario_name,
            result=result,
        )

    # 重点打印动态主备的故障事件。
    print_failure_events(
        scenario_name=results[3][0],
        result=results[3][1],
    )

    tables_dir = project_root / "results" / "tables"
    figures_dir = project_root / "results" / "figures"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_path = (
        tables_dir
        / "standby_mode_comparison.csv"
    )

    success_path = (
        figures_dir
        / "standby_request_success_rate.png"
    )

    delay_path = (
        figures_dir
        / "standby_average_delay.png"
    )

    memory_path = (
        figures_dir
        / "standby_average_memory.png"
    )

    save_summary_csv(
        results=results,
        output_path=summary_path,
    )

    plot_metric(
        results=results,
        metric_name=(
            "Standby Mode: Request Success Rate"
        ),
        values=[
            result.summary.request_success_rate
            for _, result in results
        ],
        y_label="Request Success Rate",
        output_path=success_path,
    )

    plot_metric(
        results=results,
        metric_name=(
            "Standby Mode: Average Successful Delay"
        ),
        values=[
            result.summary
            .average_successful_batch_delay_ms
            for _, result in results
        ],
        y_label="Average Delay (ms)",
        output_path=delay_path,
    )

    plot_metric(
        results=results,
        metric_name=(
            "Standby Mode: Average Active Memory"
        ),
        values=[
            result.summary.average_active_memory_mb
            for _, result in results
        ],
        y_label="Average Active Memory (MB)",
        output_path=memory_path,
    )

    print("\n" + "=" * 72)
    print("实验结果已保存")
    print("=" * 72)
    print(f"汇总表：{summary_path}")
    print(f"请求成功率图：{success_path}")
    print(f"平均时延图：{delay_path}")
    print(f"平均内存图：{memory_path}")


if __name__ == "__main__":
    main()