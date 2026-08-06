"""
run_runtime_reliability_demo.py

在完全相同的随机故障轨迹下比较：

1. 当前接入 MEC 单副本；
2. 故障域感知热备双副本。

比较指标：

1. 请求成功率；
2. 主备切换次数；
3. 服务失败次数；
4. SLA违反率；
5. 成功请求时延；
6. 热备副本内存开销。
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
from src.failure_process import (
    build_markov_failure_process,
)
from src.mobility import TrainMobilityModel
from src.network import build_linear_mec_network
from src.redundancy_placement import (
    build_reliability_aware_replica_planner,
)
from src.runtime_reliability import (
    HotStandbyRuntimeSimulator,
    RuntimeReliabilityResult,
    SingleReplicaPlanner,
)
from src.topology import build_linear_topology
from src.workload import DeterministicWorkload


def build_demo_functions() -> list[ServerlessFunction]:
    """
    创建三函数故障诊断 SFC。
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
    创建高可靠故障诊断 SFC。
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
    replica_planner,
) -> HotStandbyRuntimeSimulator:
    """
    创建一个独立仿真器。

    两种方案使用相同随机种子，
    因而得到完全相同的随机故障轨迹。
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
        build_markov_failure_process(
            config=config,
            topology=topology,
        )
    )

    return HotStandbyRuntimeSimulator(
        topology=topology,
        network=network,
        mobility_model=mobility_model,
        workload=workload,
        functions=build_demo_functions(),
        sfc=build_demo_sfc(),
        replica_planner=replica_planner,
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
    result: RuntimeReliabilityResult,
    memory_cost_per_mb_second: float,
) -> None:
    """
    分块打印一种运行方案。
    """

    summary = result.summary

    hot_memory_cost = (
        summary.total_hot_memory_mb_seconds
        * memory_cost_per_mb_second
    )

    print("\n" + "=" * 72)
    print(f"方案：{scenario_name}")
    print("=" * 72)

    print(f"  总请求数：{summary.total_requests}")
    print(f"  成功请求数：{summary.successful_requests}")
    print(f"  失败请求数：{summary.failed_requests}")

    print(
        f"  请求成功率："
        f"{summary.request_success_rate:.6f}"
    )

    print(
        f"  成功批次数："
        f"{summary.successful_batches}"
    )

    print(
        f"  失败批次数："
        f"{summary.failed_batches}"
    )

    print(
        f"  发生主备切换的批次数："
        f"{summary.failover_batches}"
    )

    print(
        f"  函数备用接管总次数："
        f"{summary.failover_function_stages}"
    )

    print(
        f"  成功但超时批次数："
        f"{summary.deadline_violations}"
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
        f"  平均热副本内存："
        f"{summary.average_hot_memory_mb:.3f} MB"
    )

    print(
        f"  峰值热副本内存："
        f"{summary.peak_hot_memory_mb:.3f} MB"
    )

    print(
        f"  热副本内存成本："
        f"{hot_memory_cost:.3f}"
    )


def print_reliability_events(
    scenario_name: str,
    result: RuntimeReliabilityResult,
) -> None:
    """
    打印故障、切换和服务失败事件。
    """

    print("\n" + "=" * 72)
    print(f"{scenario_name}：关键运行事件")
    print("=" * 72)

    event_count = 0

    for record in result.records:
        important = (
            len(record.down_domain_ids) > 0
            or len(record.down_node_ids) > 0
            or record.batch_failover
            or record.request_success is False
        )

        if not important:
            continue

        event_count += 1

        if record.down_domain_ids:
            domain_text = "、".join(
                str(domain_id)
                for domain_id
                in record.down_domain_ids
            )
        else:
            domain_text = "无"

        if record.down_node_ids:
            node_text = "、".join(
                f"MEC-{node_id + 1}"
                for node_id
                in record.down_node_ids
            )
        else:
            node_text = "无"

        if record.request_success is None:
            request_state = "当前时隙无请求"
        elif record.request_success:
            request_state = "请求成功"
        else:
            request_state = "请求失败"

        if record.selected_execution_node_ids:
            execution_text = " → ".join(
                f"MEC-{node_id + 1}"
                for node_id
                in record.selected_execution_node_ids
            )
        else:
            execution_text = "无"

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
            f"  失效故障域：{domain_text}"
        )

        print(
            f"  局部失效节点：{node_text}"
        )

        print(
            f"  当前请求数量："
            f"{record.request_count}"
        )

        print(
            f"  实际执行路径："
            f"{execution_text}"
        )

        print(
            f"  备用接管函数数："
            f"{record.failover_function_count}"
        )

        print(
            f"  故障切换时延："
            f"{record.failover_delay_ms:.3f} ms"
        )

        print(f"  请求状态：{request_state}")

        if record.end_to_end_delay_ms is not None:
            print(
                f"  端到端时延："
                f"{record.end_to_end_delay_ms:.3f} ms"
            )

    if event_count == 0:
        print("\n当前随机种子下没有出现故障事件。")


def save_summary_csv(
    results: list[
        tuple[str, RuntimeReliabilityResult]
    ],
    memory_cost_per_mb_second: float,
    output_path: Path,
) -> None:
    """
    保存方案汇总结果。
    """

    fieldnames = [
        "scenario",
        "total_requests",
        "successful_requests",
        "failed_requests",
        "request_success_rate",
        "successful_batches",
        "failed_batches",
        "failover_batches",
        "failover_function_stages",
        "deadline_violations",
        "sla_violations",
        "sla_violation_rate",
        "average_successful_batch_delay_ms",
        "p95_successful_batch_delay_ms",
        "average_hot_memory_mb",
        "peak_hot_memory_mb",
        "total_hot_memory_mb_seconds",
        "hot_memory_cost",
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
                    "total_requests": (
                        summary.total_requests
                    ),
                    "successful_requests": (
                        summary.successful_requests
                    ),
                    "failed_requests": (
                        summary.failed_requests
                    ),
                    "request_success_rate": (
                        summary.request_success_rate
                    ),
                    "successful_batches": (
                        summary.successful_batches
                    ),
                    "failed_batches": (
                        summary.failed_batches
                    ),
                    "failover_batches": (
                        summary.failover_batches
                    ),
                    "failover_function_stages": (
                        summary
                        .failover_function_stages
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
                    "average_hot_memory_mb": (
                        summary.average_hot_memory_mb
                    ),
                    "peak_hot_memory_mb": (
                        summary.peak_hot_memory_mb
                    ),
                    "total_hot_memory_mb_seconds": (
                        summary
                        .total_hot_memory_mb_seconds
                    ),
                    "hot_memory_cost": (
                        summary
                        .total_hot_memory_mb_seconds
                        * memory_cost_per_mb_second
                    ),
                }
            )


def save_event_csv(
    results: list[
        tuple[str, RuntimeReliabilityResult]
    ],
    output_path: Path,
) -> None:
    """
    保存发生故障、切换或请求失败的事件。
    """

    fieldnames = [
        "scenario",
        "time_slot",
        "position_m",
        "serving_mec",
        "request_count",
        "down_domain_ids",
        "down_node_ids",
        "selected_execution_node_ids",
        "batch_failover",
        "failover_function_count",
        "failover_delay_ms",
        "unavailable_function_ids",
        "request_success",
        "end_to_end_delay_ms",
        "deadline_met",
        "hot_memory_mb",
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
            for record in result.records:
                important = (
                    len(record.down_domain_ids) > 0
                    or len(record.down_node_ids) > 0
                    or record.batch_failover
                    or record.request_success is False
                )

                if not important:
                    continue

                writer.writerow(
                    {
                        "scenario": scenario_name,
                        "time_slot": record.time_slot,
                        "position_m": record.position_m,
                        "serving_mec": (
                            record.serving_mec
                        ),
                        "request_count": (
                            record.request_count
                        ),
                        "down_domain_ids": list(
                            record.down_domain_ids
                        ),
                        "down_node_ids": list(
                            record.down_node_ids
                        ),
                        "selected_execution_node_ids": list(
                            record
                            .selected_execution_node_ids
                        ),
                        "batch_failover": (
                            record.batch_failover
                        ),
                        "failover_function_count": (
                            record
                            .failover_function_count
                        ),
                        "failover_delay_ms": (
                            record.failover_delay_ms
                        ),
                        "unavailable_function_ids": list(
                            record
                            .unavailable_function_ids
                        ),
                        "request_success": (
                            record.request_success
                        ),
                        "end_to_end_delay_ms": (
                            record.end_to_end_delay_ms
                        ),
                        "deadline_met": (
                            record.deadline_met
                        ),
                        "hot_memory_mb": (
                            record.hot_memory_mb
                        ),
                    }
                )


def plot_success_rate(
    results: list[
        tuple[str, RuntimeReliabilityResult]
    ],
    output_path: Path,
) -> None:
    """
    绘制请求成功率。
    """

    scenario_names = [
        scenario_name
        for scenario_name, _
        in results
    ]

    success_rates = [
        result.summary.request_success_rate
        for _, result in results
    ]

    figure, axis = plt.subplots(figsize=(8, 5))

    axis.bar(
        scenario_names,
        success_rates,
    )

    axis.set_ylim(0.0, 1.05)
    axis.set_title(
        "Runtime Reliability: Request Success Rate"
    )
    axis.set_xlabel("Replica Strategy")
    axis.set_ylabel("Request Success Rate")
    axis.grid(axis="y", alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_memory(
    results: list[
        tuple[str, RuntimeReliabilityResult]
    ],
    output_path: Path,
) -> None:
    """
    绘制平均热副本内存。
    """

    scenario_names = [
        scenario_name
        for scenario_name, _
        in results
    ]

    memories = [
        result.summary.average_hot_memory_mb
        for _, result in results
    ]

    figure, axis = plt.subplots(figsize=(8, 5))

    axis.bar(
        scenario_names,
        memories,
    )

    axis.set_title(
        "Runtime Reliability: Hot Replica Memory"
    )
    axis.set_xlabel("Replica Strategy")
    axis.set_ylabel("Average Hot Memory (MB)")
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

    scenarios = [
        (
            "single_replica",
            SingleReplicaPlanner(),
        ),
        (
            "cross_domain_hot_standby",
            build_reliability_aware_replica_planner(
                config
            ),
        ),
    ]

    results: list[
        tuple[str, RuntimeReliabilityResult]
    ] = []

    memory_cost_per_mb_second = (
        config["simulation_cost"][
            "warm_memory_cost_per_mb_second"
        ]
    )

    for scenario_name, planner in scenarios:
        simulator = build_simulator(
            config=config,
            replica_planner=planner,
        )

        result = simulator.run()

        results.append(
            (scenario_name, result)
        )

        print_result(
            scenario_name=scenario_name,
            result=result,
            memory_cost_per_mb_second=(
                memory_cost_per_mb_second
            ),
        )

    # 双副本方案通常会出现备用接管事件，
    # 因此重点打印第二种方案。
    print_reliability_events(
        scenario_name=results[1][0],
        result=results[1][1],
    )

    tables_dir = project_root / "results" / "tables"
    figures_dir = project_root / "results" / "figures"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_path = (
        tables_dir
        / "runtime_reliability_summary.csv"
    )

    event_path = (
        tables_dir
        / "runtime_reliability_events.csv"
    )

    success_figure_path = (
        figures_dir
        / "runtime_request_success_rate.png"
    )

    memory_figure_path = (
        figures_dir
        / "runtime_hot_replica_memory.png"
    )

    save_summary_csv(
        results=results,
        memory_cost_per_mb_second=(
            memory_cost_per_mb_second
        ),
        output_path=summary_path,
    )

    save_event_csv(
        results=results,
        output_path=event_path,
    )

    plot_success_rate(
        results=results,
        output_path=success_figure_path,
    )

    plot_memory(
        results=results,
        output_path=memory_figure_path,
    )

    print("\n" + "=" * 72)
    print("实验结果已保存")
    print("=" * 72)
    print(f"汇总结果：{summary_path}")
    print(f"关键事件：{event_path}")
    print(f"请求成功率图：{success_figure_path}")
    print(f"热副本内存图：{memory_figure_path}")


if __name__ == "__main__":
    main()