"""
run_sfc_execution_demo.py

比较三种 SFC 执行场景：

1. 集中部署，所有函数为温实例；
2. 分布式部署，所有函数为温实例；
3. 分布式部署，所有函数都发生冷启动。

程序会输出：

1. 每个函数的输入和输出数据量；
2. 每个函数所在 MEC；
3. 函数间传输时延；
4. 冷启动时延；
5. 执行时延；
6. 端到端总时延；
7. 是否满足 SFC 时延约束。

同时会生成 CSV 表格和柱状图。
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
from src.network import build_linear_mec_network
from src.sfc_execution import (
    SFCExecutionResult,
    execute_sfc_request,
)
from src.topology import build_linear_topology


def build_demo_functions() -> list[ServerlessFunction]:
    """
    创建列车故障诊断 SFC 中的三个函数。
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
    创建列车设备故障诊断 SFC。
    """

    return SFCType(
        sfc_id=0,
        name="列车设备故障诊断",
        function_ids=[0, 1, 2],
        deadline_ms=1500.0,
        reliability_target=0.99,
        priority=ServicePriority.CRITICAL,
    )


def save_summary_csv(
    scenario_results: list[
        tuple[str, SFCExecutionResult]
    ],
    output_path: Path,
) -> None:
    """
    保存三个场景的汇总结果。
    """

    fieldnames = [
        "scenario",
        "placement_node_ids",
        "transmission_delay_ms",
        "cold_start_delay_ms",
        "execution_delay_ms",
        "end_to_end_delay_ms",
        "deadline_ms",
        "deadline_met",
        "final_output_size_mb",
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

        for scenario_name, result in scenario_results:
            writer.writerow(
                {
                    "scenario": scenario_name,
                    "placement_node_ids": (
                        list(result.placement_node_ids)
                    ),
                    "transmission_delay_ms": (
                        result.total_transmission_delay_ms
                    ),
                    "cold_start_delay_ms": (
                        result.total_cold_start_delay_ms
                    ),
                    "execution_delay_ms": (
                        result.total_execution_delay_ms
                    ),
                    "end_to_end_delay_ms": (
                        result.total_end_to_end_delay_ms
                    ),
                    "deadline_ms": result.deadline_ms,
                    "deadline_met": result.deadline_met,
                    "final_output_size_mb": (
                        result.final_output_size_mb
                    ),
                }
            )


def save_function_timeline_csv(
    scenario_results: list[
        tuple[str, SFCExecutionResult]
    ],
    output_path: Path,
) -> None:
    """
    保存每个函数阶段的详细执行记录。
    """

    fieldnames = [
        "scenario",
        "order_index",
        "function_id",
        "function_name",
        "previous_node_id",
        "node_id",
        "input_size_mb",
        "transmission_delay_ms",
        "cold_start_delay_ms",
        "execution_delay_ms",
        "output_size_mb",
        "stage_total_delay_ms",
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

        for scenario_name, result in scenario_results:
            for record in result.function_records:
                writer.writerow(
                    {
                        "scenario": scenario_name,
                        "order_index": record.order_index,
                        "function_id": record.function_id,
                        "function_name": record.function_name,
                        "previous_node_id": (
                            record.previous_node_id
                        ),
                        "node_id": record.node_id,
                        "input_size_mb": record.input_size_mb,
                        "transmission_delay_ms": (
                            record.transmission_delay_ms
                        ),
                        "cold_start_delay_ms": (
                            record.cold_start_delay_ms
                        ),
                        "execution_delay_ms": (
                            record.execution_delay_ms
                        ),
                        "output_size_mb": (
                            record.output_size_mb
                        ),
                        "stage_total_delay_ms": (
                            record.total_function_stage_delay_ms
                        ),
                    }
                )


def plot_scenario_delay(
    scenario_results: list[
        tuple[str, SFCExecutionResult]
    ],
    output_path: Path,
) -> None:
    """
    绘制三个场景的端到端时延柱状图。
    """

    scenario_names = [
        scenario_name
        for scenario_name, _ in scenario_results
    ]

    end_to_end_delays = [
        result.total_end_to_end_delay_ms
        for _, result in scenario_results
    ]

    figure, axis = plt.subplots(figsize=(9, 5))

    axis.bar(
        scenario_names,
        end_to_end_delays,
    )

    # 绘制 SFC 时延约束线。
    deadline_ms = scenario_results[0][1].deadline_ms

    axis.axhline(
        y=deadline_ms,
        linestyle="--",
        label=f"Deadline = {deadline_ms:.0f} ms",
    )

    axis.set_title("SFC End-to-End Delay")
    axis.set_xlabel("Scenario")
    axis.set_ylabel("End-to-End Delay (ms)")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def print_scenario_details(
    scenario_name: str,
    result: SFCExecutionResult,
) -> None:
    """
    在终端打印一个场景的详细执行过程。
    """

    print("\n" + "=" * 110)
    print(f"场景：{scenario_name}")
    print("=" * 110)

    print(
        f"{'顺序':>4}"
        f"{'函数':<12}"
        f"{'来源MEC':>10}"
        f"{'执行MEC':>10}"
        f"{'输入MB':>10}"
        f"{'传输ms':>12}"
        f"{'冷启动ms':>12}"
        f"{'执行ms':>10}"
        f"{'输出MB':>12}"
        f"{'阶段总ms':>12}"
    )

    print("-" * 110)

    for record in result.function_records:
        print(
            f"{record.order_index:>4d}"
            f"{record.function_name:<12}"
            f"{'MEC-' + str(record.previous_node_id + 1):>10}"
            f"{'MEC-' + str(record.node_id + 1):>10}"
            f"{record.input_size_mb:>10.3f}"
            f"{record.transmission_delay_ms:>12.3f}"
            f"{record.cold_start_delay_ms:>12.3f}"
            f"{record.execution_delay_ms:>10.3f}"
            f"{record.output_size_mb:>12.3f}"
            f"{record.total_function_stage_delay_ms:>12.3f}"
        )

    print("-" * 110)
    print(
        f"最终结果返回源 MEC 时延："
        f"{result.return_transmission_delay_ms:.3f} ms"
    )
    print(
        f"总传输时延："
        f"{result.total_transmission_delay_ms:.3f} ms"
    )
    print(
        f"总冷启动时延："
        f"{result.total_cold_start_delay_ms:.3f} ms"
    )
    print(
        f"总函数执行时延："
        f"{result.total_execution_delay_ms:.3f} ms"
    )
    print(
        f"端到端总时延："
        f"{result.total_end_to_end_delay_ms:.3f} ms"
    )
    print(
        f"SFC 时延约束："
        f"{result.deadline_ms:.3f} ms"
    )
    print(
        f"是否满足时延约束："
        f"{result.deadline_met}"
    )


def main() -> None:
    """
    程序主入口。
    """

    project_root = Path(__file__).resolve().parent

    config = load_config(
        project_root / "configs" / "debug.yaml"
    )

    topology = build_linear_topology(config)

    network = build_linear_mec_network(
        config=config,
        topology=topology,
    )

    functions = build_demo_functions()
    sfc = build_demo_sfc()

    source_node_id = (
        config["sfc_demo"]["source_mec_id"]
    )

    input_size_mb = (
        config["sfc_demo"]["input_size_mb"]
    )

    local_placement = (
        config["sfc_demo"]["local_placement"]
    )

    distributed_placement = (
        config["sfc_demo"]["distributed_placement"]
    )

    return_result_to_source = (
        config["sfc_demo"]["return_result_to_source"]
    )

    # 场景1：集中部署，所有函数已经是温实例。
    local_warm_result = execute_sfc_request(
        functions=functions,
        sfc=sfc,
        placement_node_ids=local_placement,
        source_node_id=source_node_id,
        input_size_mb=input_size_mb,
        network=network,
        cold_start_function_ids=set(),
        return_result_to_source=return_result_to_source,
    )

    # 场景2：分布式部署，所有函数已经是温实例。
    distributed_warm_result = execute_sfc_request(
        functions=functions,
        sfc=sfc,
        placement_node_ids=distributed_placement,
        source_node_id=source_node_id,
        input_size_mb=input_size_mb,
        network=network,
        cold_start_function_ids=set(),
        return_result_to_source=return_result_to_source,
    )

    # 场景3：分布式部署，三个函数全部冷启动。
    distributed_cold_result = execute_sfc_request(
        functions=functions,
        sfc=sfc,
        placement_node_ids=distributed_placement,
        source_node_id=source_node_id,
        input_size_mb=input_size_mb,
        network=network,
        cold_start_function_ids={0, 1, 2},
        return_result_to_source=return_result_to_source,
    )

    scenario_results = [
        ("local_warm", local_warm_result),
        ("distributed_warm", distributed_warm_result),
        ("distributed_all_cold", distributed_cold_result),
    ]

    for scenario_name, result in scenario_results:
        print_scenario_details(
            scenario_name=scenario_name,
            result=result,
        )

    tables_dir = project_root / "results" / "tables"
    figures_dir = project_root / "results" / "figures"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_path = (
        tables_dir / "sfc_scenario_summary.csv"
    )

    timeline_path = (
        tables_dir / "sfc_function_timeline.csv"
    )

    figure_path = (
        figures_dir / "sfc_end_to_end_delay.png"
    )

    save_summary_csv(
        scenario_results=scenario_results,
        output_path=summary_path,
    )

    save_function_timeline_csv(
        scenario_results=scenario_results,
        output_path=timeline_path,
    )

    plot_scenario_delay(
        scenario_results=scenario_results,
        output_path=figure_path,
    )

    print("\n" + "=" * 110)
    print("实验结果已保存")
    print("=" * 110)
    print(f"场景汇总表：{summary_path}")
    print(f"函数时间线：{timeline_path}")
    print(f"端到端时延图：{figure_path}")
    print("=" * 110)


if __name__ == "__main__":
    main()