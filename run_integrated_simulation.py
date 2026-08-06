"""
run_integrated_simulation.py

运行第一个完整的轨道边缘 Serverless SFC 离散时隙仿真。

当前仿真包含：

1. 列车从 MEC-1 移动到 MEC-5；
2. 周期性 SFC 请求到达；
3. 所有函数部署在当前接入 MEC；
4. 函数容器冷启动与温实例复用；
5. 旧 MEC 容器空闲后自动回收；
6. 统计端到端时延和时延违反情况；
7. 保存 CSV 和实验图片。
"""

import csv
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt

from src.config import load_config
from src.entities import (
    ServerlessFunction,
    ServicePriority,
    SFCType,
)
from src.mobility import TrainMobilityModel
from src.network import build_linear_mec_network
from src.placement import ServingMECPlacementPolicy
from src.simulator import (
    RailServerlessSFCSimulator,
    SimulationRunResult,
)
from src.topology import build_linear_topology
from src.workload import DeterministicWorkload


def build_demo_functions() -> list[ServerlessFunction]:
    """
    创建故障诊断 SFC 中的三个函数。
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
    result: SimulationRunResult,
    output_path: Path,
) -> None:
    """
    保存完整仿真的汇总指标。
    """

    summary_dict = asdict(result.summary)

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(summary_dict.keys()),
        )

        writer.writeheader()
        writer.writerow(summary_dict)


def save_timeline_csv(
    result: SimulationRunResult,
    output_path: Path,
) -> None:
    """
    保存逐时隙详细记录。
    """

    fieldnames = [
        "time_slot",
        "position_m",
        "serving_mec",
        "next_mec",
        "handover_occurred",
        "request_count",
        "placement_node_ids",
        "prewarm_target_node_ids",
        "prewarm_start_count",
        "prewarm_startup_overhead_ms",
        "prewarm_hit_count",
        "prewarm_waste_count",
        "cold_start_count",
        "warm_hit_count",
        "expiration_count",
        "transmission_delay_ms",
        "cold_start_delay_ms",
        "execution_delay_ms",
        "end_to_end_delay_ms",
        "deadline_met",
        "warm_instance_count",
        "warm_memory_mb",
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

        for record in result.records:
            row = asdict(record)

            row["placement_node_ids"] = list(
                record.placement_node_ids
            )
            row["prewarm_target_node_ids"] = list(
                record.prewarm_target_node_ids
            )

            writer.writerow(row)


def plot_end_to_end_delay(
    result: SimulationRunResult,
    deadline_ms: float,
    output_path: Path,
) -> None:
    """
    绘制有请求时隙的端到端时延。
    """

    active_records = [
        record
        for record in result.records
        if record.request_count > 0
    ]

    time_slots = [
        record.time_slot
        for record in active_records
    ]

    delays = [
        record.end_to_end_delay_ms
        for record in active_records
    ]

    figure, axis = plt.subplots(figsize=(11, 5))

    axis.plot(
        time_slots,
        delays,
        marker="o",
        markersize=3,
        linewidth=1,
    )

    axis.axhline(
        y=deadline_ms,
        linestyle="--",
        label=f"Deadline = {deadline_ms:.0f} ms",
    )

    axis.set_title(
        "Integrated Simulation: SFC End-to-End Delay"
    )

    axis.set_xlabel("Time Slot")
    axis.set_ylabel("End-to-End Delay (ms)")
    axis.grid(alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_serving_mec(
    result: SimulationRunResult,
    output_path: Path,
) -> None:
    """
    绘制列车当前接入 MEC 随时间的变化。
    """

    time_slots = [
        record.time_slot
        for record in result.records
    ]

    # node_id 从0开始；
    # 图中显示 MEC 编号时加1。
    serving_mecs = [
        record.serving_mec + 1
        for record in result.records
    ]

    figure, axis = plt.subplots(figsize=(11, 4))

    axis.step(
        time_slots,
        serving_mecs,
        where="post",
    )

    axis.set_title(
        "Train Serving MEC over Time"
    )

    axis.set_xlabel("Time Slot")
    axis.set_ylabel("Serving MEC")
    axis.set_yticks([1, 2, 3, 4, 5])
    axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def print_summary(
    result: SimulationRunResult,
) -> None:
    """
    在终端打印仿真汇总结果。
    """

    summary = result.summary

    print("\n" + "=" * 82)
    print("完整轨道边缘 Serverless SFC 仿真结果")
    print("=" * 82)

    print(f"总仿真时隙数：{summary.total_slots}")
    print(f"总请求数量：{summary.total_requests}")
    print(f"请求批次数量：{summary.request_batches}")
    print(f"MEC 切换次数：{summary.handover_count}")

    print(
        f"函数冷启动总次数："
        f"{summary.user_cold_starts}"
    )

    print(
        f"主动预热容器总次数："
        f"{summary.prewarm_starts}"
    )

    print(
        f"主动预热启动开销："
        f"{summary.total_prewarm_startup_overhead_ms:.3f} ms"
    )

    print(
        f"函数温实例命中次数："
        f"{summary.warm_hits}"
    )

    print(
        f"容器销毁总次数："
        f"{summary.expiration_count}"
    )

    print(
        f"时延违反批次数："
        f"{summary.deadline_violations}"
    )

    print(
        f"函数阶段冷启动率："
        f"{summary.function_stage_cold_start_rate:.4f}"
    )

    print(
        f"时延违反率："
        f"{summary.deadline_violation_rate:.4f}"
    )

    print(
        f"平均批次端到端时延："
        f"{summary.average_batch_delay_ms:.3f} ms"
    )

    print(
        f"P95批次端到端时延："
        f"{summary.p95_batch_delay_ms:.3f} ms"
    )

    print("=" * 82)


def print_important_events(
    result: SimulationRunResult,
) -> None:
    """
    打印 MEC 切换、冷启动、销毁和时延违反事件。
    """

    print("\n关键事件：")

    print(
        f"{'时隙':>6}"
        f"{'位置(m)':>12}"
        f"{'当前MEC':>10}"
        f"{'请求数':>8}"
        f"{'冷启动':>10}"
        f"{'温命中':>10}"
        f"{'销毁':>8}"
        f"{'端到端(ms)':>14}"
        f"{'事件':>18}"
    )

    print("-" * 106)

    for record in result.records:
        important = (
            record.handover_occurred
            or record.cold_start_count > 0
            or record.expiration_count > 0
            or record.deadline_met is False
        )

        if not important:
            continue

        events: list[str] = []

        if record.handover_occurred:
            events.append("MEC切换")

        if record.cold_start_count > 0:
            events.append("函数冷启动")

        if record.expiration_count > 0:
            events.append("旧容器销毁")

        if record.deadline_met is False:
            events.append("时延违反")

        event_text = "、".join(events)

        print(
            f"{record.time_slot:>6d}"
            f"{record.position_m:>12.2f}"
            f"{'MEC-' + str(record.serving_mec + 1):>10}"
            f"{record.request_count:>8d}"
            f"{record.cold_start_count:>10d}"
            f"{record.warm_hit_count:>10d}"
            f"{record.expiration_count:>8d}"
            f"{record.end_to_end_delay_ms:>14.3f}"
            f"{event_text:>18}"
        )


def main() -> None:
    """
    主程序入口。
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

    functions = build_demo_functions()
    sfc = build_demo_sfc()

    simulator = RailServerlessSFCSimulator(
        topology=topology,
        network=network,
        mobility_model=mobility_model,
        workload=workload,
        placement_policy=(
            ServingMECPlacementPolicy()
        ),
        functions=functions,
        sfc=sfc,
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
    )

    result = simulator.run()

    print_summary(result)
    print_important_events(result)

    tables_dir = project_root / "results" / "tables"
    figures_dir = project_root / "results" / "figures"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_path = (
        tables_dir
        / "integrated_simulation_summary.csv"
    )

    timeline_path = (
        tables_dir
        / "integrated_simulation_timeline.csv"
    )

    delay_figure_path = (
        figures_dir
        / "integrated_end_to_end_delay.png"
    )

    serving_mec_figure_path = (
        figures_dir
        / "integrated_serving_mec.png"
    )

    save_summary_csv(
        result=result,
        output_path=summary_path,
    )

    save_timeline_csv(
        result=result,
        output_path=timeline_path,
    )

    plot_end_to_end_delay(
        result=result,
        deadline_ms=sfc.deadline_ms,
        output_path=delay_figure_path,
    )

    plot_serving_mec(
        result=result,
        output_path=serving_mec_figure_path,
    )

    print("\n实验结果已保存：")
    print(f"汇总表：{summary_path}")
    print(f"逐时隙记录：{timeline_path}")
    print(f"端到端时延图：{delay_figure_path}")
    print(f"接入MEC变化图：{serving_mec_figure_path}")


if __name__ == "__main__":
    main()