"""
run_prewarming_comparison.py

比较两种 Serverless SFC 部署方式：

1. 被动响应式策略：
   新 MEC 上的第一个请求到达后才启动容器。

2. 轨迹感知提前预热策略：
   根据列车剩余驻留时间，
   提前在下一 MEC 启动整条 SFC。

程序将比较：

1. 用户冷启动次数；
2. 主动预热次数；
3. 时延违反次数；
4. 平均端到端时延；
5. P95端到端时延；
6. 后台预热启动开销。
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


def build_simulator(
    config: dict,
    prewarming_policy: PrewarmingPolicy,
) -> RailServerlessSFCSimulator:
    """
    根据指定预热策略创建一个全新的仿真器。

    两种策略必须使用独立的移动模型和容器状态，
    保证实验之间互不影响。
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
        prewarming_policy=prewarming_policy,
    )


def save_summary_csv(
    results: list[
        tuple[PrewarmingPolicy, SimulationRunResult]
    ],
    output_path: Path,
) -> None:
    """
    保存两种策略的汇总指标。
    """

    fieldnames = [
        "policy_id",
        "display_name",
        "total_requests",
        "request_batches",
        "handover_count",
        "user_cold_starts",
        "prewarm_starts",
        "prewarm_startup_overhead_ms",
        "warm_hits",
        "deadline_violations",
        "function_stage_cold_start_rate",
        "deadline_violation_rate",
        "average_batch_delay_ms",
        "p95_batch_delay_ms",
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

        for policy, result in results:
            summary = result.summary

            writer.writerow(
                {
                    "policy_id": policy.policy_id,
                    "display_name": policy.display_name,
                    "total_requests": (
                        summary.total_requests
                    ),
                    "request_batches": (
                        summary.request_batches
                    ),
                    "handover_count": (
                        summary.handover_count
                    ),
                    "user_cold_starts": (
                        summary.user_cold_starts
                    ),
                    "prewarm_starts": (
                        summary.prewarm_starts
                    ),
                    "prewarm_startup_overhead_ms": (
                        summary
                        .total_prewarm_startup_overhead_ms
                    ),
                    "warm_hits": summary.warm_hits,
                    "deadline_violations": (
                        summary.deadline_violations
                    ),
                    "function_stage_cold_start_rate": (
                        summary
                        .function_stage_cold_start_rate
                    ),
                    "deadline_violation_rate": (
                        summary.deadline_violation_rate
                    ),
                    "average_batch_delay_ms": (
                        summary.average_batch_delay_ms
                    ),
                    "p95_batch_delay_ms": (
                        summary.p95_batch_delay_ms
                    ),
                }
            )


def save_event_csv(
    results: list[
        tuple[PrewarmingPolicy, SimulationRunResult]
    ],
    output_path: Path,
) -> None:
    """
    保存预热、切换、冷启动和销毁等关键事件。
    """

    fieldnames = [
        "policy_id",
        "time_slot",
        "position_m",
        "serving_mec",
        "next_mec",
        "handover_occurred",
        "request_count",
        "prewarm_target_node_ids",
        "prewarm_start_count",
        "prewarm_startup_overhead_ms",
        "cold_start_count",
        "warm_hit_count",
        "expiration_count",
        "end_to_end_delay_ms",
        "deadline_met",
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

        for policy, result in results:
            for record in result.records:
                important = (
                    record.handover_occurred
                    or record.prewarm_start_count > 0
                    or record.cold_start_count > 0
                    or record.expiration_count > 0
                    or record.deadline_met is False
                )

                if not important:
                    continue

                writer.writerow(
                    {
                        "policy_id": policy.policy_id,
                        "time_slot": record.time_slot,
                        "position_m": record.position_m,
                        "serving_mec": record.serving_mec,
                        "next_mec": record.next_mec,
                        "handover_occurred": (
                            record.handover_occurred
                        ),
                        "request_count": (
                            record.request_count
                        ),
                        "prewarm_target_node_ids": list(
                            record.prewarm_target_node_ids
                        ),
                        "prewarm_start_count": (
                            record.prewarm_start_count
                        ),
                        "prewarm_startup_overhead_ms": (
                            record
                            .prewarm_startup_overhead_ms
                        ),
                        "cold_start_count": (
                            record.cold_start_count
                        ),
                        "warm_hit_count": (
                            record.warm_hit_count
                        ),
                        "expiration_count": (
                            record.expiration_count
                        ),
                        "end_to_end_delay_ms": (
                            record.end_to_end_delay_ms
                        ),
                        "deadline_met": (
                            record.deadline_met
                        ),
                    }
                )


def plot_average_delay(
    results: list[
        tuple[PrewarmingPolicy, SimulationRunResult]
    ],
    output_path: Path,
) -> None:
    """
    绘制平均端到端时延对比图。
    """

    policy_names = [
        policy.policy_id
        for policy, _ in results
    ]

    average_delays = [
        result.summary.average_batch_delay_ms
        for _, result in results
    ]

    figure, axis = plt.subplots(figsize=(8, 5))

    axis.bar(
        policy_names,
        average_delays,
    )

    axis.set_title(
        "Trajectory-Aware Prewarming: Average Delay"
    )
    axis.set_xlabel("Policy")
    axis.set_ylabel("Average Batch Delay (ms)")
    axis.grid(axis="y", alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_start_counts(
    results: list[
        tuple[PrewarmingPolicy, SimulationRunResult]
    ],
    output_path: Path,
) -> None:
    """
    对比用户冷启动次数和主动预热次数。
    """

    policy_names = [
        policy.policy_id
        for policy, _ in results
    ]

    user_cold_starts = [
        result.summary.user_cold_starts
        for _, result in results
    ]

    prewarm_starts = [
        result.summary.prewarm_starts
        for _, result in results
    ]

    x_positions = np.arange(
        len(policy_names)
    )

    bar_width = 0.35

    figure, axis = plt.subplots(figsize=(8, 5))

    axis.bar(
        x_positions - bar_width / 2,
        user_cold_starts,
        width=bar_width,
        label="User-triggered cold starts",
    )

    axis.bar(
        x_positions + bar_width / 2,
        prewarm_starts,
        width=bar_width,
        label="Background prewarm starts",
    )

    axis.set_title(
        "Cold Starts and Prewarm Starts"
    )
    axis.set_xlabel("Policy")
    axis.set_ylabel("Function Instance Starts")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(policy_names)
    axis.legend()
    axis.grid(axis="y", alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def print_summary(
    results: list[
        tuple[PrewarmingPolicy, SimulationRunResult]
    ],
) -> None:
    """
    以分块形式打印两种策略的汇总结果。

    不再使用超宽中文表格，
    避免 PowerShell 自动换行造成排版混乱。
    """

    print("\n" + "=" * 72)
    print("轨迹感知 Serverless SFC 提前预热策略对比")
    print("=" * 72)

    for policy, result in results:
        summary = result.summary

        print(f"\n策略：{policy.display_name}")
        print(f"策略编号：{policy.policy_id}")
        print("-" * 72)

        print(
            f"  总请求数量："
            f"{summary.total_requests}"
        )

        print(
            f"  请求批次数量："
            f"{summary.request_batches}"
        )

        print(
            f"  MEC 切换次数："
            f"{summary.handover_count}"
        )

        print(
            f"  用户触发的函数冷启动次数："
            f"{summary.user_cold_starts}"
        )

        print(
            f"  后台主动预热容器次数："
            f"{summary.prewarm_starts}"
        )

        print(
            f"  函数温实例命中次数："
            f"{summary.warm_hits}"
        )

        print(
            f"  时延违反批次数："
            f"{summary.deadline_violations}"
        )

        print(
            f"  函数阶段冷启动率："
            f"{summary.function_stage_cold_start_rate:.4f}"
        )

        print(
            f"  时延违反率："
            f"{summary.deadline_violation_rate:.4f}"
        )

        print(
            f"  平均批次端到端时延："
            f"{summary.average_batch_delay_ms:.3f} ms"
        )

        print(
            f"  P95 批次端到端时延："
            f"{summary.p95_batch_delay_ms:.3f} ms"
        )

        print(
            f"  后台预热启动开销："
            f"{summary.total_prewarm_startup_overhead_ms:.3f} ms"
        )

    print("\n" + "=" * 72)


def print_prewarming_events(
    result: SimulationRunResult,
) -> None:
    """
    分块打印轨迹感知策略中的关键事件。

    每个关键时隙单独打印一个小块，
    避免一行包含太多中文字段。
    """

    print("\n" + "=" * 72)
    print("轨迹感知策略关键事件")
    print("=" * 72)

    event_number = 0

    for record in result.records:
        # 只打印真正重要的事件。
        important = (
            record.prewarm_start_count > 0
            or record.handover_occurred
            or record.cold_start_count > 0
            or record.deadline_met is False
        )

        if not important:
            continue

        event_number += 1

        # 保存当前时隙发生的事件名称。
        events: list[str] = []

        if record.prewarm_start_count > 0:
            events.append("提前预热")

        if record.handover_occurred:
            events.append("MEC 切换")

        if record.cold_start_count > 0:
            events.append("用户请求触发冷启动")

        if record.deadline_met is False:
            events.append("违反时延约束")

        # 将预热目标节点转换成容易阅读的 MEC 名称。
        if record.prewarm_target_node_ids:
            target_text = "、".join(
                f"MEC-{node_id + 1}"
                for node_id
                in record.prewarm_target_node_ids
            )
        else:
            target_text = "无"

        # 没有请求时，deadline_met 为 None。
        if record.deadline_met is None:
            deadline_text = "当前时隙没有请求"
        elif record.deadline_met:
            deadline_text = "满足"
        else:
            deadline_text = "违反"

        print(f"\n事件 {event_number}")
        print("-" * 72)

        print(
            f"  时隙与位置："
            f"时隙 {record.time_slot}，"
            f"位置 {record.position_m:.2f} m"
        )

        print(
            f"  MEC 状态："
            f"当前 MEC-{record.serving_mec + 1}，"
            f"下一 MEC-{record.next_mec + 1}"
        )

        print(
            f"  预热目标："
            f"{target_text}"
        )

        print(
            f"  请求数量："
            f"{record.request_count}"
        )

        print(
            f"  本时隙新预热容器数："
            f"{record.prewarm_start_count}"
        )

        print(
            f"  用户触发冷启动函数数："
            f"{record.cold_start_count}"
        )

        print(
            f"  温实例命中函数数："
            f"{record.warm_hit_count}"
        )

        print(
            f"  端到端时延："
            f"{record.end_to_end_delay_ms:.3f} ms"
        )

        print(
            f"  时延约束状态："
            f"{deadline_text}"
        )

        print(
            f"  事件类型："
            f"{'、'.join(events)}"
        )

    if event_number == 0:
        print("\n没有检测到关键事件。")

    print("\n" + "=" * 72)


def main() -> None:
    """
    主程序入口。
    """

    project_root = Path(__file__).resolve().parent

    config = load_config(
        project_root / "configs" / "debug.yaml"
    )

    reactive_policy = NoPrewarmingPolicy()

    trajectory_policy = (
        TrajectoryAwarePrewarmingPolicy(
            lead_time_s=(
                config["prewarming_demo"][
                    "lead_time_s"
                ]
            )
        )
    )

    policies: list[PrewarmingPolicy] = [
        reactive_policy,
        trajectory_policy,
    ]

    results: list[
        tuple[PrewarmingPolicy, SimulationRunResult]
    ] = []

    for policy in policies:
        simulator = build_simulator(
            config=config,
            prewarming_policy=policy,
        )

        result = simulator.run()

        results.append(
            (policy, result)
        )

    print_summary(results)

    # results[1] 对应轨迹感知策略。
    print_prewarming_events(
        result=results[1][1]
    )

    tables_dir = project_root / "results" / "tables"
    figures_dir = project_root / "results" / "figures"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_path = (
        tables_dir
        / "prewarming_comparison_summary.csv"
    )

    event_path = (
        tables_dir
        / "prewarming_comparison_events.csv"
    )

    delay_figure_path = (
        figures_dir
        / "prewarming_average_delay.png"
    )

    start_figure_path = (
        figures_dir
        / "prewarming_start_counts.png"
    )

    save_summary_csv(
        results=results,
        output_path=summary_path,
    )

    save_event_csv(
        results=results,
        output_path=event_path,
    )

    plot_average_delay(
        results=results,
        output_path=delay_figure_path,
    )

    plot_start_counts(
        results=results,
        output_path=start_figure_path,
    )

    print("\n实验结果已保存：")
    print(f"策略汇总表：{summary_path}")
    print(f"关键事件表：{event_path}")
    print(f"平均时延图：{delay_figure_path}")
    print(f"启动次数图：{start_figure_path}")


if __name__ == "__main__":
    main()