"""
run_retention_comparison.py

比较三种容器保留策略：

1. 按需启动；
2. 固定生存窗口；
3. 始终保温。

程序会：

1. 读取 debug.yaml；
2. 对三种策略运行同一条请求轨迹；
3. 在终端输出汇总结果；
4. 保存汇总 CSV；
5. 保存逐时隙 CSV；
6. 绘制总成本和平均时延柱状图。
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from src.config import load_config
from src.cost_model import CostWeights
from src.entities import ServerlessFunction
from src.retention_experiment import (
    PolicyRunResult,
    run_retention_policy,
)
from src.retention_policies import (
    AlwaysWarmPolicy,
    FixedWindowPolicy,
    OnDemandPolicy,
)


def build_demo_function() -> ServerlessFunction:
    """
    创建实验使用的数据清洗函数。

    当前参数与前面 run_demo.py 中的数据清洗函数一致。
    """

    return ServerlessFunction(
        function_id=0,
        name="数据清洗",
        memory_mb=256.0,
        cpu_cycles_per_request=20.0,
        image_size_mb=80.0,
        warm_exec_time_ms=15.0,
        cold_start_time_ms=300.0,
        output_ratio=0.7,
    )


def save_summary_csv(
    results: list[PolicyRunResult],
    output_path: Path,
) -> None:
    """
    保存每种策略的汇总结果。
    """

    fieldnames = [
        "policy_id",
        "display_name",
        "total_requests",
        "request_batches",
        "user_cold_starts",
        "prewarm_starts",
        "warm_hit_batches",
        "expiration_count",
        "retained_idle_slots",
        "total_request_delay_ms",
        "average_delay_per_request_ms",
        "total_delay_cost",
        "total_retention_cost",
        "total_prewarm_cost",
        "total_cost",
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

        for result in results:
            summary = result.summary

            writer.writerow(
                {
                    field_name: getattr(summary, field_name)
                    for field_name in fieldnames
                }
            )


def save_timeline_csv(
    results: list[PolicyRunResult],
    output_path: Path,
) -> None:
    """
    保存每种策略的逐时隙详细记录。
    """

    fieldnames = [
        "policy_id",
        "time_slot",
        "request_count",
        "before_status",
        "after_status",
        "cold_start_occurred",
        "prewarm_occurred",
        "warm_hit",
        "expired",
        "idle_slots_after",
        "request_delay_ms",
        "delay_cost",
        "retention_cost",
        "prewarm_cost",
        "total_cost",
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

        for result in results:
            for record in result.records:
                writer.writerow(
                    {
                        "policy_id": result.summary.policy_id,
                        "time_slot": record.time_slot,
                        "request_count": record.request_count,
                        "before_status": (
                            record.before_status.value
                        ),
                        "after_status": (
                            record.after_status.value
                        ),
                        "cold_start_occurred": (
                            record.cold_start_occurred
                        ),
                        "prewarm_occurred": (
                            record.prewarm_occurred
                        ),
                        "warm_hit": record.warm_hit,
                        "expired": record.expired,
                        "idle_slots_after": (
                            record.idle_slots_after
                        ),
                        "request_delay_ms": (
                            record.request_delay_ms
                        ),
                        "delay_cost": record.delay_cost,
                        "retention_cost": (
                            record.retention_cost
                        ),
                        "prewarm_cost": record.prewarm_cost,
                        "total_cost": record.total_cost,
                    }
                )


def plot_total_cost(
    results: list[PolicyRunResult],
    output_path: Path,
) -> None:
    """
    绘制三种策略的总成本柱状图。
    """

    policy_names = [
        result.summary.policy_id
        for result in results
    ]

    total_costs = [
        result.summary.total_cost
        for result in results
    ]

    figure, axis = plt.subplots(figsize=(8, 5))

    axis.bar(
        policy_names,
        total_costs,
    )

    axis.set_title("Retention Policy Total Cost")
    axis.set_xlabel("Policy")
    axis.set_ylabel("Normalized Total Cost")
    axis.grid(axis="y", alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_average_delay(
    results: list[PolicyRunResult],
    output_path: Path,
) -> None:
    """
    绘制三种策略的平均请求时延柱状图。
    """

    policy_names = [
        result.summary.policy_id
        for result in results
    ]

    average_delays = [
        result.summary.average_delay_per_request_ms
        for result in results
    ]

    figure, axis = plt.subplots(figsize=(8, 5))

    axis.bar(
        policy_names,
        average_delays,
    )

    axis.set_title("Average Delay per Request")
    axis.set_xlabel("Policy")
    axis.set_ylabel("Average Delay (ms)")
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

    # 创建输出目录。
    tables_dir = project_root / "results" / "tables"
    figures_dir = project_root / "results" / "figures"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    function = build_demo_function()

    request_trace = config["retention_demo"]["request_trace"]

    slot_seconds = config["simulation"]["fast_slot_seconds"]

    cost_weights = CostWeights(
        delay_cost_per_ms=(
            config["cost"]["delay_cost_per_ms"]
        ),
        warm_memory_cost_per_mb_second=(
            config["cost"][
                "warm_memory_cost_per_mb_second"
            ]
        ),
        prewarm_start_cost_per_ms=(
            config["cost"]["prewarm_start_cost_per_ms"]
        ),
    )

    fixed_window_slots = (
        config["retention_demo"]["fixed_window_slots"]
    )

    policies = [
        OnDemandPolicy(),
        FixedWindowPolicy(
            window_slots=fixed_window_slots
        ),
        AlwaysWarmPolicy(),
    ]

    results: list[PolicyRunResult] = []

    for policy in policies:
        result = run_retention_policy(
            function=function,
            request_trace=request_trace,
            policy=policy,
            slot_seconds=slot_seconds,
            cost_weights=cost_weights,
        )

        results.append(result)

    print("=" * 126)
    print("Serverless 容器保留策略对比实验")
    print("=" * 126)

    print(
        f"{'策略':<18}"
        f"{'请求数':>8}"
        f"{'用户冷启':>10}"
        f"{'主动预热':>10}"
        f"{'温命中':>10}"
        f"{'销毁次数':>10}"
        f"{'保留时隙':>10}"
        f"{'平均时延(ms)':>16}"
        f"{'保留成本':>14}"
        f"{'预热成本':>14}"
        f"{'总成本':>14}"
    )

    print("-" * 126)

    for result in results:
        summary = result.summary

        print(
            f"{summary.display_name:<18}"
            f"{summary.total_requests:>8d}"
            f"{summary.user_cold_starts:>10d}"
            f"{summary.prewarm_starts:>10d}"
            f"{summary.warm_hit_batches:>10d}"
            f"{summary.expiration_count:>10d}"
            f"{summary.retained_idle_slots:>10d}"
            f"{summary.average_delay_per_request_ms:>16.2f}"
            f"{summary.total_retention_cost:>14.2f}"
            f"{summary.total_prewarm_cost:>14.2f}"
            f"{summary.total_cost:>14.2f}"
        )

    print("-" * 126)

    # 保存汇总结果。
    summary_path = (
        tables_dir / "retention_policy_summary.csv"
    )

    save_summary_csv(
        results=results,
        output_path=summary_path,
    )

    # 保存逐时隙结果。
    timeline_path = (
        tables_dir / "retention_policy_timeline.csv"
    )

    save_timeline_csv(
        results=results,
        output_path=timeline_path,
    )

    # 绘制总成本图。
    total_cost_figure_path = (
        figures_dir / "retention_total_cost.png"
    )

    plot_total_cost(
        results=results,
        output_path=total_cost_figure_path,
    )

    # 绘制平均时延图。
    average_delay_figure_path = (
        figures_dir / "retention_average_delay.png"
    )

    plot_average_delay(
        results=results,
        output_path=average_delay_figure_path,
    )

    print("\n实验结果已保存：")
    print(f"  汇总表：{summary_path}")
    print(f"  逐时隙表：{timeline_path}")
    print(f"  总成本图：{total_cost_figure_path}")
    print(f"  平均时延图：{average_delay_figure_path}")

    print("=" * 126)


if __name__ == "__main__":
    main()