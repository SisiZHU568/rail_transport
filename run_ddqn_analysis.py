"""
run_ddqn_statistical_analysis.py

读取三动作Double DQN独立评估结果，
执行配对置信区间和显著性检验。

输入：

results/tables/
ddqn_reliable_evaluation_episodes.csv

输出：

1. 配对比较CSV；
2. Reward配对改进置信区间图。
"""

import os

os.environ.setdefault(
    "MPLBACKEND",
    "Agg",
)

os.environ.setdefault(
    "OMP_NUM_THREADS",
    "1",
)

os.environ.setdefault(
    "MKL_NUM_THREADS",
    "1",
)

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use(
    "Agg",
    force=True,
)

import matplotlib.pyplot as plt
import numpy as np

from src.config import load_config
from src.paired_statistics import (
    build_paired_comparison_rows,
)


plt.ioff()


METRIC_LABELS = {
    "episode_reward": "Episode Reward",
    "request_success_rate": (
        "Request Success Rate"
    ),
    "sla_violation_rate": (
        "SLA Violation Rate"
    ),
    "average_delay_ms": (
        "Average Delay"
    ),
    "average_memory_mb": (
        "Average Memory"
    ),
    "total_cold_start_delay_ms": (
        "Cold-Start Delay"
    ),
    "reconfiguration_count": (
        "Reconfiguration Count"
    ),
}


def load_csv(
    input_path: Path,
) -> list[dict[str, str]]:
    """
    读取CSV文件。
    """

    if not input_path.exists():
        raise FileNotFoundError(
            "没有找到独立评估原始结果："
            f"{input_path}\n"
            "请先运行："
            "python run_ddqn_evaluation.py"
        )

    with input_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        return list(
            csv.DictReader(
                csv_file
            )
        )


def save_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """
    保存统计结果。
    """

    if not rows:
        raise ValueError(
            "没有可保存的统计结果。"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def plot_reward_improvements(
    comparison_rows: list[
        dict[str, Any]
    ],
    output_path: Path,
) -> None:
    """
    绘制Double DQN相对于各基线的Reward改进值。

    横轴大于0：

        Double DQN奖励更高。

    横轴小于0：

        Double DQN奖励更低。
    """

    reward_rows = [
        row
        for row
        in comparison_rows
        if row["metric_name"]
        == "episode_reward"
    ]

    if not reward_rows:
        raise ValueError(
            "没有Reward配对比较结果。"
        )

    baseline_names = [
        str(
            row[
                "baseline_policy"
            ]
        )
        for row
        in reward_rows
    ]

    improvement_means = np.asarray(
        [
            float(
                row[
                    "improvement_mean"
                ]
            )
            for row
            in reward_rows
        ],
        dtype=np.float64,
    )

    confidence_lows = np.asarray(
        [
            float(
                row[
                    "improvement_ci_low"
                ]
            )
            for row
            in reward_rows
        ],
        dtype=np.float64,
    )

    confidence_highs = np.asarray(
        [
            float(
                row[
                    "improvement_ci_high"
                ]
            )
            for row
            in reward_rows
        ],
        dtype=np.float64,
    )

    lower_errors = (
        improvement_means
        - confidence_lows
    )

    upper_errors = (
        confidence_highs
        - improvement_means
    )

    error_values = np.vstack(
        (
            lower_errors,
            upper_errors,
        )
    )

    y_positions = np.arange(
        len(baseline_names)
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(10, 5.5)
    )

    axis.errorbar(
        improvement_means,
        y_positions,
        xerr=error_values,
        fmt="o",
        capsize=5,
    )

    axis.axvline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    axis.set_yticks(
        y_positions
    )

    axis.set_yticklabels(
        baseline_names
    )

    axis.set_xlabel(
        "Paired Reward Improvement "
        "(Double-DQN - Baseline)"
    )

    axis.set_ylabel(
        "Baseline Policy"
    )

    axis.set_title(
        "Double-DQN Paired Reward "
        "Improvement with 95% CI"
    )

    axis.grid(
        axis="x",
        alpha=0.3,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
    )

    plt.close(figure)


def format_effect_size(
    value: float,
) -> str:
    """
    格式化效应量。
    """

    if np.isposinf(value):
        return "+inf"

    if np.isneginf(value):
        return "-inf"

    return f"{value:.3f}"


def print_metric_table(
    rows: list[dict[str, Any]],
    metric_name: str,
) -> None:
    """
    打印某个指标的配对结果。
    """

    metric_rows = [
        row
        for row in rows
        if row["metric_name"]
        == metric_name
    ]

    print(
        "\n"
        + "=" * 112
    )

    print(
        f"指标："
        f"{METRIC_LABELS[metric_name]}"
    )

    print(
        "=" * 112
    )

    print(
        f"{'Baseline':<18}"
        f"{'Improvement':>14}"
        f"{'CI-Low':>14}"
        f"{'CI-High':>14}"
        f"{'Effect dz':>12}"
        f"{'p-Holm':>12}"
        f"{'W/T/L':>12}"
        f"{'Conclusion':>16}"
    )

    print(
        "-" * 112
    )

    for row in metric_rows:
        win_count = int(
            row["win_count"]
        )

        tie_count = int(
            row["tie_count"]
        )

        loss_count = int(
            row["loss_count"]
        )

        win_tie_loss = (
            f"{win_count}/"
            f"{tie_count}/"
            f"{loss_count}"
        )

        effect_size = format_effect_size(
            float(
                row[
                    "effect_size_dz"
                ]
            )
        )

        print(
            f"{str(row['baseline_policy']):<18}"
            f"{float(row['improvement_mean']):>14.6f}"
            f"{float(row['improvement_ci_low']):>14.6f}"
            f"{float(row['improvement_ci_high']):>14.6f}"
            f"{effect_size:>12}"
            f"{float(row['holm_adjusted_p_value']):>12.6f}"
            f"{win_tie_loss:>12}"
            f"{str(row['conclusion']):>16}"
        )


def main() -> None:
    """
    执行统计分析。
    """

    project_root = (
        Path(__file__).resolve().parent
    )

    config = load_config(
        project_root
        / "configs"
        / "debug.yaml"
    )

    statistics_config = config.get(
        "ddqn_statistics",
        {},
    )

    reference_policy = str(
        statistics_config.get(
            "reference_policy",
            "Double-DQN",
        )
    )

    confidence_level = float(
        statistics_config.get(
            "confidence_level",
            0.95,
        )
    )

    bootstrap_samples = int(
        statistics_config.get(
            "bootstrap_samples",
            20000,
        )
    )

    permutation_samples = int(
        statistics_config.get(
            "permutation_samples",
            20000,
        )
    )

    random_seed = int(
        statistics_config.get(
            "random_seed",
            20260804,
        )
    )

    input_path = (
        project_root
        / statistics_config.get(
            "input_path",
            (
                "results/tables/"
                "ddqn_reliable_"
                "evaluation_episodes.csv"
            ),
        )
    )

    output_path = (
        project_root
        / statistics_config.get(
            "output_path",
            (
                "results/tables/"
                "ddqn_reliable_"
                "paired_comparisons.csv"
            ),
        )
    )

    figure_path = (
        project_root
        / statistics_config.get(
            "reward_figure_path",
            (
                "results/figures/"
                "ddqn_reliable_"
                "paired_reward.png"
            ),
        )
    )

    records = load_csv(
        input_path
    )

    comparison_rows = (
        build_paired_comparison_rows(
            records=records,
            reference_policy=(
                reference_policy
            ),
            confidence_level=(
                confidence_level
            ),
            bootstrap_samples=(
                bootstrap_samples
            ),
            permutation_samples=(
                permutation_samples
            ),
            random_seed=random_seed,
        )
    )

    save_csv(
        rows=comparison_rows,
        output_path=output_path,
    )

    plot_reward_improvements(
        comparison_rows=(
            comparison_rows
        ),
        output_path=figure_path,
    )

    policy_names = list(
        dict.fromkeys(
            str(
                record[
                    "policy_name"
                ]
            )
            for record
            in records
        )
    )

    episode_seeds = {
        int(
            float(
                record[
                    "episode_seed"
                ]
            )
        )
        for record
        in records
    }

    print("=" * 80)
    print("Double DQN配对统计分析")
    print("=" * 80)

    print(
        f"参考策略："
        f"{reference_policy}"
    )

    print(
        "比较策略："
        + ", ".join(
            policy_name
            for policy_name
            in policy_names
            if policy_name
            != reference_policy
        )
    )

    print(
        f"配对测试种子数量："
        f"{len(episode_seeds)}"
    )

    print(
        f"置信水平："
        f"{confidence_level:.2f}"
    )

    print(
        f"Bootstrap次数："
        f"{bootstrap_samples}"
    )

    print(
        f"符号翻转次数："
        f"{permutation_samples}"
    )

    print(
        "说明：Improvement大于0表示"
        "Double-DQN更优。"
    )

    # 打印最重要的指标。
    print_metric_table(
        comparison_rows,
        "episode_reward",
    )

    print_metric_table(
        comparison_rows,
        "request_success_rate",
    )

    print_metric_table(
        comparison_rows,
        "sla_violation_rate",
    )

    print_metric_table(
        comparison_rows,
        "average_delay_ms",
    )

    print_metric_table(
        comparison_rows,
        "average_memory_mb",
    )

    print("\n" + "=" * 80)
    print("统计分析完成")
    print("=" * 80)

    print(
        f"配对统计表："
        f"{output_path}"
    )

    print(
        f"Reward置信区间图："
        f"{figure_path}"
    )


if __name__ == "__main__":
    main()