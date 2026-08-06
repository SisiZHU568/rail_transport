"""
run_ddqn_evaluation.py

独立测试三动作Double DQN。

比较策略：

1. Fixed-SINGLE
2. Fixed-COLD
3. Fixed-HOT
4. Fixed-DYNAMIC
5. Rule-Based
6. Double-DQN

其中：

Fixed-SINGLE仅作为对比基线；
Double-DQN只能选择COLD、HOT、DYNAMIC。
"""

import os

# 只保存图像，不创建Qt窗口。
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

import torch
import numpy as np
import matplotlib

matplotlib.use(
    "Agg",
    force=True,
)

import matplotlib.pyplot as plt

from src.config import load_config
from src.ddqn import (
    DDQNAgent,
    build_ddqn_config,
)
from src.ddqn_evaluation import (
    EpisodeEvaluationRecord,
    build_action_distribution_rows,
    build_fixed_policy,
    build_rule_based_policy,
    build_summary_rows,
    evaluate_policy,
)
from src.rl_agent_action_space import (
    DDQN_ACTION_NAMES,
    ddqn_action_to_environment_action,
    get_ddqn_action_count,
)
from src.rl_scenario import (
    build_rl_environment,
)


plt.ioff()


def save_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """
    保存字典列表为CSV。
    """

    if not rows:
        raise ValueError(
            "没有可保存的数据。"
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


def find_summary_row(
    summary_rows: list[dict[str, Any]],
    policy_name: str,
    metric_name: str,
) -> dict[str, Any]:
    """
    查找某个策略的某项指标。
    """

    for row in summary_rows:
        same_policy = (
            row["policy_name"]
            == policy_name
        )

        same_metric = (
            row["metric_name"]
            == metric_name
        )

        if same_policy and same_metric:
            return row

    raise KeyError(
        f"没有找到策略{policy_name}的"
        f"指标{metric_name}。"
    )


def plot_metric_with_ci(
    summary_rows: list[dict[str, Any]],
    metric_name: str,
    title: str,
    y_label: str,
    output_path: Path,
    limit_zero_to_one: bool = False,
) -> None:
    """
    绘制均值和置信区间柱状图。
    """

    metric_rows = [
        row
        for row in summary_rows
        if row["metric_name"]
        == metric_name
    ]

    policy_names = [
        str(row["policy_name"])
        for row in metric_rows
    ]

    means = np.asarray(
        [
            float(row["mean"])
            for row in metric_rows
        ],
        dtype=np.float64,
    )

    lows = np.asarray(
        [
            float(
                row[
                    "confidence_interval_low"
                ]
            )
            for row in metric_rows
        ],
        dtype=np.float64,
    )

    highs = np.asarray(
        [
            float(
                row[
                    "confidence_interval_high"
                ]
            )
            for row in metric_rows
        ],
        dtype=np.float64,
    )

    lower_errors = means - lows
    upper_errors = highs - means

    error_bars = np.vstack(
        (
            lower_errors,
            upper_errors,
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(11, 5.5)
    )

    x_positions = np.arange(
        len(policy_names)
    )

    axis.bar(
        x_positions,
        means,
        yerr=error_bars,
        capsize=5,
    )

    axis.set_xticks(
        x_positions
    )

    axis.set_xticklabels(
        policy_names,
        rotation=20,
        ha="right",
    )

    axis.set_title(title)
    axis.set_ylabel(y_label)

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    if limit_zero_to_one:
        axis.set_ylim(
            0.0,
            1.0,
        )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
    )

    plt.close(figure)


def plot_action_distribution(
    action_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """
    绘制动作分布堆叠柱状图。
    """

    policy_names = [
        str(row["policy_name"])
        for row in action_rows
    ]

    action_items = (
        (
            "single_ratio",
            "SINGLE",
        ),
        (
            "cold_ratio",
            "COLD",
        ),
        (
            "hot_ratio",
            "HOT",
        ),
        (
            "dynamic_ratio",
            "DYNAMIC",
        ),
    )

    x_positions = np.arange(
        len(policy_names)
    )

    bottom = np.zeros(
        len(policy_names),
        dtype=np.float64,
    )

    figure, axis = plt.subplots(
        figsize=(11, 5.5)
    )

    for field_name, label in action_items:
        values = np.asarray(
            [
                float(row[field_name])
                for row in action_rows
            ],
            dtype=np.float64,
        )

        axis.bar(
            x_positions,
            values,
            bottom=bottom,
            label=label,
        )

        bottom += values

    axis.set_xticks(
        x_positions
    )

    axis.set_xticklabels(
        policy_names,
        rotation=20,
        ha="right",
    )

    axis.set_ylim(
        0.0,
        1.0,
    )

    axis.set_ylabel(
        "Action Ratio"
    )

    axis.set_title(
        "Action Distribution"
    )

    axis.legend()

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=300,
    )

    plt.close(figure)


def main() -> None:
    """
    执行独立测试集评估。
    """

    project_root = (
        Path(__file__).resolve().parent
    )

    config_path = (
        project_root
        / "configs"
        / "debug.yaml"
    )

    config = load_config(
        config_path
    )

    evaluation_config = config.get(
        "ddqn_evaluation",
        {},
    )

    num_episodes = int(
        evaluation_config.get(
            "num_episodes",
            50,
        )
    )

    seed_start = int(
        evaluation_config.get(
            "seed_start",
            30000,
        )
    )

    confidence_level = float(
        evaluation_config.get(
            "confidence_level",
            0.95,
        )
    )

    if num_episodes <= 0:
        raise ValueError(
            "测试Episode数量必须大于0。"
        )

    episode_seeds = [
        seed_start + index
        for index in range(
            num_episodes
        )
    ]

    # --------------------------------------------------------
    # 创建参考环境
    # --------------------------------------------------------

    reference_environment = (
        build_rl_environment(
            config
        )
    )

    environment_action_count = int(
        reference_environment.action_count
    )

    ddqn_action_count = (
        get_ddqn_action_count()
    )

    if environment_action_count != 4:
        raise RuntimeError(
            "环境动作数量应为4，"
            f"实际为{environment_action_count}。"
        )

    if ddqn_action_count != 3:
        raise RuntimeError(
            "Double DQN动作数量应为3，"
            f"实际为{ddqn_action_count}。"
        )

    # --------------------------------------------------------
    # 创建智能体并加载三动作模型
    # --------------------------------------------------------

    agent_config = build_ddqn_config(
        config
    )

    ddqn_agent = DDQNAgent(
        state_dim=(
            reference_environment.state_dim
        ),
        action_count=(
            ddqn_action_count
        ),
        config=agent_config,
    )

    model_path = (
        project_root
        / "results"
        / "models"
        / "ddqn_reliable_best.pt"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            "没有找到三动作最优模型："
            f"{model_path}\n"
            "请先运行："
            "python run_ddqn_training.py"
        )

    ddqn_agent.load(
        model_path
    )

    # --------------------------------------------------------
    # Double DQN策略
    # --------------------------------------------------------

    def ddqn_policy(
        state: np.ndarray,
        observation_info: dict[str, Any],
    ) -> int:
        """
        网络输出0～2，转换为环境动作1～3。
        """

        del observation_info

        agent_action = (
            ddqn_agent.select_action(
                state=state,
                epsilon=0.0,
            )
        )

        environment_action = (
            ddqn_action_to_environment_action(
                agent_action
            )
        )

        return environment_action

    high_risk_threshold = float(
        config["two_timescale"][
            "high_failure_risk_threshold"
        ]
    )

    policy_definitions = [
        (
            "Fixed-SINGLE",
            build_fixed_policy(0),
        ),
        (
            "Fixed-COLD",
            build_fixed_policy(1),
        ),
        (
            "Fixed-HOT",
            build_fixed_policy(2),
        ),
        (
            "Fixed-DYNAMIC",
            build_fixed_policy(3),
        ),
        (
            "Rule-Based",
            build_rule_based_policy(
                high_risk_threshold
            ),
        ),
        (
            "Double-DQN",
            ddqn_policy,
        ),
    ]

    # --------------------------------------------------------
    # 执行评估
    # --------------------------------------------------------

    all_records: list[
        EpisodeEvaluationRecord
    ] = []

    print("=" * 80)
    print("三动作Double DQN独立测试集评估")
    print("=" * 80)

    print(
        f"设备：{ddqn_agent.device}"
    )

    print(
        f"状态维度："
        f"{reference_environment.state_dim}"
    )

    print(
        f"环境动作数量："
        f"{environment_action_count}"
    )

    print(
        f"Double DQN动作数量："
        f"{ddqn_action_count}"
    )

    print(
        "Double DQN可选动作："
        + ", ".join(
            DDQN_ACTION_NAMES
        )
    )

    print(
        f"模型：{model_path}"
    )

    print(
        f"测试Episode数量："
        f"{num_episodes}"
    )

    print(
        f"测试种子："
        f"{episode_seeds[0]}"
        f"～{episode_seeds[-1]}"
    )

    print(
        f"置信水平："
        f"{confidence_level:.2f}"
    )

    for policy_name, policy in (
        policy_definitions
    ):
        print(
            "\n正在评估："
            f"{policy_name}"
        )

        # 每种策略独立创建环境。
        environment = build_rl_environment(
            config
        )

        policy_records = evaluate_policy(
            environment=environment,
            policy_name=policy_name,
            policy=policy,
            episode_seeds=episode_seeds,
        )

        all_records.extend(
            policy_records
        )

        mean_reward = float(
            np.mean(
                [
                    record.episode_reward
                    for record
                    in policy_records
                ]
            )
        )

        print(
            f"完成，平均奖励："
            f"{mean_reward:.6f}"
        )

    # --------------------------------------------------------
    # 汇总结果
    # --------------------------------------------------------

    summary_rows = build_summary_rows(
        records=all_records,
        confidence_level=(
            confidence_level
        ),
    )

    action_rows = (
        build_action_distribution_rows(
            all_records
        )
    )

    episode_rows = [
        record.to_dict()
        for record in all_records
    ]

    # --------------------------------------------------------
    # 输出路径
    # --------------------------------------------------------

    episode_records_path = (
        project_root
        / "results"
        / "tables"
        / "ddqn_reliable_evaluation_episodes.csv"
    )

    summary_path = (
        project_root
        / "results"
        / "tables"
        / "ddqn_reliable_evaluation_summary.csv"
    )

    action_distribution_path = (
        project_root
        / "results"
        / "tables"
        / "ddqn_reliable_action_distribution.csv"
    )

    figures_dir = (
        project_root
        / "results"
        / "figures"
        / "ddqn_reliable_evaluation"
    )

    save_csv(
        rows=episode_rows,
        output_path=episode_records_path,
    )

    save_csv(
        rows=summary_rows,
        output_path=summary_path,
    )

    save_csv(
        rows=action_rows,
        output_path=(
            action_distribution_path
        ),
    )

    # --------------------------------------------------------
    # 打印结果表
    # --------------------------------------------------------

    print("\n" + "=" * 110)
    print("策略对比结果（均值）")
    print("=" * 110)

    print(
        f"{'Policy':<18}"
        f"{'Reward':>12}"
        f"{'Success':>12}"
        f"{'SLA-V':>12}"
        f"{'Delay(ms)':>14}"
        f"{'Memory(MB)':>14}"
        f"{'Cold(ms)':>14}"
    )

    print("-" * 110)

    for policy_name, _ in (
        policy_definitions
    ):
        reward_row = find_summary_row(
            summary_rows,
            policy_name,
            "episode_reward",
        )

        success_row = find_summary_row(
            summary_rows,
            policy_name,
            "request_success_rate",
        )

        sla_row = find_summary_row(
            summary_rows,
            policy_name,
            "sla_violation_rate",
        )

        delay_row = find_summary_row(
            summary_rows,
            policy_name,
            "average_delay_ms",
        )

        memory_row = find_summary_row(
            summary_rows,
            policy_name,
            "average_memory_mb",
        )

        cold_row = find_summary_row(
            summary_rows,
            policy_name,
            "total_cold_start_delay_ms",
        )

        reward_mean = float(
            reward_row["mean"]
        )

        success_mean = float(
            success_row["mean"]
        )

        sla_mean = float(
            sla_row["mean"]
        )

        delay_mean = float(
            delay_row["mean"]
        )

        memory_mean = float(
            memory_row["mean"]
        )

        cold_mean = float(
            cold_row["mean"]
        )

        print(
            f"{policy_name:<18}"
            f"{reward_mean:>12.4f}"
            f"{success_mean:>12.4f}"
            f"{sla_mean:>12.4f}"
            f"{delay_mean:>14.3f}"
            f"{memory_mean:>14.3f}"
            f"{cold_mean:>14.3f}"
        )

    # --------------------------------------------------------
    # 检查Double DQN是否选择了SINGLE
    # --------------------------------------------------------

    ddqn_action_row = next(
        row
        for row in action_rows
        if row["policy_name"]
        == "Double-DQN"
    )

    ddqn_single_count = int(
        ddqn_action_row[
            "single_count"
        ]
    )

    if ddqn_single_count != 0:
        raise RuntimeError(
            "三动作Double DQN不应选择SINGLE，"
            f"但检测到{ddqn_single_count}次。"
        )

    print("\nDouble-DQN动作分布：")

    print(
        f"  SINGLE："
        f"{float(ddqn_action_row['single_ratio']):.4f}"
    )

    print(
        f"  COLD："
        f"{float(ddqn_action_row['cold_ratio']):.4f}"
    )

    print(
        f"  HOT："
        f"{float(ddqn_action_row['hot_ratio']):.4f}"
    )

    print(
        f"  DYNAMIC："
        f"{float(ddqn_action_row['dynamic_ratio']):.4f}"
    )

    # --------------------------------------------------------
    # 绘图
    # --------------------------------------------------------

    plot_metric_with_ci(
        summary_rows=summary_rows,
        metric_name="episode_reward",
        title=(
            "Average Episode Reward "
            "with Confidence Interval"
        ),
        y_label="Episode Reward",
        output_path=(
            figures_dir
            / "episode_reward.png"
        ),
    )

    plot_metric_with_ci(
        summary_rows=summary_rows,
        metric_name=(
            "request_success_rate"
        ),
        title=(
            "Request Success Rate "
            "with Confidence Interval"
        ),
        y_label="Request Success Rate",
        output_path=(
            figures_dir
            / "request_success_rate.png"
        ),
        limit_zero_to_one=True,
    )

    plot_metric_with_ci(
        summary_rows=summary_rows,
        metric_name=(
            "sla_violation_rate"
        ),
        title=(
            "SLA Violation Rate "
            "with Confidence Interval"
        ),
        y_label="SLA Violation Rate",
        output_path=(
            figures_dir
            / "sla_violation_rate.png"
        ),
        limit_zero_to_one=True,
    )

    plot_metric_with_ci(
        summary_rows=summary_rows,
        metric_name=(
            "average_delay_ms"
        ),
        title=(
            "Average End-to-End Delay "
            "with Confidence Interval"
        ),
        y_label="Delay (ms)",
        output_path=(
            figures_dir
            / "average_delay_ms.png"
        ),
    )

    plot_metric_with_ci(
        summary_rows=summary_rows,
        metric_name=(
            "average_memory_mb"
        ),
        title=(
            "Average Active Memory "
            "with Confidence Interval"
        ),
        y_label="Memory (MB)",
        output_path=(
            figures_dir
            / "average_memory_mb.png"
        ),
    )

    plot_action_distribution(
        action_rows=action_rows,
        output_path=(
            figures_dir
            / "action_distribution.png"
        ),
    )

    print("\n" + "=" * 80)
    print("评估完成")
    print("=" * 80)

    print(
        f"Episode原始结果："
        f"{episode_records_path}"
    )

    print(
        f"统计汇总："
        f"{summary_path}"
    )

    print(
        f"动作分布："
        f"{action_distribution_path}"
    )

    print(
        f"图像目录："
        f"{figures_dir}"
    )


if __name__ == "__main__":
    main()