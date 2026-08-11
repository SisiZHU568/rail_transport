"""加载在线 DDPO 检查点并生成论文可复用的独立评估报告。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import load_config
from src.dppo_checkpoint import (
    DPPOCheckpointMetadata,
    load_dppo_online_checkpoint,
    resolve_torch_device,
)
from src.dppo_dataset import compute_config_hash
from src.dppo_evaluation import (
    DPPOEpisodeEvaluation,
    delay_statistics,
    evaluate_dppo_episodes,
    summarize_evaluation,
)
from src.dppo_scenario import build_dppo_environment
from src.dppo_slow_timescale_env import DPPOSlowTimescaleEnvironment


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """解析评估参数；种子范围、检查点和输出目录均由调用方显式指定。"""

    parser = argparse.ArgumentParser(
        description="在固定仿真种子上独立评估 DDPO 在线检查点。"
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "configs" / "debug.yaml"),
        help="YAML 配置文件路径。",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="第 14 批生成的 DDPO 在线检查点。",
    )
    parser.add_argument(
        "--episodes",
        required=True,
        type=int,
        help="独立评估 Episode 数量。",
    )
    parser.add_argument(
        "--seed-start",
        required=True,
        type=int,
        help="第一个评估 Episode 的随机种子。",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="评估 CSV 和 PNG 的独立输出目录。",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
        help="覆盖训练配置中的 PyTorch 设备。",
    )
    return parser.parse_args(arguments)


def _checkpoint_metadata(
    config: dict[str, Any],
    environment: DPPOSlowTimescaleEnvironment,
) -> DPPOCheckpointMetadata:
    """从当前配置和动态场景维度构造在线检查点兼容性约束。"""

    dimensions = environment.dimensions
    action = config["dppo"]["action"]
    diffusion = config["dppo"]["diffusion"]
    return DPPOCheckpointMetadata(
        state_schema_version=str(config["dppo"]["training"]["state_schema_version"]),
        action_schema_version=str(action["schema_version"]),
        state_dim=dimensions.state_dim,
        action_dim=dimensions.action_dim,
        mec_count=dimensions.mec_count,
        compute_node_count=dimensions.compute_node_count,
        function_count=dimensions.function_count,
        diffusion_steps=int(diffusion["steps"]),
        fine_tuned_steps=int(diffusion["fine_tuned_steps"]),
        maximum_retention_seconds=float(action["maximum_retention_seconds"]),
        minimum_replicas=int(action["minimum_replicas"]),
        maximum_replicas=int(action["maximum_replicas"]),
        config_hash=compute_config_hash(config),
    )


def _write_csv(
    path: Path,
    rows: Sequence[dict[str, object]],
    fieldnames: Sequence[str],
) -> None:
    """以 UTF-8-SIG 写表，保证中文路径环境下 Excel 可直接打开。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=tuple(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _episode_rows(
    evaluations: Sequence[DPPOEpisodeEvaluation],
) -> tuple[list[dict[str, object]], tuple[str, ...]]:
    """把不可变 Episode 结果转换成包含动态 VNF 列的宽表。"""

    rows = [
        {"episode_seed": evaluation.episode_seed, **dict(evaluation.metrics)}
        for evaluation in evaluations
    ]
    metric_names = sorted(set.union(*(set(row.keys()) for row in rows)) - {"episode_seed"})
    return rows, ("episode_seed", *metric_names)


def _plot_overview(
    evaluations: Sequence[DPPOEpisodeEvaluation],
    output_path: Path,
) -> None:
    """绘制紧凑性能概览，详细数值仍以 CSV 为准。"""

    metrics = [evaluation.metrics for evaluation in evaluations]
    rate_names = ("success_rate", "sla_success_rate", "mean_reliability")
    rate_values = [float(np.mean([row[name] for row in metrics])) for name in rate_names]
    pooled_delays = tuple(
        value
        for evaluation in evaluations
        for value in evaluation.fast_slot_delay_samples_ms
    )
    delay_values = delay_statistics(pooled_delays)
    delay_names = ("mean_delay_ms", "p95_delay_ms", "p99_delay_ms", "peak_delay_ms")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(
        ("Success", "SLA success", "Reliability"),
        rate_values,
        color=("#4C78A8", "#59A14F", "#F28E2B"),
    )
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Rate")
    axes[0].set_title("Service and reliability")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(
        ("Mean", "P95", "P99", "Peak"),
        [delay_values[name] for name in delay_names],
        color="#4C78A8",
    )
    axes[1].set_ylabel("Delay (ms)")
    axes[1].set_title("Pooled fast-slot delay")
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle("DDPO Evaluation Overview")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def write_evaluation_reports(
    evaluations: Sequence[DPPOEpisodeEvaluation],
    output_root: str | Path,
) -> dict[str, Path]:
    """分别输出性能、原始时延、副本动作、保留时间和概览图。"""

    results = tuple(evaluations)
    if not results:
        raise ValueError("evaluations 不能为空。")
    if any(not isinstance(item, DPPOEpisodeEvaluation) for item in results):
        raise TypeError("evaluations 必须只包含 DPPOEpisodeEvaluation。")
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "episode_metrics": output_path / "episode_metrics.csv",
        "summary_metrics": output_path / "summary_metrics.csv",
        "delay_samples": output_path / "delay_samples.csv",
        "pooled_delay_metrics": output_path / "pooled_delay_metrics.csv",
        "replica_distribution": output_path / "replica_distribution.csv",
        "retention_distribution": output_path / "retention_distribution.csv",
        "overview": output_path / "evaluation_overview.png",
    }

    episode_rows, episode_fields = _episode_rows(results)
    _write_csv(paths["episode_metrics"], episode_rows, episode_fields)
    summary_rows = list(
        summarize_evaluation([evaluation.metrics for evaluation in results])
    )
    _write_csv(
        paths["summary_metrics"],
        summary_rows,
        (
            "metric",
            "mean",
            "sample_count",
            "standard_deviation",
            "confidence_interval_95",
        ),
    )

    delay_rows: list[dict[str, object]] = []
    pooled_delays: list[float] = []
    for evaluation in results:
        for sample_index, delay_ms in enumerate(
            evaluation.fast_slot_delay_samples_ms
        ):
            delay_rows.append(
                {
                    "episode_seed": evaluation.episode_seed,
                    "sample_index": sample_index,
                    "delay_ms": delay_ms,
                }
            )
            pooled_delays.append(delay_ms)
    _write_csv(
        paths["delay_samples"],
        delay_rows,
        ("episode_seed", "sample_index", "delay_ms"),
    )
    pooled_delay_row: dict[str, object] = {
        "sample_count": len(pooled_delays),
        **delay_statistics(pooled_delays),
    }
    _write_csv(
        paths["pooled_delay_metrics"],
        [pooled_delay_row],
        (
            "sample_count",
            "mean_delay_ms",
            "peak_delay_ms",
            "p95_delay_ms",
            "p99_delay_ms",
        ),
    )

    action_records = tuple(
        record for evaluation in results for record in evaluation.action_records
    )
    replica_rows = [
        {
            "episode_seed": record.episode_seed,
            "decision_index": record.decision_index,
            "function_id": record.function_id,
            "replica_count": record.replica_count,
            "cloud_selected": int(record.cloud_selected),
        }
        for record in action_records
    ]
    _write_csv(
        paths["replica_distribution"],
        replica_rows,
        (
            "episode_seed",
            "decision_index",
            "function_id",
            "replica_count",
            "cloud_selected",
        ),
    )
    retention_rows = [
        {
            "episode_seed": record.episode_seed,
            "decision_index": record.decision_index,
            "function_id": record.function_id,
            "primary_retention_seconds": record.primary_retention_seconds,
            "backup_retention_seconds": record.backup_retention_seconds,
        }
        for record in action_records
    ]
    _write_csv(
        paths["retention_distribution"],
        retention_rows,
        (
            "episode_seed",
            "decision_index",
            "function_id",
            "primary_retention_seconds",
            "backup_retention_seconds",
        ),
    )
    _plot_overview(results, paths["overview"])
    return paths


def main(arguments: Sequence[str] | None = None) -> None:
    """恢复在线双层策略，在固定新种子上评估并保存隔离报告。"""

    parsed = parse_arguments(arguments)
    if parsed.episodes <= 0:
        raise ValueError("episodes 必须是正整数。")
    if parsed.seed_start < 0:
        raise ValueError("seed_start 必须是非负整数。")
    config = load_config(parsed.config)
    device_name = (
        str(config["dppo"]["training"]["device"])
        if parsed.device is None
        else parsed.device
    )
    device = resolve_torch_device(device_name)
    environment = build_dppo_environment(config)
    metadata = _checkpoint_metadata(config, environment)
    loaded = load_dppo_online_checkpoint(
        parsed.checkpoint,
        expected=metadata,
        device=device,
    )
    evaluations = evaluate_dppo_episodes(
        environment,
        loaded.agent,
        episodes=parsed.episodes,
        seed_start=parsed.seed_start,
    )
    paths = write_evaluation_reports(evaluations, parsed.output_root)
    print(
        f"episodes={parsed.episodes} seed_start={parsed.seed_start} "
        f"state_dim={environment.dimensions.state_dim} "
        f"action_dim={environment.dimensions.action_dim}"
    )
    for name, path in paths.items():
        print(f"{name}：{path.resolve()}")


if __name__ == "__main__":
    main()
