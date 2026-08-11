"""用共享 Episode 种子短校准 DPPO 的 PPO 裁剪率。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

from run_dppo_training import _checkpoint_metadata, collect_rollout
from src.config import load_config
from src.dppo import DPPOAgent, DPPORolloutBuffer
from src.dppo_checkpoint import load_dppo_checkpoint, resolve_torch_device
from src.dppo_dataset import compute_config_hash
from src.dppo_diffusion import CosineNoiseSchedule
from src.dppo_scenario import build_dppo_environment
from src.dppo_stability import (
    DPPOCalibrationCandidateResult,
    DPPOStabilityProfile,
    evaluate_calibration_candidate,
    save_stability_profile,
    select_stability_profile,
    sha256_file,
)
from src.dppo_training_config import (
    build_dppo_agent_config,
    load_dppo_stability_settings,
)


CANDIDATE_METRICS_COLUMNS = (
    "candidate",
    "mean_clip_fraction",
    "mean_approximate_kl",
    "maximum_approximate_kl",
    "optimizer_step_count",
    "finite",
    "qualified",
    "failure_reasons",
    "execution_error",
)


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用短 rollout 公平校准 DPPO 的稳定 clip_ratio。"
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "configs" / "debug.yaml"),
        help="YAML 配置文件路径。",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        required=True,
        help="所有候选共同重新加载的预训练检查点。",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="候选指标和稳定性 profile 输出目录。",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
        help="覆盖 dppo.training.device。",
    )
    return parser.parse_args(arguments)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    return sum(values) / len(values)


def _write_candidate_metrics(
    path: Path,
    rows: Sequence[tuple[DPPOCalibrationCandidateResult, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CANDIDATE_METRICS_COLUMNS)
        writer.writeheader()
        for result, execution_error in rows:
            writer.writerow(
                {
                    "candidate": result.clip_ratio,
                    "mean_clip_fraction": result.mean_clip_fraction,
                    "mean_approximate_kl": result.mean_approximate_kl,
                    "maximum_approximate_kl": result.maximum_approximate_kl,
                    "optimizer_step_count": result.optimizer_step_count,
                    "finite": json.dumps(result.finite),
                    "qualified": json.dumps(result.qualified),
                    "failure_reasons": json.dumps(
                        result.failure_reasons,
                        ensure_ascii=False,
                    ),
                    "execution_error": execution_error,
                }
            )


def calibrate_dppo_stability(
    *,
    config: dict[str, Any],
    pretrained_checkpoint: str | Path,
    output_root: str | Path,
    device: str,
) -> DPPOStabilityProfile:
    """公平运行全部候选，并无条件写出候选 CSV 和稳定性 profile。"""

    settings = load_dppo_stability_settings(config)
    resolved_device = resolve_torch_device(device)
    checkpoint_path = Path(pretrained_checkpoint)
    output_path = Path(output_root)
    episode_count = (
        settings.calibration_iterations
        * settings.calibration_episodes_per_iteration
    )
    episode_seeds = tuple(
        range(
            settings.calibration_seed_start,
            settings.calibration_seed_start + episode_count,
        )
    )

    execution_rows: list[tuple[DPPOCalibrationCandidateResult, str]] = []
    for candidate in sorted(settings.clip_ratio_candidates, reverse=True):
        # 每个候选从同一检查点重建模型、价值网络和两个优化器，不能继承前一候选状态。
        environment = build_dppo_environment(config)
        metadata = _checkpoint_metadata(config, environment)
        loaded = load_dppo_checkpoint(
            checkpoint_path,
            expected=metadata,
            device=resolved_device,
        )
        agent_config = build_dppo_agent_config(config, clip_ratio=candidate)
        agent = DPPOAgent(
            loaded.model,
            CosineNoiseSchedule(metadata.diffusion_steps),
            agent_config,
            device=resolved_device,
        )

        clip_fractions: list[float] = []
        approximate_kls: list[float] = []
        maximum_kls: list[float] = []
        optimizer_step_count = 0
        execution_error = ""
        try:
            for iteration in range(settings.calibration_iterations):
                rollout_buffer = DPPORolloutBuffer()
                first_seed_index = (
                    iteration * settings.calibration_episodes_per_iteration
                )
                for episode_index in range(
                    settings.calibration_episodes_per_iteration
                ):
                    # 候选共享同一固定 seed 序列，避免场景差异污染横向比较。
                    seed = episode_seeds[first_seed_index + episode_index]
                    collect_rollout(environment, agent, rollout_buffer, seed)
                metrics = agent.update(rollout_buffer)
                clip_fractions.append(float(metrics["clip_fraction"]))
                approximate_kls.append(float(metrics["approximate_kl"]))
                maximum_kls.append(float(metrics["maximum_approximate_kl"]))
                optimizer_step_count += int(metrics["optimizer_step_count"])
        except ArithmeticError as error:
            # 数值失败只淘汰当前候选；KeyboardInterrupt/SystemExit 不属于该层级。
            execution_error = f"{type(error).__name__}: {error}"
            mean_clip_fraction = math.nan
            mean_approximate_kl = math.nan
            maximum_approximate_kl = math.nan
        else:
            mean_clip_fraction = _mean(clip_fractions)
            mean_approximate_kl = _mean(approximate_kls)
            maximum_approximate_kl = max(maximum_kls, default=math.nan)

        result = evaluate_calibration_candidate(
            clip_ratio=candidate,
            mean_clip_fraction=mean_clip_fraction,
            mean_approximate_kl=mean_approximate_kl,
            maximum_approximate_kl=maximum_approximate_kl,
            optimizer_step_count=optimizer_step_count,
            settings=settings,
        )
        execution_rows.append((result, execution_error))

    profile = select_stability_profile(
        config_hash=compute_config_hash(config),
        pretrained_checkpoint_sha256=sha256_file(checkpoint_path),
        settings=settings,
        episode_seeds=episode_seeds,
        candidate_results=tuple(result for result, _ in execution_rows),
    )
    canonical_by_candidate = {
        result.clip_ratio: result for result in profile.candidate_results
    }
    canonical_rows = tuple(
        (canonical_by_candidate[result.clip_ratio], execution_error)
        for result, execution_error in execution_rows
    )
    _write_candidate_metrics(
        output_path / "candidate_metrics.csv",
        canonical_rows,
    )
    save_stability_profile(output_path / "stability_profile.json", profile)
    return profile


def main(arguments: Sequence[str] | None = None) -> None:
    parsed = parse_arguments(arguments)
    config = load_config(parsed.config)
    training = config["dppo"]["training"]
    device = str(training["device"]) if parsed.device is None else parsed.device
    profile = calibrate_dppo_stability(
        config=config,
        pretrained_checkpoint=parsed.pretrained_checkpoint,
        output_root=parsed.output_root,
        device=device,
    )
    output_path = Path(parsed.output_root)
    if not profile.qualified:
        raise SystemExit(
            "DPPO 稳定性校准没有合格候选；失败 profile 和候选指标已写出。"
        )
    print(f"selected_clip_ratio={profile.selected_clip_ratio}")
    print(f"candidate_metrics={output_path / 'candidate_metrics.csv'}")
    print(f"stability_profile={output_path / 'stability_profile.json'}")


if __name__ == "__main__":
    main()
