"""在慢时间尺度轨道环境中收集轨迹并在线训练 DDPO 主算法。"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.config import load_config
from src.dppo import (
    DPPOAgent,
    DPPORolloutBuffer,
    DPPORolloutTransition,
)
from src.dppo_checkpoint import (
    DPPOCheckpointMetadata,
    load_dppo_checkpoint,
    load_dppo_online_checkpoint,
    resolve_torch_device,
    save_dppo_online_checkpoint,
    validate_dppo_stability_binding,
)
from src.dppo_dataset import compute_config_hash
from src.dppo_diffusion import CosineNoiseSchedule
from src.dppo_scenario import build_dppo_environment
from src.dppo_slow_timescale_env import DPPOSlowTimescaleEnvironment
from src.dppo_stability import (
    DPPOStabilityProfile,
    load_stability_profile,
    sha256_file,
    stability_profile_sha256,
    validate_stability_profile,
)
from src.dppo_training_config import (
    build_dppo_agent_config,
    load_dppo_stability_settings,
)


TRAINING_HISTORY_COLUMNS = (
    "iteration",
    "episode_count",
    "transition_count",
    "mean_reward",
    "policy_loss",
    "value_loss",
    "raw_feasibility_rate",
    "projection_change_ratio",
    "projection_rejection_rate",
    "repair_success_rate",
    "gradient_norm",
    "approximate_kl",
    "maximum_approximate_kl",
    "optimizer_step_count",
    "kl_early_stopped",
    "clip_fraction",
    "selected_clip_ratio",
    "stability_profile_sha256",
)


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """解析在线训练参数；检查点和输出目录必须由调用方明确给出。"""

    parser = argparse.ArgumentParser(
        description="在轨道慢时间尺度环境中在线训练 DDPO 双层扩散策略。"
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "configs" / "debug.yaml"),
        help="YAML 配置文件路径。",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="覆盖配置中的在线更新迭代次数。",
    )
    parser.add_argument(
        "--episodes-per-iteration",
        type=int,
        default=None,
        help="覆盖配置中每次更新前收集的完整 Episode 数量。",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        required=True,
        help="第 12 批生成的扩散预训练检查点。",
    )
    parser.add_argument(
        "--stability-profile",
        required=True,
        help="校准通过且与当前配置、预训练检查点绑定的稳定性配置。",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="在线检查点和训练历史的独立输出目录。",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
        help="覆盖配置中的训练设备。",
    )
    return parser.parse_args(arguments)


def _checkpoint_metadata(
    config: dict[str, Any],
    environment: DPPOSlowTimescaleEnvironment,
) -> DPPOCheckpointMetadata:
    """从配置和动态场景维度构造预训练、在线训练共用的兼容性元数据。"""

    dimensions = environment.dimensions
    action_config = config["dppo"]["action"]
    diffusion_config = config["dppo"]["diffusion"]
    return DPPOCheckpointMetadata(
        state_schema_version=str(config["dppo"]["training"]["state_schema_version"]),
        action_schema_version=str(action_config["schema_version"]),
        state_dim=dimensions.state_dim,
        action_dim=dimensions.action_dim,
        mec_count=dimensions.mec_count,
        compute_node_count=dimensions.compute_node_count,
        function_count=dimensions.function_count,
        diffusion_steps=int(diffusion_config["steps"]),
        fine_tuned_steps=int(diffusion_config["fine_tuned_steps"]),
        maximum_retention_seconds=float(
            action_config["maximum_retention_seconds"]
        ),
        minimum_replicas=int(action_config["minimum_replicas"]),
        maximum_replicas=int(action_config["maximum_replicas"]),
        config_hash=compute_config_hash(config),
    )


def collect_rollout(
    environment: DPPOSlowTimescaleEnvironment,
    agent: DPPOAgent,
    rollout_buffer: DPPORolloutBuffer,
    seed: int,
) -> dict[str, float]:
    """收集一个完整 Episode，并保留成功和拒绝动作的全部去噪轨迹。"""

    if not isinstance(environment, DPPOSlowTimescaleEnvironment):
        raise TypeError("environment 必须是 DPPOSlowTimescaleEnvironment。")
    if not isinstance(agent, DPPOAgent):
        raise TypeError("agent 必须是 DPPOAgent。")
    if not isinstance(rollout_buffer, DPPORolloutBuffer):
        raise TypeError("rollout_buffer 必须是 DPPORolloutBuffer。")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed 必须是非负整数。")
    if (
        agent.state_dim != environment.dimensions.state_dim
        or agent.action_dim != environment.dimensions.action_dim
    ):
        raise ValueError("智能体状态/动作维度与环境不一致。")

    state = environment.reset(seed=seed)
    rewards: list[float] = []
    raw_feasible_count = 0
    projection_change_sum = 0.0
    projection_rejection_count = 0
    repair_attempts = 0
    repair_successes = 0
    repair_failures = 0
    episode_step = 0
    while True:
        # 使用 Episode seed 和局部步号，避免 Buffer 已有数据改变当前 Episode 轨迹。
        sample = agent.sample_action(state, seed=seed + episode_step)
        raw_action = sample.actions[-1, 0].detach().cpu().numpy()
        next_state, reward, terminated, truncated, info = environment.step(raw_action)
        rollout_buffer.append(
            DPPORolloutTransition(
                state=state.copy(),
                raw_action=raw_action,
                denoising_actions=(
                    sample.actions[:, 0].detach().cpu().numpy()
                ),
                old_log_probabilities=(
                    sample.log_probabilities[:, 0].detach().cpu().numpy()
                ),
                reward=float(reward),
                value=float(agent.value(state)),
                terminated=bool(terminated),
            )
        )
        rewards.append(float(reward))
        raw_feasible_count += int(bool(info["raw_feasible"]))
        projection_change_sum += float(info["projection_change_ratio"])
        projection_rejection_count += int(not bool(info["projection_success"]))
        repair_attempts += int(info["fast_repair_attempts"])
        repair_successes += int(info["fast_repair_successes"])
        repair_failures += int(info["fast_repair_failures"])
        state = next_state
        episode_step += 1
        if terminated or truncated:
            break

    transition_count = len(rewards)
    if transition_count == 0:
        raise RuntimeError("环境没有产生任何慢时间尺度转移。")
    repair_success_rate = (
        repair_successes / repair_attempts if repair_attempts > 0 else 0.0
    )
    return {
        "mean_reward": float(np.mean(rewards)),
        "reward_sum": float(np.sum(rewards)),
        "transition_count": float(transition_count),
        "raw_feasible_count": float(raw_feasible_count),
        "raw_feasibility_rate": raw_feasible_count / transition_count,
        "projection_change_sum": projection_change_sum,
        "projection_change_ratio": projection_change_sum / transition_count,
        "projection_rejection_count": float(projection_rejection_count),
        "projection_rejection_rate": (
            projection_rejection_count / transition_count
        ),
        "repair_attempts": float(repair_attempts),
        "repair_successes": float(repair_successes),
        "repair_failures": float(repair_failures),
        "repair_success_rate": repair_success_rate,
    }


def _aggregate_rollout_metrics(
    episode_metrics: Sequence[dict[str, float]],
) -> dict[str, float]:
    """按实际转移数汇总多个 Episode，避免长短 Episode 权重失真。"""

    if not episode_metrics:
        raise ValueError("episode_metrics 不能为空。")
    transition_count = sum(item["transition_count"] for item in episode_metrics)
    if transition_count <= 0.0:
        raise ValueError("transition_count 必须大于零。")
    repair_attempts = sum(item["repair_attempts"] for item in episode_metrics)
    repair_successes = sum(item["repair_successes"] for item in episode_metrics)
    return {
        "transition_count": transition_count,
        "mean_reward": (
            sum(item["reward_sum"] for item in episode_metrics) / transition_count
        ),
        "raw_feasibility_rate": (
            sum(item["raw_feasible_count"] for item in episode_metrics)
            / transition_count
        ),
        "projection_change_ratio": (
            sum(item["projection_change_sum"] for item in episode_metrics)
            / transition_count
        ),
        "projection_rejection_rate": (
            sum(item["projection_rejection_count"] for item in episode_metrics)
            / transition_count
        ),
        "repair_success_rate": (
            repair_successes / repair_attempts if repair_attempts > 0.0 else 0.0
        ),
    }


def _write_training_history(
    history_path: Path,
    history: Sequence[dict[str, float | str]],
) -> None:
    """用 Excel 友好的 UTF-8-SIG 编码覆盖写入当前完整训练历史。"""

    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=TRAINING_HISTORY_COLUMNS)
        writer.writeheader()
        writer.writerows(history)


def train_dppo(
    environment: DPPOSlowTimescaleEnvironment,
    agent: DPPOAgent,
    metadata: DPPOCheckpointMetadata,
    *,
    iterations: int,
    episodes_per_iteration: int,
    seed: int,
    output_root: str | Path,
    stability_profile: DPPOStabilityProfile,
) -> tuple[dict[str, float | str], ...]:
    """执行配置化在线训练，并保存 last/best 检查点及逐迭代 CSV。"""

    for name, value in (
        ("iterations", iterations),
        ("episodes_per_iteration", episodes_per_iteration),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} 必须是正整数。")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed 必须是非负整数。")
    # profile 是正式训练门禁；所有绑定检查必须先于输出目录创建。
    validate_dppo_stability_binding(agent, metadata, stability_profile)
    profile_digest = stability_profile_sha256(stability_profile)
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    last_checkpoint = output_path / "dppo_online_last.pt"
    best_checkpoint = output_path / "dppo_online_best.pt"
    history_path = output_path / "training_history.csv"

    history: list[dict[str, float | str]] = []
    best_mean_reward: float | None = None
    for iteration in range(iterations):
        buffer = DPPORolloutBuffer()
        episode_metrics: list[dict[str, float]] = []
        for episode_index in range(episodes_per_iteration):
            episode_seed = (
                seed + iteration * episodes_per_iteration + episode_index
            )
            episode_metrics.append(
                collect_rollout(environment, agent, buffer, episode_seed)
            )
        rollout_metrics = _aggregate_rollout_metrics(episode_metrics)
        update_metrics = agent.update(buffer)
        row = {
            "iteration": float(iteration),
            "episode_count": float(episodes_per_iteration),
            "transition_count": rollout_metrics["transition_count"],
            "mean_reward": rollout_metrics["mean_reward"],
            "policy_loss": update_metrics["policy_loss"],
            "value_loss": update_metrics["value_loss"],
            "raw_feasibility_rate": rollout_metrics["raw_feasibility_rate"],
            "projection_change_ratio": rollout_metrics[
                "projection_change_ratio"
            ],
            "projection_rejection_rate": rollout_metrics[
                "projection_rejection_rate"
            ],
            "repair_success_rate": rollout_metrics["repair_success_rate"],
            "gradient_norm": update_metrics["gradient_norm"],
            "approximate_kl": update_metrics["approximate_kl"],
            "maximum_approximate_kl": update_metrics[
                "maximum_approximate_kl"
            ],
            "optimizer_step_count": update_metrics["optimizer_step_count"],
            "kl_early_stopped": update_metrics["kl_early_stopped"],
            "clip_fraction": update_metrics["clip_fraction"],
            "selected_clip_ratio": stability_profile.selected_clip_ratio,
            "stability_profile_sha256": profile_digest,
        }
        if not all(
            isinstance(value, str) or math.isfinite(float(value))
            for value in row.values()
        ):
            raise FloatingPointError("在线训练历史出现非有限指标。")
        history.append(row)
        current_reward = row["mean_reward"]
        if best_mean_reward is None or current_reward > best_mean_reward:
            best_mean_reward = current_reward
            save_dppo_online_checkpoint(
                best_checkpoint,
                agent,
                metadata,
                iteration=iteration,
                best_mean_reward=best_mean_reward,
                stability_profile=stability_profile,
            )
        save_dppo_online_checkpoint(
            last_checkpoint,
            agent,
            metadata,
            iteration=iteration,
            best_mean_reward=best_mean_reward,
            stability_profile=stability_profile,
        )
        _write_training_history(history_path, history)
        print(
            f"iteration={iteration + 1}/{iterations} "
            f"mean_reward={row['mean_reward']:.6f} "
            f"policy_loss={row['policy_loss']:.6f} "
            f"value_loss={row['value_loss']:.6f} "
            f"maximum_approximate_kl={row['maximum_approximate_kl']:.6f} "
            f"clip_ratio={row['selected_clip_ratio']:.6f} "
            f"kl_early_stopped={bool(row['kl_early_stopped'])}"
        )
    return tuple(history)


def main(arguments: Sequence[str] | None = None) -> None:
    """加载预训练扩散模型，执行在线训练并验证最后检查点可以恢复。"""

    parsed = parse_arguments(arguments)
    config = load_config(parsed.config)
    settings = load_dppo_stability_settings(config)
    stability_profile = load_stability_profile(parsed.stability_profile)
    config_hash = compute_config_hash(config)
    pretrained_sha256 = sha256_file(parsed.pretrained_checkpoint)
    if stability_profile.pretrained_checkpoint_sha256 != pretrained_sha256:
        raise ValueError("稳定性配置的预训练检查点 SHA256 不匹配。")
    validate_stability_profile(
        stability_profile,
        config_hash=config_hash,
        pretrained_checkpoint_path=parsed.pretrained_checkpoint,
        settings=settings,
    )
    if stability_profile.selected_clip_ratio is None:
        raise ValueError("稳定性配置未选择 clip_ratio。")
    agent_config = build_dppo_agent_config(
        config,
        clip_ratio=stability_profile.selected_clip_ratio,
    )
    training = config["dppo"]["training"]
    iterations = (
        int(training["iterations"])
        if parsed.iterations is None
        else parsed.iterations
    )
    episodes_per_iteration = (
        int(training["episodes_per_iteration"])
        if parsed.episodes_per_iteration is None
        else parsed.episodes_per_iteration
    )
    if iterations <= 0 or episodes_per_iteration <= 0:
        raise ValueError("iterations 和 episodes_per_iteration 必须是正整数。")
    device_name = str(training["device"]) if parsed.device is None else parsed.device
    device = resolve_torch_device(device_name)

    environment = build_dppo_environment(config)
    metadata = _checkpoint_metadata(config, environment)
    pretrained = load_dppo_checkpoint(
        parsed.pretrained_checkpoint,
        expected=metadata,
        device=device,
    )
    agent = DPPOAgent(
        pretrained.model,
        CosineNoiseSchedule(metadata.diffusion_steps),
        agent_config,
        device=device,
    )
    print(
        f"state_dim={environment.dimensions.state_dim} "
        f"action_dim={environment.dimensions.action_dim} "
        f"frozen_steps={agent.frozen_denoising_steps} "
        f"trainable_steps={agent.trainable_denoising_steps}"
    )
    train_dppo(
        environment,
        agent,
        metadata,
        iterations=iterations,
        episodes_per_iteration=episodes_per_iteration,
        seed=int(training["seed"]),
        output_root=parsed.output_root,
        stability_profile=stability_profile,
    )
    last_checkpoint = Path(parsed.output_root) / "dppo_online_last.pt"
    restored = load_dppo_online_checkpoint(
        last_checkpoint,
        expected=metadata,
        device=device,
    )
    if restored.stability_profile_sha256 != stability_profile_sha256(
        stability_profile
    ):
        raise ValueError("恢复检查点的稳定性配置摘要与本次正式训练不一致。")
    print(f"最后检查点：{last_checkpoint.resolve()}")
    print(f"恢复迭代：{restored.iteration}")
    print(f"最佳平均奖励：{restored.best_mean_reward:.6f}")


if __name__ == "__main__":
    main()
