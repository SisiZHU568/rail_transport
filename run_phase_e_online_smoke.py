"""在 VS Code 终端运行 GPU DPPO + CPU CLARABEL 的真实双时间尺度最小闭环。"""

import argparse

import numpy as np
import torch

from src.config import load_config
from src.dppo import DPPOAgent, DPPOBehaviorCloningBatch
from src.dppo_diffusion import ConditionalDiffusionMLP, CosineNoiseSchedule
from src.dppo_training_config import build_dppo_agent_config
from src.phase_e_environment import ArrivalBatch
from src.phase_e_online_trainer import PhaseEOnlineTrainer
from src.phase_e_runtime import build_phase_e_runtime
from src.phase_e_training_entry import (
    load_phase_e_pretrained_policy,
    load_teacher_dataset,
)


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="阶段 E 双时间尺度在线冒烟训练")
    parser.add_argument("--config", default="configs/debug.yaml")
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--clip-ratio", type=float, default=0.01)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--pretrained-checkpoint")
    parser.add_argument("--teacher-dataset")
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> None:
    args = parse_arguments(arguments)
    if args.frames <= 0:
        raise SystemExit("--frames 必须为正整数。")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用，请先激活 rail-dppo-gpu 环境。")
    if bool(args.pretrained_checkpoint) != bool(args.teacher_dataset):
        raise SystemExit("正式在线训练必须同时提供预训练检查点和教师数据集。")
    raw = load_config(args.config)
    phase_runtime = build_phase_e_runtime(raw, total_slow_frames=args.frames)
    config = phase_runtime.config
    controller = phase_runtime.controller
    environment = phase_runtime.environment
    observation_adapter = phase_runtime.observation_adapter
    runtime_settings = raw["phase_e_runtime"]
    trainer: PhaseEOnlineTrainer | None = None
    current_decision = None
    current_state = None
    previous_result = None
    update_metrics = None

    for frame_index in range(args.frames):
        start_slot = frame_index * config.slow_frame_slots

        def policy(context):
            nonlocal trainer, current_decision, current_state, update_metrics
            observation = observation_adapter.encode(
                context, previous_frame=previous_result
            )
            state = np.asarray(observation.values, dtype=np.float32)
            if trainer is None:
                dppo_config = build_dppo_agent_config(raw, clip_ratio=args.clip_ratio)
                teacher_batch = None
                if args.pretrained_checkpoint:
                    loaded_policy = load_phase_e_pretrained_policy(
                        args.pretrained_checkpoint,
                        expected_observation_hash=observation.spec.sha256,
                        expected_action_hash=controller.action_spec.sha256,
                        state_dim=observation.spec.dimension,
                        action_dim=controller.action_spec.action_dim,
                        device=args.device,
                    )
                    teacher_dataset = load_teacher_dataset(
                        args.teacher_dataset,
                        expected_observation_hash=observation.spec.sha256,
                        expected_action_hash=controller.action_spec.sha256,
                        state_dim=observation.spec.dimension,
                        action_dim=controller.action_spec.action_dim,
                    )
                    policy_model = loaded_policy.model
                    teacher_batch = DPPOBehaviorCloningBatch(
                        teacher_dataset.observations,
                        teacher_dataset.unbounded_actions,
                    )
                else:
                    # 不提供正式产物时保留随机初始化，只用于验证环境/求解器接线。
                    policy_model = ConditionalDiffusionMLP(
                        observation.spec.dimension,
                        controller.action_spec.action_dim,
                        tuple(raw["dppo"]["diffusion"]["hidden_dims"]),
                    )
                agent = DPPOAgent(
                    policy_model,
                    CosineNoiseSchedule(dppo_config.diffusion_steps),
                    dppo_config,
                    device=args.device,
                )
                trainer = PhaseEOnlineTrainer(
                    agent,
                    rollout_length_slow_frames=int(
                        raw["dppo"]["training"]["rollout_length_slow_frames"]
                    ),
                    teacher_batch=teacher_batch,
                    total_online_updates=(
                        int(raw["dppo"]["training"]["iterations"])
                        if teacher_batch is not None
                        else 0
                    ),
                )
                print(
                    "policy_initialization="
                    + ("pretrained_with_teacher_bc" if teacher_batch is not None else "random_smoke_only"),
                    flush=True,
                )
            update_metrics = trainer.update_if_ready(state)
            current_decision = trainer.sample_decision(
                state,
                seed=phase_runtime.training_seed + frame_index,
            )
            current_state = state
            return current_decision.scores

        result = environment.run_slow_frame(
            start_slot=start_slot,
            policy=policy,
            arrivals_by_slot={
                start_slot: (
                    ArrivalBatch(
                        f"smoke-{frame_index}",
                        0,
                        float(runtime_settings["smoke_arrival_equivalent_bits"]),
                        start_slot * config.fast_slot_seconds
                        + float(runtime_settings["smoke_deadline_seconds"]),
                    ),
                )
            },
            network_for_slot=lambda slot: phase_runtime.network_for_slot(
                slot,
                frame_index=frame_index,
            ),
            training_mode=True,
        )
        discarded = trainer.record_result(
            current_decision,
            result,
            terminated=frame_index == args.frames - 1,
        )
        if result.code != "OK":
            raise SystemExit(
                f"frame={frame_index} internal_failure={result.code} "
                f"discarded_transitions={discarded}"
            )
        mean_fast_ms = 1000.0 * sum(
            item.fast_result.optimization.primary_solve_seconds
            + item.fast_result.optimization.secondary_solve_seconds
            for item in result.slots
        ) / len(result.slots)
        print(
            f"frame={frame_index + 1}/{args.frames} "
            f"reward={result.reward.reward:.6f} "
            f"violation={result.reward.violation_rate:.6f} "
            f"deficit={result.reward.deficit_rate:.6f} "
            f"cost={result.raw_cost:.6f} "
            f"fast_mean_ms={mean_fast_ms:.3f} "
            f"policy_device={trainer.agent.device}",
            flush=True,
        )
        if update_metrics is not None:
            print(
                f"update={trainer.completed_update_count} "
                f"policy_loss={update_metrics['policy_loss']:.6f} "
                f"value_loss={update_metrics['value_loss']:.6f} "
                f"max_kl={update_metrics['maximum_approximate_kl']:.6f} "
                f"bc_weight={update_metrics['behavior_cloning_weight']:.6f}",
                flush=True,
            )
        previous_result = result

    if trainer.pending_count == trainer.rollout_length_slow_frames:
        final_metrics = trainer.update_if_ready(current_state)
        print(
            f"update={trainer.completed_update_count} "
            f"policy_loss={final_metrics['policy_loss']:.6f} "
            f"value_loss={final_metrics['value_loss']:.6f} "
            f"max_kl={final_metrics['maximum_approximate_kl']:.6f} "
            f"bc_weight={final_metrics['behavior_cloning_weight']:.6f}",
            flush=True,
        )
    elif trainer.pending_count:
        print(
            f"clean_tail_frames={trainer.pending_count}（不足一个事务 rollout，未更新）",
            flush=True,
        )


if __name__ == "__main__":
    main()
