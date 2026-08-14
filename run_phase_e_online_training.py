"""运行可恢复、可审计的 Phase E 正式在线 PPO 训练。"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from src.config import load_config
from src.dppo import DPPOAgent, DPPOBehaviorCloningBatch
from src.dppo_diffusion import CosineNoiseSchedule
from src.dppo_training_config import build_dppo_agent_config
from src.phase_e_environment import ArrivalBatch
from src.phase_e_online_checkpoint import (
    PhaseEOnlineMetadata,
    load_phase_e_online_checkpoint,
    read_phase_e_online_metadata,
    save_phase_e_online_checkpoint,
)
from src.phase_e_online_trainer import PhaseEOnlineTrainer
from src.phase_e_runtime import build_phase_e_runtime
from src.phase_e_training_entry import (
    load_phase_e_pretrained_policy,
    load_teacher_dataset,
)


_HISTORY_FIELDS = (
    "completed_update_count",
    "frame_start",
    "frame_end",
    "mean_reward",
    "mean_violation_rate",
    "mean_deficit_rate",
    "mean_raw_cost",
    "validation_reward",
    "policy_loss",
    "value_loss",
    "maximum_approximate_kl",
    "clip_fraction",
    "gradient_norm",
    "behavior_cloning_weight",
)


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase E 正式在线 PPO 训练")
    parser.add_argument("--config", default="configs/debug.yaml")
    parser.add_argument("--pretrained-checkpoint", required=True)
    parser.add_argument("--teacher-dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--clip-ratio", type=float, default=0.01)
    parser.add_argument("--resume-checkpoint")
    return parser.parse_args(arguments)


def _config_sha256(raw: dict, *, clip_ratio: float, device: str) -> str:
    binding = {
        "config": raw,
        "online_overrides": {
            "clip_ratio": float(clip_ratio),
            "device": device,
        },
    }
    encoded = json.dumps(
        binding,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_history(path: Path, row: dict[str, float | int]) -> None:
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_HISTORY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _validate_resume_history(path: Path, completed_update_count: int) -> None:
    if not path.is_file():
        raise ValueError("恢复训练要求已有 training_history.csv。")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or int(rows[-1]["completed_update_count"]) != completed_update_count:
        raise ValueError("恢复 checkpoint 与 training_history.csv 不一致。")


def _validation_reward(
    raw: dict,
    trainer: PhaseEOnlineTrainer,
    *,
    completed_update_count: int,
) -> float:
    runtime = build_phase_e_runtime(raw, total_slow_frames=1)
    settings = raw["phase_e_runtime"]
    policy_seed = (
        int(raw["dppo"]["stability"]["calibration_seed_start"])
    )

    def policy(context):
        observation = runtime.observation_adapter.encode(context)
        decision = trainer.sample_decision(
            np.asarray(observation.values, dtype=np.float32),
            seed=policy_seed,
        )
        return decision.scores

    result = runtime.environment.run_slow_frame(
        start_slot=0,
        policy=policy,
        arrivals_by_slot={
            0: (
                ArrivalBatch(
                    f"validation-{completed_update_count}",
                    0,
                    float(settings["smoke_arrival_equivalent_bits"]),
                    float(settings["smoke_deadline_seconds"]),
                ),
            )
        },
        network_for_slot=lambda slot: runtime.network_for_slot(
            slot,
            frame_index=0,
        ),
        training_mode=False,
    )
    if (
        result.code != "OK"
        or result.reward is None
        or any(not item.fast_result.optimization.succeeded for item in result.slots)
    ):
        raise RuntimeError("Phase E 固定验证发生内部失败。")
    return result.reward.reward


def main(arguments: list[str] | None = None) -> None:
    args = parse_arguments(arguments)
    raw = load_config(args.config)
    rollout_length = int(raw["dppo"]["training"]["rollout_length_slow_frames"])
    if args.frames <= 0 or args.frames % rollout_length != 0:
        raise SystemExit(
            f"--frames 必须是事务 rollout 长度 {rollout_length} 的正整数倍。"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用。")
    config_sha256 = _config_sha256(
        raw,
        clip_ratio=args.clip_ratio,
        device=args.device,
    )
    output = Path(args.output)
    history_path = output / "training_history.csv"
    last_path = output / "dppo_phase_e_last.pt"
    best_path = output / "dppo_phase_e_best.pt"
    start_frame = 0
    resume_metadata = None
    if args.resume_checkpoint:
        resume_metadata = read_phase_e_online_metadata(args.resume_checkpoint)
        start_frame = resume_metadata.next_frame_index
        if start_frame >= args.frames:
            raise SystemExit("恢复 checkpoint 已达到或超过目标 --frames。")
        _validate_resume_history(
            history_path,
            resume_metadata.completed_update_count,
        )

    trainer: PhaseEOnlineTrainer | None = None
    observation_spec_hash: str | None = None
    best_reward = (
        None if resume_metadata is None else resume_metadata.best_validation_reward
    )
    global_frame = start_frame
    while global_frame < args.frames:
        runtime = build_phase_e_runtime(raw, total_slow_frames=rollout_length)
        settings = raw["phase_e_runtime"]
        previous_result = None
        current_decision = None
        current_state = None
        frame_results = []
        rollout_start = global_frame

        for local_frame in range(rollout_length):
            frame_index = global_frame
            start_slot = local_frame * runtime.config.slow_frame_slots

            def policy(context):
                nonlocal trainer, current_decision, current_state
                nonlocal observation_spec_hash
                observation = runtime.observation_adapter.encode(
                    context,
                    previous_frame=previous_result,
                )
                if observation_spec_hash is None:
                    observation_spec_hash = observation.spec.sha256
                elif observation_spec_hash != observation.spec.sha256:
                    raise ValueError("Phase E 观察规格在训练过程中发生变化。")
                state = np.asarray(observation.values, dtype=np.float32)
                if trainer is None:
                    loaded_policy = load_phase_e_pretrained_policy(
                        args.pretrained_checkpoint,
                        expected_observation_hash=observation.spec.sha256,
                        expected_action_hash=runtime.controller.action_spec.sha256,
                        state_dim=observation.spec.dimension,
                        action_dim=runtime.controller.action_spec.action_dim,
                        device=args.device,
                    )
                    teacher_dataset = load_teacher_dataset(
                        args.teacher_dataset,
                        expected_observation_hash=observation.spec.sha256,
                        expected_action_hash=runtime.controller.action_spec.sha256,
                        state_dim=observation.spec.dimension,
                        action_dim=runtime.controller.action_spec.action_dim,
                    )
                    agent = DPPOAgent(
                        loaded_policy.model,
                        CosineNoiseSchedule(
                            int(raw["dppo"]["diffusion"]["steps"])
                        ),
                        build_dppo_agent_config(raw, clip_ratio=args.clip_ratio),
                        device=args.device,
                    )
                    trainer = PhaseEOnlineTrainer(
                        agent,
                        rollout_length_slow_frames=rollout_length,
                        teacher_batch=DPPOBehaviorCloningBatch(
                            teacher_dataset.observations,
                            teacher_dataset.unbounded_actions,
                        ),
                        total_online_updates=int(
                            raw["dppo"]["training"]["iterations"]
                        ),
                    )
                    if args.resume_checkpoint:
                        loaded = load_phase_e_online_checkpoint(
                            args.resume_checkpoint,
                            agent,
                            expected_observation_spec_hash=observation.spec.sha256,
                            expected_action_spec_hash=(
                                runtime.controller.action_spec.sha256
                            ),
                            expected_config_sha256=config_sha256,
                        )
                        trainer.completed_update_count = (
                            loaded.metadata.completed_update_count
                        )
                current_decision = trainer.sample_decision(
                    state,
                    seed=runtime.training_seed + frame_index,
                )
                current_state = state
                return current_decision.scores

            result = runtime.environment.run_slow_frame(
                start_slot=start_slot,
                policy=policy,
                arrivals_by_slot={
                    start_slot: (
                        ArrivalBatch(
                            f"train-{frame_index}",
                            0,
                            float(settings["smoke_arrival_equivalent_bits"]),
                            start_slot * runtime.config.fast_slot_seconds
                            + float(settings["smoke_deadline_seconds"]),
                        ),
                    )
                },
                network_for_slot=lambda slot: runtime.network_for_slot(
                    slot,
                    frame_index=frame_index,
                ),
                training_mode=True,
            )
            assert trainer is not None and current_decision is not None
            discarded = trainer.record_result(
                current_decision,
                result,
                terminated=local_frame == rollout_length - 1,
            )
            if result.code != "OK":
                _write_json_atomic(
                    output / "failure_audit.json",
                    {
                        "failure_code": result.code,
                        "frame_index": frame_index,
                        "start_slot": start_slot,
                        "discarded_transitions": discarded,
                        "completed_update_count": trainer.completed_update_count,
                        "last_checkpoint": str(last_path) if last_path.exists() else None,
                    },
                )
                raise SystemExit(f"Phase E internal failure: {result.code}")
            frame_results.append(result)
            previous_result = result
            global_frame += 1

        assert trainer is not None and current_state is not None
        metrics = trainer.update_if_ready(current_state)
        if metrics is None:
            raise RuntimeError("完整事务 rollout 未触发 PPO 更新。")
        mean_reward = float(np.mean([item.reward.reward for item in frame_results]))
        validation_reward = _validation_reward(
            raw,
            trainer,
            completed_update_count=trainer.completed_update_count,
        )
        improved = best_reward is None or validation_reward > best_reward
        if improved:
            best_reward = validation_reward
        assert observation_spec_hash is not None
        metadata = PhaseEOnlineMetadata(
            "phase-e-online-v1",
            observation_spec_hash,
            runtime.controller.action_spec.sha256,
            config_sha256,
            global_frame,
            trainer.completed_update_count,
            best_reward,
        )
        row = {
            "completed_update_count": trainer.completed_update_count,
            "frame_start": rollout_start,
            "frame_end": global_frame - 1,
            "mean_reward": mean_reward,
            "mean_violation_rate": float(
                np.mean([item.reward.violation_rate for item in frame_results])
            ),
            "mean_deficit_rate": float(
                np.mean([item.reward.deficit_rate for item in frame_results])
            ),
            "mean_raw_cost": float(np.mean([item.raw_cost for item in frame_results])),
            "validation_reward": validation_reward,
            "policy_loss": metrics["policy_loss"],
            "value_loss": metrics["value_loss"],
            "maximum_approximate_kl": metrics["maximum_approximate_kl"],
            "clip_fraction": metrics["clip_fraction"],
            "gradient_norm": metrics["gradient_norm"],
            "behavior_cloning_weight": metrics["behavior_cloning_weight"],
        }
        _append_history(history_path, row)
        save_phase_e_online_checkpoint(last_path, trainer.agent, metadata)
        if improved:
            save_phase_e_online_checkpoint(best_path, trainer.agent, metadata)
        print(
            f"update={trainer.completed_update_count} frames={rollout_start}-{global_frame - 1} "
            f"reward={mean_reward:.6f} validation={validation_reward:.6f} "
            f"max_kl={metrics['maximum_approximate_kl']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
