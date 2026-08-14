"""使用已通过安全解码复验的阶段 E 教师数据进行 GPU 扩散预训练。"""

import argparse
from pathlib import Path

import torch

from src.dppo_checkpoint import pretrain_diffusion_step
from src.dppo_diffusion import ConditionalDiffusionMLP, CosineNoiseSchedule
from src.phase_e_training_entry import load_teacher_dataset


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="阶段 E 正式教师预训练")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--observation-spec-hash", required=True)
    parser.add_argument("--action-spec-hash", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> None:
    args = parse_arguments(arguments)
    # 先读取维度，再用同一入口执行严格的规格与 v/u 一致性校验。
    with __import__("numpy").load(args.dataset, allow_pickle=False) as raw:
        state_dim = int(raw["observations"].shape[1])
        action_dim = int(raw["unbounded_actions"].shape[1])
    dataset = load_teacher_dataset(
        args.dataset,
        expected_observation_hash=args.observation_spec_hash,
        expected_action_hash=args.action_spec_hash,
        state_dim=state_dim,
        action_dim=action_dim,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用。")
    states = torch.from_numpy(dataset.observations.copy())
    actions = torch.from_numpy(dataset.unbounded_actions.copy())
    model = ConditionalDiffusionMLP(state_dim, action_dim, (256, 256)).to(device)
    schedule = CosineNoiseSchedule(20)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    for step in range(args.steps):
        loss = pretrain_diffusion_step(
            model, schedule, optimizer, states, actions,
            optimizer_step=step, batch_size=args.batch_size, seed=12000,
            device=device, gradient_clip_norm=5.0,
        )
        print(f"step={step + 1}/{args.steps} diffusion_loss={loss:.6f} device={device}", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": "phase-e-pretrained-v1",
        "observation_spec_hash": args.observation_spec_hash,
        "action_spec_hash": args.action_spec_hash,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "model_hidden_dims": (256, 256),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_steps": args.steps,
    }, output)
    print(f"checkpoint={output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
