"""在 VS Code 终端验证阶段 E 的配置尺寸、CUDA 与一次扩散优化步骤。"""

import argparse

import torch

from src.config import load_config
from src.dppo_diffusion import ConditionalDiffusionMLP, CosineNoiseSchedule, diffusion_noise_loss
from src.orchestration_config import load_phase_a_config
from src.phase_d_policy_specs import ObservationFeature, ObservationSpec
from src.safe_deployment_decoder import ActionSpec


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="阶段 E GPU 冒烟训练")
    parser.add_argument("--config", default="configs/debug.yaml")
    parser.add_argument("--steps", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用，请先激活 rail-dppo-gpu 环境。")
    config = load_phase_a_config(load_config(args.config))
    action_spec = ActionSpec.from_phase_a_config(config)
    # 冒烟入口只验证 GPU 和动态动作尺寸；正式训练必须使用阶段 D 的完整观察规格。
    observation_spec = ObservationSpec(
        (ObservationFeature("gpu_smoke", 12, "none", 1.0),),
        tuple(sorted({function_id for function_id, _ in action_spec.pairs})),
        tuple(sorted(config.node_resources)), "phase-e-gpu-smoke-v1",
    )
    policy = ConditionalDiffusionMLP(
        observation_spec.dimension, action_spec.action_dim, (64, 64)
    ).cuda()
    schedule = CosineNoiseSchedule(steps=4)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    states = torch.zeros((8, observation_spec.dimension), device="cuda")
    actions = torch.zeros((8, action_spec.action_dim), device="cuda")
    for step in range(args.steps):
        generator = torch.Generator(device="cuda").manual_seed(10_000 + step)
        optimizer.zero_grad(set_to_none=True)
        loss = diffusion_noise_loss(policy, schedule, states, actions, generator)
        loss.backward()
        optimizer.step()
        print(
            f"step={step + 1}/{args.steps} loss={loss.item():.6f} "
            f"device={next(policy.parameters()).device} action_dim={action_spec.action_dim}",
            flush=True,
        )


if __name__ == "__main__":
    main()
