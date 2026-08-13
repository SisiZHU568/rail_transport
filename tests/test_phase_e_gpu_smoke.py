import torch

from src.config import load_config
from src.dppo import DPPOAgent, DPPOConfig
from src.dppo_diffusion import ConditionalDiffusionMLP, CosineNoiseSchedule, diffusion_noise_loss
from src.orchestration_config import load_phase_a_config
from src.phase_d_policy_specs import ObservationFeature, ObservationSpec
from src.safe_deployment_decoder import ActionSpec


def test_config_driven_phase_e_policy_runs_one_gpu_training_step() -> None:
    if not torch.cuda.is_available():
        return
    config = load_phase_a_config(load_config("configs/debug.yaml"))
    action_spec = ActionSpec.from_phase_a_config(config)
    observation_spec = ObservationSpec(
        (ObservationFeature("smoke", 12, "none", 1.0),),
        (0, 1, 2), tuple(sorted(config.node_resources)), "gpu-smoke-v1",
    )
    policy = ConditionalDiffusionMLP(observation_spec.dimension, action_spec.action_dim, (32, 32)).cuda()
    schedule = CosineNoiseSchedule(steps=4)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    states = torch.zeros((2, observation_spec.dimension), device="cuda")
    actions = torch.zeros((2, action_spec.action_dim), device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(7)

    optimizer.zero_grad()
    loss = diffusion_noise_loss(policy, schedule, states, actions, generator)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert next(policy.parameters()).is_cuda
    agent = DPPOAgent(
        policy, schedule,
        DPPOConfig(diffusion_steps=4, fine_tuned_steps=2, value_hidden_dims=(32,)),
        device="cuda",
    )
    assert agent.action_dim == action_spec.action_dim == 36
