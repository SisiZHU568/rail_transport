"""测试 DPPO 条件扩散策略的噪声调度、训练损失和反向采样。"""

import pytest
import torch

from src.dppo_diffusion import (
    ConditionalDiffusionMLP,
    CosineNoiseSchedule,
    diffusion_noise_loss,
    sample_denoising_chain,
    sinusoidal_timestep_embedding,
)


def test_cosine_schedule_has_finite_probabilities() -> None:
    """余弦调度的 beta 必须严格位于 (0, 1)，累计信号逐步减弱。"""

    schedule = CosineNoiseSchedule(steps=20)

    assert schedule.betas.shape == (20,)
    assert schedule.alphas.shape == (20,)
    assert schedule.alpha_cumulative_products.shape == (20,)
    assert torch.isfinite(schedule.betas).all()
    assert torch.all((schedule.betas > 0.0) & (schedule.betas < 1.0))
    assert torch.all(
        schedule.alpha_cumulative_products[1:]
        < schedule.alpha_cumulative_products[:-1]
    )


def test_q_sample_preserves_batch_action_shape_and_is_seeded() -> None:
    """前向加噪不得改变动作形状，同一噪声必须得到相同结果。"""

    schedule = CosineNoiseSchedule(steps=20)
    clean_actions = torch.zeros((3, 14), dtype=torch.float32)
    noise = torch.arange(42, dtype=torch.float32).reshape(3, 14) / 42.0
    timesteps = torch.tensor([0, 5, 19], dtype=torch.long)

    first = schedule.q_sample(clean_actions, timesteps, noise)
    second = schedule.q_sample(clean_actions, timesteps, noise)

    assert first.shape == clean_actions.shape
    assert first.dtype == clean_actions.dtype
    assert torch.equal(first, second)
    assert torch.linalg.vector_norm(first[2]) > torch.linalg.vector_norm(first[0])


def test_sinusoidal_embedding_supports_odd_dynamic_dimension() -> None:
    """时间嵌入维度由网络结构决定，奇数维也必须正确补齐。"""

    embedding = sinusoidal_timestep_embedding(
        torch.tensor([0, 1, 2], dtype=torch.long),
        embedding_dim=7,
    )

    assert embedding.shape == (3, 7)
    assert torch.isfinite(embedding).all()
    assert not torch.equal(embedding[0], embedding[1])


@pytest.mark.parametrize(
    ("state_dim", "action_dim", "hidden_dims"),
    [(58, 14, (32, 32)), (78, 27, (64, 48, 32))],
)
def test_conditional_mlp_predicts_noise_with_configured_dimensions(
    state_dim: int,
    action_dim: int,
    hidden_dims: tuple[int, ...],
) -> None:
    """改变 MEC/VNF 规模后，网络输入输出维度必须自动跟随配置。"""

    model = ConditionalDiffusionMLP(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
    )
    states = torch.zeros((4, state_dim), dtype=torch.float32)
    noisy_actions = torch.zeros((4, action_dim), dtype=torch.float32)
    timesteps = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    predicted_noise = model(noisy_actions, timesteps, states)

    assert predicted_noise.shape == noisy_actions.shape
    assert torch.isfinite(predicted_noise).all()


def test_diffusion_noise_loss_is_finite_and_differentiable() -> None:
    """专家动作预训练损失必须能够反向传播到条件扩散网络。"""

    torch.manual_seed(11)
    model = ConditionalDiffusionMLP(58, 14, (32, 32))
    schedule = CosineNoiseSchedule(steps=20)
    states = torch.randn((5, 58), dtype=torch.float32)
    clean_actions = torch.tanh(torch.randn((5, 14), dtype=torch.float32))
    generator = torch.Generator(device="cpu").manual_seed(17)

    loss = diffusion_noise_loss(
        model,
        schedule,
        states,
        clean_actions,
        generator,
    )
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_seeded_denoising_chain_is_reproducible() -> None:
    """相同模型、状态和种子必须返回完全相同的反向去噪轨迹。"""

    torch.manual_seed(23)
    schedule = CosineNoiseSchedule(steps=20)
    model = ConditionalDiffusionMLP(
        state_dim=58,
        action_dim=14,
        hidden_dims=(64, 64),
    )
    states = torch.zeros((2, 58), dtype=torch.float32)

    first = sample_denoising_chain(model, schedule, states, seed=31)
    second = sample_denoising_chain(model, schedule, states, seed=31)

    assert first.actions.shape == (21, 2, 14)
    assert first.means.shape == (20, 2, 14)
    assert first.standard_deviations.shape == (20, 2, 14)
    assert first.log_probabilities.shape == (20, 2)
    assert torch.equal(first.actions, second.actions)
    assert torch.equal(first.means, second.means)
    assert torch.equal(first.standard_deviations, second.standard_deviations)
    assert torch.equal(first.log_probabilities, second.log_probabilities)
    assert torch.all(first.standard_deviations > 0.0)
    assert torch.isfinite(first.actions).all()
    assert torch.isfinite(first.log_probabilities).all()


def test_different_sampling_seeds_change_initial_and_final_actions() -> None:
    """改变随机种子应改变探索轨迹，而不是退化为固定确定性动作。"""

    torch.manual_seed(29)
    schedule = CosineNoiseSchedule(steps=5)
    model = ConditionalDiffusionMLP(10, 4, (16, 16))
    states = torch.zeros((2, 10), dtype=torch.float32)

    first = sample_denoising_chain(model, schedule, states, seed=41)
    second = sample_denoising_chain(model, schedule, states, seed=42)

    assert not torch.equal(first.actions[0], second.actions[0])
    assert not torch.equal(first.actions[-1], second.actions[-1])


def test_diffusion_components_reject_wrong_shapes() -> None:
    """状态、动作或时间步错位时必须尽早给出明确错误。"""

    schedule = CosineNoiseSchedule(steps=20)
    model = ConditionalDiffusionMLP(58, 14, (32, 32))

    with pytest.raises(ValueError, match="clean_actions"):
        schedule.q_sample(
            torch.zeros((2, 14)),
            torch.tensor([0, 1]),
            torch.zeros((2, 13)),
        )
    with pytest.raises(ValueError, match="state_dim"):
        sample_denoising_chain(
            model,
            schedule,
            torch.zeros((2, 57)),
            seed=1,
        )
