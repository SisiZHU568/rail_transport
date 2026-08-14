"""验证从官方 DPPO 适配的最小算法核心。"""

import pytest
import torch

from src.config import load_config
from src.dppo import DPPOConfig, _gaussian_log_probability
from src.dppo_official_core import (
    official_clip_schedule,
    official_dppo_policy_loss,
)
from src.dppo_training_config import build_dppo_agent_config


def test_official_clip_schedule_grows_from_base_to_maximum() -> None:
    """不同去噪阶段应使用官方实现中的指数增长裁剪范围。"""

    schedule = official_clip_schedule(
        fine_tuned_steps=5,
        maximum_clip_ratio=0.20,
        base_clip_ratio=0.001,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert schedule.shape == (5,)
    assert schedule[0].item() == pytest.approx(0.001)
    assert schedule[-1].item() == pytest.approx(0.20)
    assert torch.all(schedule[1:] > schedule[:-1])


def test_single_fine_tuned_step_uses_maximum_clip_ratio() -> None:
    """只有一个可训练去噪步骤时直接使用最终裁剪范围。"""

    schedule = official_clip_schedule(
        fine_tuned_steps=1,
        maximum_clip_ratio=0.15,
        base_clip_ratio=0.001,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    assert schedule.tolist() == pytest.approx([0.15])


def test_official_policy_loss_is_finite_and_backpropagates() -> None:
    """官方 DPPO 损失应保持可微，并向当前策略概率传递有限梯度。"""

    new_log_probabilities = torch.tensor(
        [[0.00, 0.05, -0.03]],
        dtype=torch.float64,
        requires_grad=True,
    )
    old_log_probabilities = torch.zeros_like(new_log_probabilities)
    advantages = torch.tensor([1.5], dtype=torch.float64)

    result = official_dppo_policy_loss(
        new_log_probabilities,
        old_log_probabilities,
        advantages,
        gamma_denoising=0.99,
        maximum_clip_ratio=0.20,
        base_clip_ratio=0.001,
    )

    assert torch.isfinite(result.policy_loss)
    assert torch.isfinite(result.approximate_kl)
    assert 0.0 <= result.clip_fraction.item() <= 1.0
    result.policy_loss.backward()
    assert new_log_probabilities.grad is not None
    assert torch.isfinite(new_log_probabilities.grad).all()


def test_denoising_log_probability_averages_action_dimensions() -> None:
    """轨道联合动作必须采用官方 DPPO 的动作维平均概率口径。"""

    samples = torch.tensor([[0.0, 1.0, -1.0]], dtype=torch.float64)
    means = torch.zeros_like(samples)
    standard_deviations = torch.ones_like(samples)
    expected = torch.distributions.Normal(
        means,
        standard_deviations,
    ).log_prob(samples).mean(dim=-1)

    actual = _gaussian_log_probability(
        samples,
        means,
        standard_deviations,
    )

    assert torch.equal(actual, expected)


def test_official_clip_settings_are_loaded_from_yaml() -> None:
    """官方裁剪调度参数应由配置文件统一传入代理，而不是写死在算法中。"""

    config = load_config("configs/debug.yaml")
    agent_config = build_dppo_agent_config(config, clip_ratio=0.20)

    assert agent_config.clip_ratio_base == pytest.approx(0.001)


@pytest.mark.parametrize(
    "settings",
    (
        {"clip_ratio": 0.10, "clip_ratio_base": 0.20},
    ),
)
def test_dppo_config_rejects_invalid_official_clip_settings(
    settings: dict[str, float],
) -> None:
    """裁剪下限不能超过上限，指数增长速率也必须为正。"""

    with pytest.raises(ValueError):
        DPPOConfig(**settings)
