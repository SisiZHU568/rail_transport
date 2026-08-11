"""验证从官方 DPPO 适配的最小算法核心。"""

import pytest
import torch

from src.dppo_official_core import (
    official_clip_schedule,
    official_dppo_policy_loss,
)


def test_official_clip_schedule_grows_from_base_to_maximum() -> None:
    """不同去噪阶段应使用官方实现中的指数增长裁剪范围。"""

    schedule = official_clip_schedule(
        fine_tuned_steps=5,
        maximum_clip_ratio=0.20,
        base_clip_ratio=0.001,
        growth_rate=3.0,
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
        growth_rate=3.0,
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
        growth_rate=3.0,
    )

    assert torch.isfinite(result.policy_loss)
    assert torch.isfinite(result.approximate_kl)
    assert 0.0 <= result.clip_fraction.item() <= 1.0
    result.policy_loss.backward()
    assert new_log_probabilities.grad is not None
    assert torch.isfinite(new_log_probabilities.grad).all()
