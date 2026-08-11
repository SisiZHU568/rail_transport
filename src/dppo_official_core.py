"""适配官方 DPPO 的去噪折扣与 PPO 策略损失。

上游来源：https://github.com/irom-princeton/dppo
主要参考：model/diffusion/diffusion_ppo.py 中的 ``PPODiffusion.loss``。
本文件只把官方张量布局适配为本项目使用的 ``(批次, 去噪步骤)``，
轨道状态编码、联合动作解释和约束修复仍由本项目原有模块负责。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class OfficialDPPOLoss:
    """保存一次官方 DPPO 策略损失及其核心诊断量。"""

    policy_loss: torch.Tensor
    approximate_kl: torch.Tensor
    clip_fraction: torch.Tensor
    mean_ratio: torch.Tensor
    clip_ratios: torch.Tensor


def official_clip_schedule(
    *,
    fine_tuned_steps: int,
    maximum_clip_ratio: float,
    base_clip_ratio: float,
    growth_rate: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """按官方 DPPO 公式生成各去噪步骤的 PPO 裁剪范围。"""

    if isinstance(fine_tuned_steps, bool) or fine_tuned_steps <= 0:
        raise ValueError("fine_tuned_steps 必须是正整数。")
    maximum = float(maximum_clip_ratio)
    base = float(base_clip_ratio)
    rate = float(growth_rate)
    if not (0.0 < base <= maximum < 1.0):
        raise ValueError("裁剪范围必须满足 0 < base <= maximum < 1。")
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("growth_rate 必须是正有限数。")

    # 只有一个可训练去噪步骤时，它同时也是最接近最终动作的步骤。
    if fine_tuned_steps == 1:
        return torch.full(
            (1,),
            maximum,
            device=device,
            dtype=dtype,
        )

    indices = torch.arange(fine_tuned_steps, device=device, dtype=dtype)
    progress = indices / (fine_tuned_steps - 1)
    # 官方实现采用指数插值：早期去噪更新保守，接近最终动作时逐步放宽。
    return base + (maximum - base) * (
        torch.exp(rate * progress) - 1.0
    ) / math.expm1(rate)


def official_dppo_policy_loss(
    new_log_probabilities: torch.Tensor,
    old_log_probabilities: torch.Tensor,
    advantages: torch.Tensor,
    *,
    gamma_denoising: float,
    maximum_clip_ratio: float,
    base_clip_ratio: float,
    growth_rate: float,
) -> OfficialDPPOLoss:
    """计算官方 DPPO 的去噪折扣 clipped PPO 策略损失。

    ``new_log_probabilities`` 和 ``old_log_probabilities`` 的形状为
    ``(B, K)``。这与官方实现把环境批次和去噪步骤展平后的计算等价。
    """

    if (
        new_log_probabilities.ndim != 2
        or new_log_probabilities.shape != old_log_probabilities.shape
    ):
        raise ValueError("新旧对数概率必须具有相同的二维形状 (B, K)。")
    if advantages.shape != (new_log_probabilities.shape[0],):
        raise ValueError("advantages 必须为每个环境样本提供一个标量。")
    gamma = float(gamma_denoising)
    if not math.isfinite(gamma) or not 0.0 < gamma <= 1.0:
        raise ValueError("gamma_denoising 必须位于 (0, 1]。")

    fine_tuned_steps = new_log_probabilities.shape[1]
    clip_ratios = official_clip_schedule(
        fine_tuned_steps=fine_tuned_steps,
        maximum_clip_ratio=maximum_clip_ratio,
        base_clip_ratio=base_clip_ratio,
        growth_rate=growth_rate,
        device=new_log_probabilities.device,
        dtype=new_log_probabilities.dtype,
    )

    # 环境优势先在完整轨迹上归一化；这里再按去噪 MDP 的时间方向折扣。
    exponents = torch.arange(
        fine_tuned_steps - 1,
        -1,
        -1,
        device=new_log_probabilities.device,
        dtype=new_log_probabilities.dtype,
    )
    weighted_advantages = advantages.unsqueeze(1) * (gamma**exponents)

    log_ratio = new_log_probabilities - old_log_probabilities
    ratio = log_ratio.exp()
    lower_ratio = 1.0 - clip_ratios
    upper_ratio = 1.0 + clip_ratios
    clipped_ratio = torch.maximum(
        torch.minimum(ratio, upper_ratio),
        lower_ratio,
    )

    policy_loss_unclipped = -weighted_advantages * ratio
    policy_loss_clipped = -weighted_advantages * clipped_ratio
    policy_loss = torch.maximum(
        policy_loss_unclipped,
        policy_loss_clipped,
    ).mean()

    # 与官方代码一致，使用 Schulman k3 估计器报告近似 KL。
    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
    clip_fraction = (
        (ratio - 1.0).abs() > clip_ratios
    ).to(dtype=new_log_probabilities.dtype).mean()
    return OfficialDPPOLoss(
        policy_loss=policy_loss,
        approximate_kl=approximate_kl,
        clip_fraction=clip_fraction,
        mean_ratio=ratio.mean(),
        clip_ratios=clip_ratios,
    )
