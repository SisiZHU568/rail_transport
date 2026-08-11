"""实现 DPPO 使用的条件扩散动作网络和完整反向去噪轨迹。"""

from collections.abc import Sequence
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as functional


class CosineNoiseSchedule:
    """预计算 DDPM 余弦噪声调度及反向后验方差。"""

    def __init__(
        self,
        steps: int,
        *,
        cosine_offset: float = 0.008,
        maximum_beta: float = 0.999,
    ) -> None:
        """根据扩散步数构造前向噪声和反向采样所需常量。"""

        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError("steps 必须是正整数。")
        for name, value in (
            ("cosine_offset", cosine_offset),
            ("maximum_beta", maximum_beta),
        ):
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"{name} 必须是有限数。")
        if float(cosine_offset) < 0.0:
            raise ValueError("cosine_offset 不能小于零。")
        if not 0.0 < float(maximum_beta) < 1.0:
            raise ValueError("maximum_beta 必须位于 (0, 1)。")

        self.steps = steps
        # 先计算 t=0...T 共 T+1 个累计信号比例，再转成每一步 beta。
        time_points = torch.linspace(0.0, 1.0, steps + 1, dtype=torch.float64)
        offset = float(cosine_offset)
        cumulative = torch.cos(
            (time_points + offset) / (1.0 + offset) * math.pi / 2.0
        ).square()
        cumulative = cumulative / cumulative[0]
        betas = 1.0 - cumulative[1:] / cumulative[:-1]
        self.betas = betas.clamp(min=torch.finfo(torch.float64).eps, max=maximum_beta)
        self.alphas = 1.0 - self.betas
        self.alpha_cumulative_products = torch.cumprod(self.alphas, dim=0)
        previous_cumulative = torch.cat(
            (
                torch.ones(1, dtype=torch.float64),
                self.alpha_cumulative_products[:-1],
            )
        )
        # q(a_{t-1}|a_t,a_0) 的理论后验方差；t=0 时等于零。
        self.posterior_variances = (
            self.betas
            * (1.0 - previous_cumulative)
            / (1.0 - self.alpha_cumulative_products)
        )

    def _validate_timesteps(
        self,
        timesteps: torch.Tensor,
        batch_size: int,
    ) -> None:
        """检查每个批样本都有一个合法整数时间步。"""

        if (
            not isinstance(timesteps, torch.Tensor)
            or timesteps.dtype != torch.long
            or timesteps.shape != (batch_size,)
        ):
            raise ValueError("timesteps 必须是形状为 (batch,) 的 long 张量。")
        if batch_size > 0 and (
            int(timesteps.min().item()) < 0
            or int(timesteps.max().item()) >= self.steps
        ):
            raise ValueError(f"timesteps 必须位于 [0, {self.steps - 1}]。")

    @staticmethod
    def _extract(
        values: torch.Tensor,
        timesteps: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """按批提取调度常量，并扩展成可与动作广播的列向量。"""

        selected = values.to(
            device=reference.device,
            dtype=reference.dtype,
        )[timesteps]
        return selected.reshape(reference.shape[0], *([1] * (reference.ndim - 1)))

    def q_sample(
        self,
        clean_actions: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """按闭式公式生成任意时间步的带噪动作 ``a_t``。

        ``a_t = sqrt(alpha_bar_t) * a_0 + sqrt(1-alpha_bar_t) * epsilon``
        """

        if not isinstance(clean_actions, torch.Tensor) or clean_actions.ndim != 2:
            raise ValueError("clean_actions 必须是二维张量。")
        if not clean_actions.is_floating_point():
            raise ValueError("clean_actions 必须是浮点张量。")
        if not isinstance(noise, torch.Tensor) or noise.shape != clean_actions.shape:
            raise ValueError("noise 形状必须与 clean_actions 完全一致。")
        if noise.device != clean_actions.device:
            raise ValueError("noise 和 clean_actions 必须位于同一设备。")
        self._validate_timesteps(timesteps, clean_actions.shape[0])
        if timesteps.device != clean_actions.device:
            raise ValueError("timesteps 和 clean_actions 必须位于同一设备。")
        cumulative = self._extract(
            self.alpha_cumulative_products,
            timesteps,
            clean_actions,
        )
        return cumulative.sqrt() * clean_actions + (1.0 - cumulative).sqrt() * noise

    def reverse_mean(
        self,
        noisy_actions: torch.Tensor,
        predicted_noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """根据网络预测噪声计算 ``p(a_{t-1}|a_t,s)`` 的均值。"""

        if predicted_noise.shape != noisy_actions.shape:
            raise ValueError("predicted_noise 形状必须与 noisy_actions 一致。")
        self._validate_timesteps(timesteps, noisy_actions.shape[0])
        alphas = self._extract(self.alphas, timesteps, noisy_actions)
        betas = self._extract(self.betas, timesteps, noisy_actions)
        cumulative = self._extract(
            self.alpha_cumulative_products,
            timesteps,
            noisy_actions,
        )
        return (
            noisy_actions
            - betas / (1.0 - cumulative).sqrt() * predicted_noise
        ) / alphas.sqrt()

    def reverse_standard_deviation(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        minimum_standard_deviation: float,
    ) -> torch.Tensor:
        """返回带数值下限且扩展到动作形状的反向标准差。"""

        if (
            isinstance(minimum_standard_deviation, bool)
            or not math.isfinite(float(minimum_standard_deviation))
            or float(minimum_standard_deviation) <= 0.0
        ):
            raise ValueError("minimum_standard_deviation 必须是大于零的有限数。")
        self._validate_timesteps(timesteps, noisy_actions.shape[0])
        variances = self._extract(
            self.posterior_variances,
            timesteps,
            noisy_actions,
        )
        standard_deviations = variances.sqrt()
        return standard_deviations.clamp_min(
            float(minimum_standard_deviation)
        ).expand_as(noisy_actions)


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
) -> torch.Tensor:
    """把离散扩散时间步编码为固定正弦/余弦特征。"""

    if (
        not isinstance(embedding_dim, int)
        or isinstance(embedding_dim, bool)
        or embedding_dim <= 0
    ):
        raise ValueError("embedding_dim 必须是正整数。")
    if not isinstance(timesteps, torch.Tensor) or timesteps.ndim != 1:
        raise ValueError("timesteps 必须是一维张量。")
    half_dim = embedding_dim // 2
    if half_dim == 0:
        return timesteps.to(dtype=torch.float32).unsqueeze(1)
    frequency_indices = torch.arange(
        half_dim,
        device=timesteps.device,
        dtype=torch.float32,
    )
    frequencies = torch.exp(
        -math.log(10000.0) * frequency_indices / max(half_dim - 1, 1)
    )
    angles = timesteps.to(dtype=torch.float32).unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
    if embedding_dim % 2 == 1:
        embedding = functional.pad(embedding, (0, 1))
    return embedding


class ConditionalDiffusionMLP(nn.Module):
    """根据状态、带噪动作和时间步预测当前动作噪声。"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
    ) -> None:
        """从场景维度和配置化隐藏层构造条件 MLP。"""

        super().__init__()
        for name, value in (("state_dim", state_dim), ("action_dim", action_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} 必须是正整数。")
        widths = tuple(hidden_dims)
        if not widths or any(
            isinstance(width, bool) or not isinstance(width, int) or width <= 0
            for width in widths
        ):
            raise ValueError("hidden_dims 必须包含至少一个正整数。")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = widths
        self.timestep_embedding_dim = widths[0]

        layers: list[nn.Module] = []
        input_width = state_dim + action_dim + self.timestep_embedding_dim
        for output_width in widths:
            layers.extend((nn.Linear(input_width, output_width), nn.SiLU()))
            input_width = output_width
        layers.append(nn.Linear(input_width, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        states: torch.Tensor,
    ) -> torch.Tensor:
        """预测与输入动作同形状的标准高斯噪声。"""

        if (
            not isinstance(noisy_actions, torch.Tensor)
            or noisy_actions.ndim != 2
            or noisy_actions.shape[1] != self.action_dim
        ):
            raise ValueError(
                f"noisy_actions 必须具有形状 (batch, {self.action_dim})。"
            )
        if (
            not isinstance(states, torch.Tensor)
            or states.ndim != 2
            or states.shape != (noisy_actions.shape[0], self.state_dim)
        ):
            raise ValueError(
                f"states 的 state_dim 必须为 {self.state_dim} 且批大小一致。"
            )
        if states.device != noisy_actions.device or states.dtype != noisy_actions.dtype:
            raise ValueError("states 和 noisy_actions 必须具有相同设备与数据类型。")
        if (
            not isinstance(timesteps, torch.Tensor)
            or timesteps.dtype != torch.long
            or timesteps.shape != (noisy_actions.shape[0],)
            or timesteps.device != noisy_actions.device
        ):
            raise ValueError("timesteps 必须是同设备、形状为 (batch,) 的 long 张量。")
        time_embedding = sinusoidal_timestep_embedding(
            timesteps,
            self.timestep_embedding_dim,
        ).to(dtype=noisy_actions.dtype)
        inputs = torch.cat((noisy_actions, states, time_embedding), dim=1)
        return self.network(inputs)


def diffusion_noise_loss(
    model: ConditionalDiffusionMLP,
    schedule: CosineNoiseSchedule,
    states: torch.Tensor,
    clean_actions: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """随机抽取扩散时间步，计算网络预测噪声的均方误差。"""

    if clean_actions.shape != (states.shape[0], model.action_dim):
        raise ValueError("clean_actions 的批大小或 action_dim 与模型不一致。")
    if states.shape != (clean_actions.shape[0], model.state_dim):
        raise ValueError("states 的批大小或 state_dim 与模型不一致。")
    timesteps = torch.randint(
        0,
        schedule.steps,
        (states.shape[0],),
        generator=generator,
        device=states.device,
    )
    noise = torch.randn(
        clean_actions.shape,
        generator=generator,
        device=clean_actions.device,
        dtype=clean_actions.dtype,
    )
    noisy_actions = schedule.q_sample(clean_actions, timesteps, noise)
    predicted_noise = model(noisy_actions, timesteps, states)
    return functional.mse_loss(predicted_noise, noise)


@dataclass(frozen=True)
class DenoisingSample:
    """保存一次反向扩散的全部动作、分布参数和转移对数概率。"""

    actions: torch.Tensor
    means: torch.Tensor
    standard_deviations: torch.Tensor
    log_probabilities: torch.Tensor


def _gaussian_log_probability(
    samples: torch.Tensor,
    means: torch.Tensor,
    standard_deviations: torch.Tensor,
) -> torch.Tensor:
    """计算独立动作维高斯密度，并对动作维求和。"""

    normalized = (samples - means) / standard_deviations
    elementwise = -0.5 * (
        normalized.square()
        + 2.0 * standard_deviations.log()
        + math.log(2.0 * math.pi)
    )
    return elementwise.sum(dim=-1)


def sample_denoising_chain(
    model: ConditionalDiffusionMLP,
    schedule: CosineNoiseSchedule,
    states: torch.Tensor,
    *,
    seed: int,
    minimum_sampling_standard_deviation: float = 0.001,
) -> DenoisingSample:
    """从标准高斯 ``a_T`` 开始，采样并记录完整条件反向链。"""

    if (
        not isinstance(states, torch.Tensor)
        or states.ndim != 2
        or states.shape[1] != model.state_dim
    ):
        raise ValueError(f"states 的 state_dim 必须为 {model.state_dim}。")
    if not states.is_floating_point():
        raise ValueError("states 必须是浮点张量。")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed 必须是非负整数。")
    try:
        parameter = next(model.parameters())
    except StopIteration as error:
        raise ValueError("model 必须包含可训练参数。") from error
    if parameter.device != states.device or parameter.dtype != states.dtype:
        raise ValueError("model 和 states 必须具有相同设备与数据类型。")

    generator = torch.Generator(device=states.device).manual_seed(seed)
    current_actions = torch.randn(
        (states.shape[0], model.action_dim),
        generator=generator,
        device=states.device,
        dtype=states.dtype,
    )
    action_history = [current_actions]
    mean_history: list[torch.Tensor] = []
    deviation_history: list[torch.Tensor] = []
    log_probability_history: list[torch.Tensor] = []
    # 采样轨迹用于环境交互，不建立反向传播图；训练时会重新计算新策略概率。
    with torch.no_grad():
        for timestep_value in reversed(range(schedule.steps)):
            timesteps = torch.full(
                (states.shape[0],),
                timestep_value,
                device=states.device,
                dtype=torch.long,
            )
            predicted_noise = model(current_actions, timesteps, states)
            means = schedule.reverse_mean(
                current_actions,
                predicted_noise,
                timesteps,
            )
            standard_deviations = schedule.reverse_standard_deviation(
                current_actions,
                timesteps,
                minimum_standard_deviation=minimum_sampling_standard_deviation,
            )
            noise = torch.randn(
                current_actions.shape,
                generator=generator,
                device=states.device,
                dtype=states.dtype,
            )
            next_actions = means + standard_deviations * noise
            log_probabilities = _gaussian_log_probability(
                next_actions,
                means,
                standard_deviations,
            )
            mean_history.append(means)
            deviation_history.append(standard_deviations)
            log_probability_history.append(log_probabilities)
            action_history.append(next_actions)
            current_actions = next_actions
    return DenoisingSample(
        actions=torch.stack(action_history),
        means=torch.stack(mean_history),
        standard_deviations=torch.stack(deviation_history),
        log_probabilities=torch.stack(log_probability_history),
    )
