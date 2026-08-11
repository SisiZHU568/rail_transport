"""实现 DDPO 在线阶段的轨迹缓存、优势估计和双层策略更新。"""

from __future__ import annotations

from collections.abc import Sequence
import copy
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from src.dppo_diffusion import (
    ConditionalDiffusionMLP,
    CosineNoiseSchedule,
    DenoisingSample,
)


def _finite_float(name: str, value: float) -> float:
    """把数值参数统一转成有限浮点数，并给出便于定位的错误。"""

    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是有限数。")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} 必须是有限数。")
    return converted


def _positive_integer(name: str, value: int) -> int:
    """校验不能被布尔值冒充的正整数参数。"""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} 必须是正整数。")
    return value


@dataclass(frozen=True)
class DPPOConfig:
    """集中保存 DDPO 在线更新超参数，避免算法内部写死实验规模。"""

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.20
    denoising_discount: float = 0.99
    policy_learning_rate: float = 3e-4
    value_learning_rate: float = 3e-4
    batch_size: int = 64
    update_epochs: int = 10
    gradient_clip_norm: float = 5.0
    diffusion_steps: int = 20
    fine_tuned_steps: int = 5
    value_hidden_dims: tuple[int, ...] = (256, 256)
    seed: int = 13000

    def __post_init__(self) -> None:
        """在训练开始前一次性拦截无效超参数。"""

        gamma = _finite_float("gamma", self.gamma)
        gae_lambda = _finite_float("gae_lambda", self.gae_lambda)
        clip_ratio = _finite_float("clip_ratio", self.clip_ratio)
        denoising_discount = _finite_float(
            "denoising_discount",
            self.denoising_discount,
        )
        policy_learning_rate = _finite_float(
            "policy_learning_rate",
            self.policy_learning_rate,
        )
        value_learning_rate = _finite_float(
            "value_learning_rate",
            self.value_learning_rate,
        )
        gradient_clip_norm = _finite_float(
            "gradient_clip_norm",
            self.gradient_clip_norm,
        )
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma 必须位于 (0, 1]。")
        if not 0.0 <= gae_lambda <= 1.0:
            raise ValueError("gae_lambda 必须位于 [0, 1]。")
        if not 0.0 < clip_ratio < 1.0:
            raise ValueError("clip_ratio 必须位于 (0, 1)。")
        if not 0.0 < denoising_discount <= 1.0:
            raise ValueError("denoising_discount 必须位于 (0, 1]。")
        if policy_learning_rate <= 0.0 or value_learning_rate <= 0.0:
            raise ValueError("策略和价值网络学习率必须大于零。")
        if gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm 必须大于零。")

        diffusion_steps = _positive_integer("diffusion_steps", self.diffusion_steps)
        fine_tuned_steps = _positive_integer(
            "fine_tuned_steps",
            self.fine_tuned_steps,
        )
        if fine_tuned_steps > diffusion_steps:
            raise ValueError("fine_tuned_steps 不能超过 diffusion_steps。")
        _positive_integer("batch_size", self.batch_size)
        _positive_integer("update_epochs", self.update_epochs)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed 必须是非负整数。")

        hidden_dims = tuple(self.value_hidden_dims)
        if not hidden_dims:
            raise ValueError("value_hidden_dims 至少需要一层。")
        for width in hidden_dims:
            _positive_integer("value_hidden_dims 中的宽度", width)

        # frozen dataclass 仍可在初始化阶段把列表规范化为不可变元组。
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "gae_lambda", gae_lambda)
        object.__setattr__(self, "clip_ratio", clip_ratio)
        object.__setattr__(self, "denoising_discount", denoising_discount)
        object.__setattr__(self, "policy_learning_rate", policy_learning_rate)
        object.__setattr__(self, "value_learning_rate", value_learning_rate)
        object.__setattr__(self, "gradient_clip_norm", gradient_clip_norm)
        object.__setattr__(self, "value_hidden_dims", hidden_dims)


class ValueNetwork(nn.Module):
    """估计一次慢尺度联合动作之前的状态价值。"""

    def __init__(self, state_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        _positive_integer("state_dim", state_dim)
        widths = tuple(hidden_dims)
        if not widths:
            raise ValueError("hidden_dims 至少需要一层。")
        for width in widths:
            _positive_integer("hidden_dims 中的宽度", width)

        layers: list[nn.Module] = []
        input_width = state_dim
        for output_width in widths:
            layers.extend((nn.Linear(input_width, output_width), nn.SiLU()))
            input_width = output_width
        layers.append(nn.Linear(input_width, 1))
        self.state_dim = state_dim
        self.hidden_dims = widths
        self.network = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """返回每个状态的一个标量价值。"""

        if (
            not isinstance(states, torch.Tensor)
            or states.ndim != 2
            or states.shape[1] != self.state_dim
        ):
            raise ValueError(f"states 必须具有形状 (batch, {self.state_dim})。")
        return self.network(states).squeeze(-1)


def _immutable_float_array(name: str, value: np.ndarray, ndim: int) -> np.ndarray:
    """保存只读浮点副本，防止环境复用数组时悄悄改写旧轨迹。"""

    array = np.asarray(value, dtype=np.float32)
    if array.ndim != ndim:
        raise ValueError(f"{name} 必须是 {ndim} 维数组。")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} 必须只包含有限数。")
    result = np.array(array, dtype=np.float32, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class DPPORolloutTransition:
    """保存一个环境转移及其内部完整扩散去噪轨迹。"""

    state: np.ndarray
    raw_action: np.ndarray
    denoising_actions: np.ndarray
    old_log_probabilities: np.ndarray
    reward: float
    value: float
    terminated: bool

    def __post_init__(self) -> None:
        """校验环境层与扩散层的动作、步数和概率严格对齐。"""

        state = _immutable_float_array("state", self.state, 1)
        raw_action = _immutable_float_array("raw_action", self.raw_action, 1)
        denoising_actions = _immutable_float_array(
            "denoising_actions",
            self.denoising_actions,
            2,
        )
        old_log_probabilities = _immutable_float_array(
            "old_log_probabilities",
            self.old_log_probabilities,
            1,
        )
        if denoising_actions.shape[0] != old_log_probabilities.shape[0] + 1:
            raise ValueError("完整动作链长度必须比旧对数概率数量多 1。")
        if denoising_actions.shape[1] != raw_action.shape[0]:
            raise ValueError("raw_action 维度必须与去噪动作维度一致。")
        if not np.allclose(raw_action, denoising_actions[-1], rtol=1e-6, atol=1e-7):
            raise ValueError("raw_action 必须等于去噪链最后一个动作 a_0。")
        reward = _finite_float("reward", self.reward)
        value = _finite_float("value", self.value)
        if not isinstance(self.terminated, (bool, np.bool_)):
            raise ValueError("terminated 必须是布尔值。")

        object.__setattr__(self, "state", state)
        object.__setattr__(self, "raw_action", raw_action)
        object.__setattr__(self, "denoising_actions", denoising_actions)
        object.__setattr__(self, "old_log_probabilities", old_log_probabilities)
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "terminated", bool(self.terminated))


class DPPORolloutBuffer:
    """只保存当前策略产生的一批 on-policy 环境转移。"""

    def __init__(self) -> None:
        self.transitions: list[DPPORolloutTransition] = []

    def append(self, transition: DPPORolloutTransition) -> None:
        """追加一个联合动作，并检查同批数据的维度保持一致。"""

        if not isinstance(transition, DPPORolloutTransition):
            raise TypeError("transition 必须是 DPPORolloutTransition。")
        if self.transitions:
            reference = self.transitions[0]
            for name in (
                "state",
                "raw_action",
                "denoising_actions",
                "old_log_probabilities",
            ):
                if getattr(transition, name).shape != getattr(reference, name).shape:
                    raise ValueError(f"同一 rollout 中的 {name} 形状必须一致。")
        self.transitions.append(transition)

    def clear(self) -> None:
        """清除旧策略数据，保证下一批仍满足 on-policy 条件。"""

        self.transitions.clear()

    def __len__(self) -> int:
        return len(self.transitions)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    terminated: np.ndarray,
    *,
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """按环境时间倒序计算广义优势估计和价值网络目标。"""

    rewards_array = np.asarray(rewards, dtype=np.float64)
    values_array = np.asarray(values, dtype=np.float64)
    terminated_array = np.asarray(terminated, dtype=np.bool_)
    if rewards_array.ndim != 1 or rewards_array.size == 0:
        raise ValueError("rewards 必须是一维非空数组。")
    if values_array.shape != rewards_array.shape:
        raise ValueError("values 形状必须与 rewards 一致。")
    if terminated_array.shape != rewards_array.shape:
        raise ValueError("terminated 形状必须与 rewards 一致。")
    if not np.isfinite(rewards_array).all() or not np.isfinite(values_array).all():
        raise ValueError("rewards 和 values 必须只包含有限数。")
    bootstrap = _finite_float("next_value", next_value)
    gamma_value = _finite_float("gamma", gamma)
    lambda_value = _finite_float("gae_lambda", gae_lambda)
    if not 0.0 <= gamma_value <= 1.0:
        raise ValueError("gamma 必须位于 [0, 1]。")
    if not 0.0 <= lambda_value <= 1.0:
        raise ValueError("gae_lambda 必须位于 [0, 1]。")

    advantages = np.zeros_like(rewards_array, dtype=np.float64)
    next_advantage = 0.0
    for index in reversed(range(rewards_array.size)):
        nonterminal = 0.0 if terminated_array[index] else 1.0
        following_value = (
            bootstrap if index == rewards_array.size - 1 else values_array[index + 1]
        )
        temporal_difference = (
            rewards_array[index]
            + gamma_value * following_value * nonterminal
            - values_array[index]
        )
        next_advantage = (
            temporal_difference
            + gamma_value * lambda_value * nonterminal * next_advantage
        )
        advantages[index] = next_advantage
    return advantages, advantages + values_array


def clipped_policy_surrogate(
    new_log_probabilities: torch.Tensor,
    old_log_probabilities: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算 PPO 裁剪损失，并同时返回原始和裁剪后的概率比供审计。"""

    if (
        new_log_probabilities.shape != old_log_probabilities.shape
        or advantages.shape != new_log_probabilities.shape
    ):
        raise ValueError("新旧对数概率和优势必须具有完全相同的形状。")
    if not (
        torch.isfinite(new_log_probabilities).all()
        and torch.isfinite(old_log_probabilities).all()
        and torch.isfinite(advantages).all()
    ):
        raise ValueError("新旧对数概率和优势必须只包含有限数。")
    clip = _finite_float("clip_ratio", clip_ratio)
    if not 0.0 < clip < 1.0:
        raise ValueError("clip_ratio 必须位于 (0, 1)。")

    # exp(log π_new - log π_old) 明确使用采样时保存的旧策略概率。
    ratios = torch.exp(new_log_probabilities - old_log_probabilities)
    clipped_ratios = ratios.clamp(1.0 - clip, 1.0 + clip)
    surrogate = torch.minimum(ratios * advantages, clipped_ratios * advantages)
    return -surrogate.mean(), ratios, clipped_ratios


def _gaussian_log_probability(
    samples: torch.Tensor,
    means: torch.Tensor,
    standard_deviations: torch.Tensor,
) -> torch.Tensor:
    """计算独立动作维高斯分布的联合对数概率。"""

    if samples.shape != means.shape or samples.shape != standard_deviations.shape:
        raise ValueError("高斯样本、均值和标准差形状必须一致。")
    if torch.any(standard_deviations <= 0.0):
        raise ValueError("高斯标准差必须大于零。")
    normalized = (samples - means) / standard_deviations
    elementwise = -0.5 * (
        normalized.square()
        + 2.0 * standard_deviations.log()
        + math.log(2.0 * math.pi)
    )
    return elementwise.sum(dim=-1)


class DPPOAgent:
    """使用固定扩散前段探索，并以 PPO 微调最后若干去噪步骤。"""

    def __init__(
        self,
        pretrained_policy: ConditionalDiffusionMLP,
        schedule: CosineNoiseSchedule,
        config: DPPOConfig,
        *,
        device: str | torch.device,
    ) -> None:
        if not isinstance(pretrained_policy, ConditionalDiffusionMLP):
            raise TypeError("pretrained_policy 必须是 ConditionalDiffusionMLP。")
        if not isinstance(schedule, CosineNoiseSchedule):
            raise TypeError("schedule 必须是 CosineNoiseSchedule。")
        if not isinstance(config, DPPOConfig):
            raise TypeError("config 必须是 DPPOConfig。")
        resolved_device = torch.device(device)
        if resolved_device.type not in {"cpu", "cuda"}:
            raise ValueError("DDPO 训练设备只能是 cpu 或 cuda。")
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("配置请求 CUDA，但当前环境不可用。")
        if schedule.steps != config.diffusion_steps:
            raise ValueError("schedule.steps 必须与 config.diffusion_steps 一致。")

        self.config = config
        self.schedule = schedule
        self.device = resolved_device
        self.frozen_denoising_steps = config.diffusion_steps - config.fine_tuned_steps
        self.trainable_denoising_steps = config.fine_tuned_steps

        # 两个副本初始参数相同；固定副本永不进入优化器。
        self.frozen_policy = copy.deepcopy(pretrained_policy).to(resolved_device)
        self.trainable_policy = copy.deepcopy(pretrained_policy).to(resolved_device)
        self.frozen_policy.eval()
        for parameter in self.frozen_policy.parameters():
            parameter.requires_grad_(False)
        self.trainable_policy.train()

        # 独立随机上下文确保价值网络初始化可复现，又不污染调用方 RNG。
        with torch.random.fork_rng():
            torch.manual_seed(config.seed)
            value_network = ValueNetwork(
                pretrained_policy.state_dim,
                config.value_hidden_dims,
            )
        self.value_network = value_network.to(resolved_device)
        self.policy_optimizer = torch.optim.Adam(
            self.trainable_policy.parameters(),
            lr=config.policy_learning_rate,
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_network.parameters(),
            lr=config.value_learning_rate,
        )

    @property
    def state_dim(self) -> int:
        return self.trainable_policy.state_dim

    @property
    def action_dim(self) -> int:
        return self.trainable_policy.action_dim

    def _state_tensor(self, states: np.ndarray | torch.Tensor) -> torch.Tensor:
        """把单状态或状态批次统一送到策略所在设备。"""

        parameter = next(self.trainable_policy.parameters())
        tensor = torch.as_tensor(states, dtype=parameter.dtype, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[1] != self.state_dim:
            raise ValueError(f"states 必须具有形状 ({self.state_dim},) 或 (batch, {self.state_dim})。")
        if not torch.isfinite(tensor).all():
            raise ValueError("states 必须只包含有限数。")
        return tensor

    def estimate_value(self, states: np.ndarray | torch.Tensor) -> float | np.ndarray:
        """估计慢尺度状态价值；单状态返回 float，批量状态返回数组。"""

        is_single = np.ndim(states) == 1 if not isinstance(states, torch.Tensor) else states.ndim == 1
        state_tensor = self._state_tensor(states)
        with torch.no_grad():
            values = self.value_network(state_tensor)
        if is_single:
            return float(values[0].item())
        return values.detach().cpu().numpy()

    def value(self, states: np.ndarray | torch.Tensor) -> float | np.ndarray:
        """提供训练循环使用的简洁价值接口。"""

        return self.estimate_value(states)

    def sample_action(
        self,
        states: np.ndarray | torch.Tensor,
        *,
        seed: int,
    ) -> DenoisingSample:
        """由固定前段和可训练末段共同生成一条完整联合动作轨迹。"""

        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed 必须是非负整数。")
        state_tensor = self._state_tensor(states)
        parameter = next(self.trainable_policy.parameters())
        generator = torch.Generator(device=self.device).manual_seed(seed)
        current_actions = torch.randn(
            (state_tensor.shape[0], self.action_dim),
            generator=generator,
            device=self.device,
            dtype=parameter.dtype,
        )
        action_history = [current_actions]
        mean_history: list[torch.Tensor] = []
        deviation_history: list[torch.Tensor] = []
        log_probability_history: list[torch.Tensor] = []

        # 采样只记录轨迹；更新阶段会用当前策略重新计算末段概率。
        with torch.no_grad():
            for chain_index, timestep_value in enumerate(
                reversed(range(self.schedule.steps))
            ):
                policy = (
                    self.frozen_policy
                    if chain_index < self.frozen_denoising_steps
                    else self.trainable_policy
                )
                timesteps = torch.full(
                    (state_tensor.shape[0],),
                    timestep_value,
                    device=self.device,
                    dtype=torch.long,
                )
                predicted_noise = policy(current_actions, timesteps, state_tensor)
                means = self.schedule.reverse_mean(
                    current_actions,
                    predicted_noise,
                    timesteps,
                )
                standard_deviations = self.schedule.reverse_standard_deviation(
                    current_actions,
                    timesteps,
                )
                noise = torch.randn(
                    current_actions.shape,
                    generator=generator,
                    device=self.device,
                    dtype=current_actions.dtype,
                )
                next_actions = means + standard_deviations * noise
                log_probabilities = _gaussian_log_probability(
                    next_actions,
                    means,
                    standard_deviations,
                )
                action_history.append(next_actions)
                mean_history.append(means)
                deviation_history.append(standard_deviations)
                log_probability_history.append(log_probabilities)
                current_actions = next_actions

        return DenoisingSample(
            actions=torch.stack(action_history),
            means=torch.stack(mean_history),
            standard_deviations=torch.stack(deviation_history),
            log_probabilities=torch.stack(log_probability_history),
        )

    def _current_trainable_log_probabilities(
        self,
        states: torch.Tensor,
        denoising_actions: torch.Tensor,
    ) -> torch.Tensor:
        """只重算最后 M 个去噪转移在当前策略下的对数概率。"""

        log_probabilities: list[torch.Tensor] = []
        for chain_index in range(
            self.frozen_denoising_steps,
            self.config.diffusion_steps,
        ):
            timestep_value = self.config.diffusion_steps - 1 - chain_index
            timesteps = torch.full(
                (states.shape[0],),
                timestep_value,
                device=self.device,
                dtype=torch.long,
            )
            current_actions = denoising_actions[:, chain_index]
            next_actions = denoising_actions[:, chain_index + 1]
            predicted_noise = self.trainable_policy(
                current_actions,
                timesteps,
                states,
            )
            means = self.schedule.reverse_mean(
                current_actions,
                predicted_noise,
                timesteps,
            )
            standard_deviations = self.schedule.reverse_standard_deviation(
                current_actions,
                timesteps,
            )
            log_probabilities.append(
                _gaussian_log_probability(
                    next_actions,
                    means,
                    standard_deviations,
                )
            )
        return torch.stack(log_probabilities, dim=1)

    @staticmethod
    def _require_finite_loss(name: str, loss: torch.Tensor) -> None:
        """在反向传播前终止非有限损失，避免污染模型参数。"""

        if loss.ndim != 0 or not torch.isfinite(loss):
            raise FloatingPointError(f"{name} 出现非有限值。")

    @staticmethod
    def _clip_and_check_gradients(
        name: str,
        parameters: Sequence[torch.nn.Parameter],
        maximum_norm: float,
    ) -> float:
        """裁剪梯度，检查并返回裁剪前的梯度范数。"""

        norm = torch.nn.utils.clip_grad_norm_(parameters, maximum_norm)
        if not torch.isfinite(norm):
            raise FloatingPointError(f"{name} 梯度出现非有限值。")
        return float(norm.detach().cpu().item())

    def update(
        self,
        buffer: DPPORolloutBuffer,
        *,
        next_value: float = 0.0,
    ) -> dict[str, float]:
        """用当前 rollout 执行多轮裁剪 PPO 和价值网络更新。"""

        if not isinstance(buffer, DPPORolloutBuffer):
            raise TypeError("buffer 必须是 DPPORolloutBuffer。")
        if len(buffer) == 0:
            raise ValueError("空 rollout buffer 不能执行更新。")
        transitions = tuple(buffer.transitions)
        expected_chain_shape = (
            self.config.diffusion_steps + 1,
            self.action_dim,
        )
        for transition in transitions:
            if transition.state.shape != (self.state_dim,):
                raise ValueError("rollout state_dim 与策略不一致。")
            if transition.denoising_actions.shape != expected_chain_shape:
                raise ValueError("rollout 去噪链形状与策略不一致。")

        rewards = np.asarray([item.reward for item in transitions], dtype=np.float64)
        old_values = np.asarray([item.value for item in transitions], dtype=np.float64)
        terminated = np.asarray(
            [item.terminated for item in transitions],
            dtype=np.bool_,
        )
        advantages, returns = compute_gae(
            rewards,
            old_values,
            terminated,
            next_value=next_value,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

        parameter = next(self.trainable_policy.parameters())
        states = torch.as_tensor(
            np.stack([item.state for item in transitions]),
            dtype=parameter.dtype,
            device=self.device,
        )
        chains = torch.as_tensor(
            np.stack([item.denoising_actions for item in transitions]),
            dtype=parameter.dtype,
            device=self.device,
        )
        old_log_probabilities = torch.as_tensor(
            np.stack([item.old_log_probabilities for item in transitions])[
                :, self.frozen_denoising_steps :
            ],
            dtype=parameter.dtype,
            device=self.device,
        )
        outer_advantages = torch.as_tensor(
            advantages,
            dtype=parameter.dtype,
            device=self.device,
        )
        value_targets = torch.as_tensor(
            returns,
            dtype=parameter.dtype,
            device=self.device,
        )

        # 最接近最终动作 a_0 的转移指数为 0；越靠前的可训练步骤折扣越多。
        denoising_exponents = torch.arange(
            self.trainable_denoising_steps - 1,
            -1,
            -1,
            device=self.device,
            dtype=parameter.dtype,
        )
        denoising_weights = self.config.denoising_discount**denoising_exponents

        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        policy_losses: list[float] = []
        value_losses: list[float] = []
        approximate_kls: list[float] = []
        clip_fractions: list[float] = []
        gradient_norms: list[float] = []
        sample_count = len(transitions)
        for _ in range(self.config.update_epochs):
            permutation = torch.randperm(sample_count, generator=generator)
            for start in range(0, sample_count, self.config.batch_size):
                cpu_indices = permutation[start : start + self.config.batch_size]
                indices = cpu_indices.to(self.device)
                batch_states = states.index_select(0, indices)
                batch_chains = chains.index_select(0, indices)
                batch_old_log_probabilities = old_log_probabilities.index_select(
                    0,
                    indices,
                )
                batch_advantages = outer_advantages.index_select(0, indices)
                step_advantages = (
                    batch_advantages.unsqueeze(1) * denoising_weights.unsqueeze(0)
                )

                self.policy_optimizer.zero_grad(set_to_none=True)
                new_log_probabilities = self._current_trainable_log_probabilities(
                    batch_states,
                    batch_chains,
                )
                policy_loss, ratios, clipped_ratios = clipped_policy_surrogate(
                    new_log_probabilities,
                    batch_old_log_probabilities,
                    step_advantages,
                    clip_ratio=self.config.clip_ratio,
                )
                self._require_finite_loss("policy_loss", policy_loss)
                policy_loss.backward()
                policy_parameters = tuple(self.trainable_policy.parameters())
                policy_gradient_norm = self._clip_and_check_gradients(
                    "策略网络",
                    policy_parameters,
                    self.config.gradient_clip_norm,
                )
                self.policy_optimizer.step()

                self.value_optimizer.zero_grad(set_to_none=True)
                predicted_values = self.value_network(batch_states)
                batch_targets = value_targets.index_select(0, indices)
                value_loss = functional.mse_loss(predicted_values, batch_targets)
                self._require_finite_loss("value_loss", value_loss)
                value_loss.backward()
                value_parameters = tuple(self.value_network.parameters())
                self._clip_and_check_gradients(
                    "价值网络",
                    value_parameters,
                    self.config.gradient_clip_norm,
                )
                self.value_optimizer.step()

                with torch.no_grad():
                    approximate_kl = (
                        batch_old_log_probabilities - new_log_probabilities
                    ).mean()
                    clip_fraction = (
                        torch.abs(ratios - clipped_ratios) > 0.0
                    ).to(dtype=parameter.dtype).mean()
                policy_losses.append(float(policy_loss.detach().cpu().item()))
                value_losses.append(float(value_loss.detach().cpu().item()))
                approximate_kls.append(float(approximate_kl.cpu().item()))
                clip_fractions.append(float(clip_fraction.cpu().item()))
                gradient_norms.append(policy_gradient_norm)

        metrics = {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "approximate_kl": float(np.mean(approximate_kls)),
            "clip_fraction": float(np.mean(clip_fractions)),
            "gradient_norm": float(np.mean(gradient_norms)),
        }
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("DDPO 更新指标出现非有限值。")

        # 只有完整更新成功后才清空，异常时保留数据便于复现和排查。
        buffer.clear()
        return metrics
