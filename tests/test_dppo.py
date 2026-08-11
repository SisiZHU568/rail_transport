"""测试 DDPO 在线更新中的轨迹存储、GAE 和裁剪策略优化。"""

import copy
import math

import numpy as np
import pytest
import torch

from src.dppo import (
    DPPOAgent,
    DPPOConfig,
    DPPORolloutBuffer,
    DPPORolloutTransition,
    clipped_policy_surrogate,
    compute_gae,
)
from src.dppo_diffusion import ConditionalDiffusionMLP, CosineNoiseSchedule


def _small_agent(*, diffusion_steps: int = 4, fine_tuned_steps: int = 2) -> DPPOAgent:
    """构造运行速度快、但包含完整双层策略结构的测试智能体。"""

    torch.manual_seed(7)
    policy = ConditionalDiffusionMLP(6, 3, (12, 12))
    schedule = CosineNoiseSchedule(steps=diffusion_steps)
    config = DPPOConfig(
        diffusion_steps=diffusion_steps,
        fine_tuned_steps=fine_tuned_steps,
        value_hidden_dims=(12, 12),
        batch_size=2,
        update_epochs=2,
        policy_learning_rate=1e-3,
        value_learning_rate=1e-3,
        seed=101,
    )
    return DPPOAgent(policy, schedule, config, device="cpu")


def _transition_from_sample(
    agent: DPPOAgent,
    state: np.ndarray,
    *,
    reward: float,
    terminated: bool,
    seed: int,
) -> DPPORolloutTransition:
    """把一次扩散采样整理成一个外层环境转移。"""

    sample = agent.sample_action(state, seed=seed)
    chain = sample.actions[:, 0].cpu().numpy()
    old_log_probabilities = sample.log_probabilities[:, 0].cpu().numpy()
    return DPPORolloutTransition(
        state=state,
        raw_action=chain[-1],
        denoising_actions=chain,
        old_log_probabilities=old_log_probabilities,
        reward=reward,
        value=agent.value(state),
        terminated=terminated,
    )


def test_rollout_transition_keeps_full_denoising_chain_as_immutable_copy() -> None:
    """每个慢尺度动作必须保存完整去噪链，且不受外部数组后续修改影响。"""

    state = np.arange(6, dtype=np.float32)
    chain = np.arange(15, dtype=np.float32).reshape(5, 3)
    old_log_probabilities = np.arange(4, dtype=np.float32)
    transition = DPPORolloutTransition(
        state=state,
        raw_action=chain[-1],
        denoising_actions=chain,
        old_log_probabilities=old_log_probabilities,
        reward=1.5,
        value=0.25,
        terminated=False,
    )

    state[0] = -100.0
    chain[-1, 0] = -100.0
    old_log_probabilities[0] = -100.0

    assert transition.state.shape == (6,)
    assert transition.denoising_actions.shape == (5, 3)
    assert transition.old_log_probabilities.shape == (4,)
    assert transition.state[0] == 0.0
    assert transition.raw_action[0] == 12.0
    assert transition.old_log_probabilities[0] == 0.0
    assert not transition.denoising_actions.flags.writeable


def test_rollout_buffer_clear_removes_all_grouped_transitions() -> None:
    """一次更新完成后必须清空旧策略轨迹，避免破坏 on-policy 假设。"""

    buffer = DPPORolloutBuffer()
    transition = DPPORolloutTransition(
        state=np.zeros(2, dtype=np.float32),
        raw_action=np.zeros(1, dtype=np.float32),
        denoising_actions=np.zeros((3, 1), dtype=np.float32),
        old_log_probabilities=np.zeros(2, dtype=np.float32),
        reward=0.0,
        value=0.0,
        terminated=True,
    )
    buffer.append(transition)

    assert len(buffer) == 1
    buffer.clear()
    assert len(buffer) == 0


def test_gae_matches_three_step_manual_example() -> None:
    """用可手算样例验证 GAE 的倒序递推和回报值。"""

    advantages, returns = compute_gae(
        rewards=np.asarray([1.0, 2.0, 3.0]),
        values=np.asarray([0.5, 1.0, 1.5]),
        terminated=np.asarray([False, False, True]),
        next_value=0.0,
        gamma=1.0,
        gae_lambda=1.0,
    )

    assert np.allclose(returns, [6.0, 5.0, 3.0])
    assert np.allclose(advantages, [5.5, 4.0, 1.5])


def test_gae_ignores_bootstrap_value_after_terminal_transition() -> None:
    """终止状态之后不存在未来价值，即使误传非零 bootstrap 也必须屏蔽。"""

    advantages, returns = compute_gae(
        rewards=np.asarray([2.0]),
        values=np.asarray([0.5]),
        terminated=np.asarray([True]),
        next_value=999.0,
        gamma=0.99,
        gae_lambda=0.95,
    )

    assert np.allclose(advantages, [1.5])
    assert np.allclose(returns, [2.0])


def test_clipped_surrogate_uses_old_log_probability_and_limits_ratio() -> None:
    """概率比必须以采样时旧概率为分母，并由 PPO 区间限制过大更新。"""

    old_log_probabilities = torch.tensor([math.log(0.25), math.log(0.5)])
    new_log_probabilities = torch.tensor([math.log(0.5), math.log(0.5)])
    advantages = torch.ones(2)

    loss, ratios, clipped_ratios = clipped_policy_surrogate(
        new_log_probabilities,
        old_log_probabilities,
        advantages,
        clip_ratio=0.2,
    )

    assert torch.allclose(ratios, torch.tensor([2.0, 1.0]))
    assert torch.allclose(clipped_ratios, torch.tensor([1.2, 1.0]))
    assert loss.item() == pytest.approx(-1.1)


def test_agent_assigns_first_15_steps_to_frozen_policy_and_last_5_to_trainable() -> None:
    """论文默认的 20 步去噪必须明确划分为固定 15 步和微调 5 步。"""

    agent = _small_agent(diffusion_steps=20, fine_tuned_steps=5)

    assert agent.frozen_denoising_steps == 15
    assert agent.trainable_denoising_steps == 5
    assert all(not parameter.requires_grad for parameter in agent.frozen_policy.parameters())
    assert all(parameter.requires_grad for parameter in agent.trainable_policy.parameters())


def test_config_has_separate_validated_diffusion_standard_deviation_floors() -> None:
    """训练探索、PPO 概率和独立评估必须拥有各自可校验的标准差下限。"""

    config = DPPOConfig()

    assert config.training_sampling_min_std == pytest.approx(0.01)
    assert config.probability_min_std == pytest.approx(0.10)
    assert config.evaluation_sampling_min_std == pytest.approx(0.001)
    for field_name in (
        "training_sampling_min_std",
        "probability_min_std",
        "evaluation_sampling_min_std",
    ):
        with pytest.raises(ValueError, match=field_name):
            DPPOConfig(**{field_name: 0.0})


def test_seeded_hybrid_sampler_records_every_reverse_transition() -> None:
    """双层策略采样仍应返回初始噪声、全部中间动作和每步旧概率。"""

    agent = _small_agent()
    state = np.linspace(-1.0, 1.0, 6, dtype=np.float32)

    first = agent.sample_action(state, seed=19)
    second = agent.sample_action(state, seed=19)

    assert first.actions.shape == (5, 1, 3)
    assert first.log_probabilities.shape == (4, 1)
    assert torch.equal(first.actions, second.actions)
    assert torch.equal(first.log_probabilities, second.log_probabilities)
    assert torch.isfinite(first.actions).all()
    assert torch.isfinite(first.log_probabilities).all()


def test_hybrid_sampler_separates_sampling_and_probability_floors() -> None:
    """实际动作按 0.01 探索，但旧策略概率必须在同一动作上按 0.10 计算。"""

    agent = _small_agent()
    state = np.linspace(-1.0, 1.0, 6, dtype=np.float32)

    sample = agent.sample_action(state, seed=53)
    final_actions = sample.actions[-1]
    final_means = sample.means[-1]
    probability_standard_deviations = torch.full_like(final_means, 0.10)
    sampling_standard_deviations = torch.full_like(final_means, 0.01)
    expected_probability_logp = torch.distributions.Normal(
        final_means,
        probability_standard_deviations,
    ).log_prob(final_actions).sum(dim=-1)
    sampling_logp = torch.distributions.Normal(
        final_means,
        sampling_standard_deviations,
    ).log_prob(final_actions).sum(dim=-1)

    assert torch.all(sample.standard_deviations[-1] >= 0.01)
    assert torch.allclose(sample.log_probabilities[-1], expected_probability_logp)
    assert not torch.allclose(sample.log_probabilities[-1], sampling_logp)


def test_current_trainable_log_probabilities_use_probability_floor() -> None:
    """PPO 更新重算的新策略概率必须与采样时保存的旧策略使用同一 0.10 尺度。"""

    agent = _small_agent()
    states = torch.zeros((1, 6), dtype=torch.float32)
    sample = agent.sample_action(states, seed=59)

    current_log_probabilities = agent._current_trainable_log_probabilities(
        states,
        sample.actions.permute(1, 0, 2),
    )
    expected_final_logp = torch.distributions.Normal(
        sample.means[-1],
        torch.full_like(sample.means[-1], 0.10),
    ).log_prob(sample.actions[-1]).sum(dim=-1)
    sampling_final_logp = torch.distributions.Normal(
        sample.means[-1],
        torch.full_like(sample.means[-1], 0.01),
    ).log_prob(sample.actions[-1]).sum(dim=-1)

    assert torch.allclose(current_log_probabilities[:, -1], expected_final_logp)
    assert not torch.allclose(current_log_probabilities[:, -1], sampling_final_logp)


def test_update_changes_only_trainable_policy_and_clears_buffer() -> None:
    """PPO 更新只能改变末段策略和价值网络，预训练前段必须保持不变。"""

    agent = _small_agent()
    buffer = DPPORolloutBuffer()
    for index, reward in enumerate((1.0, 0.5, -0.25, 2.0)):
        state = np.linspace(-0.5, 0.5, 6, dtype=np.float32) + index * 0.05
        buffer.append(
            _transition_from_sample(
                agent,
                state,
                reward=reward,
                terminated=index == 3,
                seed=30 + index,
            )
        )

    frozen_before = copy.deepcopy(agent.frozen_policy.state_dict())
    trainable_before = copy.deepcopy(agent.trainable_policy.state_dict())
    value_before = copy.deepcopy(agent.value_network.state_dict())

    metrics = agent.update(buffer)

    assert len(buffer) == 0
    assert all(
        torch.equal(value, agent.frozen_policy.state_dict()[name])
        for name, value in frozen_before.items()
    )
    assert any(
        not torch.equal(value, agent.trainable_policy.state_dict()[name])
        for name, value in trainable_before.items()
    )
    assert any(
        not torch.equal(value, agent.value_network.state_dict()[name])
        for name, value in value_before.items()
    )
    assert set(metrics) == {
        "policy_loss",
        "value_loss",
        "approximate_kl",
        "clip_fraction",
        "gradient_norm",
    }
    assert all(math.isfinite(value) for value in metrics.values())
