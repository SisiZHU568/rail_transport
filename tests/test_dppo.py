"""测试 DDPO 在线更新中的轨迹存储、GAE 和裁剪策略优化。"""

import copy
import math

import numpy as np
import pytest
import torch

import src.dppo as dppo_module
from src.dppo import (
    DPPOAgent,
    DPPOConfig,
    DPPORolloutBuffer,
    DPPORolloutTransition,
    approximate_kl_divergence,
    clipped_policy_surrogate,
    compute_gae,
    normalize_advantages,
)
from src.dppo_diffusion import ConditionalDiffusionMLP, CosineNoiseSchedule


def _small_agent(
    *,
    diffusion_steps: int = 4,
    fine_tuned_steps: int = 2,
    batch_size: int = 2,
    update_epochs: int = 2,
    policy_learning_rate: float = 1e-3,
    target_kl: float = 1.0,
    normalize_advantages: bool = True,
) -> DPPOAgent:
    """构造运行速度快、但包含完整双层策略结构的测试智能体。"""

    torch.manual_seed(7)
    policy = ConditionalDiffusionMLP(6, 3, (12, 12))
    schedule = CosineNoiseSchedule(steps=diffusion_steps)
    config = DPPOConfig(
        diffusion_steps=diffusion_steps,
        fine_tuned_steps=fine_tuned_steps,
        value_hidden_dims=(12, 12),
        batch_size=batch_size,
        update_epochs=update_epochs,
        policy_learning_rate=policy_learning_rate,
        value_learning_rate=1e-3,
        seed=101,
        target_kl=target_kl,
        normalize_advantages=normalize_advantages,
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


def _rollout_buffer(agent: DPPOAgent, rewards: tuple[float, ...]) -> DPPORolloutBuffer:
    """按给定奖励构造一条确定性的完整环境轨迹。"""

    buffer = DPPORolloutBuffer()
    for index, reward in enumerate(rewards):
        state = np.linspace(-0.5, 0.5, 6, dtype=np.float32) + index * 0.05
        buffer.append(
            _transition_from_sample(
                agent,
                state,
                reward=reward,
                terminated=index == len(rewards) - 1,
                seed=30 + index,
            )
        )
    return buffer


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


def test_normalize_advantages_uses_full_trajectory_population_statistics() -> None:
    """优势按整条环境轨迹的总体统计量归一化，并统一为 float32。"""

    normalized = normalize_advantages(np.asarray([1.0, 2.0, 5.0, 8.0]))

    assert normalized.dtype == np.float32
    assert normalized.mean() == pytest.approx(0.0, abs=1e-7)
    assert normalized.std(ddof=0) == pytest.approx(1.0, abs=1e-7)


@pytest.mark.parametrize(
    "advantages",
    (
        np.asarray([3.5], dtype=np.float64),
        np.asarray([2.0, 2.0, 2.0], dtype=np.float64),
    ),
)
def test_normalize_advantages_keeps_singleton_and_constant_values(
    advantages: np.ndarray,
) -> None:
    """统计量不足或标准差过小时，返回原值的 float32 副本。"""

    normalized = normalize_advantages(advantages)

    assert normalized.dtype == np.float32
    assert np.array_equal(normalized, advantages.astype(np.float32))
    assert not np.shares_memory(normalized, advantages)


def test_normalize_advantages_handles_large_finite_float32_values() -> None:
    """大幅值优势用高精度统计，不能在求方差时溢出并丢失尺度。"""

    normalized = normalize_advantages(
        np.asarray([3e38, -3e38], dtype=np.float32),
    )
    high_precision = normalized.astype(np.float64)

    assert normalized.dtype == np.float32
    assert np.isfinite(normalized).all()
    assert high_precision.mean() == pytest.approx(0.0, abs=1e-12)
    assert high_precision.std(ddof=0) == pytest.approx(1.0, abs=1e-7)


def test_normalize_advantages_rejects_non_vector_input() -> None:
    """优势必须明确对应一条一维环境轨迹。"""

    with pytest.raises(ValueError, match="一维"):
        normalize_advantages(np.ones((2, 2), dtype=np.float32))


@pytest.mark.parametrize("threshold", (-1.0, math.inf, math.nan, True))
def test_normalize_advantages_rejects_invalid_minimum_standard_deviation(
    threshold: object,
) -> None:
    """归一化阈值只能是非负有限数，布尔值不能冒充数字。"""

    with pytest.raises(ValueError, match="minimum_standard_deviation"):
        normalize_advantages(
            np.asarray([1.0, 2.0], dtype=np.float32),
            minimum_standard_deviation=threshold,
        )


def test_approximate_kl_divergence_matches_exact_stable_formula() -> None:
    """KL 诊断严格采用 (exp(log_ratio)-1)-log_ratio 的批均值。"""

    new_log_probabilities = torch.tensor([[0.1, -0.4], [0.7, -1.2]])
    old_log_probabilities = torch.tensor([[-0.2, -0.1], [0.5, -0.8]])
    log_ratio = new_log_probabilities.double() - old_log_probabilities.double()
    expected = (torch.expm1(log_ratio) - log_ratio).mean()

    actual = approximate_kl_divergence(
        new_log_probabilities,
        old_log_probabilities,
    )

    assert torch.equal(actual, expected)


def test_approximate_kl_divergence_preserves_small_log_ratio_precision() -> None:
    """正负小量必须使用高精度 expm1，避免相消产生负 KL 或丢失信号。"""

    new_log_probabilities = torch.tensor([1e-4, -1e-4], dtype=torch.float32)
    old_log_probabilities = torch.zeros(2, dtype=torch.float32)
    represented_ratios = [float(value) for value in new_log_probabilities]
    expected = sum(
        math.expm1(log_ratio) - log_ratio for log_ratio in represented_ratios
    ) / len(represented_ratios)

    actual = approximate_kl_divergence(
        new_log_probabilities,
        old_log_probabilities,
    )

    assert actual.dtype == torch.float64
    assert actual.item() >= 0.0
    assert actual.item() == pytest.approx(expected, rel=1e-12, abs=1e-18)


def test_approximate_kl_divergence_keeps_large_finite_log_ratio_finite() -> None:
    """float32 会溢出的有限漂移仍应产生可用于早停的有限大 KL。"""

    actual = approximate_kl_divergence(
        torch.tensor([90.0], dtype=torch.float32),
        torch.zeros(1, dtype=torch.float32),
    )

    assert torch.isfinite(actual)
    assert actual.item() > 1.0


def test_approximate_kl_divergence_saturates_float64_overflow() -> None:
    """极端有限漂移饱和到 float64 最大值，以便更新路径正常执行 KL 早停。"""

    actual = approximate_kl_divergence(
        torch.tensor([1000.0], dtype=torch.float64),
        torch.zeros(1, dtype=torch.float64),
    )

    assert torch.isfinite(actual)
    assert actual.item() == torch.finfo(torch.float64).max


def test_approximate_kl_divergence_rejects_empty_tensors() -> None:
    """空批次没有可解释的 KL 均值，必须明确拒绝。"""

    with pytest.raises(ValueError, match="空"):
        approximate_kl_divergence(torch.empty(0), torch.empty(0))


def test_approximate_kl_divergence_rejects_mismatched_shapes() -> None:
    """新旧策略对数概率必须逐元素对应。"""

    with pytest.raises(ValueError, match="形状"):
        approximate_kl_divergence(torch.zeros(2), torch.zeros(2, 1))


@pytest.mark.parametrize("invalid_value", (math.inf, -math.inf, math.nan))
def test_approximate_kl_divergence_rejects_non_finite_content(
    invalid_value: float,
) -> None:
    """非有限对数概率不得进入指数运算污染训练诊断。"""

    with pytest.raises(ValueError, match="有限"):
        approximate_kl_divergence(
            torch.tensor([0.0, invalid_value]),
            torch.zeros(2),
        )


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


def test_config_preserves_original_positional_parameter_order() -> None:
    """新增配置不得改变原有位置参数含义，以免旧训练脚本静默错绑参数。"""

    config = DPPOConfig(0.99, 0.95, 0.20, 0.99, 3e-4, 3e-4, 32)

    assert config.batch_size == 32
    assert config.training_sampling_min_std == pytest.approx(0.01)


def test_config_appends_ppo_stability_options_with_validated_defaults() -> None:
    """稳定性开关追加在已有字段末尾，避免破坏旧位置参数调用。"""

    config = DPPOConfig()

    assert config.target_kl == pytest.approx(1.0)
    assert config.normalize_advantages is True
    assert tuple(DPPOConfig.__dataclass_fields__)[-2:] == (
        "target_kl",
        "normalize_advantages",
    )


@pytest.mark.parametrize("target_kl", (0.0, -0.1, math.inf, math.nan, True))
def test_config_rejects_invalid_target_kl(target_kl: object) -> None:
    """KL 早停阈值必须是严格正的有限数。"""

    with pytest.raises(ValueError, match="target_kl"):
        DPPOConfig(target_kl=target_kl)


@pytest.mark.parametrize("normalize", (0, 1, np.bool_(True), "yes", None))
def test_config_requires_real_bool_for_advantage_normalization(
    normalize: object,
) -> None:
    """归一化开关必须是真正的 bool，不能接受 truthy/falsy 替代品。"""

    with pytest.raises(ValueError, match="normalize_advantages"):
        DPPOConfig(normalize_advantages=normalize)


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


def test_update_normalizes_full_gae_before_applying_denoising_discount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """优势只在环境层归一化一次，随后才展开并乘每个去噪步骤的折扣。"""

    agent = _small_agent(batch_size=4, update_epochs=1)
    buffer = _rollout_buffer(agent, (1.0, -0.5, 3.0, 0.25))
    transitions = tuple(buffer.transitions)
    raw_advantages, _ = compute_gae(
        rewards=np.asarray([item.reward for item in transitions]),
        values=np.asarray([item.value for item in transitions]),
        terminated=np.asarray([item.terminated for item in transitions]),
        next_value=0.0,
        gamma=agent.config.gamma,
        gae_lambda=agent.config.gae_lambda,
    )
    permutation = torch.randperm(
        len(transitions),
        generator=torch.Generator(device="cpu").manual_seed(agent.config.seed),
    ).numpy()
    denoising_weights = agent.config.denoising_discount ** np.arange(
        agent.trainable_denoising_steps - 1,
        -1,
        -1,
    )
    expected = (
        normalize_advantages(raw_advantages)[permutation, None]
        * denoising_weights[None, :]
    )
    captured_advantages: list[np.ndarray] = []
    original_surrogate = dppo_module.clipped_policy_surrogate

    def capture_surrogate(
        new_log_probabilities: torch.Tensor,
        old_log_probabilities: torch.Tensor,
        advantages: torch.Tensor,
        *,
        clip_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        captured_advantages.append(advantages.detach().cpu().numpy().copy())
        return original_surrogate(
            new_log_probabilities,
            old_log_probabilities,
            advantages,
            clip_ratio=clip_ratio,
        )

    monkeypatch.setattr(dppo_module, "clipped_policy_surrogate", capture_surrogate)

    agent.update(buffer)

    assert len(captured_advantages) == 1
    assert np.allclose(captured_advantages[0], expected, atol=1e-6)


def test_update_can_keep_raw_gae_before_denoising_discount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """关闭归一化时，裁剪目标必须保留原始 GAE，仅施加去噪折扣。"""

    agent = _small_agent(
        batch_size=4,
        update_epochs=1,
        normalize_advantages=False,
    )
    buffer = _rollout_buffer(agent, (1.0, -0.5, 3.0, 0.25))
    transitions = tuple(buffer.transitions)
    raw_advantages, _ = compute_gae(
        rewards=np.asarray([item.reward for item in transitions]),
        values=np.asarray([item.value for item in transitions]),
        terminated=np.asarray([item.terminated for item in transitions]),
        next_value=0.0,
        gamma=agent.config.gamma,
        gae_lambda=agent.config.gae_lambda,
    )
    permutation = torch.randperm(
        len(transitions),
        generator=torch.Generator(device="cpu").manual_seed(agent.config.seed),
    ).numpy()
    denoising_weights = agent.config.denoising_discount ** np.arange(
        agent.trainable_denoising_steps - 1,
        -1,
        -1,
    )
    expected = raw_advantages[permutation, None] * denoising_weights[None, :]
    captured_advantages: list[np.ndarray] = []
    original_surrogate = dppo_module.clipped_policy_surrogate

    def capture_surrogate(
        new_log_probabilities: torch.Tensor,
        old_log_probabilities: torch.Tensor,
        advantages: torch.Tensor,
        *,
        clip_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        captured_advantages.append(advantages.detach().cpu().numpy().copy())
        return original_surrogate(
            new_log_probabilities,
            old_log_probabilities,
            advantages,
            clip_ratio=clip_ratio,
        )

    monkeypatch.setattr(dppo_module, "clipped_policy_surrogate", capture_surrogate)

    agent.update(buffer)

    assert len(captured_advantages) == 1
    assert np.allclose(captured_advantages[0], expected, atol=1e-6)


def test_update_stops_future_batches_after_kl_reaches_tiny_target() -> None:
    """至少一次优化后触及极小 KL 阈值时，应停止当前余下的全部更新。"""

    agent = _small_agent(
        batch_size=4,
        update_epochs=6,
        policy_learning_rate=1e-2,
        target_kl=1e-12,
    )
    buffer = _rollout_buffer(agent, (1.0, -0.5, 3.0, 0.25))
    expected_batches = agent.config.update_epochs

    metrics = agent.update(buffer)

    assert 1.0 <= metrics["optimizer_step_count"] < expected_batches
    assert metrics["kl_early_stopped"] == 1.0
    assert metrics["maximum_approximate_kl"] >= agent.config.target_kl
    assert len(buffer) == 0
    assert all(math.isfinite(value) for value in metrics.values())


def test_update_stops_before_surrogate_when_log_ratio_is_extreme(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """极大 KL 必须在 PPO 的 exp 概率比之前早停，且首批不产生优化指标。"""

    agent = _small_agent(batch_size=1, update_epochs=2, target_kl=1.0)
    buffer = _rollout_buffer(agent, (1.0,))
    old_log_probabilities = torch.as_tensor(
        np.array(
            buffer.transitions[0].old_log_probabilities[
                agent.frozen_denoising_steps :
            ],
            copy=True,
        ),
        dtype=torch.float32,
    ).unsqueeze(0)

    def extreme_current_log_probabilities(
        states: torch.Tensor,
        denoising_actions: torch.Tensor,
    ) -> torch.Tensor:
        del states, denoising_actions
        return old_log_probabilities + 90.0

    def unexpected_call(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("KL 早停后不得计算 surrogate 或执行 optimizer.step")

    monkeypatch.setattr(
        agent,
        "_current_trainable_log_probabilities",
        extreme_current_log_probabilities,
    )
    monkeypatch.setattr(dppo_module, "clipped_policy_surrogate", unexpected_call)
    monkeypatch.setattr(agent.policy_optimizer, "step", unexpected_call)
    monkeypatch.setattr(agent.value_optimizer, "step", unexpected_call)

    metrics = agent.update(buffer)

    assert metrics["kl_early_stopped"] == 1.0
    assert metrics["optimizer_step_count"] == 0.0
    assert metrics["policy_loss"] == 0.0
    assert metrics["value_loss"] == 0.0
    assert metrics["gradient_norm"] == 0.0
    assert metrics["approximate_kl"] == metrics["maximum_approximate_kl"]
    assert math.isfinite(metrics["approximate_kl"])
    assert metrics["approximate_kl"] > agent.config.target_kl
    assert metrics["clip_fraction"] == 1.0
    assert len(buffer) == 0


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
        "maximum_approximate_kl",
        "clip_fraction",
        "gradient_norm",
        "optimizer_step_count",
        "kl_early_stopped",
    }
    assert metrics["optimizer_step_count"] == 4.0
    assert metrics["kl_early_stopped"] == 0.0
    assert metrics["maximum_approximate_kl"] >= metrics["approximate_kl"]
    assert all(math.isfinite(value) for value in metrics.values())
