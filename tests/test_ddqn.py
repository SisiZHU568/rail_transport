"""
test_ddqn.py

测试Double DQN核心组件。
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.ddqn import (
    DDQNAgent,
    DDQNConfig,
    QNetwork,
    ReplayBuffer,
    linear_epsilon,
)


def build_test_config() -> DDQNConfig:
    return DDQNConfig(
        gamma=0.95,
        learning_rate=0.001,
        hidden_dims=(32, 32),
        replay_capacity=100,
        replay_start_size=8,
        batch_size=8,
        target_update_interval=5,
        gradient_clip_norm=5.0,
        random_seed=42,
    )


def fill_buffer(
    buffer: ReplayBuffer,
    count: int,
) -> None:
    for index in range(count):
        state = np.full(
            buffer.state_dim,
            index / 100.0,
            dtype=np.float32,
        )

        next_state = np.full(
            buffer.state_dim,
            (index + 1) / 100.0,
            dtype=np.float32,
        )

        buffer.add(
            state=state,
            action=index % 3,
            reward=-0.1,
            next_state=next_state,
            done=(index % 10 == 0),
        )


def test_q_network_output_shape() -> None:
    """
    15维状态、4个动作应输出[B,4]。
    """

    network = QNetwork(
        state_dim=15,
        action_count=3,
        hidden_dims=(32, 32),
    )

    states = torch.zeros(
        (5, 15),
        dtype=torch.float32,
    )

    output = network(states)

    assert output.shape == (5, 3)


def test_replay_buffer_sample_shape() -> None:
    """
    检查经验回放采样形状。
    """

    buffer = ReplayBuffer(
        capacity=100,
        state_dim=15,
    )

    fill_buffer(
        buffer,
        count=20,
    )

    batch = buffer.sample(
        batch_size=8,
        rng=np.random.default_rng(42),
    )

    assert batch.states.shape == (8, 15)
    assert batch.actions.shape == (8,)
    assert batch.rewards.shape == (8,)
    assert batch.next_states.shape == (8, 15)
    assert batch.dones.shape == (8,)


def test_linear_epsilon_boundaries() -> None:
    """
    epsilon从1.0衰减到0.05。
    """

    assert linear_epsilon(
        episode_index=0,
        epsilon_start=1.0,
        epsilon_end=0.05,
        decay_episodes=100,
    ) == pytest.approx(1.0)

    assert linear_epsilon(
        episode_index=100,
        epsilon_start=1.0,
        epsilon_end=0.05,
        decay_episodes=100,
    ) == pytest.approx(0.05)

    assert linear_epsilon(
        episode_index=200,
        epsilon_start=1.0,
        epsilon_end=0.05,
        decay_episodes=100,
    ) == pytest.approx(0.05)


def test_selected_action_is_valid() -> None:
    """
    动作必须位于0～3。
    """

    agent = DDQNAgent(
        state_dim=15,
        action_count=3,
        config=build_test_config(),
        device="cpu",
    )

    state = np.zeros(
        15,
        dtype=np.float32,
    )

    for _ in range(20):
        action = agent.select_action(
            state=state,
            epsilon=1.0,
        )

        assert 0 <= action < 3


def test_learning_waits_for_enough_experience() -> None:
    """
    经验不足时不执行网络更新。
    """

    agent = DDQNAgent(
        state_dim=15,
        action_count=3,
        config=build_test_config(),
        device="cpu",
    )

    buffer = ReplayBuffer(
        capacity=100,
        state_dim=15,
    )

    fill_buffer(
        buffer,
        count=7,
    )

    assert agent.learn(buffer) is None


def test_learning_returns_finite_loss() -> None:
    """
    经验充足时应返回有限损失值。
    """

    agent = DDQNAgent(
        state_dim=15,
        action_count=3,
        config=build_test_config(),
        device="cpu",
    )

    buffer = ReplayBuffer(
        capacity=100,
        state_dim=15,
    )

    fill_buffer(
        buffer,
        count=20,
    )

    loss = agent.learn(buffer)

    assert loss is not None
    assert np.isfinite(loss)
    assert loss >= 0.0


def test_model_save_and_load(
    tmp_path: Path,
) -> None:
    """
    检查模型保存和加载。
    """

    config = build_test_config()

    first_agent = DDQNAgent(
        state_dim=15,
        action_count=3,
        config=config,
        device="cpu",
    )

    model_path = (
        tmp_path / "ddqn_test.pt"
    )

    first_agent.save(model_path)

    second_agent = DDQNAgent(
        state_dim=15,
        action_count=3,
        config=config,
        device="cpu",
    )

    second_agent.load(model_path)

    first_parameters = list(
        first_agent
        .online_network
        .parameters()
    )

    second_parameters = list(
        second_agent
        .online_network
        .parameters()
    )

    for first, second in zip(
        first_parameters,
        second_parameters,
    ):
        assert torch.allclose(
            first,
            second,
        )