"""测试带合法动作掩码和模型版本信息的 Double DQN 核心组件。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import src.ddqn as ddqn
from src.ddqn import (
    DDQNAgent,
    DDQNConfig,
    QNetwork,
    ReplayBuffer,
    linear_epsilon,
)


STATE_DIM = 78
ACTION_COUNT = 12


def build_test_config(
    *,
    replay_start_size: int = 8,
    batch_size: int = 8,
    gamma: float = 0.95,
) -> DDQNConfig:
    """创建规模较小、运行较快的单元测试配置。"""

    return DDQNConfig(
        gamma=gamma,
        learning_rate=0.001,
        hidden_dims=(32, 32),
        replay_capacity=100,
        replay_start_size=replay_start_size,
        batch_size=batch_size,
        target_update_interval=5,
        gradient_clip_norm=5.0,
        random_seed=42,
    )


def build_metadata(
    *,
    state_schema_version: str = "ddqn-v2-78",
    action_schema_version: str = "structured-slow-v1",
    mec_count: int = 5,
    function_count: int = 3,
) -> ddqn.DDQNCheckpointMetadata:
    """创建与当前五个 MEC、三个 VNF 场景一致的模型元数据。"""

    return ddqn.DDQNCheckpointMetadata(
        state_schema_version=state_schema_version,
        action_schema_version=action_schema_version,
        mec_count=mec_count,
        function_count=function_count,
    )


def build_agent(
    *,
    config: DDQNConfig | None = None,
    metadata: ddqn.DDQNCheckpointMetadata | None = None,
) -> DDQNAgent:
    """统一创建当前 78 维状态、12 个动作的测试智能体。"""

    return DDQNAgent(
        state_dim=STATE_DIM,
        action_count=ACTION_COUNT,
        config=config or build_test_config(),
        metadata=metadata or build_metadata(),
        device="cpu",
    )


def fill_buffer(
    buffer: ReplayBuffer,
    count: int,
) -> None:
    """填充合法的测试经验；终止状态也保存全 True 掩码。"""

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
            action=index % buffer.action_count,
            reward=-0.1,
            next_state=next_state,
            done=(index % 10 == 0),
            next_action_mask=np.ones(
                buffer.action_count,
                dtype=np.bool_,
            ),
        )


def test_q_network_output_shape() -> None:
    """78 维状态经过网络后，应输出 12 个动作的 Q 值。"""

    network = QNetwork(
        state_dim=STATE_DIM,
        action_count=ACTION_COUNT,
        hidden_dims=(32, 32),
    )
    states = torch.zeros(
        (5, STATE_DIM),
        dtype=torch.float32,
    )

    output = network(states)

    assert output.shape == (5, ACTION_COUNT)


def test_replay_batch_contains_next_action_masks() -> None:
    """每条经验必须携带下一状态的 12 维合法动作掩码。"""

    buffer = ReplayBuffer(
        capacity=10,
        state_dim=STATE_DIM,
        action_count=ACTION_COUNT,
    )
    next_action_mask = np.zeros(
        ACTION_COUNT,
        dtype=np.bool_,
    )
    next_action_mask[[2, 9]] = True
    buffer.add(
        state=np.zeros(STATE_DIM, dtype=np.float32),
        action=2,
        reward=-0.1,
        next_state=np.zeros(STATE_DIM, dtype=np.float32),
        done=False,
        next_action_mask=next_action_mask,
    )

    batch = buffer.sample(1, np.random.default_rng(1))

    assert batch.states.shape == (1, STATE_DIM)
    assert batch.actions.shape == (1,)
    assert batch.rewards.shape == (1,)
    assert batch.next_states.shape == (1, STATE_DIM)
    assert batch.dones.shape == (1,)
    assert batch.next_action_masks.shape == (1, ACTION_COUNT)
    assert batch.next_action_masks.dtype == np.bool_
    assert np.array_equal(batch.next_action_masks[0], next_action_mask)


def test_replay_buffer_rejects_non_boolean_action_mask() -> None:
    """用整数数组冒充掩码时应直接报错，避免把 0/1 含义弄反。"""

    buffer = ReplayBuffer(
        capacity=10,
        state_dim=STATE_DIM,
        action_count=ACTION_COUNT,
    )

    with pytest.raises(ValueError, match="布尔"):
        buffer.add(
            state=np.zeros(STATE_DIM, dtype=np.float32),
            action=2,
            reward=-0.1,
            next_state=np.zeros(STATE_DIM, dtype=np.float32),
            done=False,
            next_action_mask=np.ones(ACTION_COUNT, dtype=np.int64),
        )


def test_linear_epsilon_boundaries() -> None:
    """epsilon 应从 1.0 线性衰减到 0.05，并保持下界。"""

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


def test_exploration_never_selects_masked_action() -> None:
    """随机探索也只能从当前合法动作集合中抽样。"""

    agent = build_agent()
    action_mask = np.zeros(ACTION_COUNT, dtype=np.bool_)
    action_mask[[2, 9]] = True

    actions = {
        agent.select_action(
            np.zeros(STATE_DIM, dtype=np.float32),
            epsilon=1.0,
            action_mask=action_mask,
        )
        for _ in range(100)
    }

    assert actions <= {2, 9}
    assert actions == {2, 9}


def test_greedy_selection_never_selects_masked_action() -> None:
    """即使非法动作 Q 值最高，贪心选择也必须忽略它。"""

    agent = build_agent()
    action_mask = np.zeros(ACTION_COUNT, dtype=np.bool_)
    action_mask[[2, 9]] = True

    with torch.no_grad():
        for parameter in agent.online_network.parameters():
            parameter.zero_()
        output_layer = agent.online_network.network[-1]
        assert isinstance(output_layer, torch.nn.Linear)
        output_layer.bias[0] = 100.0
        output_layer.bias[9] = 5.0

    action = agent.select_action(
        np.zeros(STATE_DIM, dtype=np.float32),
        epsilon=0.0,
        action_mask=action_mask,
    )

    assert action == 9


def test_action_selection_rejects_empty_mask() -> None:
    """没有任何合法动作时应明确报错，不能偷偷返回动作 0。"""

    agent = build_agent()

    with pytest.raises(ValueError, match="没有合法"):
        agent.select_action(
            np.zeros(STATE_DIM, dtype=np.float32),
            epsilon=0.0,
            action_mask=np.zeros(ACTION_COUNT, dtype=np.bool_),
        )


def test_learning_waits_for_enough_experience() -> None:
    """经验不足时不执行网络更新。"""

    agent = build_agent()
    buffer = ReplayBuffer(
        capacity=100,
        state_dim=STATE_DIM,
        action_count=ACTION_COUNT,
    )
    fill_buffer(buffer, count=7)

    assert agent.learn(buffer) is None


def test_masked_double_dqn_target_ignores_invalid_high_q_action() -> None:
    """计算目标值时，在线网络不能选择被掩码排除的高 Q 动作。"""

    config = build_test_config(
        replay_start_size=1,
        batch_size=1,
        gamma=1.0,
    )
    agent = build_agent(config=config)
    buffer = ReplayBuffer(
        capacity=10,
        state_dim=STATE_DIM,
        action_count=ACTION_COUNT,
    )
    next_action_mask = np.zeros(ACTION_COUNT, dtype=np.bool_)
    next_action_mask[2] = True
    buffer.add(
        state=np.zeros(STATE_DIM, dtype=np.float32),
        action=1,
        reward=0.0,
        next_state=np.zeros(STATE_DIM, dtype=np.float32),
        done=False,
        next_action_mask=next_action_mask,
    )

    # 将两张网络固定为可手算的常数输出：
    # 非法动作 0 的在线 Q 值最高，但合法动作只有 2。
    with torch.no_grad():
        for parameter in agent.online_network.parameters():
            parameter.zero_()
        for parameter in agent.target_network.parameters():
            parameter.zero_()
        online_output = agent.online_network.network[-1]
        target_output = agent.target_network.network[-1]
        assert isinstance(online_output, torch.nn.Linear)
        assert isinstance(target_output, torch.nn.Linear)
        online_output.bias[0] = 10.0
        online_output.bias[2] = 2.0
        target_output.bias[0] = 100.0
        target_output.bias[2] = 3.0

    loss = agent.learn(buffer)

    # 当前 Q=0、合法目标 Q=3，因此 SmoothL1Loss 为 3-0.5=2.5。
    assert loss == pytest.approx(2.5)


def test_learning_returns_finite_loss() -> None:
    """经验充足时应返回非负的有限损失值。"""

    agent = build_agent()
    buffer = ReplayBuffer(
        capacity=100,
        state_dim=STATE_DIM,
        action_count=ACTION_COUNT,
    )
    fill_buffer(buffer, count=20)

    loss = agent.learn(buffer)

    assert loss is not None
    assert np.isfinite(loss)
    assert loss >= 0.0


def test_agent_rejects_obsolete_state_dimension() -> None:
    """新智能体不能再用旧版 15 维状态创建。"""

    with pytest.raises(ValueError, match="78"):
        DDQNAgent(
            state_dim=15,
            action_count=ACTION_COUNT,
            config=build_test_config(),
            metadata=build_metadata(),
            device="cpu",
        )


def test_agent_rejects_wrong_scenario_metadata() -> None:
    """元数据必须明确表示当前五 MEC、三 VNF 场景。"""

    with pytest.raises(ValueError, match="MEC"):
        build_agent(metadata=build_metadata(mec_count=4))


def test_model_save_and_load(tmp_path: Path) -> None:
    """相同状态、动作版本的模型应能保存并恢复。"""

    config = build_test_config()
    first_agent = build_agent(config=config)
    model_path = tmp_path / "ddqn_test.pt"
    first_agent.save(model_path)

    second_agent = build_agent(config=config)
    second_agent.load(model_path)

    for first, second in zip(
        first_agent.online_network.parameters(),
        second_agent.online_network.parameters(),
    ):
        assert torch.allclose(first, second)


def test_old_schema_checkpoint_is_rejected(tmp_path: Path) -> None:
    """不带版本元数据的旧模型必须被拒绝，避免静默误加载。"""

    agent = build_agent()
    model_path = tmp_path / "old.pt"
    torch.save(
        {
            "state_dim": STATE_DIM,
            "action_count": ACTION_COUNT,
        },
        model_path,
    )

    with pytest.raises(ValueError, match="状态模式版本"):
        agent.load(model_path)


def test_checkpoint_with_other_action_schema_is_rejected(tmp_path: Path) -> None:
    """动作模式版本不同的模型不能加载到当前智能体。"""

    source_agent = build_agent()
    model_path = tmp_path / "wrong_action_schema.pt"
    source_agent.save(model_path)
    checkpoint = torch.load(model_path, weights_only=True)
    checkpoint["metadata"]["action_schema_version"] = "legacy-action-v0"
    torch.save(checkpoint, model_path)

    with pytest.raises(ValueError, match="动作模式版本"):
        build_agent().load(model_path)
