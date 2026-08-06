"""
ddqn.py

Double DQN核心实现。

包含：

1. 经验回放池；
2. Q网络；
3. epsilon-greedy动作选择；
4. Double DQN目标值计算；
5. 目标网络同步；
6. 模型保存和加载。
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DDQNConfig:
    """
    Double DQN超参数。
    """

    gamma: float
    learning_rate: float

    hidden_dims: tuple[int, ...]

    replay_capacity: int
    replay_start_size: int
    batch_size: int

    target_update_interval: int
    gradient_clip_norm: float

    random_seed: int

    def __post_init__(self) -> None:
        if not 0 <= self.gamma <= 1:
            raise ValueError(
                "gamma必须位于[0,1]。"
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate必须大于0。"
            )

        if not self.hidden_dims:
            raise ValueError(
                "至少需要一个隐藏层。"
            )

        if any(
            hidden_dim <= 0
            for hidden_dim in self.hidden_dims
        ):
            raise ValueError(
                "隐藏层维度必须大于0。"
            )

        if self.replay_capacity <= 0:
            raise ValueError(
                "经验回放池容量必须大于0。"
            )

        if self.replay_start_size <= 0:
            raise ValueError(
                "replay_start_size必须大于0。"
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size必须大于0。"
            )

        if self.replay_start_size > self.replay_capacity:
            raise ValueError(
                "replay_start_size不能超过回放池容量。"
            )

        if self.target_update_interval <= 0:
            raise ValueError(
                "目标网络更新间隔必须大于0。"
            )

        if self.gradient_clip_norm <= 0:
            raise ValueError(
                "梯度裁剪阈值必须大于0。"
            )


@dataclass(frozen=True)
class ReplayBatch:
    """
    从经验回放池中采样的一批数据。
    """

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray


class ReplayBuffer:
    """
    固定容量循环经验回放池。
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
    ) -> None:
        if capacity <= 0:
            raise ValueError(
                "capacity必须大于0。"
            )

        if state_dim <= 0:
            raise ValueError(
                "state_dim必须大于0。"
            )

        self.capacity = capacity
        self.state_dim = state_dim

        self.states = np.zeros(
            (capacity, state_dim),
            dtype=np.float32,
        )

        self.actions = np.zeros(
            capacity,
            dtype=np.int64,
        )

        self.rewards = np.zeros(
            capacity,
            dtype=np.float32,
        )

        self.next_states = np.zeros(
            (capacity, state_dim),
            dtype=np.float32,
        )

        self.dones = np.zeros(
            capacity,
            dtype=np.float32,
        )

        self._position = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        添加一条状态转移。
        """

        state_array = np.asarray(
            state,
            dtype=np.float32,
        )

        next_state_array = np.asarray(
            next_state,
            dtype=np.float32,
        )

        expected_shape = (
            self.state_dim,
        )

        if state_array.shape != expected_shape:
            raise ValueError(
                "state维度错误："
                f"期望{expected_shape}，"
                f"实际{state_array.shape}。"
            )

        if next_state_array.shape != expected_shape:
            raise ValueError(
                "next_state维度错误。"
            )

        self.states[
            self._position
        ] = state_array

        self.actions[
            self._position
        ] = int(action)

        self.rewards[
            self._position
        ] = float(reward)

        self.next_states[
            self._position
        ] = next_state_array

        self.dones[
            self._position
        ] = float(done)

        self._position = (
            self._position + 1
        ) % self.capacity

        self._size = min(
            self._size + 1,
            self.capacity,
        )

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
    ) -> ReplayBatch:
        """
        无放回随机采样。
        """

        if batch_size <= 0:
            raise ValueError(
                "batch_size必须大于0。"
            )

        if self._size < batch_size:
            raise ValueError(
                "经验数量不足，无法采样。"
            )

        indices = rng.choice(
            self._size,
            size=batch_size,
            replace=False,
        )

        return ReplayBatch(
            states=self.states[
                indices
            ].copy(),
            actions=self.actions[
                indices
            ].copy(),
            rewards=self.rewards[
                indices
            ].copy(),
            next_states=self.next_states[
                indices
            ].copy(),
            dones=self.dones[
                indices
            ].copy(),
        )


class QNetwork(nn.Module):
    """
    输入状态，输出每个离散动作的Q值。
    """

    def __init__(
        self,
        state_dim: int,
        action_count: int,
        hidden_dims: tuple[int, ...],
    ) -> None:
        super().__init__()

        if state_dim <= 0:
            raise ValueError(
                "state_dim必须大于0。"
            )

        if action_count <= 0:
            raise ValueError(
                "action_count必须大于0。"
            )

        layers: list[nn.Module] = []

        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(
                        input_dim,
                        hidden_dim,
                    ),
                    nn.ReLU(),
                ]
            )

            input_dim = hidden_dim

        layers.append(
            nn.Linear(
                input_dim,
                action_count,
            )
        )

        self.network = nn.Sequential(
            *layers
        )

    def forward(
        self,
        states: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(states)


class DDQNAgent:
    """
    Double DQN智能体。
    """

    def __init__(
        self,
        state_dim: int,
        action_count: int,
        config: DDQNConfig,
        device: str | torch.device | None = None,
    ) -> None:
        self.state_dim = state_dim
        self.action_count = action_count
        self.config = config

        if device is None:
            selected_device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            selected_device = device

        self.device = torch.device(
            selected_device
        )

        torch.manual_seed(
            config.random_seed
        )

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                config.random_seed
            )

        self.rng = np.random.default_rng(
            config.random_seed
        )

        self.online_network = QNetwork(
            state_dim=state_dim,
            action_count=action_count,
            hidden_dims=config.hidden_dims,
        ).to(self.device)

        self.target_network = QNetwork(
            state_dim=state_dim,
            action_count=action_count,
            hidden_dims=config.hidden_dims,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=config.learning_rate,
        )

        self.training_steps = 0

        self.sync_target_network()

    def sync_target_network(self) -> None:
        """
        将在线网络参数复制到目标网络。
        """

        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )

        self.target_network.eval()

    def select_action(
        self,
        state: np.ndarray,
        epsilon: float,
    ) -> int:
        """
        epsilon-greedy动作选择。
        """

        if not 0 <= epsilon <= 1:
            raise ValueError(
                "epsilon必须位于[0,1]。"
            )

        if self.rng.random() < epsilon:
            return int(
                self.rng.integers(
                    low=0,
                    high=self.action_count,
                )
            )

        state_array = np.asarray(
            state,
            dtype=np.float32,
        )

        if state_array.shape != (
            self.state_dim,
        ):
            raise ValueError(
                "动作选择时状态维度错误。"
            )

        state_tensor = torch.from_numpy(
            state_array
        ).unsqueeze(0).to(
            self.device
        )

        self.online_network.eval()

        with torch.no_grad():
            q_values = self.online_network(
                state_tensor
            )

            action = int(
                torch.argmax(
                    q_values,
                    dim=1,
                ).item()
            )

        self.online_network.train()

        return action

    def learn(
        self,
        replay_buffer: ReplayBuffer,
    ) -> float | None:
        """
        从经验回放池采样并更新一次网络。

        经验不足时返回None。
        """

        required_size = max(
            self.config.batch_size,
            self.config.replay_start_size,
        )

        if len(replay_buffer) < required_size:
            return None

        batch = replay_buffer.sample(
            batch_size=self.config.batch_size,
            rng=self.rng,
        )

        states = torch.from_numpy(
            batch.states
        ).to(self.device)

        actions = torch.from_numpy(
            batch.actions
        ).long().unsqueeze(1).to(
            self.device
        )

        rewards = torch.from_numpy(
            batch.rewards
        ).unsqueeze(1).to(
            self.device
        )

        next_states = torch.from_numpy(
            batch.next_states
        ).to(self.device)

        dones = torch.from_numpy(
            batch.dones
        ).unsqueeze(1).to(
            self.device
        )

        current_q_values = (
            self.online_network(states)
            .gather(
                dim=1,
                index=actions,
            )
        )

        # Double DQN：
        #
        # 1. 在线网络选择下一动作；
        # 2. 目标网络评价该动作。
        with torch.no_grad():
            next_actions = (
                self.online_network(
                    next_states
                )
                .argmax(
                    dim=1,
                    keepdim=True,
                )
            )

            next_q_values = (
                self.target_network(
                    next_states
                )
                .gather(
                    dim=1,
                    index=next_actions,
                )
            )

            target_q_values = (
                rewards
                +
                self.config.gamma
                * (1.0 - dones)
                * next_q_values
            )

        loss = F.smooth_l1_loss(
            current_q_values,
            target_q_values,
        )

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.online_network.parameters(),
            self.config.gradient_clip_norm,
        )

        self.optimizer.step()

        self.training_steps += 1

        if (
            self.training_steps
            % self.config.target_update_interval
            == 0
        ):
            self.sync_target_network()

        return float(
            loss.item()
        )

    def save(
        self,
        path: str | Path,
    ) -> None:
        """
        保存模型与优化器状态。
        """

        output_path = Path(path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint = {
            "state_dim": self.state_dim,
            "action_count": self.action_count,
            "config": asdict(
                self.config
            ),
            "online_network": (
                self.online_network.state_dict()
            ),
            "target_network": (
                self.target_network.state_dict()
            ),
            "optimizer": (
                self.optimizer.state_dict()
            ),
            "training_steps": (
                self.training_steps
            ),
        }

        torch.save(
            checkpoint,
            output_path,
        )

    def load(
        self,
        path: str | Path,
    ) -> None:
        """
        加载模型。
        """

        checkpoint = torch.load(
            Path(path),
            map_location=self.device,
        )

        if (
            int(checkpoint["state_dim"])
            != self.state_dim
        ):
            raise ValueError(
                "模型状态维度与环境不一致。"
            )

        if (
            int(checkpoint["action_count"])
            != self.action_count
        ):
            raise ValueError(
                "模型动作数量与环境不一致。"
            )

        self.online_network.load_state_dict(
            checkpoint["online_network"]
        )

        self.target_network.load_state_dict(
            checkpoint["target_network"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        self.training_steps = int(
            checkpoint["training_steps"]
        )


def linear_epsilon(
    episode_index: int,
    epsilon_start: float,
    epsilon_end: float,
    decay_episodes: int,
) -> float:
    """
    线性epsilon衰减。

    episode_index从0开始。
    """

    if episode_index < 0:
        raise ValueError(
            "episode_index不能小于0。"
        )

    if not 0 <= epsilon_end <= epsilon_start <= 1:
        raise ValueError(
            "epsilon参数不合法。"
        )

    if decay_episodes <= 0:
        raise ValueError(
            "decay_episodes必须大于0。"
        )

    progress = min(
        episode_index / decay_episodes,
        1.0,
    )

    return float(
        epsilon_start
        +
        progress
        * (
            epsilon_end
            - epsilon_start
        )
    )


def build_ddqn_config(
    config: dict[str, Any],
) -> DDQNConfig:
    """
    根据debug.yaml创建DDQN配置。
    """

    ddqn_config = config["ddqn"]

    return DDQNConfig(
        gamma=float(
            ddqn_config["gamma"]
        ),
        learning_rate=float(
            ddqn_config["learning_rate"]
        ),
        hidden_dims=tuple(
            int(value)
            for value
            in ddqn_config["hidden_dims"]
        ),
        replay_capacity=int(
            ddqn_config["replay_capacity"]
        ),
        replay_start_size=int(
            ddqn_config[
                "replay_start_size"
            ]
        ),
        batch_size=int(
            ddqn_config["batch_size"]
        ),
        target_update_interval=int(
            ddqn_config[
                "target_update_interval"
            ]
        ),
        gradient_clip_norm=float(
            ddqn_config[
                "gradient_clip_norm"
            ]
        ),
        random_seed=int(
            ddqn_config["random_seed"]
        ),
    )