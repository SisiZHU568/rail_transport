"""Double DQN 核心实现。

本模块负责：经验回放、Q 网络、带合法动作掩码的 epsilon-greedy、
带合法动作掩码的 Double-DQN 目标值，以及带模式版本校验的模型存取。
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# 当前研究方案已经固定为 78 维状态和 12 个结构化慢层动作。
# 将这些值集中定义，能防止训练脚本误用旧版 15 维状态或 4 动作模型。
CURRENT_STATE_DIM = 78
CURRENT_ACTION_COUNT = 12
CURRENT_STATE_SCHEMA_VERSION = "ddqn-v2-78"
CURRENT_ACTION_SCHEMA_VERSION = "structured-slow-v1"
CURRENT_MEC_COUNT = 5
CURRENT_FUNCTION_COUNT = 3


@dataclass(frozen=True)
class DDQNConfig:
    """Double DQN 超参数。"""

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
            raise ValueError("gamma 必须位于 [0, 1]。")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate 必须大于 0。")
        if not self.hidden_dims:
            raise ValueError("至少需要一个隐藏层。")
        if any(hidden_dim <= 0 for hidden_dim in self.hidden_dims):
            raise ValueError("隐藏层维度必须大于 0。")
        if self.replay_capacity <= 0:
            raise ValueError("经验回放池容量必须大于 0。")
        if self.replay_start_size <= 0:
            raise ValueError("replay_start_size 必须大于 0。")
        if self.batch_size <= 0:
            raise ValueError("batch_size 必须大于 0。")
        if self.replay_start_size > self.replay_capacity:
            raise ValueError("replay_start_size 不能超过回放池容量。")
        if self.batch_size > self.replay_capacity:
            raise ValueError("batch_size 不能超过回放池容量。")
        if self.target_update_interval <= 0:
            raise ValueError("目标网络更新间隔必须大于 0。")
        if self.gradient_clip_norm <= 0:
            raise ValueError("梯度裁剪阈值必须大于 0。")


@dataclass(frozen=True)
class DDQNCheckpointMetadata:
    """标识模型所使用的状态、动作和实验场景版本。

    神经网络参数本身看不出各维特征的实际含义，因此保存这些元数据，
    才能阻止“张量形状碰巧相同、语义却不同”的旧模型被误加载。
    """

    state_schema_version: str
    action_schema_version: str
    mec_count: int
    function_count: int

    def __post_init__(self) -> None:
        if not self.state_schema_version:
            raise ValueError("状态模式版本不能为空。")
        if not self.action_schema_version:
            raise ValueError("动作模式版本不能为空。")
        if self.mec_count <= 0:
            raise ValueError("MEC 数量必须大于 0。")
        if self.function_count <= 0:
            raise ValueError("VNF 数量必须大于 0。")


@dataclass(frozen=True)
class ReplayBatch:
    """从经验回放池中采样的一批状态转移。"""

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    next_action_masks: np.ndarray


class ReplayBuffer:
    """固定容量的循环经验回放池。"""

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_count: int,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity 必须大于 0。")
        if state_dim <= 0:
            raise ValueError("state_dim 必须大于 0。")
        if action_count <= 0:
            raise ValueError("action_count 必须大于 0。")

        self.capacity = capacity
        self.state_dim = state_dim
        self.action_count = action_count
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.next_action_masks = np.zeros(
            (capacity, action_count),
            dtype=np.bool_,
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
        next_action_mask: np.ndarray,
    ) -> None:
        """添加一条状态转移及其下一状态合法动作掩码。"""

        state_array = np.asarray(state, dtype=np.float32)
        next_state_array = np.asarray(next_state, dtype=np.float32)
        expected_state_shape = (self.state_dim,)
        if state_array.shape != expected_state_shape:
            raise ValueError(
                "state 维度错误："
                f"期望 {expected_state_shape}，实际 {state_array.shape}。"
            )
        if next_state_array.shape != expected_state_shape:
            raise ValueError(
                "next_state 维度错误："
                f"期望 {expected_state_shape}，实际 {next_state_array.shape}。"
            )
        if not 0 <= int(action) < self.action_count:
            raise ValueError(
                f"action 必须位于 [0, {self.action_count - 1}]。"
            )

        # 不在这里自动把 0/1 整数转成布尔值，因为上游把含义写反时，
        # 静默转换会让错误经验进入回放池，之后很难定位训练异常。
        next_mask_array = np.asarray(next_action_mask)
        if next_mask_array.dtype != np.bool_:
            raise ValueError("next_action_mask 必须是布尔数组。")
        expected_mask_shape = (self.action_count,)
        if next_mask_array.shape != expected_mask_shape:
            raise ValueError(
                "next_action_mask 维度错误："
                f"期望 {expected_mask_shape}，实际 {next_mask_array.shape}。"
            )
        if not np.any(next_mask_array):
            raise ValueError("next_action_mask 至少要包含一个合法动作。")

        position = self._position
        self.states[position] = state_array
        self.actions[position] = int(action)
        self.rewards[position] = float(reward)
        self.next_states[position] = next_state_array
        self.dones[position] = float(done)
        self.next_action_masks[position] = next_mask_array
        self._position = (position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
    ) -> ReplayBatch:
        """从已有经验中进行无放回随机采样。"""

        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0。")
        if self._size < batch_size:
            raise ValueError("经验数量不足，无法采样。")

        indices = rng.choice(
            self._size,
            size=batch_size,
            replace=False,
        )
        return ReplayBatch(
            states=self.states[indices].copy(),
            actions=self.actions[indices].copy(),
            rewards=self.rewards[indices].copy(),
            next_states=self.next_states[indices].copy(),
            dones=self.dones[indices].copy(),
            next_action_masks=self.next_action_masks[indices].copy(),
        )


class QNetwork(nn.Module):
    """输入一批状态，输出每个离散动作的 Q 值。"""

    def __init__(
        self,
        state_dim: int,
        action_count: int,
        hidden_dims: tuple[int, ...],
    ) -> None:
        super().__init__()
        if state_dim <= 0:
            raise ValueError("state_dim 必须大于 0。")
        if action_count <= 0:
            raise ValueError("action_count 必须大于 0。")

        layers: list[nn.Module] = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                ]
            )
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_count))
        self.network = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states)


class DDQNAgent:
    """面向当前 78 维状态、12 个结构化动作的 Double DQN 智能体。"""

    def __init__(
        self,
        state_dim: int,
        action_count: int,
        config: DDQNConfig,
        metadata: DDQNCheckpointMetadata,
        device: str | torch.device | None = None,
    ) -> None:
        """创建带状态/动作模式版本信息的 DDQN 智能体。"""

        self._validate_current_schema(
            state_dim=state_dim,
            action_count=action_count,
            metadata=metadata,
        )
        self.state_dim = state_dim
        self.action_count = action_count
        self.config = config
        self.metadata = metadata

        if device is None:
            selected_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            selected_device = device
        self.device = torch.device(selected_device)

        torch.manual_seed(config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.random_seed)
        self.rng = np.random.default_rng(config.random_seed)

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

    @staticmethod
    def _validate_current_schema(
        *,
        state_dim: int,
        action_count: int,
        metadata: DDQNCheckpointMetadata,
    ) -> None:
        """在创建网络前校验当前研究方案的固定模式。"""

        if state_dim != CURRENT_STATE_DIM:
            raise ValueError(
                f"当前 DDQN 状态维度必须为 {CURRENT_STATE_DIM}，"
                f"实际为 {state_dim}。"
            )
        if action_count != CURRENT_ACTION_COUNT:
            raise ValueError(
                f"当前 DDQN 动作数量必须为 {CURRENT_ACTION_COUNT}，"
                f"实际为 {action_count}。"
            )
        if metadata.state_schema_version != CURRENT_STATE_SCHEMA_VERSION:
            raise ValueError(
                "状态模式版本必须为 "
                f"{CURRENT_STATE_SCHEMA_VERSION}。"
            )
        if metadata.action_schema_version != CURRENT_ACTION_SCHEMA_VERSION:
            raise ValueError(
                "动作模式版本必须为 "
                f"{CURRENT_ACTION_SCHEMA_VERSION}。"
            )
        if metadata.mec_count != CURRENT_MEC_COUNT:
            raise ValueError(
                f"当前场景的 MEC 数量必须为 {CURRENT_MEC_COUNT}。"
            )
        if metadata.function_count != CURRENT_FUNCTION_COUNT:
            raise ValueError(
                f"当前场景的 VNF 数量必须为 {CURRENT_FUNCTION_COUNT}。"
            )

    @staticmethod
    def _validate_action_mask(
        action_mask: np.ndarray,
        action_count: int,
    ) -> np.ndarray:
        """校验单个状态的合法动作掩码并返回原布尔数组。"""

        mask_array = np.asarray(action_mask)
        if mask_array.dtype != np.bool_:
            raise ValueError("action_mask 必须是布尔数组。")
        expected_shape = (action_count,)
        if mask_array.shape != expected_shape:
            raise ValueError(
                "action_mask 维度错误："
                f"期望 {expected_shape}，实际 {mask_array.shape}。"
            )
        if not np.any(mask_array):
            raise ValueError("当前状态没有合法的 DDQN 动作。")
        return mask_array

    def sync_target_network(self) -> None:
        """将在线网络参数复制到目标网络。"""

        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

    def select_action(
        self,
        state: np.ndarray,
        epsilon: float,
        action_mask: np.ndarray,
    ) -> int:
        """使用带合法动作掩码的 epsilon-greedy 选择动作。"""

        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon 必须位于 [0, 1]。")
        state_array = np.asarray(state, dtype=np.float32)
        if state_array.shape != (self.state_dim,):
            raise ValueError(
                f"动作选择状态应为 ({self.state_dim},)，"
                f"实际为 {state_array.shape}。"
            )
        mask_array = self._validate_action_mask(
            action_mask,
            self.action_count,
        )
        valid_ids = np.flatnonzero(mask_array)

        # 探索阶段也只在合法集合内抽样，不能先随机再事后修复。
        if self.rng.random() < epsilon:
            return int(self.rng.choice(valid_ids))

        state_tensor = torch.from_numpy(state_array).unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(mask_array).to(self.device)
        was_training = self.online_network.training
        self.online_network.eval()
        with torch.no_grad():
            q_values = self.online_network(state_tensor)
            masked_q_values = q_values.masked_fill(
                ~mask_tensor.unsqueeze(0),
                float("-inf"),
            )
            action = int(masked_q_values.argmax(dim=1).item())
        self.online_network.train(was_training)
        return action

    def learn(self, replay_buffer: ReplayBuffer) -> float | None:
        """从回放池采样并更新一次网络；经验不足时返回 ``None``。"""

        if replay_buffer.state_dim != self.state_dim:
            raise ValueError("回放池状态维度与 DDQN 智能体不一致。")
        if replay_buffer.action_count != self.action_count:
            raise ValueError("回放池动作数量与 DDQN 智能体不一致。")

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
        states = torch.from_numpy(batch.states).to(self.device)
        actions = torch.from_numpy(batch.actions).long().unsqueeze(1).to(
            self.device
        )
        rewards = torch.from_numpy(batch.rewards).unsqueeze(1).to(self.device)
        next_states = torch.from_numpy(batch.next_states).to(self.device)
        dones = torch.from_numpy(batch.dones).unsqueeze(1).to(self.device)
        next_action_masks = torch.from_numpy(batch.next_action_masks).to(
            self.device
        )

        current_q_values = self.online_network(states).gather(
            dim=1,
            index=actions,
        )

        # Double DQN 分工保持不变：在线网络“选动作”，目标网络“评估动作”。
        # 新增的关键点是在线网络在 argmax 前先屏蔽下一状态的非法动作。
        with torch.no_grad():
            next_online_q_values = self.online_network(next_states)
            masked_next_online_q_values = next_online_q_values.masked_fill(
                ~next_action_masks,
                float("-inf"),
            )
            next_actions = masked_next_online_q_values.argmax(
                dim=1,
                keepdim=True,
            )
            next_q_values = self.target_network(next_states).gather(
                dim=1,
                index=next_actions,
            )
            target_q_values = (
                rewards
                + self.config.gamma * (1.0 - dones) * next_q_values
            )

        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.online_network.parameters(),
            self.config.gradient_clip_norm,
        )
        self.optimizer.step()

        self.training_steps += 1
        if self.training_steps % self.config.target_update_interval == 0:
            self.sync_target_network()
        return float(loss.item())

    def save(self, path: str | Path) -> None:
        """保存网络、优化器及状态/动作模式版本信息。"""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "state_dim": self.state_dim,
            "action_count": self.action_count,
            "metadata": asdict(self.metadata),
            "config": asdict(self.config),
            "online_network": self.online_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "training_steps": self.training_steps,
        }
        torch.save(checkpoint, output_path)

    def load(self, path: str | Path) -> None:
        """校验模型模式完全一致后，再加载网络和优化器参数。"""

        checkpoint = torch.load(
            Path(path),
            map_location=self.device,
            # 这里只需要张量和基础 Python 类型，不允许反序列化任意对象。
            weights_only=True,
        )
        if not isinstance(checkpoint, Mapping):
            raise ValueError("模型文件格式错误。")
        self._validate_checkpoint_schema(checkpoint)

        required_keys = (
            "online_network",
            "target_network",
            "optimizer",
            "training_steps",
        )
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            raise ValueError(
                "模型文件缺少必要字段：" + ", ".join(missing_keys) + "。"
            )

        self.online_network.load_state_dict(checkpoint["online_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.training_steps = int(checkpoint["training_steps"])

    def _validate_checkpoint_schema(self, checkpoint: Mapping[str, Any]) -> None:
        """给模型不兼容问题提供可理解的中文错误，而不是张量报错。"""

        metadata = checkpoint.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("模型缺少状态模式版本，不能加载旧版模型。")

        expected_metadata = asdict(self.metadata)
        if metadata.get("state_schema_version") != expected_metadata[
            "state_schema_version"
        ]:
            raise ValueError("模型状态模式版本与当前智能体不一致。")
        if metadata.get("action_schema_version") != expected_metadata[
            "action_schema_version"
        ]:
            raise ValueError("模型动作模式版本与当前智能体不一致。")
        if metadata.get("mec_count") != expected_metadata["mec_count"]:
            raise ValueError("模型 MEC 数量与当前实验场景不一致。")
        if metadata.get("function_count") != expected_metadata[
            "function_count"
        ]:
            raise ValueError("模型 VNF 数量与当前实验场景不一致。")
        if dict(metadata) != expected_metadata:
            raise ValueError("模型模式元数据包含无法识别的字段。")

        if checkpoint.get("state_dim") != self.state_dim:
            raise ValueError("模型状态维度与当前环境不一致。")
        if checkpoint.get("action_count") != self.action_count:
            raise ValueError("模型动作数量与当前环境不一致。")


def linear_epsilon(
    episode_index: int,
    epsilon_start: float,
    epsilon_end: float,
    decay_episodes: int,
) -> float:
    """计算从起始值线性衰减到结束值的 epsilon。"""

    if episode_index < 0:
        raise ValueError("episode_index 不能小于 0。")
    if not 0 <= epsilon_end <= epsilon_start <= 1:
        raise ValueError("epsilon 参数不合法。")
    if decay_episodes <= 0:
        raise ValueError("decay_episodes 必须大于 0。")

    progress = min(episode_index / decay_episodes, 1.0)
    return float(
        epsilon_start + progress * (epsilon_end - epsilon_start)
    )


def build_ddqn_config(config: dict[str, Any]) -> DDQNConfig:
    """根据项目配置字典创建 DDQN 超参数。"""

    ddqn_config = config["ddqn"]
    return DDQNConfig(
        gamma=float(ddqn_config["gamma"]),
        learning_rate=float(ddqn_config["learning_rate"]),
        hidden_dims=tuple(int(value) for value in ddqn_config["hidden_dims"]),
        replay_capacity=int(ddqn_config["replay_capacity"]),
        replay_start_size=int(ddqn_config["replay_start_size"]),
        batch_size=int(ddqn_config["batch_size"]),
        target_update_interval=int(ddqn_config["target_update_interval"]),
        gradient_clip_norm=float(ddqn_config["gradient_clip_norm"]),
        random_seed=int(ddqn_config["random_seed"]),
    )
