"""
run_ddqn_training.py

训练轨道边缘双时间尺度Double DQN。
"""
from src.rl_agent_action_space import (
    DDQNAction,
    DDQN_ACTION_NAMES,
    ddqn_action_to_environment_action,
    get_ddqn_action_count,
)
import os

# 使用非交互式绘图后端。
# 训练脚本只保存图片，不需要Qt窗口。
os.environ.setdefault(
    "MPLBACKEND",
    "Agg",
)

# 限制底层并行线程，减少Windows环境中的运行库冲突。
os.environ.setdefault(
    "OMP_NUM_THREADS",
    "1",
)

os.environ.setdefault(
    "MKL_NUM_THREADS",
    "1",
)

import csv
import random
from collections import deque
from pathlib import Path

import matplotlib

# 必须在导入pyplot之前设置。
matplotlib.use(
    "Agg",
    force=True,
)

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import load_config
from src.ddqn import (
    DDQNAgent,
    ReplayBuffer,
    build_ddqn_config,
    linear_epsilon,
)
from src.rl_scenario import (
    build_rl_environment,
)

# 禁用Matplotlib交互模式。
plt.ioff()


def set_global_seed(
    random_seed: int,
) -> None:
    """
    固定Python、NumPy和PyTorch随机种子。
    """

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            random_seed
        )


def save_history(
    rows: list[dict],
    output_path: Path,
) -> None:
    """
    保存训练历史。
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def plot_series(
    x_values: list[int],
    y_values: list[float],
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
) -> None:
    """
    绘制一条训练曲线。
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(10, 5)
    )

    axis.plot(
        x_values,
        y_values,
    )

    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(alpha=0.3)

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
    )

    plt.close(figure)


def main() -> None:
    project_root = (
        Path(__file__).resolve().parent
    )

    config = load_config(
        project_root
        / "configs"
        / "debug.yaml"
    )

    yaml_config = config["ddqn"]

    random_seed = int(
        yaml_config["random_seed"]
    )

    set_global_seed(
        random_seed
    )

    environment = build_rl_environment(
        config
    )

    agent_config = build_ddqn_config(
        config
    )

    # 环境仍支持4种动作，
    # 但Double DQN只输出3种动作。
    ddqn_action_count = (
        get_ddqn_action_count()
    )

    agent = DDQNAgent(
        state_dim=environment.state_dim,
        action_count=ddqn_action_count,
        config=agent_config,
    )

    replay_buffer = ReplayBuffer(
        capacity=(
            agent_config.replay_capacity
        ),
        state_dim=environment.state_dim,
    )

    episodes = int(
        yaml_config["episodes"]
    )

    train_seed_start = int(
        yaml_config["train_seed_start"]
    )

    epsilon_start = float(
        yaml_config["epsilon_start"]
    )

    epsilon_end = float(
        yaml_config["epsilon_end"]
    )

    epsilon_decay_episodes = int(
        yaml_config[
            "epsilon_decay_episodes"
        ]
    )

    best_model_path = (
        project_root
        / yaml_config["best_model_path"]
    )

    last_model_path = (
        project_root
        / yaml_config["last_model_path"]
    )

    history_path = (
        project_root
        / yaml_config["history_path"]
    )

    figures_dir = (
        project_root
        / "results"
        / "figures"
    )

    reward_curve_path = (
        figures_dir
        / "ddqn_reliable_training_reward.png"
    )

    loss_curve_path = (
        figures_dir
        / "ddqn_reliable_training_loss.png"
    )

    epsilon_curve_path = (
        figures_dir
        / "ddqn_reliable_training_epsilon.png"
    )

    reward_window: deque[float] = deque(
        maxlen=20
    )

    history_rows: list[dict] = []

    best_average_reward = float(
        "-inf"
    )

    print("=" * 72)
    print("Double DQN训练")
    print("=" * 72)

    print(
        f"设备：{agent.device}"
    )

    print(
        f"状态维度："
        f"{environment.state_dim}"
    )

    print(
        f"环境动作数量："
        f"{environment.action_count}"
    )

    print(
        f"Double DQN动作数量："
        f"{ddqn_action_count}"
    )

    print(
        "Double DQN可选动作："
        + ", ".join(
            DDQN_ACTION_NAMES
        )
    )

    print(
        f"训练Episode："
        f"{episodes}"
    )

    print(
        f"网络隐藏层："
        f"{agent_config.hidden_dims}"
    )

    for episode in range(
        1,
        episodes + 1,
    ):
        episode_seed = (
            train_seed_start
            + episode
            - 1
        )

        state, _ = environment.reset(
            seed=episode_seed
        )

        epsilon = linear_epsilon(
            episode_index=episode - 1,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            decay_episodes=(
                epsilon_decay_episodes
            ),
        )

        episode_reward = 0.0
        episode_losses: list[float] = []

        episode_decisions = 0
        action_counts = [
            0
            for _ in range(
                ddqn_action_count
            )
        ]

        terminated = False
        truncated = False

        while not (
            terminated or truncated
        ):
                        # 智能体内部动作范围为0～2。
            agent_action = (
                agent.select_action(
                    state=state,
                    epsilon=epsilon,
                )
            )

            action_counts[
                agent_action
            ] += 1

            # 转换为环境动作1～3。
            environment_action = (
                ddqn_action_to_environment_action(
                    agent_action
                )
            )

            (
                next_state,
                reward,
                terminated,
                truncated,
                _,
            ) = environment.step(
                environment_action
            )

            done = (
                terminated or truncated
            )

            # 回放池保存智能体内部动作0～2。
            #
            # 不能保存环境动作1～3，
            # 否则动作3会超过网络输出范围。
            replay_buffer.add(
                state=state,
                action=agent_action,
                reward=reward,
                next_state=next_state,
                done=done,
            )
            loss = agent.learn(
                replay_buffer
            )

            if loss is not None:
                episode_losses.append(
                    loss
                )

            state = next_state

            episode_reward += reward
            episode_decisions += 1

        reward_window.append(
            episode_reward
        )

        average_reward_20 = float(
            np.mean(reward_window)
        )

        average_loss = (
            float(
                np.mean(
                    episode_losses
                )
            )
            if episode_losses
            else 0.0
        )

        history_rows.append(
            {
                "episode": episode,
                "episode_seed": episode_seed,
                "reward": episode_reward,
                "average_reward_20": (
                    average_reward_20
                ),
                "epsilon": epsilon,
                "average_loss": (
                    average_loss
                ),
                "decision_count": (
                    episode_decisions
                ),
                "replay_size": len(
                    replay_buffer
                ),
                                # SINGLE不属于智能体动作空间。
                "single_count": 0,

                "cold_count": (
                    action_counts[
                        int(
                            DDQNAction.COLD
                        )
                    ]
                ),

                "hot_count": (
                    action_counts[
                        int(
                            DDQNAction.HOT
                        )
                    ]
                ),

                "dynamic_count": (
                    action_counts[
                        int(
                            DDQNAction.DYNAMIC
                        )
                    ]
                ),
            }
        )

        # 至少积累20个Episode后，
        # 再使用Avg20选择最优模型。
        if (
            episode >= 20
            and average_reward_20
            > best_average_reward
        ):
            best_average_reward = (
                average_reward_20
            )

            agent.save(
                best_model_path
            )

        print(
            f"Ep {episode:4d}/{episodes}"
            f" | Reward: "
            f"{episode_reward:8.4f}"
            f" | Avg20: "
            f"{average_reward_20:8.4f}"
            f" | Epsilon: "
            f"{epsilon:6.3f}"
            f" | Loss: "
            f"{average_loss:9.6f}"
        )

    agent.save(
        last_model_path
    )

    save_history(
        rows=history_rows,
        output_path=history_path,
    )

    episode_numbers = [
        int(row["episode"])
        for row in history_rows
    ]

    plot_series(
        x_values=episode_numbers,
        y_values=[
            float(
                row["average_reward_20"]
            )
            for row in history_rows
        ],
        title=(
            "Double DQN Training Reward"
        ),
        x_label="Episode",
        y_label="Average Reward over 20 Episodes",
        output_path=reward_curve_path,
    )

    plot_series(
        x_values=episode_numbers,
        y_values=[
            float(
                row["average_loss"]
            )
            for row in history_rows
        ],
        title=(
            "Double DQN Training Loss"
        ),
        x_label="Episode",
        y_label="Average TD Loss",
        output_path=loss_curve_path,
    )

    plot_series(
        x_values=episode_numbers,
        y_values=[
            float(
                row["epsilon"]
            )
            for row in history_rows
        ],
        title=(
            "Double DQN Epsilon Decay"
        ),
        x_label="Episode",
        y_label="Epsilon",
        output_path=epsilon_curve_path,
    )

    print("\n" + "=" * 72)
    print("Double DQN训练完成")
    print("=" * 72)

    print(
        f"最优Avg20："
        f"{best_average_reward:.6f}"
    )

    print(
        f"最优模型："
        f"{best_model_path}"
    )

    print(
        f"最终模型："
        f"{last_model_path}"
    )

    print(
        f"训练历史："
        f"{history_path}"
    )

    print(
        f"奖励曲线："
        f"{reward_curve_path}"
    )

    print(
        f"损失曲线："
        f"{loss_curve_path}"
    )

    print(
        f"探索率曲线："
        f"{epsilon_curve_path}"
    )


if __name__ == "__main__":
    main()