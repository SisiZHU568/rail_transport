"""
rl_agent_action_space.py

Double DQN智能体使用的三动作空间。

环境仍支持四种动作：

    0：SINGLE
    1：COLD
    2：HOT
    3：DYNAMIC

Double DQN只允许选择：

    0：COLD
    1：HOT
    2：DYNAMIC

因此需要将智能体动作转换为环境动作。
"""

from enum import IntEnum

from src.slow_timescale_rl_env import (
    SlowControlAction,
)


class DDQNAction(IntEnum):
    """
    Double DQN内部动作编号。
    """

    COLD = 0
    HOT = 1
    DYNAMIC = 2


DDQN_ACTION_NAMES = (
    "COLD",
    "HOT",
    "DYNAMIC",
)


_AGENT_TO_ENVIRONMENT_ACTION = {
    DDQNAction.COLD: SlowControlAction.COLD,
    DDQNAction.HOT: SlowControlAction.HOT,
    DDQNAction.DYNAMIC: SlowControlAction.DYNAMIC,
}


_ENVIRONMENT_TO_AGENT_ACTION = {
    environment_action: agent_action
    for agent_action, environment_action
    in _AGENT_TO_ENVIRONMENT_ACTION.items()
}


def get_ddqn_action_count() -> int:
    """
    返回Double DQN动作数量。
    """

    return len(DDQNAction)


def ddqn_action_to_environment_action(
    action: int | DDQNAction,
) -> int:
    """
    将Double DQN动作映射为环境动作。

    映射关系：

        DDQN 0 -> 环境1 COLD
        DDQN 1 -> 环境2 HOT
        DDQN 2 -> 环境3 DYNAMIC
    """

    try:
        ddqn_action = DDQNAction(
            int(action)
        )

    except ValueError as error:
        raise ValueError(
            "非法Double DQN动作："
            f"{action}，合法范围为0～2。"
        ) from error

    return int(
        _AGENT_TO_ENVIRONMENT_ACTION[
            ddqn_action
        ]
    )


def environment_action_to_ddqn_action(
    action: int | SlowControlAction,
) -> int:
    """
    将环境动作转换为Double DQN动作。

    SINGLE不属于智能体动作空间。
    """

    try:
        environment_action = (
            SlowControlAction(
                int(action)
            )
        )

    except ValueError as error:
        raise ValueError(
            f"非法环境动作：{action}。"
        ) from error

    if (
        environment_action
        == SlowControlAction.SINGLE
    ):
        raise ValueError(
            "SINGLE只作为对比基线，"
            "不属于Double DQN动作空间。"
        )

    return int(
        _ENVIRONMENT_TO_AGENT_ACTION[
            environment_action
        ]
    )