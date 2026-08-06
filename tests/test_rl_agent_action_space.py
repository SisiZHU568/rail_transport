"""
test_rl_agent_action_space.py

测试三动作Double DQN与四动作环境之间的动作映射。
"""

import pytest

from src.rl_agent_action_space import (
    DDQNAction,
    ddqn_action_to_environment_action,
    environment_action_to_ddqn_action,
    get_ddqn_action_count,
)
from src.slow_timescale_rl_env import (
    SlowControlAction,
)


def test_ddqn_has_three_actions() -> None:
    assert get_ddqn_action_count() == 3

    assert list(DDQNAction) == [
        DDQNAction.COLD,
        DDQNAction.HOT,
        DDQNAction.DYNAMIC,
    ]


def test_ddqn_to_environment_mapping() -> None:
    assert (
        ddqn_action_to_environment_action(
            DDQNAction.COLD
        )
        == int(
            SlowControlAction.COLD
        )
    )

    assert (
        ddqn_action_to_environment_action(
            DDQNAction.HOT
        )
        == int(
            SlowControlAction.HOT
        )
    )

    assert (
        ddqn_action_to_environment_action(
            DDQNAction.DYNAMIC
        )
        == int(
            SlowControlAction.DYNAMIC
        )
    )


def test_environment_to_ddqn_mapping() -> None:
    assert (
        environment_action_to_ddqn_action(
            SlowControlAction.COLD
        )
        == int(
            DDQNAction.COLD
        )
    )

    assert (
        environment_action_to_ddqn_action(
            SlowControlAction.HOT
        )
        == int(
            DDQNAction.HOT
        )
    )

    assert (
        environment_action_to_ddqn_action(
            SlowControlAction.DYNAMIC
        )
        == int(
            DDQNAction.DYNAMIC
        )
    )


def test_single_is_not_ddqn_action() -> None:
    with pytest.raises(
        ValueError,
        match="不属于Double DQN动作空间",
    ):
        environment_action_to_ddqn_action(
            SlowControlAction.SINGLE
        )


def test_invalid_ddqn_action_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="非法Double DQN动作",
    ):
        ddqn_action_to_environment_action(
            3
        )