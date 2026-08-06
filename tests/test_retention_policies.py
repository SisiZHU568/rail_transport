"""
test_retention_policies.py

测试三种基础容器保留策略。
"""

from src.entities import FunctionInstance
from src.retention_policies import (
    AlwaysWarmPolicy,
    FixedWindowPolicy,
    OnDemandPolicy,
)


def test_on_demand_policy() -> None:
    """
    按需策略使用1个时隙的生存窗口，
    并且永远不主动保温。
    """

    policy = OnDemandPolicy()
    instance = FunctionInstance(node_id=0, function_id=0)

    assert policy.survival_slots == 1

    assert policy.should_keep_warm(
        time_slot=0,
        request_count=0,
        instance=instance,
    ) is False


def test_fixed_window_policy() -> None:
    """
    固定窗口策略应正确保存指定窗口长度。
    """

    policy = FixedWindowPolicy(window_slots=3)
    instance = FunctionInstance(node_id=0, function_id=0)

    assert policy.survival_slots == 3

    assert policy.should_keep_warm(
        time_slot=0,
        request_count=0,
        instance=instance,
    ) is False


def test_always_warm_policy() -> None:
    """
    始终保温策略应在所有时隙返回 True。
    """

    policy = AlwaysWarmPolicy()
    instance = FunctionInstance(node_id=0, function_id=0)

    assert policy.should_keep_warm(
        time_slot=0,
        request_count=0,
        instance=instance,
    ) is True