"""测试 DPPO 主副本与备用副本相互独立的连续保留时间。"""

import pytest

from src.continuous_retention import ContinuousRetentionTracker


def test_primary_and_backup_expire_at_independent_slots() -> None:
    """主副本与备用副本使用不同秒数，因此可以在不同快时隙到期。"""

    tracker = ContinuousRetentionTracker(slot_seconds=1.0)
    tracker.apply_intent(
        slot=2,
        function_id=0,
        primary_node_id=1,
        backup_node_ids=(2,),
        primary_seconds=4.0,
        backup_seconds=1.0,
    )

    assert tracker.hot_node_ids(0, slot=3) == frozenset({1, 2})
    assert tracker.hot_node_ids(0, slot=4) == frozenset({1})
    assert tracker.hot_node_ids(0, slot=7) == frozenset()


def test_fractional_seconds_round_up_to_complete_slots() -> None:
    """非整数保留时间向上取整，避免容器比动作要求更早释放。"""

    tracker = ContinuousRetentionTracker(slot_seconds=1.0)
    tracker.apply_intent(
        slot=5,
        function_id=0,
        primary_node_id=1,
        backup_node_ids=(2,),
        primary_seconds=1.2,
        backup_seconds=0.1,
    )

    assert tracker.hot_node_ids(0, slot=6) == frozenset({1, 2})
    assert tracker.hot_node_ids(0, slot=7) == frozenset({1})
    assert tracker.hot_node_ids(0, slot=8) == frozenset()


def test_refresh_extends_but_does_not_shorten_existing_retention() -> None:
    """新决策只能延长已有到期时隙，较短动作不能意外清除温热实例。"""

    tracker = ContinuousRetentionTracker(slot_seconds=1.0)
    tracker.apply_intent(
        slot=2,
        function_id=0,
        primary_node_id=1,
        backup_node_ids=(2,),
        primary_seconds=4.0,
        backup_seconds=4.0,
    )
    tracker.apply_intent(
        slot=3,
        function_id=0,
        primary_node_id=1,
        backup_node_ids=(2,),
        primary_seconds=1.0,
        backup_seconds=6.0,
    )

    assert tracker.hot_node_ids(0, slot=6) == frozenset({1, 2})
    assert tracker.hot_node_ids(0, slot=7) == frozenset({2})
    assert tracker.hot_node_ids(0, slot=10) == frozenset()


def test_zero_seconds_does_not_create_or_shorten_retention() -> None:
    """零秒不创建新温热实例，也不缩短同一实例尚未结束的保留期。"""

    tracker = ContinuousRetentionTracker(slot_seconds=1.0)
    tracker.apply_intent(
        slot=1,
        function_id=0,
        primary_node_id=1,
        backup_node_ids=(2,),
        primary_seconds=3.0,
        backup_seconds=0.0,
    )
    tracker.apply_intent(
        slot=2,
        function_id=0,
        primary_node_id=1,
        backup_node_ids=(3,),
        primary_seconds=0.0,
        backup_seconds=0.0,
    )

    assert tracker.hot_node_ids(0, slot=3) == frozenset({1})
    assert tracker.hot_node_ids(0, slot=5) == frozenset()


def test_failed_nodes_are_removed_immediately() -> None:
    """节点一旦故障，其上的温热容器不能继续被状态编码器视为可用。"""

    tracker = ContinuousRetentionTracker(slot_seconds=1.0)
    tracker.apply_intent(
        slot=0,
        function_id=0,
        primary_node_id=1,
        backup_node_ids=(2, 3),
        primary_seconds=10.0,
        backup_seconds=10.0,
    )

    tracker.remove_failed_nodes({1, 3})

    assert tracker.hot_node_ids(0, slot=1) == frozenset({2})


def test_reset_clears_retention_from_previous_episode() -> None:
    """开始新 Episode 时必须清空旧状态，避免训练样本之间相互泄漏。"""

    tracker = ContinuousRetentionTracker(slot_seconds=1.0)
    tracker.apply_intent(
        slot=0,
        function_id=0,
        primary_node_id=1,
        backup_node_ids=(2,),
        primary_seconds=10.0,
        backup_seconds=10.0,
    )

    tracker.reset()

    assert tracker.hot_node_ids(0, slot=1) == frozenset()


@pytest.mark.parametrize("slot_seconds", [0.0, -1.0, float("inf"), float("nan")])
def test_tracker_rejects_invalid_slot_length(slot_seconds: float) -> None:
    """快时隙长度必须为正有限数，否则秒数无法可靠换算。"""

    with pytest.raises(ValueError, match="slot_seconds"):
        ContinuousRetentionTracker(slot_seconds=slot_seconds)


@pytest.mark.parametrize(
    ("primary_seconds", "backup_seconds"),
    [
        (-1.0, 1.0),
        (1.0, -1.0),
        (float("inf"), 1.0),
        (1.0, float("nan")),
    ],
)
def test_apply_intent_rejects_invalid_retention_seconds(
    primary_seconds: float,
    backup_seconds: float,
) -> None:
    """动作中的保留时间必须是非负有限数。"""

    tracker = ContinuousRetentionTracker(slot_seconds=1.0)

    with pytest.raises(ValueError, match="retention seconds"):
        tracker.apply_intent(
            slot=0,
            function_id=0,
            primary_node_id=1,
            backup_node_ids=(2,),
            primary_seconds=primary_seconds,
            backup_seconds=backup_seconds,
        )


def test_apply_intent_rejects_duplicate_replica_nodes() -> None:
    """同一 VNF 的主副本和备用副本不能指向同一计算节点。"""

    tracker = ContinuousRetentionTracker(slot_seconds=1.0)

    with pytest.raises(ValueError, match="unique"):
        tracker.apply_intent(
            slot=0,
            function_id=0,
            primary_node_id=1,
            backup_node_ids=(1,),
            primary_seconds=1.0,
            backup_seconds=1.0,
        )
