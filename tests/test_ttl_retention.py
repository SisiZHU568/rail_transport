"""
test_ttl_retention.py

测试跨窗口 TTL 容器保留逻辑。
"""

import pytest

from src.ttl_retention import (
    TTLAction,
    TTLRetentionManager,
)


TTL_OPTIONS = [0, 3, 6, 10]


def test_ttl_options_validation() -> None:
    manager = TTLRetentionManager(
        TTL_OPTIONS
    )

    assert manager.ttl_slots_for_action(
        TTLAction.TTL_0
    ) == 0

    assert manager.ttl_slots_for_action(
        TTLAction.TTL_3
    ) == 10


def test_positive_action_prewarms_targets() -> None:
    manager = TTLRetentionManager(
        TTL_OPTIONS
    )

    result = manager.apply_action(
        action=TTLAction.TTL_1,
        target_backup_keys=[
            (0, 2),
            (1, 2),
            (2, 2),
        ],
    )

    assert result.ttl_slots == 3
    assert len(
        result.newly_prewarmed_keys
    ) == 3

    assert manager.is_warm(
        (0, 2)
    )

    assert manager.remaining_slots_for(
        (0, 2)
    ) == 3


def test_ttl_expires_after_exact_number_of_slots() -> None:
    manager = TTLRetentionManager(
        TTL_OPTIONS
    )

    manager.apply_action(
        action=TTLAction.TTL_1,
        target_backup_keys=[
            (0, 2),
        ],
    )

    assert manager.tick() == ()
    assert manager.remaining_slots_for(
        (0, 2)
    ) == 2

    assert manager.tick() == ()
    assert manager.remaining_slots_for(
        (0, 2)
    ) == 1

    expired = manager.tick()

    assert expired == (
        (0, 2),
    )

    assert not manager.is_warm(
        (0, 2)
    )


def test_refresh_can_shorten_previous_ttl() -> None:
    manager = TTLRetentionManager(
        TTL_OPTIONS
    )

    manager.apply_action(
        action=TTLAction.TTL_3,
        target_backup_keys=[
            (0, 2),
        ],
    )

    assert manager.remaining_slots_for(
        (0, 2)
    ) == 10

    result = manager.apply_action(
        action=TTLAction.TTL_1,
        target_backup_keys=[
            (0, 2),
        ],
    )

    assert result.refreshed_keys == (
        (0, 2),
    )

    assert manager.remaining_slots_for(
        (0, 2)
    ) == 3


def test_zero_action_releases_all_standby_containers() -> None:
    manager = TTLRetentionManager(
        TTL_OPTIONS
    )

    manager.apply_action(
        action=TTLAction.TTL_2,
        target_backup_keys=[
            (0, 2),
            (1, 2),
        ],
    )

    result = manager.apply_action(
        action=TTLAction.TTL_0,
        target_backup_keys=[],
    )

    assert result.released_keys == (
        (0, 2),
        (1, 2),
    )

    assert manager.snapshot().warm_keys == ()


def test_old_non_target_container_keeps_counting_down() -> None:
    manager = TTLRetentionManager(
        TTL_OPTIONS
    )

    manager.apply_action(
        action=TTLAction.TTL_2,
        target_backup_keys=[
            (0, 2),
        ],
    )

    manager.tick(
        elapsed_slots=2
    )

    # 列车切换后，新的目标备用节点变为4。
    manager.apply_action(
        action=TTLAction.TTL_1,
        target_backup_keys=[
            (0, 4),
        ],
    )

    assert manager.remaining_slots_for(
        (0, 2)
    ) == 4

    assert manager.remaining_slots_for(
        (0, 4)
    ) == 3


def test_request_execution_refreshes_selected_backup() -> None:
    manager = TTLRetentionManager(
        TTL_OPTIONS
    )

    newly_retained = (
        manager.retain_after_request(
            executed_backup_keys=[
                (2, 3),
            ],
            action=TTLAction.TTL_2,
        )
    )

    assert newly_retained == (
        (2, 3),
    )

    assert manager.remaining_slots_for(
        (2, 3)
    ) == 6


def test_active_memory_uses_function_memory() -> None:
    manager = TTLRetentionManager(
        TTL_OPTIONS
    )

    manager.apply_action(
        action=TTLAction.TTL_1,
        target_backup_keys=[
            (0, 2),
            (1, 2),
            (2, 2),
        ],
    )

    memory = (
        manager.active_standby_memory_mb(
            {
                0: 256.0,
                1: 512.0,
                2: 768.0,
            }
        )
    )

    assert memory == pytest.approx(
        1536.0
    )


def test_duplicate_keys_are_deduplicated() -> None:
    manager = TTLRetentionManager(
        TTL_OPTIONS
    )

    result = manager.apply_action(
        action=TTLAction.TTL_1,
        target_backup_keys=[
            (0, 2),
            (0, 2),
        ],
    )

    assert result.target_keys == (
        (0, 2),
    )

    assert len(
        result.newly_prewarmed_keys
    ) == 1
