"""测试 DPPO 面向整条 SFC 的连续联合动作布局。"""

import numpy as np
import pytest

from src.dppo_action_space import (
    DPPOActionSpace,
    DecodedFunctionAction,
)
from src.scenario_dimensions import ScenarioDimensions


def test_decode_default_joint_action() -> None:
    """默认 27 维动作应同时解码节点排名、副本数和主备保留时间。"""

    dimensions = ScenarioDimensions(tuple(range(5)), 5, (0, 1, 2))
    space = DPPOActionSpace(
        dimensions,
        maximum_retention_seconds=20.0,
        minimum_replicas=2,
        maximum_replicas=3,
    )
    action = np.zeros(dimensions.action_dim, dtype=np.float32)
    action[:6] = np.asarray([0.2, 0.9, -0.1, 0.7, 0.1, 0.3])

    decoded = space.decode(action)

    first = decoded.function_actions[0]
    assert first.replica_count == 3
    assert first.ranked_node_ids[:3] == (1, 3, 5)
    assert first.primary_retention_seconds == pytest.approx(10.0)
    assert first.backup_retention_seconds == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("mec_ids", "function_ids", "expected_action_dim"),
    [
        ((0, 1, 2), (0, 1), 14),
        ((0, 1, 2, 3, 4), (0, 1, 2), 27),
        (tuple(range(8)), (0, 1, 2), 36),
    ],
)
def test_action_slices_follow_scenario_dimensions(
    mec_ids: tuple[int, ...],
    function_ids: tuple[int, ...],
    expected_action_dim: int,
) -> None:
    """改变实验规模时，每段动作长度都必须由统一维度对象计算。"""

    dimensions = ScenarioDimensions(mec_ids, len(mec_ids), function_ids)
    space = DPPOActionSpace(
        dimensions,
        maximum_retention_seconds=20.0,
        minimum_replicas=2,
        maximum_replicas=3,
    )

    assert space.action_dim == expected_action_dim
    assert space.node_score_count == len(function_ids) * (len(mec_ids) + 1)
    assert space.replica_score_count == len(function_ids)
    assert space.retention_value_count == 2 * len(function_ids)


def test_replica_quantization_and_interleaved_retention_are_per_function() -> None:
    """每个 VNF 独立选择副本数，并按“主、备”顺序读取连续保留时间。"""

    dimensions = ScenarioDimensions((0, 1, 2), 3, (10, 20))
    space = DPPOActionSpace(
        dimensions,
        maximum_retention_seconds=20.0,
        minimum_replicas=2,
        maximum_replicas=3,
    )
    action = np.zeros(dimensions.action_dim, dtype=np.float32)
    action[space.replica_score_slice] = np.asarray([-0.01, 0.0])
    action[space.retention_slice] = np.asarray([-1.0, 1.0, -0.5, 0.5])

    decoded = space.decode(action)
    first, second = decoded.function_actions

    assert first.replica_count == 2
    assert second.replica_count == 3
    assert first.primary_retention_seconds == pytest.approx(0.0)
    assert first.backup_retention_seconds == pytest.approx(20.0)
    assert second.primary_retention_seconds == pytest.approx(5.0)
    assert second.backup_retention_seconds == pytest.approx(15.0)


def test_equal_node_scores_are_broken_by_node_id() -> None:
    """评分完全相同时按节点 ID 升序，保证相同种子可以重复实验。"""

    dimensions = ScenarioDimensions((10, 2, 7), 99, (0,))
    space = DPPOActionSpace(
        dimensions,
        maximum_retention_seconds=20.0,
        minimum_replicas=2,
        maximum_replicas=3,
    )

    decoded = space.decode(np.zeros(dimensions.action_dim, dtype=np.float32))

    assert decoded.function_actions[0].ranked_node_ids == (2, 7, 10, 99)


def test_clip_rejects_invalid_actions_and_does_not_modify_input() -> None:
    """执行动作可以截断，但错误形状和非有限值必须明确拒绝。"""

    dimensions = ScenarioDimensions((0, 1, 2), 3, (0, 1))
    space = DPPOActionSpace(
        dimensions,
        maximum_retention_seconds=20.0,
        minimum_replicas=2,
        maximum_replicas=3,
    )
    raw_action = np.linspace(-2.0, 2.0, dimensions.action_dim, dtype=np.float64)
    original = raw_action.copy()

    clipped = space.clip(raw_action)

    np.testing.assert_array_equal(raw_action, original)
    assert clipped.dtype == np.float32
    assert np.all(clipped >= -1.0)
    assert np.all(clipped <= 1.0)

    with pytest.raises(ValueError, match="动作形状"):
        space.decode(np.zeros((1, dimensions.action_dim), dtype=np.float32))

    non_finite = np.zeros(dimensions.action_dim, dtype=np.float32)
    non_finite[0] = np.nan
    with pytest.raises(ValueError, match="有限值"):
        space.decode(non_finite)


def test_teacher_action_round_trip_preserves_semantics() -> None:
    """教师动作编码后再解码，应保留节点顺序、副本数和保留时间。"""

    dimensions = ScenarioDimensions((0, 1, 2), 3, (10, 20))
    space = DPPOActionSpace(
        dimensions,
        maximum_retention_seconds=20.0,
        minimum_replicas=2,
        maximum_replicas=3,
    )
    teacher_actions = (
        DecodedFunctionAction(10, (3, 2, 1, 0), 2, 4.0, 8.0),
        DecodedFunctionAction(20, (1, 0, 2, 3), 3, 6.0, 12.0),
    )

    encoded = space.encode_teacher_action(teacher_actions)
    decoded = space.decode(encoded)

    for expected, actual in zip(
        teacher_actions,
        decoded.function_actions,
        strict=True,
    ):
        assert actual.function_id == expected.function_id
        assert actual.ranked_node_ids == expected.ranked_node_ids
        assert actual.replica_count == expected.replica_count
        assert actual.primary_retention_seconds == pytest.approx(
            expected.primary_retention_seconds
        )
        assert actual.backup_retention_seconds == pytest.approx(
            expected.backup_retention_seconds
        )


def test_action_space_rejects_invalid_hyperparameters() -> None:
    """训练与评估必须使用有物理意义且可编码的相同动作参数。"""

    dimensions = ScenarioDimensions((0, 1, 2), 3, (0,))

    with pytest.raises(ValueError, match="maximum_retention_seconds"):
        DPPOActionSpace(
            dimensions,
            maximum_retention_seconds=0.0,
            minimum_replicas=2,
            maximum_replicas=3,
        )

    with pytest.raises(ValueError, match="minimum_replicas"):
        DPPOActionSpace(
            dimensions,
            maximum_retention_seconds=20.0,
            minimum_replicas=0,
            maximum_replicas=3,
        )

    with pytest.raises(ValueError, match="minimum_replicas.*maximum_replicas"):
        DPPOActionSpace(
            dimensions,
            maximum_retention_seconds=20.0,
            minimum_replicas=4,
            maximum_replicas=3,
        )

    with pytest.raises(ValueError, match="maximum_replicas.*compute_node_count"):
        DPPOActionSpace(
            dimensions,
            maximum_retention_seconds=20.0,
            minimum_replicas=2,
            maximum_replicas=5,
        )


def test_replica_score_uniformly_decodes_configured_interval() -> None:
    """一个连续分量被均匀量化，副本上限扩展时动作维度不变。"""

    dimensions = ScenarioDimensions(tuple(range(5)), 5, (0, 1, 2, 3))
    space = DPPOActionSpace(
        dimensions,
        maximum_retention_seconds=20.0,
        minimum_replicas=2,
        maximum_replicas=5,
    )
    action = np.zeros(space.action_dim, dtype=np.float32)
    action[space.replica_score_slice] = np.asarray(
        [-1.0, -0.25, 0.25, 1.0],
        dtype=np.float32,
    )

    decoded_counts = [
        item.replica_count for item in space.decode(action).function_actions
    ]

    assert decoded_counts == [2, 3, 4, 5]


def test_fixed_replica_interval_round_trips_teacher_action() -> None:
    """上下限相同时仍保留同一动作布局，并稳定解码为固定副本数。"""

    dimensions = ScenarioDimensions(tuple(range(5)), 5, (0,))
    space = DPPOActionSpace(
        dimensions,
        maximum_retention_seconds=20.0,
        minimum_replicas=4,
        maximum_replicas=4,
    )
    teacher = DecodedFunctionAction(
        function_id=0,
        ranked_node_ids=dimensions.compute_node_ids,
        replica_count=4,
        primary_retention_seconds=10.0,
        backup_retention_seconds=5.0,
    )

    encoded = space.encode_teacher_action((teacher,))

    assert space.decode(encoded).function_actions[0].replica_count == 4
