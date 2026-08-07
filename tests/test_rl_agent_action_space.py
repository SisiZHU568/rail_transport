"""测试 DDQN 结构化慢层动作的编码、解码与边界检查。"""

import numpy as np
import pytest

from src.rl_agent_action_space import (
    ActionFeasibilityContext,
    CloudPolicy,
    DDQN_ACTION_NAMES,
    RetentionPolicy,
    build_valid_action_mask,
    decode_ddqn_action,
    encode_ddqn_action,
    get_ddqn_action_count,
)


def test_action_mask_uses_cloud_to_restore_domain_diversity() -> None:
    """仅边缘故障域不足时，允许云的同类动作仍可合法。"""

    context = ActionFeasibilityContext(
        operational_edge_node_ids=frozenset({0, 1}),
        operational_cloud_node_id=5,
        node_free_memory_mb={
            0: 4000.0,
            1: 4000.0,
            5: 64000.0,
        },
        node_fault_domains={0: 0, 1: 0, 5: 3},
        total_function_memory_mb=1536.0,
        minimum_distinct_fault_domains=2,
    )

    mask = build_valid_action_mask(context)

    assert mask.shape == (12,)
    assert mask.dtype == np.bool_
    assert not bool(mask[0])
    assert bool(mask[1])


def test_action_mask_rejects_impossible_all_warm_memory() -> None:
    """全副本保温的内存下界超限时，只屏蔽对应保温动作。"""

    context = ActionFeasibilityContext(
        operational_edge_node_ids=frozenset({0, 1}),
        operational_cloud_node_id=None,
        node_free_memory_mb={0: 100.0, 1: 100.0},
        node_fault_domains={0: 0, 1: 1},
        total_function_memory_mb=150.0,
        minimum_distinct_fault_domains=2,
    )

    mask = build_valid_action_mask(context)

    assert bool(mask[0])
    assert not bool(mask[4])
    assert not bool(mask[6])


@pytest.mark.parametrize(
    ("action_id", "replicas", "retention", "cloud"),
    [
        (0, 2, RetentionPolicy.ON_DEMAND, CloudPolicy.EDGE_ONLY),
        (5, 2, RetentionPolicy.ALL_WARM, CloudPolicy.CLOUD_ALLOWED),
        (6, 3, RetentionPolicy.ON_DEMAND, CloudPolicy.EDGE_ONLY),
        (11, 3, RetentionPolicy.ALL_WARM, CloudPolicy.CLOUD_ALLOWED),
    ],
)
def test_structured_action_decoding(
    action_id: int,
    replicas: int,
    retention: RetentionPolicy,
    cloud: CloudPolicy,
) -> None:
    """代表性动作必须能按论文定义双向转换。"""

    action = decode_ddqn_action(action_id)

    assert action.replica_count == replicas
    assert action.retention_policy is retention
    assert action.cloud_policy is cloud
    assert encode_ddqn_action(
        replica_count=replicas,
        retention_policy=retention,
        cloud_policy=cloud,
    ) == action_id


def test_action_catalog_has_twelve_unique_entries() -> None:
    """2×3×2 的笛卡尔积必须生成 12 个互不重复的动作。"""

    actions = {
        decode_ddqn_action(action_id)
        for action_id in range(12)
    }

    assert get_ddqn_action_count() == 12
    assert len(actions) == 12
    assert len(DDQN_ACTION_NAMES) == 12
    assert len(set(DDQN_ACTION_NAMES)) == 12


def test_every_action_round_trips() -> None:
    """完整检查 0～11，防止只正确处理示例动作。"""

    for action_id in range(get_ddqn_action_count()):
        action = decode_ddqn_action(action_id)
        encoded = encode_ddqn_action(
            replica_count=action.replica_count,
            retention_policy=action.retention_policy,
            cloud_policy=action.cloud_policy,
        )
        assert encoded == action_id


@pytest.mark.parametrize("action_id", [-1, 12])
def test_invalid_action_id_is_rejected(action_id: int) -> None:
    """超出动作编号上下界时必须给出便于排查的错误。"""

    with pytest.raises(ValueError, match="0～11"):
        decode_ddqn_action(action_id)


def test_invalid_replica_count_is_rejected() -> None:
    """DDQN 慢层只允许研究方案中的 2 或 3 副本。"""

    with pytest.raises(ValueError, match="只能是2或3"):
        encode_ddqn_action(
            replica_count=1,
            retention_policy=RetentionPolicy.ON_DEMAND,
            cloud_policy=CloudPolicy.EDGE_ONLY,
        )
