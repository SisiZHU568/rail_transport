"""定义 DDQN 使用的结构化慢时间尺度动作空间。"""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class RetentionPolicy(IntEnum):
    """慢层允许的 Serverless 实例保留等级。"""

    ON_DEMAND = 0
    PRIMARY_WARM = 1
    ALL_WARM = 2


class CloudPolicy(IntEnum):
    """慢层是否允许快层把副本部署到中心云。"""

    EDGE_ONLY = 0
    CLOUD_ALLOWED = 1


@dataclass(frozen=True)
class ActionFeasibilityContext:
    """保存慢层动作掩码所需的当前基础设施摘要。"""

    # 这里只保存仍可工作的轨旁节点；故障节点不能参与慢层预检查。
    operational_edge_node_ids: frozenset[int]
    operational_cloud_node_id: int | None
    node_free_memory_mb: dict[int, float]
    node_fault_domains: dict[int, int]
    total_function_memory_mb: float
    minimum_distinct_fault_domains: int


@dataclass(frozen=True)
class StructuredSlowAction:
    """保存一个可解释、可写入论文表格的 DDQN 组合动作。"""

    action_id: int
    replica_count: int
    retention_policy: RetentionPolicy
    cloud_policy: CloudPolicy


def get_ddqn_action_count() -> int:
    """返回 2 种副本数、3 种保留策略和 2 种云策略的组合数。"""

    return 12


def encode_ddqn_action(
    replica_count: int,
    retention_policy: RetentionPolicy,
    cloud_policy: CloudPolicy,
) -> int:
    """把三个慢层决策分量编码成稳定的 0～11 动作编号。"""

    if replica_count not in (2, 3):
        raise ValueError(
            "DDQN副本数只能是2或3。"
        )

    # IntEnum 转换同时完成类型统一和非法枚举值检查。
    retention = RetentionPolicy(
        retention_policy
    )
    cloud = CloudPolicy(cloud_policy)

    # 编码顺序固定为：副本数 → 保留策略 → 云策略。
    # 固定顺序能保证训练、评估和论文动作表使用同一编号。
    return (
        (replica_count - 2) * 3
        + int(retention)
    ) * 2 + int(cloud)


def decode_ddqn_action(
    action: int,
) -> StructuredSlowAction:
    """把 0～11 动作编号还原成三个慢层决策分量。"""

    action_id = int(action)
    if not 0 <= action_id < get_ddqn_action_count():
        raise ValueError(
            "非法DDQN动作，合法范围为0～11。"
        )

    replica_count = 2 + action_id // 6
    remainder = action_id % 6
    retention = RetentionPolicy(
        remainder // 2
    )
    cloud = CloudPolicy(remainder % 2)

    return StructuredSlowAction(
        action_id=action_id,
        replica_count=replica_count,
        retention_policy=retention,
        cloud_policy=cloud,
    )


def build_valid_action_mask(
    context: ActionFeasibilityContext,
) -> np.ndarray:
    """返回12个动作的必要条件掩码，True表示可继续精确检查。"""

    mask = np.zeros(
        get_ddqn_action_count(),
        dtype=np.bool_,
    )

    for action_id in range(get_ddqn_action_count()):
        action = decode_ddqn_action(action_id)
        candidate_ids = set(
            context.operational_edge_node_ids
        )

        # CLOUD_ALLOWED只是把中心云加入候选，不代表一定使用中心云。
        if (
            action.cloud_policy
            is CloudPolicy.CLOUD_ALLOWED
            and context.operational_cloud_node_id
            is not None
        ):
            candidate_ids.add(
                context.operational_cloud_node_id
            )

        enough_nodes = (
            len(candidate_ids) >= action.replica_count
        )
        enough_domains = len(
            {
                context.node_fault_domains[node_id]
                for node_id in candidate_ids
            }
        ) >= context.minimum_distinct_fault_domains

        # 只有ALL_WARM要求全部计划副本常驻内存；另外两种策略
        # 的实际冷启动与瞬时资源约束交给快层精确审计。
        enough_hot_memory = True
        if (
            action.retention_policy
            is RetentionPolicy.ALL_WARM
        ):
            required_memory_mb = (
                action.replica_count
                * context.total_function_memory_mb
            )
            available_memory_mb = sum(
                context.node_free_memory_mb.get(
                    node_id,
                    0.0,
                )
                for node_id in candidate_ids
            )
            enough_hot_memory = (
                available_memory_mb
                >= required_memory_mb
            )

        mask[action_id] = (
            enough_nodes
            and enough_domains
            and enough_hot_memory
        )

    return mask


DDQN_ACTION_NAMES = tuple(
    (
        f"R{action.replica_count}_"
        f"{action.retention_policy.name}_"
        f"{action.cloud_policy.name}"
    )
    for action in (
        decode_ddqn_action(action_id)
        for action_id in range(
            get_ddqn_action_count()
        )
    )
)


# ------------------------------------------------------------------
# 迁移期兼容接口
# ------------------------------------------------------------------
# 训练入口和旧 RL 环境将在后续任务中一起迁移。暂时保留下面的三动作
# 适配器，避免中间提交出现导入错误；新 DDQN 动作选择不得调用这些接口。
class DDQNAction(IntEnum):
    """旧三动作编号，仅供尚未迁移的入口临时导入。"""

    COLD = 0
    HOT = 1
    DYNAMIC = 2


_LEGACY_AGENT_TO_ENVIRONMENT_ACTION = {
    # 旧冷备和动态策略都对应“两副本、主副本保温、仅边缘”。
    DDQNAction.COLD: 2,
    DDQNAction.HOT: 4,
    DDQNAction.DYNAMIC: 2,
}


_LEGACY_ENVIRONMENT_TO_AGENT_ACTION = {
    environment_action: agent_action
    for agent_action, environment_action
    in _LEGACY_AGENT_TO_ENVIRONMENT_ACTION.items()
}


def ddqn_action_to_environment_action(
    action: int | DDQNAction,
) -> int:
    """临时把旧三动作编号映射到新的结构化动作编号。"""

    try:
        ddqn_action = DDQNAction(int(action))
    except ValueError as error:
        raise ValueError(
            "迁移期旧DDQN动作只允许0～2；"
            "结构化动作不能通过旧适配器执行。"
        ) from error

    return int(
        _LEGACY_AGENT_TO_ENVIRONMENT_ACTION[
            ddqn_action
        ]
    )


def environment_action_to_ddqn_action(
    action: int,
) -> int:
    """临时把可兼容的结构化动作编号映射回旧三动作编号。"""

    try:
        environment_action = int(action)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"非法环境动作：{action}。"
        ) from error

    if environment_action not in _LEGACY_ENVIRONMENT_TO_AGENT_ACTION:
        raise ValueError("该结构化动作没有唯一的旧三动作映射。")
    return int(_LEGACY_ENVIRONMENT_TO_AGENT_ACTION[environment_action])
