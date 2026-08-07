"""定义 DDQN 使用的结构化慢时间尺度动作空间。"""

from dataclasses import dataclass
from enum import IntEnum

from src.slow_timescale_rl_env import (
    SlowControlAction,
)


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
    DDQNAction.COLD: SlowControlAction.COLD,
    DDQNAction.HOT: SlowControlAction.HOT,
    DDQNAction.DYNAMIC: (
        SlowControlAction.DYNAMIC
    ),
}


_LEGACY_ENVIRONMENT_TO_AGENT_ACTION = {
    environment_action: agent_action
    for agent_action, environment_action
    in _LEGACY_AGENT_TO_ENVIRONMENT_ACTION.items()
}


def ddqn_action_to_environment_action(
    action: int | DDQNAction,
) -> int:
    """临时把旧三动作编号映射到尚未迁移的环境动作。"""

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
    action: int | SlowControlAction,
) -> int:
    """临时把非 SINGLE 的旧环境动作映射回旧三动作编号。"""

    try:
        environment_action = SlowControlAction(
            int(action)
        )
    except ValueError as error:
        raise ValueError(
            f"非法环境动作：{action}。"
        ) from error

    if environment_action is SlowControlAction.SINGLE:
        raise ValueError(
            "SINGLE只作为对比基线，"
            "不属于Double DQN动作空间。"
        )

    return int(
        _LEGACY_ENVIRONMENT_TO_AGENT_ACTION[
            environment_action
        ]
    )
