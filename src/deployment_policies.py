"""保存规则基线仍需使用的算法无关部署策略枚举。"""

from enum import IntEnum


class RetentionPolicy(IntEnum):
    """规则慢层允许的 Serverless 实例保留等级。"""

    ON_DEMAND = 0
    PRIMARY_WARM = 1
    ALL_WARM = 2


class CloudPolicy(IntEnum):
    """规则慢层是否允许快层把副本部署到中心云。"""

    EDGE_ONLY = 0
    CLOUD_ALLOWED = 1
