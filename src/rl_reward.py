
"""
rl_reward.py

本文件定义双时间尺度强化学习环境的奖励函数。

奖励函数同时考虑：

1. 用户端到端时延；
2. 温实例内存占用；
3. 冷备用启动时延；
4. SLA违反率；
5. 副本重配置开销。

所有指标先归一化到[0,1]，
再计算加权成本。

最终奖励：

    reward = -normalized_cost

因此：

    最好情况接近 0
    最差情况接近 -1
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RLRewardWeights:
    """
    强化学习奖励权重。
    """

    delay: float
    memory: float
    cold_start: float
    sla_violation: float
    reconfiguration: float

    def __post_init__(self) -> None:
        """
        检查奖励权重。
        """

        values = (
            self.delay,
            self.memory,
            self.cold_start,
            self.sla_violation,
            self.reconfiguration,
        )

        if any(
            not math.isfinite(value)
            for value in values
        ):
            raise ValueError(
                "奖励权重必须是有限数值。"
            )

        if any(value < 0 for value in values):
            raise ValueError(
                "奖励权重不能小于0。"
            )

        if sum(values) <= 0:
            raise ValueError(
                "至少需要一个大于0的奖励权重。"
            )

    @property
    def total_weight(self) -> float:
        """
        返回所有权重之和。
        """

        return (
            self.delay
            + self.memory
            + self.cold_start
            + self.sla_violation
            + self.reconfiguration
        )


@dataclass(frozen=True)
class RLWindowMetrics:
    """
    一个慢时间尺度窗口的运行指标。
    """

    total_requests: int
    successful_requests: int

    request_batches: int
    successful_batches: int
    failed_batches: int

    deadline_violations: int
    sla_violations: int

    request_success_rate: float
    sla_violation_rate: float

    average_successful_delay_ms: float
    total_cold_start_delay_ms: float

    average_active_memory_mb: float
    total_active_memory_mb_seconds: float

    failover_function_stages: int
    cold_start_function_stages: int

    reconfigured_function_stages: int

    def __post_init__(self) -> None:
        """
        检查窗口指标是否合法。
        """

        integer_values = (
            self.total_requests,
            self.successful_requests,
            self.request_batches,
            self.successful_batches,
            self.failed_batches,
            self.deadline_violations,
            self.sla_violations,
            self.failover_function_stages,
            self.cold_start_function_stages,
            self.reconfigured_function_stages,
        )

        if any(value < 0 for value in integer_values):
            raise ValueError(
                "窗口计数指标不能小于0。"
            )

        if not 0 <= self.request_success_rate <= 1:
            raise ValueError(
                "请求成功率必须位于[0,1]。"
            )

        if not 0 <= self.sla_violation_rate <= 1:
            raise ValueError(
                "SLA违反率必须位于[0,1]。"
            )

        continuous_values = (
            self.average_successful_delay_ms,
            self.total_cold_start_delay_ms,
            self.average_active_memory_mb,
            self.total_active_memory_mb_seconds,
        )

        if any(
            not math.isfinite(value)
            or value < 0
            for value in continuous_values
        ):
            raise ValueError(
                "连续窗口指标必须是非负有限数值。"
            )


@dataclass(frozen=True)
class RLRewardBreakdown:
    """
    奖励函数的分项结果。
    """

    normalized_delay: float
    normalized_memory: float
    normalized_cold_start: float
    normalized_sla_violation: float
    normalized_reconfiguration: float

    weighted_cost: float
    reward: float


def calculate_rl_reward(
    metrics: RLWindowMetrics,
    deadline_ms: float,
    maximum_active_memory_mb: float,
    maximum_cold_start_delay_ms_per_batch: float,
    function_count: int,
    weights: RLRewardWeights,
) -> RLRewardBreakdown:
    """
    计算一个慢时间尺度窗口的奖励。

    Parameters
    ----------
    metrics:
        当前窗口运行指标。

    deadline_ms:
        SFC时延约束。

    maximum_active_memory_mb:
        全热备情况下的最大活动内存。

    maximum_cold_start_delay_ms_per_batch:
        一批请求中全部函数都冷启动时的最大启动时延。

    function_count:
        SFC函数数量。

    weights:
        奖励权重。
    """

    if deadline_ms <= 0:
        raise ValueError(
            "SFC时延约束必须大于0。"
        )

    if maximum_active_memory_mb <= 0:
        raise ValueError(
            "最大活动内存必须大于0。"
        )

    if maximum_cold_start_delay_ms_per_batch <= 0:
        raise ValueError(
            "最大冷启动时延必须大于0。"
        )

    if function_count <= 0:
        raise ValueError(
            "函数数量必须大于0。"
        )

    # 成功请求的平均时延归一化。
    #
    # 超过deadline的部分由SLA违反项惩罚，
    # 因此这里截断到1。
    normalized_delay = min(
        metrics.average_successful_delay_ms
        / deadline_ms,
        1.0,
    )

    normalized_memory = min(
        metrics.average_active_memory_mb
        / maximum_active_memory_mb,
        1.0,
    )

    if metrics.request_batches > 0:
        cold_start_denominator = (
            maximum_cold_start_delay_ms_per_batch
            * metrics.request_batches
        )

        normalized_cold_start = min(
            metrics.total_cold_start_delay_ms
            / cold_start_denominator,
            1.0,
        )
    else:
        normalized_cold_start = 0.0

    normalized_sla_violation = (
        metrics.sla_violation_rate
    )

    normalized_reconfiguration = min(
        metrics.reconfigured_function_stages
        / function_count,
        1.0,
    )

    weighted_sum = (
        weights.delay
        * normalized_delay
        +
        weights.memory
        * normalized_memory
        +
        weights.cold_start
        * normalized_cold_start
        +
        weights.sla_violation
        * normalized_sla_violation
        +
        weights.reconfiguration
        * normalized_reconfiguration
    )

    weighted_cost = (
        weighted_sum
        / weights.total_weight
    )

    reward = -weighted_cost

    return RLRewardBreakdown(
        normalized_delay=normalized_delay,
        normalized_memory=normalized_memory,
        normalized_cold_start=(
            normalized_cold_start
        ),
        normalized_sla_violation=(
            normalized_sla_violation
        ),
        normalized_reconfiguration=(
            normalized_reconfiguration
        ),
        weighted_cost=weighted_cost,
        reward=reward,
    )