
"""
rl_reward.py

本文件定义双时间尺度强化学习环境的奖励函数。

正式 DDQN 将使用简化奖励：

    total_cost = run_cost + route_cost + cold_start_cost
    reward = -(normalized_total_cost + violation_penalty)

旧的五权重奖励接口暂时保留，供尚未迁移的演示和训练入口使用；
这些调用方在下一步迁移后会与旧接口一起删除。
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RLWindowCostMetrics:
    """保存一个慢窗口内直接计价的三个成本和违约标志。"""

    run_cost: float
    route_cost: float
    cold_start_cost: float
    has_violation: bool

    def __post_init__(self) -> None:
        """拒绝会破坏奖励稳定性的负成本或非有限成本。"""

        values = (
            self.run_cost,
            self.route_cost,
            self.cold_start_cost,
        )
        if any(
            not math.isfinite(value)
            or value < 0
            for value in values
        ):
            raise ValueError(
                "窗口成本必须是非负有限值。"
            )


@dataclass(frozen=True)
class RLCostRewardBreakdown:
    """保存简化奖励的原始成本、归一化结果和最终奖励。"""

    run_cost: float
    route_cost: float
    cold_start_cost: float
    total_cost: float
    normalized_total_cost: float
    violation_penalty: float
    reward: float


def calculate_cost_reward(
    metrics: RLWindowCostMetrics,
    maximum_window_cost: float,
) -> RLCostRewardBreakdown:
    """按“归一化总成本 + 单一违约项”计算慢窗口奖励。"""

    if (
        not math.isfinite(maximum_window_cost)
        or maximum_window_cost <= 0
    ):
        raise ValueError(
            "窗口最大成本必须是正有限值。"
        )

    total_cost = (
        metrics.run_cost
        + metrics.route_cost
        + metrics.cold_start_cost
    )

    # 极端窗口的成本统一截断为 1，避免个别异常值主导训练。
    normalized_total_cost = min(
        total_cost / maximum_window_cost,
        1.0,
    )

    # 资源、可靠性和 SLA 约束统一折叠为一个二值硬违约项，
    # 不再为每种约束分别引入需要调优的奖励权重。
    violation_penalty = (
        1.0 if metrics.has_violation else 0.0
    )
    reward = -(
        normalized_total_cost
        + violation_penalty
    )

    return RLCostRewardBreakdown(
        run_cost=metrics.run_cost,
        route_cost=metrics.route_cost,
        cold_start_cost=(
            metrics.cold_start_cost
        ),
        total_cost=total_cost,
        normalized_total_cost=(
            normalized_total_cost
        ),
        violation_penalty=violation_penalty,
        reward=reward,
    )


# 以下接口是旧 RL 环境的迁移期兼容代码。
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

    # 以下字段直接汇总共享快层执行器的结果。默认值用于兼容尚未迁移的统计代码；
    # 正式 DDQN 环境会显式填写这些字段，便于论文逐项报告原始结果。
    total_run_cost: float = 0.0
    total_route_cost: float = 0.0
    total_cold_start_cost: float = 0.0
    fast_repair_attempts: int = 0
    fast_repair_successes: int = 0
    fast_repair_failures: int = 0
    constraint_rejected_batches: int = 0
    cloud_used_slots: int = 0
    cloud_usage_rate: float = 0.0
    minimum_exact_sfc_reliability: float = 0.0
    mean_exact_sfc_reliability: float = 0.0

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
            self.fast_repair_attempts,
            self.fast_repair_successes,
            self.fast_repair_failures,
            self.constraint_rejected_batches,
            self.cloud_used_slots,
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
            self.total_run_cost,
            self.total_route_cost,
            self.total_cold_start_cost,
            self.minimum_exact_sfc_reliability,
            self.mean_exact_sfc_reliability,
        )

        if any(
            not math.isfinite(value)
            or value < 0
            for value in continuous_values
        ):
            raise ValueError(
                "连续窗口指标必须是非负有限数值。"
            )

        if self.fast_repair_successes + self.fast_repair_failures > (
            self.fast_repair_attempts
        ):
            raise ValueError("快层修复成功数与失败数不能超过尝试数。")
        if not 0.0 <= self.cloud_usage_rate <= 1.0:
            raise ValueError("中心云使用率必须位于 [0, 1]。")
        if not 0.0 <= self.minimum_exact_sfc_reliability <= 1.0:
            raise ValueError("最小精确可靠性必须位于 [0, 1]。")
        if not 0.0 <= self.mean_exact_sfc_reliability <= 1.0:
            raise ValueError("平均精确可靠性必须位于 [0, 1]。")

    @property
    def total_cost(self) -> float:
        """返回简化奖励使用的三项原始成本之和。"""

        return (
            self.total_run_cost
            + self.total_route_cost
            + self.total_cold_start_cost
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
