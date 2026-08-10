
"""
rl_reward.py

本文件定义双时间尺度强化学习环境的简化奖励函数：

    total_cost = run_cost + route_cost + cold_start_cost
    reward = -(normalized_total_cost + violation_penalty)

仅保留论文主算法实际使用的三项成本和单一硬违约项，避免引入额外权重。
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

    # 以下字段直接汇总共享快层执行器的结果，便于论文逐项报告原始结果。
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
