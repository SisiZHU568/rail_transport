"""
cost_model.py

本文件负责计算一个时隙中的系统成本。

当前成本由三部分组成：

1. 用户请求时延成本；
2. 温容器空闲保留成本；
3. 主动预热资源成本。

当前使用归一化成本，而不是人民币、美元或实际电费。
后续获得真实测量数据后，可以替换成本参数。
"""

from dataclasses import dataclass

from src.cold_start import LifecycleResult
from src.entities import InstanceStatus, ServerlessFunction


@dataclass(frozen=True)
class CostWeights:
    """
    成本权重。

    Attributes
    ----------
    delay_cost_per_ms:
        每 1 ms 用户请求时延对应的成本。

    warm_memory_cost_per_mb_second:
        温容器每占用 1 MB 内存、持续 1 秒的成本。

    prewarm_start_cost_per_ms:
        主动预热每消耗 1 ms 启动时间对应的资源成本。
    """

    delay_cost_per_ms: float
    warm_memory_cost_per_mb_second: float
    prewarm_start_cost_per_ms: float

    def __post_init__(self) -> None:
        """
        成本权重不能为负数。
        """

        if self.delay_cost_per_ms < 0:
            raise ValueError("时延成本权重不能小于 0。")

        if self.warm_memory_cost_per_mb_second < 0:
            raise ValueError("内存保留成本权重不能小于 0。")

        if self.prewarm_start_cost_per_ms < 0:
            raise ValueError("预热成本权重不能小于 0。")


@dataclass(frozen=True)
class SlotCostBreakdown:
    """
    一个时隙的成本分解结果。
    """

    # 当前时隙真实请求感知到的总时延
    request_delay_ms: float

    # 请求时延对应的成本
    delay_cost: float

    # 空闲温容器的保留成本
    retention_cost: float

    # 主动预热产生的资源成本
    prewarm_cost: float

    @property
    def total_cost(self) -> float:
        """
        返回当前时隙总成本。
        """

        return (
            self.delay_cost
            + self.retention_cost
            + self.prewarm_cost
        )


def calculate_slot_cost(
    function: ServerlessFunction,
    lifecycle_result: LifecycleResult,
    slot_seconds: float,
    weights: CostWeights,
) -> SlotCostBreakdown:
    """
    计算一个时隙中的成本。

    Parameters
    ----------
    function:
        当前函数的资源和执行参数。

    lifecycle_result:
        容器生命周期管理器返回的时隙结果。

    slot_seconds:
        当前快时隙长度，单位为秒。

    weights:
        成本权重。

    Returns
    -------
    SlotCostBreakdown:
        时延、保留、预热和总成本。
    """

    if slot_seconds <= 0:
        raise ValueError("时隙长度必须大于 0。")

    # ========================================================
    # 1. 用户请求时延
    # ========================================================
    #
    # 有真实请求时，用户感知的时延包括：
    #
    # 冷启动时延 + 函数执行时延
    #
    # 主动预热发生在没有真实请求的时隙，
    # 因此预热启动时间不能算作用户请求时延。
    if lifecycle_result.request_count > 0:
        request_delay_ms = (
            lifecycle_result.cold_start_delay_ms
            + lifecycle_result.execution_delay_ms
        )
    else:
        request_delay_ms = 0.0

    delay_cost = (
        request_delay_ms
        * weights.delay_cost_per_ms
    )

    # ========================================================
    # 2. 空闲温容器保留成本
    # ========================================================
    #
    # 满足以下条件时收取保留成本：
    #
    # 1. 当前没有真实请求；
    # 2. 时隙结束时容器仍为 WARM；
    # 3. 容器没有在当前时隙被销毁。
    retained_during_idle_slot = (
        lifecycle_result.request_count == 0
        and lifecycle_result.after_status == InstanceStatus.WARM
        and not lifecycle_result.expired
    )

    if retained_during_idle_slot:
        retention_cost = (
            function.memory_mb
            * slot_seconds
            * weights.warm_memory_cost_per_mb_second
        )
    else:
        retention_cost = 0.0

    # ========================================================
    # 3. 主动预热成本
    # ========================================================
    #
    # 主动预热虽然不会增加用户请求时延，
    # 但会消耗创建容器和加载镜像所需的资源。
    if lifecycle_result.prewarm_occurred:
        prewarm_cost = (
            lifecycle_result.cold_start_delay_ms
            * weights.prewarm_start_cost_per_ms
        )
    else:
        prewarm_cost = 0.0

    return SlotCostBreakdown(
        request_delay_ms=request_delay_ms,
        delay_cost=delay_cost,
        retention_cost=retention_cost,
        prewarm_cost=prewarm_cost,
    )