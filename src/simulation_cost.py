"""
simulation_cost.py

本文件根据完整仿真结果计算系统综合成本。

当前综合成本包含三部分：

1. 用户请求端到端时延成本；
2. 温容器内存占用成本；
3. 后台主动预热启动成本。

预热浪费不再额外重复收费，因为浪费的预热已经产生：

1. 后台启动成本；
2. 温容器内存占用成本。

因此，预热浪费次数作为独立指标记录，
但不会再增加一个重复的“浪费惩罚成本”。
"""

from dataclasses import dataclass
import math

from src.simulator import SimulationRunResult


@dataclass(frozen=True)
class SimulationCostWeights:
    """
    完整仿真的成本权重。

    Attributes
    ----------
    delay_cost_per_ms:
        每 1 ms 用户端到端时延对应的成本。

    warm_memory_cost_per_mb_second:
        温容器占用 1 MB 内存并持续 1 秒的成本。

    prewarm_start_cost_per_ms:
        后台预热每消耗 1 ms 启动时间对应的成本。
    """

    delay_cost_per_ms: float
    warm_memory_cost_per_mb_second: float
    prewarm_start_cost_per_ms: float

    def __post_init__(self) -> None:
        """
        检查成本权重是否合法。
        """

        values = [
            self.delay_cost_per_ms,
            self.warm_memory_cost_per_mb_second,
            self.prewarm_start_cost_per_ms,
        ]

        if any(not math.isfinite(value) for value in values):
            raise ValueError("成本权重必须是有限数值。")

        if any(value < 0 for value in values):
            raise ValueError("成本权重不能小于 0。")


@dataclass(frozen=True)
class SimulationCostAnalysis:
    """
    一次完整仿真的成本分析结果。
    """

    # 用户请求感知到的端到端时延总量
    total_request_delay_ms: float

    # 温容器内存占用总量，单位为 MB·s
    total_warm_memory_mb_seconds: float

    # 后台主动预热启动时间总量
    total_prewarm_startup_overhead_ms: float

    # 预热启动、命中和浪费次数
    prewarm_starts: int
    prewarm_hits: int
    prewarm_wastes: int

    # 到仿真结束时仍未命中也未过期的预热数量
    unresolved_prewarms: int

    # 预热命中率和浪费率
    prewarm_hit_rate: float
    prewarm_waste_rate: float

    # 温容器内存统计
    average_warm_memory_mb: float
    peak_warm_memory_mb: float

    # 三类成本
    total_delay_cost: float
    total_retention_cost: float
    total_prewarm_cost: float

    # 系统综合成本
    total_system_cost: float


def analyze_simulation_cost(
    result: SimulationRunResult,
    slot_seconds: float,
    weights: SimulationCostWeights,
) -> SimulationCostAnalysis:
    """
    根据完整仿真结果计算综合成本。

    Parameters
    ----------
    result:
        RailServerlessSFCSimulator 返回的仿真结果。

    slot_seconds:
        一个快时隙的持续时间，单位为秒。

    weights:
        三类成本权重。

    Returns
    -------
    SimulationCostAnalysis:
        预热质量、内存占用和综合成本结果。
    """

    if slot_seconds <= 0:
        raise ValueError("时隙长度必须大于 0。")

    records = result.records

    # 只对真实请求产生的端到端时延进行计费。
    total_request_delay_ms = sum(
        record.end_to_end_delay_ms
        for record in records
        if record.request_count > 0
    )

    # 当前采用“时隙结束快照”统计内存占用：
    #
    # 内存占用量
    # =
    # 每个时隙结束时温实例总内存 × 时隙长度
    total_warm_memory_mb_seconds = sum(
        record.warm_memory_mb * slot_seconds
        for record in records
    )

    total_prewarm_startup_overhead_ms = sum(
        record.prewarm_startup_overhead_ms
        for record in records
    )

    prewarm_starts = result.summary.prewarm_starts

    prewarm_hits = sum(
        record.prewarm_hit_count
        for record in records
    )

    prewarm_wastes = sum(
        record.prewarm_waste_count
        for record in records
    )

    # 仿真结束时仍处于温状态、尚未命中也没有过期的预热。
    unresolved_prewarms = max(
        prewarm_starts
        - prewarm_hits
        - prewarm_wastes,
        0,
    )

    if prewarm_starts > 0:
        prewarm_hit_rate = (
            prewarm_hits / prewarm_starts
        )

        prewarm_waste_rate = (
            prewarm_wastes / prewarm_starts
        )
    else:
        prewarm_hit_rate = 0.0
        prewarm_waste_rate = 0.0

    if len(records) > 0:
        average_warm_memory_mb = (
            sum(
                record.warm_memory_mb
                for record in records
            )
            / len(records)
        )

        peak_warm_memory_mb = max(
            record.warm_memory_mb
            for record in records
        )
    else:
        average_warm_memory_mb = 0.0
        peak_warm_memory_mb = 0.0

    total_delay_cost = (
        total_request_delay_ms
        * weights.delay_cost_per_ms
    )

    total_retention_cost = (
        total_warm_memory_mb_seconds
        * weights.warm_memory_cost_per_mb_second
    )

    total_prewarm_cost = (
        total_prewarm_startup_overhead_ms
        * weights.prewarm_start_cost_per_ms
    )

    total_system_cost = (
        total_delay_cost
        + total_retention_cost
        + total_prewarm_cost
    )

    return SimulationCostAnalysis(
        total_request_delay_ms=(
            total_request_delay_ms
        ),
        total_warm_memory_mb_seconds=(
            total_warm_memory_mb_seconds
        ),
        total_prewarm_startup_overhead_ms=(
            total_prewarm_startup_overhead_ms
        ),
        prewarm_starts=prewarm_starts,
        prewarm_hits=prewarm_hits,
        prewarm_wastes=prewarm_wastes,
        unresolved_prewarms=unresolved_prewarms,
        prewarm_hit_rate=prewarm_hit_rate,
        prewarm_waste_rate=prewarm_waste_rate,
        average_warm_memory_mb=(
            average_warm_memory_mb
        ),
        peak_warm_memory_mb=peak_warm_memory_mb,
        total_delay_cost=total_delay_cost,
        total_retention_cost=(
            total_retention_cost
        ),
        total_prewarm_cost=total_prewarm_cost,
        total_system_cost=total_system_cost,
    )