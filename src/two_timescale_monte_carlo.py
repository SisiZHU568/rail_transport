"""
two_timescale_monte_carlo.py

本文件负责双时间尺度控制方案的配对蒙特卡洛实验。

对于每一个随机种子：

    fixed_single
    fixed_cold
    fixed_hot
    fixed_dynamic
    two_timescale

五种方案使用相同的基础设施随机故障轨迹。

本模块保存：

1. 每次实验原始指标；
2. 每种方案的均值；
3. 样本标准差；
4. 置信区间；
5. 最小值和最大值。
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from src.monte_carlo import (
    MetricSummary,
    summarize_metric,
)


class TwoTimescaleSimulatorProtocol(
    Protocol
):
    """
    双时间尺度蒙特卡洛仿真器接口。
    """

    def run(self):
        """
        执行一次完整仿真。
        """

        ...


@dataclass(frozen=True)
class TwoTimescaleMonteCarloRunRecord:
    """
    一种方案在一个随机种子下的原始结果。
    """

    scenario_name: str
    run_index: int
    random_seed: int

    request_success_rate: float
    sla_violation_rate: float

    average_successful_batch_delay_ms: float
    p95_successful_batch_delay_ms: float

    average_active_memory_mb: float
    peak_active_memory_mb: float

    cold_start_function_stages: int
    total_cold_start_delay_ms: float

    slow_decision_updates: int
    slow_mode_switches: int

    fast_repair_attempts: int
    fast_repair_successes: int
    fast_repair_failures: int
    fast_repair_success_rate: float
    constraint_rejected_batches: int

    total_request_delay_cost: float
    total_memory_cost: float
    total_cold_start_cost: float
    total_sla_penalty: float
    total_slow_control_cost: float
    total_system_cost: float


@dataclass(frozen=True)
class TwoTimescaleMonteCarloScenarioSummary:
    """
    一种方案经过多次实验后的统计结果。
    """

    scenario_name: str
    run_count: int

    request_success_rate: MetricSummary
    sla_violation_rate: MetricSummary

    average_successful_batch_delay_ms: MetricSummary
    p95_successful_batch_delay_ms: MetricSummary

    average_active_memory_mb: MetricSummary
    peak_active_memory_mb: MetricSummary

    cold_start_function_stages: MetricSummary
    total_cold_start_delay_ms: MetricSummary

    slow_decision_updates: MetricSummary
    slow_mode_switches: MetricSummary

    fast_repair_attempts: MetricSummary
    fast_repair_successes: MetricSummary
    fast_repair_failures: MetricSummary
    fast_repair_success_rate: MetricSummary
    constraint_rejected_batches: MetricSummary

    total_request_delay_cost: MetricSummary
    total_memory_cost: MetricSummary
    total_cold_start_cost: MetricSummary
    total_sla_penalty: MetricSummary
    total_slow_control_cost: MetricSummary
    total_system_cost: MetricSummary


@dataclass(frozen=True)
class TwoTimescaleMonteCarloResult:
    """
    完整的双时间尺度蒙特卡洛结果。
    """

    confidence_level: float

    run_records: tuple[
        TwoTimescaleMonteCarloRunRecord,
        ...,
    ]

    scenario_summaries: tuple[
        TwoTimescaleMonteCarloScenarioSummary,
        ...,
    ]


def run_two_timescale_monte_carlo(
    scenario_builders: dict[
        str,
        Callable[
            [int],
            TwoTimescaleSimulatorProtocol,
        ],
    ],
    num_runs: int,
    base_seed: int,
    confidence_level: float,
) -> TwoTimescaleMonteCarloResult:
    """
    执行配对随机种子蒙特卡洛实验。
    """

    if not scenario_builders:
        raise ValueError(
            "至少需要配置一个对比方案。"
        )

    if num_runs <= 0:
        raise ValueError(
            "蒙特卡洛运行次数必须大于0。"
        )

    if not isinstance(base_seed, int):
        raise TypeError(
            "base_seed必须是整数。"
        )

    if not 0 < confidence_level < 1:
        raise ValueError(
            "置信水平必须位于(0,1)。"
        )

    run_records: list[
        TwoTimescaleMonteCarloRunRecord
    ] = []

    # 外层遍历随机种子，
    # 保证每个随机种子被全部方案共同使用。
    for run_index in range(num_runs):
        random_seed = (
            base_seed + run_index
        )

        for (
            scenario_name,
            simulator_builder,
        ) in scenario_builders.items():
            if not scenario_name:
                raise ValueError(
                    "方案名称不能为空。"
                )

            simulator = simulator_builder(
                random_seed
            )

            result = simulator.run()
            summary = result.summary

            run_records.append(
                TwoTimescaleMonteCarloRunRecord(
                    scenario_name=scenario_name,
                    run_index=run_index,
                    random_seed=random_seed,
                    request_success_rate=(
                        summary.request_success_rate
                    ),
                    sla_violation_rate=(
                        summary.sla_violation_rate
                    ),
                    average_successful_batch_delay_ms=(
                        summary
                        .average_successful_batch_delay_ms
                    ),
                    p95_successful_batch_delay_ms=(
                        summary
                        .p95_successful_batch_delay_ms
                    ),
                    average_active_memory_mb=(
                        summary.average_active_memory_mb
                    ),
                    peak_active_memory_mb=(
                        summary.peak_active_memory_mb
                    ),
                    cold_start_function_stages=(
                        summary
                        .cold_start_function_stages
                    ),
                    total_cold_start_delay_ms=(
                        summary.total_cold_start_delay_ms
                    ),
                    slow_decision_updates=(
                        summary.slow_decision_updates
                    ),
                    slow_mode_switches=(
                        summary.slow_mode_switches
                    ),
                    fast_repair_attempts=(
                        summary.fast_repair_attempts
                    ),
                    fast_repair_successes=(
                        summary.fast_repair_successes
                    ),
                    fast_repair_failures=(
                        summary.fast_repair_failures
                    ),
                    fast_repair_success_rate=(
                        summary.fast_repair_success_rate
                    ),
                    constraint_rejected_batches=(
                        summary.constraint_rejected_batches
                    ),
                    total_request_delay_cost=(
                        summary.total_request_delay_cost
                    ),
                    total_memory_cost=(
                        summary.total_memory_cost
                    ),
                    total_cold_start_cost=(
                        summary.total_cold_start_cost
                    ),
                    total_sla_penalty=(
                        summary.total_sla_penalty
                    ),
                    total_slow_control_cost=(
                        summary.total_slow_control_cost
                    ),
                    total_system_cost=(
                        summary.total_system_cost
                    ),
                )
            )

    scenario_summaries: list[
        TwoTimescaleMonteCarloScenarioSummary
    ] = []

    for scenario_name in scenario_builders:
        scenario_records = [
            record
            for record in run_records
            if record.scenario_name
            == scenario_name
        ]

        if len(scenario_records) != num_runs:
            raise RuntimeError(
                f"方案{scenario_name}的运行次数异常。"
            )

        def metric(
            attribute_name: str,
        ) -> MetricSummary:
            """
            汇总一个属性。
            """

            return summarize_metric(
                [
                    float(
                        getattr(
                            record,
                            attribute_name,
                        )
                    )
                    for record in scenario_records
                ],
                confidence_level=(
                    confidence_level
                ),
            )

        scenario_summaries.append(
            TwoTimescaleMonteCarloScenarioSummary(
                scenario_name=scenario_name,
                run_count=num_runs,
                request_success_rate=metric(
                    "request_success_rate"
                ),
                sla_violation_rate=metric(
                    "sla_violation_rate"
                ),
                average_successful_batch_delay_ms=metric(
                    "average_successful_batch_delay_ms"
                ),
                p95_successful_batch_delay_ms=metric(
                    "p95_successful_batch_delay_ms"
                ),
                average_active_memory_mb=metric(
                    "average_active_memory_mb"
                ),
                peak_active_memory_mb=metric(
                    "peak_active_memory_mb"
                ),
                cold_start_function_stages=metric(
                    "cold_start_function_stages"
                ),
                total_cold_start_delay_ms=metric(
                    "total_cold_start_delay_ms"
                ),
                slow_decision_updates=metric(
                    "slow_decision_updates"
                ),
                slow_mode_switches=metric(
                    "slow_mode_switches"
                ),
                fast_repair_attempts=metric(
                    "fast_repair_attempts"
                ),
                fast_repair_successes=metric(
                    "fast_repair_successes"
                ),
                fast_repair_failures=metric(
                    "fast_repair_failures"
                ),
                fast_repair_success_rate=metric(
                    "fast_repair_success_rate"
                ),
                constraint_rejected_batches=metric(
                    "constraint_rejected_batches"
                ),
                total_request_delay_cost=metric(
                    "total_request_delay_cost"
                ),
                total_memory_cost=metric(
                    "total_memory_cost"
                ),
                total_cold_start_cost=metric(
                    "total_cold_start_cost"
                ),
                total_sla_penalty=metric(
                    "total_sla_penalty"
                ),
                total_slow_control_cost=metric(
                    "total_slow_control_cost"
                ),
                total_system_cost=metric(
                    "total_system_cost"
                ),
            )
        )

    return TwoTimescaleMonteCarloResult(
        confidence_level=confidence_level,
        run_records=tuple(run_records),
        scenario_summaries=tuple(
            scenario_summaries
        ),
    )
