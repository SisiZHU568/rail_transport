"""
monte_carlo.py

本文件负责执行多随机种子蒙特卡洛实验。

蒙特卡洛实验的基本流程：

1. 生成一组随机种子；
2. 对每个随机种子生成一条随机故障轨迹；
3. 所有对比方案使用同一个种子；
4. 记录每个方案在每次实验中的指标；
5. 计算均值、标准差和置信区间。

这种设计称为“配对随机种子实验”。

例如在随机种子1000下：

    单副本、冷备、热备、动态主备

都面对同一条基础设施故障轨迹，
从而避免不同随机故障造成不公平比较。
"""

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite, sqrt
from statistics import (
    NormalDist,
    mean,
    stdev,
)
from typing import Any, Protocol


class MonteCarloSimulatorProtocol(Protocol):
    """
    蒙特卡洛实验需要的仿真器接口。
    """

    def run(self) -> Any:
        """
        执行一次仿真。
        """

        ...


@dataclass(frozen=True)
class MetricSummary:
    """
    一个指标的统计汇总结果。

    Attributes
    ----------
    count:
        样本数量。

    mean:
        样本均值。

    std:
        样本标准差，使用 n-1 作为分母。

    ci_low:
        置信区间下界。

    ci_high:
        置信区间上界。

    minimum:
        最小观测值。

    maximum:
        最大观测值。
    """

    count: int
    mean: float
    std: float
    ci_low: float
    ci_high: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class MonteCarloRunRecord:
    """
    一个方案在一次随机种子实验中的结果。
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

    failover_batches: int
    hot_failover_function_stages: int
    cold_failover_function_stages: int

    total_cold_backup_startup_delay_ms: float


@dataclass(frozen=True)
class MonteCarloScenarioSummary:
    """
    一个方案经过多次实验后的统计结果。
    """

    scenario_name: str
    run_count: int

    request_success_rate: MetricSummary
    sla_violation_rate: MetricSummary

    average_successful_batch_delay_ms: MetricSummary
    p95_successful_batch_delay_ms: MetricSummary

    average_active_memory_mb: MetricSummary
    peak_active_memory_mb: MetricSummary

    failover_batches: MetricSummary
    hot_failover_function_stages: MetricSummary
    cold_failover_function_stages: MetricSummary

    total_cold_backup_startup_delay_ms: MetricSummary


@dataclass(frozen=True)
class MonteCarloExperimentResult:
    """
    完整蒙特卡洛实验结果。
    """

    confidence_level: float

    run_records: tuple[
        MonteCarloRunRecord,
        ...,
    ]

    scenario_summaries: tuple[
        MonteCarloScenarioSummary,
        ...,
    ]


def summarize_metric(
    values: list[float] | tuple[float, ...],
    confidence_level: float = 0.95,
) -> MetricSummary:
    """
    计算一个指标的统计量。

    置信区间采用正态近似：

        均值 ± z × 标准差 / sqrt(n)

    其中：

        confidence_level=0.95
        z≈1.96

    正式论文实验中，样本数量建议不少于30。
    """

    if not 0 < confidence_level < 1:
        raise ValueError(
            "confidence_level 必须位于 (0, 1)。"
        )

    numeric_values = tuple(
        float(value)
        for value in values
    )

    if len(numeric_values) == 0:
        raise ValueError(
            "统计指标不能为空。"
        )

    if any(
        not isfinite(value)
        for value in numeric_values
    ):
        raise ValueError(
            "统计指标必须是有限数值。"
        )

    sample_count = len(numeric_values)

    sample_mean = float(
        mean(numeric_values)
    )

    if sample_count > 1:
        sample_std = float(
            stdev(numeric_values)
        )
    else:
        sample_std = 0.0

    z_value = NormalDist().inv_cdf(
        0.5 + confidence_level / 2.0
    )

    confidence_margin = (
        z_value
        * sample_std
        / sqrt(sample_count)
    )

    return MetricSummary(
        count=sample_count,
        mean=sample_mean,
        std=sample_std,
        ci_low=(
            sample_mean
            - confidence_margin
        ),
        ci_high=(
            sample_mean
            + confidence_margin
        ),
        minimum=min(numeric_values),
        maximum=max(numeric_values),
    )


def run_paired_monte_carlo(
    scenario_builders: dict[
        str,
        Callable[
            [int],
            MonteCarloSimulatorProtocol,
        ],
    ],
    num_runs: int,
    base_seed: int,
    confidence_level: float = 0.95,
) -> MonteCarloExperimentResult:
    """
    执行配对随机种子蒙特卡洛实验。

    Parameters
    ----------
    scenario_builders:
        方案名称到仿真器构建函数的映射。

        构建函数输入一个随机种子，
        返回一个全新的仿真器。

    num_runs:
        每个方案运行次数。

    base_seed:
        第0次实验使用的种子。

        第k次实验种子：

            base_seed + k

    confidence_level:
        置信水平。

    Returns
    -------
    MonteCarloExperimentResult:
        原始逐次结果和统计汇总结果。
    """

    if len(scenario_builders) == 0:
        raise ValueError(
            "至少需要配置一个对比方案。"
        )

    if num_runs <= 0:
        raise ValueError(
            "蒙特卡洛运行次数必须大于0。"
        )

    if not isinstance(base_seed, int):
        raise TypeError(
            "base_seed 必须是整数。"
        )

    if not 0 < confidence_level < 1:
        raise ValueError(
            "confidence_level 必须位于 (0, 1)。"
        )

    run_records: list[
        MonteCarloRunRecord
    ] = []

    # 外层遍历随机种子，
    # 内层遍历方案。
    #
    # 这样能够明确保证每一个随机种子
    # 都被所有方案共同使用。
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
                MonteCarloRunRecord(
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
                    failover_batches=(
                        summary.failover_batches
                    ),
                    hot_failover_function_stages=(
                        summary
                        .hot_failover_function_stages
                    ),
                    cold_failover_function_stages=(
                        summary
                        .cold_failover_function_stages
                    ),
                    total_cold_backup_startup_delay_ms=(
                        summary
                        .total_cold_backup_startup_delay_ms
                    ),
                )
            )

    scenario_summaries: list[
        MonteCarloScenarioSummary
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
                f"方案 {scenario_name} 的实验数量异常。"
            )

        scenario_summaries.append(
            MonteCarloScenarioSummary(
                scenario_name=scenario_name,
                run_count=num_runs,
                request_success_rate=summarize_metric(
                    [
                        record.request_success_rate
                        for record
                        in scenario_records
                    ],
                    confidence_level,
                ),
                sla_violation_rate=summarize_metric(
                    [
                        record.sla_violation_rate
                        for record
                        in scenario_records
                    ],
                    confidence_level,
                ),
                average_successful_batch_delay_ms=(
                    summarize_metric(
                        [
                            record
                            .average_successful_batch_delay_ms
                            for record
                            in scenario_records
                        ],
                        confidence_level,
                    )
                ),
                p95_successful_batch_delay_ms=(
                    summarize_metric(
                        [
                            record
                            .p95_successful_batch_delay_ms
                            for record
                            in scenario_records
                        ],
                        confidence_level,
                    )
                ),
                average_active_memory_mb=summarize_metric(
                    [
                        record.average_active_memory_mb
                        for record
                        in scenario_records
                    ],
                    confidence_level,
                ),
                peak_active_memory_mb=summarize_metric(
                    [
                        record.peak_active_memory_mb
                        for record
                        in scenario_records
                    ],
                    confidence_level,
                ),
                failover_batches=summarize_metric(
                    [
                        record.failover_batches
                        for record
                        in scenario_records
                    ],
                    confidence_level,
                ),
                hot_failover_function_stages=(
                    summarize_metric(
                        [
                            record
                            .hot_failover_function_stages
                            for record
                            in scenario_records
                        ],
                        confidence_level,
                    )
                ),
                cold_failover_function_stages=(
                    summarize_metric(
                        [
                            record
                            .cold_failover_function_stages
                            for record
                            in scenario_records
                        ],
                        confidence_level,
                    )
                ),
                total_cold_backup_startup_delay_ms=(
                    summarize_metric(
                        [
                            record
                            .total_cold_backup_startup_delay_ms
                            for record
                            in scenario_records
                        ],
                        confidence_level,
                    )
                ),
            )
        )

    return MonteCarloExperimentResult(
        confidence_level=confidence_level,
        run_records=tuple(run_records),
        scenario_summaries=tuple(
            scenario_summaries
        ),
    )