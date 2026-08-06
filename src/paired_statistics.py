"""
paired_statistics.py

对相同测试种子下的策略结果进行配对统计分析。

统计方法：

1. Bootstrap均值置信区间；
2. 配对符号翻转检验；
3. Holm多重比较校正；
4. 配对效应量Cohen's dz。

统一规定：

    improvement > 0：Double DQN更好
    improvement < 0：Double DQN更差

对于Reward和Success，数值越大越好。

对于SLA违反率、时延、内存、冷启动时延和重配置次数，
数值越小越好。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


# 各指标的优化方向。
METRIC_DIRECTIONS: dict[str, str] = {
    "episode_reward": "higher",
    "request_success_rate": "higher",
    "sla_violation_rate": "lower",
    "average_delay_ms": "lower",
    "average_memory_mb": "lower",
    "total_cold_start_delay_ms": "lower",
    "reconfiguration_count": "lower",
}


@dataclass(frozen=True)
class MeanConfidenceInterval:
    """
    均值及置信区间。
    """

    mean: float
    low: float
    high: float


def _as_float_array(
    values: Sequence[float],
    name: str,
) -> np.ndarray:
    """
    将输入转换为一维浮点数组。
    """

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.ndim != 1:
        raise ValueError(
            f"{name}必须是一维序列。"
        )

    if len(array) == 0:
        raise ValueError(
            f"{name}不能为空。"
        )

    if not np.all(
        np.isfinite(array)
    ):
        raise ValueError(
            f"{name}中存在NaN或无穷值。"
        )

    return array


def calculate_improvements(
    reference_values: Sequence[float],
    baseline_values: Sequence[float],
    direction: str,
) -> np.ndarray:
    """
    计算参考策略相对于基线的改进值。

    direction == "higher"：

        improvement = reference - baseline

    direction == "lower"：

        improvement = baseline - reference

    因此无论哪种指标：

        improvement > 0
        都表示参考策略更好。
    """

    reference_array = _as_float_array(
        reference_values,
        "reference_values",
    )

    baseline_array = _as_float_array(
        baseline_values,
        "baseline_values",
    )

    if (
        reference_array.shape
        != baseline_array.shape
    ):
        raise ValueError(
            "参考策略和基线策略的样本数量不一致。"
        )

    if direction == "higher":
        return (
            reference_array
            - baseline_array
        )

    if direction == "lower":
        return (
            baseline_array
            - reference_array
        )

    raise ValueError(
        "direction只能为higher或lower。"
    )


def bootstrap_mean_confidence_interval(
    values: Sequence[float],
    confidence_level: float,
    num_samples: int,
    random_generator: np.random.Generator,
) -> MeanConfidenceInterval:
    """
    使用Bootstrap计算均值置信区间。
    """

    value_array = _as_float_array(
        values,
        "values",
    )

    if not (
        0.0
        < confidence_level
        < 1.0
    ):
        raise ValueError(
            "confidence_level必须位于(0,1)。"
        )

    if num_samples <= 0:
        raise ValueError(
            "num_samples必须大于0。"
        )

    sample_count = len(
        value_array
    )

    sampled_indices = (
        random_generator.integers(
            low=0,
            high=sample_count,
            size=(
                num_samples,
                sample_count,
            ),
        )
    )

    bootstrap_means = np.mean(
        value_array[
            sampled_indices
        ],
        axis=1,
    )

    tail_probability = (
        1.0 - confidence_level
    ) / 2.0

    low = float(
        np.quantile(
            bootstrap_means,
            tail_probability,
        )
    )

    high = float(
        np.quantile(
            bootstrap_means,
            1.0 - tail_probability,
        )
    )

    return MeanConfidenceInterval(
        mean=float(
            np.mean(value_array)
        ),
        low=low,
        high=high,
    )


def paired_sign_flip_p_value(
    paired_differences: Sequence[float],
    num_samples: int,
    random_generator: np.random.Generator,
    exact_sample_limit: int = 16,
) -> float:
    """
    对配对差值进行双侧符号翻转检验。

    原假设：

        配对改进值的均值为0。

    小样本时枚举所有符号组合；
    大样本时采用Monte Carlo符号翻转。
    """

    differences = _as_float_array(
        paired_differences,
        "paired_differences",
    )

    if num_samples <= 0:
        raise ValueError(
            "num_samples必须大于0。"
        )

    observed_statistic = abs(
        float(
            np.mean(differences)
        )
    )

    tolerance = 1e-15

    if observed_statistic <= tolerance:
        return 1.0

    sample_count = len(
        differences
    )

    # 小样本进行精确枚举。
    if sample_count <= exact_sample_limit:
        total_combinations = (
            1 << sample_count
        )

        extreme_count = 0

        for mask in range(
            total_combinations
        ):
            signs = np.empty(
                sample_count,
                dtype=np.float64,
            )

            for index in range(
                sample_count
            ):
                if (
                    mask
                    & (1 << index)
                ):
                    signs[index] = 1.0

                else:
                    signs[index] = -1.0

            permuted_statistic = abs(
                float(
                    np.mean(
                        signs
                        * differences
                    )
                )
            )

            if (
                permuted_statistic
                >= observed_statistic
                - tolerance
            ):
                extreme_count += 1

        return (
            extreme_count
            / total_combinations
        )

    # 大样本进行Monte Carlo符号翻转。
    random_bits = (
        random_generator.integers(
            low=0,
            high=2,
            size=(
                num_samples,
                sample_count,
            ),
            dtype=np.int8,
        )
    )

    random_signs = (
        random_bits.astype(
            np.float64
        )
        * 2.0
        - 1.0
    )

    permuted_means = np.mean(
        random_signs
        * differences[
            np.newaxis,
            :
        ],
        axis=1,
    )

    extreme_count = int(
        np.count_nonzero(
            np.abs(
                permuted_means
            )
            >= observed_statistic
            - tolerance
        )
    )

    # 加1修正，避免Monte Carlo p值等于0。
    return (
        extreme_count + 1
    ) / (
        num_samples + 1
    )


def paired_effect_size_dz(
    paired_differences: Sequence[float],
) -> float:
    """
    计算配对效应量Cohen's dz。

        dz = mean(difference)
             / std(difference)
    """

    differences = _as_float_array(
        paired_differences,
        "paired_differences",
    )

    if len(differences) == 1:
        return 0.0

    mean_difference = float(
        np.mean(differences)
    )

    standard_deviation = float(
        np.std(
            differences,
            ddof=1,
        )
    )

    tolerance = 1e-12

    if standard_deviation <= tolerance:
        if abs(
            mean_difference
        ) <= tolerance:
            return 0.0

        return float(
            np.sign(
                mean_difference
            )
            * np.inf
        )

    return (
        mean_difference
        / standard_deviation
    )


def holm_adjust_p_values(
    p_values: Sequence[float],
) -> list[float]:
    """
    使用Holm方法校正一组p值。

    返回顺序与输入顺序一致。
    """

    p_value_array = _as_float_array(
        p_values,
        "p_values",
    )

    if np.any(
        p_value_array < 0.0
    ) or np.any(
        p_value_array > 1.0
    ):
        raise ValueError(
            "p值必须位于[0,1]。"
        )

    comparison_count = len(
        p_value_array
    )

    sorted_indices = np.argsort(
        p_value_array
    )

    adjusted_values = np.empty(
        comparison_count,
        dtype=np.float64,
    )

    running_maximum = 0.0

    for rank, original_index in enumerate(
        sorted_indices
    ):
        correction_factor = (
            comparison_count
            - rank
        )

        candidate = (
            correction_factor
            * float(
                p_value_array[
                    original_index
                ]
            )
        )

        running_maximum = max(
            running_maximum,
            candidate,
        )

        adjusted_values[
            original_index
        ] = min(
            running_maximum,
            1.0,
        )

    return [
        float(value)
        for value
        in adjusted_values
    ]


def _group_records_by_policy_and_seed(
    records: Sequence[
        Mapping[str, Any]
    ],
) -> tuple[
    list[str],
    dict[
        str,
        dict[int, Mapping[str, Any]],
    ],
]:
    """
    按策略和测试种子组织Episode结果。
    """

    policy_order: list[str] = []

    grouped: dict[
        str,
        dict[int, Mapping[str, Any]],
    ] = {}

    for record in records:
        policy_name = str(
            record["policy_name"]
        )

        episode_seed = int(
            float(
                record["episode_seed"]
            )
        )

        if policy_name not in grouped:
            grouped[
                policy_name
            ] = {}

            policy_order.append(
                policy_name
            )

        if (
            episode_seed
            in grouped[policy_name]
        ):
            raise ValueError(
                f"策略{policy_name}中存在重复测试种子："
                f"{episode_seed}。"
            )

        grouped[
            policy_name
        ][episode_seed] = record

    return policy_order, grouped


def build_paired_comparison_rows(
    records: Sequence[
        Mapping[str, Any]
    ],
    reference_policy: str = "Double-DQN",
    baseline_policies: Sequence[str] | None = None,
    metric_names: Sequence[str] | None = None,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 20000,
    permutation_samples: int = 20000,
    random_seed: int = 20260804,
    significance_level: float = 0.05,
) -> list[dict[str, Any]]:
    """
    构建参考策略与各基线之间的配对比较表。
    """

    if len(records) == 0:
        raise ValueError(
            "records不能为空。"
        )

    policy_order, grouped = (
        _group_records_by_policy_and_seed(
            records
        )
    )

    if reference_policy not in grouped:
        raise KeyError(
            f"没有找到参考策略："
            f"{reference_policy}。"
        )

    if baseline_policies is None:
        selected_baselines = [
            policy_name
            for policy_name
            in policy_order
            if policy_name
            != reference_policy
        ]

    else:
        selected_baselines = [
            str(policy_name)
            for policy_name
            in baseline_policies
        ]

    for baseline_policy in (
        selected_baselines
    ):
        if baseline_policy not in grouped:
            raise KeyError(
                f"没有找到基线策略："
                f"{baseline_policy}。"
            )

    if metric_names is None:
        selected_metrics = list(
            METRIC_DIRECTIONS.keys()
        )

    else:
        selected_metrics = [
            str(metric_name)
            for metric_name
            in metric_names
        ]

    for metric_name in selected_metrics:
        if metric_name not in (
            METRIC_DIRECTIONS
        ):
            raise KeyError(
                f"没有定义指标{metric_name}"
                "的优化方向。"
            )

    reference_seeds = set(
        grouped[
            reference_policy
        ].keys()
    )

    rows: list[
        dict[str, Any]
    ] = []

    for metric_index, metric_name in enumerate(
        selected_metrics
    ):
        metric_rows: list[
            dict[str, Any]
        ] = []

        direction = (
            METRIC_DIRECTIONS[
                metric_name
            ]
        )

        for baseline_index, baseline_policy in enumerate(
            selected_baselines
        ):
            baseline_seeds = set(
                grouped[
                    baseline_policy
                ].keys()
            )

            if (
                baseline_seeds
                != reference_seeds
            ):
                missing_in_baseline = sorted(
                    reference_seeds
                    - baseline_seeds
                )

                missing_in_reference = sorted(
                    baseline_seeds
                    - reference_seeds
                )

                raise ValueError(
                    f"{reference_policy}与"
                    f"{baseline_policy}的"
                    "测试种子不一致。"
                    f"基线缺少：{missing_in_baseline}；"
                    f"参考策略缺少：{missing_in_reference}。"
                )

            sorted_seeds = sorted(
                reference_seeds
            )

            reference_values = np.asarray(
                [
                    float(
                        grouped[
                            reference_policy
                        ][seed][
                            metric_name
                        ]
                    )
                    for seed
                    in sorted_seeds
                ],
                dtype=np.float64,
            )

            baseline_values = np.asarray(
                [
                    float(
                        grouped[
                            baseline_policy
                        ][seed][
                            metric_name
                        ]
                    )
                    for seed
                    in sorted_seeds
                ],
                dtype=np.float64,
            )

            improvements = (
                calculate_improvements(
                    reference_values=(
                        reference_values
                    ),
                    baseline_values=(
                        baseline_values
                    ),
                    direction=direction,
                )
            )

            # 每个比较使用独立且可复现的随机数生成器。
            child_seed = np.random.SeedSequence(
                [
                    random_seed,
                    metric_index,
                    baseline_index,
                ]
            )

            random_generator = (
                np.random.default_rng(
                    child_seed
                )
            )

            confidence_interval = (
                bootstrap_mean_confidence_interval(
                    values=improvements,
                    confidence_level=(
                        confidence_level
                    ),
                    num_samples=(
                        bootstrap_samples
                    ),
                    random_generator=(
                        random_generator
                    ),
                )
            )

            raw_p_value = (
                paired_sign_flip_p_value(
                    paired_differences=(
                        improvements
                    ),
                    num_samples=(
                        permutation_samples
                    ),
                    random_generator=(
                        random_generator
                    ),
                )
            )

            effect_size = (
                paired_effect_size_dz(
                    improvements
                )
            )

            tolerance = 1e-12

            win_count = int(
                np.count_nonzero(
                    improvements
                    > tolerance
                )
            )

            tie_count = int(
                np.count_nonzero(
                    np.abs(
                        improvements
                    )
                    <= tolerance
                )
            )

            loss_count = int(
                np.count_nonzero(
                    improvements
                    < -tolerance
                )
            )

            raw_difference = (
                reference_values
                - baseline_values
            )

            metric_rows.append(
                {
                    "reference_policy": (
                        reference_policy
                    ),
                    "baseline_policy": (
                        baseline_policy
                    ),
                    "metric_name": (
                        metric_name
                    ),
                    "optimization_direction": (
                        direction
                    ),
                    "sample_count": len(
                        sorted_seeds
                    ),
                    "reference_mean": float(
                        np.mean(
                            reference_values
                        )
                    ),
                    "baseline_mean": float(
                        np.mean(
                            baseline_values
                        )
                    ),
                    "raw_difference_mean": float(
                        np.mean(
                            raw_difference
                        )
                    ),
                    "improvement_mean": (
                        confidence_interval.mean
                    ),
                    "improvement_ci_low": (
                        confidence_interval.low
                    ),
                    "improvement_ci_high": (
                        confidence_interval.high
                    ),
                    "ci_excludes_zero": bool(
                        confidence_interval.low
                        > 0.0
                        or confidence_interval.high
                        < 0.0
                    ),
                    "effect_size_dz": (
                        effect_size
                    ),
                    "win_count": win_count,
                    "tie_count": tie_count,
                    "loss_count": loss_count,
                    "raw_p_value": (
                        raw_p_value
                    ),
                }
            )

        # 同一指标下，对不同基线的p值做Holm校正。
        adjusted_p_values = (
            holm_adjust_p_values(
                [
                    float(
                        row[
                            "raw_p_value"
                        ]
                    )
                    for row
                    in metric_rows
                ]
            )
        )

        for row, adjusted_p_value in zip(
            metric_rows,
            adjusted_p_values,
            strict=True,
        ):
            improvement_mean = float(
                row[
                    "improvement_mean"
                ]
            )

            significant = bool(
                adjusted_p_value
                < significance_level
            )

            if not significant:
                conclusion = (
                    "差异不显著"
                )

            elif improvement_mean > 0.0:
                conclusion = (
                    "Double-DQN显著更优"
                )

            elif improvement_mean < 0.0:
                conclusion = (
                    "Double-DQN显著更差"
                )

            else:
                conclusion = (
                    "差异不显著"
                )

            row[
                "holm_adjusted_p_value"
            ] = adjusted_p_value

            row[
                "significant_0_05"
            ] = significant

            row[
                "conclusion"
            ] = conclusion

        rows.extend(
            metric_rows
        )

    return rows