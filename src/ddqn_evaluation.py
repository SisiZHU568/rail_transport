"""
ddqn_evaluation.py

Double DQN独立测试集评估工具。

功能：

1. 在固定测试种子下运行一种策略；
2. 统计每个Episode的奖励、可靠性、时延和资源开销；
3. 计算均值、标准差和置信区间；
4. 统计四种动作的使用比例。
"""

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from math import sqrt
from statistics import NormalDist
from typing import Any

import numpy as np


PolicyFunction = Callable[
    [np.ndarray, dict[str, Any]],
    int,
]


@dataclass(frozen=True)
class EpisodeEvaluationRecord:
    """
    一种策略在一个测试Episode中的结果。
    """

    policy_name: str
    episode_index: int
    episode_seed: int

    episode_reward: float
    decision_count: int

    total_requests: float
    successful_requests: float
    request_success_rate: float

    request_batches: float
    sla_violation_batches: float
    sla_violation_rate: float

    average_delay_ms: float
    average_memory_mb: float

    total_cold_start_delay_ms: float
    reconfiguration_count: float

    single_count: int
    cold_count: int
    hot_count: int
    dynamic_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConfidenceSummary:
    """
    一个指标的统计结果。
    """

    sample_count: int
    mean: float
    standard_deviation: float
    confidence_interval_low: float
    confidence_interval_high: float


def _read_number(
    source: Any,
    names: Sequence[str],
    default: float | None = None,
) -> float:
    """
    从对象或字典中读取第一个存在的数值字段。

    设置多个候选名称，是为了兼容项目中可能存在的
    singular/plural命名差异。
    """

    for name in names:
        if isinstance(source, dict):
            if name in source:
                return float(source[name])

        elif hasattr(source, name):
            return float(
                getattr(source, name)
            )

    if default is None:
        raise AttributeError(
            "没有找到指标字段："
            + ", ".join(names)
        )

    return float(default)


def build_fixed_policy(
    action: int,
) -> PolicyFunction:
    """
    创建固定动作策略。
    """

    fixed_action = int(action)

    def policy(
        state: np.ndarray,
        observation_info: dict[str, Any],
    ) -> int:
        del state
        del observation_info

        return fixed_action

    return policy


def build_rule_based_policy(
    high_risk_threshold: float,
) -> PolicyFunction:
    """
    创建与环境演示程序一致的人工规则策略。

    高风险：
        HOT

    普通风险：
        DYNAMIC
    """

    if not 0.0 <= high_risk_threshold <= 1.0:
        raise ValueError(
            "high_risk_threshold必须位于[0,1]。"
        )

    def policy(
        state: np.ndarray,
        observation_info: dict[str, Any],
    ) -> int:
        del state

        failure_risk = float(
            observation_info[
                "predicted_failure_risk"
            ]
        )

        if failure_risk >= high_risk_threshold:
            # HOT
            return 2

        # DYNAMIC
        return 3

    return policy


def summarize_values(
    values: Sequence[float],
    confidence_level: float = 0.95,
) -> ConfidenceSummary:
    """
    计算均值、样本标准差和双侧置信区间。

    使用正态近似：

        mean ± z * std / sqrt(n)
    """

    value_array = np.asarray(
        values,
        dtype=np.float64,
    )

    if value_array.ndim != 1:
        raise ValueError(
            "values必须是一维序列。"
        )

    if len(value_array) == 0:
        raise ValueError(
            "values不能为空。"
        )

    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "confidence_level必须位于(0,1)。"
        )

    sample_count = len(value_array)

    mean = float(
        np.mean(value_array)
    )

    if sample_count == 1:
        standard_deviation = 0.0
        half_width = 0.0

    else:
        standard_deviation = float(
            np.std(
                value_array,
                ddof=1,
            )
        )

        quantile_probability = (
            1.0
            -
            (
                1.0 - confidence_level
            )
            / 2.0
        )

        z_value = NormalDist().inv_cdf(
            quantile_probability
        )

        standard_error = (
            standard_deviation
            / sqrt(sample_count)
        )

        half_width = (
            z_value
            * standard_error
        )

    return ConfidenceSummary(
        sample_count=sample_count,
        mean=mean,
        standard_deviation=(
            standard_deviation
        ),
        confidence_interval_low=(
            mean - half_width
        ),
        confidence_interval_high=(
            mean + half_width
        ),
    )


def evaluate_policy(
    environment: Any,
    policy_name: str,
    policy: PolicyFunction,
    episode_seeds: Sequence[int],
) -> list[EpisodeEvaluationRecord]:
    """
    在指定测试种子上评估一种策略。

    同一组episode_seeds会用于所有策略，
    从而保证故障轨迹和工作负载一致。
    """

    records: list[
        EpisodeEvaluationRecord
    ] = []

    for episode_index, episode_seed in enumerate(
        episode_seeds,
        start=1,
    ):
        state, observation_info = (
            environment.reset(
                seed=int(episode_seed)
            )
        )

        episode_reward = 0.0
        decision_count = 0

        action_counts = [
            0,
            0,
            0,
            0,
        ]

        total_requests = 0.0
        successful_requests = 0.0

        request_batches = 0.0
        sla_violation_batches = 0.0

        delay_sum_ms = 0.0
        delay_weight = 0.0

        memory_slot_sum = 0.0
        memory_slot_count = 0.0

        total_cold_start_delay_ms = 0.0
        reconfiguration_count = 0.0

        terminated = False
        truncated = False

        while not (
            terminated or truncated
        ):
            action = int(
                policy(
                    state,
                    observation_info,
                )
            )

            if not (
                0
                <= action
                < environment.action_count
            ):
                raise ValueError(
                    f"策略{policy_name}返回了"
                    f"非法动作：{action}。"
                )

            action_counts[action] += 1

            (
                next_state,
                reward,
                terminated,
                truncated,
                step_info,
            ) = environment.step(action)

            metrics = step_info["metrics"]

            window_length = float(
                step_info["window_length"]
            )

            # ----------------------------------------
            # 请求成功率统计
            # ----------------------------------------

            window_requests = _read_number(
                metrics,
                names=(
                    "total_requests",
                    "request_count",
                ),
                default=0.0,
            )

            window_success_rate = _read_number(
                metrics,
                names=(
                    "request_success_rate",
                ),
                default=1.0,
            )

            window_successful_requests = (
                _read_number(
                    metrics,
                    names=(
                        "successful_requests",
                        "successful_request_count",
                    ),
                    default=(
                        window_success_rate
                        * window_requests
                    ),
                )
            )

            total_requests += window_requests

            successful_requests += (
                window_successful_requests
            )

            # ----------------------------------------
            # SLA违反率统计
            # ----------------------------------------

            window_sla_rate = _read_number(
                metrics,
                names=(
                    "sla_violation_rate",
                ),
                default=0.0,
            )

            window_batches = _read_number(
                metrics,
                names=(
                    "request_batches",
                    "request_batch_count",
                    "total_request_batches",
                    "total_batches",
                ),
                # 如果当前实现没有公开批次数，
                # 则使用请求数作为权重。
                default=window_requests,
            )

            window_sla_violations = (
                _read_number(
                    metrics,
                    names=(
                        "sla_violation_batches",
                        "sla_violation_batch_count",
                        "sla_violations",
                    ),
                    default=(
                        window_sla_rate
                        * window_batches
                    ),
                )
            )

            request_batches += window_batches

            sla_violation_batches += (
                window_sla_violations
            )

            # ----------------------------------------
            # 成功请求时延统计
            # ----------------------------------------

            window_average_delay_ms = (
                _read_number(
                    metrics,
                    names=(
                        "average_successful_delay_ms",
                        "average_delay_ms",
                    ),
                    default=0.0,
                )
            )

            window_successful_batches = (
                _read_number(
                    metrics,
                    names=(
                        "successful_batches",
                        "successful_batch_count",
                    ),
                    default=(
                        window_successful_requests
                    ),
                )
            )

            window_delay_sum_ms = _read_number(
                metrics,
                names=(
                    "total_successful_delay_ms",
                    "successful_delay_sum_ms",
                ),
                default=(
                    window_average_delay_ms
                    * window_successful_batches
                ),
            )

            delay_sum_ms += window_delay_sum_ms
            delay_weight += window_successful_batches

            # ----------------------------------------
            # 内存、冷启动和重配置
            # ----------------------------------------

            window_average_memory_mb = (
                _read_number(
                    metrics,
                    names=(
                        "average_active_memory_mb",
                        "average_memory_mb",
                    ),
                    default=0.0,
                )
            )

            memory_slot_sum += (
                window_average_memory_mb
                * window_length
            )

            memory_slot_count += window_length

            total_cold_start_delay_ms += (
                _read_number(
                    metrics,
                    names=(
                        "total_cold_start_delay_ms",
                        "cold_start_delay_ms",
                    ),
                    default=0.0,
                )
            )

            reconfiguration_count += (
                _read_number(
                    metrics,
                    names=(
                        "reconfiguration_count",
                        "replica_reconfiguration_count",
                        "reconfiguration_events",
                    ),
                    default=_read_number(
                        step_info,
                        names=(
                            "reconfiguration_count",
                            "replica_reconfiguration_count",
                        ),
                        default=0.0,
                    ),
                )
            )

            episode_reward += float(reward)
            decision_count += 1

            state = next_state

            if not (
                terminated or truncated
            ):
                if (
                    "next_observation"
                    not in step_info
                ):
                    raise KeyError(
                        "step_info缺少"
                        "next_observation。"
                    )

                observation_info = (
                    step_info[
                        "next_observation"
                    ]
                )

        request_success_rate = (
            successful_requests
            / total_requests
            if total_requests > 0.0
            else 1.0
        )

        sla_violation_rate = (
            sla_violation_batches
            / request_batches
            if request_batches > 0.0
            else 0.0
        )

        average_delay_ms = (
            delay_sum_ms
            / delay_weight
            if delay_weight > 0.0
            else 0.0
        )

        average_memory_mb = (
            memory_slot_sum
            / memory_slot_count
            if memory_slot_count > 0.0
            else 0.0
        )

        records.append(
            EpisodeEvaluationRecord(
                policy_name=policy_name,
                episode_index=episode_index,
                episode_seed=int(
                    episode_seed
                ),
                episode_reward=float(
                    episode_reward
                ),
                decision_count=(
                    decision_count
                ),
                total_requests=(
                    total_requests
                ),
                successful_requests=(
                    successful_requests
                ),
                request_success_rate=(
                    request_success_rate
                ),
                request_batches=(
                    request_batches
                ),
                sla_violation_batches=(
                    sla_violation_batches
                ),
                sla_violation_rate=(
                    sla_violation_rate
                ),
                average_delay_ms=(
                    average_delay_ms
                ),
                average_memory_mb=(
                    average_memory_mb
                ),
                total_cold_start_delay_ms=(
                    total_cold_start_delay_ms
                ),
                reconfiguration_count=(
                    reconfiguration_count
                ),
                single_count=(
                    action_counts[0]
                ),
                cold_count=(
                    action_counts[1]
                ),
                hot_count=(
                    action_counts[2]
                ),
                dynamic_count=(
                    action_counts[3]
                ),
            )
        )

    return records


SUMMARY_METRICS = (
    "episode_reward",
    "request_success_rate",
    "sla_violation_rate",
    "average_delay_ms",
    "average_memory_mb",
    "total_cold_start_delay_ms",
    "reconfiguration_count",
)


def build_summary_rows(
    records: Sequence[
        EpisodeEvaluationRecord
    ],
    confidence_level: float,
) -> list[dict[str, Any]]:
    """
    按策略和指标生成长表格式的统计结果。
    """

    policy_names = list(
        dict.fromkeys(
            record.policy_name
            for record in records
        )
    )

    rows: list[dict[str, Any]] = []

    for policy_name in policy_names:
        policy_records = [
            record
            for record in records
            if record.policy_name
            == policy_name
        ]

        for metric_name in SUMMARY_METRICS:
            values = [
                float(
                    getattr(
                        record,
                        metric_name,
                    )
                )
                for record in policy_records
            ]

            summary = summarize_values(
                values=values,
                confidence_level=(
                    confidence_level
                ),
            )

            rows.append(
                {
                    "policy_name": (
                        policy_name
                    ),
                    "metric_name": (
                        metric_name
                    ),
                    "sample_count": (
                        summary.sample_count
                    ),
                    "mean": summary.mean,
                    "standard_deviation": (
                        summary
                        .standard_deviation
                    ),
                    "confidence_interval_low": (
                        summary
                        .confidence_interval_low
                    ),
                    "confidence_interval_high": (
                        summary
                        .confidence_interval_high
                    ),
                }
            )

    return rows


def build_action_distribution_rows(
    records: Sequence[
        EpisodeEvaluationRecord
    ],
) -> list[dict[str, Any]]:
    """
    统计每种策略的动作使用比例。
    """

    policy_names = list(
        dict.fromkeys(
            record.policy_name
            for record in records
        )
    )

    rows: list[dict[str, Any]] = []

    for policy_name in policy_names:
        policy_records = [
            record
            for record in records
            if record.policy_name
            == policy_name
        ]

        single_count = sum(
            record.single_count
            for record in policy_records
        )

        cold_count = sum(
            record.cold_count
            for record in policy_records
        )

        hot_count = sum(
            record.hot_count
            for record in policy_records
        )

        dynamic_count = sum(
            record.dynamic_count
            for record in policy_records
        )

        total_count = (
            single_count
            + cold_count
            + hot_count
            + dynamic_count
        )

        denominator = max(
            total_count,
            1,
        )

        rows.append(
            {
                "policy_name": policy_name,
                "total_decisions": (
                    total_count
                ),
                "single_count": (
                    single_count
                ),
                "cold_count": cold_count,
                "hot_count": hot_count,
                "dynamic_count": (
                    dynamic_count
                ),
                "single_ratio": (
                    single_count
                    / denominator
                ),
                "cold_ratio": (
                    cold_count
                    / denominator
                ),
                "hot_ratio": (
                    hot_count
                    / denominator
                ),
                "dynamic_ratio": (
                    dynamic_count
                    / denominator
                ),
            }
        )

    return rows