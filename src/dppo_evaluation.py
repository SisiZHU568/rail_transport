"""实现 DDPO 的无梯度独立 Episode 评估与跨种子统计汇总。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from types import MappingProxyType

import numpy as np
import torch

from src.dppo import DPPOAgent
from src.dppo_slow_timescale_env import DPPOSlowTimescaleEnvironment


REQUIRED_EVALUATION_COLUMNS = frozenset(
    {
        "success_rate",
        "sla_success_rate",
        "mean_delay_ms",
        "peak_delay_ms",
        "p95_delay_ms",
        "p99_delay_ms",
        "mean_reliability",
        "cold_start_count",
        "mean_active_memory_mb",
        "cloud_replica_ratio",
        "mean_primary_retention_s",
        "mean_backup_retention_s",
        "raw_feasibility_rate",
        "projection_change_ratio",
        "repair_attempts",
        "repair_successes",
        "repair_failures",
        "rejection_count",
        "run_cost",
        "route_cost",
        "cold_start_cost",
        "total_cost",
    }
)


@dataclass(frozen=True)
class DPPODecisionActionRecord:
    """保存一个慢决策中单个 VNF 的可解释动作，供分布表直接导出。"""

    episode_seed: int
    decision_index: int
    function_id: int
    replica_count: int
    cloud_selected: bool
    primary_retention_seconds: float
    backup_retention_seconds: float

    def __post_init__(self) -> None:
        for name in ("episode_seed", "decision_index", "function_id"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} 必须是非负整数。")
        if self.replica_count not in {2, 3}:
            raise ValueError("replica_count 必须是 2 或 3。")
        if not isinstance(self.cloud_selected, bool):
            raise ValueError("cloud_selected 必须是布尔值。")
        for name in (
            "primary_retention_seconds",
            "backup_retention_seconds",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} 必须是非负有限数。")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class DPPOEpisodeEvaluation:
    """保存一个固定种子 Episode 的性能、原始时延和动作分布。"""

    episode_seed: int
    metrics: Mapping[str, float]
    fast_slot_delay_samples_ms: tuple[float, ...]
    replica_counts_by_function: Mapping[int, float]
    action_records: tuple[DPPODecisionActionRecord, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.episode_seed, bool)
            or not isinstance(self.episode_seed, int)
            or self.episode_seed < 0
        ):
            raise ValueError("episode_seed 必须是非负整数。")
        copied_metrics = {str(key): float(value) for key, value in self.metrics.items()}
        if not REQUIRED_EVALUATION_COLUMNS <= copied_metrics.keys():
            missing = sorted(REQUIRED_EVALUATION_COLUMNS - copied_metrics.keys())
            raise ValueError(f"Episode 评估缺少指标：{missing}。")
        if not all(math.isfinite(value) for value in copied_metrics.values()):
            raise ValueError("Episode 评估指标必须全部为有限数。")
        delay_samples = tuple(float(value) for value in self.fast_slot_delay_samples_ms)
        if any(not math.isfinite(value) or value < 0.0 for value in delay_samples):
            raise ValueError("快时隙时延样本必须是非负有限数。")
        replica_counts = {
            int(function_id): float(value)
            for function_id, value in self.replica_counts_by_function.items()
        }
        if any(
            function_id < 0
            or not math.isfinite(value)
            or not 2.0 <= value <= 3.0
            for function_id, value in replica_counts.items()
        ):
            raise ValueError("每个 VNF 的平均副本数必须位于 [2, 3]。")
        records = tuple(self.action_records)
        if any(
            not isinstance(record, DPPODecisionActionRecord)
            or record.episode_seed != self.episode_seed
            for record in records
        ):
            raise ValueError("动作记录必须属于当前 Episode。")
        object.__setattr__(self, "metrics", MappingProxyType(copied_metrics))
        object.__setattr__(self, "fast_slot_delay_samples_ms", delay_samples)
        object.__setattr__(
            self,
            "replica_counts_by_function",
            MappingProxyType(replica_counts),
        )
        object.__setattr__(self, "action_records", records)


def delay_statistics(samples_ms: Sequence[float]) -> dict[str, float]:
    """直接在原始快时隙时延样本上计算均值、峰值、P95 和 P99。"""

    values = np.asarray(tuple(samples_ms), dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("samples_ms 必须是一维序列。")
    if values.size == 0:
        return {
            "mean_delay_ms": 0.0,
            "peak_delay_ms": 0.0,
            "p95_delay_ms": 0.0,
            "p99_delay_ms": 0.0,
        }
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("samples_ms 必须只包含非负有限数。")
    return {
        "mean_delay_ms": float(np.mean(values)),
        "peak_delay_ms": float(np.max(values)),
        "p95_delay_ms": float(np.percentile(values, 95)),
        "p99_delay_ms": float(np.percentile(values, 99)),
    }


def _safe_rate(numerator: float, denominator: float, *, empty: float) -> float:
    """计算比例，并显式规定没有观测时的语义。"""

    return float(numerator / denominator) if denominator > 0.0 else float(empty)


def _run_evaluation_episode(
    environment: DPPOSlowTimescaleEnvironment,
    agent: DPPOAgent,
    episode_seed: int,
) -> DPPOEpisodeEvaluation:
    """执行一次不建立计算图的完整线路 Episode。"""

    state = environment.reset(seed=episode_seed)
    cloud_node_id = environment.dimensions.cloud_node_id
    function_ids = environment.dimensions.function_ids
    replica_counts: dict[int, list[int]] = {
        function_id: [] for function_id in function_ids
    }
    action_records: list[DPPODecisionActionRecord] = []
    delay_samples: list[float] = []
    reliability_samples: list[float] = []
    rewards: list[float] = []

    total_requests = 0
    successful_requests = 0
    request_batches = 0
    sla_violations = 0
    cold_start_count = 0
    total_active_memory_mb_seconds = 0.0
    total_duration_seconds = 0.0
    cloud_used_slots = 0
    total_fast_slots = 0
    raw_feasible_count = 0
    projection_change_sum = 0.0
    repair_attempts = 0
    repair_successes = 0
    repair_failures = 0
    rejection_count = 0
    run_cost = 0.0
    route_cost = 0.0
    cold_start_cost = 0.0
    cloud_replica_count = 0
    requested_replica_count = 0
    decision_index = 0

    while True:
        sample = agent.sample_action(state, seed=episode_seed + decision_index)
        raw_action = sample.actions[-1, 0].detach().cpu().numpy()
        next_state, reward, terminated, truncated, info = environment.step(raw_action)
        rewards.append(float(reward))

        decoded_action = info["decoded_action"]
        for function_action in decoded_action.function_actions:
            selected_node_ids = function_action.ranked_node_ids[
                : function_action.replica_count
            ]
            cloud_selected = cloud_node_id in selected_node_ids
            replica_counts[function_action.function_id].append(
                function_action.replica_count
            )
            requested_replica_count += function_action.replica_count
            cloud_replica_count += int(cloud_selected)
            action_records.append(
                DPPODecisionActionRecord(
                    episode_seed=episode_seed,
                    decision_index=decision_index,
                    function_id=function_action.function_id,
                    replica_count=function_action.replica_count,
                    cloud_selected=cloud_selected,
                    primary_retention_seconds=(
                        function_action.primary_retention_seconds
                    ),
                    backup_retention_seconds=(
                        function_action.backup_retention_seconds
                    ),
                )
            )

        metrics = info["final_metrics"]
        window_length = int(info["window_length"])
        total_requests += metrics.total_requests
        successful_requests += metrics.successful_requests
        request_batches += metrics.request_batches
        sla_violations += metrics.sla_violations
        cold_start_count += metrics.cold_start_function_stages
        total_active_memory_mb_seconds += metrics.total_active_memory_mb_seconds
        total_duration_seconds += window_length * environment.execution_core.slot_seconds
        cloud_used_slots += metrics.cloud_used_slots
        total_fast_slots += window_length
        raw_feasible_count += int(bool(info["raw_feasible"]))
        projection_change_sum += float(info["projection_change_ratio"])
        repair_attempts += metrics.fast_repair_attempts
        repair_successes += metrics.fast_repair_successes
        repair_failures += metrics.fast_repair_failures
        rejection_count += int(bool(info["projection_rejected"]))
        run_cost += metrics.total_run_cost
        route_cost += metrics.total_route_cost
        cold_start_cost += metrics.total_cold_start_cost
        delay_samples.extend(float(value) for value in info["fast_slot_delay_samples_ms"])
        reliability_samples.extend(
            float(value) for value in info["exact_sfc_reliability_samples"]
        )

        state = next_state
        decision_index += 1
        if terminated or truncated:
            break

    if decision_index == 0:
        raise RuntimeError("评估 Episode 没有产生任何慢决策。")
    delay_metrics = delay_statistics(delay_samples)
    has_rejection = rejection_count > 0
    success_rate = _safe_rate(
        successful_requests,
        total_requests,
        empty=0.0 if has_rejection else 1.0,
    )
    sla_success_rate = 1.0 - _safe_rate(
        sla_violations,
        request_batches,
        empty=1.0 if has_rejection else 0.0,
    )
    sla_success_rate = min(1.0, max(0.0, sla_success_rate))
    mean_reliability = (
        float(np.mean(reliability_samples)) if reliability_samples else 0.0
    )
    minimum_reliability = min(reliability_samples) if reliability_samples else 0.0
    mean_active_memory_mb = _safe_rate(
        total_active_memory_mb_seconds,
        total_duration_seconds,
        empty=0.0,
    )
    cloud_replica_ratio = _safe_rate(
        cloud_replica_count,
        requested_replica_count,
        empty=0.0,
    )
    mean_primary_retention = float(
        np.mean([record.primary_retention_seconds for record in action_records])
    )
    mean_backup_retention = float(
        np.mean([record.backup_retention_seconds for record in action_records])
    )
    total_cost = run_cost + route_cost + cold_start_cost
    averaged_replica_counts = {
        function_id: float(np.mean(values))
        for function_id, values in replica_counts.items()
    }
    metrics_row: dict[str, float] = {
        "success_rate": success_rate,
        "sla_success_rate": sla_success_rate,
        **delay_metrics,
        "mean_reliability": mean_reliability,
        "minimum_reliability": float(minimum_reliability),
        "cold_start_count": float(cold_start_count),
        "mean_active_memory_mb": mean_active_memory_mb,
        "cloud_replica_ratio": cloud_replica_ratio,
        "cloud_usage_rate": _safe_rate(
            cloud_used_slots,
            total_fast_slots,
            empty=0.0,
        ),
        "mean_primary_retention_s": mean_primary_retention,
        "mean_backup_retention_s": mean_backup_retention,
        "raw_feasibility_rate": raw_feasible_count / decision_index,
        "projection_change_ratio": projection_change_sum / decision_index,
        "repair_attempts": float(repair_attempts),
        "repair_successes": float(repair_successes),
        "repair_failures": float(repair_failures),
        "rejection_count": float(rejection_count),
        "run_cost": run_cost,
        "route_cost": route_cost,
        "cold_start_cost": cold_start_cost,
        "total_cost": total_cost,
        "mean_reward": float(np.mean(rewards)),
        "total_reward": float(np.sum(rewards)),
        "decision_count": float(decision_index),
        "total_requests": float(total_requests),
    }
    for function_id, value in averaged_replica_counts.items():
        metrics_row[f"mean_replica_count_vnf_{function_id}"] = value
    return DPPOEpisodeEvaluation(
        episode_seed=episode_seed,
        metrics=metrics_row,
        fast_slot_delay_samples_ms=tuple(delay_samples),
        replica_counts_by_function=averaged_replica_counts,
        action_records=tuple(action_records),
    )


def evaluate_dppo_episode(
    environment: DPPOSlowTimescaleEnvironment,
    agent: DPPOAgent,
    *,
    episode_seed: int,
) -> DPPOEpisodeEvaluation:
    """关闭梯度完成独立评估，并恢复调用前的网络 train/eval 模式。"""

    if not isinstance(environment, DPPOSlowTimescaleEnvironment):
        raise TypeError("environment 必须是 DPPOSlowTimescaleEnvironment。")
    if not isinstance(agent, DPPOAgent):
        raise TypeError("agent 必须是 DPPOAgent。")
    if isinstance(episode_seed, bool) or not isinstance(episode_seed, int) or episode_seed < 0:
        raise ValueError("episode_seed 必须是非负整数。")
    if (
        agent.state_dim != environment.dimensions.state_dim
        or agent.action_dim != environment.dimensions.action_dim
    ):
        raise ValueError("智能体状态/动作维度与评估环境不一致。")

    networks = (agent.frozen_policy, agent.trainable_policy, agent.value_network)
    previous_modes = tuple(network.training for network in networks)
    for network in networks:
        network.eval()
    try:
        with torch.no_grad():
            return _run_evaluation_episode(environment, agent, episode_seed)
    finally:
        for network, was_training in zip(networks, previous_modes, strict=True):
            network.train(was_training)


def evaluate_dppo_episodes(
    environment: DPPOSlowTimescaleEnvironment,
    agent: DPPOAgent,
    *,
    episodes: int,
    seed_start: int,
) -> tuple[DPPOEpisodeEvaluation, ...]:
    """按连续固定种子执行多个相互独立的评估 Episode。"""

    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0:
        raise ValueError("episodes 必须是正整数。")
    if isinstance(seed_start, bool) or not isinstance(seed_start, int) or seed_start < 0:
        raise ValueError("seed_start 必须是非负整数。")
    return tuple(
        evaluate_dppo_episode(
            environment,
            agent,
            episode_seed=seed_start + offset,
        )
        for offset in range(episodes)
    )


def summarize_evaluation(
    evaluation_rows: Sequence[Mapping[str, float]],
) -> tuple[dict[str, str | int | float], ...]:
    """对每个数值指标报告均值、样本标准差和 95% 置信区间半宽。"""

    rows = tuple(evaluation_rows)
    if not rows:
        raise ValueError("evaluation_rows 不能为空。")
    if any(not REQUIRED_EVALUATION_COLUMNS <= row.keys() for row in rows):
        raise ValueError("每个评估行都必须包含完整论文指标。")
    metric_names = sorted(set.intersection(*(set(row.keys()) for row in rows)))
    summary: list[dict[str, str | int | float]] = []
    for metric_name in metric_names:
        if metric_name == "episode_seed":
            continue
        try:
            values = np.asarray(
                [float(row[metric_name]) for row in rows],
                dtype=np.float64,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"指标 {metric_name} 必须是数值。") from error
        if not np.isfinite(values).all():
            raise ValueError(f"指标 {metric_name} 必须只包含有限数。")
        sample_count = int(values.size)
        standard_deviation = (
            float(np.std(values, ddof=1)) if sample_count > 1 else 0.0
        )
        summary.append(
            {
                "metric": metric_name,
                "mean": float(np.mean(values)),
                "sample_count": sample_count,
                "standard_deviation": standard_deviation,
                "confidence_interval_95": (
                    1.96 * standard_deviation / math.sqrt(sample_count)
                ),
            }
        )
    return tuple(summary)
