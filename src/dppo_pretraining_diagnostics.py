"""提供 DPPO 预训练诊断所需的纯指标计算。"""

from dataclasses import dataclass
from collections import Counter
import math

import numpy as np

from src.dppo_action_space import DPPOActionSpace
from src.dppo_projection import ProjectionResult


def _finite_float(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} 必须是有限数。")
    return number


def _action_vector(name: str, value: np.ndarray, action_dim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (action_dim,):
        raise ValueError(f"{name} 形状必须是 ({action_dim},)。")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} 必须只包含有限数。")
    result = np.array(array, dtype=np.float32, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class GeneratedActionMetric:
    """保存一个模型在一条专家记录上的拟合与投影指标。"""

    partition: str
    model_name: str
    episode_seed: int
    slow_step: int
    teacher_name: str
    action_mse: float
    action_mae: float
    replica_accuracy: float
    node_prefix_exact_rate: float
    node_prefix_set_rate: float
    primary_retention_mae_seconds: float
    backup_retention_mae_seconds: float
    raw_feasible: bool
    projection_success: bool
    projection_change_ratio: float
    projection_reasons: tuple[str, ...]
    generated_replica_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.partition not in {"validation", "test"}:
            raise ValueError("partition 必须是 validation 或 test。")
        if self.model_name not in {"random", "pretrained"}:
            raise ValueError("model_name 必须是 random 或 pretrained。")
        if not isinstance(self.teacher_name, str) or not self.teacher_name:
            raise ValueError("teacher_name 必须是非空字符串。")
        for name in ("episode_seed", "slow_step"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} 必须是非负整数。")
        for name in (
            "action_mse",
            "action_mae",
            "replica_accuracy",
            "node_prefix_exact_rate",
            "node_prefix_set_rate",
            "primary_retention_mae_seconds",
            "backup_retention_mae_seconds",
            "projection_change_ratio",
        ):
            value = _finite_float(name, getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} 不能为负数。")
            object.__setattr__(self, name, value)
        for name in (
            "replica_accuracy",
            "node_prefix_exact_rate",
            "node_prefix_set_rate",
            "projection_change_ratio",
        ):
            if getattr(self, name) > 1.0:
                raise ValueError(f"{name} 不能大于 1。")
        if not isinstance(self.raw_feasible, bool):
            raise ValueError("raw_feasible 必须是布尔值。")
        if not isinstance(self.projection_success, bool):
            raise ValueError("projection_success 必须是布尔值。")
        object.__setattr__(self, "projection_reasons", tuple(self.projection_reasons))
        object.__setattr__(
            self,
            "generated_replica_counts",
            tuple(int(value) for value in self.generated_replica_counts),
        )


def evaluate_generated_action(
    *,
    partition: str,
    model_name: str,
    episode_seed: int,
    slow_step: int,
    teacher_name: str,
    generated_action: np.ndarray,
    expert_action: np.ndarray,
    action_space: DPPOActionSpace,
    projection_result: ProjectionResult,
) -> GeneratedActionMetric:
    """比较单条生成动作和专家动作，并保留原始投影结果。"""

    if not isinstance(action_space, DPPOActionSpace):
        raise TypeError("action_space 必须是 DPPOActionSpace。")
    if not isinstance(projection_result, ProjectionResult):
        raise TypeError("projection_result 必须是 ProjectionResult。")
    generated = _action_vector(
        "generated_action", generated_action, action_space.action_dim
    )
    expert = _action_vector("expert_action", expert_action, action_space.action_dim)
    generated_decoded = action_space.decode(generated)
    expert_decoded = action_space.decode(expert)
    generated_by_function = {
        action.function_id: action
        for action in generated_decoded.function_actions
    }
    if tuple(generated_by_function) != tuple(
        action.function_id for action in expert_decoded.function_actions
    ):
        raise ValueError("生成动作和专家动作的 VNF 顺序不一致。")

    replica_matches = 0
    prefix_exact_matches = 0
    prefix_set_matches = 0
    primary_errors: list[float] = []
    backup_errors: list[float] = []
    replica_counts: list[int] = []
    for expert_function in expert_decoded.function_actions:
        generated_function = generated_by_function[expert_function.function_id]
        replica_counts.append(generated_function.replica_count)
        replica_matches += int(
            generated_function.replica_count == expert_function.replica_count
        )
        # 专家副本数就是标签实际要部署的节点数，因此只比较这个前缀。
        prefix_length = expert_function.replica_count
        expert_prefix = expert_function.ranked_node_ids[:prefix_length]
        generated_prefix = generated_function.ranked_node_ids[:prefix_length]
        prefix_exact_matches += int(generated_prefix == expert_prefix)
        prefix_set_matches += int(set(generated_prefix) == set(expert_prefix))
        primary_errors.append(
            abs(
                generated_function.primary_retention_seconds
                - expert_function.primary_retention_seconds
            )
        )
        backup_errors.append(
            abs(
                generated_function.backup_retention_seconds
                - expert_function.backup_retention_seconds
            )
        )

    function_count = len(expert_decoded.function_actions)
    difference = generated.astype(np.float64) - expert.astype(np.float64)
    return GeneratedActionMetric(
        partition=partition,
        model_name=model_name,
        episode_seed=episode_seed,
        slow_step=slow_step,
        teacher_name=teacher_name,
        action_mse=float(np.mean(difference**2)),
        action_mae=float(np.mean(np.abs(difference))),
        replica_accuracy=replica_matches / function_count,
        node_prefix_exact_rate=prefix_exact_matches / function_count,
        node_prefix_set_rate=prefix_set_matches / function_count,
        primary_retention_mae_seconds=float(np.mean(primary_errors)),
        backup_retention_mae_seconds=float(np.mean(backup_errors)),
        raw_feasible=projection_result.raw_feasible,
        projection_success=projection_result.success,
        projection_change_ratio=projection_result.change_ratio,
        projection_reasons=projection_result.reasons,
        generated_replica_counts=tuple(replica_counts),
    )


def _mean(rows: tuple[GeneratedActionMetric, ...], name: str) -> float:
    return float(np.mean([float(getattr(row, name)) for row in rows]))


def _model_summary(rows: tuple[GeneratedActionMetric, ...]) -> dict[str, object]:
    failure_counts = Counter(
        reason for row in rows for reason in row.projection_reasons
    )
    replica_counts = Counter(
        count for row in rows for count in row.generated_replica_counts
    )
    return {
        "mean_action_mse": _mean(rows, "action_mse"),
        "median_action_mse": float(np.median([row.action_mse for row in rows])),
        "mean_action_mae": _mean(rows, "action_mae"),
        "median_action_mae": float(np.median([row.action_mae for row in rows])),
        "replica_accuracy": _mean(rows, "replica_accuracy"),
        "node_prefix_exact_rate": _mean(rows, "node_prefix_exact_rate"),
        "node_prefix_set_rate": _mean(rows, "node_prefix_set_rate"),
        "primary_retention_mae_seconds": _mean(
            rows, "primary_retention_mae_seconds"
        ),
        "backup_retention_mae_seconds": _mean(
            rows, "backup_retention_mae_seconds"
        ),
        "raw_feasibility_rate": _mean(rows, "raw_feasible"),
        "projection_success_rate": _mean(rows, "projection_success"),
        "mean_projection_change_ratio": _mean(rows, "projection_change_ratio"),
        "projection_failure_counts": {
            key: failure_counts[key] for key in sorted(failure_counts)
        },
        "replica_count_distribution": {
            str(key): replica_counts[key] for key in sorted(replica_counts)
        },
    }


def summarize_diagnostic_rows(
    rows: tuple[GeneratedActionMetric, ...],
) -> dict[str, object]:
    """成对汇总随机模型和预训练模型，避免缺失记录造成偏置。"""

    normalized = tuple(rows)
    if not normalized or any(
        not isinstance(row, GeneratedActionMetric) for row in normalized
    ):
        raise ValueError("rows 必须包含 GeneratedActionMetric。")
    by_key: dict[tuple[str, int, int], dict[str, GeneratedActionMetric]] = {}
    for row in normalized:
        key = (row.partition, row.episode_seed, row.slow_step)
        model_rows = by_key.setdefault(key, {})
        if row.model_name in model_rows:
            raise ValueError("同一记录不能包含重复模型指标。")
        model_rows[row.model_name] = row
    if any(set(model_rows) != {"random", "pretrained"} for model_rows in by_key.values()):
        raise ValueError("每条记录必须包含成对的 random 和 pretrained 指标。")

    random_rows = tuple(by_key[key]["random"] for key in sorted(by_key))
    pretrained_rows = tuple(by_key[key]["pretrained"] for key in sorted(by_key))
    random_summary = _model_summary(random_rows)
    pretrained_summary = _model_summary(pretrained_rows)
    random_mean_mse = float(random_summary["mean_action_mse"])
    pretrained_mean_mse = float(pretrained_summary["mean_action_mse"])
    if random_mean_mse <= 0.0:
        relative_reduction = 0.0 if pretrained_mean_mse == 0.0 else -1.0
    else:
        relative_reduction = (
            random_mean_mse - pretrained_mean_mse
        ) / random_mean_mse
    improvement_rate = float(
        np.mean(
            [
                pretrained.action_mse < random.action_mse
                for random, pretrained in zip(
                    random_rows, pretrained_rows, strict=True
                )
            ]
        )
    )
    comparison: dict[str, float] = {
        "relative_mean_mse_reduction": relative_reduction,
        "record_mse_improvement_rate": improvement_rate,
    }
    delta_names = (
        "replica_accuracy",
        "node_prefix_exact_rate",
        "node_prefix_set_rate",
        "raw_feasibility_rate",
        "projection_success_rate",
    )
    for name in delta_names:
        comparison[f"{name}_delta"] = float(pretrained_summary[name]) - float(
            random_summary[name]
        )
    for name in (
        "primary_retention_mae_seconds",
        "backup_retention_mae_seconds",
        "mean_projection_change_ratio",
    ):
        comparison[f"{name}_delta"] = float(pretrained_summary[name]) - float(
            random_summary[name]
        )

    continuous_learned = relative_reduction >= 0.10 and improvement_rate > 0.50
    deployment_improved = any(
        comparison[name] > 0.0
        for name in (
            "replica_accuracy_delta",
            "node_prefix_exact_rate_delta",
            "node_prefix_set_rate_delta",
            "raw_feasibility_rate_delta",
        )
    )
    if not continuous_learned:
        conclusion = "continuous_signal_not_learned"
    elif deployment_improved:
        conclusion = "continuous_and_deployment_signal_learned"
    else:
        conclusion = "continuous_only_mapping_problem"
    return {
        "record_count": len(by_key),
        "models": {
            "random": random_summary,
            "pretrained": pretrained_summary,
        },
        "comparison": comparison,
        "conclusion": conclusion,
    }
