"""阶段 D 的配置驱动观察快照；只编码显式传入的只读状态。"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from src.orchestration_config import PhaseAConfig
from src.phase_d_policy_specs import ObservationFeature, ObservationSpec
from src.safe_deployment_decoder import ActionSpec


@dataclass(frozen=True)
class ObservationBundleVersions:
    queue_version: int
    lifecycle_version: int
    failure_version: int
    network_version: int
    observation_bundle_version: int


@dataclass(frozen=True)
class ObservationContext:
    current_slot: int
    total_episode_slots: int
    maximum_drain_slots: int
    is_drain_phase: bool
    serving_mec: int
    next_mec: int
    node_health: Mapping[int, bool]
    node_free_memory_ratio: Mapping[int, float]
    warm_counts: Mapping[tuple[int, int], int]
    starting_counts: Mapping[tuple[int, int], int]
    locked_counts: Mapping[tuple[int, int], int]
    retention_remaining_ratios: Mapping[tuple[int, int], tuple[float, float, float]]
    queue_load_ratios: Mapping[int, float]
    minimum_slack_ratios: Mapping[int, float]
    structural_reliability: float
    warm_reliability: float
    previous_service_deficit: float
    previous_cost_ratio: float
    previous_sla_violation: float


@dataclass(frozen=True)
class ObservationSnapshot:
    values: tuple[float, ...]
    spec: ObservationSpec
    versions: ObservationBundleVersions


def _spec(function_ids: tuple[int, ...], node_ids: tuple[int, ...], pair_count: int) -> ObservationSpec:
    stage_count = len(function_ids) + 1  # 上行队列加全部 VNF 阶段队列。
    features = (
        ObservationFeature("episode_progress", 1, "fixed_scale", 1.0),
        ObservationFeature("slow_frame_position", 1, "fixed_scale", 1.0),
        ObservationFeature("is_drain_phase", 1, "none", 1.0),
        ObservationFeature("serving_mec_one_hot", len(node_ids), "none", 1.0),
        ObservationFeature("next_mec_one_hot", len(node_ids), "none", 1.0),
        ObservationFeature("node_health", len(node_ids), "none", 1.0),
        ObservationFeature("node_free_memory", len(node_ids), "fixed_scale", 1.0),
        ObservationFeature("warm_instance_count", pair_count, "running", 1.0),
        ObservationFeature("starting_instance_count", pair_count, "running", 1.0),
        ObservationFeature("locked_instance_count", pair_count, "running", 1.0),
        ObservationFeature("retention_min_max_mean", pair_count * 3, "fixed_scale", 1.0),
        ObservationFeature("queue_load", stage_count, "running", 1.0),
        ObservationFeature("minimum_slack", stage_count, "fixed_scale", 1.0),
        ObservationFeature("reliability", 2, "none", 1.0),
        ObservationFeature("previous_frame_summary", 3, "none", 1.0),
    )
    return ObservationSpec(features, function_ids, node_ids, "phase-d-observation-v1")


def build_observation_snapshot(
    config: PhaseAConfig,
    action_spec: ActionSpec,
    context: ObservationContext,
    versions: ObservationBundleVersions,
) -> ObservationSnapshot:
    """按规范实体顺序生成扁平观察，规模改变时无需改代码。"""

    function_ids = tuple(sorted({function_id for function_id, _ in action_spec.pairs}))
    node_ids = tuple(sorted(config.node_resources))
    denominator = context.total_episode_slots + context.maximum_drain_slots
    if denominator <= 0:
        raise ValueError("实验时长必须为正。")
    values: list[float] = [
        min(1.0, context.current_slot / denominator),
        (context.current_slot % config.slow_frame_slots) / config.slow_frame_slots,
        float(context.is_drain_phase),
    ]
    values.extend(float(node_id == context.serving_mec) for node_id in node_ids)
    values.extend(float(node_id == context.next_mec) for node_id in node_ids)
    values.extend(float(context.node_health[node_id]) for node_id in node_ids)
    values.extend(float(context.node_free_memory_ratio[node_id]) for node_id in node_ids)
    for mapping in (context.warm_counts, context.starting_counts, context.locked_counts):
        values.extend(float(mapping.get(pair, 0)) for pair in action_spec.pairs)
    for pair in action_spec.pairs:
        values.extend(context.retention_remaining_ratios.get(pair, (0.0, 0.0, 0.0)))
    queue_ids = (-1, *function_ids)
    values.extend(float(context.queue_load_ratios.get(queue_id, 0.0)) for queue_id in queue_ids)
    values.extend(float(context.minimum_slack_ratios.get(queue_id, 1.0)) for queue_id in queue_ids)
    values.extend((context.structural_reliability, context.warm_reliability))
    values.extend((
        context.previous_service_deficit,
        context.previous_cost_ratio,
        context.previous_sla_violation,
    ))
    spec = _spec(function_ids, node_ids, len(action_spec.pairs))
    if len(values) != spec.dimension:
        raise RuntimeError("ObservationSpec 与实际编码维度不一致。")
    return ObservationSnapshot(tuple(values), spec, versions)
