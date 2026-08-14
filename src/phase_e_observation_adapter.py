"""把在线四类只读快照编码为配置驱动的阶段 D ObservationSnapshot。"""

import math

from src.instance_lifecycle import LifecycleStatus
from src.orchestration_config import PhaseAConfig
from src.phase_d_observation import (
    ObservationContext,
    ObservationSnapshot,
    build_observation_snapshot,
)
from src.phase_e_environment import PhaseESlowFrameResult, SlowDecisionContext
from src.safe_deployment_decoder import ActionSpec


class PhaseEObservationAdapter:
    """只读编码器；运行统计不写入 ObservationSpec 哈希。"""

    def __init__(
        self,
        config: PhaseAConfig,
        action_spec: ActionSpec,
        *,
        total_episode_slots: int,
        maximum_drain_slots: int,
    ) -> None:
        if total_episode_slots <= 0 or maximum_drain_slots < 0:
            raise ValueError("实验时隙数必须为正，排空时隙数不能为负。")
        self.config = config
        self.action_spec = action_spec
        self.total_episode_slots = total_episode_slots
        self.maximum_drain_slots = maximum_drain_slots

    def encode(
        self,
        context: SlowDecisionContext,
        *,
        next_mec: int | None = None,
        is_drain_phase: bool = False,
        structural_reliability: float = 0.0,
        warm_reliability: float = 0.0,
        previous_frame: PhaseESlowFrameResult | None = None,
    ) -> ObservationSnapshot:
        lifecycle = context.lifecycle_snapshot
        queue = context.queue_snapshot
        current_slot = queue.current_slot
        max_retention = max(1, max(self.action_spec.retention_slot_options))

        warm_counts: dict[tuple[int, int], int] = {}
        starting_counts: dict[tuple[int, int], int] = {}
        locked_counts: dict[tuple[int, int], int] = {}
        remaining: dict[tuple[int, int], tuple[float, float, float]] = {}
        for pair in self.action_spec.pairs:
            pair_batches = tuple(
                batch for batch in lifecycle.batches
                if (batch.function_id, batch.node_id) == pair
            )
            warm_counts[pair] = sum(
                batch.count for batch in pair_batches
                if batch.status is LifecycleStatus.WARM
            )
            starting_counts[pair] = sum(
                batch.count for batch in pair_batches
                if batch.status is LifecycleStatus.STARTING
            )
            locked_counts[pair] = lifecycle.locked_count(
                *pair, current_slot=current_slot
            )
            values = [
                max(0.0, batch.retention_deadline_slot - current_slot)
                / max_retention
                for batch in pair_batches
                for _ in range(batch.count)
            ]
            remaining[pair] = (
                (min(values), max(values), sum(values) / len(values))
                if values else (0.0, 0.0, 0.0)
            )

        node_free_memory = {
            node_id: max(
                0.0,
                1.0
                - lifecycle.memory_used_mb_by_node.get(node_id, 0.0)
                / resource.memory_capacity_mb,
            )
            for node_id, resource in self.config.node_resources.items()
        }
        batches = {item.batch_id: item for item in queue.batches}
        queue_loads: dict[int, float] = {}
        slack_by_stage: dict[int, list[float]] = {}
        slow_seconds = self.config.slow_frame_slots * self.config.fast_slot_seconds
        for fragment in (*queue.uplink_fragments, *queue.stage_fragments):
            if fragment.available_slot > current_slot:
                continue
            stage_id = fragment.stage_id
            # 使用 Mbit 作为固定物理尺度；后续 running normalization 独立保存。
            queue_loads[stage_id] = (
                queue_loads.get(stage_id, 0.0)
                + fragment.input_equivalent_bits / 1e6
            )
            batch = batches.get(fragment.batch_id)
            if batch is not None:
                slack = (
                    batch.absolute_deadline_time
                    - current_slot * self.config.fast_slot_seconds
                ) / slow_seconds
                slack_by_stage.setdefault(stage_id, []).append(
                    min(1.0, max(-1.0, slack))
                )
        minimum_slack = {
            stage_id: min(values) for stage_id, values in slack_by_stage.items()
        }
        if previous_frame is None or previous_frame.reward is None:
            previous_deficit = previous_cost = previous_violation = 0.0
        else:
            previous_deficit = previous_frame.reward.deficit_rate
            previous_cost = previous_frame.reward.normalized_cost
            previous_violation = previous_frame.reward.violation_rate
        values = (
            structural_reliability,
            warm_reliability,
            previous_deficit,
            previous_cost,
            previous_violation,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("可靠性和上一慢帧摘要必须为有限数。")
        observation_context = ObservationContext(
            current_slot=current_slot,
            total_episode_slots=self.total_episode_slots,
            maximum_drain_slots=self.maximum_drain_slots,
            is_drain_phase=is_drain_phase,
            serving_mec=context.network_snapshot.serving_mec,
            next_mec=(
                context.network_snapshot.serving_mec
                if next_mec is None else next_mec
            ),
            node_health=context.failure_snapshot.effective_node_up,
            node_free_memory_ratio=node_free_memory,
            warm_counts=warm_counts,
            starting_counts=starting_counts,
            locked_counts=locked_counts,
            retention_remaining_ratios=remaining,
            queue_load_ratios=queue_loads,
            minimum_slack_ratios=minimum_slack,
            structural_reliability=structural_reliability,
            warm_reliability=warm_reliability,
            previous_service_deficit=previous_deficit,
            previous_cost_ratio=previous_cost,
            previous_sla_violation=previous_violation,
        )
        return build_observation_snapshot(
            self.config,
            self.action_spec,
            observation_context,
            context.versions,
        )
