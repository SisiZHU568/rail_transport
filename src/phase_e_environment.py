"""阶段 E 的单慢帧主环境：慢层部署、快层求解、EDF 提交和奖励汇总。"""

from dataclasses import dataclass
import math
from typing import Callable, Mapping

import numpy as np

from src.cost_ledger import CostLedger
from src.failure_process import FailureSnapshot
from src.fast_resource_model import NetworkSnapshot
from src.fast_resource_optimizer import FastResourceOptimizer
from src.instance_lifecycle import InstanceLifecycleManager, LifecycleSnapshot
from src.orchestration_core import (
    FastResourceSlotResult,
    PhaseBSlotCoordinator,
    solve_and_commit_fast_resources,
)
from src.phase_d_observation import ObservationBundleVersions
from src.phase_e_main_controller import PhaseEMainController
from src.phase_e_training_workflow import FrameReward, FrameRewardInput, compute_frame_reward
from src.queue_manager import QueueStateManager
from src.queue_state import QueueSnapshot


@dataclass(frozen=True)
class ArrivalBatch:
    batch_id: str
    service_id: int
    input_equivalent_bits: float
    absolute_deadline_time: float

    def __post_init__(self) -> None:
        if (
            not self.batch_id
            or self.service_id < 0
            or not math.isfinite(self.input_equivalent_bits)
            or self.input_equivalent_bits <= 0.0
            or not math.isfinite(self.absolute_deadline_time)
        ):
            raise ValueError("到达批次字段无效。")


@dataclass(frozen=True)
class SlowDecisionContext:
    queue_snapshot: QueueSnapshot
    lifecycle_snapshot: LifecycleSnapshot
    failure_snapshot: FailureSnapshot
    network_snapshot: NetworkSnapshot
    versions: ObservationBundleVersions


@dataclass(frozen=True)
class PhaseEFastSlotResult:
    slot: int
    queue_equivalent_bits: float
    deficit_equivalent_bits: float
    fast_result: FastResourceSlotResult


@dataclass(frozen=True)
class PhaseESlowFrameResult:
    code: str
    reward: FrameReward | None
    slots: tuple[PhaseEFastSlotResult, ...]
    queue_equivalent_bits: float
    deficit_equivalent_bits: float
    raw_cost: float
    new_violation_count: int
    at_risk_batch_count: int


class PhaseESlowFrameEnvironment:
    """只运行一个固定长度慢帧；内部失败不生成可写入 PPO 的奖励转移。"""

    _INTERNAL_FAILURES = {
        "DECODER_SEARCH_LIMIT",
        "DECODER_INTERNAL_FAILURE",
        "INVALID_POLICY_SCORE",
        "FAST_SOLVER_FAILURE",
        "STALE_SNAPSHOT",
        "INVALID_ALLOCATION_PLAN",
        "FLOW_CONSERVATION_VIOLATION",
    }

    def __init__(
        self,
        *,
        controller: PhaseEMainController,
        coordinator: PhaseBSlotCoordinator,
        optimizer: FastResourceOptimizer,
        lifecycle_manager: InstanceLifecycleManager,
        queue_manager: QueueStateManager,
        cost_ledger: CostLedger,
        required_replica_nodes: Mapping[int, int],
        reference_cost: float,
        alpha: float = 0.8,
        fault_domain_by_node: Mapping[int, int] | None = None,
        minimum_fault_domains: Mapping[int, int] | None = None,
        domain_availability: Mapping[int, float] | None = None,
        node_conditional_availability: Mapping[int, float] | None = None,
        maximum_vnf_unavailability: Mapping[int, float] | None = None,
    ) -> None:
        if not math.isfinite(reference_cost) or reference_cost <= 0.0:
            raise ValueError("reference_cost 必须为正有限数。")
        self.controller = controller
        self.coordinator = coordinator
        self.optimizer = optimizer
        self.lifecycle_manager = lifecycle_manager
        self.queue_manager = queue_manager
        self.cost_ledger = cost_ledger
        self.required_replica_nodes = dict(required_replica_nodes)
        self.fault_domain_by_node = dict(fault_domain_by_node or {})
        self.minimum_fault_domains = dict(minimum_fault_domains or {})
        self.domain_availability = dict(domain_availability or {})
        self.node_conditional_availability = dict(
            node_conditional_availability or {}
        )
        self.maximum_vnf_unavailability = dict(
            maximum_vnf_unavailability or {}
        )
        self.reference_cost = reference_cost
        self.alpha = alpha

    @staticmethod
    def _available_queue_bits(snapshot: QueueSnapshot) -> float:
        return sum(
            item.input_equivalent_bits
            for item in (*snapshot.uplink_fragments, *snapshot.stage_fragments)
            if item.available_slot <= snapshot.current_slot
        )

    @staticmethod
    def _failure_result(
        code: str,
        slots: list[PhaseEFastSlotResult],
        queue_bits: float,
        deficit_bits: float,
        raw_cost: float,
        violations: int,
        at_risk: int,
    ) -> PhaseESlowFrameResult:
        return PhaseESlowFrameResult(
            code, None, tuple(slots), queue_bits, deficit_bits,
            raw_cost, violations, at_risk,
        )

    def run_slow_frame(
        self,
        *,
        start_slot: int,
        policy: Callable[[SlowDecisionContext], np.ndarray],
        arrivals_by_slot: Mapping[int, tuple[ArrivalBatch, ...]],
        network_for_slot: Callable[[int], NetworkSnapshot],
        training_mode: bool,
    ) -> PhaseESlowFrameResult:
        config = self.controller.config
        if start_slot % config.slow_frame_slots != 0:
            raise ValueError("慢帧必须从 slow_frame_slots 的整数边界开始。")
        initial_ledger_size = len(self.cost_ledger.entries)
        at_risk_ids = {
            batch.batch_id
            for batch in self.queue_manager.snapshot().batches
            if batch.completion_slot is None and not batch.violation_recorded
        }
        total_queue = 0.0
        total_deficit = 0.0
        fast_cost = 0.0
        new_violations = 0
        slot_results: list[PhaseEFastSlotResult] = []

        for offset in range(config.slow_frame_slots):
            slot = start_slot + offset
            network = network_for_slot(slot)
            if network.current_slot != slot:
                return self._failure_result(
                    "STALE_SNAPSHOT", slot_results, total_queue,
                    total_deficit, fast_cost, new_violations, len(at_risk_ids),
                )
            boundary = self.coordinator.begin_slot(
                slot, network_version=network.version
            )
            new_violations += boundary.sla_report.new_violation_count
            for arrival in arrivals_by_slot.get(slot, ()):
                arrival_time = slot * config.fast_slot_seconds
                if arrival.absolute_deadline_time < arrival_time:
                    raise ValueError("到达批次截止时间不能早于到达时刻。")
                self.queue_manager.admit_batch(
                    arrival.batch_id,
                    arrival.service_id,
                    arrival_time,
                    arrival.absolute_deadline_time,
                    arrival.input_equivalent_bits,
                )
                at_risk_ids.add(arrival.batch_id)

            if offset == 0:
                queue_snapshot = self.queue_manager.snapshot()
                lifecycle_snapshot = self.lifecycle_manager.snapshot()
                versions = ObservationBundleVersions(
                    queue_snapshot.version,
                    lifecycle_snapshot.version,
                    boundary.failure_snapshot.version,
                    network.version,
                    start_slot // config.slow_frame_slots,
                )
                context = SlowDecisionContext(
                    queue_snapshot,
                    lifecycle_snapshot,
                    boundary.failure_snapshot,
                    network,
                    versions,
                )
                proposal = self.controller.propose_deployment(
                    policy(context),
                    lifecycle_manager=self.lifecycle_manager,
                    failure_snapshot=boundary.failure_snapshot,
                    observation_versions=versions,
                    required_replica_nodes=self.required_replica_nodes,
                    fault_domain_by_node=self.fault_domain_by_node,
                    minimum_fault_domains=self.minimum_fault_domains,
                    domain_availability=self.domain_availability,
                    node_conditional_availability=(
                        self.node_conditional_availability
                    ),
                    maximum_vnf_unavailability=(
                        self.maximum_vnf_unavailability
                    ),
                )
                if proposal.code == "OK" and proposal.lifecycle_plan is not None:
                    committed = self.coordinator.phase_a.commit_deployment(
                        proposal.lifecycle_plan
                    )
                    if not committed.accepted:
                        return self._failure_result(
                            committed.code, slot_results, total_queue,
                            total_deficit, fast_cost, new_violations,
                            len(at_risk_ids),
                        )
                elif proposal.code != "NO_SAFE_FEASIBLE_DEPLOYMENT":
                    return self._failure_result(
                        proposal.code, slot_results, total_queue,
                        total_deficit, fast_cost, new_violations,
                        len(at_risk_ids),
                    )

            queue_before = self.queue_manager.snapshot()
            queue_bits = self._available_queue_bits(queue_before)
            total_queue += queue_bits
            fast = solve_and_commit_fast_resources(
                self.optimizer,
                self.queue_manager,
                self.lifecycle_manager.snapshot(),
                boundary.failure_snapshot,
                network,
            )
            if not fast.optimization.succeeded:
                deficit = queue_bits
                if training_mode:
                    return self._failure_result(
                        fast.optimization.code, slot_results, total_queue,
                        total_deficit + deficit, fast_cost, new_violations,
                        len(at_risk_ids),
                    )
            else:
                deficit = min(
                    queue_bits,
                    max(0.0, fast.optimization.service_shortfall_equivalent_bits),
                )
                fast_cost += fast.optimization.total_resource_cost
                if fast.commit is not None and not fast.commit.accepted:
                    return self._failure_result(
                        fast.commit.code, slot_results, total_queue,
                        total_deficit + deficit, fast_cost, new_violations,
                        len(at_risk_ids),
                    )
            total_deficit += deficit
            self.coordinator.phase_a.finalize_slot_costing()
            slot_results.append(
                PhaseEFastSlotResult(slot, queue_bits, deficit, fast)
            )

        lifecycle_cost = sum(
            entry.amount
            for entry in self.cost_ledger.entries[initial_ledger_size:]
        )
        raw_cost = lifecycle_cost + fast_cost
        reward = compute_frame_reward(
            FrameRewardInput(
                len(at_risk_ids),
                new_violations,
                total_queue,
                total_deficit,
                raw_cost,
                self.reference_cost,
            ),
            alpha=self.alpha,
        )
        return PhaseESlowFrameResult(
            "OK", reward, tuple(slot_results), total_queue, total_deficit,
            raw_cost, new_violations, len(at_risk_ids),
        )
