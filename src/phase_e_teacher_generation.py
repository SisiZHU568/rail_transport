"""阶段 E 教师候选的完整安全搜索、生命周期预演与解码往返审计。"""

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Callable, Mapping

import numpy as np

from src.cost_ledger import CostLedger
from src.failure_process import FailureProcess, FailureSnapshot
from src.fast_resource_model import NetworkSnapshot
from src.fast_resource_optimizer import FastResourceOptimizer
from src.instance_lifecycle import (
    DeploymentTarget,
    InstanceLifecycleManager,
    LifecycleCommitResult,
    LifecycleDeploymentPlan,
)
from src.phase_e_teachers import TeacherCandidate, TeacherLabel, select_teacher_labels
from src.orchestration_core import (
    PhaseASlotCoordinator,
    solve_and_commit_fast_resources,
)
from src.phase_e_environment import ArrivalBatch
from src.phase_e_main_controller import PhaseEMainController
from src.queue_manager import QueueStateManager
from src.safe_deployment_decoder import (
    DecoderInput,
    DeploymentPlan,
    SafeDeploymentDecoder,
)


@dataclass(frozen=True)
class TeacherMetrics:
    predicted_deficit: float
    predicted_cost: float
    structural_reliability_margin: float
    balance_maximum: float
    balance_variance: float

    def __post_init__(self) -> None:
        values = (
            self.predicted_deficit,
            self.predicted_cost,
            self.structural_reliability_margin,
            self.balance_maximum,
            self.balance_variance,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("教师代理指标必须全部为有限数。")
        if self.predicted_deficit < 0.0 or self.predicted_cost < 0.0:
            raise ValueError("教师缺口和成本不能为负。")
        if self.balance_maximum < 0.0 or self.balance_variance < 0.0:
            raise ValueError("教师利用率均衡指标不能为负。")


@dataclass(frozen=True)
class TeacherCandidateBuildResult:
    code: str
    candidate: TeacherCandidate | None
    preview: LifecycleCommitResult | None


@dataclass(frozen=True)
class ExactTeacherGenerationResult:
    code: str
    labels: tuple[TeacherLabel, ...]
    candidate_count: int
    generated_patterns: int


class _ObservedStateFailureProcess(FailureProcess):
    """教师只保持当前可观测故障状态，不读取任何未来故障轨迹。"""

    def __init__(self, snapshot: FailureSnapshot) -> None:
        self.snapshot = snapshot

    def reset(self) -> None:
        return None

    def state_for_slot(self, time_slot: int) -> FailureSnapshot:
        if time_slot < self.snapshot.time_slot:
            raise ValueError("教师代理不能回退到观察时刻之前。")
        return FailureSnapshot(
            time_slot=time_slot,
            domain_up=dict(self.snapshot.domain_up),
            node_local_up=dict(self.snapshot.node_local_up),
            effective_node_up=dict(self.snapshot.effective_node_up),
            version=self.snapshot.version + time_slot - self.snapshot.time_slot,
            newly_unavailable_node_ids=(),
            newly_available_node_ids=(),
        )


class CausalTeacherProxy:
    """在状态副本上运行真实快层与 EDF，生成不含未来信息的教师指标。"""

    def __init__(
        self,
        *,
        controller: PhaseEMainController,
        optimizer: FastResourceOptimizer,
        lifecycle_manager: InstanceLifecycleManager,
        queue_manager: QueueStateManager,
        failure_snapshot: FailureSnapshot,
        structural_reliability_margin: Callable[[DeploymentPlan], float],
        function_cycles_per_input_bit: Mapping[int, float],
        network_for_slot: Callable[[int], NetworkSnapshot],
        predicted_arrivals_by_slot: Mapping[int, tuple[ArrivalBatch, ...]],
    ) -> None:
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in function_cycles_per_input_bit.values()
        ):
            raise ValueError("教师 CPU 周期参数必须为正有限数。")
        self.controller = controller
        self.optimizer = optimizer
        self.lifecycle_manager = lifecycle_manager
        self.queue_manager = queue_manager
        self.failure_snapshot = failure_snapshot
        self.structural_reliability_margin = structural_reliability_margin
        self.function_cycles_per_input_bit = dict(function_cycles_per_input_bit)
        self.network_for_slot = network_for_slot
        self.predicted_arrivals_by_slot = {
            slot: tuple(items) for slot, items in predicted_arrivals_by_slot.items()
        }

    def evaluate(
        self,
        plan: DeploymentPlan,
        preview: LifecycleCommitResult,
    ) -> TeacherMetrics:
        """候选先做生命周期预演，再在独立副本上逐快时隙求解。"""

        if not preview.accepted:
            raise ValueError("只有通过生命周期预审的候选才能评估。")
        config = self.controller.config
        start_slot = self.queue_manager.snapshot().current_slot
        lifecycle = InstanceLifecycleManager.from_snapshot(
            config=config,
            function_memory_mb=self.lifecycle_manager.function_memory_mb,
            snapshot=preview.snapshot,
        )
        queues = QueueStateManager.from_snapshot(
            self.queue_manager.flow_config,
            self.queue_manager.snapshot(),
            slot_seconds=self.queue_manager.slot_seconds,
            flow_absolute_tolerance_bits=max(
                self.queue_manager.flow_absolute_tolerance_bits,
                # 快层 residual_tolerance 的内部数据单位是 Mbit。
                self.optimizer.config.residual_tolerance * 1e6,
            ),
            flow_relative_tolerance=self.queue_manager.flow_relative_tolerance,
        )
        ledger = CostLedger()
        coordinator = PhaseASlotCoordinator(
            _ObservedStateFailureProcess(self.failure_snapshot),
            lifecycle,
            ledger,
        )

        # 新建实例的一次性费用在原预演中已经发生；代理副本不再重复提交计划，
        # 因此在这里按预演新增批次显式计入。
        event_cost = 0.0
        for batch in preview.created_batches:
            pair = config.deployment_pairs[(batch.function_id, batch.node_id)]
            event_cost += batch.count * (
                pair.deployment_cost_per_instance
                + pair.cold_start_cost_per_instance
            )
        weighted_deficit = 0.0
        fast_cost = 0.0
        work_by_node = {node_id: 0.0 for node_id in config.node_resources}
        for offset in range(config.slow_frame_slots):
            slot = start_slot + offset
            queues.begin_slot(slot)
            boundary = coordinator.begin_slot(slot)
            for arrival in self.predicted_arrivals_by_slot.get(slot, ()):
                queues.admit_batch(
                    arrival.batch_id,
                    arrival.service_id,
                    slot * config.fast_slot_seconds,
                    arrival.absolute_deadline_time,
                    arrival.input_equivalent_bits,
                )
            network = self.network_for_slot(slot)
            if network.current_slot != slot:
                raise RuntimeError("TEACHER_INTERNAL_FAILURE: stale network snapshot")
            fast = solve_and_commit_fast_resources(
                self.optimizer,
                queues,
                boundary.lifecycle_snapshot,
                boundary.failure_snapshot,
                network,
            )
            if not fast.optimization.succeeded:
                raise RuntimeError(
                    f"TEACHER_INTERNAL_FAILURE: {fast.optimization.code}"
                )
            if fast.commit is not None and not fast.commit.accepted:
                raise RuntimeError(
                    f"TEACHER_INTERNAL_FAILURE: {fast.commit.code}"
                )
            # 一级目标已经包含 EDF 紧迫度权重，单位由 Mbit 换回等效 bit。
            weighted_deficit += max(0.0, fast.optimization.primary_optimum) * 1e6
            fast_cost += fast.optimization.total_resource_cost
            if fast.optimization.plan is not None:
                for operation in fast.optimization.plan.operations:
                    if operation.operation_type != "execute":
                        continue
                    function_id = operation.queue_key.stage_id
                    node_id = operation.queue_key.location
                    work_by_node[node_id] += (
                        operation.physical_bits
                        * self.function_cycles_per_input_bit[function_id]
                    )
            coordinator.finalize_slot_costing()

        running_cost = sum(item.amount for item in ledger.entries)
        frame_seconds = config.slow_frame_slots * config.fast_slot_seconds
        utilizations = tuple(
            min(
                1.0,
                work_by_node[node_id]
                / (resource.cpu_capacity_cycles_per_second * frame_seconds),
            )
            for node_id, resource in sorted(config.node_resources.items())
        )
        return TeacherMetrics(
            predicted_deficit=weighted_deficit,
            predicted_cost=event_cost + running_cost + fast_cost,
            structural_reliability_margin=float(
                self.structural_reliability_margin(plan)
            ),
            balance_maximum=max(utilizations, default=0.0),
            balance_variance=(float(np.var(utilizations)) if utilizations else 0.0),
        )


def _canonical_plan(plan: DeploymentPlan) -> tuple[str, str]:
    counts = tuple(
        (function_id, node_id, int(value))
        for (function_id, node_id), value in sorted(plan.instance_counts.items())
    )
    retention = tuple(
        (function_id, node_id, int(value))
        for (function_id, node_id), value in sorted(plan.retention_slots.items())
    )
    serialized = json.dumps(
        {"instance_counts": counts, "retention_slots": retention},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    # 规范键先比较实例数、再比较保留档，哈希只负责最终消歧。
    key = json.dumps((counts, retention, digest), separators=(",", ":"))
    return digest, key


def _lifecycle_plan(
    plan: DeploymentPlan,
    lifecycle_manager: InstanceLifecycleManager,
    failure_snapshot: FailureSnapshot,
) -> LifecycleDeploymentPlan:
    snapshot = lifecycle_manager.snapshot()
    return LifecycleDeploymentPlan(
        expected_lifecycle_version=snapshot.version,
        expected_failure_version=failure_snapshot.version,
        current_slot=snapshot.current_slot,
        targets=tuple(
            DeploymentTarget(
                function_id,
                node_id,
                plan.instance_counts[(function_id, node_id)],
                plan.retention_slots[(function_id, node_id)],
            )
            for function_id, node_id in sorted(plan.instance_counts)
        ),
    )


def build_teacher_candidate(
    *,
    plan: DeploymentPlan,
    eligible_teacher_types: tuple[str, ...],
    decoder: SafeDeploymentDecoder,
    decoder_input: DecoderInput,
    lifecycle_manager: InstanceLifecycleManager,
    failure_snapshot: FailureSnapshot,
    metrics: TeacherMetrics,
) -> TeacherCandidateBuildResult:
    """只在生命周期预演和同解码器往返都通过后建立候选。"""

    allowed_types = {"COST", "RELIABILITY", "BALANCE"}
    if not eligible_teacher_types or not set(eligible_teacher_types) <= allowed_types:
        raise ValueError("教师类型必须来自 COST/RELIABILITY/BALANCE。")
    preview = lifecycle_manager.preview_deployment(
        _lifecycle_plan(plan, lifecycle_manager, failure_snapshot),
        failure_snapshot,
    )
    if not preview.accepted:
        return TeacherCandidateBuildResult(preview.code, None, preview)
    encoded = decoder.encode_plan(plan, decoder_input)
    if encoded.code != "OK":
        return TeacherCandidateBuildResult(encoded.code, None, preview)
    plan_hash, canonical_key = _canonical_plan(plan)
    mask = tuple(
        value
        for pair in decoder.action_spec.pairs
        for value in (True, plan.instance_counts[pair] > 0)
    )
    candidate = TeacherCandidate(
        plan_hash=plan_hash,
        scores=encoded.scores,
        predicted_deficit=metrics.predicted_deficit,
        predicted_cost=metrics.predicted_cost,
        structural_reliability_margin=metrics.structural_reliability_margin,
        balance_score=metrics.balance_maximum,
        canonical_key=canonical_key,
        plan=plan,
        effective_action_mask=mask,
        balance_variance=metrics.balance_variance,
        eligible_teacher_types=eligible_teacher_types,
    )
    return TeacherCandidateBuildResult("OK", candidate, preview)


def generate_exact_teacher_labels(
    *,
    decoder: SafeDeploymentDecoder,
    decoder_input: DecoderInput,
    lifecycle_manager: InstanceLifecycleManager,
    failure_snapshot: FailureSnapshot,
    retention_profiles: Mapping[str, Mapping[tuple[int, int], int]],
    evaluator: Callable[[DeploymentPlan, LifecycleCommitResult], TeacherMetrics],
    max_generated_patterns: int,
) -> ExactTeacherGenerationResult:
    """完整枚举每类教师的固定保留规则；任一搜索被截断就不输出部分最优。"""

    candidates: list[TeacherCandidate] = []
    generated_patterns = 0
    for teacher_type in ("COST", "RELIABILITY", "BALANCE"):
        if teacher_type not in retention_profiles:
            continue
        enumeration = decoder.enumerate_safe_plans(
            decoder_input,
            max_generated_patterns=max_generated_patterns,
            retention_slots_by_pair=retention_profiles[teacher_type],
        )
        generated_patterns += enumeration.generated_patterns
        if enumeration.code == "DECODER_SEARCH_LIMIT":
            return ExactTeacherGenerationResult(
                "TEACHER_SEARCH_LIMIT", (), 0, generated_patterns
            )
        if enumeration.code == "NO_SAFE_FEASIBLE_DEPLOYMENT":
            continue
        try:
            for plan in enumeration.plans:
                preview = lifecycle_manager.preview_deployment(
                    _lifecycle_plan(plan, lifecycle_manager, failure_snapshot),
                    failure_snapshot,
                )
                if not preview.accepted:
                    continue
                built = build_teacher_candidate(
                    plan=plan,
                    eligible_teacher_types=(teacher_type,),
                    decoder=decoder,
                    decoder_input=decoder_input,
                    lifecycle_manager=lifecycle_manager,
                    failure_snapshot=failure_snapshot,
                    metrics=evaluator(plan, preview),
                )
                if built.code == "OK" and built.candidate is not None:
                    candidates.append(built.candidate)
        except (ArithmeticError, ValueError, RuntimeError):
            return ExactTeacherGenerationResult(
                "TEACHER_INTERNAL_FAILURE", (), 0, generated_patterns
            )
    if not candidates:
        return ExactTeacherGenerationResult(
            "NO_SAFE_FEASIBLE_DEPLOYMENT", (), 0, generated_patterns
        )
    labels = select_teacher_labels(tuple(candidates))
    return ExactTeacherGenerationResult(
        "OK", labels, len(candidates), generated_patterns
    )
