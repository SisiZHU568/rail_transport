"""阶段 B 队列状态所有者：跨时隙事件、确定性 EDF 与原子提交。"""

from dataclasses import dataclass, replace
import hashlib
import math
from types import MappingProxyType
from typing import Mapping

from src.queue_state import (
    BatchRecord,
    CompletionEvent,
    InTransitRecord,
    QueueFragment,
    QueueSnapshot,
    StageFlowConfig,
)


def _stable_id(prefix: str, *parts: object) -> str:
    """以规范字段生成可复现 ID，避免把随机数引入 EDF 结果。"""

    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:20]}"


@dataclass(frozen=True, order=True)
class QueueKey:
    queue_type: str
    service_id: int = -1
    stage_id: int = -1
    location: int = -1
    routing_target_node: int = -1

    @classmethod
    def uplink(cls, service_id: int) -> "QueueKey":
        return cls("uplink", service_id=service_id)

    @classmethod
    def stage(
        cls,
        stage_id: int,
        location: int,
        routing_target_node: int | None,
    ) -> "QueueKey":
        return cls(
            "stage",
            stage_id=stage_id,
            location=location,
            routing_target_node=(
                -1 if routing_target_node is None else routing_target_node
            ),
        )

    @property
    def optional_target(self) -> int | None:
        return None if self.routing_target_node < 0 else self.routing_target_node


@dataclass(frozen=True)
class AllocationOperation:
    queue_key: QueueKey
    operation_type: str
    physical_bits: float
    routing_target_node: int | None = None
    destination_node_id: int | None = None
    link_id: int | None = None
    propagation_slots: int = 1

    def __post_init__(self) -> None:
        if self.operation_type not in {"uplink", "execute", "forward"}:
            raise ValueError("未知的队列操作类型。")
        if not math.isfinite(self.physical_bits) or self.physical_bits <= 0.0:
            raise ValueError("physical_bits 必须是正有限数。")

    @property
    def sort_key(self) -> tuple[str, int, int, int]:
        return (
            self.operation_type,
            -1 if self.routing_target_node is None else self.routing_target_node,
            -1 if self.destination_node_id is None else self.destination_node_id,
            -1 if self.link_id is None else self.link_id,
        )


@dataclass(frozen=True)
class FastAllocationPlan:
    expected_queue_version: int
    expected_lifecycle_version: int
    expected_failure_version: int
    expected_network_version: int
    current_slot: int
    operations: tuple[AllocationOperation, ...]


@dataclass(frozen=True)
class QueueCommitContext:
    queue_version: int
    lifecycle_version: int
    failure_version: int
    network_version: int
    current_slot: int
    effective_node_up: Mapping[int, bool]
    warm_instance_counts: Mapping[tuple[int, int], int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "effective_node_up",
            MappingProxyType(dict(self.effective_node_up)),
        )
        object.__setattr__(
            self,
            "warm_instance_counts",
            MappingProxyType(dict(self.warm_instance_counts)),
        )


@dataclass(frozen=True)
class QueueCommitResult:
    accepted: bool
    code: str
    snapshot: QueueSnapshot


@dataclass(frozen=True)
class SLAReport:
    """当前边界新增违约及累计完成状态。"""

    new_violation_count: int
    on_time_completed_count: int
    late_completed_count: int


class QueueStateManager:
    """唯一持有队列运行状态；所有资源服务通过原子计划提交。"""

    def __init__(
        self,
        flow_config: StageFlowConfig,
        *,
        slot_seconds: float,
        initial_batches: tuple[BatchRecord, ...] = (),
        initial_uplink_fragments: tuple[QueueFragment, ...] = (),
        initial_stage_fragments: tuple[QueueFragment, ...] = (),
        initial_in_transit: tuple[InTransitRecord, ...] = (),
        initial_completion_events: tuple[CompletionEvent, ...] = (),
        flow_absolute_tolerance_bits: float = 1e-6,
        flow_relative_tolerance: float = 1e-12,
    ) -> None:
        if not math.isfinite(slot_seconds) or slot_seconds <= 0.0:
            raise ValueError("slot_seconds 必须是正有限数。")
        if (
            not math.isfinite(flow_absolute_tolerance_bits)
            or flow_absolute_tolerance_bits < 0.0
            or not math.isfinite(flow_relative_tolerance)
            or flow_relative_tolerance < 0.0
        ):
            raise ValueError("流量守恒容差必须为非负有限数。")
        self.flow_config = flow_config
        self.slot_seconds = slot_seconds
        self.flow_absolute_tolerance_bits = flow_absolute_tolerance_bits
        self.flow_relative_tolerance = flow_relative_tolerance
        self._version = 0
        self._current_slot = 0
        self._batches = list(initial_batches)
        self._uplink_fragments = list(initial_uplink_fragments)
        self._stage_fragments = list(initial_stage_fragments)
        self._in_transit = list(initial_in_transit)
        self._completion_events = list(initial_completion_events)

    @classmethod
    def from_snapshot(
        cls,
        flow_config: StageFlowConfig,
        snapshot: QueueSnapshot,
        *,
        slot_seconds: float,
        flow_absolute_tolerance_bits: float = 1e-6,
        flow_relative_tolerance: float = 1e-12,
    ) -> "QueueStateManager":
        """为教师/验证创建完全独立的队列副本，不回写真实环境。"""

        if not isinstance(snapshot, QueueSnapshot):
            raise TypeError("snapshot 必须是 QueueSnapshot。")
        clone = cls(
            flow_config,
            slot_seconds=slot_seconds,
            initial_batches=snapshot.batches,
            initial_uplink_fragments=snapshot.uplink_fragments,
            initial_stage_fragments=snapshot.stage_fragments,
            initial_in_transit=snapshot.in_transit,
            initial_completion_events=snapshot.completion_events,
            flow_absolute_tolerance_bits=flow_absolute_tolerance_bits,
            flow_relative_tolerance=flow_relative_tolerance,
        )
        clone._version = snapshot.version
        clone._current_slot = snapshot.current_slot
        return clone

    def _flow_tolerance(self, reference_bits: float) -> float:
        """把求解器归一化残差换回 bit 后用于提交端的同口径审计。"""

        return max(
            self.flow_absolute_tolerance_bits,
            self.flow_relative_tolerance * max(1.0, abs(reference_bits)),
        )

    def snapshot(self) -> QueueSnapshot:
        return QueueSnapshot(
            version=self._version,
            current_slot=self._current_slot,
            batches=tuple(sorted(self._batches, key=lambda item: item.batch_id)),
            uplink_fragments=tuple(
                sorted(self._uplink_fragments, key=lambda item: item.fragment_id)
            ),
            stage_fragments=tuple(
                sorted(self._stage_fragments, key=lambda item: item.fragment_id)
            ),
            in_transit=tuple(
                sorted(self._in_transit, key=lambda item: item.transit_id)
            ),
            completion_events=tuple(
                sorted(self._completion_events, key=lambda item: item.event_id)
            ),
        )

    def admit_batch(
        self,
        batch_id: str,
        service_id: int,
        arrival_time: float,
        absolute_deadline_time: float,
        input_equivalent_bits: float,
    ) -> QueueSnapshot:
        if any(batch.batch_id == batch_id for batch in self._batches):
            raise ValueError("batch_id 必须唯一。")
        batch = BatchRecord(
            batch_id,
            service_id,
            arrival_time,
            absolute_deadline_time,
            total_input_equivalent_bits=input_equivalent_bits,
        )
        fragment = QueueFragment(
            _stable_id("fragment", batch_id, "uplink", self._current_slot),
            batch_id,
            service_id,
            -1,
            None,
            None,
            input_equivalent_bits,
            self._current_slot,
        )
        self._batches.append(batch)
        self._uplink_fragments.append(fragment)
        self._version += 1
        return self.snapshot()

    def begin_slot(self, current_slot: int) -> QueueSnapshot:
        """只提交本边界到达的在途片段和完成事件。"""

        if current_slot < self._current_slot:
            raise ValueError("队列时钟不能倒退。")
        # 先在局部变量完成全部审计；任何事件守恒错误都不能污染真实状态。
        due_transit = [
            item for item in self._in_transit if item.arrival_slot <= current_slot
        ]
        remaining_transit = [
            item for item in self._in_transit if item.arrival_slot > current_slot
        ]
        next_stage_fragments = [
            *self._stage_fragments,
            *(item.fragment for item in due_transit),
        ]

        due_completion = [
            item
            for item in self._completion_events
            if item.completion_slot <= current_slot
        ]
        remaining_completion = [
            item
            for item in self._completion_events
            if item.completion_slot > current_slot
        ]
        completed_by_batch: dict[str, float] = {}
        completion_slot_by_batch: dict[str, int] = {}
        for event in due_completion:
            completed_by_batch[event.batch_id] = (
                completed_by_batch.get(event.batch_id, 0.0)
                + event.input_equivalent_bits
            )
            completion_slot_by_batch[event.batch_id] = max(
                completion_slot_by_batch.get(event.batch_id, 0),
                event.completion_slot,
            )
        updated_batches: list[BatchRecord] = []
        for batch in self._batches:
            completed = (
                batch.completed_input_equivalent_bits
                + completed_by_batch.get(batch.batch_id, 0.0)
            )
            completion_tolerance = self._flow_tolerance(
                batch.total_input_equivalent_bits
            )
            if completed > batch.total_input_equivalent_bits + completion_tolerance:
                raise ValueError("completion flow 超过批次原始输入等效量。")
            completion_slot = batch.completion_slot
            if (
                abs(completed - batch.total_input_equivalent_bits)
                <= completion_tolerance
            ):
                completed = batch.total_input_equivalent_bits
                completion_slot = completion_slot_by_batch.get(
                    batch.batch_id,
                    completion_slot,
                )
            updated_batches.append(
                replace(
                    batch,
                    completed_input_equivalent_bits=completed,
                    completion_slot=completion_slot,
                )
            )
        unknown_completion_ids = set(completed_by_batch) - {
            batch.batch_id for batch in self._batches
        }
        if unknown_completion_ids:
            raise ValueError("completion flow 引用了未知批次。")
        self._in_transit = remaining_transit
        self._stage_fragments = next_stage_fragments
        self._completion_events = remaining_completion
        self._batches = updated_batches
        if current_slot != self._current_slot or due_transit or due_completion:
            self._version += 1
        self._current_slot = current_slot
        return self.snapshot()

    def _reject(self, code: str) -> QueueCommitResult:
        return QueueCommitResult(False, code, self.snapshot())

    def clear_invalid_routing_targets(
        self,
        valid_targets: set[tuple[int, int]],
    ) -> QueueSnapshot:
        """仅解绑已经到达的队列片段；不可变在途记录保持原样。"""

        updated: list[QueueFragment] = []
        changed = False
        for fragment in self._stage_fragments:
            target = fragment.routing_target_node
            if target is not None and (fragment.stage_id, target) not in valid_targets:
                updated.append(replace(fragment, routing_target_node=None))
                changed = True
            else:
                updated.append(fragment)
        if changed:
            self._stage_fragments = updated
            self._version += 1
        return self.snapshot()

    def audit_deadlines(self) -> SLAReport:
        """严格超过截止时间才记一次违约，已违约批次仍可继续完成。"""

        current_time = self._current_slot * self.slot_seconds
        new_violations = 0
        on_time_completed = 0
        late_completed = 0
        updated: list[BatchRecord] = []
        for batch in self._batches:
            violated = batch.violation_recorded
            if (
                batch.completion_slot is None
                and current_time > batch.absolute_deadline_time
                and not violated
            ):
                violated = True
                new_violations += 1
            if batch.completion_slot is not None:
                completion_time = batch.completion_slot * self.slot_seconds
                if completion_time <= batch.absolute_deadline_time:
                    on_time_completed += 1
                else:
                    late_completed += 1
                    if not violated:
                        new_violations += 1
                    violated = True
            updated.append(replace(batch, violation_recorded=violated))
        if updated != self._batches:
            self._batches = updated
            self._version += 1
        return SLAReport(
            new_violations,
            on_time_completed,
            late_completed,
        )

    def _is_stale(
        self,
        plan: FastAllocationPlan,
        context: QueueCommitContext,
    ) -> bool:
        return (
            plan.expected_queue_version != self._version
            or context.queue_version != self._version
            or plan.expected_lifecycle_version != context.lifecycle_version
            or plan.expected_failure_version != context.failure_version
            or plan.expected_network_version != context.network_version
            or plan.current_slot != self._current_slot
            or context.current_slot != self._current_slot
        )

    def _equivalent_amount(self, operation: AllocationOperation) -> float:
        if operation.queue_key.queue_type == "uplink":
            return operation.physical_bits
        return self.flow_config.equivalent_bits(
            operation.queue_key.stage_id,
            operation.physical_bits,
        )

    def _matching_fragments(self, key: QueueKey) -> list[QueueFragment]:
        source = (
            self._uplink_fragments
            if key.queue_type == "uplink"
            else self._stage_fragments
        )
        result: list[QueueFragment] = []
        for fragment in source:
            if fragment.available_slot > self._current_slot:
                continue
            if key.queue_type == "uplink":
                if fragment.stage_id == -1 and fragment.service_id == key.service_id:
                    result.append(fragment)
            elif (
                fragment.stage_id == key.stage_id
                and fragment.location == key.location
                and fragment.routing_target_node == key.optional_target
            ):
                result.append(fragment)
        deadlines = {batch.batch_id: batch for batch in self._batches}
        return sorted(
            result,
            key=lambda item: (
                deadlines[item.batch_id].absolute_deadline_time,
                deadlines[item.batch_id].arrival_time,
                item.batch_id,
                item.fragment_id,
            ),
        )

    def _operation_is_valid(
        self,
        operation: AllocationOperation,
        context: QueueCommitContext,
    ) -> str | None:
        key = operation.queue_key
        if key.queue_type == "uplink":
            if operation.operation_type != "uplink":
                return "INVALID_STAGE_TRANSITION"
            if operation.destination_node_id is None:
                return "INVALID_ALLOCATION_PLAN"
            return None
        if key.queue_type != "stage" or not (
            0 <= key.stage_id < self.flow_config.stage_count
        ):
            return "INVALID_STAGE_TRANSITION"
        if operation.operation_type == "execute":
            if key.optional_target is not None and key.optional_target != key.location:
                return "INVALID_STAGE_TRANSITION"
            if not context.effective_node_up.get(key.location, False):
                return "INVALID_ALLOCATION_PLAN"
            if context.warm_instance_counts.get((key.stage_id, key.location), 0) <= 0:
                return "INVALID_ALLOCATION_PLAN"
            return None
        if operation.operation_type != "forward":
            return "INVALID_STAGE_TRANSITION"
        if operation.destination_node_id is None or operation.link_id is None:
            return "INVALID_ALLOCATION_PLAN"
        if key.optional_target is not None:
            if operation.routing_target_node != key.optional_target:
                return "INVALID_STAGE_TRANSITION"
        elif operation.routing_target_node is None:
            return "INVALID_ALLOCATION_PLAN"
        return None

    def commit_allocation(
        self,
        plan: FastAllocationPlan,
        context: QueueCommitContext,
    ) -> QueueCommitResult:
        """在临时集合中完成全部 EDF 扣减，成功后一次性替换真实队列。"""

        if self._is_stale(plan, context):
            return self._reject("STALE_SNAPSHOT")
        if not plan.operations:
            return self._reject("INVALID_ALLOCATION_PLAN")
        for operation in plan.operations:
            error = self._operation_is_valid(operation, context)
            if error is not None:
                return self._reject(error)

        grouped: dict[QueueKey, list[AllocationOperation]] = {}
        for operation in plan.operations:
            grouped.setdefault(operation.queue_key, []).append(operation)

        consumed_ids: set[str] = set()
        residuals: list[QueueFragment] = []
        produced_stage: list[QueueFragment] = []
        produced_transit: list[InTransitRecord] = []
        produced_completion: list[CompletionEvent] = []
        sequence = 0

        for key in sorted(grouped):
            operations = sorted(grouped[key], key=lambda item: item.sort_key)
            quotas = [self._equivalent_amount(item) for item in operations]
            fragments = self._matching_fragments(key)
            available = sum(item.input_equivalent_bits for item in fragments)
            tolerance = self._flow_tolerance(available)
            if sum(quotas) > available + tolerance:
                return self._reject("FLOW_EXCEEDS_AVAILABLE_QUEUE")

            for fragment in fragments:
                remaining = fragment.input_equivalent_bits
                consumed = 0.0
                for index, operation in enumerate(operations):
                    if quotas[index] <= 0.0 or remaining <= 0.0:
                        continue
                    amount = min(remaining, quotas[index])
                    tail = remaining - amount
                    if amount > 0.0 and 0.0 < tail <= tolerance:
                        amount = remaining
                    child_id = _stable_id(
                        "fragment",
                        fragment.fragment_id,
                        self._version,
                        sequence,
                        operation.sort_key,
                    )
                    sequence += 1
                    if operation.operation_type == "uplink":
                        produced_stage.append(
                            QueueFragment(
                                child_id,
                                fragment.batch_id,
                                fragment.service_id,
                                0,
                                operation.destination_node_id,
                                None,
                                amount,
                                self._current_slot + 1,
                            )
                        )
                    elif operation.operation_type == "execute":
                        if key.stage_id + 1 < self.flow_config.stage_count:
                            produced_stage.append(
                                QueueFragment(
                                    child_id,
                                    fragment.batch_id,
                                    fragment.service_id,
                                    key.stage_id + 1,
                                    key.location,
                                    None,
                                    amount,
                                    self._current_slot + 1,
                                )
                            )
                        else:
                            produced_completion.append(
                                CompletionEvent(
                                    _stable_id(
                                        "completion",
                                        child_id,
                                        self._current_slot + 1,
                                    ),
                                    fragment.batch_id,
                                    self._current_slot + 1,
                                    amount,
                                )
                            )
                    else:
                        arrival_slot = self._current_slot + max(
                            1,
                            operation.propagation_slots,
                        )
                        forwarded = QueueFragment(
                            child_id,
                            fragment.batch_id,
                            fragment.service_id,
                            key.stage_id,
                            operation.destination_node_id,
                            operation.routing_target_node,
                            amount,
                            arrival_slot,
                        )
                        produced_transit.append(
                            InTransitRecord(
                                _stable_id("transit", child_id, arrival_slot),
                                forwarded,
                                key.location,
                                operation.destination_node_id,
                                operation.link_id,
                                self._current_slot,
                                arrival_slot,
                                amount,
                            )
                        )
                    remaining -= amount
                    consumed += amount
                    quotas[index] -= amount
                if consumed > 0.0:
                    consumed_ids.add(fragment.fragment_id)
                    if remaining > tolerance:
                        residuals.append(
                            replace(fragment, input_equivalent_bits=remaining)
                        )
            if any(quota > tolerance for quota in quotas):
                return self._reject("FLOW_CONSERVATION_VIOLATION")

        self._uplink_fragments = [
            item
            for item in self._uplink_fragments
            if item.fragment_id not in consumed_ids
        ]
        self._stage_fragments = [
            item
            for item in self._stage_fragments
            if item.fragment_id not in consumed_ids
        ]
        for residual in residuals:
            if residual.stage_id == -1:
                self._uplink_fragments.append(residual)
            else:
                self._stage_fragments.append(residual)
        self._stage_fragments.extend(produced_stage)
        self._in_transit.extend(produced_transit)
        self._completion_events.extend(produced_completion)
        self._version += 1
        return QueueCommitResult(True, "OK", self.snapshot())
