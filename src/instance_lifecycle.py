"""实例批次生命周期状态所有者、只读快照和原子部署提交。"""

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import math
from types import MappingProxyType

from src.failure_process import FailureSnapshot
from src.orchestration_config import PhaseAConfig


class LifecycleStatus(str, Enum):
    """活动实例仅处于启动中或温状态。"""

    STARTING = "starting"
    WARM = "warm"


@dataclass(frozen=True, order=True)
class InstanceBatch:
    """具有相同启动时刻和保留期限的一组实例。"""

    batch_id: str
    function_id: int
    node_id: int
    status: LifecycleStatus
    count: int
    ready_slot: int
    retention_deadline_slot: int

    def __post_init__(self) -> None:
        if not self.batch_id:
            raise ValueError("batch_id 不能为空。")
        if self.function_id < 0 or self.node_id < 0 or self.count <= 0:
            raise ValueError("实例批次编号必须非负且 count 必须为正。")
        if self.ready_slot < 0 or self.retention_deadline_slot < 0:
            raise ValueError("实例批次时隙不能为负。")


@dataclass(frozen=True)
class LifecycleSnapshot:
    """生命周期状态的不可变副本。"""

    version: int
    current_slot: int
    batches: tuple[InstanceBatch, ...]
    memory_used_mb_by_node: dict[int, float]

    def __post_init__(self) -> None:
        """隔离管理器内部字典，保证快照消费者只能读取。"""

        object.__setattr__(
            self,
            "memory_used_mb_by_node",
            MappingProxyType(dict(self.memory_used_mb_by_node)),
        )

    def _count(
        self,
        function_id: int,
        node_id: int,
        status: LifecycleStatus | None,
    ) -> int:
        return sum(
            batch.count
            for batch in self.batches
            if batch.function_id == function_id
            and batch.node_id == node_id
            and (status is None or batch.status is status)
        )

    def active_count(self, function_id: int, node_id: int) -> int:
        """STARTING 和 WARM 都是实际存在的活动实例。"""

        return self._count(function_id, node_id, None)

    def warm_count(self, function_id: int, node_id: int) -> int:
        return self._count(function_id, node_id, LifecycleStatus.WARM)

    def starting_count(self, function_id: int, node_id: int) -> int:
        return self._count(function_id, node_id, LifecycleStatus.STARTING)

    def locked_count(
        self,
        function_id: int,
        node_id: int,
        *,
        current_slot: int,
    ) -> int:
        """只有期限严格大于当前时隙的实例仍被保留承诺锁定。"""

        return sum(
            batch.count
            for batch in self.batches
            if batch.function_id == function_id
            and batch.node_id == node_id
            and current_slot < batch.retention_deadline_slot
        )


@dataclass(frozen=True)
class DeploymentTarget:
    function_id: int
    node_id: int
    target_count: int
    retention_slots: int


@dataclass(frozen=True)
class LifecycleDeploymentPlan:
    expected_lifecycle_version: int
    expected_failure_version: int
    current_slot: int
    targets: tuple[DeploymentTarget, ...]


@dataclass(frozen=True)
class LifecycleCommitResult:
    accepted: bool
    code: str
    snapshot: LifecycleSnapshot
    created_instance_count: int = 0
    created_batches: tuple[InstanceBatch, ...] = ()


def _derived_batch_id(
    parent: str,
    lifecycle_version: int,
    sequence: int,
    label: str,
) -> str:
    """用稳定哈希生成拆分/新建批次 ID，不消费任何随机流。"""

    payload = f"{parent}:{lifecycle_version}:{sequence}:{label}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


class InstanceLifecycleManager:
    """持有唯一实例批次状态，并通过局部副本完成原子更新。"""

    def __init__(
        self,
        *,
        config: PhaseAConfig,
        function_memory_mb: dict[int, float],
        initial_batches: tuple[InstanceBatch, ...] = (),
    ) -> None:
        self.config = config
        self.function_memory_mb = dict(function_memory_mb)
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in self.function_memory_mb.values()
        ):
            raise ValueError("VNF 内存必须是正有限数。")
        self._version = 0
        self._current_slot = 0
        self._batches = list(sorted(initial_batches))
        self._assert_batches_supported(self._batches)
        self._audit_memory(self._batches)

    @classmethod
    def from_snapshot(
        cls,
        *,
        config: PhaseAConfig,
        function_memory_mb: dict[int, float],
        snapshot: LifecycleSnapshot,
    ) -> "InstanceLifecycleManager":
        """为教师或验证创建生命周期副本，保留快照版本与当前时隙。"""

        if not isinstance(snapshot, LifecycleSnapshot):
            raise TypeError("snapshot 必须是 LifecycleSnapshot。")
        clone = cls(
            config=config,
            function_memory_mb=function_memory_mb,
            initial_batches=snapshot.batches,
        )
        clone._version = snapshot.version
        clone._current_slot = snapshot.current_slot
        if clone.snapshot() != snapshot:
            raise ValueError("LifecycleSnapshot 的派生内存与实例批次不一致。")
        return clone

    def _assert_batches_supported(self, batches: list[InstanceBatch]) -> None:
        for batch in batches:
            if (batch.function_id, batch.node_id) not in self.config.deployment_pairs:
                raise ValueError("实例批次引用了不允许部署的 VNF—节点组合。")

    def _memory_usage(
        self,
        batches: list[InstanceBatch] | None = None,
    ) -> dict[int, float]:
        result = {node_id: 0.0 for node_id in self.config.node_resources}
        for batch in self._batches if batches is None else batches:
            result[batch.node_id] += (
                batch.count * self.function_memory_mb[batch.function_id]
            )
        return result

    def _audit_memory(self, batches: list[InstanceBatch]) -> None:
        usage = self._memory_usage(batches)
        if any(
            used
            > self.config.node_resources[node_id].memory_capacity_mb + 1e-9
            for node_id, used in usage.items()
        ):
            raise MemoryError("实例批次超过节点内存容量。")

    def snapshot(self) -> LifecycleSnapshot:
        """返回按规范键排序的不可变生命周期快照。"""

        return LifecycleSnapshot(
            version=self._version,
            current_slot=self._current_slot,
            batches=tuple(sorted(self._batches)),
            memory_used_mb_by_node=MappingProxyType(self._memory_usage()),
        )

    def advance_to_slot(self, current_slot: int) -> LifecycleSnapshot:
        """在边界把所有 ready_slot 已到的启动实例转换为温实例。"""

        if current_slot < self._current_slot:
            raise ValueError("生命周期时钟不能倒退。")
        changed = False
        updated: list[InstanceBatch] = []
        for batch in self._batches:
            if (
                batch.status is LifecycleStatus.STARTING
                and current_slot >= batch.ready_slot
            ):
                updated.append(replace(batch, status=LifecycleStatus.WARM))
                changed = True
            else:
                updated.append(batch)
        self._current_slot = current_slot
        if changed:
            self._batches = sorted(updated)
            self._version += 1
        return self.snapshot()

    def apply_failure_snapshot(
        self,
        failure_snapshot: FailureSnapshot,
    ) -> LifecycleSnapshot:
        """有效状态刚失效时销毁节点全部批次；恢复不自动重建。"""

        unavailable = set(failure_snapshot.newly_unavailable_node_ids)
        if unavailable:
            remaining = [
                batch for batch in self._batches if batch.node_id not in unavailable
            ]
            if len(remaining) != len(self._batches):
                self._batches = remaining
                self._version += 1
        return self.snapshot()

    def _rejected(self, code: str) -> LifecycleCommitResult:
        return LifecycleCommitResult(False, code, self.snapshot())

    def preview_deployment(
        self,
        plan: LifecycleDeploymentPlan,
        failure_snapshot: FailureSnapshot,
    ) -> LifecycleCommitResult:
        """在独立管理器副本上执行同一提交逻辑，绝不修改真实状态。"""

        preview = InstanceLifecycleManager(
            config=self.config,
            function_memory_mb=self.function_memory_mb,
            initial_batches=tuple(self._batches),
        )
        preview._version = self._version
        preview._current_slot = self._current_slot
        return preview.commit_deployment(plan, failure_snapshot)

    def commit_deployment(
        self,
        plan: LifecycleDeploymentPlan,
        failure_snapshot: FailureSnapshot,
    ) -> LifecycleCommitResult:
        """在临时列表完成全部更新和审计，通过后才替换真实状态。"""

        if (
            plan.expected_lifecycle_version != self._version
            or plan.expected_failure_version != failure_snapshot.version
            or plan.current_slot != self._current_slot
            or failure_snapshot.time_slot != plan.current_slot
        ):
            return self._rejected("STALE_SNAPSHOT")
        target_keys = [(target.function_id, target.node_id) for target in plan.targets]
        if len(target_keys) != len(set(target_keys)):
            return self._rejected("INVALID_DEPLOYMENT_PLAN")
        effective_up = failure_snapshot.effective_node_up
        if set(effective_up) != set(self.config.node_resources):
            return self._rejected("STALE_SNAPSHOT")
        for target in plan.targets:
            key = (target.function_id, target.node_id)
            pair = self.config.deployment_pairs.get(key)
            if (
                pair is None
                or target.target_count not in pair.allowed_instance_counts
                or target.retention_slots not in self.config.retention_slot_options
            ):
                return self._rejected("INVALID_DEPLOYMENT_PLAN")
            if target.target_count > 0 and not effective_up[target.node_id]:
                return self._rejected("UNHEALTHY_TARGET_NODE")

        working = list(self._batches)
        created = 0
        created_batches: list[InstanceBatch] = []
        sequence = 0
        try:
            for target in sorted(plan.targets, key=lambda item: (item.function_id, item.node_id)):
                key_batches = sorted(
                    [
                        batch
                        for batch in working
                        if (batch.function_id, batch.node_id)
                        == (target.function_id, target.node_id)
                    ],
                    key=lambda batch: (
                        batch.retention_deadline_slot,
                        batch.ready_slot,
                        batch.batch_id,
                    ),
                )
                others = [batch for batch in working if batch not in key_batches]
                locked_count = sum(
                    batch.count
                    for batch in key_batches
                    if plan.current_slot < batch.retention_deadline_slot
                )
                existing_count = sum(batch.count for batch in key_batches)
                keep_count = max(locked_count, target.target_count)
                delete_count = max(0, existing_count - keep_count)
                kept: list[InstanceBatch] = []
                for batch in key_batches:
                    removable = (
                        plan.current_slot >= batch.retention_deadline_slot
                    )
                    removed = min(delete_count, batch.count) if removable else 0
                    delete_count -= removed
                    if batch.count - removed > 0:
                        kept.append(replace(batch, count=batch.count - removed))

                refresh_remaining = min(
                    target.target_count,
                    sum(batch.count for batch in kept),
                )
                refreshed: list[InstanceBatch] = []
                for batch in kept:
                    refresh_count = min(refresh_remaining, batch.count)
                    refresh_remaining -= refresh_count
                    if refresh_count == 0:
                        refreshed.append(batch)
                        continue
                    new_deadline = max(
                        batch.retention_deadline_slot,
                        max(plan.current_slot, batch.ready_slot)
                        + target.retention_slots,
                    )
                    if refresh_count == batch.count:
                        refreshed.append(
                            replace(batch, retention_deadline_slot=new_deadline)
                        )
                    else:
                        sequence += 1
                        refreshed.append(
                            replace(batch, count=batch.count - refresh_count)
                        )
                        refreshed.append(
                            replace(
                                batch,
                                batch_id=_derived_batch_id(
                                    batch.batch_id,
                                    self._version,
                                    sequence,
                                    "refresh",
                                ),
                                count=refresh_count,
                                retention_deadline_slot=new_deadline,
                            )
                        )

                new_count = max(0, target.target_count - existing_count)
                if new_count > 0:
                    pair = self.config.deployment_pairs[
                        (target.function_id, target.node_id)
                    ]
                    ready_slot = plan.current_slot + math.ceil(
                        pair.cold_start_seconds / self.config.fast_slot_seconds
                    )
                    sequence += 1
                    new_batch = InstanceBatch(
                            batch_id=_derived_batch_id(
                                f"{target.function_id}:{target.node_id}",
                                self._version,
                                sequence,
                                "create",
                            ),
                            function_id=target.function_id,
                            node_id=target.node_id,
                            status=(
                                LifecycleStatus.WARM
                                if ready_slot <= plan.current_slot
                                else LifecycleStatus.STARTING
                            ),
                            count=new_count,
                            ready_slot=ready_slot,
                            retention_deadline_slot=(
                                ready_slot + target.retention_slots
                            ),
                        )
                    refreshed.append(new_batch)
                    created_batches.append(new_batch)
                    created += new_count
                working = [*others, *refreshed]
            self._audit_memory(working)
        except MemoryError:
            return self._rejected("MEMORY_CAPACITY_EXCEEDED")

        self._batches = sorted(working)
        self._version += 1
        return LifecycleCommitResult(
            True,
            "OK",
            self.snapshot(),
            created_instance_count=created,
            created_batches=tuple(created_batches),
        )
