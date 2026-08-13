"""阶段 A 的固定时隙边界顺序；实际状态仍由故障与生命周期层持有。"""

from dataclasses import dataclass

from src.cost_ledger import CostLedger, CostLedgerEntry
from src.failure_process import FailureProcess, FailureSnapshot
from src.instance_lifecycle import (
    InstanceLifecycleManager,
    LifecycleCommitResult,
    LifecycleDeploymentPlan,
    LifecycleSnapshot,
)


@dataclass(frozen=True)
class PhaseASlotResult:
    """一次边界推进后供后续阶段读取的两份不可变快照。"""

    failure_snapshot: FailureSnapshot
    lifecycle_snapshot: LifecycleSnapshot


class PhaseASlotCoordinator:
    """按规范顺序调用状态所有者，本身不保存重复环境状态。"""

    def __init__(
        self,
        failure_process: FailureProcess,
        lifecycle: InstanceLifecycleManager,
        cost_ledger: CostLedger,
    ) -> None:
        self.failure_process = failure_process
        self.lifecycle = lifecycle
        self.cost_ledger = cost_ledger
        self._current_failure: FailureSnapshot | None = None
        self._costing_finalized = False

    def begin_slot(self, current_slot: int) -> PhaseASlotResult:
        """先完成到期冷启动，再更新故障并立即销毁失效节点实例。"""

        self.lifecycle.advance_to_slot(current_slot)
        failure = self.failure_process.state_for_slot(current_slot)
        lifecycle = self.lifecycle.apply_failure_snapshot(failure)
        self._current_failure = failure
        self._costing_finalized = False
        return PhaseASlotResult(failure, lifecycle)

    def commit_deployment(
        self,
        plan: LifecycleDeploymentPlan,
    ) -> LifecycleCommitResult:
        """提交当前快照计划，并只对真正新建的批次登记事件费。"""

        if self._current_failure is None:
            raise RuntimeError("必须先调用 begin_slot()。")
        result = self.lifecycle.commit_deployment(plan, self._current_failure)
        if not result.accepted:
            return result
        config = self.lifecycle.config
        entries: list[CostLedgerEntry] = []
        for batch in result.created_batches:
            pair = config.deployment_pairs[(batch.function_id, batch.node_id)]
            for cost_type, unit_price in (
                ("deployment", pair.deployment_cost_per_instance),
                ("cold", pair.cold_start_cost_per_instance),
            ):
                entries.append(
                    CostLedgerEntry(
                        slot=plan.current_slot,
                        slow_frame_index=(
                            plan.current_slot // config.slow_frame_slots
                        ),
                        cost_type=cost_type,
                        function_id=batch.function_id,
                        node_id=batch.node_id,
                        instance_batch_id=batch.batch_id,
                        count=batch.count,
                        physical_quantity=float(batch.count),
                        unit_price=unit_price,
                        amount=batch.count * unit_price,
                        price_version=config.lifecycle.price_version,
                    )
                )
        self.cost_ledger.append_all(tuple(entries))
        return result

    def finalize_slot_costing(self) -> tuple[CostLedgerEntry, ...]:
        """按故障和部署提交后的活动实例，对当前服务区间计费一次。"""

        if self._current_failure is None:
            raise RuntimeError("必须先调用 begin_slot()。")
        if self._costing_finalized:
            raise RuntimeError("当前时隙已经完成计费。")
        snapshot = self.lifecycle.snapshot()
        active_counts: dict[tuple[int, int], int] = {}
        for batch in snapshot.batches:
            key = (batch.function_id, batch.node_id)
            active_counts[key] = active_counts.get(key, 0) + batch.count
        config = self.lifecycle.config
        running = CostLedger.running_entries(
            slot=snapshot.current_slot,
            slow_frame_index=(
                snapshot.current_slot // config.slow_frame_slots
            ),
            slot_seconds=config.fast_slot_seconds,
            active_counts=active_counts,
            running_prices={
                key: pair.running_cost_per_instance_second
                for key, pair in config.deployment_pairs.items()
            },
            price_version=config.lifecycle.price_version,
        )
        self.cost_ledger.append_all(running)
        self._costing_finalized = True
        return running
