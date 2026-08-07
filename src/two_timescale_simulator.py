"""
two_timescale_simulator.py

本文件将双时间尺度控制器接入完整轨道边缘运行仿真。

慢时间尺度输入：

1. 预测请求负载；
2. 预测故障风险；
3. SFC可靠性目标；
4. 当前和下一MEC。

慢时间尺度输出：

1. 每个函数的副本数量；
2. Serverless实例保留策略；
3. 中心云使用策略。

快时间尺度输入：

1. 当前请求数量；
2. 当前可用MEC节点；
3. 当前副本部署计划；
4. 列车剩余驻留时间。

快时间尺度输出：

1. 当前温实例集合；
2. 请求执行节点；
3. 主备接管；
4. 冷启动；
5. 请求成功或失败。

同时统计：

1. 服务成功率；
2. SLA违反率；
3. 时延；
4. 冷启动；
5. 内存；
6. 副本重配置；
7. 综合系统成本。
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.entities import (
    SFCType,
    SlotConstraintAudit,
)
from src.failure_process import FailureProcess
from src.failure_risk_prediction import (
    FailureRiskProvider,
)
from src.fast_slot_executor import (
    FastSlotExecutor,
    FastSlotInput,
)
from src.mobility import TrainMobilityModel
from src.two_timescale_control import (
    SlowTimescaleDecision,
    SlowTimescaleState,
    retention_policy_to_legacy_mode,
)
from src.workload import DeterministicWorkload


class TwoTimescaleControllerProtocol(Protocol):
    """
    双时间尺度控制器接口。
    """

    def reset(self) -> None:
        ...

    def get_slow_decision(
        self,
        state: SlowTimescaleState,
    ) -> SlowTimescaleDecision:
        ...

@dataclass(frozen=True)
class TwoTimescaleCostWeights:
    """
    双时间尺度综合成本权重。
    """

    delay_cost_per_ms: float

    memory_cost_per_mb_second: float

    cold_start_cost_per_ms: float

    sla_violation_penalty_per_batch: float

    slow_decision_cost: float

    replica_reconfiguration_cost_per_function: float

    def __post_init__(self) -> None:
        values = (
            self.delay_cost_per_ms,
            self.memory_cost_per_mb_second,
            self.cold_start_cost_per_ms,
            self.sla_violation_penalty_per_batch,
            self.slow_decision_cost,
            self.replica_reconfiguration_cost_per_function,
        )

        if any(value < 0 for value in values):
            raise ValueError(
                "综合成本权重不能小于0。"
            )


@dataclass(frozen=True)
class TwoTimescaleSlotRecord:
    """
    一个快时隙的双时间尺度运行记录。
    """

    time_slot: int
    position_m: float

    serving_mec: int
    next_mec: int
    remaining_dwell_time_s: float

    handover_occurred: bool
    request_count: int

    predicted_request_rate: float
    predicted_failure_risk: float

    slow_decision_updated: bool
    slow_mode_changed: bool

    slow_mode: str
    use_redundancy: bool
    replica_count: int
    retention_policy: str
    cloud_policy: str
    cloud_used: bool
    slow_reason: str

    replica_plan_changed_function_count: int

    function_replica_node_ids: dict[
        int,
        tuple[int, ...],
    ]

    function_hot_node_ids: dict[
        int,
        tuple[int, ...],
    ]

    selected_execution_node_ids: tuple[int, ...]

    backup_activation_triggered: bool

    failover_function_ids: tuple[int, ...]
    cold_start_function_ids: tuple[int, ...]
    unavailable_function_ids: tuple[int, ...]

    request_success: bool | None

    transmission_delay_ms: float | None
    execution_delay_ms: float | None
    cold_start_delay_ms: float
    failover_delay_ms: float
    end_to_end_delay_ms: float | None

    deadline_met: bool | None

    active_instance_count: int
    active_memory_mb: float

    # 修复前的审计用于解释为什么触发快层搜索。
    initial_constraint_audit: SlotConstraintAudit

    # succeeded=None表示原方案可行，因而没有执行搜索。
    fast_repair_attempted: bool
    fast_repair_succeeded: bool | None
    fast_repair_reason: str
    fast_repair_evaluated_candidate_count: int

    # 最终采用方案的审计；修复失败时保留初审作为拒绝依据。
    constraint_audit: SlotConstraintAudit

    # 最终审计或运行路径不可行时为True，且不会进入SFC执行器。
    constraint_rejected: bool

    # 与强化学习环境共享的三项原始成本。
    total_run_cost: float
    total_route_cost: float
    total_cold_start_cost: float
    total_system_cost: float


@dataclass(frozen=True)
class TwoTimescaleSummary:
    """
    双时间尺度完整仿真汇总结果。
    """

    total_slots: int
    total_requests: int
    request_batches: int

    successful_requests: int
    failed_requests: int
    request_success_rate: float

    successful_batches: int
    failed_batches: int
    batch_success_rate: float

    failover_batches: int
    failover_function_stages: int

    cold_start_function_stages: int
    total_cold_start_delay_ms: float
    total_failover_delay_ms: float

    deadline_violations: int
    sla_violations: int
    sla_violation_rate: float

    fast_repair_attempts: int
    fast_repair_successes: int
    fast_repair_failures: int
    fast_repair_success_rate: float
    constraint_rejected_batches: int

    cloud_used_slots: int
    cloud_usage_rate: float

    average_successful_batch_delay_ms: float
    p95_successful_batch_delay_ms: float

    average_active_memory_mb: float
    peak_active_memory_mb: float
    total_active_memory_mb_seconds: float

    slow_decision_updates: int
    slow_mode_switches: int

    replica_reconfiguration_function_stages: int

    total_request_delay_cost: float
    total_memory_cost: float
    total_run_cost: float
    total_route_cost: float
    total_cold_start_cost: float
    total_sla_penalty: float
    total_slow_control_cost: float
    total_system_cost: float


@dataclass(frozen=True)
class TwoTimescaleRunResult:
    """
    一次双时间尺度完整仿真结果。
    """

    summary: TwoTimescaleSummary

    records: tuple[
        TwoTimescaleSlotRecord,
        ...,
    ]


class TwoTimescaleRuntimeSimulator:
    """
    双时间尺度轨道边缘SFC运行仿真器。
    """

    def __init__(
        self,
        mobility_model: TrainMobilityModel,
        workload: DeterministicWorkload,
        sfc: SFCType,
        controller: TwoTimescaleControllerProtocol,
        fast_slot_executor: FastSlotExecutor,
        failure_process: FailureProcess,
        failure_risk_provider: FailureRiskProvider,
        prediction_horizon_slots: int,
        slot_seconds: float,
        cost_weights: TwoTimescaleCostWeights,
    ) -> None:
        """
        创建双时间尺度完整仿真器。

        仿真器只负责慢层时序、列车移动和指标汇总；副本规划、
        快层决策、约束修复、SFC执行和三项成本统一交给
        ``fast_slot_executor``，避免规则方案与RL环境出现两套口径。
        """

        if prediction_horizon_slots <= 0:
            raise ValueError(
                "负载预测窗口必须大于0。"
            )

        if slot_seconds <= 0:
            raise ValueError(
                "时隙长度必须大于0。"
            )

        self.mobility_model = mobility_model
        self.workload = workload
        self.sfc = sfc
        self.controller = controller
        self.fast_slot_executor = fast_slot_executor
        self.failure_process = failure_process
        self.failure_risk_provider = (
            failure_risk_provider
        )
        self.prediction_horizon_slots = (
            prediction_horizon_slots
        )
        self.slot_seconds = slot_seconds
        self.cost_weights = cost_weights

    def _predict_request_rate(
        self,
        time_slot: int,
    ) -> float:
        """
        预测未来慢周期内的平均请求数量。

        当前调试版本使用确定性请求轨迹做精确前视。
        """

        future_counts = [
            self.workload.request_count(
                time_slot + offset
            )
            for offset
            in range(
                self.prediction_horizon_slots
            )
        ]

        return float(
            np.mean(future_counts)
        )

    def run(
        self,
    ) -> TwoTimescaleRunResult:
        """
        从铁路起点运行至终点。
        """

        self.controller.reset()
        self.failure_process.reset()

        train_state = self.mobility_model.reset()

        records: list[
            TwoTimescaleSlotRecord
        ] = []

        previous_serving_mec: int | None = None

        previous_slow_decision_slot: int | None = None
        # 使用结构化三元组判断慢策略是否改变，避免继续依赖旧枚举。
        previous_slow_policy: (
            tuple[object, ...] | None
        ) = None

        previous_candidate_map: (
            dict[int, tuple[int, ...]] | None
        ) = None

        while True:
            handover_occurred = (
                previous_serving_mec is not None
                and train_state.serving_mec
                != previous_serving_mec
            )

            request_count = (
                self.workload.request_count(
                    train_state.time_slot
                )
            )

            predicted_request_rate = (
                self._predict_request_rate(
                    train_state.time_slot
                )
            )

            predicted_failure_risk = (
                self.failure_risk_provider.predict(
                    time_slot=train_state.time_slot,
                    train_state=train_state,
                )
            )

            slow_state = SlowTimescaleState(
                time_slot=train_state.time_slot,
                serving_mec=train_state.serving_mec,
                next_mec=train_state.next_mec,
                predicted_request_rate=(
                    predicted_request_rate
                ),
                predicted_failure_risk=(
                    predicted_failure_risk
                ),
                reliability_target=(
                    self.sfc.reliability_target
                ),
            )

            slow_decision = (
                self.controller.get_slow_decision(
                    slow_state
                )
            )

            slow_decision_updated = (
                slow_decision.decision_slot
                != previous_slow_decision_slot
            )

            current_slow_policy = (
                slow_decision.replica_count,
                slow_decision.retention_policy,
                slow_decision.cloud_policy,
            )

            slow_mode_changed = (
                previous_slow_policy is not None
                and current_slow_policy
                != previous_slow_policy
            )

            infrastructure_state = (
                self.failure_process.state_for_slot(
                    train_state.time_slot
                )
            )
            # 规则仿真器不再自行构造快层决定。这个调用是每个快时隙
            # 唯一的规划、审计、修复、执行和成本数据来源。
            fast_result = self.fast_slot_executor.execute(
                FastSlotInput(
                    train_state=train_state,
                    request_count=request_count,
                    infrastructure_state=(
                        infrastructure_state
                    ),
                    slow_decision=slow_decision,
                    previous_candidate_map=(
                        previous_candidate_map
                    ),
                )
            )
            candidate_map = dict(
                fast_result.function_replica_node_ids
            )

            records.append(
                TwoTimescaleSlotRecord(
                    time_slot=train_state.time_slot,
                    position_m=train_state.position_m,
                    serving_mec=(
                        train_state.serving_mec
                    ),
                    next_mec=train_state.next_mec,
                    remaining_dwell_time_s=(
                        train_state
                        .remaining_dwell_time_s
                    ),
                    handover_occurred=(
                        handover_occurred
                    ),
                    request_count=request_count,
                    predicted_request_rate=(
                        predicted_request_rate
                    ),
                    predicted_failure_risk=(
                        predicted_failure_risk
                    ),
                    slow_decision_updated=(
                        slow_decision_updated
                    ),
                    slow_mode_changed=(
                        slow_mode_changed
                    ),
                    slow_mode=(
                        # 暂时保留旧结果列名与取值，便于历史实验对比。
                        retention_policy_to_legacy_mode(
                            slow_decision.retention_policy,
                            slow_decision.replica_count,
                        ).value
                    ),
                    use_redundancy=(
                        slow_decision.use_redundancy
                    ),
                    replica_count=(
                        slow_decision.replica_count
                    ),
                    retention_policy=(
                        slow_decision
                        .retention_policy.name.lower()
                    ),
                    cloud_policy=(
                        slow_decision
                        .cloud_policy.name.lower()
                    ),
                    cloud_used=fast_result.used_cloud,
                    slow_reason=(
                        slow_decision.reason
                    ),
                    replica_plan_changed_function_count=(
                        fast_result.plan_change_count
                    ),
                    function_replica_node_ids=(
                        candidate_map
                    ),
                    function_hot_node_ids=(
                        fast_result.function_hot_node_ids
                    ),
                    selected_execution_node_ids=(
                        fast_result
                        .selected_execution_node_ids
                    ),
                    backup_activation_triggered=(
                        fast_result
                        .backup_activation_triggered
                    ),
                    failover_function_ids=(
                        fast_result
                        .failover_function_ids
                    ),
                    cold_start_function_ids=(
                        fast_result
                        .cold_start_function_ids
                    ),
                    unavailable_function_ids=(
                        fast_result
                        .unavailable_function_ids
                    ),
                    request_success=(
                        fast_result.request_success
                    ),
                    transmission_delay_ms=(
                        fast_result.transmission_delay_ms
                    ),
                    execution_delay_ms=(
                        fast_result.execution_delay_ms
                    ),
                    cold_start_delay_ms=(
                        fast_result.cold_start_delay_ms
                    ),
                    failover_delay_ms=(
                        fast_result.failover_delay_ms
                    ),
                    end_to_end_delay_ms=(
                        fast_result.end_to_end_delay_ms
                    ),
                    deadline_met=fast_result.deadline_met,
                    active_instance_count=(
                        fast_result.active_instance_count
                    ),
                    active_memory_mb=(
                        fast_result.active_memory_mb
                    ),
                    initial_constraint_audit=(
                        fast_result.initial_audit
                    ),
                    fast_repair_attempted=(
                        fast_result.fast_repair_attempted
                    ),
                    fast_repair_succeeded=(
                        fast_result.fast_repair_succeeded
                    ),
                    fast_repair_reason=(
                        fast_result.fast_repair_reason
                    ),
                    fast_repair_evaluated_candidate_count=(
                        fast_result
                        .fast_repair_evaluated_candidate_count
                    ),
                    constraint_audit=(
                        fast_result.final_audit
                    ),
                    constraint_rejected=(
                        fast_result.constraint_rejected
                    ),
                    total_run_cost=fast_result.run_cost,
                    total_route_cost=fast_result.route_cost,
                    total_cold_start_cost=(
                        fast_result.cold_start_cost
                    ),
                    total_system_cost=(
                        fast_result.total_cost
                    ),
                )
            )

            previous_slow_decision_slot = (
                slow_decision.decision_slot
            )

            previous_slow_policy = current_slow_policy

            previous_candidate_map = dict(
                candidate_map
            )

            if self.mobility_model.finished:
                break

            previous_serving_mec = (
                train_state.serving_mec
            )

            train_state = (
                self.mobility_model.step()
            )

        return TwoTimescaleRunResult(
            summary=self._build_summary(
                records
            ),
            records=tuple(records),
        )

    def _build_summary(
        self,
        records: list[
            TwoTimescaleSlotRecord
        ],
    ) -> TwoTimescaleSummary:
        """
        计算汇总指标和综合成本。
        """

        active_records = [
            record
            for record in records
            if record.request_count > 0
        ]

        successful_records = [
            record
            for record in active_records
            if record.request_success is True
        ]

        failed_records = [
            record
            for record in active_records
            if record.request_success is False
        ]

        total_requests = sum(
            record.request_count
            for record in active_records
        )

        successful_requests = sum(
            record.request_count
            for record in successful_records
        )

        failed_requests = sum(
            record.request_count
            for record in failed_records
        )

        request_batches = len(active_records)
        successful_batches = len(
            successful_records
        )
        failed_batches = len(failed_records)

        request_success_rate = (
            successful_requests / total_requests
            if total_requests > 0
            else 0.0
        )

        batch_success_rate = (
            successful_batches / request_batches
            if request_batches > 0
            else 0.0
        )

        failover_batches = sum(
            int(
                len(record.failover_function_ids)
                > 0
            )
            for record in successful_records
        )

        failover_function_stages = sum(
            len(record.failover_function_ids)
            for record in successful_records
        )

        cold_start_function_stages = sum(
            len(record.cold_start_function_ids)
            for record in successful_records
        )

        total_cold_start_delay_ms = sum(
            record.cold_start_delay_ms
            for record in successful_records
        )

        total_failover_delay_ms = sum(
            record.failover_delay_ms
            for record in successful_records
        )

        deadline_violations = sum(
            int(record.deadline_met is False)
            for record in successful_records
        )

        sla_violations = (
            failed_batches
            + deadline_violations
        )

        sla_violation_rate = (
            sla_violations / request_batches
            if request_batches > 0
            else 0.0
        )

        # 修复统计覆盖全部快时隙，包括无请求时的后台部署维护；
        # 约束拒绝批次数在下方单独限制为真实请求批次。
        fast_repair_attempts = sum(
            int(record.fast_repair_attempted)
            for record in records
        )
        fast_repair_successes = sum(
            int(record.fast_repair_succeeded is True)
            for record in records
        )
        fast_repair_failures = sum(
            int(record.fast_repair_succeeded is False)
            for record in records
        )
        fast_repair_success_rate = (
            fast_repair_successes
            / fast_repair_attempts
            if fast_repair_attempts > 0
            else 0.0
        )

        # 只统计有真实请求且因硬约束无解而被拒绝的批次；
        # 该批次已包含在failed_batches中，不额外增加SLA违反数。
        constraint_rejected_batches = sum(
            int(
                record.request_count > 0
                and record.constraint_rejected
            )
            for record in records
        )

        cloud_used_slots = sum(
            int(record.cloud_used)
            for record in records
        )
        cloud_usage_rate = (
            cloud_used_slots / len(records)
            if records
            else 0.0
        )

        successful_delays = [
            record.end_to_end_delay_ms
            for record in successful_records
            if record.end_to_end_delay_ms
            is not None
        ]

        if successful_delays:
            average_delay = float(
                np.mean(successful_delays)
            )

            p95_delay = float(
                np.percentile(
                    successful_delays,
                    95,
                )
            )
        else:
            average_delay = 0.0
            p95_delay = 0.0

        if records:
            average_memory = float(
                np.mean(
                    [
                        record.active_memory_mb
                        for record in records
                    ]
                )
            )

            peak_memory = max(
                record.active_memory_mb
                for record in records
            )
        else:
            average_memory = 0.0
            peak_memory = 0.0

        total_memory_mb_seconds = sum(
            record.active_memory_mb
            * self.slot_seconds
            for record in records
        )

        slow_decision_updates = sum(
            int(record.slow_decision_updated)
            for record in records
        )

        slow_mode_switches = sum(
            int(record.slow_mode_changed)
            for record in records
        )

        replica_reconfiguration_function_stages = sum(
            record
            .replica_plan_changed_function_count
            for record in records
        )

        total_request_delay_ms = sum(
            record.end_to_end_delay_ms
            for record in successful_records
            if record.end_to_end_delay_ms
            is not None
        )

        total_request_delay_cost = (
            total_request_delay_ms
            * self.cost_weights
            .delay_cost_per_ms
        )

        total_memory_cost = (
            total_memory_mb_seconds
            * self.cost_weights
            .memory_cost_per_mb_second
        )

        # 这三项由共享执行器逐时隙直接计价，规则仿真和RL环境
        # 因而使用完全相同的总成本定义。
        total_run_cost = sum(
            record.total_run_cost
            for record in records
        )
        total_route_cost = sum(
            record.total_route_cost
            for record in records
        )
        total_cold_start_cost = sum(
            record.total_cold_start_cost
            for record in records
        )

        total_sla_penalty = (
            sla_violations
            * self.cost_weights
            .sla_violation_penalty_per_batch
        )

        total_slow_control_cost = (
            slow_decision_updates
            * self.cost_weights
            .slow_decision_cost
            +
            replica_reconfiguration_function_stages
            * self.cost_weights
            .replica_reconfiguration_cost_per_function
        )

        total_system_cost = (
            total_run_cost
            + total_route_cost
            + total_cold_start_cost
        )

        return TwoTimescaleSummary(
            total_slots=len(records),
            total_requests=total_requests,
            request_batches=request_batches,
            successful_requests=(
                successful_requests
            ),
            failed_requests=failed_requests,
            request_success_rate=(
                request_success_rate
            ),
            successful_batches=(
                successful_batches
            ),
            failed_batches=failed_batches,
            batch_success_rate=(
                batch_success_rate
            ),
            failover_batches=failover_batches,
            failover_function_stages=(
                failover_function_stages
            ),
            cold_start_function_stages=(
                cold_start_function_stages
            ),
            total_cold_start_delay_ms=(
                total_cold_start_delay_ms
            ),
            total_failover_delay_ms=(
                total_failover_delay_ms
            ),
            deadline_violations=(
                deadline_violations
            ),
            sla_violations=sla_violations,
            sla_violation_rate=(
                sla_violation_rate
            ),
            fast_repair_attempts=(
                fast_repair_attempts
            ),
            fast_repair_successes=(
                fast_repair_successes
            ),
            fast_repair_failures=(
                fast_repair_failures
            ),
            fast_repair_success_rate=(
                fast_repair_success_rate
            ),
            constraint_rejected_batches=(
                constraint_rejected_batches
            ),
            cloud_used_slots=cloud_used_slots,
            cloud_usage_rate=cloud_usage_rate,
            average_successful_batch_delay_ms=(
                average_delay
            ),
            p95_successful_batch_delay_ms=(
                p95_delay
            ),
            average_active_memory_mb=(
                average_memory
            ),
            peak_active_memory_mb=(
                peak_memory
            ),
            total_active_memory_mb_seconds=(
                total_memory_mb_seconds
            ),
            slow_decision_updates=(
                slow_decision_updates
            ),
            slow_mode_switches=(
                slow_mode_switches
            ),
            replica_reconfiguration_function_stages=(
                replica_reconfiguration_function_stages
            ),
            total_request_delay_cost=(
                total_request_delay_cost
            ),
            total_memory_cost=(
                total_memory_cost
            ),
            total_run_cost=total_run_cost,
            total_route_cost=total_route_cost,
            total_cold_start_cost=(
                total_cold_start_cost
            ),
            total_sla_penalty=(
                total_sla_penalty
            ),
            total_slow_control_cost=(
                total_slow_control_cost
            ),
            total_system_cost=(
                total_system_cost
            ),
        )
