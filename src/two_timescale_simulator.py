"""
two_timescale_simulator.py

本文件将双时间尺度控制器接入完整轨道边缘运行仿真。

慢时间尺度输入：

1. 预测请求负载；
2. 预测故障风险；
3. SFC可靠性目标；
4. 当前和下一MEC。

慢时间尺度输出：

1. 是否启用跨故障域冗余；
2. 副本数量；
3. 单副本、冷备或热备模式。

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
    ServerlessFunction,
    SFCType,
    SlotConstraintAudit,
    TrainState,
)
from src.failure_process import FailureProcess
from src.failure_risk_prediction import (
    FailureRiskProvider,
)
from src.mobility import TrainMobilityModel
from src.network import LinearMECNetwork
from src.redundancy_placement import (
    ReplicaPlacementPlan,
)
from src.reliability import FaultDomainReliabilityModel
from src.sfc_execution import execute_sfc_batch
from src.topology import LinearRailTopology
from src.two_timescale_control import (
    FastTimescaleDecision,
    FastTimescaleState,
    SlowTimescaleDecision,
    SlowTimescaleState,
    StandbyMode,
)
from src.workload import DeterministicWorkload


class ReplicaPlannerProtocol(Protocol):
    """
    副本规划器接口。
    """

    def plan(
        self,
        sfc: SFCType,
        train_state: TrainState,
        topology: LinearRailTopology,
    ) -> ReplicaPlacementPlan:
        ...


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

    def get_fast_decision(
        self,
        state: FastTimescaleState,
        slow_decision: SlowTimescaleDecision,
    ) -> FastTimescaleDecision:
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

    # 第一阶段只记录约束违反，不改变本时隙的请求执行结果。
    constraint_audit: SlotConstraintAudit


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
        topology: LinearRailTopology,
        network: LinearMECNetwork,
        mobility_model: TrainMobilityModel,
        workload: DeterministicWorkload,
        functions: list[ServerlessFunction],
        sfc: SFCType,
        controller: TwoTimescaleControllerProtocol,
        single_replica_planner: ReplicaPlannerProtocol,
        redundant_replica_planner: ReplicaPlannerProtocol,
        failure_process: FailureProcess,
        failure_risk_provider: FailureRiskProvider,
        reliability_model: FaultDomainReliabilityModel,
        prediction_horizon_slots: int,
        input_size_mb_per_request: float,
        slot_seconds: float,
        failover_delay_ms_per_function: float,
        cost_weights: TwoTimescaleCostWeights,
        return_result_to_source: bool = True,
    ) -> None:
        """
        创建双时间尺度完整仿真器。
        """

        if len(functions) == 0:
            raise ValueError(
                "至少需要一个Serverless函数。"
            )

        if prediction_horizon_slots <= 0:
            raise ValueError(
                "负载预测窗口必须大于0。"
            )

        if input_size_mb_per_request < 0:
            raise ValueError(
                "单请求输入数据量不能小于0。"
            )

        if slot_seconds <= 0:
            raise ValueError(
                "时隙长度必须大于0。"
            )

        if failover_delay_ms_per_function < 0:
            raise ValueError(
                "故障切换时延不能小于0。"
            )

        function_map: dict[
            int,
            ServerlessFunction,
        ] = {}

        for function in functions:
            if function.function_id in function_map:
                raise ValueError(
                    f"函数编号{function.function_id}重复。"
                )

            function_map[
                function.function_id
            ] = function

        for function_id in sfc.function_ids:
            if function_id not in function_map:
                raise KeyError(
                    f"缺少function_id={function_id}。"
                )

        self.topology = topology
        self.network = network
        self.mobility_model = mobility_model
        self.workload = workload

        self.functions = functions
        self.function_map = function_map
        self.sfc = sfc

        self.controller = controller

        self.single_replica_planner = (
            single_replica_planner
        )

        self.redundant_replica_planner = (
            redundant_replica_planner
        )

        self.failure_process = failure_process

        self.failure_risk_provider = (
            failure_risk_provider
        )

        # 复用 reliability.py 中考虑共享故障域的精确模型，
        # 避免仿真器内部出现第二套不一致的可靠性公式。
        self.reliability_model = reliability_model

        self.prediction_horizon_slots = (
            prediction_horizon_slots
        )

        self.input_size_mb_per_request = (
            input_size_mb_per_request
        )

        self.slot_seconds = slot_seconds

        self.failover_delay_ms_per_function = (
            failover_delay_ms_per_function
        )

        self.cost_weights = cost_weights

        self.return_result_to_source = (
            return_result_to_source
        )

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

    def _build_candidate_map(
        self,
        train_state: TrainState,
        slow_decision: SlowTimescaleDecision,
    ) -> dict[int, tuple[int, ...]]:
        """
        根据慢动作选择单副本或跨域双副本规划器。
        """

        if slow_decision.use_redundancy:
            planner = (
                self.redundant_replica_planner
            )
        else:
            planner = (
                self.single_replica_planner
            )

        plan = planner.plan(
            sfc=self.sfc,
            train_state=train_state,
            topology=self.topology,
        )

        candidate_map = {
            function_id: tuple(node_ids)
            for function_id, node_ids
            in plan.function_replica_node_ids.items()
        }

        if set(candidate_map) != set(
            self.sfc.function_ids
        ):
            raise ValueError(
                "副本计划与SFC函数集合不一致。"
            )

        return candidate_map

    def _count_plan_changes(
        self,
        previous_map: (
            dict[int, tuple[int, ...]] | None
        ),
        current_map: dict[
            int,
            tuple[int, ...],
        ],
    ) -> int:
        """
        统计有多少个函数的副本计划发生变化。

        首个时隙视为初始部署，
        因此所有函数都计为一次配置操作。
        """

        if previous_map is None:
            return len(self.sfc.function_ids)

        return sum(
            int(
                previous_map[function_id]
                != current_map[function_id]
            )
            for function_id
            in self.sfc.function_ids
        )

    def _calculate_active_memory(
        self,
        function_hot_node_ids: dict[
            int,
            tuple[int, ...],
        ],
        cold_activated_pairs: set[
            tuple[int, int],
        ],
    ) -> tuple[int, float]:
        """
        统计温实例和本时隙临时启动的冷备用内存。
        """

        active_pairs: set[
            tuple[int, int]
        ] = set()

        for function_id, node_ids in (
            function_hot_node_ids.items()
        ):
            for node_id in node_ids:
                active_pairs.add(
                    (
                        function_id,
                        node_id,
                    )
                )

        active_pairs.update(
            cold_activated_pairs
        )

        active_memory_mb = sum(
            self.function_map[
                function_id
            ].memory_mb
            for function_id, _
            in active_pairs
        )

        return (
            len(active_pairs),
            active_memory_mb,
        )

    def _audit_slot_constraints(
        self,
        request_count: int,
        candidate_map: dict[
            int,
            tuple[int, ...],
        ],
        selected_execution_node_ids: tuple[int, ...],
        request_success: bool | None,
        function_hot_node_ids: dict[
            int,
            tuple[int, ...],
        ],
        cold_activated_pairs: set[
            tuple[int, int]
        ],
    ) -> SlotConstraintAudit:
        """
        审计本时隙的资源、部署计划和可靠性约束。

        当前阶段采用“只记录、不拦截”的方式：即使发现资源超限
        或可靠性不足，也不会在这里改写请求成功结果。这样可以先
        观察现有策略产生了哪些不可行方案，再由后续优化器修复。
        """

        node_map = {
            site.node.node_id: site.node
            for site in self.topology.sites
        }

        required_function_ids = set(
            self.sfc.function_ids
        )
        supplied_function_ids = set(candidate_map)

        missing_function_ids = tuple(
            sorted(
                required_function_ids
                - supplied_function_ids
            )
        )
        extra_function_ids = tuple(
            sorted(
                supplied_function_ids
                - required_function_ids
            )
        )

        invalid_replica_node_ids: set[int] = set()

        replica_plan_valid = not (
            missing_function_ids
            or extra_function_ids
        )

        reasons: list[str] = []

        if missing_function_ids:
            reasons.append(
                "副本计划缺少函数"
                f"{list(missing_function_ids)}。"
            )

        if extra_function_ids:
            reasons.append(
                "副本计划包含额外函数"
                f"{list(extra_function_ids)}。"
            )

        for function_id in self.sfc.function_ids:
            node_ids = candidate_map.get(
                function_id,
                (),
            )

            if not node_ids:
                replica_plan_valid = False
                reasons.append(
                    f"函数{function_id}没有部署副本。"
                )

            if len(node_ids) != len(set(node_ids)):
                replica_plan_valid = False
                reasons.append(
                    f"函数{function_id}存在重复副本节点。"
                )

            for node_id in node_ids:
                if node_id not in node_map:
                    invalid_replica_node_ids.add(
                        node_id
                    )
                    replica_plan_valid = False

        if invalid_replica_node_ids:
            reasons.append(
                "副本计划引用未知节点"
                f"{sorted(invalid_replica_node_ids)}。"
            )

        # 只有已经保持温热，或本时隙实际发生冷启动的实例，
        # 才占用活动实例内存。尚未启动的冷备用不在这里计费。
        active_pairs = {
            (function_id, node_id)
            for function_id, node_ids
            in function_hot_node_ids.items()
            for node_id in node_ids
        }

        # 使用集合合并可以避免同一实例同时出现在“温实例”和
        # “本时隙冷启动实例”中时被重复计算。
        active_pairs.update(cold_activated_pairs)

        node_memory_demand_mb: dict[
            int,
            float,
        ] = {}

        for function_id, node_id in active_pairs:
            node_memory_demand_mb[node_id] = (
                node_memory_demand_mb.get(
                    node_id,
                    0.0,
                )
                + self.function_map[
                    function_id
                ].memory_mb
            )

        node_cpu_demand: dict[int, float] = {}

        # 失败或没有请求的时隙没有实际函数执行，因此 CPU 需求为零。
        if request_success is True:
            for function_id, node_id in zip(
                self.sfc.function_ids,
                selected_execution_node_ids,
            ):
                node_cpu_demand[node_id] = (
                    node_cpu_demand.get(
                        node_id,
                        0.0,
                    )
                    + self.function_map[
                        function_id
                    ].cpu_demand(request_count)
                )

        cpu_violation_node_ids = tuple(
            sorted(
                node_id
                for node_id, demand
                in node_cpu_demand.items()
                if demand
                > node_map[node_id].cpu_capacity
            )
        )

        memory_violation_node_ids = tuple(
            sorted(
                node_id
                for node_id, demand
                in node_memory_demand_mb.items()
                if demand
                > node_map[
                    node_id
                ].memory_capacity_mb
            )
        )

        for node_id in cpu_violation_node_ids:
            reasons.append(
                f"节点{node_id}的CPU需求"
                f"{node_cpu_demand[node_id]:.3f}超过容量"
                f"{node_map[node_id].cpu_capacity:.3f}。"
            )

        for node_id in memory_violation_node_ids:
            reasons.append(
                f"节点{node_id}的内存需求"
                f"{node_memory_demand_mb[node_id]:.3f}MB"
                "超过容量"
                f"{node_map[node_id].memory_capacity_mb:.3f}MB。"
            )

        exact_sfc_reliability: float | None = None
        reliability_target_met = False

        if replica_plan_valid:
            reliability_result = (
                self.reliability_model.evaluate_sfc(
                    sfc=self.sfc,
                    function_replica_node_ids=(
                        candidate_map
                    ),
                )
            )

            exact_sfc_reliability = (
                reliability_result
                .exact_shared_failure_availability
            )

            # target_met 同时检查可靠性数值和最小故障域数量。
            reliability_target_met = (
                reliability_result.target_met
            )

            if not reliability_target_met:
                reasons.append(
                    "SFC可靠性"
                    f"{exact_sfc_reliability:.6f}"
                    "未达到目标"
                    f"{self.sfc.reliability_target:.6f}，"
                    "或故障域隔离要求未满足。"
                )

        resource_constraints_met = not (
            cpu_violation_node_ids
            or memory_violation_node_ids
        )

        all_constraints_met = (
            resource_constraints_met
            and replica_plan_valid
            and reliability_target_met
        )

        return SlotConstraintAudit(
            node_cpu_demand=dict(
                sorted(node_cpu_demand.items())
            ),
            node_memory_demand_mb=dict(
                sorted(
                    node_memory_demand_mb.items()
                )
            ),
            cpu_violation_node_ids=(
                cpu_violation_node_ids
            ),
            memory_violation_node_ids=(
                memory_violation_node_ids
            ),
            invalid_replica_node_ids=tuple(
                sorted(invalid_replica_node_ids)
            ),
            missing_function_ids=(
                missing_function_ids
            ),
            exact_sfc_reliability=(
                exact_sfc_reliability
            ),
            reliability_target=(
                self.sfc.reliability_target
            ),
            reliability_target_met=(
                reliability_target_met
            ),
            resource_constraints_met=(
                resource_constraints_met
            ),
            replica_plan_valid=replica_plan_valid,
            all_constraints_met=all_constraints_met,
            violation_reasons=tuple(reasons),
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
        previous_slow_mode: StandbyMode | None = None

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

            slow_mode_changed = (
                previous_slow_mode is not None
                and slow_decision.standby_mode
                is not previous_slow_mode
            )

            candidate_map = (
                self._build_candidate_map(
                    train_state=train_state,
                    slow_decision=slow_decision,
                )
            )

            plan_change_count = (
                self._count_plan_changes(
                    previous_map=(
                        previous_candidate_map
                    ),
                    current_map=candidate_map,
                )
            )

            infrastructure_state = (
                self.failure_process.state_for_slot(
                    train_state.time_slot
                )
            )

            operational_node_ids = frozenset(
                site.node.node_id
                for site in self.topology.sites
                if infrastructure_state
                .is_node_operational(
                    node_id=site.node.node_id,
                    topology=self.topology,
                )
            )

            fast_state = FastTimescaleState(
                time_slot=train_state.time_slot,
                serving_mec=train_state.serving_mec,
                remaining_dwell_time_s=(
                    train_state
                    .remaining_dwell_time_s
                ),
                request_count=request_count,
                function_ids=tuple(
                    self.sfc.function_ids
                ),
                candidate_node_ids=candidate_map,
                operational_node_ids=(
                    operational_node_ids
                ),
            )

            fast_decision = (
                self.controller.get_fast_decision(
                    state=fast_state,
                    slow_decision=slow_decision,
                )
            )

            transmission_delay_ms: (
                float | None
            ) = None

            execution_delay_ms: (
                float | None
            ) = None

            cold_start_delay_ms = 0.0
            failover_delay_ms = 0.0

            end_to_end_delay_ms: (
                float | None
            ) = None

            deadline_met: bool | None = None

            cold_activated_pairs: set[
                tuple[int, int]
            ] = set()

            if fast_decision.request_success is True:
                cold_start_ids = set(
                    fast_decision
                    .cold_start_function_ids
                )

                sfc_result = execute_sfc_batch(
                    functions=self.functions,
                    sfc=self.sfc,
                    placement_node_ids=list(
                        fast_decision
                        .selected_execution_node_ids
                    ),
                    source_node_id=(
                        train_state.serving_mec
                    ),
                    input_size_mb_per_request=(
                        self.input_size_mb_per_request
                    ),
                    request_count=request_count,
                    network=self.network,
                    cold_start_function_ids=(
                        cold_start_ids
                    ),
                    return_result_to_source=(
                        self.return_result_to_source
                    ),
                )

                transmission_delay_ms = (
                    sfc_result
                    .total_transmission_delay_ms
                )

                execution_delay_ms = (
                    sfc_result
                    .total_execution_delay_ms
                )

                cold_start_delay_ms = (
                    sfc_result
                    .total_cold_start_delay_ms
                )

                failover_delay_ms = (
                    len(
                        fast_decision
                        .failover_function_ids
                    )
                    * self
                    .failover_delay_ms_per_function
                )

                end_to_end_delay_ms = (
                    sfc_result
                    .total_end_to_end_delay_ms
                    + failover_delay_ms
                )

                deadline_met = (
                    end_to_end_delay_ms
                    <= self.sfc.deadline_ms
                )

                selected_nodes = (
                    fast_decision
                    .selected_execution_node_ids
                )

                for index, function_id in enumerate(
                    self.sfc.function_ids
                ):
                    if function_id in cold_start_ids:
                        cold_activated_pairs.add(
                            (
                                function_id,
                                selected_nodes[index],
                            )
                        )

            (
                active_instance_count,
                active_memory_mb,
            ) = self._calculate_active_memory(
                function_hot_node_ids=(
                    fast_decision
                    .function_hot_node_ids
                ),
                cold_activated_pairs=(
                    cold_activated_pairs
                ),
            )

            constraint_audit = (
                self._audit_slot_constraints(
                    request_count=request_count,
                    candidate_map=candidate_map,
                    selected_execution_node_ids=(
                        fast_decision
                        .selected_execution_node_ids
                    ),
                    request_success=(
                        fast_decision.request_success
                    ),
                    function_hot_node_ids=(
                        fast_decision
                        .function_hot_node_ids
                    ),
                    cold_activated_pairs=(
                        cold_activated_pairs
                    ),
                )
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
                        slow_decision
                        .standby_mode
                        .value
                    ),
                    use_redundancy=(
                        slow_decision.use_redundancy
                    ),
                    replica_count=(
                        slow_decision.replica_count
                    ),
                    slow_reason=(
                        slow_decision.reason
                    ),
                    replica_plan_changed_function_count=(
                        plan_change_count
                    ),
                    function_replica_node_ids=(
                        candidate_map
                    ),
                    function_hot_node_ids=(
                        fast_decision
                        .function_hot_node_ids
                    ),
                    selected_execution_node_ids=(
                        fast_decision
                        .selected_execution_node_ids
                    ),
                    backup_activation_triggered=(
                        fast_decision
                        .backup_activation_triggered
                    ),
                    failover_function_ids=(
                        fast_decision
                        .failover_function_ids
                    ),
                    cold_start_function_ids=(
                        fast_decision
                        .cold_start_function_ids
                    ),
                    unavailable_function_ids=(
                        fast_decision
                        .unavailable_function_ids
                    ),
                    request_success=(
                        fast_decision.request_success
                    ),
                    transmission_delay_ms=(
                        transmission_delay_ms
                    ),
                    execution_delay_ms=(
                        execution_delay_ms
                    ),
                    cold_start_delay_ms=(
                        cold_start_delay_ms
                    ),
                    failover_delay_ms=(
                        failover_delay_ms
                    ),
                    end_to_end_delay_ms=(
                        end_to_end_delay_ms
                    ),
                    deadline_met=deadline_met,
                    active_instance_count=(
                        active_instance_count
                    ),
                    active_memory_mb=(
                        active_memory_mb
                    ),
                    constraint_audit=(
                        constraint_audit
                    ),
                )
            )

            previous_slow_decision_slot = (
                slow_decision.decision_slot
            )

            previous_slow_mode = (
                slow_decision.standby_mode
            )

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

        total_cold_start_cost = (
            total_cold_start_delay_ms
            * self.cost_weights
            .cold_start_cost_per_ms
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
            total_request_delay_cost
            + total_memory_cost
            + total_cold_start_cost
            + total_sla_penalty
            + total_slow_control_cost
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
