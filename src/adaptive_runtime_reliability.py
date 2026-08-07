"""
adaptive_runtime_reliability.py

本文件比较不同主备温状态模式下的运行行为。

支持：

1. 单副本；
2. 跨故障域冷备用；
3. 跨故障域全热备；
4. 轨迹感知动态主备。

当主实例失效时：

- 温备用接管：
  只产生故障检测与切换时延。

- 冷备用接管：
  产生备用实例冷启动时延，
  同时产生故障检测与切换时延。

当前冷备用是简化模型：

    冷备用在本次接管时启动；
    下一个时隙重新根据策略决定温状态。

后续将加入跨时隙的副本状态迁移。
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.entities import (
    ServerlessFunction,
    SFCType,
    TrainState,
)
from src.failure_process import FailureProcess
from src.mobility import TrainMobilityModel
from src.network import TransferNetworkProtocol
from src.redundancy_placement import (
    ReplicaPlacementPlan,
)
from src.sfc_execution import execute_sfc_batch
from src.standby_policy import (
    StandbyActivationPolicy,
)
from src.topology import LinearRailTopology
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
        """
        返回当前列车状态下的候选副本计划。
        """

        ...


@dataclass(frozen=True)
class AdaptiveReliabilitySlotRecord:
    """
    一个时隙的动态主备运行记录。
    """

    time_slot: int
    position_m: float
    serving_mec: int
    next_mec: int
    remaining_dwell_time_s: float
    handover_occurred: bool

    request_count: int

    down_domain_ids: tuple[int, ...]
    down_node_ids: tuple[int, ...]

    # 每个函数的全部候选副本。
    function_replica_node_ids: dict[
        int,
        tuple[int, ...],
    ]

    # 当前时隙处于温状态的副本。
    function_hot_node_ids: dict[
        int,
        tuple[int, ...],
    ]

    # 实际执行每个函数的节点。
    selected_execution_node_ids: tuple[int, ...]

    unavailable_function_ids: tuple[int, ...]

    request_success: bool | None

    # 备用接管统计。
    failover_function_count: int
    hot_failover_function_count: int
    cold_failover_function_count: int

    failover_delay_ms: float
    cold_backup_startup_delay_ms: float

    transmission_delay_ms: float | None
    execution_delay_ms: float | None
    end_to_end_delay_ms: float | None
    deadline_met: bool | None

    # 本时隙实际处于活动或温状态的实例内存。
    active_replica_count: int
    active_memory_mb: float


@dataclass(frozen=True)
class AdaptiveReliabilitySummary:
    """
    动态主备仿真汇总指标。
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

    hot_failover_function_stages: int
    cold_failover_function_stages: int

    total_failover_delay_ms: float
    total_cold_backup_startup_delay_ms: float

    deadline_violations: int
    sla_violations: int
    sla_violation_rate: float

    average_successful_batch_delay_ms: float
    p95_successful_batch_delay_ms: float

    average_active_memory_mb: float
    peak_active_memory_mb: float
    total_active_memory_mb_seconds: float


@dataclass(frozen=True)
class AdaptiveReliabilityResult:
    """
    一次动态主备仿真的完整结果。
    """

    summary: AdaptiveReliabilitySummary

    records: tuple[
        AdaptiveReliabilitySlotRecord,
        ...,
    ]


class AdaptiveStandbyRuntimeSimulator:
    """
    支持热备、冷备和动态主备的运行态仿真器。
    """

    def __init__(
        self,
        topology: LinearRailTopology,
        network: TransferNetworkProtocol,
        mobility_model: TrainMobilityModel,
        workload: DeterministicWorkload,
        functions: list[ServerlessFunction],
        sfc: SFCType,
        replica_planner: ReplicaPlannerProtocol,
        standby_policy: StandbyActivationPolicy,
        failure_process: FailureProcess,
        input_size_mb_per_request: float,
        slot_seconds: float,
        failover_delay_ms_per_function: float,
        return_result_to_source: bool = True,
    ) -> None:
        """
        创建动态主备仿真器。
        """

        if len(functions) == 0:
            raise ValueError(
                "至少需要配置一个函数。"
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
                    f"函数编号 {function.function_id} 重复。"
                )

            function_map[function.function_id] = function

        for function_id in sfc.function_ids:
            if function_id not in function_map:
                raise KeyError(
                    f"缺少 function_id={function_id}。"
                )

        self.topology = topology
        self.network = network
        self.mobility_model = mobility_model
        self.workload = workload
        self.functions = functions
        self.function_map = function_map
        self.sfc = sfc

        self.replica_planner = replica_planner
        self.standby_policy = standby_policy
        self.failure_process = failure_process

        self.input_size_mb_per_request = (
            input_size_mb_per_request
        )

        self.slot_seconds = slot_seconds

        self.failover_delay_ms_per_function = (
            failover_delay_ms_per_function
        )

        self.return_result_to_source = (
            return_result_to_source
        )

    def _build_replica_maps(
        self,
        train_state: TrainState,
    ) -> tuple[
        dict[int, tuple[int, ...]],
        dict[int, tuple[int, ...]],
    ]:
        """
        生成候选副本和温副本映射。
        """

        placement_plan = self.replica_planner.plan(
            sfc=self.sfc,
            train_state=train_state,
            topology=self.topology,
        )

        candidate_map = {
            function_id: tuple(node_ids)
            for function_id, node_ids
            in placement_plan
            .function_replica_node_ids
            .items()
        }

        if set(candidate_map) != set(
            self.sfc.function_ids
        ):
            raise ValueError(
                "副本计划与 SFC 函数集合不一致。"
            )

        hot_map: dict[int, tuple[int, ...]] = {}

        for function_id in self.sfc.function_ids:
            candidate_node_ids = candidate_map[
                function_id
            ]

            if len(candidate_node_ids) == 0:
                raise ValueError(
                    f"函数 {function_id} 没有候选副本。"
                )

            hot_node_ids = (
                self.standby_policy
                .hot_replica_node_ids(
                    function_id=function_id,
                    candidate_node_ids=(
                        candidate_node_ids
                    ),
                    train_state=train_state,
                    topology=self.topology,
                )
            )

            if len(hot_node_ids) == 0:
                raise ValueError(
                    "主实例必须保持温状态。"
                )

            if not set(hot_node_ids).issubset(
                candidate_node_ids
            ):
                raise ValueError(
                    "温副本必须属于候选副本集合。"
                )

            # 第一个候选节点是主实例，
            # 主实例必须始终处于温状态。
            if candidate_node_ids[0] not in hot_node_ids:
                raise ValueError(
                    "主实例必须处于温状态。"
                )

            hot_map[function_id] = tuple(
                hot_node_ids
            )

        return candidate_map, hot_map

    def _calculate_active_memory(
        self,
        function_hot_node_ids: dict[
            int,
            tuple[int, ...],
        ],
        additionally_activated_pairs: set[
            tuple[int, int]
        ],
    ) -> tuple[int, float]:
        """
        计算本时隙实际活动实例数量和内存。

        additionally_activated_pairs 用于记录：
        因故障接管而临时启动的冷备用实例。
        """

        active_pairs: set[
            tuple[int, int]
        ] = set()

        for function_id, hot_node_ids in (
            function_hot_node_ids.items()
        ):
            for node_id in hot_node_ids:
                active_pairs.add(
                    (function_id, node_id)
                )

        active_pairs.update(
            additionally_activated_pairs
        )

        active_memory_mb = 0.0

        for function_id, _ in active_pairs:
            active_memory_mb += (
                self.function_map[
                    function_id
                ].memory_mb
            )

        return (
            len(active_pairs),
            active_memory_mb,
        )

    def run(self) -> AdaptiveReliabilityResult:
        """
        从铁路线路起点运行到终点。
        """

        self.failure_process.reset()

        train_state = self.mobility_model.reset()

        records: list[
            AdaptiveReliabilitySlotRecord
        ] = []

        previous_serving_mec: int | None = None

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

            infrastructure_state = (
                self.failure_process.state_for_slot(
                    train_state.time_slot
                )
            )

            (
                function_replica_node_ids,
                function_hot_node_ids,
            ) = self._build_replica_maps(
                train_state=train_state
            )

            selected_execution_node_ids: list[
                int
            ] = []

            unavailable_function_ids: list[int] = []

            failover_function_count = 0
            hot_failover_function_count = 0
            cold_failover_function_count = 0

            cold_start_function_ids: set[int] = set()

            # 本时隙由于冷备接管临时激活的实例。
            cold_activated_pairs: set[
                tuple[int, int]
            ] = set()

            request_success: bool | None = None

            failover_delay_ms = 0.0
            cold_backup_startup_delay_ms = 0.0

            transmission_delay_ms: float | None = None
            execution_delay_ms: float | None = None
            end_to_end_delay_ms: float | None = None
            deadline_met: bool | None = None

            if request_count > 0:
                for function_id in self.sfc.function_ids:
                    candidate_node_ids = (
                        function_replica_node_ids[
                            function_id
                        ]
                    )

                    hot_node_ids = set(
                        function_hot_node_ids[
                            function_id
                        ]
                    )

                    operational_candidates = [
                        node_id
                        for node_id
                        in candidate_node_ids
                        if infrastructure_state
                        .is_node_operational(
                            node_id=node_id,
                            topology=self.topology,
                        )
                    ]

                    if not operational_candidates:
                        unavailable_function_ids.append(
                            function_id
                        )
                        continue

                    # 候选副本列表按主、备顺序排列，
                    # 因此选择第一个正常节点。
                    selected_node_id = (
                        operational_candidates[0]
                    )

                    selected_execution_node_ids.append(
                        selected_node_id
                    )

                    primary_node_id = (
                        candidate_node_ids[0]
                    )

                    if selected_node_id != primary_node_id:
                        failover_function_count += 1

                        if selected_node_id in hot_node_ids:
                            hot_failover_function_count += 1
                        else:
                            cold_failover_function_count += 1

                            cold_start_function_ids.add(
                                function_id
                            )

                            cold_activated_pairs.add(
                                (
                                    function_id,
                                    selected_node_id,
                                )
                            )

                if unavailable_function_ids:
                    request_success = False
                    selected_execution_node_ids = []
                else:
                    request_success = True

                    sfc_result = execute_sfc_batch(
                        functions=self.functions,
                        sfc=self.sfc,
                        placement_node_ids=(
                            selected_execution_node_ids
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
                            cold_start_function_ids
                        ),
                        return_result_to_source=(
                            self.return_result_to_source
                        ),
                    )

                    failover_delay_ms = (
                        failover_function_count
                        * self
                        .failover_delay_ms_per_function
                    )

                    cold_backup_startup_delay_ms = (
                        sfc_result
                        .total_cold_start_delay_ms
                    )

                    transmission_delay_ms = (
                        sfc_result
                        .total_transmission_delay_ms
                    )

                    execution_delay_ms = (
                        sfc_result
                        .total_execution_delay_ms
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

            (
                active_replica_count,
                active_memory_mb,
            ) = self._calculate_active_memory(
                function_hot_node_ids=(
                    function_hot_node_ids
                ),
                additionally_activated_pairs=(
                    cold_activated_pairs
                ),
            )

            records.append(
                AdaptiveReliabilitySlotRecord(
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
                    down_domain_ids=(
                        infrastructure_state
                        .down_domain_ids
                    ),
                    down_node_ids=(
                        infrastructure_state
                        .down_node_ids
                    ),
                    function_replica_node_ids=(
                        function_replica_node_ids
                    ),
                    function_hot_node_ids=(
                        function_hot_node_ids
                    ),
                    selected_execution_node_ids=tuple(
                        selected_execution_node_ids
                    ),
                    unavailable_function_ids=tuple(
                        unavailable_function_ids
                    ),
                    request_success=request_success,
                    failover_function_count=(
                        failover_function_count
                    ),
                    hot_failover_function_count=(
                        hot_failover_function_count
                    ),
                    cold_failover_function_count=(
                        cold_failover_function_count
                    ),
                    failover_delay_ms=(
                        failover_delay_ms
                    ),
                    cold_backup_startup_delay_ms=(
                        cold_backup_startup_delay_ms
                    ),
                    transmission_delay_ms=(
                        transmission_delay_ms
                    ),
                    execution_delay_ms=(
                        execution_delay_ms
                    ),
                    end_to_end_delay_ms=(
                        end_to_end_delay_ms
                    ),
                    deadline_met=deadline_met,
                    active_replica_count=(
                        active_replica_count
                    ),
                    active_memory_mb=(
                        active_memory_mb
                    ),
                )
            )

            if self.mobility_model.finished:
                break

            previous_serving_mec = (
                train_state.serving_mec
            )

            train_state = (
                self.mobility_model.step()
            )

        summary = self._build_summary(records)

        return AdaptiveReliabilityResult(
            summary=summary,
            records=tuple(records),
        )

    def _build_summary(
        self,
        records: list[
            AdaptiveReliabilitySlotRecord
        ],
    ) -> AdaptiveReliabilitySummary:
        """
        计算动态主备仿真汇总指标。
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
        successful_batches = len(successful_records)
        failed_batches = len(failed_records)

        if total_requests > 0:
            request_success_rate = (
                successful_requests
                / total_requests
            )
        else:
            request_success_rate = 0.0

        if request_batches > 0:
            batch_success_rate = (
                successful_batches
                / request_batches
            )
        else:
            batch_success_rate = 0.0

        failover_batches = sum(
            int(record.failover_function_count > 0)
            for record in active_records
        )

        failover_function_stages = sum(
            record.failover_function_count
            for record in active_records
        )

        hot_failover_function_stages = sum(
            record.hot_failover_function_count
            for record in active_records
        )

        cold_failover_function_stages = sum(
            record.cold_failover_function_count
            for record in active_records
        )

        total_failover_delay_ms = sum(
            record.failover_delay_ms
            for record in active_records
        )

        total_cold_backup_startup_delay_ms = sum(
            record.cold_backup_startup_delay_ms
            for record in active_records
        )

        deadline_violations = sum(
            int(record.deadline_met is False)
            for record in successful_records
        )

        sla_violations = (
            failed_batches
            + deadline_violations
        )

        if request_batches > 0:
            sla_violation_rate = (
                sla_violations
                / request_batches
            )
        else:
            sla_violation_rate = 0.0

        successful_delays = [
            record.end_to_end_delay_ms
            for record in successful_records
            if record.end_to_end_delay_ms is not None
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
            average_active_memory_mb = float(
                np.mean(
                    [
                        record.active_memory_mb
                        for record in records
                    ]
                )
            )

            peak_active_memory_mb = max(
                record.active_memory_mb
                for record in records
            )
        else:
            average_active_memory_mb = 0.0
            peak_active_memory_mb = 0.0

        total_active_memory_mb_seconds = sum(
            record.active_memory_mb
            * self.slot_seconds
            for record in records
        )

        return AdaptiveReliabilitySummary(
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
            hot_failover_function_stages=(
                hot_failover_function_stages
            ),
            cold_failover_function_stages=(
                cold_failover_function_stages
            ),
            total_failover_delay_ms=(
                total_failover_delay_ms
            ),
            total_cold_backup_startup_delay_ms=(
                total_cold_backup_startup_delay_ms
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
                average_active_memory_mb
            ),
            peak_active_memory_mb=(
                peak_active_memory_mb
            ),
            total_active_memory_mb_seconds=(
                total_active_memory_mb_seconds
            ),
        )
