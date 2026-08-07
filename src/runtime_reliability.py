
"""
runtime_reliability.py

本文件实现运行态高可靠 Serverless SFC 仿真器。

当前假设采用热备模式：

1. 主实例和备用实例都已经处于温状态；
2. 主实例正常时优先使用主实例；
3. 主实例失效时尝试使用备用实例；
4. 备用实例接管会产生故障切换时延；
5. 所有副本都持续占用温容器内存；
6. 当前阶段暂时不考虑副本迁移和同步流量。

本模块重点研究：

1. 服务成功率；
2. 主备故障切换；
3. SLA 违反；
4. 热备副本内存开销。
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
from src.topology import LinearRailTopology
from src.workload import DeterministicWorkload


class ReplicaPlannerProtocol(Protocol):
    """
    运行态副本规划器接口。
    """

    def plan(
        self,
        sfc: SFCType,
        train_state: TrainState,
        topology: LinearRailTopology,
    ) -> ReplicaPlacementPlan:
        """
        返回当前列车状态下的副本部署计划。
        """

        ...


class SingleReplicaPlanner:
    """
    单副本基线规划器。

    所有函数只部署在当前接入 MEC。
    """

    def plan(
        self,
        sfc: SFCType,
        train_state: TrainState,
        topology: LinearRailTopology,
    ) -> ReplicaPlacementPlan:
        """
        为每个函数生成一个主实例。
        """

        topology.get_site(
            train_state.serving_mec
        )

        replica_nodes = (
            train_state.serving_mec,
        )

        return ReplicaPlacementPlan(
            policy_id="single_replica",
            display_name="当前接入MEC单副本",
            primary_node_id=(
                train_state.serving_mec
            ),
            shared_replica_node_ids=(
                replica_nodes
            ),
            function_replica_node_ids={
                function_id: replica_nodes
                for function_id
                in sfc.function_ids
            },
        )


@dataclass(frozen=True)
class RuntimeReliabilitySlotRecord:
    """
    一个时隙的运行态可靠性记录。
    """

    time_slot: int
    position_m: float
    serving_mec: int
    next_mec: int
    handover_occurred: bool

    request_count: int

    down_domain_ids: tuple[int, ...]
    down_node_ids: tuple[int, ...]

    # 每个函数的候选副本节点。
    function_replica_node_ids: dict[
        int,
        tuple[int, ...],
    ]

    # 实际选择的函数执行节点。
    #
    # 请求失败或没有请求时为空元组。
    selected_execution_node_ids: tuple[int, ...]

    # 是否发生了至少一个函数的备用接管。
    batch_failover: bool

    # 当前批次中有多少个函数切换到了备用实例。
    failover_function_count: int

    # 故障切换附加时延。
    failover_delay_ms: float

    # 没有可用副本的函数编号。
    unavailable_function_ids: tuple[int, ...]

    # 没有请求时为 None。
    request_success: bool | None

    transmission_delay_ms: float | None
    execution_delay_ms: float | None
    end_to_end_delay_ms: float | None

    # 请求执行失败时为 None，
    # 因为失败请求没有完成时延。
    deadline_met: bool | None

    # 当前部署计划中的温副本数量和内存。
    hot_replica_count: int
    hot_memory_mb: float


@dataclass(frozen=True)
class RuntimeReliabilitySummary:
    """
    运行态高可靠仿真汇总指标。
    """

    total_slots: int
    total_requests: int
    request_batches: int

    successful_requests: int
    failed_requests: int

    successful_batches: int
    failed_batches: int

    # 按请求数量计算的服务成功率。
    request_success_rate: float

    # 按请求批次计算的服务成功率。
    batch_success_rate: float

    failover_batches: int
    failover_function_stages: int

    # 成功执行但超过时延约束的批次数量。
    deadline_violations: int

    # 服务失败或成功但超时均属于 SLA 违反。
    sla_violations: int
    sla_violation_rate: float

    average_successful_batch_delay_ms: float
    p95_successful_batch_delay_ms: float

    average_hot_memory_mb: float
    peak_hot_memory_mb: float
    total_hot_memory_mb_seconds: float


@dataclass(frozen=True)
class RuntimeReliabilityResult:
    """
    一次运行态高可靠仿真的完整结果。
    """

    summary: RuntimeReliabilitySummary
    records: tuple[
        RuntimeReliabilitySlotRecord,
        ...,
    ]


class HotStandbyRuntimeSimulator:
    """
    热备主备副本运行态仿真器。
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
        failure_process: FailureProcess,
        input_size_mb_per_request: float,
        slot_seconds: float,
        failover_delay_ms_per_function: float,
        return_result_to_source: bool = True,
    ) -> None:
        """
        创建热备运行态仿真器。
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

            function_map[function.function_id] = (
                function
            )

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

    def _calculate_hot_replica_memory(
        self,
        function_replica_node_ids: dict[
            int,
            tuple[int, ...],
        ],
    ) -> tuple[int, float]:
        """
        计算当前主备计划持续占用的温容器内存。
        """

        hot_replica_count = 0
        hot_memory_mb = 0.0

        for function_id in self.sfc.function_ids:
            replica_node_ids = (
                function_replica_node_ids[
                    function_id
                ]
            )

            function = self.function_map[
                function_id
            ]

            hot_replica_count += len(
                replica_node_ids
            )

            hot_memory_mb += (
                function.memory_mb
                * len(replica_node_ids)
            )

        return (
            hot_replica_count,
            hot_memory_mb,
        )

    def run(self) -> RuntimeReliabilityResult:
        """
        从线路起点运行到终点。
        """

        self.failure_process.reset()

        train_state = self.mobility_model.reset()

        records: list[
            RuntimeReliabilitySlotRecord
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

            placement_plan = (
                self.replica_planner.plan(
                    sfc=self.sfc,
                    train_state=train_state,
                    topology=self.topology,
                )
            )

            function_replica_node_ids = {
                function_id: tuple(node_ids)
                for function_id, node_ids
                in placement_plan
                .function_replica_node_ids
                .items()
            }

            if set(function_replica_node_ids) != set(
                self.sfc.function_ids
            ):
                raise ValueError(
                    "副本计划与 SFC 函数集合不一致。"
                )

            (
                hot_replica_count,
                hot_memory_mb,
            ) = self._calculate_hot_replica_memory(
                function_replica_node_ids
            )

            selected_execution_node_ids: list[
                int
            ] = []

            unavailable_function_ids: list[int] = []

            failover_function_count = 0

            request_success: bool | None = None
            transmission_delay_ms: float | None = None
            execution_delay_ms: float | None = None
            end_to_end_delay_ms: float | None = None
            deadline_met: bool | None = None

            # 只有存在真实请求时才进行副本选择。
            if request_count > 0:
                for function_id in self.sfc.function_ids:
                    candidate_node_ids = (
                        function_replica_node_ids[
                            function_id
                        ]
                    )

                    if len(candidate_node_ids) == 0:
                        raise ValueError(
                            f"函数 {function_id} 没有副本。"
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

                    if len(operational_candidates) == 0:
                        unavailable_function_ids.append(
                            function_id
                        )
                        continue

                    selected_node_id = (
                        operational_candidates[0]
                    )

                    selected_execution_node_ids.append(
                        selected_node_id
                    )

                    # 每个函数副本列表中的第一个节点
                    # 被视为主实例。
                    primary_node_id = (
                        candidate_node_ids[0]
                    )

                    if selected_node_id != primary_node_id:
                        failover_function_count += 1

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
                        cold_start_function_ids=set(),
                        return_result_to_source=(
                            self.return_result_to_source
                        ),
                    )

                    failover_delay_ms = (
                        failover_function_count
                        * self
                        .failover_delay_ms_per_function
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

            failover_delay_ms = (
                failover_function_count
                * self.failover_delay_ms_per_function
            )

            records.append(
                RuntimeReliabilitySlotRecord(
                    time_slot=train_state.time_slot,
                    position_m=train_state.position_m,
                    serving_mec=(
                        train_state.serving_mec
                    ),
                    next_mec=train_state.next_mec,
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
                    selected_execution_node_ids=tuple(
                        selected_execution_node_ids
                    ),
                    batch_failover=(
                        failover_function_count > 0
                    ),
                    failover_function_count=(
                        failover_function_count
                    ),
                    failover_delay_ms=(
                        failover_delay_ms
                    ),
                    unavailable_function_ids=tuple(
                        unavailable_function_ids
                    ),
                    request_success=request_success,
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
                    hot_replica_count=(
                        hot_replica_count
                    ),
                    hot_memory_mb=hot_memory_mb,
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

        return RuntimeReliabilityResult(
            summary=summary,
            records=tuple(records),
        )

    def _build_summary(
        self,
        records: list[
            RuntimeReliabilitySlotRecord
        ],
    ) -> RuntimeReliabilitySummary:
        """
        汇总运行态可靠性指标。
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
            int(record.batch_failover)
            for record in successful_records
        )

        failover_function_stages = sum(
            record.failover_function_count
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
            if record.end_to_end_delay_ms
            is not None
        ]

        if successful_delays:
            average_successful_batch_delay_ms = (
                float(
                    np.mean(successful_delays)
                )
            )

            p95_successful_batch_delay_ms = (
                float(
                    np.percentile(
                        successful_delays,
                        95,
                    )
                )
            )
        else:
            average_successful_batch_delay_ms = 0.0
            p95_successful_batch_delay_ms = 0.0

        if records:
            average_hot_memory_mb = float(
                np.mean(
                    [
                        record.hot_memory_mb
                        for record in records
                    ]
                )
            )

            peak_hot_memory_mb = max(
                record.hot_memory_mb
                for record in records
            )
        else:
            average_hot_memory_mb = 0.0
            peak_hot_memory_mb = 0.0

        total_hot_memory_mb_seconds = sum(
            record.hot_memory_mb
            * self.slot_seconds
            for record in records
        )

        return RuntimeReliabilitySummary(
            total_slots=len(records),
            total_requests=total_requests,
            request_batches=request_batches,
            successful_requests=(
                successful_requests
            ),
            failed_requests=failed_requests,
            successful_batches=(
                successful_batches
            ),
            failed_batches=failed_batches,
            request_success_rate=(
                request_success_rate
            ),
            batch_success_rate=batch_success_rate,
            failover_batches=failover_batches,
            failover_function_stages=(
                failover_function_stages
            ),
            deadline_violations=(
                deadline_violations
            ),
            sla_violations=sla_violations,
            sla_violation_rate=(
                sla_violation_rate
            ),
            average_successful_batch_delay_ms=(
                average_successful_batch_delay_ms
            ),
            p95_successful_batch_delay_ms=(
                p95_successful_batch_delay_ms
            ),
            average_hot_memory_mb=(
                average_hot_memory_mb
            ),
            peak_hot_memory_mb=(
                peak_hot_memory_mb
            ),
            total_hot_memory_mb_seconds=(
                total_hot_memory_mb_seconds
            ),
        )
