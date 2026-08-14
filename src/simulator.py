"""
simulator.py

本文件实现轨道边缘 Serverless SFC 离散时隙仿真器。

当前仿真器连接：

1. 列车移动模型；
2. 请求负载模型；
3. SFC 函数部署策略；
4. 轨迹感知提前预热策略；
5. Serverless 容器生命周期；
6. MEC 网络传输模型；
7. SFC 批量执行模型。

本版本支持比较：

    被动响应式部署
    与
    轨迹感知提前预热部署
"""

from dataclasses import dataclass

import numpy as np

from src.cold_start import ContainerLifecycleManager
from src.entities import (
    FunctionInstance,
    InstanceStatus,
    ServerlessFunction,
    SFCType,
    TrainState,
)
from src.mobility import TrainMobilityModel
from src.network import TransferNetworkProtocol
from src.placement import PlacementPolicy
from src.prewarming import (
    NoPrewarmingPolicy,
    PrewarmingPolicy,
)
from src.sfc_execution import execute_sfc_batch
from src.topology import LinearRailTopology
from src.workload import DeterministicWorkload


@dataclass(frozen=True)
class SlotSimulationRecord:
    """
    一个快时隙的完整仿真记录。
    """

    # 当前快时隙编号
    time_slot: int

    # 列车位置，单位为米
    position_m: float

    # 当前接入 MEC
    serving_mec: int

    # 下一个 MEC
    next_mec: int

    # 当前时隙是否发生 MEC 切换
    handover_occurred: bool

    # 当前时隙到达的请求数量
    request_count: int

    # 当前 SFC 的函数部署节点
    placement_node_ids: tuple[int, ...]

    # 当前时隙在哪些 MEC 上执行了提前预热
    prewarm_target_node_ids: tuple[int, ...]

    # 当前时隙新创建的预热容器数量
    prewarm_start_count: int

    # 当前时隙预热产生的后台启动开销
    #
    # 单位为毫秒。
    # 该数值不是用户请求时延，
    # 而是后台资源启动开销。
    prewarm_startup_overhead_ms: float

    # 当前时隙有多少个提前预热的函数实例
    # 后来被真实请求成功使用
    prewarm_hit_count: int

    # 当前时隙有多少个提前预热的函数实例
    # 在没有被请求使用的情况下被销毁
    prewarm_waste_count: int

    # 当前真实请求触发的冷启动函数数量
    cold_start_count: int

    # 当前请求命中的温实例数量
    warm_hit_count: int

    # 当前时隙被销毁的旧容器数量
    expiration_count: int

    # SFC 总传输时延
    transmission_delay_ms: float

    # 用户请求承担的总冷启动时延
    cold_start_delay_ms: float

    # SFC 函数执行时延
    execution_delay_ms: float

    # SFC 端到端时延
    end_to_end_delay_ms: float

    # 是否满足时延约束
    #
    # 没有请求时为 None。
    deadline_met: bool | None

    # 当前时隙结束时仍处于 WARM 状态的实例数量
    warm_instance_count: int

    # 当前时隙结束时所有温实例占用的总内存
    # 单位为 MB
    warm_memory_mb: float


@dataclass(frozen=True)
class SimulationSummary:
    """
    一次完整仿真的汇总指标。
    """

    # 总仿真时隙数量
    total_slots: int

    # 总请求数量
    total_requests: int

    # 有请求到达的时隙数量
    request_batches: int

    # MEC 切换次数
    handover_count: int

    # 用户请求触发的函数冷启动总次数
    user_cold_starts: int

    # 主动预热创建的容器总次数
    prewarm_starts: int

    # 主动预热产生的后台启动开销
    total_prewarm_startup_overhead_ms: float

    # 函数温实例命中总次数
    warm_hits: int

    # 容器销毁总次数
    expiration_count: int

    # 违反 SFC 时延约束的请求批次数量
    deadline_violations: int

    # 函数阶段冷启动率
    function_stage_cold_start_rate: float

    # 请求批次时延违反率
    deadline_violation_rate: float

    # 平均请求批次端到端时延
    average_batch_delay_ms: float

    # P95 请求批次端到端时延
    p95_batch_delay_ms: float


@dataclass(frozen=True)
class SimulationRunResult:
    """
    一次完整仿真的全部结果。
    """

    summary: SimulationSummary
    records: tuple[SlotSimulationRecord, ...]


class RailServerlessSFCSimulator:
    """
    单列车、单 SFC 的离散时隙仿真器。

    当前假设：

    1. 只有一列列车；
    2. 每个时隙只有一种 SFC 请求；
    3. 一个函数在一个 MEC 上最多有一个实例；
    4. 请求批次在当前时隙内处理完成；
    5. 暂时不考虑 CPU 排队与带宽竞争；
    6. 预热操作在后台完成；
    7. 预热启动时间不计入用户端到端时延。
    """

    def __init__(
        self,
        topology: LinearRailTopology,
        network: TransferNetworkProtocol,
        mobility_model: TrainMobilityModel,
        workload: DeterministicWorkload,
        placement_policy: PlacementPolicy,
        functions: list[ServerlessFunction],
        sfc: SFCType,
        survival_slots: int,
        input_size_mb_per_request: float,
        return_result_to_source: bool = True,
        prewarming_policy: PrewarmingPolicy | None = None,
    ) -> None:
        """
        创建完整仿真器。

        Parameters
        ----------
        prewarming_policy:
            提前预热策略。

            None 表示不进行预热，
            等价于 NoPrewarmingPolicy。
        """

        if survival_slots <= 0:
            raise ValueError(
                "容器生存窗口必须大于 0。"
            )

        if input_size_mb_per_request < 0:
            raise ValueError(
                "单请求输入数据量不能小于 0。"
            )

        if len(functions) == 0:
            raise ValueError(
                "系统中至少需要一个函数。"
            )

        # 当前仿真器要求一条 SFC 内函数编号唯一。
        if len(sfc.function_ids) != len(
            set(sfc.function_ids)
        ):
            raise ValueError(
                "当前仿真器要求 SFC 中函数编号不能重复。"
            )

        function_map: dict[int, ServerlessFunction] = {}

        for function in functions:
            if function.function_id in function_map:
                raise ValueError(
                    f"函数编号 {function.function_id} 重复。"
                )

            function_map[function.function_id] = function

        # 检查 SFC 所需函数是否全部存在。
        for function_id in sfc.function_ids:
            if function_id not in function_map:
                raise KeyError(
                    f"系统中不存在 function_id={function_id}。"
                )

        self.topology = topology
        self.network = network
        self.mobility_model = mobility_model
        self.workload = workload
        self.placement_policy = placement_policy
        self.functions = functions
        self.function_map = function_map
        self.sfc = sfc

        self.input_size_mb_per_request = (
            input_size_mb_per_request
        )

        self.return_result_to_source = (
            return_result_to_source
        )

        # 没有传入预热策略时，
        # 默认使用“不提前预热”策略。
        if prewarming_policy is None:
            self.prewarming_policy = (
                NoPrewarmingPolicy()
            )
        else:
            self.prewarming_policy = (
                prewarming_policy
            )

        self.lifecycle_manager = (
            ContainerLifecycleManager(
                survival_slots=survival_slots
            )
        )

        # 字典键：
        #     (node_id, function_id)
        #
        # 字典值：
        #     对应的函数实例状态。
        self.instances: dict[
            tuple[int, int],
            FunctionInstance,
        ] = {}

                # 保存“已经主动预热，但还没有被真实请求使用”
        # 的函数实例。
        #
        # 字典键格式：
        #     (node_id, function_id)
        #
        # 后续有两种可能：
        #
        # 1. 被真实请求使用：
        #    记为一次预热命中；
        #
        # 2. 没有被请求使用就过期：
        #    记为一次预热浪费。
        self.pending_prewarmed_instances: set[
            tuple[int, int]
        ] = set()

    def _get_or_create_instance(
        self,
        node_id: int,
        function_id: int,
    ) -> FunctionInstance:
        """
        获取指定 MEC 上的函数实例。

        实例从未出现过时，
        创建一个 ABSENT 状态对象。
        """

        instance_key = (
            node_id,
            function_id,
        )

        if instance_key not in self.instances:
            self.instances[instance_key] = (
                FunctionInstance(
                    node_id=node_id,
                    function_id=function_id,
                    status=InstanceStatus.ABSENT,
                )
            )

        return self.instances[instance_key]

    def _process_prewarming(
        self,
        train_state: TrainState,
    ) -> tuple[
        set[tuple[int, int]],
        tuple[int, ...],
        int,
        float,
    ]:
        """
        执行当前时隙的提前预热操作。

        Returns
        -------
        prewarmed_instance_keys:
            当前时隙被预热或保活的函数实例键。

        target_node_ids:
            当前时隙预热目标 MEC。

        prewarm_start_count:
            当前时隙真正新启动的预热容器数量。

        prewarm_startup_overhead_ms:
            当前时隙预热产生的后台启动开销。
        """

        target_node_ids = (
            self.prewarming_policy.target_node_ids(
                train_state=train_state,
                sfc=self.sfc,
                topology=self.topology,
            )
        )

        # 去除重复目标，同时保持原有顺序。
        unique_target_node_ids = list(
            dict.fromkeys(target_node_ids)
        )

        prewarmed_instance_keys: set[
            tuple[int, int]
        ] = set()

        prewarm_start_count = 0
        prewarm_startup_overhead_ms = 0.0

        for target_node_id in unique_target_node_ids:
            # 检查目标 MEC 是否存在。
            self.topology.get_site(target_node_id)

            # 当前策略在目标 MEC 上预热整条 SFC。
            for function_id in self.sfc.function_ids:
                function = self.function_map[
                    function_id
                ]

                instance = self._get_or_create_instance(
                    node_id=target_node_id,
                    function_id=function_id,
                )

                lifecycle_result = (
                    self.lifecycle_manager.process_slot(
                        instance=instance,
                        function=function,
                        request_count=0,
                        keep_warm=True,
                    )
                )

                instance_key = (
                    target_node_id,
                    function_id,
                )

                prewarmed_instance_keys.add(
                    instance_key
                )

                # 只有实例原本不存在时，
                # 才会产生一次新的主动预热启动。
                if lifecycle_result.prewarm_occurred:
                    prewarm_start_count += 1

                    prewarm_startup_overhead_ms += (
                        lifecycle_result
                        .cold_start_delay_ms
                    )

                    # 这个实例刚刚由主动预热创建，
                    # 暂时还不知道以后是否会被真实请求使用。
                    self.pending_prewarmed_instances.add(
                        instance_key
                    )

        return (
            prewarmed_instance_keys,
            tuple(unique_target_node_ids),
            prewarm_start_count,
            prewarm_startup_overhead_ms,
        )

    def _process_active_request_batch(
        self,
        placement_node_ids: list[int],
        request_count: int,
    ) -> tuple[
        set[tuple[int, int]],
        set[int],
        int,
        int,
        int,
    ]:
        """
        更新当前真实请求使用的函数容器状态。

        Returns
        -------
        used_instance_keys:
            当前请求使用的实例键。

        cold_start_function_ids:
            当前发生冷启动的函数编号集合。

        cold_start_count:
            用户请求触发的冷启动函数数量。

        warm_hit_count:
            命中的温实例数量。

        prewarm_hit_count:
            命中的温实例中，有多少个来自提前预热。
        """

        if len(placement_node_ids) != len(
            self.sfc.function_ids
        ):
            raise ValueError(
                "部署节点数量与 SFC 函数数量不一致。"
            )

        used_instance_keys: set[
            tuple[int, int]
        ] = set()

        cold_start_function_ids: set[int] = set()

        cold_start_count = 0
        warm_hit_count = 0
        prewarm_hit_count = 0

        for function_id, node_id in zip(
            self.sfc.function_ids,
            placement_node_ids,
        ):
            function = self.function_map[
                function_id
            ]

            instance = self._get_or_create_instance(
                node_id=node_id,
                function_id=function_id,
            )

            instance_key = (
                node_id,
                function_id,
            )

            lifecycle_result = (
                self.lifecycle_manager.process_slot(
                    instance=instance,
                    function=function,
                    request_count=request_count,
                    keep_warm=False,
                )
            )

            used_instance_keys.add(instance_key)

            if lifecycle_result.cold_start_occurred:
                cold_start_function_ids.add(
                    function_id
                )

                cold_start_count += 1

            if lifecycle_result.warm_hit:
                warm_hit_count += 1

                # 该温实例以前由主动预热创建，
                # 现在第一次被真实请求成功使用。
                if (
                    instance_key
                    in self.pending_prewarmed_instances
                ):
                    prewarm_hit_count += 1

                    # 已经成功命中，不再处于“等待验证”状态。
                    self.pending_prewarmed_instances.discard(
                        instance_key
                    )

        return (
            used_instance_keys,
            cold_start_function_ids,
            cold_start_count,
            warm_hit_count,
            prewarm_hit_count,
        )

    def _age_unused_instances(
        self,
        used_instance_keys: set[tuple[int, int]],
    ) -> tuple[int, int]:
        """
        更新当前时隙没有使用的旧容器。

        达到生存窗口后，容器会从 WARM 变成 ABSENT。

        Returns
        -------
        expiration_count:
            当前时隙销毁的全部容器数量。

        prewarm_waste_count:
            销毁容器中，有多少个属于从未被请求使用的
            主动预热容器。
        """

        expiration_count = 0
        prewarm_waste_count = 0

        for instance_key, instance in (
            self.instances.items()
        ):
            # 真实请求使用的实例和当前正在预热的实例，
            # 不应在同一个时隙再次累计空闲时间。
            if instance_key in used_instance_keys:
                continue

            function = self.function_map[
                instance.function_id
            ]

            lifecycle_result = (
                self.lifecycle_manager.process_slot(
                    instance=instance,
                    function=function,
                    request_count=0,
                    keep_warm=False,
                )
            )

            if lifecycle_result.expired:
                expiration_count += 1

                # 该实例由主动预热创建，
                # 但直到被销毁也没有真实请求使用它。
                if (
                    instance_key
                    in self.pending_prewarmed_instances
                ):
                    prewarm_waste_count += 1

                    self.pending_prewarmed_instances.discard(
                        instance_key
                    )

        return (
            expiration_count,
            prewarm_waste_count,
        )
    
    def _get_warm_memory_snapshot(
        self,
    ) -> tuple[int, float]:
        """
        统计当前时隙结束时的温实例数量和内存占用。

        Returns
        -------
        warm_instance_count:
            当前处于 WARM 状态的实例数量。

        warm_memory_mb:
            所有温实例占用的总内存，单位为 MB。
        """

        warm_instance_count = 0
        warm_memory_mb = 0.0

        for instance in self.instances.values():
            if instance.status != InstanceStatus.WARM:
                continue

            warm_instance_count += 1

            function = self.function_map[
                instance.function_id
            ]

            warm_memory_mb += function.memory_mb

        return (
            warm_instance_count,
            warm_memory_mb,
        )

    def run(self) -> SimulationRunResult:
        """
        从铁路起点运行到铁路终点。
        """

        # 清除上一次运行留下的容器状态。
        self.instances.clear()

        # 清除上一次运行留下的待验证预热状态。
        self.pending_prewarmed_instances.clear()

        # 列车恢复到初始位置。
        train_state = self.mobility_model.reset()

        records: list[SlotSimulationRecord] = []

        previous_serving_mec: int | None = None

        while True:
            # 判断当前时隙是否发生 MEC 切换。
            handover_occurred = (
                previous_serving_mec is not None
                and train_state.serving_mec
                != previous_serving_mec
            )

            # 获取当前时隙的请求数量。
            request_count = (
                self.workload.request_count(
                    train_state.time_slot
                )
            )

            # 确定真实请求的函数部署位置。
            placement_node_ids = (
                self.placement_policy.place_functions(
                    sfc=self.sfc,
                    train_state=train_state,
                    topology=self.topology,
                )
            )

            # =================================================
            # 1. 处理轨迹感知提前预热
            # =================================================

            (
                prewarmed_instance_keys,
                prewarm_target_node_ids,
                prewarm_start_count,
                prewarm_startup_overhead_ms,
            ) = self._process_prewarming(
                train_state=train_state
            )

            # 当前时隙预热过的实例不能再累计空闲时间。
            used_instance_keys = set(
                prewarmed_instance_keys
            )

            cold_start_count = 0
            warm_hit_count = 0

            # 当前时隙成功使用的预热实例数量
            prewarm_hit_count = 0

            transmission_delay_ms = 0.0
            cold_start_delay_ms = 0.0
            execution_delay_ms = 0.0
            end_to_end_delay_ms = 0.0
            deadline_met: bool | None = None

            # =================================================
            # 2. 处理当前时隙的真实请求
            # =================================================

            if request_count > 0:
                (
                    request_instance_keys,
                    cold_start_function_ids,
                    cold_start_count,
                    warm_hit_count,
                    prewarm_hit_count,
                ) = self._process_active_request_batch(
                    placement_node_ids=(
                        placement_node_ids
                    ),
                    request_count=request_count,
                )

                # 合并真实请求使用的实例与预热实例。
                used_instance_keys.update(
                    request_instance_keys
                )

                sfc_result = execute_sfc_batch(
                    functions=self.functions,
                    sfc=self.sfc,
                    placement_node_ids=(
                        placement_node_ids
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

                transmission_delay_ms = (
                    sfc_result
                    .total_transmission_delay_ms
                )

                cold_start_delay_ms = (
                    sfc_result
                    .total_cold_start_delay_ms
                )

                execution_delay_ms = (
                    sfc_result
                    .total_execution_delay_ms
                )

                end_to_end_delay_ms = (
                    sfc_result
                    .total_end_to_end_delay_ms
                )

                deadline_met = (
                    sfc_result.deadline_met
                )

            # =================================================
            # 3. 更新没有使用的旧容器
            # =================================================

            (
                expiration_count,
                prewarm_waste_count,
            ) = self._age_unused_instances(
                used_instance_keys=(
                    used_instance_keys
                )
            )

            # 所有请求、预热和容器回收处理完成后，
            # 统计时隙结束时的温实例内存占用。
            (
                warm_instance_count,
                warm_memory_mb,
            ) = self._get_warm_memory_snapshot()

            # =================================================
            # 4. 保存当前时隙记录
            # =================================================

            records.append(
                SlotSimulationRecord(
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
                    placement_node_ids=tuple(
                        placement_node_ids
                    ),
                    prewarm_target_node_ids=(
                        prewarm_target_node_ids
                    ),
                    prewarm_start_count=(
                        prewarm_start_count
                    ),
                    prewarm_startup_overhead_ms=(
                        prewarm_startup_overhead_ms
                    ),
                    prewarm_hit_count=(
                        prewarm_hit_count
                    ),
                    prewarm_waste_count=(
                        prewarm_waste_count
                    ),
                    cold_start_count=(
                        cold_start_count
                    ),
                    warm_hit_count=warm_hit_count,
                    expiration_count=(
                        expiration_count
                    ),
                    transmission_delay_ms=(
                        transmission_delay_ms
                    ),
                    cold_start_delay_ms=(
                        cold_start_delay_ms
                    ),
                    execution_delay_ms=(
                        execution_delay_ms
                    ),
                    end_to_end_delay_ms=(
                        end_to_end_delay_ms
                    ),
                    deadline_met=deadline_met,
                    warm_instance_count=(
                        warm_instance_count
                    ),
                    warm_memory_mb=warm_memory_mb,
                )
            )

            # 到达线路终点后结束。
            if self.mobility_model.finished:
                break

            previous_serving_mec = (
                train_state.serving_mec
            )

            train_state = (
                self.mobility_model.step()
            )

        summary = self._build_summary(records)

        return SimulationRunResult(
            summary=summary,
            records=tuple(records),
        )

    def _build_summary(
        self,
        records: list[SlotSimulationRecord],
    ) -> SimulationSummary:
        """
        根据逐时隙记录计算汇总指标。
        """

        active_records = [
            record
            for record in records
            if record.request_count > 0
        ]

        total_requests = sum(
            record.request_count
            for record in records
        )

        request_batches = len(active_records)

        handover_count = sum(
            int(record.handover_occurred)
            for record in records
        )

        user_cold_starts = sum(
            record.cold_start_count
            for record in records
        )

        prewarm_starts = sum(
            record.prewarm_start_count
            for record in records
        )

        total_prewarm_startup_overhead_ms = sum(
            record.prewarm_startup_overhead_ms
            for record in records
        )

        warm_hits = sum(
            record.warm_hit_count
            for record in records
        )

        expiration_count = sum(
            record.expiration_count
            for record in records
        )

        deadline_violations = sum(
            int(record.deadline_met is False)
            for record in active_records
        )

        total_function_stages = (
            request_batches
            * len(self.sfc.function_ids)
        )

        if total_function_stages > 0:
            function_stage_cold_start_rate = (
                user_cold_starts
                / total_function_stages
            )
        else:
            function_stage_cold_start_rate = 0.0

        if request_batches > 0:
            deadline_violation_rate = (
                deadline_violations
                / request_batches
            )

            delays = [
                record.end_to_end_delay_ms
                for record in active_records
            ]

            average_batch_delay_ms = float(
                np.mean(delays)
            )

            p95_batch_delay_ms = float(
                np.percentile(delays, 95)
            )
        else:
            deadline_violation_rate = 0.0
            average_batch_delay_ms = 0.0
            p95_batch_delay_ms = 0.0

        return SimulationSummary(
            total_slots=len(records),
            total_requests=total_requests,
            request_batches=request_batches,
            handover_count=handover_count,
            user_cold_starts=user_cold_starts,
            prewarm_starts=prewarm_starts,
            total_prewarm_startup_overhead_ms=(
                total_prewarm_startup_overhead_ms
            ),
            warm_hits=warm_hits,
            expiration_count=expiration_count,
            deadline_violations=(
                deadline_violations
            ),
            function_stage_cold_start_rate=(
                function_stage_cold_start_rate
            ),
            deadline_violation_rate=(
                deadline_violation_rate
            ),
            average_batch_delay_ms=(
                average_batch_delay_ms
            ),
            p95_batch_delay_ms=(
                p95_batch_delay_ms
            ),
        )
