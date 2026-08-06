"""
retention_experiment.py

本文件负责运行一条请求轨迹下的容器保留策略实验。

它将以下模块连接起来：

1. RetentionPolicy：
   决定是否主动保温，以及生存窗口大小。

2. ContainerLifecycleManager：
   更新容器状态。

3. cost_model：
   计算每个时隙的成本。

最终输出：

1. 每个时隙的详细记录；
2. 整条请求轨迹的汇总指标。
"""

from dataclasses import dataclass

from src.cold_start import ContainerLifecycleManager
from src.cost_model import (
    CostWeights,
    SlotCostBreakdown,
    calculate_slot_cost,
)
from src.entities import (
    FunctionInstance,
    InstanceStatus,
    ServerlessFunction,
)
from src.retention_policies import RetentionPolicy


@dataclass(frozen=True)
class SlotExperimentRecord:
    """
    一个时隙的详细实验记录。
    """

    time_slot: int
    request_count: int

    before_status: InstanceStatus
    after_status: InstanceStatus

    cold_start_occurred: bool
    prewarm_occurred: bool
    warm_hit: bool
    expired: bool

    idle_slots_after: int

    request_delay_ms: float
    delay_cost: float
    retention_cost: float
    prewarm_cost: float
    total_cost: float


@dataclass(frozen=True)
class PolicySummary:
    """
    一种策略在完整请求轨迹上的汇总结果。
    """

    policy_id: str
    display_name: str

    # 请求总数量
    total_requests: int

    # 有请求的时隙数量
    request_batches: int

    # 由真实请求触发的冷启动次数
    user_cold_starts: int

    # 主动预热次数
    prewarm_starts: int

    # 命中温容器的请求批次数量
    warm_hit_batches: int

    # 容器被回收的次数
    expiration_count: int

    # 空闲但仍然保留温容器的时隙数量
    retained_idle_slots: int

    # 全部用户请求感知到的总时延
    total_request_delay_ms: float

    # 平均每个请求的时延
    average_delay_per_request_ms: float

    # 三种成本分项
    total_delay_cost: float
    total_retention_cost: float
    total_prewarm_cost: float

    # 总成本
    total_cost: float


@dataclass(frozen=True)
class PolicyRunResult:
    """
    一次策略实验的完整结果。
    """

    summary: PolicySummary
    records: list[SlotExperimentRecord]


def run_retention_policy(
    function: ServerlessFunction,
    request_trace: list[int],
    policy: RetentionPolicy,
    slot_seconds: float,
    cost_weights: CostWeights,
    node_id: int = 0,
) -> PolicyRunResult:
    """
    在指定请求轨迹上运行一种容器保留策略。

    Parameters
    ----------
    function:
        需要执行的 Serverless 函数。

    request_trace:
        每个时隙到达的请求数量。

    policy:
        当前采用的容器保留策略。

    slot_seconds:
        一个快时隙的长度。

    cost_weights:
        成本权重。

    node_id:
        函数实例所在节点编号。
        当前默认放在 MEC-1，也就是节点0。

    Returns
    -------
    PolicyRunResult:
        包含逐时隙记录和汇总指标。
    """

    if len(request_trace) == 0:
        raise ValueError("请求轨迹不能为空。")

    if any(request_count < 0 for request_count in request_trace):
        raise ValueError("请求数量不能小于 0。")

    if slot_seconds <= 0:
        raise ValueError("时隙长度必须大于 0。")

    # 每运行一种新策略，都创建一个全新的函数实例。
    #
    # 这样可以保证不同策略之间互不干扰，
    # 所有策略都从 ABSENT 状态开始。
    instance = FunctionInstance(
        node_id=node_id,
        function_id=function.function_id,
        status=InstanceStatus.ABSENT,
    )

    # 每种策略可以使用不同的容器生存窗口。
    lifecycle_manager = ContainerLifecycleManager(
        survival_slots=policy.survival_slots
    )

    records: list[SlotExperimentRecord] = []

    total_requests = 0
    request_batches = 0
    user_cold_starts = 0
    prewarm_starts = 0
    warm_hit_batches = 0
    expiration_count = 0
    retained_idle_slots = 0

    total_request_delay_ms = 0.0
    total_delay_cost = 0.0
    total_retention_cost = 0.0
    total_prewarm_cost = 0.0

    for time_slot, request_count in enumerate(request_trace):
        # 询问当前策略是否要主动保温。
        keep_warm = policy.should_keep_warm(
            time_slot=time_slot,
            request_count=request_count,
            instance=instance,
        )

        # 更新容器生命周期状态。
        lifecycle_result = lifecycle_manager.process_slot(
            instance=instance,
            function=function,
            request_count=request_count,
            keep_warm=keep_warm,
        )

        # 计算当前时隙的各类成本。
        slot_cost: SlotCostBreakdown = calculate_slot_cost(
            function=function,
            lifecycle_result=lifecycle_result,
            slot_seconds=slot_seconds,
            weights=cost_weights,
        )

        # ----------------------------------------------------
        # 更新统计指标
        # ----------------------------------------------------

        total_requests += request_count

        if request_count > 0:
            request_batches += 1

        # 主动预热不属于“用户请求触发的冷启动”。
        if (
            lifecycle_result.cold_start_occurred
            and not lifecycle_result.prewarm_occurred
        ):
            user_cold_starts += 1

        if lifecycle_result.prewarm_occurred:
            prewarm_starts += 1

        if lifecycle_result.warm_hit:
            warm_hit_batches += 1

        if lifecycle_result.expired:
            expiration_count += 1

        if slot_cost.retention_cost > 0:
            retained_idle_slots += 1

        total_request_delay_ms += slot_cost.request_delay_ms
        total_delay_cost += slot_cost.delay_cost
        total_retention_cost += slot_cost.retention_cost
        total_prewarm_cost += slot_cost.prewarm_cost

        # 保存当前时隙的详细记录。
        records.append(
            SlotExperimentRecord(
                time_slot=time_slot,
                request_count=request_count,
                before_status=lifecycle_result.before_status,
                after_status=lifecycle_result.after_status,
                cold_start_occurred=(
                    lifecycle_result.cold_start_occurred
                ),
                prewarm_occurred=(
                    lifecycle_result.prewarm_occurred
                ),
                warm_hit=lifecycle_result.warm_hit,
                expired=lifecycle_result.expired,
                idle_slots_after=(
                    lifecycle_result.idle_slots_after
                ),
                request_delay_ms=slot_cost.request_delay_ms,
                delay_cost=slot_cost.delay_cost,
                retention_cost=slot_cost.retention_cost,
                prewarm_cost=slot_cost.prewarm_cost,
                total_cost=slot_cost.total_cost,
            )
        )

    if total_requests > 0:
        average_delay_per_request_ms = (
            total_request_delay_ms / total_requests
        )
    else:
        average_delay_per_request_ms = 0.0

    total_cost = (
        total_delay_cost
        + total_retention_cost
        + total_prewarm_cost
    )

    summary = PolicySummary(
        policy_id=policy.policy_id,
        display_name=policy.display_name,
        total_requests=total_requests,
        request_batches=request_batches,
        user_cold_starts=user_cold_starts,
        prewarm_starts=prewarm_starts,
        warm_hit_batches=warm_hit_batches,
        expiration_count=expiration_count,
        retained_idle_slots=retained_idle_slots,
        total_request_delay_ms=total_request_delay_ms,
        average_delay_per_request_ms=(
            average_delay_per_request_ms
        ),
        total_delay_cost=total_delay_cost,
        total_retention_cost=total_retention_cost,
        total_prewarm_cost=total_prewarm_cost,
        total_cost=total_cost,
    )

    return PolicyRunResult(
        summary=summary,
        records=records,
    )