"""
two_timescale_control.py

本文件实现轨道边缘 Serverless SFC 的双时间尺度控制器。

慢时间尺度负责：

1. 决定每个函数的副本数量；
2. 选择实例按需、主实例保温或全部保温；
3. 决定是否允许使用中心云；
4. 决策在多个快时隙内保持有效。

快时间尺度负责：

1. 根据列车剩余驻留时间临时激活备用实例；
2. 根据基础设施状态选择实际执行节点；
3. 判断是否发生主备接管；
4. 判断备用接管是否需要冷启动；
5. 判断当前请求能否成功执行。

当前实现的是规则策略。
后续强化学习方法可以继承相同的状态、动作和结果接口。
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.rl_agent_action_space import (
    CloudPolicy,
    RetentionPolicy,
)
from src.sfc_deployment_intent import (
    FunctionDeploymentIntent,
    SFCDeploymentIntent,
)
from src.topology import LinearRailTopology


class StandbyMode(Enum):
    """
    慢时间尺度选择的基础主备模式。
    """

    # 每个函数只使用一个主副本
    SINGLE = "single"

    # 使用主备双副本，但默认只保持主实例温热
    COLD = "cold"

    # 主实例和备用实例都保持温热
    HOT = "hot"

    # 冷备基础上，允许快层在切换窗口临时激活备用
    DYNAMIC = "dynamic"


def legacy_mode_to_policy(
    standby_mode: StandbyMode,
) -> tuple[int, RetentionPolicy, CloudPolicy]:
    """把旧实验模式显式转换为新的结构化慢层策略。"""

    # 旧基线没有中心云开关，因此统一映射为仅边缘部署。
    mapping = {
        StandbyMode.SINGLE: (
            1,
            RetentionPolicy.PRIMARY_WARM,
            CloudPolicy.EDGE_ONLY,
        ),
        StandbyMode.COLD: (
            2,
            RetentionPolicy.PRIMARY_WARM,
            CloudPolicy.EDGE_ONLY,
        ),
        StandbyMode.HOT: (
            2,
            RetentionPolicy.ALL_WARM,
            CloudPolicy.EDGE_ONLY,
        ),
        StandbyMode.DYNAMIC: (
            2,
            RetentionPolicy.PRIMARY_WARM,
            CloudPolicy.EDGE_ONLY,
        ),
    }
    return mapping[StandbyMode(standby_mode)]


def retention_policy_to_legacy_mode(
    retention_policy: RetentionPolicy,
    replica_count: int,
) -> StandbyMode:
    """临时把新保留策略转换为旧快层接口可识别的模式。"""

    # 下一开发步骤会让快层直接理解 RetentionPolicy；在此之前，
    # 该转换只负责保证旧仿真与新慢层决策能够平稳衔接。
    if replica_count == 1:
        return StandbyMode.SINGLE
    if retention_policy is RetentionPolicy.ALL_WARM:
        return StandbyMode.HOT
    return StandbyMode.COLD


@dataclass(frozen=True)
class SlowTimescaleState:
    """
    慢时间尺度观测状态。

    Attributes
    ----------
    time_slot:
        当前快时隙编号。

    serving_mec:
        当前接入MEC。

    next_mec:
        下一接入MEC。

    predicted_request_rate:
        下一慢周期内预测的平均请求数。

    predicted_failure_risk:
        下一慢周期内预测的基础设施故障风险，
        取值范围为[0, 1]。

    reliability_target:
        当前SFC的可靠性要求。
    """

    time_slot: int
    serving_mec: int
    next_mec: int

    predicted_request_rate: float
    predicted_failure_risk: float
    reliability_target: float

    def __post_init__(self) -> None:
        """
        检查慢时间尺度状态是否合法。
        """

        if self.time_slot < 0:
            raise ValueError(
                "time_slot 不能小于0。"
            )

        if self.predicted_request_rate < 0:
            raise ValueError(
                "预测请求率不能小于0。"
            )

        if not 0 <= self.predicted_failure_risk <= 1:
            raise ValueError(
                "预测故障风险必须位于[0, 1]。"
            )

        if not 0 < self.reliability_target <= 1:
            raise ValueError(
                "可靠性目标必须位于(0, 1]。"
            )


@dataclass(frozen=True)
class SlowTimescaleDecision:
    """
    一次慢时间尺度决策。

    Attributes
    ----------
    decision_slot:
        该决策生成的时隙。

    valid_until_slot:
        该决策正常情况下持续有效到哪个时隙。

    replica_count:
        每个函数的副本数量。

    retention_policy:
        Serverless实例保留策略。

    cloud_policy:
        是否允许快层使用中心云节点。

    reason:
        生成该决策的原因，便于调试和论文分析。
    """

    decision_slot: int
    valid_until_slot: int

    replica_count: int
    retention_policy: RetentionPolicy
    cloud_policy: CloudPolicy

    reason: str

    @property
    def use_redundancy(self) -> bool:
        """副本数大于1时，说明慢层启用了冗余。"""

        return self.replica_count > 1


def build_rule_based_deployment_intent(
    *,
    slow_decision: SlowTimescaleDecision,
    candidate_map: dict[int, tuple[int, ...]],
    slot_seconds: float,
    backup_activation_triggered: bool,
) -> SFCDeploymentIntent:
    """把规则控制器的旧策略结果适配为共享逐 VNF 部署意图。

    此函数只服务于仍保留的规则实验，不改变 DPPO 的连续动作定义。旧决策的
    ``valid_until_slot`` 是闭区间，而通用部署意图使用右开区间，因此加一。
    """

    if slot_seconds <= 0:
        raise ValueError("slot_seconds must be positive.")
    if not candidate_map:
        raise ValueError("candidate_map cannot be empty.")
    if any(
        len(node_ids) != slow_decision.replica_count
        for node_ids in candidate_map.values()
    ):
        raise ValueError(
            "Rule-based candidate counts must match the slow decision."
        )

    exclusive_valid_until = slow_decision.valid_until_slot + 1
    retention_seconds = (
        exclusive_valid_until - slow_decision.decision_slot
    ) * float(slot_seconds)
    if slow_decision.retention_policy is RetentionPolicy.ON_DEMAND:
        primary_seconds = 0.0
        backup_seconds = 0.0
    elif slow_decision.retention_policy is RetentionPolicy.ALL_WARM:
        primary_seconds = retention_seconds
        backup_seconds = retention_seconds
    else:
        primary_seconds = retention_seconds
        backup_seconds = retention_seconds if backup_activation_triggered else 0.0

    function_intents = tuple(
        FunctionDeploymentIntent(
            function_id=function_id,
            preferred_node_ids=tuple(node_ids),
            replica_count=len(node_ids),
            primary_retention_seconds=primary_seconds,
            backup_retention_seconds=backup_seconds,
        )
        for function_id, node_ids in candidate_map.items()
    )
    return SFCDeploymentIntent(
        decision_slot=slow_decision.decision_slot,
        valid_until_slot=exclusive_valid_until,
        function_intents=function_intents,
        source_algorithm="rule_based",
    )


@dataclass(frozen=True)
class FastTimescaleState:
    """
    快时间尺度观测状态。

    Attributes
    ----------
    time_slot:
        当前快时隙。

    serving_mec:
        当前接入MEC。

    remaining_dwell_time_s:
        当前MEC剩余驻留时间。

    request_count:
        当前时隙请求数量。

    function_ids:
        SFC函数顺序。

    candidate_node_ids:
        每个函数的候选副本节点。

        每个元组中的第一个节点被视为主实例。

    operational_node_ids:
        当前可以正常工作的MEC节点集合。
    """

    time_slot: int
    serving_mec: int
    remaining_dwell_time_s: float
    request_count: int

    function_ids: tuple[int, ...]

    candidate_node_ids: dict[
        int,
        tuple[int, ...],
    ]

    operational_node_ids: frozenset[int]

    def __post_init__(self) -> None:
        """
        检查快时间尺度状态是否合法。
        """

        if self.time_slot < 0:
            raise ValueError(
                "time_slot 不能小于0。"
            )

        if self.remaining_dwell_time_s < 0:
            raise ValueError(
                "剩余驻留时间不能小于0。"
            )

        if self.request_count < 0:
            raise ValueError(
                "请求数量不能小于0。"
            )

        if len(self.function_ids) == 0:
            raise ValueError(
                "SFC至少需要一个函数。"
            )

        if len(self.function_ids) != len(
            set(self.function_ids)
        ):
            raise ValueError(
                "function_ids 中不能出现重复编号。"
            )

        if set(self.candidate_node_ids) != set(
            self.function_ids
        ):
            raise ValueError(
                "候选副本映射与SFC函数集合不一致。"
            )

        for function_id in self.function_ids:
            nodes = self.candidate_node_ids[
                function_id
            ]

            if len(nodes) == 0:
                raise ValueError(
                    f"函数{function_id}没有候选副本。"
                )

            if len(nodes) != len(set(nodes)):
                raise ValueError(
                    f"函数{function_id}的候选节点重复。"
                )


@dataclass(frozen=True)
class FastTimescaleDecision:
    """
    一次快时间尺度决策。

    function_hot_node_ids:
        当前时隙每个函数的温实例节点。

    selected_execution_node_ids:
        实际执行整条SFC的节点序列。

        没有请求或请求失败时为空元组。

    backup_activation_triggered:
        是否因为接近MEC切换而临时激活备用实例。

    failover_function_ids:
        从主实例切换到备用实例的函数。

    cold_start_function_ids:
        备用接管或修复迁移后的执行节点原来不是温实例，
        因而需要冷启动的函数。

    unavailable_function_ids:
        没有任何可用副本的函数。

    request_success:
        没有请求时为None；
        有请求且所有函数都有可用副本时为True；
        否则为False。
    """

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


def build_fast_decision_for_hot_nodes(
    state: FastTimescaleState,
    function_hot_node_ids: dict[int, tuple[int, ...]],
    *,
    previously_hot_node_ids: dict[int, tuple[int, ...]] | None = None,
    backup_activation_triggered: bool = False,
) -> FastTimescaleDecision:
    """根据显式温热节点生成快层执行路径，不读取任何学习算法动作类型。

    候选元组第一项仍是主副本。若主副本故障，则按照部署意图给出的后续
    节点顺序接管；是否冷启动只取决于执行前已知的温热节点集合。
    """

    if set(function_hot_node_ids) != set(state.function_ids):
        raise ValueError("Hot-node mapping must cover the complete SFC.")
    normalized_hot_nodes: dict[int, tuple[int, ...]] = {}
    for function_id in state.function_ids:
        node_ids = tuple(function_hot_node_ids[function_id])
        if len(node_ids) != len(set(node_ids)):
            raise ValueError(f"Function {function_id} has duplicate hot nodes.")
        normalized_hot_nodes[function_id] = node_ids

    selected_execution_node_ids: list[int] = []
    failover_function_ids: list[int] = []
    cold_start_function_ids: list[int] = []
    unavailable_function_ids: list[int] = []

    for function_id in state.function_ids:
        candidate_nodes = state.candidate_node_ids[function_id]
        primary_node_id = candidate_nodes[0]
        if state.request_count == 0:
            continue

        operational_candidates = [
            node_id
            for node_id in candidate_nodes
            if node_id in state.operational_node_ids
        ]
        if not operational_candidates:
            unavailable_function_ids.append(function_id)
            continue

        selected_node_id = operational_candidates[0]
        selected_execution_node_ids.append(selected_node_id)
        if selected_node_id != primary_node_id:
            failover_function_ids.append(function_id)

        known_hot_nodes = (
            normalized_hot_nodes[function_id]
            if previously_hot_node_ids is None
            else previously_hot_node_ids.get(function_id, ())
        )
        if selected_node_id not in known_hot_nodes:
            cold_start_function_ids.append(function_id)

    if state.request_count == 0:
        request_success: bool | None = None
    elif unavailable_function_ids:
        request_success = False
        # 任一 VNF 无可用副本时，整条 SFC 都不能执行部分路径。
        selected_execution_node_ids = []
    else:
        request_success = True

    return FastTimescaleDecision(
        function_hot_node_ids=normalized_hot_nodes,
        selected_execution_node_ids=tuple(selected_execution_node_ids),
        backup_activation_triggered=backup_activation_triggered,
        failover_function_ids=tuple(failover_function_ids),
        cold_start_function_ids=tuple(cold_start_function_ids),
        unavailable_function_ids=tuple(unavailable_function_ids),
        request_success=request_success,
    )


def build_fast_decision_for_plan(
    state: FastTimescaleState,
    retention_policy: RetentionPolicy,
    backup_activation_triggered: bool,
    previously_hot_node_ids: (
        dict[int, tuple[int, ...]] | None
    ) = None,
) -> FastTimescaleDecision:
    """根据给定副本方案生成无副作用的快层路由结果。"""

    function_hot_node_ids: dict[
        int,
        tuple[int, ...],
    ] = {}
    selected_execution_node_ids: list[int] = []
    failover_function_ids: list[int] = []
    cold_start_function_ids: list[int] = []
    unavailable_function_ids: list[int] = []

    for function_id in state.function_ids:
        all_candidate_nodes = state.candidate_node_ids[
            function_id
        ]

        # 副本数量已经由慢层模板和候选图确定；快层只在这些
        # 计划副本中选择执行节点，不能自行增删副本。
        usable_candidate_nodes = all_candidate_nodes
        primary_node_id = usable_candidate_nodes[0]

        if retention_policy is RetentionPolicy.ON_DEMAND:
            # 按需策略不提前保留容器，包括主副本也需要冷启动。
            hot_node_ids: tuple[int, ...] = ()
        elif (
            retention_policy
            is RetentionPolicy.ALL_WARM
        ):
            hot_node_ids = tuple(usable_candidate_nodes)
        elif backup_activation_triggered:
            # PRIMARY_WARM在切换窗口只额外激活第一个备用，
            # 三副本时不会把第二个备用也误当成常驻温实例。
            hot_node_ids = tuple(
                usable_candidate_nodes[:2]
            )
        else:
            hot_node_ids = (primary_node_id,)

        function_hot_node_ids[function_id] = hot_node_ids

        # 修复后新建但不在本次执行路径上的HOT备用实例仍计入内存；
        # 当前阶段暂不把这类后台启动开销叠加到用户端到端时延，
        # 后续接入完整实例生命周期模型时再细化该部分成本。

        # 没有请求时只维护实例温热状态，不构造执行路径。
        if state.request_count == 0:
            continue

        operational_candidates = [
            node_id
            for node_id in usable_candidate_nodes
            if node_id in state.operational_node_ids
        ]

        if not operational_candidates:
            unavailable_function_ids.append(function_id)
            continue

        selected_node_id = operational_candidates[0]
        selected_execution_node_ids.append(selected_node_id)

        if selected_node_id != primary_node_id:
            failover_function_ids.append(function_id)

        # 普通控制器依据当前计划判断温实例；修复器需要传入修复前
        # 真实温实例，避免把刚迁移的新节点错误地当成已经预热。
        known_hot_node_ids = (
            hot_node_ids
            if previously_hot_node_ids is None
            else previously_hot_node_ids.get(
                function_id,
                (),
            )
        )
        if selected_node_id not in known_hot_node_ids:
            cold_start_function_ids.append(function_id)

    if state.request_count == 0:
        request_success: bool | None = None
    elif unavailable_function_ids:
        request_success = False

        # 任一函数无可用副本时，整条SFC路径不完整，不能部分执行。
        selected_execution_node_ids = []
    else:
        request_success = True

    return FastTimescaleDecision(
        function_hot_node_ids=function_hot_node_ids,
        selected_execution_node_ids=tuple(
            selected_execution_node_ids
        ),
        backup_activation_triggered=(
            backup_activation_triggered
        ),
        failover_function_ids=tuple(
            failover_function_ids
        ),
        cold_start_function_ids=tuple(
            cold_start_function_ids
        ),
        unavailable_function_ids=tuple(
            unavailable_function_ids
        ),
        request_success=request_success,
    )


class RuleBasedTwoTimescaleController:
    """
    规则式双时间尺度控制器。

    慢时间尺度规则
    ----------------
    1. 当SFC可靠性目标达到冗余阈值时，
       启用跨故障域双副本；

    2. 当预测故障风险较高或预测负载较高时，
       采用全热备；

    3. 其他需要冗余的情况采用冷备；

    4. 可靠性要求较低且故障风险较低时，
       可以使用单副本。

    快时间尺度规则
    ----------------
    1. HOT模式下，主备实例始终保持温状态；

    2. COLD模式下，通常只有主实例保持温状态；

    3. 当列车进入切换窗口时，
       快时间尺度临时将备用实例激活为温状态；

    4. 主实例故障时，按候选顺序选择第一个正常副本。
    """

    def __init__(
        self,
        slow_period_slots: int,
        handover_hot_window_s: float,
        redundancy_reliability_threshold: float,
        high_failure_risk_threshold: float,
        hot_load_threshold_requests_per_slot: float,
    ) -> None:
        """
        创建双时间尺度控制器。
        """

        if slow_period_slots <= 0:
            raise ValueError(
                "慢时间尺度周期必须大于0。"
            )

        if handover_hot_window_s <= 0:
            raise ValueError(
                "切换预热窗口必须大于0。"
            )

        if not 0 < redundancy_reliability_threshold <= 1:
            raise ValueError(
                "冗余可靠性阈值必须位于(0,1]。"
            )

        if not 0 <= high_failure_risk_threshold <= 1:
            raise ValueError(
                "高故障风险阈值必须位于[0,1]。"
            )

        if hot_load_threshold_requests_per_slot < 0:
            raise ValueError(
                "高负载阈值不能小于0。"
            )

        self.slow_period_slots = slow_period_slots

        self.handover_hot_window_s = (
            handover_hot_window_s
        )

        self.redundancy_reliability_threshold = (
            redundancy_reliability_threshold
        )

        self.high_failure_risk_threshold = (
            high_failure_risk_threshold
        )

        self.hot_load_threshold_requests_per_slot = (
            hot_load_threshold_requests_per_slot
        )

        self._last_slow_decision: (
            SlowTimescaleDecision | None
        ) = None

        self._last_serving_mec: int | None = None

    def reset(self) -> None:
        """
        清除上一轮仿真的慢时间尺度缓存。
        """

        self._last_slow_decision = None
        self._last_serving_mec = None

    def get_slow_decision(
        self,
        state: SlowTimescaleState,
    ) -> SlowTimescaleDecision:
        """
        获取当前有效的慢时间尺度决策。

        以下情况会重新计算：

        1. 仿真刚开始；
        2. 上一决策已经过期；
        3. 当前接入MEC发生变化。
        """

        handover_occurred = (
            self._last_serving_mec is not None
            and state.serving_mec
            != self._last_serving_mec
        )

        decision_expired = (
            self._last_slow_decision is None
            or state.time_slot
            > self._last_slow_decision.valid_until_slot
        )

        if (
            self._last_slow_decision is not None
            and not decision_expired
            and not handover_occurred
        ):
            return self._last_slow_decision

        decision = self._compute_slow_decision(
            state
        )

        self._last_slow_decision = decision
        self._last_serving_mec = state.serving_mec

        return decision

    def _compute_slow_decision(
        self,
        state: SlowTimescaleState,
    ) -> SlowTimescaleDecision:
        """
        根据预测状态生成新的慢时间尺度决策。
        """

        reliability_requires_redundancy = (
            state.reliability_target
            >= self.redundancy_reliability_threshold
        )

        risk_requires_redundancy = (
            state.predicted_failure_risk
            >= self.high_failure_risk_threshold
        )

        use_redundancy = (
            reliability_requires_redundancy
            or risk_requires_redundancy
        )

        if not use_redundancy:
            replica_count = 1
            retention_policy = (
                RetentionPolicy.PRIMARY_WARM
            )

            reason = (
                "可靠性目标和预测故障风险均较低，"
                "采用单副本。"
            )

        else:
            replica_count = 2

            high_failure_risk = (
                state.predicted_failure_risk
                >= self.high_failure_risk_threshold
            )

            high_request_load = (
                state.predicted_request_rate
                >= self
                .hot_load_threshold_requests_per_slot
            )

            if high_failure_risk:
                retention_policy = (
                    RetentionPolicy.ALL_WARM
                )

                reason = (
                    "预测故障风险较高，"
                    "采用跨故障域全热备。"
                )

            elif high_request_load:
                retention_policy = (
                    RetentionPolicy.ALL_WARM
                )

                reason = (
                    "预测请求负载较高，"
                    "采用跨故障域全热备。"
                )

            else:
                retention_policy = (
                    RetentionPolicy.PRIMARY_WARM
                )

                reason = (
                    "可靠性要求需要双副本，"
                    "但当前风险和负载较低，"
                    "采用跨故障域冷备。"
                )

        return SlowTimescaleDecision(
            decision_slot=state.time_slot,
            valid_until_slot=(
                state.time_slot
                + self.slow_period_slots
                - 1
            ),
            replica_count=replica_count,
            retention_policy=retention_policy,
            # 当前规则基线仍只在边缘侧规划副本。
            cloud_policy=CloudPolicy.EDGE_ONLY,
            reason=reason,
        )

    def get_fast_decision(
        self,
        state: FastTimescaleState,
        slow_decision: SlowTimescaleDecision,
    ) -> FastTimescaleDecision:
        """
        根据慢时间尺度动作和当前时隙状态，
        生成快时间尺度决策。
        """

        # 冷备模式在接近MEC切换时，
        # 由快时间尺度临时激活备用实例。
        near_handover = (
            state.remaining_dwell_time_s
            <= self.handover_hot_window_s
        )

        backup_activation_triggered = (
            slow_decision.retention_policy
            is RetentionPolicy.PRIMARY_WARM
            and near_handover
            and slow_decision.use_redundancy
        )

        return build_fast_decision_for_plan(
            state=state,
            retention_policy=(
                slow_decision.retention_policy
            ),
            backup_activation_triggered=(
                backup_activation_triggered
            ),
        )


def build_rule_based_two_timescale_controller(
    config: dict[str, Any],
) -> RuleBasedTwoTimescaleController:
    """
    根据debug.yaml创建双时间尺度控制器。
    """

    control_config = config["two_timescale"]

    return RuleBasedTwoTimescaleController(
        slow_period_slots=int(
            control_config["slow_period_slots"]
        ),
        handover_hot_window_s=float(
            control_config["handover_hot_window_s"]
        ),
        redundancy_reliability_threshold=float(
            control_config[
                "redundancy_reliability_threshold"
            ]
        ),
        high_failure_risk_threshold=float(
            control_config[
                "high_failure_risk_threshold"
            ]
        ),
        hot_load_threshold_requests_per_slot=float(
            control_config[
                "hot_load_threshold_requests_per_slot"
            ]
        ),
    )

class FixedModeTwoTimescaleController:
    """
    固定主备模式控制器。

    该控制器用于构建正式实验中的固定基线：

    1. 固定单副本；
    2. 固定冷备用；
    3. 固定全热备；
    4. 固定轨迹感知动态主备。

    与规则式双时间尺度控制器不同，
    它不会根据故障风险或请求负载改变慢动作。
    """

    def __init__(
        self,
        standby_mode: StandbyMode,
        handover_hot_window_s: float,
        enable_handover_activation: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        standby_mode:
            固定使用的主备模式。

        handover_hot_window_s:
            接近切换时的备用激活窗口。

        enable_handover_activation:
            是否允许冷备在切换窗口内转为温备。

            False：
                固定冷备。

            True：
                固定轨迹感知动态主备。
        """

        if handover_hot_window_s <= 0:
            raise ValueError(
                "切换窗口必须大于0。"
            )

        if (
            standby_mode is StandbyMode.SINGLE
            and enable_handover_activation
        ):
            raise ValueError(
                "单副本模式不存在备用激活。"
            )

        self.standby_mode = standby_mode

        self.handover_hot_window_s = (
            handover_hot_window_s
        )

        self.enable_handover_activation = (
            enable_handover_activation
        )

        self._decision: (
            SlowTimescaleDecision | None
        ) = None

    def reset(self) -> None:
        """
        清除上一次仿真的固定决策。
        """

        self._decision = None

    def get_slow_decision(
        self,
        state: SlowTimescaleState,
    ) -> SlowTimescaleDecision:
        """
        返回固定慢时间尺度决策。

        固定策略只在仿真开始时生成一次慢动作。
        """

        if self._decision is not None:
            return self._decision

        (
            replica_count,
            retention_policy,
            cloud_policy,
        ) = legacy_mode_to_policy(
            self.standby_mode
        )

        if self.standby_mode is StandbyMode.SINGLE:
            reason = "固定单副本基线。"

        elif self.standby_mode is StandbyMode.HOT:
            reason = "固定跨故障域全热备基线。"

        elif (
            self.enable_handover_activation
            or self.standby_mode is StandbyMode.DYNAMIC
        ):
            reason = (
                "固定跨故障域冷备，"
                "切换窗口内动态激活备用。"
            )

        else:
            reason = "固定跨故障域冷备基线。"

        self._decision = SlowTimescaleDecision(
            decision_slot=state.time_slot,
            valid_until_slot=10**12,
            replica_count=replica_count,
            retention_policy=retention_policy,
            cloud_policy=cloud_policy,
            reason=reason,
        )

        return self._decision

    def get_fast_decision(
        self,
        state: FastTimescaleState,
        slow_decision: SlowTimescaleDecision,
    ) -> FastTimescaleDecision:
        """
        根据固定主备模式执行快时间尺度路由。
        """

        near_handover = (
            state.remaining_dwell_time_s
            <= self.handover_hot_window_s
        )

        backup_activation_triggered = (
            self.standby_mode
            in (StandbyMode.COLD, StandbyMode.DYNAMIC)
            and (
                self.enable_handover_activation
                or self.standby_mode is StandbyMode.DYNAMIC
            )
            and near_handover
        )

        return build_fast_decision_for_plan(
            state=state,
            retention_policy=(
                slow_decision.retention_policy
            ),
            backup_activation_triggered=(
                backup_activation_triggered
            ),
        )
