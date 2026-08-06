"""
two_timescale_control.py

本文件实现轨道边缘 Serverless SFC 的双时间尺度控制器。

慢时间尺度负责：

1. 是否启用跨故障域冗余；
2. 每个函数使用一个副本还是两个副本；
3. 备用副本采用冷备还是全热备；
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

    use_redundancy:
        是否使用跨故障域冗余副本。

    replica_count:
        每个函数的副本数量。

    standby_mode:
        基础主备模式。

    reason:
        生成该决策的原因，便于调试和论文分析。
    """

    decision_slot: int
    valid_until_slot: int

    use_redundancy: bool
    replica_count: int
    standby_mode: StandbyMode

    reason: str


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
        备用实例接管时仍为冷状态，
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
            standby_mode = StandbyMode.SINGLE
            replica_count = 1

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
                standby_mode = StandbyMode.HOT

                reason = (
                    "预测故障风险较高，"
                    "采用跨故障域全热备。"
                )

            elif high_request_load:
                standby_mode = StandbyMode.HOT

                reason = (
                    "预测请求负载较高，"
                    "采用跨故障域全热备。"
                )

            else:
                standby_mode = StandbyMode.COLD

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
            use_redundancy=use_redundancy,
            replica_count=replica_count,
            standby_mode=standby_mode,
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

        function_hot_node_ids: dict[
            int,
            tuple[int, ...],
        ] = {}

        selected_execution_node_ids: list[int] = []

        failover_function_ids: list[int] = []
        cold_start_function_ids: list[int] = []
        unavailable_function_ids: list[int] = []

        # 冷备模式在接近MEC切换时，
        # 由快时间尺度临时激活备用实例。
        near_handover = (
            state.remaining_dwell_time_s
            <= self.handover_hot_window_s
        )

        backup_activation_triggered = (
            slow_decision.standby_mode
            is StandbyMode.COLD
            and near_handover
            and slow_decision.use_redundancy
        )

        for function_id in state.function_ids:
            all_candidate_nodes = (
                state.candidate_node_ids[
                    function_id
                ]
            )

            # 慢时间尺度选择单副本时，
            # 即使外部候选计划中存在备用节点，
            # 快时间尺度也只允许使用主节点。
            if (
                slow_decision.standby_mode
                is StandbyMode.SINGLE
            ):
                usable_candidate_nodes = (
                    all_candidate_nodes[0],
                )
            else:
                usable_candidate_nodes = (
                    all_candidate_nodes
                )

            primary_node_id = (
                usable_candidate_nodes[0]
            )

            # 确定哪些副本处于温状态。
            if (
                slow_decision.standby_mode
                is StandbyMode.HOT
            ):
                hot_node_ids = (
                    usable_candidate_nodes
                )

            elif backup_activation_triggered:
                hot_node_ids = (
                    usable_candidate_nodes
                )

            else:
                hot_node_ids = (
                    primary_node_id,
                )

            function_hot_node_ids[
                function_id
            ] = hot_node_ids

            # 当前时隙没有请求时，
            # 只进行备用激活，不进行执行路由。
            if state.request_count == 0:
                continue

            operational_candidates = [
                node_id
                for node_id in usable_candidate_nodes
                if node_id
                in state.operational_node_ids
            ]

            if not operational_candidates:
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

            if selected_node_id != primary_node_id:
                failover_function_ids.append(
                    function_id
                )

                # 备用节点未处于温状态时，
                # 本次接管需要进行冷启动。
                if selected_node_id not in hot_node_ids:
                    cold_start_function_ids.append(
                        function_id
                    )

        if state.request_count == 0:
            request_success: bool | None = None

        elif unavailable_function_ids:
            request_success = False

            # 整条SFC无法完成，
            # 清空不完整的执行路径。
            selected_execution_node_ids = []

        else:
            request_success = True

        return FastTimescaleDecision(
            function_hot_node_ids=(
                function_hot_node_ids
            ),
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

        use_redundancy = (
            self.standby_mode
            is not StandbyMode.SINGLE
        )

        replica_count = (
            2 if use_redundancy else 1
        )

        if self.standby_mode is StandbyMode.SINGLE:
            reason = "固定单副本基线。"

        elif self.standby_mode is StandbyMode.HOT:
            reason = "固定跨故障域全热备基线。"

        elif self.enable_handover_activation:
            reason = (
                "固定跨故障域冷备，"
                "切换窗口内动态激活备用。"
            )

        else:
            reason = "固定跨故障域冷备基线。"

        self._decision = SlowTimescaleDecision(
            decision_slot=state.time_slot,
            valid_until_slot=10**12,
            use_redundancy=use_redundancy,
            replica_count=replica_count,
            standby_mode=self.standby_mode,
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

        function_hot_node_ids: dict[
            int,
            tuple[int, ...],
        ] = {}

        selected_execution_node_ids: list[int] = []

        failover_function_ids: list[int] = []
        cold_start_function_ids: list[int] = []
        unavailable_function_ids: list[int] = []

        near_handover = (
            state.remaining_dwell_time_s
            <= self.handover_hot_window_s
        )

        backup_activation_triggered = (
            self.standby_mode is StandbyMode.COLD
            and self.enable_handover_activation
            and near_handover
        )

        for function_id in state.function_ids:
            all_candidate_nodes = (
                state.candidate_node_ids[
                    function_id
                ]
            )

            if self.standby_mode is StandbyMode.SINGLE:
                usable_candidate_nodes = (
                    all_candidate_nodes[0],
                )
            else:
                usable_candidate_nodes = (
                    all_candidate_nodes
                )

            primary_node_id = (
                usable_candidate_nodes[0]
            )

            if self.standby_mode is StandbyMode.HOT:
                hot_node_ids = usable_candidate_nodes

            elif backup_activation_triggered:
                hot_node_ids = usable_candidate_nodes

            else:
                hot_node_ids = (
                    primary_node_id,
                )

            function_hot_node_ids[
                function_id
            ] = tuple(hot_node_ids)

            if state.request_count == 0:
                continue

            operational_candidates = [
                node_id
                for node_id in usable_candidate_nodes
                if node_id
                in state.operational_node_ids
            ]

            if not operational_candidates:
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

            if selected_node_id != primary_node_id:
                failover_function_ids.append(
                    function_id
                )

                if selected_node_id not in hot_node_ids:
                    cold_start_function_ids.append(
                        function_id
                    )

        if state.request_count == 0:
            request_success: bool | None = None

        elif unavailable_function_ids:
            request_success = False
            selected_execution_node_ids = []

        else:
            request_success = True

        return FastTimescaleDecision(
            function_hot_node_ids=(
                function_hot_node_ids
            ),
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