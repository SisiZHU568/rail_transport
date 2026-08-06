"""
slow_timescale_rl_env.py

本文件将轨道边缘Serverless SFC控制问题
封装为标准强化学习环境。

环境接口与Gymnasium保持一致：

    state, info = env.reset(seed=42)

    next_state, reward, terminated, truncated, info = (
        env.step(action)
    )

但本文件不强制依赖gymnasium，
便于先独立测试环境逻辑。

一个强化学习step对应一次慢时间尺度动作。

动作持续到：

1. slow_period_slots达到；或
2. 列车发生MEC切换；

二者中较早发生者。
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from src.entities import (
    ServerlessFunction,
    SFCType,
    TrainState,
)
from src.failure_process import (
    FailureProcess,
    InfrastructureState,
)
from src.failure_risk_prediction import (
    FailureRiskProvider,
)
from src.mobility import TrainMobilityModel
from src.network import LinearMECNetwork
from src.rl_reward import (
    RLRewardBreakdown,
    RLRewardWeights,
    RLWindowMetrics,
    calculate_rl_reward,
)
from src.sfc_execution import execute_sfc_batch
from src.topology import LinearRailTopology
from src.workload import DeterministicWorkload


class SlowControlAction(IntEnum):
    """
    强化学习慢时间尺度离散动作。
    """

    SINGLE = 0
    COLD = 1
    HOT = 2
    DYNAMIC = 3


@dataclass(frozen=True)
class EpisodeTraceSlot:
    """
    预生成的一条Episode时隙状态。
    """

    train_state: TrainState
    request_count: int
    infrastructure_state: InfrastructureState
    predicted_failure_risk: float


class SlowTimescaleRLEnvironment:
    """
    双时间尺度强化学习环境。

    状态共15维：

    0. 线路运行进度
    1. 当前接入MEC编号
    2. 下一MEC编号
    3. 当前MEC剩余驻留时间
    4. 预测平均请求负载
    5. 预测故障风险
    6. 是否存在上一窗口历史

    7～10. 上一动作one-hot编码
        7  SINGLE
        8  COLD
        9  HOT
        10 DYNAMIC

    11. 上一窗口请求成功率
    12. 上一窗口SLA违反率
    13. 上一窗口平均时延归一化值
    14. 上一窗口平均内存归一化值
    """

    STATE_FEATURE_NAMES = (
        "episode_progress",
        "serving_mec",
        "next_mec",
        "remaining_dwell_time",
        "predicted_request_rate",
        "predicted_failure_risk",
        "has_previous_window",
        "previous_action_single",
        "previous_action_cold",
        "previous_action_hot",
        "previous_action_dynamic",
        "previous_request_success_rate",
        "previous_sla_violation_rate",
        "previous_normalized_delay",
        "previous_normalized_memory",
    )

    def __init__(
        self,
        topology: LinearRailTopology,
        network: LinearMECNetwork,
        mobility_model: TrainMobilityModel,
        workload: DeterministicWorkload,
        functions: list[ServerlessFunction],
        sfc: SFCType,
        single_replica_planner,
        redundant_replica_planner,
        failure_process_builder: Callable[
            [int],
            FailureProcess,
        ],
        failure_risk_provider: FailureRiskProvider,
        slow_period_slots: int,
        handover_hot_window_s: float,
        input_size_mb_per_request: float,
        slot_seconds: float,
        failover_delay_ms_per_function: float,
        reward_weights: RLRewardWeights,
        default_seed: int,
        return_result_to_source: bool = True,
    ) -> None:
        """
        创建强化学习环境。
        """

        if len(functions) == 0:
            raise ValueError(
                "至少需要配置一个函数。"
            )

        if slow_period_slots <= 0:
            raise ValueError(
                "慢时间尺度周期必须大于0。"
            )

        if handover_hot_window_s <= 0:
            raise ValueError(
                "切换激活窗口必须大于0。"
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

        if not isinstance(default_seed, int):
            raise TypeError(
                "默认随机种子必须是整数。"
            )

        function_map: dict[
            int,
            ServerlessFunction,
        ] = {}

        for function in functions:
            if function.function_id in function_map:
                raise ValueError(
                    f"函数编号"
                    f"{function.function_id}重复。"
                )

            function_map[
                function.function_id
            ] = function

        for function_id in sfc.function_ids:
            if function_id not in function_map:
                raise KeyError(
                    f"缺少function_id="
                    f"{function_id}。"
                )

        self.topology = topology
        self.network = network
        self.mobility_model = mobility_model
        self.workload = workload

        self.functions = functions
        self.function_map = function_map
        self.sfc = sfc

        self.single_replica_planner = (
            single_replica_planner
        )

        self.redundant_replica_planner = (
            redundant_replica_planner
        )

        self.failure_process_builder = (
            failure_process_builder
        )

        self.failure_risk_provider = (
            failure_risk_provider
        )

        self.slow_period_slots = (
            slow_period_slots
        )

        self.handover_hot_window_s = (
            handover_hot_window_s
        )

        self.input_size_mb_per_request = (
            input_size_mb_per_request
        )

        self.slot_seconds = slot_seconds

        self.failover_delay_ms_per_function = (
            failover_delay_ms_per_function
        )

        self.reward_weights = reward_weights
        self.default_seed = default_seed

        self.return_result_to_source = (
            return_result_to_source
        )

        self.state_dim = len(
            self.STATE_FEATURE_NAMES
        )

        self.action_count = len(
            SlowControlAction
        )

        self.maximum_active_memory_mb = (
            2.0
            * sum(
                function.memory_mb
                for function in self.functions
            )
        )

        self.maximum_cold_start_delay_ms_per_batch = (
            sum(
                function.cold_start_time_ms
                for function in self.functions
            )
        )

        self._trace: tuple[
            EpisodeTraceSlot,
            ...,
        ] = ()

        self._cursor = 0
        self._terminated = False
        self._episode_seed = default_seed

        self._maximum_request_count = 1
        self._maximum_dwell_time_s = 1.0

        self._previous_action: (
            SlowControlAction | None
        ) = None

        self._previous_metrics: (
            RLWindowMetrics | None
        ) = None

        self._previous_reward_breakdown: (
            RLRewardBreakdown | None
        ) = None

        self._previous_candidate_map: (
            dict[int, tuple[int, ...]]
            | None
        ) = None

    def _build_episode_trace(
        self,
        seed: int,
    ) -> tuple[EpisodeTraceSlot, ...]:
        """
        预生成一整条列车运行、请求和故障轨迹。

        同一Episode中所有动作面对相同的故障轨迹，
        动作不会改变随机数的生成顺序。
        """

        failure_process = (
            self.failure_process_builder(
                seed
            )
        )

        failure_process.reset()

        train_state = (
            self.mobility_model.reset()
        )

        trace: list[
            EpisodeTraceSlot
        ] = []

        while True:
            request_count = (
                self.workload.request_count(
                    train_state.time_slot
                )
            )

            infrastructure_state = (
                failure_process.state_for_slot(
                    train_state.time_slot
                )
            )

            predicted_failure_risk = (
                self.failure_risk_provider.predict(
                    time_slot=(
                        train_state.time_slot
                    ),
                    train_state=train_state,
                )
            )

            trace.append(
                EpisodeTraceSlot(
                    train_state=train_state,
                    request_count=request_count,
                    infrastructure_state=(
                        infrastructure_state
                    ),
                    predicted_failure_risk=(
                        predicted_failure_risk
                    ),
                )
            )

            if self.mobility_model.finished:
                break

            train_state = (
                self.mobility_model.step()
            )

        return tuple(trace)

    def reset(
        self,
        seed: int | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        重置一个强化学习Episode。

        Returns
        -------
        state:
            15维float32状态向量。

        info:
            Episode初始信息。
        """

        selected_seed = (
            self.default_seed
            if seed is None
            else seed
        )

        if not isinstance(selected_seed, int):
            raise TypeError(
                "随机种子必须是整数。"
            )

        self._episode_seed = selected_seed

        self._trace = (
            self._build_episode_trace(
                seed=selected_seed
            )
        )

        if len(self._trace) == 0:
            raise RuntimeError(
                "Episode轨迹不能为空。"
            )

        self._cursor = 0
        self._terminated = False

        self._previous_action = None
        self._previous_metrics = None
        self._previous_reward_breakdown = None
        self._previous_candidate_map = None

        self._maximum_request_count = max(
            1,
            max(
                slot.request_count
                for slot in self._trace
            ),
        )

        self._maximum_dwell_time_s = max(
            1.0,
            max(
                slot.train_state
                .remaining_dwell_time_s
                for slot in self._trace
            ),
        )

        state = self._build_state()

        info = self._build_observation_info()

        info.update(
            {
                "episode_seed": (
                    self._episode_seed
                ),
                "total_fast_slots": (
                    len(self._trace)
                ),
                "state_feature_names": (
                    self.STATE_FEATURE_NAMES
                ),
            }
        )

        return state, info

    def _predict_request_rate(
        self,
        start_index: int,
    ) -> float:
        """
        计算未来一个慢周期内的平均请求数。
        """

        end_index = min(
            start_index
            + self.slow_period_slots,
            len(self._trace),
        )

        request_counts = [
            self._trace[index].request_count
            for index in range(
                start_index,
                end_index,
            )
        ]

        if not request_counts:
            return 0.0

        return float(
            np.mean(request_counts)
        )

    def _build_state(
        self,
    ) -> np.ndarray:
        """
        构建当前15维状态。
        """

        if self._terminated:
            return self._build_terminal_state()

        slot = self._trace[self._cursor]

        train_state = slot.train_state

        node_count = len(
            self.topology.sites
        )

        node_denominator = max(
            node_count - 1,
            1,
        )

        progress_denominator = max(
            len(self._trace) - 1,
            1,
        )

        episode_progress = (
            self._cursor
            / progress_denominator
        )

        serving_mec_normalized = (
            train_state.serving_mec
            / node_denominator
        )

        next_mec_normalized = (
            train_state.next_mec
            / node_denominator
        )

        dwell_normalized = min(
            train_state.remaining_dwell_time_s
            / self._maximum_dwell_time_s,
            1.0,
        )

        predicted_request_rate = (
            self._predict_request_rate(
                self._cursor
            )
        )

        request_rate_normalized = min(
            predicted_request_rate
            / self._maximum_request_count,
            1.0,
        )

        action_one_hot = np.zeros(
            self.action_count,
            dtype=np.float32,
        )

        if self._previous_action is not None:
            action_one_hot[
                int(self._previous_action)
            ] = 1.0

        if self._previous_metrics is None:
            has_history = 0.0
            previous_success_rate = 0.0
            previous_sla_rate = 0.0
            previous_delay = 0.0
            previous_memory = 0.0

        else:
            has_history = 1.0

            previous_success_rate = (
                self._previous_metrics
                .request_success_rate
            )

            previous_sla_rate = (
                self._previous_metrics
                .sla_violation_rate
            )

            if (
                self._previous_reward_breakdown
                is None
            ):
                raise RuntimeError(
                    "奖励分解记录缺失。"
                )

            previous_delay = (
                self._previous_reward_breakdown
                .normalized_delay
            )

            previous_memory = (
                self._previous_reward_breakdown
                .normalized_memory
            )

        state = np.array(
            [
                episode_progress,
                serving_mec_normalized,
                next_mec_normalized,
                dwell_normalized,
                request_rate_normalized,
                slot.predicted_failure_risk,
                has_history,
                *action_one_hot.tolist(),
                previous_success_rate,
                previous_sla_rate,
                previous_delay,
                previous_memory,
            ],
            dtype=np.float32,
        )

        if state.shape != (
            self.state_dim,
        ):
            raise RuntimeError(
                "强化学习状态维度异常。"
            )

        return state

    def _build_terminal_state(
        self,
    ) -> np.ndarray:
        """
        构建Episode结束后的终止状态。
        """

        state = np.zeros(
            self.state_dim,
            dtype=np.float32,
        )

        # episode_progress
        state[0] = 1.0

        if self._previous_action is not None:
            state[
                7 + int(self._previous_action)
            ] = 1.0

        if self._previous_metrics is not None:
            state[6] = 1.0

            state[11] = (
                self._previous_metrics
                .request_success_rate
            )

            state[12] = (
                self._previous_metrics
                .sla_violation_rate
            )

            if (
                self._previous_reward_breakdown
                is not None
            ):
                state[13] = (
                    self._previous_reward_breakdown
                    .normalized_delay
                )

                state[14] = (
                    self._previous_reward_breakdown
                    .normalized_memory
                )

        return state

    def _build_observation_info(
        self,
    ) -> dict:
        """
        返回当前决策点的原始可解释信息。
        """

        if self._terminated:
            return {
                "terminated": True,
                "decision_slot": None,
                "serving_mec": None,
                "next_mec": None,
                "remaining_dwell_time_s": 0.0,
                "predicted_request_rate": 0.0,
                "predicted_failure_risk": 0.0,
            }

        slot = self._trace[self._cursor]

        return {
            "terminated": False,
            "decision_slot": (
                slot.train_state.time_slot
            ),
            "serving_mec": (
                slot.train_state.serving_mec
            ),
            "next_mec": (
                slot.train_state.next_mec
            ),
            "remaining_dwell_time_s": (
                slot.train_state
                .remaining_dwell_time_s
            ),
            "predicted_request_rate": (
                self._predict_request_rate(
                    self._cursor
                )
            ),
            "predicted_failure_risk": (
                slot.predicted_failure_risk
            ),
        }

    def _next_decision_boundary(
        self,
        start_index: int,
    ) -> tuple[int, str]:
        """
        计算当前动作的结束位置。

        返回的end_index不包含在当前窗口中。
        """

        natural_end = min(
            start_index
            + self.slow_period_slots,
            len(self._trace),
        )

        current_serving_mec = (
            self._trace[start_index]
            .train_state
            .serving_mec
        )

        for index in range(
            start_index + 1,
            natural_end,
        ):
            serving_mec = (
                self._trace[index]
                .train_state
                .serving_mec
            )

            if serving_mec != current_serving_mec:
                return index, "handover"

        if natural_end >= len(self._trace):
            return natural_end, "episode_end"

        return natural_end, "period_end"

    def _build_candidate_map(
        self,
        action: SlowControlAction,
        train_state: TrainState,
    ) -> dict[int, tuple[int, ...]]:
        """
        根据强化学习动作生成候选副本计划。
        """

        if action is SlowControlAction.SINGLE:
            planner = self.single_replica_planner
        else:
            planner = (
                self.redundant_replica_planner
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

    def _hot_nodes_for_action(
        self,
        action: SlowControlAction,
        candidate_node_ids: tuple[int, ...],
        train_state: TrainState,
    ) -> tuple[int, ...]:
        """
        根据动作确定当前温实例节点。
        """

        primary_node_id = (
            candidate_node_ids[0]
        )

        if action is SlowControlAction.SINGLE:
            return (primary_node_id,)

        if action is SlowControlAction.COLD:
            return (primary_node_id,)

        if action is SlowControlAction.HOT:
            return candidate_node_ids

        if action is SlowControlAction.DYNAMIC:
            has_future_handover = (
                train_state.next_mec
                != train_state.serving_mec
            )

            near_handover = (
                train_state
                .remaining_dwell_time_s
                <= self.handover_hot_window_s
            )

            if (
                has_future_handover
                and near_handover
            ):
                return candidate_node_ids

            return (primary_node_id,)

        raise ValueError(
            f"不支持的动作：{action}"
        )

    def _count_plan_changes(
        self,
        previous_map: (
            dict[int, tuple[int, ...]]
            | None
        ),
        current_map: dict[
            int,
            tuple[int, ...],
        ],
    ) -> int:
        """
        统计副本节点发生变化的函数数量。
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

    def step(
        self,
        action: int,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict,
    ]:
        """
        执行一次慢时间尺度动作。

        Returns
        -------
        next_state:
            下一决策点状态。

        reward:
            当前窗口奖励。

        terminated:
            是否到达线路终点。

        truncated:
            是否因外部时间限制截断。
            当前始终为False。

        info:
            当前窗口详细指标。
        """

        if self._terminated:
            raise RuntimeError(
                "Episode已经结束，"
                "请先调用reset()。"
            )

        try:
            selected_action = (
                SlowControlAction(action)
            )
        except ValueError as error:
            raise ValueError(
                f"动作必须位于"
                f"[0,{self.action_count - 1}]。"
            ) from error

        start_index = self._cursor

        (
            end_index,
            boundary_reason,
        ) = self._next_decision_boundary(
            start_index
        )

        first_trace_slot = (
            self._trace[start_index]
        )

        initial_candidate_map = (
            self._build_candidate_map(
                action=selected_action,
                train_state=(
                    first_trace_slot.train_state
                ),
            )
        )

        reconfigured_function_stages = (
            self._count_plan_changes(
                previous_map=(
                    self._previous_candidate_map
                ),
                current_map=(
                    initial_candidate_map
                ),
            )
        )

        total_requests = 0
        successful_requests = 0

        request_batches = 0
        successful_batches = 0
        failed_batches = 0

        deadline_violations = 0

        successful_delays: list[float] = []

        total_cold_start_delay_ms = 0.0
        total_memory_mb_seconds = 0.0

        failover_function_stages = 0
        cold_start_function_stages = 0

        last_candidate_map = (
            initial_candidate_map
        )

        for index in range(
            start_index,
            end_index,
        ):
            trace_slot = self._trace[index]

            train_state = (
                trace_slot.train_state
            )

            request_count = (
                trace_slot.request_count
            )

            candidate_map = (
                self._build_candidate_map(
                    action=selected_action,
                    train_state=train_state,
                )
            )

            last_candidate_map = candidate_map

            operational_node_ids = {
                site.node.node_id
                for site in self.topology.sites
                if trace_slot
                .infrastructure_state
                .is_node_operational(
                    node_id=(
                        site.node.node_id
                    ),
                    topology=self.topology,
                )
            }

            function_hot_node_ids: dict[
                int,
                tuple[int, ...],
            ] = {}

            selected_execution_node_ids: list[
                int
            ] = []

            unavailable_function_ids: list[
                int
            ] = []

            failover_function_ids: list[
                int
            ] = []

            cold_start_function_ids: list[
                int
            ] = []

            for function_id in self.sfc.function_ids:
                candidate_node_ids = (
                    candidate_map[
                        function_id
                    ]
                )

                hot_node_ids = (
                    self._hot_nodes_for_action(
                        action=selected_action,
                        candidate_node_ids=(
                            candidate_node_ids
                        ),
                        train_state=train_state,
                    )
                )

                function_hot_node_ids[
                    function_id
                ] = hot_node_ids

                if request_count == 0:
                    continue

                operational_candidates = [
                    node_id
                    for node_id
                    in candidate_node_ids
                    if node_id
                    in operational_node_ids
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

                primary_node_id = (
                    candidate_node_ids[0]
                )

                if selected_node_id != primary_node_id:
                    failover_function_ids.append(
                        function_id
                    )

                    if (
                        selected_node_id
                        not in hot_node_ids
                    ):
                        cold_start_function_ids.append(
                            function_id
                        )

            total_requests += request_count

            request_success: bool | None

            if request_count == 0:
                request_success = None

            elif unavailable_function_ids:
                request_success = False

            else:
                request_success = True

            cold_activated_pairs: set[
                tuple[int, int]
            ] = set()

            if request_count > 0:
                request_batches += 1

                if request_success is False:
                    failed_batches += 1

                else:
                    successful_batches += 1
                    successful_requests += (
                        request_count
                    )

                    cold_start_ids = set(
                        cold_start_function_ids
                    )

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
                        request_count=(
                            request_count
                        ),
                        network=self.network,
                        cold_start_function_ids=(
                            cold_start_ids
                        ),
                        return_result_to_source=(
                            self.return_result_to_source
                        ),
                    )

                    failover_delay_ms = (
                        len(
                            failover_function_ids
                        )
                        * self
                        .failover_delay_ms_per_function
                    )

                    end_to_end_delay_ms = (
                        sfc_result
                        .total_end_to_end_delay_ms
                        + failover_delay_ms
                    )

                    successful_delays.append(
                        end_to_end_delay_ms
                    )

                    if (
                        end_to_end_delay_ms
                        > self.sfc.deadline_ms
                    ):
                        deadline_violations += 1

                    total_cold_start_delay_ms += (
                        sfc_result
                        .total_cold_start_delay_ms
                    )

                    failover_function_stages += len(
                        failover_function_ids
                    )

                    cold_start_function_stages += len(
                        cold_start_function_ids
                    )

                    for (
                        function_id,
                        selected_node_id,
                    ) in zip(
                        self.sfc.function_ids,
                        selected_execution_node_ids,
                    ):
                        if (
                            function_id
                            in cold_start_ids
                        ):
                            cold_activated_pairs.add(
                                (
                                    function_id,
                                    selected_node_id,
                                )
                            )

            active_pairs: set[
                tuple[int, int]
            ] = set()

            for (
                function_id,
                node_ids,
            ) in function_hot_node_ids.items():
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

            total_memory_mb_seconds += (
                active_memory_mb
                * self.slot_seconds
            )

        sla_violations = (
            failed_batches
            + deadline_violations
        )

        if total_requests > 0:
            request_success_rate = (
                successful_requests
                / total_requests
            )
        else:
            request_success_rate = 1.0

        if request_batches > 0:
            sla_violation_rate = (
                sla_violations
                / request_batches
            )
        else:
            sla_violation_rate = 0.0

        if successful_delays:
            average_successful_delay_ms = (
                float(
                    np.mean(
                        successful_delays
                    )
                )
            )
        else:
            average_successful_delay_ms = 0.0

        window_length = (
            end_index - start_index
        )

        if window_length <= 0:
            raise RuntimeError(
                "慢时间尺度窗口长度异常。"
            )

        window_duration_seconds = (
            window_length
            * self.slot_seconds
        )

        average_active_memory_mb = (
            total_memory_mb_seconds
            / window_duration_seconds
        )

        metrics = RLWindowMetrics(
            total_requests=total_requests,
            successful_requests=(
                successful_requests
            ),
            request_batches=request_batches,
            successful_batches=(
                successful_batches
            ),
            failed_batches=failed_batches,
            deadline_violations=(
                deadline_violations
            ),
            sla_violations=sla_violations,
            request_success_rate=(
                request_success_rate
            ),
            sla_violation_rate=(
                sla_violation_rate
            ),
            average_successful_delay_ms=(
                average_successful_delay_ms
            ),
            total_cold_start_delay_ms=(
                total_cold_start_delay_ms
            ),
            average_active_memory_mb=(
                average_active_memory_mb
            ),
            total_active_memory_mb_seconds=(
                total_memory_mb_seconds
            ),
            failover_function_stages=(
                failover_function_stages
            ),
            cold_start_function_stages=(
                cold_start_function_stages
            ),
            reconfigured_function_stages=(
                reconfigured_function_stages
            ),
        )

        reward_breakdown = (
            calculate_rl_reward(
                metrics=metrics,
                deadline_ms=(
                    self.sfc.deadline_ms
                ),
                maximum_active_memory_mb=(
                    self.maximum_active_memory_mb
                ),
                maximum_cold_start_delay_ms_per_batch=(
                    self
                    .maximum_cold_start_delay_ms_per_batch
                ),
                function_count=len(
                    self.sfc.function_ids
                ),
                weights=self.reward_weights,
            )
        )

        self._previous_action = (
            selected_action
        )

        self._previous_metrics = metrics

        self._previous_reward_breakdown = (
            reward_breakdown
        )

        self._previous_candidate_map = dict(
            last_candidate_map
        )

        self._cursor = end_index

        self._terminated = (
            self._cursor
            >= len(self._trace)
        )

        next_state = self._build_state()

        info = {
            "episode_seed": self._episode_seed,
            "action": int(selected_action),
            "action_name": selected_action.name,
            "window_start_index": start_index,
            "window_end_index": end_index,
            "window_start_slot": (
                self._trace[start_index]
                .train_state
                .time_slot
            ),
            "window_end_slot": (
                self._trace[end_index - 1]
                .train_state
                .time_slot
            ),
            "window_length": window_length,
            "boundary_reason": boundary_reason,
            "handover_boundary_reached": (
                boundary_reason == "handover"
            ),
            "metrics": metrics,
            "reward_breakdown": (
                reward_breakdown
            ),
            "next_observation": (
                self._build_observation_info()
            ),
        }

        terminated = self._terminated
        truncated = False

        return (
            next_state,
            reward_breakdown.reward,
            terminated,
            truncated,
            info,
        )