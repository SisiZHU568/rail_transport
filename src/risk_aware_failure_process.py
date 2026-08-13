"""
risk_aware_failure_process.py

本文件实现带高风险时间窗口的随机故障过程。

普通时段：

    使用由长期可用率反推得到的基础失效概率。

高风险时段：

    基础失效概率 × high_risk_failure_multiplier。

例如：

    基础节点失效概率 = 0.01
    高风险倍数 = 8

则高风险窗口中的节点失效概率约为：

    0.01 × 8 = 0.08

五种控制策略在相同随机种子下，会得到完全相同的
节点和故障域随机状态，从而实现公平对比。
"""

from dataclasses import dataclass
import random
from typing import Any

from src.failure_process import (
    FailureProcess,
    InfrastructureState,
)
from src.topology import LinearRailTopology
from src.orchestration_config import load_ctmc_rate_maps


@dataclass(frozen=True)
class FailureRiskMultiplierWindow:
    """
    一个失效概率放大窗口。

    start_slot和end_slot均包含在窗口内。
    """

    start_slot: int
    end_slot: int
    multiplier: float

    def __post_init__(self) -> None:
        """
        检查窗口参数。
        """

        if self.start_slot < 0:
            raise ValueError(
                "高风险窗口起始时隙不能小于0。"
            )

        if self.end_slot < self.start_slot:
            raise ValueError(
                "高风险窗口结束时隙不能早于起始时隙。"
            )

        if self.multiplier <= 0:
            raise ValueError(
                "失效概率放大倍数必须大于0。"
            )

    def contains(
        self,
        time_slot: int,
    ) -> bool:
        """
        判断时隙是否位于当前窗口。
        """

        return (
            self.start_slot
            <= time_slot
            <= self.end_slot
        )


class WindowedMarkovFailureProcess(
    FailureProcess
):
    """
    带时变失效概率的二状态马尔可夫故障过程。

    对每个故障域和节点：

        UP --失效概率--> DOWN
        DOWN --恢复概率--> UP

    失效概率受到高风险时间窗口影响，
    恢复概率暂时保持不变。
    """

    def __init__(
        self,
        topology: LinearRailTopology,
        domain_failure_probabilities: dict[
            int,
            float,
        ],
        domain_recovery_probabilities: dict[
            int,
            float,
        ],
        node_failure_probabilities: dict[
            int,
            float,
        ],
        node_recovery_probabilities: dict[
            int,
            float,
        ],
        risk_windows: list[
            FailureRiskMultiplierWindow
        ],
        random_seed: int,
    ) -> None:
        """
        创建时变随机故障过程。
        """

        self.topology = topology
        self.random_seed = random_seed
        self.risk_windows = tuple(risk_windows)

        self.domain_ids = tuple(
            sorted(
                {
                    node.fault_domain
                    for node in topology.compute_nodes
                }
            )
        )

        self.node_ids = tuple(
            sorted(
                node.node_id
                for node in topology.compute_nodes
            )
        )

        self.domain_failure_probabilities = dict(
            domain_failure_probabilities
        )

        self.domain_recovery_probabilities = dict(
            domain_recovery_probabilities
        )

        self.node_failure_probabilities = dict(
            node_failure_probabilities
        )

        self.node_recovery_probabilities = dict(
            node_recovery_probabilities
        )

        self._validate_probability_dictionary(
            values=self.domain_failure_probabilities,
            expected_ids=set(self.domain_ids),
            parameter_name="故障域失效概率",
        )

        self._validate_probability_dictionary(
            values=self.domain_recovery_probabilities,
            expected_ids=set(self.domain_ids),
            parameter_name="故障域恢复概率",
        )

        self._validate_probability_dictionary(
            values=self.node_failure_probabilities,
            expected_ids=set(self.node_ids),
            parameter_name="节点失效概率",
        )

        self._validate_probability_dictionary(
            values=self.node_recovery_probabilities,
            expected_ids=set(self.node_ids),
            parameter_name="节点恢复概率",
        )

        self._random = random.Random(
            self.random_seed
        )

        self._domain_up: dict[int, bool] = {}
        self._node_local_up: dict[int, bool] = {}
        self._next_time_slot = 0

        self.reset()

    @staticmethod
    def _validate_probability_dictionary(
        values: dict[int, float],
        expected_ids: set[int],
        parameter_name: str,
    ) -> None:
        """
        检查概率字典。
        """

        if set(values) != expected_ids:
            raise ValueError(
                f"{parameter_name}的编号集合不完整。"
            )

        for entity_id, probability in values.items():
            if not 0 <= probability <= 1:
                raise ValueError(
                    f"{parameter_name}中编号"
                    f"{entity_id}的概率不在[0,1]。"
                )

    def _multiplier_for_slot(
        self,
        time_slot: int,
    ) -> float:
        """
        返回指定时隙使用的失效概率倍数。

        多个窗口重叠时取最大倍数。
        """

        active_multipliers = [
            window.multiplier
            for window in self.risk_windows
            if window.contains(time_slot)
        ]

        if not active_multipliers:
            return 1.0

        return max(active_multipliers)

    @staticmethod
    def _scaled_probability(
        base_probability: float,
        multiplier: float,
    ) -> float:
        """
        对基础概率进行放大，并限制在[0,1]。
        """

        return min(
            base_probability * multiplier,
            1.0,
        )

    def reset(self) -> None:
        """
        恢复随机种子以及初始全正常状态。
        """

        self._random = random.Random(
            self.random_seed
        )

        self._domain_up = {
            domain_id: True
            for domain_id in self.domain_ids
        }

        self._node_local_up = {
            node_id: True
            for node_id in self.node_ids
        }

        self._next_time_slot = 0

    def _update_states(
        self,
        time_slot: int,
    ) -> None:
        """
        更新到当前时隙的基础设施状态。
        """

        multiplier = self._multiplier_for_slot(
            time_slot
        )

        for domain_id in self.domain_ids:
            currently_up = self._domain_up[
                domain_id
            ]

            if currently_up:
                base_probability = (
                    self.domain_failure_probabilities[
                        domain_id
                    ]
                )

                transition_probability = (
                    self._scaled_probability(
                        base_probability=(
                            base_probability
                        ),
                        multiplier=multiplier,
                    )
                )

                if (
                    self._random.random()
                    < transition_probability
                ):
                    self._domain_up[
                        domain_id
                    ] = False

            else:
                recovery_probability = (
                    self.domain_recovery_probabilities[
                        domain_id
                    ]
                )

                if (
                    self._random.random()
                    < recovery_probability
                ):
                    self._domain_up[
                        domain_id
                    ] = True

        for node_id in self.node_ids:
            currently_up = self._node_local_up[
                node_id
            ]

            if currently_up:
                base_probability = (
                    self.node_failure_probabilities[
                        node_id
                    ]
                )

                transition_probability = (
                    self._scaled_probability(
                        base_probability=(
                            base_probability
                        ),
                        multiplier=multiplier,
                    )
                )

                if (
                    self._random.random()
                    < transition_probability
                ):
                    self._node_local_up[
                        node_id
                    ] = False

            else:
                recovery_probability = (
                    self.node_recovery_probabilities[
                        node_id
                    ]
                )

                if (
                    self._random.random()
                    < recovery_probability
                ):
                    self._node_local_up[
                        node_id
                    ] = True

    def state_for_slot(
        self,
        time_slot: int,
    ) -> InfrastructureState:
        """
        按照0、1、2、...顺序返回基础设施状态。
        """

        if time_slot != self._next_time_slot:
            raise ValueError(
                "WindowedMarkovFailureProcess必须"
                "按照0、1、2、...顺序访问时隙。"
            )

        # 时隙0使用初始全正常状态。
        if time_slot > 0:
            self._update_states(
                time_slot=time_slot
            )

        state = InfrastructureState(
            time_slot=time_slot,
            domain_up=dict(
                self._domain_up
            ),
            node_local_up=dict(
                self._node_local_up
            ),
        )

        self._next_time_slot += 1

        return state


def _failure_probability_from_availability(
    availability: float,
    recovery_probability: float,
) -> float:
    """
    根据稳态可用率反推每时隙失效概率。

        A = μ / (λ + μ)

    所以：

        λ = μ(1-A) / A
    """

    if not 0 < availability <= 1:
        raise ValueError(
            "长期可用率必须位于(0,1]。"
        )

    if not 0 <= recovery_probability <= 1:
        raise ValueError(
            "恢复概率必须位于[0,1]。"
        )

    failure_probability = (
        recovery_probability
        * (1.0 - availability)
        / availability
    )

    if failure_probability > 1:
        raise ValueError(
            "反推得到的失效概率大于1。"
        )

    return failure_probability


def build_windowed_markov_failure_process(
    config: dict[str, Any],
    topology: LinearRailTopology,
    random_seed: int | None = None,
) -> WindowedMarkovFailureProcess:
    """
    根据debug.yaml创建时变随机故障过程。
    """

    runtime_config = config[
        "runtime_failure"
    ]

    selected_random_seed = (
        runtime_config["random_seed"]
        if random_seed is None
        else random_seed
    )

    if not isinstance(
        selected_random_seed,
        int,
    ):
        raise TypeError(
            "随机种子必须是整数。"
        )

    domain_recovery_probability = float(
        runtime_config[
            "domain_recovery_probability_per_slot"
        ]
    )

    node_recovery_probability = float(
        runtime_config[
            "node_recovery_probability_per_slot"
        ]
    )

    domain_rates, _ = load_ctmc_rate_maps(config)
    fault_domain_availability = {
        domain_id: rates.steady_availability
        for domain_id, rates in domain_rates.items()
    }

    topology_domain_ids = {
        node.fault_domain
        for node in topology.compute_nodes
    }

    missing_domain_ids = (
        topology_domain_ids
        - set(fault_domain_availability)
    )

    if missing_domain_ids:
        raise ValueError(
            "以下计算节点故障域缺少可靠性配置："
            f"{sorted(missing_domain_ids)}。"
        )

    # 只读取当前计算拓扑实际使用的故障域。
    # 因此关闭中心云时，配置文件可以继续保留云故障域参数。
    domain_failure_probabilities = {
        domain_id: (
            _failure_probability_from_availability(
                availability=(
                    fault_domain_availability[
                        domain_id
                    ]
                ),
                recovery_probability=(
                    domain_recovery_probability
                ),
            )
        )
        for domain_id in topology_domain_ids
    }

    domain_recovery_probabilities = {
        domain_id: domain_recovery_probability
        for domain_id in topology_domain_ids
    }

    node_failure_probabilities = {
        node.node_id: (
            _failure_probability_from_availability(
                availability=(
                    node.reliability
                ),
                recovery_probability=(
                    node_recovery_probability
                ),
            )
        )
        for node in topology.compute_nodes
    }

    node_recovery_probabilities = {
        node.node_id: node_recovery_probability
        for node in topology.compute_nodes
    }

    multiplier = float(
        config["two_timescale_monte_carlo"][
            "high_risk_failure_multiplier"
        ]
    )

    risk_windows = [
        FailureRiskMultiplierWindow(
            start_slot=int(
                item["start_slot"]
            ),
            end_slot=int(
                item["end_slot"]
            ),
            multiplier=multiplier,
        )
        for item in config[
            "two_timescale_experiment"
        ]["high_risk_windows"]
    ]

    return WindowedMarkovFailureProcess(
        topology=topology,
        domain_failure_probabilities=(
            domain_failure_probabilities
        ),
        domain_recovery_probabilities=(
            domain_recovery_probabilities
        ),
        node_failure_probabilities=(
            node_failure_probabilities
        ),
        node_recovery_probabilities=(
            node_recovery_probabilities
        ),
        risk_windows=risk_windows,
        random_seed=selected_random_seed,
    )
