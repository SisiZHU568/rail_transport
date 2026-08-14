"""在指定风险窗口放大 CTMC 失效率的唯一故障过程扩展。"""

from dataclasses import dataclass
from typing import Any

from src.failure_process import CTMCTransition, MarkovFailureProcess
from src.orchestration_config import CTMCRates, load_phase_a_config
from src.topology import LinearRailTopology


@dataclass(frozen=True)
class FailureRiskMultiplierWindow:
    """一个包含左右边界的连续时间失效率放大窗口。"""

    start_slot: int
    end_slot: int
    multiplier: float

    def __post_init__(self) -> None:
        if self.start_slot < 0:
            raise ValueError("高风险窗口起始时隙不能小于0。")
        if self.end_slot < self.start_slot:
            raise ValueError("高风险窗口结束时隙不能早于起始时隙。")
        if self.multiplier <= 0.0:
            raise ValueError("失效率放大倍数必须大于0。")

    def contains(self, time_slot: int) -> bool:
        """判断指定时隙是否位于当前窗口。"""

        return self.start_slot <= time_slot <= self.end_slot


class WindowedMarkovFailureProcess(MarkovFailureProcess):
    """复用精确 CTMC、稳态初始化和独立随机流的风险窗口过程。"""

    def __init__(
        self,
        *,
        topology: LinearRailTopology,
        domain_rates: dict[int, CTMCRates],
        node_rates: dict[int, CTMCRates],
        slot_seconds: float,
        risk_windows: list[FailureRiskMultiplierWindow],
        random_seed: int,
    ) -> None:
        self.risk_windows = tuple(risk_windows)
        super().__init__(
            topology=topology,
            domain_rates=domain_rates,
            node_rates=node_rates,
            slot_seconds=slot_seconds,
            random_seed=random_seed,
        )

    def _multiplier_for_slot(self, time_slot: int) -> float:
        """多个风险窗口重叠时使用最大的失效率倍数。"""

        active = [
            window.multiplier
            for window in self.risk_windows
            if window.contains(time_slot)
        ]
        return max(active, default=1.0)

    def transition_for_slot(
        self,
        entity_kind: str,
        entity_id: int,
        *,
        time_slot: int,
    ) -> CTMCTransition:
        """只放大 λ，随后用同一矩阵指数重新计算完整转移矩阵。"""

        multiplier = self._multiplier_for_slot(time_slot)
        if entity_kind == "domain":
            base_rates = self.domain_rates[entity_id]
        elif entity_kind == "node":
            base_rates = self.node_rates[entity_id]
        else:
            raise ValueError("entity_kind 必须是 domain 或 node。")
        return CTMCTransition.from_rates(
            CTMCRates(
                failure_rate_per_second=(
                    base_rates.failure_rate_per_second * multiplier
                ),
                recovery_rate_per_second=base_rates.recovery_rate_per_second,
            ),
            self.slot_seconds,
        )

    def _update_states(self) -> None:
        """按当前边界时隙的时变 CTMC 矩阵更新所有隐藏链。"""

        time_slot = self._next_time_slot
        for domain_id in self.domain_ids:
            transition = self.transition_for_slot(
                "domain", domain_id, time_slot=time_slot
            )
            draw = self._domain_random[domain_id].random()
            if self._domain_up[domain_id]:
                if draw < transition.failure_probability:
                    self._domain_up[domain_id] = False
            elif draw < transition.recovery_probability:
                self._domain_up[domain_id] = True

        for node_id in self.node_ids:
            transition = self.transition_for_slot(
                "node", node_id, time_slot=time_slot
            )
            draw = self._node_random[node_id].random()
            if self._node_local_up[node_id]:
                if draw < transition.failure_probability:
                    self._node_local_up[node_id] = False
            elif draw < transition.recovery_probability:
                self._node_local_up[node_id] = True


def build_windowed_markov_failure_process(
    config: dict[str, Any],
    topology: LinearRailTopology,
    random_seed: int | None = None,
) -> WindowedMarkovFailureProcess:
    """从阶段 A 配置构造带风险窗口的精确 CTMC 过程。"""

    phase_a = load_phase_a_config(config)
    selected_seed = phase_a.failure_base_seed if random_seed is None else random_seed
    if isinstance(selected_seed, bool) or not isinstance(selected_seed, int):
        raise TypeError("随机种子必须是整数。")
    topology_domain_ids = {
        node.fault_domain for node in topology.compute_nodes
    }
    multiplier = float(
        config["two_timescale_monte_carlo"]["high_risk_failure_multiplier"]
    )
    risk_windows = [
        FailureRiskMultiplierWindow(
            start_slot=int(item["start_slot"]),
            end_slot=int(item["end_slot"]),
            multiplier=multiplier,
        )
        for item in config["two_timescale_experiment"]["high_risk_windows"]
    ]
    return WindowedMarkovFailureProcess(
        topology=topology,
        domain_rates={
            domain_id: phase_a.domain_rates[domain_id]
            for domain_id in topology_domain_ids
        },
        node_rates={
            node.node_id: phase_a.node_rates[node.node_id]
            for node in topology.compute_nodes
        },
        slot_seconds=phase_a.fast_slot_seconds,
        risk_windows=risk_windows,
        random_seed=selected_seed,
    )
