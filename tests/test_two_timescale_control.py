"""
test_two_timescale_control.py

测试双时间尺度控制器：

1. 慢时间尺度缓存；
2. MEC切换强制更新；
3. 单副本、冷备和热备决策；
4. 快时间尺度备用激活；
5. 热备用和冷备用接管；
6. 无可用副本时请求失败。
"""

import pytest

from src.two_timescale_control import (
    FastTimescaleState,
    RuleBasedTwoTimescaleController,
    SlowTimescaleDecision,
    SlowTimescaleState,
    StandbyMode,
)


def build_controller() -> (
    RuleBasedTwoTimescaleController
):
    """
    创建测试控制器。
    """

    return RuleBasedTwoTimescaleController(
        slow_period_slots=10,
        handover_hot_window_s=3.0,
        redundancy_reliability_threshold=0.99,
        high_failure_risk_threshold=0.10,
        hot_load_threshold_requests_per_slot=2.0,
    )


def build_slow_state(
    time_slot: int = 0,
    serving_mec: int = 0,
    next_mec: int = 1,
    request_rate: float = 1.0,
    failure_risk: float = 0.01,
    reliability_target: float = 0.99,
) -> SlowTimescaleState:
    """
    创建慢时间尺度测试状态。
    """

    return SlowTimescaleState(
        time_slot=time_slot,
        serving_mec=serving_mec,
        next_mec=next_mec,
        predicted_request_rate=request_rate,
        predicted_failure_risk=failure_risk,
        reliability_target=reliability_target,
    )


def build_fast_state(
    remaining_dwell_time_s: float,
    request_count: int,
    operational_node_ids: set[int],
) -> FastTimescaleState:
    """
    创建两个函数、跨故障域双副本的快状态。

    两个函数的主实例均为MEC-1，
    备用实例均为MEC-3。
    """

    return FastTimescaleState(
        time_slot=1,
        serving_mec=0,
        remaining_dwell_time_s=(
            remaining_dwell_time_s
        ),
        request_count=request_count,
        function_ids=(0, 1),
        candidate_node_ids={
            0: (0, 2),
            1: (0, 2),
        },
        operational_node_ids=frozenset(
            operational_node_ids
        ),
    )


def build_slow_decision(
    standby_mode: StandbyMode,
) -> SlowTimescaleDecision:
    """
    直接构造快时间尺度测试使用的慢动作。
    """

    use_redundancy = (
        standby_mode is not StandbyMode.SINGLE
    )

    return SlowTimescaleDecision(
        decision_slot=0,
        valid_until_slot=9,
        use_redundancy=use_redundancy,
        replica_count=(
            1
            if standby_mode is StandbyMode.SINGLE
            else 2
        ),
        standby_mode=standby_mode,
        reason="测试动作",
    )


def test_invalid_slow_period_is_rejected() -> None:
    """
    慢时间尺度周期必须大于0。
    """

    with pytest.raises(ValueError):
        RuleBasedTwoTimescaleController(
            slow_period_slots=0,
            handover_hot_window_s=3.0,
            redundancy_reliability_threshold=0.99,
            high_failure_risk_threshold=0.1,
            hot_load_threshold_requests_per_slot=2.0,
        )


def test_low_target_and_low_risk_choose_single_replica() -> None:
    """
    可靠性目标较低且故障风险较低时，
    选择单副本。
    """

    controller = build_controller()

    decision = controller.get_slow_decision(
        build_slow_state(
            reliability_target=0.95,
            failure_risk=0.01,
            request_rate=1.0,
        )
    )

    assert decision.use_redundancy is False
    assert decision.replica_count == 1
    assert decision.standby_mode is StandbyMode.SINGLE


def test_high_reliability_target_chooses_cold_redundancy() -> None:
    """
    可靠性要求达到0.99，
    但风险和负载都较低时使用跨域冷备。
    """

    controller = build_controller()

    decision = controller.get_slow_decision(
        build_slow_state(
            reliability_target=0.99,
            failure_risk=0.01,
            request_rate=1.0,
        )
    )

    assert decision.use_redundancy is True
    assert decision.replica_count == 2
    assert decision.standby_mode is StandbyMode.COLD


def test_high_failure_risk_chooses_hot_standby() -> None:
    """
    预测故障风险较高时使用全热备。
    """

    controller = build_controller()

    decision = controller.get_slow_decision(
        build_slow_state(
            reliability_target=0.99,
            failure_risk=0.20,
            request_rate=1.0,
        )
    )

    assert decision.standby_mode is StandbyMode.HOT


def test_slow_decision_is_cached_inside_period() -> None:
    """
    同一个慢周期内且未发生MEC切换时，
    应继续使用原决策。
    """

    controller = build_controller()

    first_decision = controller.get_slow_decision(
        build_slow_state(
            time_slot=0,
            failure_risk=0.01,
        )
    )

    # 即使时隙5的故障风险发生变化，
    # 慢周期尚未结束，因此仍返回原对象。
    second_decision = controller.get_slow_decision(
        build_slow_state(
            time_slot=5,
            failure_risk=0.50,
        )
    )

    assert second_decision is first_decision
    assert (
        second_decision.standby_mode
        is StandbyMode.COLD
    )


def test_handover_forces_new_slow_decision() -> None:
    """
    即使慢周期尚未结束，
    MEC切换也必须强制重新决策。
    """

    controller = build_controller()

    first_decision = controller.get_slow_decision(
        build_slow_state(
            time_slot=0,
            serving_mec=0,
            next_mec=1,
            failure_risk=0.01,
        )
    )

    second_decision = controller.get_slow_decision(
        build_slow_state(
            time_slot=5,
            serving_mec=1,
            next_mec=2,
            failure_risk=0.20,
        )
    )

    assert second_decision is not first_decision
    assert second_decision.decision_slot == 5
    assert (
        second_decision.standby_mode
        is StandbyMode.HOT
    )


def test_cold_mode_activates_backup_near_handover() -> None:
    """
    冷备模式接近MEC切换时，
    快时间尺度应临时激活备用节点。
    """

    controller = build_controller()

    decision = controller.get_fast_decision(
        state=build_fast_state(
            remaining_dwell_time_s=2.0,
            request_count=0,
            operational_node_ids={0, 2},
        ),
        slow_decision=build_slow_decision(
            StandbyMode.COLD
        ),
    )

    assert (
        decision.backup_activation_triggered
        is True
    )

    assert decision.function_hot_node_ids == {
        0: (0, 2),
        1: (0, 2),
    }

    assert decision.request_success is None


def test_cold_backup_failover_requires_cold_start() -> None:
    """
    非切换窗口内主节点失效时，
    冷备用可以接管，但需要冷启动。
    """

    controller = build_controller()

    decision = controller.get_fast_decision(
        state=build_fast_state(
            remaining_dwell_time_s=10.0,
            request_count=1,
            operational_node_ids={2},
        ),
        slow_decision=build_slow_decision(
            StandbyMode.COLD
        ),
    )

    assert decision.request_success is True

    assert (
        decision.selected_execution_node_ids
        == (2, 2)
    )

    assert decision.failover_function_ids == (
        0,
        1,
    )

    assert decision.cold_start_function_ids == (
        0,
        1,
    )


def test_hot_backup_failover_has_no_cold_start() -> None:
    """
    全热备模式下备用接管不需要冷启动。
    """

    controller = build_controller()

    decision = controller.get_fast_decision(
        state=build_fast_state(
            remaining_dwell_time_s=10.0,
            request_count=1,
            operational_node_ids={2},
        ),
        slow_decision=build_slow_decision(
            StandbyMode.HOT
        ),
    )

    assert decision.request_success is True

    assert decision.failover_function_ids == (
        0,
        1,
    )

    assert decision.cold_start_function_ids == ()