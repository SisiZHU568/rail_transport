"""
test_two_timescale_simulator.py

测试双时间尺度控制器接入完整仿真后的行为。
"""

import pytest

from src.config import load_config
from src.entities import (
    ServerlessFunction,
    ServicePriority,
    SFCType,
)
from src.failure_process import (
    ScriptedFailureProcess,
)
from src.failure_risk_prediction import (
    ConstantFailureRiskProvider,
)
from src.mobility import TrainMobilityModel
from src.network import build_linear_mec_network
from src.redundancy_placement import (
    build_reliability_aware_replica_planner,
)
from src.reliability import (
    build_fault_domain_reliability_model,
)
from src.runtime_reliability import (
    SingleReplicaPlanner,
)
from src.topology import build_linear_topology
from src.two_timescale_control import (
    FixedModeTwoTimescaleController,
    RuleBasedTwoTimescaleController,
    StandbyMode,
)
from src.two_timescale_simulator import (
    TwoTimescaleCostWeights,
    TwoTimescaleRuntimeSimulator,
)
from src.workload import DeterministicWorkload


def build_test_function() -> ServerlessFunction:
    return ServerlessFunction(
        function_id=0,
        name="测试函数",
        memory_mb=100.0,
        cpu_cycles_per_request=10.0,
        image_size_mb=50.0,
        warm_exec_time_ms=10.0,
        cold_start_time_ms=100.0,
        output_ratio=0.5,
    )


def build_test_sfc() -> SFCType:
    return SFCType(
        sfc_id=0,
        name="测试SFC",
        function_ids=[0],
        deadline_ms=500.0,
        reliability_target=0.99,
        priority=ServicePriority.CRITICAL,
    )


def build_cost_weights() -> (
    TwoTimescaleCostWeights
):
    return TwoTimescaleCostWeights(
        delay_cost_per_ms=1.0,
        memory_cost_per_mb_second=0.1,
        cold_start_cost_per_ms=0.2,
        sla_violation_penalty_per_batch=1000.0,
        slow_decision_cost=10.0,
        replica_reconfiguration_cost_per_function=20.0,
    )


def build_test_simulator(
    controller,
    risk: float,
    functions: list[ServerlessFunction] | None = None,
    sfc: SFCType | None = None,
    request_trace: list[int] | None = None,
    down_nodes_by_slot: (
        dict[int, set[int]] | None
    ) = None,
) -> TwoTimescaleRuntimeSimulator:
    """
    请求在时隙3到达。

    时隙3：
        列车位于1500米，
        当前接入MEC-2；
        MEC-2局部失效。
    """

    config = load_config(
        "configs/debug.yaml"
    )

    topology = build_linear_topology(config)

    network = build_linear_mec_network(
        config=config,
        topology=topology,
    )

    mobility_model = TrainMobilityModel(
        topology=topology,
        initial_position_m=0.0,
        speed_mps=500.0,
        slot_seconds=1.0,
    )

    selected_functions = (
        [build_test_function()]
        if functions is None
        else functions
    )

    selected_sfc = (
        build_test_sfc()
        if sfc is None
        else sfc
    )

    selected_request_trace = (
        [0, 0, 0, 1]
        if request_trace is None
        else request_trace
    )

    selected_down_nodes = (
        {3: {1}}
        if down_nodes_by_slot is None
        else down_nodes_by_slot
    )

    workload = DeterministicWorkload(
        request_trace=selected_request_trace,
        repeat=False,
    )

    failure_process = ScriptedFailureProcess(
        topology=topology,
        down_nodes_by_slot=selected_down_nodes,
    )

    return TwoTimescaleRuntimeSimulator(
        topology=topology,
        network=network,
        mobility_model=mobility_model,
        workload=workload,
        functions=selected_functions,
        sfc=selected_sfc,
        controller=controller,
        single_replica_planner=(
            SingleReplicaPlanner()
        ),
        redundant_replica_planner=(
            build_reliability_aware_replica_planner(
                config
            )
        ),
        failure_process=failure_process,
        failure_risk_provider=(
            ConstantFailureRiskProvider(
                risk=risk
            )
        ),
        reliability_model=(
            build_fault_domain_reliability_model(
                config=config,
                topology=topology,
            )
        ),
        prediction_horizon_slots=4,
        input_size_mb_per_request=1.0,
        slot_seconds=1.0,
        failover_delay_ms_per_function=20.0,
        cost_weights=build_cost_weights(),
        return_result_to_source=True,
    )


def build_rule_controller(
    handover_window_s: float = 1.0,
) -> RuleBasedTwoTimescaleController:
    return RuleBasedTwoTimescaleController(
        slow_period_slots=10,
        handover_hot_window_s=(
            handover_window_s
        ),
        redundancy_reliability_threshold=0.99,
        high_failure_risk_threshold=0.10,
        hot_load_threshold_requests_per_slot=2.0,
    )


def test_fixed_single_replica_fails() -> None:
    """
    主节点失效时，固定单副本请求失败。
    """

    controller = (
        FixedModeTwoTimescaleController(
            standby_mode=StandbyMode.SINGLE,
            handover_hot_window_s=1.0,
        )
    )

    result = build_test_simulator(
        controller=controller,
        risk=0.01,
    ).run()

    record = result.records[3]

    assert record.request_success is False
    assert result.summary.failed_requests == 1


def test_low_risk_rule_uses_cold_backup() -> None:
    """
    低风险时规则控制器选择冷备。

    时隙3不在1秒切换窗口内，
    因此备用接管需要冷启动。
    """

    result = build_test_simulator(
        controller=build_rule_controller(
            handover_window_s=1.0
        ),
        risk=0.01,
    ).run()

    record = result.records[3]

    assert record.slow_mode == "cold"
    assert record.request_success is True

    assert (
        record.cold_start_function_ids
        == (0,)
    )

    assert record.cold_start_delay_ms == pytest.approx(
        100.0
    )


def test_high_risk_rule_uses_hot_backup() -> None:
    """
    高风险时规则控制器选择全热备，
    备用接管不需要冷启动。
    """

    result = build_test_simulator(
        controller=build_rule_controller(),
        risk=0.20,
    ).run()

    record = result.records[3]

    assert record.slow_mode == "hot"
    assert record.request_success is True

    assert (
        record.failover_function_ids
        == (0,)
    )

    assert (
        record.cold_start_function_ids
        == ()
    )

    assert record.cold_start_delay_ms == 0.0


def test_handover_forces_slow_decision_update() -> None:
    """
    列车切换到MEC-2时，
    规则式慢时间尺度应重新决策。
    """

    result = build_test_simulator(
        controller=build_rule_controller(),
        risk=0.20,
    ).run()

    assert result.records[0].slow_decision_updated is True
    assert result.records[3].handover_occurred is True
    assert result.records[3].slow_decision_updated is True

    assert (
        result.summary.slow_decision_updates
        >= 2
    )


def test_total_cost_equals_all_cost_components() -> None:
    """
    综合成本必须等于五类成本之和。
    """

    result = build_test_simulator(
        controller=build_rule_controller(),
        risk=0.20,
    ).run()

    summary = result.summary

    expected = (
        summary.total_request_delay_cost
        + summary.total_memory_cost
        + summary.total_cold_start_cost
        + summary.total_sla_penalty
        + summary.total_slow_control_cost
    )

    assert summary.total_system_cost == pytest.approx(
        expected
    )


def test_slot_record_contains_resource_and_reliability_audit() -> None:
    """双副本时隙应保存完整且可行的约束审计结果。"""

    result = build_test_simulator(
        controller=build_rule_controller(),
        risk=0.20,
    ).run()

    audit = result.records[3].constraint_audit

    assert audit.node_cpu_demand == {
        2: pytest.approx(10.0)
    }
    assert audit.resource_constraints_met is True
    assert audit.replica_plan_valid is True
    assert audit.reliability_target_met is True
    assert audit.all_constraints_met is True
    assert audit.violation_reasons == ()


def test_single_replica_reliability_violation_is_recorded() -> None:
    """单副本可靠性不足时应保存原因，解释后续修复失败。"""

    controller = FixedModeTwoTimescaleController(
        standby_mode=StandbyMode.SINGLE,
        handover_hot_window_s=1.0,
    )

    result = build_test_simulator(
        controller=controller,
        risk=0.01,
    ).run()

    audit = result.records[0].constraint_audit

    assert audit.replica_plan_valid is True
    assert audit.reliability_target_met is False
    assert audit.all_constraints_met is False
    assert any(
        "可靠性" in reason
        for reason in audit.violation_reasons
    )


def test_resource_overload_is_repaired_before_execution() -> None:
    """资源超限方案必须先分散副本，再执行真实请求。"""

    functions = [
        ServerlessFunction(
            function_id=function_id,
            name=f"资源测试函数{function_id}",
            memory_mb=5000.0,
            cpu_cycles_per_request=60.0,
            image_size_mb=50.0,
            warm_exec_time_ms=10.0,
            cold_start_time_ms=100.0,
            output_ratio=1.0,
        )
        for function_id in (0, 1)
    ]

    sfc = SFCType(
        sfc_id=1,
        name="资源审计SFC",
        function_ids=[0, 1],
        deadline_ms=500.0,
        reliability_target=0.90,
        priority=ServicePriority.CRITICAL,
    )

    controller = FixedModeTwoTimescaleController(
        standby_mode=StandbyMode.HOT,
        handover_hot_window_s=1.0,
    )

    result = build_test_simulator(
        controller=controller,
        risk=0.01,
        functions=functions,
        sfc=sfc,
        request_trace=[1],
        down_nodes_by_slot={},
    ).run()

    record = result.records[0]

    assert (
        record.initial_constraint_audit
        .resource_constraints_met
        is False
    )
    assert record.fast_repair_attempted is True
    assert record.fast_repair_succeeded is True
    assert record.constraint_audit.all_constraints_met is True
    assert record.request_success is True
    assert all(
        demand <= 8192.0
        for demand in (
            record.constraint_audit
            .node_memory_demand_mb.values()
        )
    )


def test_unrepairable_single_replica_request_is_rejected() -> None:
    """单副本无法达到双故障域要求时，活动请求不得执行。"""

    controller = FixedModeTwoTimescaleController(
        standby_mode=StandbyMode.SINGLE,
        handover_hot_window_s=1.0,
    )
    result = build_test_simulator(
        controller=controller,
        risk=0.01,
        request_trace=[1],
        down_nodes_by_slot={},
    ).run()

    record = result.records[0]

    assert record.fast_repair_attempted is True
    assert record.fast_repair_succeeded is False
    assert record.request_success is False
    assert record.selected_execution_node_ids == ()
    assert record.end_to_end_delay_ms is None


def test_unrepairable_no_request_slot_is_not_counted_as_request_failure() -> None:
    """无请求时修复失败只记录约束问题，不生成虚假的失败批次。"""

    controller = FixedModeTwoTimescaleController(
        standby_mode=StandbyMode.SINGLE,
        handover_hot_window_s=1.0,
    )
    result = build_test_simulator(
        controller=controller,
        risk=0.01,
        request_trace=[0],
        down_nodes_by_slot={},
    ).run()

    record = result.records[0]

    assert record.fast_repair_succeeded is False
    assert record.request_success is None
    assert result.summary.failed_batches == 0


def test_summary_counts_repairs_and_constraint_rejections() -> None:
    """修复失败批次应被统计一次，并且只产生一次SLA惩罚。"""

    controller = FixedModeTwoTimescaleController(
        standby_mode=StandbyMode.SINGLE,
        handover_hot_window_s=1.0,
    )
    result = build_test_simulator(
        controller=controller,
        risk=0.01,
        request_trace=[1],
        down_nodes_by_slot={},
    ).run()

    summary = result.summary

    # 单副本部署在整条线路的每个时隙都无法满足双故障域要求，
    # 因此后台部署修复次数应覆盖全部时隙，而非只统计请求时隙。
    assert summary.fast_repair_attempts == summary.total_slots
    assert summary.fast_repair_successes == 0
    assert summary.fast_repair_failures == summary.total_slots
    assert summary.fast_repair_success_rate == 0.0
    assert summary.constraint_rejected_batches == 1
    assert summary.failed_batches == 1
    assert summary.sla_violations == 1
