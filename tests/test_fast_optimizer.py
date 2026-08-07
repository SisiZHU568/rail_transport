"""测试慢层模板约束下的确定性快层可行性修复。"""

from src.constraint_audit import SlotConstraintAuditor
from src.entities import (
    EdgeNode,
    NodeType,
    ServerlessFunction,
    ServicePriority,
    SFCType,
)
from src.fast_optimizer import (
    FastFeasibilityOptimizer,
    FastOptimizationResult,
)
from src.reliability import FaultDomainReliabilityModel
from src.topology import LinearRailTopology, TracksideSite
from src.two_timescale_control import (
    FastTimescaleState,
    SlowTimescaleDecision,
    StandbyMode,
    build_fast_decision_for_plan,
    legacy_mode_to_policy,
    retention_policy_to_legacy_mode,
)


def optimize_plan(
    *,
    auditor: SlotConstraintAuditor,
    topology: LinearRailTopology,
    state: FastTimescaleState,
    slow_decision: SlowTimescaleDecision,
) -> FastOptimizationResult:
    """先审计原方案，再调用修复器，模拟真实执行顺序。"""

    initial_decision = build_fast_decision_for_plan(
        state=state,
        standby_mode=retention_policy_to_legacy_mode(
            slow_decision.retention_policy,
            slow_decision.replica_count,
        ),
        backup_activation_triggered=False,
    )
    cold_pairs = {
        (
            function_id,
            initial_decision.selected_execution_node_ids[index],
        )
        for index, function_id in enumerate(
            state.function_ids
        )
        if function_id
        in initial_decision.cold_start_function_ids
        and initial_decision.request_success is True
    }
    initial_audit = auditor.audit(
        request_count=state.request_count,
        expected_replica_count=slow_decision.replica_count,
        candidate_map=state.candidate_node_ids,
        selected_execution_node_ids=(
            initial_decision.selected_execution_node_ids
        ),
        request_success=initial_decision.request_success,
        function_hot_node_ids=(
            initial_decision.function_hot_node_ids
        ),
        cold_activated_pairs=cold_pairs,
    )
    optimizer = FastFeasibilityOptimizer(
        functions=auditor.functions,
        sfc=auditor.sfc,
        topology=topology,
        auditor=auditor,
        return_result_to_source=True,
    )
    return optimizer.optimize(
        state=state,
        slow_decision=slow_decision,
        initial_decision=initial_decision,
        initial_audit=initial_audit,
    )


def optimize_plan_for_test(
    *,
    replica_count: int,
    standby_mode: StandbyMode,
    candidate_map: dict[int, tuple[int, ...]],
    request_count: int,
    minimum_distinct_fault_domains: int,
    fault_domains: tuple[int, ...] = (0, 1, 2, 3, 4),
    operational_node_ids: set[int] | None = None,
    function_memory_mb: float = 600.0,
    cpu_per_request: float = 60.0,
) -> tuple[FastOptimizationResult, dict[int, int]]:
    """构造资源容量、故障域和节点状态可控的小型场景。"""

    node_fault_domains = {
        node_id: fault_domain
        for node_id, fault_domain in enumerate(fault_domains)
    }
    topology = LinearRailTopology(
        sites=[
            TracksideSite(
                node=EdgeNode(
                    node_id=node_id,
                    name=f"MEC-{node_id + 1}",
                    node_type=NodeType.TRACKSIDE,
                    cpu_capacity=100.0,
                    memory_capacity_mb=1000.0,
                    reliability=0.99,
                    fault_domain=(
                        node_fault_domains[node_id]
                    ),
                ),
                position_m=float(node_id * 1000),
                coverage_radius_m=1200.0,
            )
            for node_id in range(len(fault_domains))
        ]
    )
    functions = [
        ServerlessFunction(
            function_id=function_id,
            name=f"测试函数{function_id}",
            memory_mb=function_memory_mb,
            cpu_cycles_per_request=cpu_per_request,
            image_size_mb=10.0,
            warm_exec_time_ms=10.0,
            cold_start_time_ms=100.0,
            output_ratio=1.0,
        )
        for function_id in (0, 1)
    ]
    sfc = SFCType(
        sfc_id=0,
        name="修复测试SFC",
        function_ids=[0, 1],
        deadline_ms=500.0,
        reliability_target=0.90,
        priority=ServicePriority.CRITICAL,
    )
    reliability_model = FaultDomainReliabilityModel(
        topology=topology,
        fault_domain_availability={
            fault_domain: 0.999
            for fault_domain in set(fault_domains)
        },
        minimum_distinct_fault_domains=(
            minimum_distinct_fault_domains
        ),
    )
    auditor = SlotConstraintAuditor(
        functions=functions,
        sfc=sfc,
        topology=topology,
        reliability_model=reliability_model,
    )
    state = FastTimescaleState(
        time_slot=0,
        serving_mec=0,
        remaining_dwell_time_s=10.0,
        request_count=request_count,
        function_ids=(0, 1),
        candidate_node_ids=candidate_map,
        operational_node_ids=frozenset(
            set(range(len(fault_domains)))
            if operational_node_ids is None
            else operational_node_ids
        ),
    )
    (
        _,
        retention_policy,
        cloud_policy,
    ) = legacy_mode_to_policy(standby_mode)
    slow_decision = SlowTimescaleDecision(
        decision_slot=0,
        valid_until_slot=9,
        replica_count=replica_count,
        retention_policy=retention_policy,
        cloud_policy=cloud_policy,
        reason="测试慢动作",
    )
    return (
        optimize_plan(
            auditor=auditor,
            topology=topology,
            state=state,
            slow_decision=slow_decision,
        ),
        node_fault_domains,
    )


def build_overload_result() -> FastOptimizationResult:
    """创建一个需要把两个函数分散到不同节点的结果。"""

    result, _ = optimize_plan_for_test(
        replica_count=1,
        standby_mode=StandbyMode.SINGLE,
        candidate_map={0: (0,), 1: (0,)},
        request_count=1,
        minimum_distinct_fault_domains=1,
    )
    return result


def test_feasible_initial_plan_is_returned_without_search() -> None:
    """原方案满足全部硬约束时，不应浪费时间执行搜索。"""

    result, _ = optimize_plan_for_test(
        replica_count=1,
        standby_mode=StandbyMode.SINGLE,
        candidate_map={0: (0,), 1: (1,)},
        request_count=1,
        minimum_distinct_fault_domains=1,
    )

    assert result.attempted is False
    assert result.succeeded is None
    assert result.evaluated_candidate_count == 0
    assert result.function_replica_node_ids == {
        0: (0,),
        1: (1,),
    }


def test_cpu_and_memory_overload_is_repaired_by_spreading_functions() -> None:
    """同节点CPU和内存超限时，应优先只迁移一个函数。"""

    result, _ = optimize_plan_for_test(
        replica_count=1,
        standby_mode=StandbyMode.SINGLE,
        candidate_map={0: (0,), 1: (0,)},
        request_count=1,
        minimum_distinct_fault_domains=1,
    )

    assert result.initial_audit.resource_constraints_met is False
    assert result.succeeded is True
    assert result.final_audit.all_constraints_met is True
    assert result.function_replica_node_ids == {
        0: (0,),
        1: (1,),
    }


def test_same_domain_replicas_are_moved_across_fault_domains() -> None:
    """双副本位于同一故障域时，应迁移为跨故障域部署。"""

    result, node_fault_domains = optimize_plan_for_test(
        replica_count=2,
        standby_mode=StandbyMode.COLD,
        candidate_map={0: (0, 1), 1: (0, 1)},
        request_count=1,
        minimum_distinct_fault_domains=2,
        fault_domains=(0, 0, 1, 1, 2),
        function_memory_mb=100.0,
        cpu_per_request=10.0,
    )

    assert result.initial_audit.reliability_target_met is False
    assert result.succeeded is True
    assert result.final_audit.reliability_target_met is True
    for node_ids in result.function_replica_node_ids.values():
        assert len({
            node_fault_domains[node_id]
            for node_id in node_ids
        }) == 2


def test_unavailable_initial_path_is_repaired_to_operational_nodes() -> None:
    """静态审计可行但执行节点故障时，仍必须触发修复。"""

    result, _ = optimize_plan_for_test(
        replica_count=1,
        standby_mode=StandbyMode.SINGLE,
        candidate_map={0: (0,), 1: (0,)},
        request_count=1,
        minimum_distinct_fault_domains=1,
        operational_node_ids={1, 2, 3, 4},
        function_memory_mb=100.0,
        cpu_per_request=10.0,
    )

    assert result.initial_audit.all_constraints_met is True
    assert result.succeeded is True
    assert result.decision.request_success is True
    assert set(
        result.decision.selected_execution_node_ids
    ) <= {1, 2, 3, 4}


def test_single_replica_cannot_bypass_two_domain_requirement() -> None:
    """快层不得擅自增加副本来绕过慢层模板。"""

    result, _ = optimize_plan_for_test(
        replica_count=1,
        standby_mode=StandbyMode.SINGLE,
        candidate_map={0: (0,), 1: (1,)},
        request_count=1,
        minimum_distinct_fault_domains=2,
    )

    assert result.attempted is True
    assert result.succeeded is False
    assert result.decision.request_success is False
    assert result.decision.selected_execution_node_ids == ()
    assert "副本数量" in result.reason


def test_no_request_remains_none_when_repair_is_impossible() -> None:
    """无请求时即使计划无法修复，也不能伪造一次请求失败。"""

    result, _ = optimize_plan_for_test(
        replica_count=1,
        standby_mode=StandbyMode.SINGLE,
        candidate_map={0: (0,), 1: (1,)},
        request_count=0,
        minimum_distinct_fault_domains=2,
    )

    assert result.succeeded is False
    assert result.decision.request_success is None


def test_too_few_operational_nodes_returns_normal_failure() -> None:
    """正常节点少于副本数时，应返回可解释失败而不是异常。"""

    result, _ = optimize_plan_for_test(
        replica_count=2,
        standby_mode=StandbyMode.HOT,
        candidate_map={0: (0, 1), 1: (0, 1)},
        request_count=1,
        minimum_distinct_fault_domains=2,
        operational_node_ids={0},
        function_memory_mb=100.0,
        cpu_per_request=10.0,
    )

    assert result.succeeded is False
    assert result.evaluated_candidate_count == 0
    assert "正常MEC" in result.reason


def test_every_node_resource_shortage_returns_failure() -> None:
    """所有节点都放不下实例时，应检查候选后拒绝活动请求。"""

    result, _ = optimize_plan_for_test(
        replica_count=1,
        standby_mode=StandbyMode.SINGLE,
        candidate_map={0: (0,), 1: (1,)},
        request_count=1,
        minimum_distinct_fault_domains=1,
        function_memory_mb=1200.0,
        cpu_per_request=10.0,
    )

    assert result.succeeded is False
    assert result.evaluated_candidate_count > 0
    assert result.decision.request_success is False


def test_repair_is_deterministic() -> None:
    """相同输入重复运行必须得到完全相同的修复结果。"""

    first = build_overload_result()
    second = build_overload_result()

    assert (
        first.function_replica_node_ids
        == second.function_replica_node_ids
    )
    assert first.reason == second.reason
    assert (
        first.evaluated_candidate_count
        == second.evaluated_candidate_count
    )
