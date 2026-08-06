"""测试快时隙约束审计器。"""

from src.constraint_audit import SlotConstraintAuditor
from src.entities import (
    EdgeNode,
    NodeType,
    ServerlessFunction,
    ServicePriority,
    SFCType,
)
from src.reliability import FaultDomainReliabilityModel
from src.topology import LinearRailTopology, TracksideSite


def build_auditor() -> SlotConstraintAuditor:
    """创建一个资源容量和故障域都容易核对的审计器。"""

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
                    fault_domain=node_id,
                ),
                position_m=float(node_id * 1000),
                coverage_radius_m=1200.0,
            )
            for node_id in range(3)
        ]
    )
    functions = [
        ServerlessFunction(
            function_id=function_id,
            name=f"测试函数{function_id}",
            memory_mb=600.0,
            cpu_cycles_per_request=60.0,
            image_size_mb=10.0,
            warm_exec_time_ms=10.0,
            cold_start_time_ms=100.0,
            output_ratio=1.0,
        )
        for function_id in (0, 1)
    ]
    sfc = SFCType(
        sfc_id=0,
        name="审计测试SFC",
        function_ids=[0, 1],
        deadline_ms=500.0,
        reliability_target=0.90,
        priority=ServicePriority.CRITICAL,
    )
    reliability_model = FaultDomainReliabilityModel(
        topology=topology,
        fault_domain_availability={
            0: 0.999,
            1: 0.999,
            2: 0.999,
        },
        minimum_distinct_fault_domains=1,
    )
    return SlotConstraintAuditor(
        functions=functions,
        sfc=sfc,
        topology=topology,
        reliability_model=reliability_model,
    )


def test_expected_replica_count_is_a_hard_constraint() -> None:
    """实际副本数与慢层要求不一致时，计划必须判为无效。"""

    audit = build_auditor().audit(
        request_count=1,
        expected_replica_count=2,
        candidate_map={0: (0,), 1: (1,)},
        selected_execution_node_ids=(0, 1),
        request_success=True,
        function_hot_node_ids={0: (0,), 1: (1,)},
        cold_activated_pairs=set(),
    )

    assert audit.replica_count_violation_function_ids == (0, 1)
    assert audit.replica_plan_valid is False
    assert audit.all_constraints_met is False
    assert any(
        "副本数量" in reason
        for reason in audit.violation_reasons
    )


def test_same_node_resource_demand_is_accumulated() -> None:
    """两个函数位于同一节点时，CPU和内存需求必须累加。"""

    auditor = build_auditor()
    audit = auditor.audit(
        request_count=1,
        expected_replica_count=1,
        candidate_map={0: (0,), 1: (0,)},
        selected_execution_node_ids=(0, 0),
        request_success=True,
        function_hot_node_ids={0: (0,), 1: (0,)},
        cold_activated_pairs=set(),
    )

    assert audit.node_cpu_demand == {0: 120.0}
    assert audit.node_memory_demand_mb == {0: 1200.0}
    assert audit.cpu_violation_node_ids == (0,)
    assert audit.memory_violation_node_ids == (0,)
    assert auditor.resource_constraints_met(
        request_count=1,
        selected_execution_node_ids=(0, 0),
        request_success=True,
        function_hot_node_ids={0: (0,), 1: (0,)},
        cold_activated_pairs=set(),
    ) is False


def test_capacity_equal_to_demand_is_still_feasible() -> None:
    """需求等于容量不是超限，只有严格大于容量才算违规。"""

    auditor = build_auditor()
    auditor.node_map[0].cpu_capacity = 120.0
    auditor.node_map[0].memory_capacity_mb = 1200.0

    audit = auditor.audit(
        request_count=1,
        expected_replica_count=1,
        candidate_map={0: (0,), 1: (0,)},
        selected_execution_node_ids=(0, 0),
        request_success=True,
        function_hot_node_ids={0: (0,), 1: (0,)},
        cold_activated_pairs=set(),
    )

    assert audit.cpu_violation_node_ids == ()
    assert audit.memory_violation_node_ids == ()
    assert audit.resource_constraints_met is True


def test_missing_function_is_reported_as_invalid_plan() -> None:
    """SFC中的任一函数没有部署时，不能继续计算可靠性。"""

    audit = build_auditor().audit(
        request_count=1,
        expected_replica_count=1,
        candidate_map={0: (0,)},
        selected_execution_node_ids=(0,),
        request_success=True,
        function_hot_node_ids={0: (0,)},
        cold_activated_pairs=set(),
    )

    assert audit.missing_function_ids == (1,)
    assert audit.replica_plan_valid is False
    assert audit.exact_sfc_reliability is None
