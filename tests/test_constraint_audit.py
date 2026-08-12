"""测试快时隙约束审计器。"""

import pytest

from src.constraint_audit import SlotConstraintAuditor
from src.entities import (
    EdgeNode,
    NodeType,
    ServerlessFunction,
    ServicePriority,
    SFCType,
)
from src.fast_convex_scheduler import FastScheduledBatch
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
        ],
        cloud_node=EdgeNode(
            node_id=3,
            name="中心云",
            node_type=NodeType.CLOUD,
            cpu_capacity=1000.0,
            memory_capacity_mb=5000.0,
            reliability=0.999,
            fault_domain=3,
        ),
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
            3: 0.999,
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


def test_cloud_candidate_is_audited_as_known_compute_node() -> None:
    """中心云候选必须参与容量和可靠性审计，不能被当作未知节点。"""

    audit = build_auditor().audit(
        request_count=1,
        expected_replica_count=1,
        candidate_map={0: (3,), 1: (3,)},
        selected_execution_node_ids=(3, 3),
        request_success=True,
        function_hot_node_ids={
            0: (3,),
            1: (3,),
        },
        cold_activated_pairs=set(),
    )

    assert audit.invalid_replica_node_ids == ()
    assert audit.node_cpu_demand == {3: 120.0}
    assert audit.node_memory_demand_mb == {
        3: 1200.0
    }
    assert audit.resource_constraints_met is True
    assert audit.replica_plan_valid is True
    assert audit.exact_sfc_reliability == (
        pytest.approx(0.998001)
    )
    assert audit.all_constraints_met is True


def test_deployment_audit_allows_redundancy_after_one_replica_fails() -> None:
    """一个副本故障后仍有正常副本时，主备部署仍可进入快层。"""

    audit = build_auditor().audit_deployment(
        expected_replica_count={0: 2, 1: 2},
        candidate_map={0: (0, 1), 1: (1, 2)},
        function_hot_node_ids={0: (0,), 1: (2,)},
        operational_node_ids=frozenset({0, 2, 3}),
    )

    assert audit.all_constraints_met is True


def test_deployment_audit_rejects_function_with_no_operational_replica() -> None:
    """某个 VNF 的所有副本都故障时，完整 SFC 已无法执行。"""

    audit = build_auditor().audit_deployment(
        expected_replica_count={0: 1, 1: 1},
        candidate_map={0: (0,), 1: (1,)},
        function_hot_node_ids={0: (0,), 1: ()},
        operational_node_ids=frozenset({0, 2, 3}),
    )

    assert audit.all_constraints_met is False
    assert any("函数1" in reason and "正常副本" in reason for reason in audit.violation_reasons)


def test_multi_path_audit_accumulates_integer_batch_cpu() -> None:
    """最终审计必须累计所有路径批次，不能只检查第一条路径。"""

    auditor = build_auditor()
    auditor.node_map[0].cpu_capacity = 240.0
    auditor.node_map[1].cpu_capacity = 120.0
    audit = auditor.audit_scheduled_batches(
        request_count=3,
        expected_replica_count={0: 1, 1: 2},
        candidate_map={0: (0,), 1: (0, 1)},
        function_hot_node_ids={0: (), 1: ()},
        scheduled_batches=(
            FastScheduledBatch(1, (0, 0)),
            FastScheduledBatch(2, (0, 1)),
        ),
    )

    assert audit.node_cpu_demand == {0: 240.0, 1: 120.0}
    assert audit.node_memory_demand_mb == {0: 1200.0, 1: 600.0}
    assert audit.cpu_violation_node_ids == ()
    assert audit.memory_violation_node_ids == (0,)


def test_multi_path_audit_requires_request_conservation() -> None:
    """整数批次少分或多分请求都必须被最终门禁拒绝。"""

    audit = build_auditor().audit_scheduled_batches(
        request_count=3,
        expected_replica_count={0: 1, 1: 1},
        candidate_map={0: (0,), 1: (1,)},
        function_hot_node_ids={0: (0,), 1: (1,)},
        scheduled_batches=(FastScheduledBatch(2, (0, 1)),),
    )

    assert audit.all_constraints_met is False
    assert any("请求总数" in reason for reason in audit.violation_reasons)
