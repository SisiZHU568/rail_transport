"""
test_entities.py

测试基础实体类是否按照预期工作。

pytest 会自动寻找：
1. 名称以 test_ 开头的文件；
2. 名称以 test_ 开头的函数。
"""

import pytest

from src.entities import (
    EdgeNode,
    NodeType,
    ServerlessFunction,
    ServicePriority,
    SFCType,
    SlotConstraintAudit,
)


def test_onboard_is_not_a_compute_node_type() -> None:
    """首版函数部署节点只包含轨旁 MEC 和中心云。"""

    assert {item.value for item in NodeType} == {"trackside", "cloud"}


def test_create_edge_node() -> None:
    """
    测试能否创建一个合法的轨旁 MEC。
    """

    node = EdgeNode(
        node_id=0,
        name="MEC-1",
        node_type=NodeType.TRACKSIDE,
        cpu_capacity=100.0,
        memory_capacity_mb=8192.0,
        reliability=0.98,
        fault_domain=0,
    )

    assert node.node_id == 0
    assert node.name == "MEC-1"
    assert node.node_type == NodeType.TRACKSIDE
    assert node.reliability == 0.98


def test_invalid_node_reliability() -> None:
    """
    节点可靠性大于1时，程序应主动抛出 ValueError。
    """

    with pytest.raises(ValueError):
        EdgeNode(
            node_id=0,
            name="错误节点",
            node_type=NodeType.TRACKSIDE,
            cpu_capacity=100.0,
            memory_capacity_mb=8192.0,
            reliability=1.2,
            fault_domain=0,
        )


def test_create_serverless_function() -> None:
    """
    测试能否创建合法的 Serverless 函数。
    """

    function = ServerlessFunction(
        function_id=0,
        name="数据清洗",
        memory_mb=256.0,
        cpu_cycles_per_request=20.0,
        image_size_mb=80.0,
        warm_exec_time_ms=15.0,
        cold_start_time_ms=300.0,
        output_ratio=0.7,
    )

    assert function.function_id == 0
    assert function.name == "数据清洗"
    assert function.cold_start_time_ms > function.warm_exec_time_ms


def test_create_sfc() -> None:
    """
    测试函数链的执行顺序是否可以正确保存。
    """

    sfc = SFCType(
        sfc_id=0,
        name="故障诊断",
        function_ids=[0, 1, 2],
        deadline_ms=1500.0,
        reliability_target=0.99,
        priority=ServicePriority.CRITICAL,
    )

    assert sfc.function_ids == [0, 1, 2]
    assert sfc.priority == ServicePriority.CRITICAL


def test_empty_sfc_is_invalid() -> None:
    """
    不包含任何函数的 SFC 应被判定为非法。
    """

    with pytest.raises(ValueError):
        SFCType(
            sfc_id=0,
            name="空SFC",
            function_ids=[],
            deadline_ms=1500.0,
            reliability_target=0.99,
            priority=ServicePriority.NORMAL,
        )


def build_capacity_test_node() -> EdgeNode:
    """创建容量较小、便于验证资源边界的测试节点。"""

    return EdgeNode(
        node_id=0,
        name="容量测试节点",
        node_type=NodeType.TRACKSIDE,
        cpu_capacity=100.0,
        memory_capacity_mb=512.0,
        reliability=0.98,
        fault_domain=0,
    )


def test_node_accepts_demand_equal_to_capacity() -> None:
    """资源需求恰好等于容量时，节点仍然可以承载负载。"""

    node = build_capacity_test_node()

    assert node.has_sufficient_capacity(
        cpu_demand=100.0,
        memory_demand_mb=512.0,
    ) is True


@pytest.mark.parametrize(
    ("cpu_demand", "memory_demand_mb"),
    [
        (100.1, 512.0),
        (100.0, 512.1),
    ],
)
def test_node_rejects_capacity_overload(
    cpu_demand: float,
    memory_demand_mb: float,
) -> None:
    """CPU 或内存任一超限时，节点都不能承载该负载。"""

    node = build_capacity_test_node()

    assert node.has_sufficient_capacity(
        cpu_demand=cpu_demand,
        memory_demand_mb=memory_demand_mb,
    ) is False


@pytest.mark.parametrize(
    ("cpu_demand", "memory_demand_mb"),
    [
        (-0.1, 0.0),
        (0.0, -0.1),
    ],
)
def test_node_rejects_negative_resource_demand(
    cpu_demand: float,
    memory_demand_mb: float,
) -> None:
    """负数资源需求没有物理意义，应被明确拒绝。"""

    node = build_capacity_test_node()

    with pytest.raises(ValueError):
        node.has_sufficient_capacity(
            cpu_demand=cpu_demand,
            memory_demand_mb=memory_demand_mb,
        )


def build_cpu_test_function() -> ServerlessFunction:
    """创建每个请求需要 20 个抽象 CPU 单位的测试函数。"""

    return ServerlessFunction(
        function_id=0,
        name="CPU测试函数",
        memory_mb=128.0,
        cpu_cycles_per_request=20.0,
        image_size_mb=50.0,
        warm_exec_time_ms=10.0,
        cold_start_time_ms=100.0,
        output_ratio=0.5,
    )


def test_function_calculates_batch_cpu_demand() -> None:
    """批量 CPU 需求应等于单请求需求乘以请求数量。"""

    function = build_cpu_test_function()

    assert function.cpu_demand(
        request_count=3
    ) == pytest.approx(60.0)

    assert function.cpu_demand(
        request_count=0
    ) == 0.0


def test_function_rejects_negative_request_count() -> None:
    """负数请求数量没有实际意义，应被明确拒绝。"""

    function = build_cpu_test_function()

    with pytest.raises(ValueError):
        function.cpu_demand(request_count=-1)


def test_constraint_audit_preserves_diagnostic_details() -> None:
    """审计结果应完整保存资源和可靠性诊断信息。"""

    audit = SlotConstraintAudit(
        node_cpu_demand={0: 120.0},
        node_memory_demand_mb={0: 600.0},
        cpu_violation_node_ids=(0,),
        memory_violation_node_ids=(0,),
        invalid_replica_node_ids=(),
        missing_function_ids=(),
        exact_sfc_reliability=0.98,
        reliability_target=0.99,
        reliability_target_met=False,
        resource_constraints_met=False,
        replica_plan_valid=True,
        all_constraints_met=False,
        violation_reasons=(
            "节点0的CPU需求120.000超过容量100.000。",
        ),
        replica_count_violation_function_ids=(0,),
    )

    assert audit.cpu_violation_node_ids == (0,)
    assert audit.replica_count_violation_function_ids == (0,)
    assert audit.exact_sfc_reliability == pytest.approx(
        0.98
    )
    assert audit.all_constraints_met is False
