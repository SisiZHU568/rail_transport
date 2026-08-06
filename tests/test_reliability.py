"""
test_reliability.py

测试故障域感知可靠性模型。
"""

import pytest

from src.config import load_config
from src.entities import (
    ServicePriority,
    SFCType,
)
from src.reliability import (
    build_fault_domain_reliability_model,
)
from src.topology import build_linear_topology


def build_test_model():
    """
    创建可靠性模型。
    """

    config = load_config("configs/debug.yaml")
    topology = build_linear_topology(config)

    return build_fault_domain_reliability_model(
        config=config,
        topology=topology,
    )


def build_test_sfc() -> SFCType:
    """
    创建三函数测试 SFC。
    """

    return SFCType(
        sfc_id=0,
        name="高可靠故障诊断SFC",
        function_ids=[0, 1, 2],
        deadline_ms=1500.0,
        reliability_target=0.99,
        priority=ServicePriority.CRITICAL,
    )


def test_single_replica_availability() -> None:
    """
    MEC-1 位于故障域0。

    有效可用率：

        0.995 × 0.98 = 0.9751
    """

    model = build_test_model()

    result = model.evaluate_function(
        function_id=0,
        replica_node_ids=[0],
    )

    assert result.availability == pytest.approx(
        0.9751
    )

    assert result.distinct_fault_domain_count == 1
    assert result.fault_domain_diversity_met is False


def test_same_domain_replica_availability() -> None:
    """
    MEC-1 和 MEC-2 都属于故障域0。

    可用率：

        0.995 × [1-(1-0.98)^2]
        = 0.994602
    """

    model = build_test_model()

    result = model.evaluate_function(
        function_id=0,
        replica_node_ids=[0, 1],
    )

    assert result.availability == pytest.approx(
        0.994602
    )

    assert result.fault_domain_ids == (0,)
    assert result.fault_domain_diversity_met is False


def test_cross_domain_replicas_are_more_reliable() -> None:
    """
    MEC-1 属于故障域0；
    MEC-3 属于故障域1。

    跨故障域副本应比同故障域副本更可靠。
    """

    model = build_test_model()

    same_domain = model.evaluate_function(
        function_id=0,
        replica_node_ids=[0, 1],
    )

    cross_domain = model.evaluate_function(
        function_id=0,
        replica_node_ids=[0, 2],
    )

    assert cross_domain.availability == pytest.approx(
        0.99937999
    )

    assert (
        cross_domain.availability
        > same_domain.availability
    )

    assert cross_domain.fault_domain_ids == (0, 1)
    assert cross_domain.fault_domain_diversity_met is True


def test_exact_sfc_reliability_handles_shared_nodes() -> None:
    """
    三个函数都只部署在 MEC-1。

    因为三个函数共享同一个节点，
    只要 MEC-1 正常，整条 SFC 就可用。

    精确共享故障可靠性应为：

        0.9751

    而函数阶段独立近似为：

        0.9751^3
        = 0.927144591751
    """

    model = build_test_model()
    sfc = build_test_sfc()

    plan = {
        0: (0,),
        1: (0,),
        2: (0,),
    }

    result = model.evaluate_sfc(
        sfc=sfc,
        function_replica_node_ids=plan,
    )

    assert (
        result.exact_shared_failure_availability
        == pytest.approx(0.9751)
    )

    assert (
        result.approximate_stage_product_availability
        == pytest.approx(0.927144591751)
    )

    assert result.target_met is False


def test_cross_domain_plan_meets_reliability_target() -> None:
    """
    三个函数都部署在 MEC-1 和 MEC-3。

    两个节点位于不同故障域，
    应满足0.99可靠性目标及故障域隔离要求。
    """

    model = build_test_model()
    sfc = build_test_sfc()

    plan = {
        0: (0, 2),
        1: (0, 2),
        2: (0, 2),
    }

    result = model.evaluate_sfc(
        sfc=sfc,
        function_replica_node_ids=plan,
    )

    assert (
        result.exact_shared_failure_availability
        == pytest.approx(0.99937999)
    )

    assert result.fault_domain_diversity_met is True
    assert result.target_met is True