"""测试 DPPO 固定部署下的 CLARABEL 快层请求调度。"""

import cvxpy as cp

from src.entities import (
    EdgeNode,
    NodeType,
    ServerlessFunction,
    ServicePriority,
    SFCType,
)
from src.fast_convex_scheduler import FastConvexScheduler
from src.network import LinearMECNetwork
from src.topology import LinearRailTopology, TracksideSite
from src.two_timescale_control import FastTimescaleState


def _build_scheduler(*, node_zero_cpu: float = 100.0) -> FastConvexScheduler:
    """构造只有两级 VNF 的小型真实网络，方便核对路径分流。"""

    nodes = [
        EdgeNode(
            node_id=node_id,
            name=f"MEC-{node_id}",
            node_type=NodeType.TRACKSIDE,
            cpu_capacity=node_zero_cpu if node_id == 0 else 100.0,
            memory_capacity_mb=1000.0,
            reliability=0.999,
            fault_domain=node_id,
        )
        for node_id in (0, 1)
    ]
    topology = LinearRailTopology(
        sites=[
            TracksideSite(node=node, position_m=node.node_id * 1000.0, coverage_radius_m=1200.0)
            for node in nodes
        ]
    )
    functions = [
        ServerlessFunction(
            function_id=function_id,
            name=f"VNF-{function_id}",
            memory_mb=100.0,
            cpu_cycles_per_request=1.0,
            image_size_mb=10.0,
            warm_exec_time_ms=1.0,
            cold_start_time_ms=10.0,
            output_ratio=1.0,
        )
        for function_id in (0, 1)
    ]
    sfc = SFCType(
        sfc_id=0,
        name="测试链",
        function_ids=[0, 1],
        deadline_ms=1000.0,
        reliability_target=0.9,
        priority=ServicePriority.CRITICAL,
    )
    network = LinearMECNetwork(
        node_ids=[0, 1],
        adjacent_bandwidth_mbps=1000.0,
        propagation_delay_per_hop_ms=1.0,
        edge_data_cost_per_mb_hop=1.0,
    )
    return FastConvexScheduler(
        topology=topology,
        network=network,
        functions=functions,
        sfc=sfc,
        input_size_mb_per_request=1.0,
        slot_seconds=1.0,
        edge_cpu_cost_per_unit=1.0,
        edge_memory_cost_per_mb_second=0.0,
        cloud_cpu_cost_per_unit=2.0,
        cloud_memory_cost_per_mb_second=0.0,
        cold_start_cost_per_ms=1.0,
        solver_name="CLARABEL",
        max_iterations=200,
        feasibility_tolerance=1.0e-7,
    )


def _state(
    *,
    request_count: int,
    second_function_nodes: tuple[int, ...] = (0, 1),
    operational_nodes: frozenset[int] = frozenset({0, 1}),
) -> FastTimescaleState:
    return FastTimescaleState(
        time_slot=0,
        serving_mec=0,
        remaining_dwell_time_s=10.0,
        request_count=request_count,
        function_ids=(0, 1),
        candidate_node_ids={0: (0,), 1: second_function_nodes},
        operational_node_ids=operational_nodes,
    )


def test_no_requests_skip_clarabel(monkeypatch) -> None:
    """空时隙不需要建立和求解数学模型。"""

    def forbidden_solve(*args: object, **kwargs: object) -> None:
        raise AssertionError("空时隙不应调用求解器")

    monkeypatch.setattr(cp.Problem, "solve", forbidden_solve)
    result = _build_scheduler().schedule(
        state=_state(request_count=0),
        function_hot_node_ids={0: (0,), 1: (0, 1)},
    )

    assert result.succeeded is True
    assert result.solver_status == "not_run"
    assert result.scheduled_batches == ()


def test_low_load_uses_the_lowest_cost_complete_path() -> None:
    """资源充足时，全部请求应选择网络成本最低的完整路径。"""

    result = _build_scheduler().schedule(
        state=_state(request_count=4),
        function_hot_node_ids={0: (0,), 1: (0, 1)},
    )

    assert result.succeeded is True
    assert result.solver_status == "optimal"
    assert [(batch.request_count, batch.execution_node_ids) for batch in result.scheduled_batches] == [
        (4, (0, 0))
    ]


def test_cpu_capacity_splits_requests_and_preserves_total_count() -> None:
    """最低成本路径容量不足时，CLARABEL 应把请求分到第二条路径。"""

    result = _build_scheduler(node_zero_cpu=6.0).schedule(
        state=_state(request_count=4),
        function_hot_node_ids={0: (0,), 1: (0, 1)},
    )

    assert result.succeeded is True
    allocations = {
        batch.execution_node_ids: batch.request_count
        for batch in result.scheduled_batches
    }
    assert allocations == {(0, 0): 2, (0, 1): 2}
    assert sum(allocations.values()) == 4


def test_failed_node_is_never_used() -> None:
    """快层只能在当前正常且已部署的副本之间调度。"""

    result = _build_scheduler().schedule(
        state=_state(
            request_count=3,
            second_function_nodes=(1, 0),
            operational_nodes=frozenset({0}),
        ),
        function_hot_node_ids={0: (0,), 1: (0, 1)},
    )

    assert result.succeeded is True
    assert result.scheduled_batches[0].execution_node_ids == (0, 0)


def test_non_optimal_solver_status_is_rejected_without_fallback(monkeypatch) -> None:
    """CLARABEL 未给出严格 optimal 时直接拒绝，不切换其他算法。"""

    calls: list[object] = []

    def report_inaccurate(problem: cp.Problem, *args: object, **kwargs: object) -> None:
        calls.append(kwargs.get("solver"))
        problem._status = cp.OPTIMAL_INACCURATE

    monkeypatch.setattr(cp.Problem, "solve", report_inaccurate)
    result = _build_scheduler().schedule(
        state=_state(request_count=2),
        function_hot_node_ids={0: (0,), 1: (0, 1)},
    )

    assert result.succeeded is False
    assert result.solver_status == "optimal_inaccurate"
    assert calls == ["CLARABEL"]
    assert result.scheduled_batches == ()
