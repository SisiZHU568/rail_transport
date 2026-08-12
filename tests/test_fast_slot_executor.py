"""测试训练环境和规则仿真器共用的快时隙执行闭环。"""

from dataclasses import replace

import pytest

from src.constraint_audit import SlotConstraintAuditor
from src.entities import (
    EdgeNode,
    NodeType,
    ServerlessFunction,
    ServicePriority,
    SFCType,
    TrainState,
)
from src.failure_process import InfrastructureState
from src.fast_optimizer import FastFeasibilityOptimizer
from src.fast_convex_scheduler import FastConvexScheduler
from src.fast_slot_executor import (
    FastSlotExecutor,
    FastSlotInput,
    RuntimeCostRates,
)
from src.network import LinearMECNetwork
from src.redundancy_placement import (
    ReliabilityAwareReplicaPlanner,
)
from src.reliability import FaultDomainReliabilityModel
from src.deployment_policies import (
    CloudPolicy,
    RetentionPolicy,
)
from src.runtime_reliability import SingleReplicaPlanner
from src.topology import LinearRailTopology, TracksideSite
from src.two_timescale_control import SlowTimescaleDecision


def build_executor(
    *,
    unrepairable: bool,
    function_ids: tuple[int, ...] = (0, 1),
    node_count: int = 3,
) -> FastSlotExecutor:
    """构造两函数、三轨旁节点的最小可核对场景。"""

    topology = LinearRailTopology(
        sites=[
            TracksideSite(
                node=EdgeNode(
                    node_id=node_id,
                    name=f"MEC-{node_id + 1}",
                    node_type=NodeType.TRACKSIDE,
                    cpu_capacity=1000.0,
                    memory_capacity_mb=1000.0,
                    reliability=0.99,
                    fault_domain=node_id,
                ),
                position_m=float(node_id * 1000),
                coverage_radius_m=1200.0,
            )
            for node_id in range(node_count)
        ]
    )
    functions = [
        ServerlessFunction(
            function_id=function_id,
            name=f"测试函数{function_id}",
            # 两函数采用双副本时一共需要四个实例。
            # 每个实例600 MB且只有三个节点，因此必有节点超限，
            # 用于构造快层也无法修复的确定性场景。
            memory_mb=600.0 if unrepairable else 100.0,
            cpu_cycles_per_request=10.0,
            image_size_mb=10.0,
            warm_exec_time_ms=10.0,
            cold_start_time_ms=100.0,
            output_ratio=1.0,
        )
        for function_id in function_ids
    ]
    sfc = SFCType(
        sfc_id=0,
        name="共享执行器测试SFC",
        function_ids=list(function_ids),
        deadline_ms=500.0,
        reliability_target=0.90,
        priority=ServicePriority.CRITICAL,
    )
    auditor = SlotConstraintAuditor(
        functions=functions,
        sfc=sfc,
        topology=topology,
        reliability_model=FaultDomainReliabilityModel(
            topology=topology,
            fault_domain_availability={
                node_id: 0.999
                for node_id in range(node_count)
            },
            minimum_distinct_fault_domains=2,
        ),
    )
    network = LinearMECNetwork(
        node_ids=list(range(node_count)),
        adjacent_bandwidth_mbps=100.0,
        propagation_delay_per_hop_ms=1.0,
        edge_data_cost_per_mb_hop=0.01,
    )
    cost_rates = RuntimeCostRates(
        edge_cpu_cost_per_unit=0.01,
        edge_memory_cost_per_mb_second=0.001,
        cloud_cpu_cost_per_unit=0.05,
        cloud_memory_cost_per_mb_second=0.005,
        cold_start_cost_per_ms=0.10,
    )
    optimizer = FastFeasibilityOptimizer(
        functions=functions,
        sfc=sfc,
        topology=topology,
        auditor=auditor,
        network=network,
        edge_cpu_cost_per_unit=(
            cost_rates.edge_cpu_cost_per_unit
        ),
        edge_memory_cost_per_mb_second=(
            cost_rates.edge_memory_cost_per_mb_second
        ),
        cloud_cpu_cost_per_unit=(
            cost_rates.cloud_cpu_cost_per_unit
        ),
        cloud_memory_cost_per_mb_second=(
            cost_rates.cloud_memory_cost_per_mb_second
        ),
        cold_start_cost_per_ms=(
            cost_rates.cold_start_cost_per_ms
        ),
        input_size_mb_per_request=2.0,
        slot_seconds=1.0,
    )
    fast_convex_scheduler = FastConvexScheduler(
        topology=topology,
        network=network,
        functions=functions,
        sfc=sfc,
        input_size_mb_per_request=2.0,
        slot_seconds=1.0,
        edge_cpu_cost_per_unit=cost_rates.edge_cpu_cost_per_unit,
        edge_memory_cost_per_mb_second=cost_rates.edge_memory_cost_per_mb_second,
        cloud_cpu_cost_per_unit=cost_rates.cloud_cpu_cost_per_unit,
        cloud_memory_cost_per_mb_second=cost_rates.cloud_memory_cost_per_mb_second,
        cold_start_cost_per_ms=cost_rates.cold_start_cost_per_ms,
        solver_name="CLARABEL",
        max_iterations=200,
        feasibility_tolerance=1.0e-7,
    )

    return FastSlotExecutor(
        topology=topology,
        network=network,
        functions=functions,
        sfc=sfc,
        replica_planners={
            1: SingleReplicaPlanner(),
            2: ReliabilityAwareReplicaPlanner(
                replica_count=2,
                minimum_distinct_fault_domains=2,
            ),
            3: ReliabilityAwareReplicaPlanner(
                replica_count=3,
                minimum_distinct_fault_domains=2,
            ),
        },
        constraint_auditor=auditor,
        fast_optimizer=optimizer,
        fast_convex_scheduler=fast_convex_scheduler,
        input_size_mb_per_request=2.0,
        slot_seconds=1.0,
        handover_hot_window_s=2.0,
        failover_delay_ms_per_function=5.0,
        cost_rates=cost_rates,
    )


def build_slot_input(*, node_count: int = 3) -> FastSlotInput:
    """生成所有轨旁节点都正常、包含一个请求的快时隙输入。"""

    return FastSlotInput(
        train_state=TrainState(
            time_slot=0,
            position_m=0.0,
            speed_mps=20.0,
            serving_mec=0,
            next_mec=1,
            remaining_dwell_time_s=10.0,
        ),
        request_count=1,
        infrastructure_state=InfrastructureState(
            time_slot=0,
            domain_up={node_id: True for node_id in range(node_count)},
            node_local_up={node_id: True for node_id in range(node_count)},
        ),
        slow_decision=SlowTimescaleDecision(
            decision_slot=0,
            valid_until_slot=9,
            replica_count=2,
            retention_policy=RetentionPolicy.ALL_WARM,
            cloud_policy=CloudPolicy.EDGE_ONLY,
            reason="测试慢动作",
        ),
        previous_candidate_map=None,
    )


def test_infeasible_final_audit_never_executes_sfc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """硬约束不可修复时必须正常拒绝，不能执行部分SFC。"""

    calls = 0

    def forbidden_execute(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("不可行方案不应进入SFC执行器")

    monkeypatch.setattr(
        "src.fast_slot_executor.execute_sfc_batch",
        forbidden_execute,
    )

    result = build_executor(unrepairable=True).execute(
        build_slot_input()
    )

    assert calls == 0
    assert result.final_audit.all_constraints_met is False
    assert result.request_success is False
    assert result.constraint_rejected is True


def test_shared_executor_returns_three_cost_components() -> None:
    """一个可行时隙必须返回奖励函数需要的三项原始成本。"""

    slot_input = build_slot_input()
    # 改为按需启动，让测试可以直接核对冷启动成本是否来自真实执行。
    slot_input = replace(
        slot_input,
        slow_decision=replace(
            slot_input.slow_decision,
            retention_policy=RetentionPolicy.ON_DEMAND,
        ),
    )
    result = build_executor(unrepairable=False).execute(
        slot_input
    )

    assert result.final_audit.all_constraints_met is True
    # CPU：2个函数 × 10单位 × 0.01；
    # 内存：2个活动实例 × 100 MB × 1秒 × 0.001。
    assert result.run_cost == pytest.approx(0.4)
    assert result.route_cost == pytest.approx(0.0)
    # 两个函数各冷启动100 ms，单价为0.10/毫秒。
    assert result.cold_start_delay_ms == pytest.approx(200.0)
    assert result.cold_start_cost == pytest.approx(20.0)
    assert result.cold_start_function_ids == (0, 1)
    assert result.failover_function_ids == ()
    assert result.transmission_delay_ms == pytest.approx(0.0)
    assert result.execution_delay_ms == pytest.approx(20.0)
    assert result.active_instance_count == 2
    assert result.active_memory_mb == pytest.approx(200.0)
    assert result.total_cost == pytest.approx(
        result.run_cost
        + result.route_cost
        + result.cold_start_cost
    )
