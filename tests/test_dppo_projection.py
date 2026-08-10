"""测试 DPPO 连续动作到可执行 SFC 部署意图之间的确定性投影。"""

import numpy as np
import pytest

from src.dppo_action_space import (
    DPPOActionSpace,
    DecodedDPPOAction,
    DecodedFunctionAction,
)
from src.dppo_intent_adapter import DPPOIntentAdapter
from src.dppo_projection import DPPOProjector, ProjectionResourceDemand
from src.scenario_dimensions import ScenarioDimensions
from src.sfc_deployment_intent import FunctionDeploymentIntent


def _decoded_action(
    function_ids: tuple[int, ...],
    ranking: tuple[int, ...],
    *,
    replica_count: int = 2,
) -> DecodedDPPOAction:
    """构造小型、可读的解码动作，避免测试依赖固定的动作向量切片。"""

    return DecodedDPPOAction(
        function_actions=tuple(
            DecodedFunctionAction(
                function_id=function_id,
                ranked_node_ids=ranking,
                replica_count=replica_count,
                primary_retention_seconds=6.0,
                backup_retention_seconds=3.0,
            )
            for function_id in function_ids
        )
    )


def _uniform_resources(
    node_ids: tuple[int, ...],
    *,
    cpu: float = 100.0,
    memory_mb: float = 4096.0,
) -> tuple[dict[int, float], dict[int, float]]:
    """为每个节点生成相同的可用资源，突出当前测试真正关注的约束。"""

    return (
        {node_id: cpu for node_id in node_ids},
        {node_id: memory_mb for node_id in node_ids},
    )


def test_projection_skips_failed_nodes_and_spreads_fault_domains() -> None:
    """最高分节点故障时应顺延，并优先把副本放到不同故障域。"""

    dimensions = ScenarioDimensions(tuple(range(5)), 5, (0,))
    action_space = DPPOActionSpace(dimensions, maximum_retention_seconds=20.0)
    raw_action = np.zeros(dimensions.action_dim, dtype=np.float32)
    raw_action[:6] = np.asarray([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
    free_cpu, free_memory_mb = _uniform_resources(dimensions.compute_node_ids)
    projector = DPPOProjector(
        dimensions=dimensions,
        resource_demands={
            0: ProjectionResourceDemand(cpu=1.0, memory_mb=64.0),
        },
        minimum_distinct_fault_domains=2,
    )

    result = projector.project(
        decoded_action=action_space.decode(raw_action),
        operational_node_ids=frozenset({1, 2, 3, 4, 5}),
        free_cpu=free_cpu,
        free_memory_mb=free_memory_mb,
        fault_domains={0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 3},
    )

    assert result.success is True
    assert result.raw_feasible is False
    assert result.function_intents is not None
    intent = result.function_intents[0]
    assert 0 not in intent.preferred_node_ids
    assert len(intent.preferred_node_ids) == intent.replica_count
    assert len(
        {result.fault_domains[node_id] for node_id in intent.preferred_node_ids}
    ) >= 2
    assert result.changed_assignment_count > 0
    assert result.change_ratio == pytest.approx(
        result.changed_assignment_count / result.requested_assignment_count
    )


def test_raw_feasible_action_is_not_changed() -> None:
    """原动作满足必要条件时，投影不应无故改变演员网络给出的顺序。"""

    dimensions = ScenarioDimensions((0, 1, 2), 3, (10,))
    projector = DPPOProjector(
        dimensions=dimensions,
        resource_demands={10: ProjectionResourceDemand(1.0, 64.0)},
        minimum_distinct_fault_domains=2,
    )
    free_cpu, free_memory_mb = _uniform_resources(dimensions.compute_node_ids)

    result = projector.project(
        decoded_action=_decoded_action((10,), (0, 1, 2, 3)),
        operational_node_ids=frozenset(dimensions.compute_node_ids),
        free_cpu=free_cpu,
        free_memory_mb=free_memory_mb,
        fault_domains={0: 0, 1: 1, 2: 2, 3: 3},
    )

    assert result.success is True
    assert result.raw_feasible is True
    assert result.function_intents is not None
    assert result.function_intents[0].preferred_node_ids == (0, 1)
    assert result.changed_assignment_count == 0
    assert result.requested_assignment_count == 2
    assert result.change_ratio == pytest.approx(0.0)


def test_projection_uses_remaining_capacity_across_functions() -> None:
    """多个 VNF 共用节点时必须累计扣减容量，不能让每个 VNF 都看到初始余量。"""

    dimensions = ScenarioDimensions((0, 1, 2), 3, (10, 20))
    projector = DPPOProjector(
        dimensions=dimensions,
        resource_demands={
            10: ProjectionResourceDemand(1.0, 64.0),
            20: ProjectionResourceDemand(1.0, 64.0),
        },
        minimum_distinct_fault_domains=1,
    )

    result = projector.project(
        decoded_action=_decoded_action((10, 20), (0, 1, 2, 3)),
        operational_node_ids=frozenset(dimensions.compute_node_ids),
        free_cpu={0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},
        free_memory_mb={node_id: 64.0 for node_id in dimensions.compute_node_ids},
        fault_domains={0: 0, 1: 1, 2: 2, 3: 3},
    )

    assert result.success is True
    assert result.function_intents is not None
    first, second = result.function_intents
    assert first.preferred_node_ids == (0, 1)
    assert second.preferred_node_ids == (2, 3)


def test_projection_reports_machine_readable_failure_reasons() -> None:
    """必要条件无法满足时不生成半成品意图，并给出可统计的失败代码。"""

    dimensions = ScenarioDimensions((0, 1, 2), 3, (0,))
    decoded = _decoded_action((0,), (0, 1, 2, 3), replica_count=3)
    projector = DPPOProjector(
        dimensions=dimensions,
        resource_demands={0: ProjectionResourceDemand(1.0, 64.0)},
        minimum_distinct_fault_domains=2,
    )
    free_cpu, free_memory_mb = _uniform_resources(dimensions.compute_node_ids)

    too_few_nodes = projector.project(
        decoded_action=decoded,
        operational_node_ids=frozenset({0, 1}),
        free_cpu=free_cpu,
        free_memory_mb=free_memory_mb,
        fault_domains={0: 0, 1: 1, 2: 2, 3: 3},
    )
    one_fault_domain = projector.project(
        decoded_action=decoded,
        operational_node_ids=frozenset(dimensions.compute_node_ids),
        free_cpu=free_cpu,
        free_memory_mb=free_memory_mb,
        fault_domains={node_id: 0 for node_id in dimensions.compute_node_ids},
    )

    assert too_few_nodes.success is False
    assert too_few_nodes.function_intents is None
    assert too_few_nodes.reasons == (
        "insufficient_operational_nodes:function_id=0",
    )
    assert too_few_nodes.requested_assignment_count == 3
    assert one_fault_domain.success is False
    assert one_fault_domain.function_intents is None
    assert one_fault_domain.reasons == (
        "insufficient_fault_domains:function_id=0",
    )


def test_projection_distinguishes_cpu_and_memory_shortage() -> None:
    """CPU 与内存不足分开编码，便于后续实验统计真正的拒绝原因。"""

    dimensions = ScenarioDimensions((0, 1, 2), 3, (0,))
    decoded = _decoded_action((0,), (0, 1, 2, 3))
    projector = DPPOProjector(
        dimensions=dimensions,
        resource_demands={0: ProjectionResourceDemand(2.0, 128.0)},
        minimum_distinct_fault_domains=1,
    )
    node_ids = dimensions.compute_node_ids

    cpu_failure = projector.project(
        decoded_action=decoded,
        operational_node_ids=frozenset(node_ids),
        free_cpu={node_id: 1.0 for node_id in node_ids},
        free_memory_mb={node_id: 4096.0 for node_id in node_ids},
        fault_domains={node_id: node_id for node_id in node_ids},
    )
    memory_failure = projector.project(
        decoded_action=decoded,
        operational_node_ids=frozenset(node_ids),
        free_cpu={node_id: 100.0 for node_id in node_ids},
        free_memory_mb={node_id: 64.0 for node_id in node_ids},
        fault_domains={node_id: node_id for node_id in node_ids},
    )

    assert cpu_failure.reasons == ("insufficient_cpu:function_id=0",)
    assert memory_failure.reasons == ("insufficient_memory:function_id=0",)


def test_intent_adapter_attaches_window_metadata_only_after_success() -> None:
    """慢尺度有效期由适配器统一添加，通用意图类型不绑定 DPPO。"""

    dimensions = ScenarioDimensions((0, 1, 2), 3, (0,))
    projector = DPPOProjector(
        dimensions=dimensions,
        resource_demands={0: ProjectionResourceDemand(1.0, 64.0)},
        minimum_distinct_fault_domains=2,
    )
    free_cpu, free_memory_mb = _uniform_resources(dimensions.compute_node_ids)
    projection = projector.project(
        decoded_action=_decoded_action((0,), (0, 1, 2, 3)),
        operational_node_ids=frozenset(dimensions.compute_node_ids),
        free_cpu=free_cpu,
        free_memory_mb=free_memory_mb,
        fault_domains={node_id: node_id for node_id in dimensions.compute_node_ids},
    )

    intent = DPPOIntentAdapter().to_intent(
        projection,
        decision_slot=4,
        valid_until_slot=9,
    )

    assert intent.decision_slot == 4
    assert intent.valid_until_slot == 9
    assert intent.source_algorithm == "dppo"
    assert intent.function_intents == projection.function_intents

    failed_projection = projector.project(
        decoded_action=_decoded_action((0,), (0, 1, 2, 3)),
        operational_node_ids=frozenset({0}),
        free_cpu=free_cpu,
        free_memory_mb=free_memory_mb,
        fault_domains={node_id: node_id for node_id in dimensions.compute_node_ids},
    )
    with pytest.raises(ValueError, match="failed projection"):
        DPPOIntentAdapter().to_intent(
            failed_projection,
            decision_slot=4,
            valid_until_slot=9,
        )


def test_function_intent_rejects_duplicate_or_mismatched_nodes() -> None:
    """执行意图自身也保持最小不变量，防止错误数据绕过投影器进入快层。"""

    with pytest.raises(ValueError, match="unique"):
        FunctionDeploymentIntent(0, (1, 1), 2, 1.0, 1.0)
    with pytest.raises(ValueError, match="replica_count"):
        FunctionDeploymentIntent(0, (1, 2), 3, 1.0, 1.0)
