"""测试共享快层执行器直接消费逐 VNF 部署意图。"""

from dataclasses import replace

from src.sfc_deployment_intent import (
    FunctionDeploymentIntent,
    SFCDeploymentIntent,
)
from tests.test_fast_slot_executor import build_executor, build_slot_input


def test_algorithm_independent_intent_can_represent_single_replica_baseline() -> None:
    """共享意图允许规则基线用 1 副本，但 DPPO 解码器仍只生成 2/3 副本。"""

    intent = FunctionDeploymentIntent(0, (1,), 1, 2.0, 0.0)
    assert intent.replica_count == 1
    assert intent.preferred_node_ids == (1,)


def _mixed_intent() -> SFCDeploymentIntent:
    """构造副本数为 (2, 3, 2) 且主备保留时间不同的完整 SFC 意图。"""

    return SFCDeploymentIntent(
        decision_slot=0,
        valid_until_slot=4,
        function_intents=(
            FunctionDeploymentIntent(0, (0, 1), 2, 4.0, 1.0),
            FunctionDeploymentIntent(1, (1, 2, 3), 3, 6.0, 2.0),
            FunctionDeploymentIntent(2, (2, 3), 2, 8.0, 3.0),
        ),
        source_algorithm="test",
    )


def test_executor_uses_per_function_replica_counts() -> None:
    """快层必须保留每个 VNF 自己的副本数，不能重新套用一个全局数量。"""

    executor = build_executor(
        unrepairable=False,
        function_ids=(0, 1, 2),
        node_count=5,
    )
    slot_input = replace(
        build_slot_input(node_count=5),
        slow_decision=None,
        deployment_intent=_mixed_intent(),
    )

    result = executor.execute(slot_input)

    assert result.initial_candidate_map == {
        0: (0, 1),
        1: (1, 2, 3),
        2: (2, 3),
    }
    assert tuple(
        len(result.initial_candidate_map[function_id])
        for function_id in (0, 1, 2)
    ) == (2, 3, 2)
    assert result.final_audit is not None
    assert result.final_audit.all_constraints_met is True
    assert result.retained_hot_node_ids_by_function[0] == frozenset({0, 1})


def test_continuous_retention_expires_without_reapplying_same_intent() -> None:
    """同一慢层意图每个快时隙都会被读取，但保留期只能在决策时隙刷新一次。"""

    executor = build_executor(
        unrepairable=False,
        function_ids=(0, 1, 2),
        node_count=5,
    )
    first_input = replace(
        build_slot_input(node_count=5),
        slow_decision=None,
        deployment_intent=_mixed_intent(),
    )
    executor.execute(first_input)

    slot_two_state = replace(first_input.train_state, time_slot=2)
    slot_two_infrastructure = replace(
        first_input.infrastructure_state,
        time_slot=2,
    )
    result = executor.execute(
        replace(
            first_input,
            train_state=slot_two_state,
            infrastructure_state=slot_two_infrastructure,
        )
    )

    # VNF 0 的备用节点只保留 1 秒，在时隙 2 已到期；主节点仍然温热。
    assert result.retained_hot_node_ids_by_function[0] == frozenset({0})


def test_executor_reset_clears_retention_between_episodes() -> None:
    """仿真器开始新 Episode 时必须清除上一条轨迹留下的温热实例。"""

    executor = build_executor(
        unrepairable=False,
        function_ids=(0, 1, 2),
        node_count=5,
    )
    first_input = replace(
        build_slot_input(node_count=5),
        slow_decision=None,
        deployment_intent=_mixed_intent(),
    )
    executor.execute(first_input)

    executor.reset()

    slot_one_input = replace(
        first_input,
        train_state=replace(first_input.train_state, time_slot=1),
        infrastructure_state=replace(first_input.infrastructure_state, time_slot=1),
    )
    result = executor.execute(slot_one_input)
    assert result.retained_hot_node_ids_by_function == {
        0: frozenset(),
        1: frozenset(),
        2: frozenset(),
    }
