"""测试只读取当前公开观测的确定性 DPPO 仿真教师。"""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from src.config import load_config
from src.dppo_scenario import build_dppo_scenario
from src.dppo_teacher import build_simulation_teacher


@pytest.mark.parametrize(
    ("teacher_name", "expected_replicas"),
    [("cost", 2), ("reliability", 3), ("balanced", 3)],
)
def test_teacher_is_deterministic_and_uses_declared_replica_policy(
    teacher_name: str,
    expected_replicas: int,
) -> None:
    """同一公开快照必须生成完全相同且语义明确的教师动作。"""

    scenario = build_dppo_scenario(load_config("configs/debug.yaml"))
    scenario.reset(seed=123)
    public_snapshot = scenario.current_public_snapshot()
    teacher = build_simulation_teacher(teacher_name, scenario)

    first = teacher.propose(public_snapshot)
    second = teacher.propose(public_snapshot)

    np.testing.assert_array_equal(first.relaxed_action, second.relaxed_action)
    assert first.teacher_name == teacher_name
    assert first.decoded_action == second.decoded_action
    assert all(
        item.replica_count == expected_replicas
        for item in first.decoded_action.function_actions
    )


def test_cost_teacher_prefers_near_operational_edge_nodes() -> None:
    """成本教师应先排除故障节点，再优先选择低时延轨旁节点。"""

    scenario = build_dppo_scenario(load_config("configs/debug.yaml"))
    scenario.reset(seed=123)
    snapshot = scenario.current_public_snapshot()
    observations = tuple(
        replace(item, operational=False)
        if item.node_id == snapshot.serving_mec
        else item
        for item in snapshot.node_observations
    )
    snapshot_with_failed_serving_node = replace(
        snapshot,
        node_observations=observations,
    )

    proposal = build_simulation_teacher("cost", scenario).propose(
        snapshot_with_failed_serving_node
    )
    first_action = proposal.decoded_action.function_actions[0]

    assert first_action.ranked_node_ids[0] != snapshot.serving_mec
    assert first_action.ranked_node_ids[0] == snapshot.next_mec
    assert first_action.ranked_node_ids[-1] == snapshot.serving_mec


@pytest.mark.parametrize("teacher_name", ["reliability", "balanced"])
def test_reliable_teachers_put_distinct_fault_domains_in_replica_prefix(
    teacher_name: str,
) -> None:
    """三副本教师的前三名必须来自三个不同故障域。"""

    scenario = build_dppo_scenario(load_config("configs/debug.yaml"))
    scenario.reset(seed=123)
    proposal = build_simulation_teacher(teacher_name, scenario).propose(
        scenario.current_public_snapshot()
    )
    fault_domain_by_node = {
        node.node_id: node.fault_domain
        for node in scenario.execution_core.topology.compute_nodes
    }

    for action in proposal.decoded_action.function_actions:
        selected_domains = {
            fault_domain_by_node[node_id]
            for node_id in action.ranked_node_ids[: action.replica_count]
        }
        assert len(selected_domains) == action.replica_count


def test_balanced_teacher_uses_edge_cost_order_after_reliability() -> None:
    """平衡教师先满足跨域约束，再用时延和云成本排列可选节点。"""

    scenario = build_dppo_scenario(load_config("configs/debug.yaml"))
    scenario.reset(seed=123)
    proposal = build_simulation_teacher("balanced", scenario).propose(
        scenario.current_public_snapshot()
    )
    first_ranking = proposal.decoded_action.function_actions[0].ranked_node_ids
    cloud_id = scenario.execution_core.topology.cloud_node.node_id

    # 默认场景有三个轨旁故障域，因而无需为了跨域而提前使用高成本中心云。
    assert cloud_id not in first_ranking[:3]
    assert first_ranking[:3] == (0, 2, 4)


def test_hidden_future_requests_do_not_change_teacher_action() -> None:
    """教师不能通过场景对象读取当前决策点之后的真实请求。"""

    first_config = load_config("configs/debug.yaml")
    second_config = deepcopy(first_config)
    second_config["integrated_simulation"]["request_trace"][-1] += 100
    first_scenario = build_dppo_scenario(first_config)
    second_scenario = build_dppo_scenario(second_config)
    first_scenario.reset(seed=123)
    second_scenario.reset(seed=123)

    first_teacher = build_simulation_teacher("balanced", first_scenario)
    second_teacher = build_simulation_teacher("balanced", second_scenario)
    first_action = first_teacher.propose(
        first_scenario.current_public_snapshot()
    ).relaxed_action
    second_action = second_teacher.propose(
        second_scenario.current_public_snapshot()
    ).relaxed_action

    np.testing.assert_array_equal(first_action, second_action)


def test_unknown_teacher_name_is_rejected() -> None:
    """配置拼写错误必须立即报错，不能静默改用另一种教师。"""

    scenario = build_dppo_scenario(load_config("configs/debug.yaml"))

    with pytest.raises(ValueError, match="未知的仿真教师"):
        build_simulation_teacher("unknown", scenario)


def test_teachers_derive_replica_count_from_configured_bounds() -> None:
    """三类教师从同一配置区间派生低、中、高副本策略。"""

    config = load_config("configs/debug.yaml")
    config["dppo"]["action"]["minimum_replicas"] = 2
    config["dppo"]["action"]["maximum_replicas"] = 5
    scenario = build_dppo_scenario(config)
    scenario.reset(seed=123)
    snapshot = scenario.current_public_snapshot()
    expected_counts = {"cost": 2, "balanced": 3, "reliability": 5}

    for teacher_name, expected_count in expected_counts.items():
        proposal = build_simulation_teacher(teacher_name, scenario).propose(snapshot)
        assert {
            item.replica_count
            for item in proposal.decoded_action.function_actions
        } == {expected_count}
