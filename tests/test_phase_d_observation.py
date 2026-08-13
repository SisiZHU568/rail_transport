from src.config import load_config
from src.orchestration_config import load_phase_a_config
from src.phase_d_observation import (
    ObservationBundleVersions,
    ObservationContext,
    build_observation_snapshot,
)
from src.safe_deployment_decoder import ActionSpec


def test_observation_is_config_driven_and_carries_all_source_versions() -> None:
    config = load_phase_a_config(load_config("configs/debug.yaml"))
    action_spec = ActionSpec.from_phase_a_config(config)
    context = ObservationContext(
        current_slot=5,
        total_episode_slots=100,
        maximum_drain_slots=20,
        is_drain_phase=False,
        serving_mec=0,
        next_mec=1,
        node_health={node_id: node_id != 2 for node_id in config.node_resources},
        node_free_memory_ratio={node_id: 0.5 for node_id in config.node_resources},
        warm_counts={pair: 1 for pair in action_spec.pairs},
        starting_counts={pair: 0 for pair in action_spec.pairs},
        locked_counts={pair: 1 for pair in action_spec.pairs},
        retention_remaining_ratios={pair: (0.2, 0.4, 0.3) for pair in action_spec.pairs},
        queue_load_ratios={-1: 0.1, 0: 0.2, 1: 0.3, 2: 0.4},
        minimum_slack_ratios={-1: 0.8, 0: 0.7, 1: 0.6, 2: 0.5},
        structural_reliability=0.999,
        warm_reliability=0.995,
        previous_service_deficit=0.1,
        previous_cost_ratio=0.2,
        previous_sla_violation=0.0,
    )
    versions = ObservationBundleVersions(3, 4, 5, 6, 7)

    snapshot = build_observation_snapshot(config, action_spec, context, versions)

    assert snapshot.versions == versions
    assert len(snapshot.values) == snapshot.spec.dimension
    assert snapshot.spec.ordered_function_ids == (0, 1, 2)
    assert snapshot.spec.ordered_node_ids == (0, 1, 2, 3, 4, 5)
    # 时间进度使用“正常期 + 最大排空期”作为固定分母。
    assert snapshot.values[0] == 5 / 120
    assert snapshot.values[2] == 0.0
