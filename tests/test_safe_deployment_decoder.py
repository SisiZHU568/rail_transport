"""测试配置驱动动作规格与确定性安全部署解码。"""

import numpy as np

from src.config import load_config
from src.orchestration_config import load_phase_a_config
from src.safe_deployment_decoder import (
    ActionSpec,
    DecoderInput,
    SafeDeploymentDecoder,
)


def test_action_spec_uses_two_scores_per_allowed_pair() -> None:
    config = load_phase_a_config(load_config("configs/debug.yaml"))
    spec = ActionSpec.from_phase_a_config(config)

    assert spec.pairs[0] == (0, 0)
    assert spec.pairs[-1] == (2, 5)
    assert spec.action_dim == 36
    assert len(spec.sha256) == 64
    assert spec.entries[0].action_type == "instance_count_score"
    assert spec.entries[1].action_type == "retention_time_score"


def test_decoder_selects_only_globally_completable_counts() -> None:
    config = load_phase_a_config(load_config("configs/debug.yaml"))
    spec = ActionSpec.from_phase_a_config(config)
    decoder = SafeDeploymentDecoder(config, spec, function_memory_mb={0: 256, 1: 512, 2: 768})
    scores = np.zeros(spec.action_dim, dtype=np.float64)
    # 所有实例评分倾向最大值；内存约束会迫使解码器保留后续 VNF 的空间。
    scores[0::2] = 1.0
    result = decoder.decode(
        scores,
        DecoderInput(
            effective_node_up={node_id: True for node_id in config.node_resources},
            locked_instance_counts={},
            required_replica_nodes={0: 1, 1: 1, 2: 1},
        ),
    )

    assert result.code == "OK"
    assert result.plan is not None
    assert set(result.plan.instance_counts) == set(spec.pairs)
    assert all(
        sum(
            max(result.plan.instance_counts[(function_id, node_id)], 0)
            * {0: 256, 1: 512, 2: 768}[function_id]
            for function_id in (0, 1, 2)
        ) <= config.node_resources[node_id].memory_capacity_mb
        for node_id in config.node_resources
    )


def test_decoder_reports_no_safe_deployment_without_projection() -> None:
    config = load_phase_a_config(load_config("configs/debug.yaml"))
    spec = ActionSpec.from_phase_a_config(config)
    result = SafeDeploymentDecoder(
        config, spec, function_memory_mb={0: 256, 1: 512, 2: 768}
    ).decode(
        np.full(spec.action_dim, 0.5),
        DecoderInput(
            effective_node_up={node_id: False for node_id in config.node_resources},
            locked_instance_counts={},
            required_replica_nodes={0: 1, 1: 1, 2: 1},
        ),
    )

    assert result.code == "NO_SAFE_FEASIBLE_DEPLOYMENT"
    assert result.plan is None
    assert result.used_fallback is False


def test_zero_target_marks_only_retention_dimension_ineffective() -> None:
    config = load_phase_a_config(load_config("configs/debug.yaml"))
    spec = ActionSpec.from_phase_a_config(config)
    scores = np.zeros(spec.action_dim)
    scores[0::2] = 0.0
    result = SafeDeploymentDecoder(
        config, spec, function_memory_mb={0: 256, 1: 512, 2: 768}
    ).decode(
        scores,
        DecoderInput(
            effective_node_up={node_id: True for node_id in config.node_resources},
            locked_instance_counts={},
            required_replica_nodes={0: 1, 1: 1, 2: 1},
        ),
    )

    assert all(result.effective_action_mask[0::2])
    for pair_index, pair in enumerate(spec.pairs):
        if result.plan.instance_counts[pair] == 0:
            assert result.effective_action_mask[2 * pair_index + 1] is False


def test_decoder_enforces_fault_domain_dispersion() -> None:
    config = load_phase_a_config(load_config("configs/debug.yaml"))
    spec = ActionSpec.from_phase_a_config(config)
    result = SafeDeploymentDecoder(
        config, spec, function_memory_mb={0: 256, 1: 512, 2: 768}
    ).decode(
        np.zeros(spec.action_dim),
        DecoderInput(
            effective_node_up={node_id: True for node_id in config.node_resources},
            locked_instance_counts={},
            required_replica_nodes={0: 2, 1: 1, 2: 1},
            fault_domain_by_node={0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 3},
            minimum_fault_domains={0: 2},
        ),
    )

    assert result.code == "OK"
    selected_domains = {
        {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 3}[node_id]
        for (function_id, node_id), count in result.plan.instance_counts.items()
        if function_id == 0 and count > 0
    }
    assert len(selected_domains) >= 2


def test_decoder_uses_union_bound_unavailability_budget() -> None:
    config = load_phase_a_config(load_config("configs/debug.yaml"))
    spec = ActionSpec.from_phase_a_config(config)
    result = SafeDeploymentDecoder(
        config, spec, function_memory_mb={0: 256, 1: 512, 2: 768}
    ).decode(
        np.zeros(spec.action_dim),
        DecoderInput(
            effective_node_up={node_id: True for node_id in config.node_resources},
            locked_instance_counts={},
            required_replica_nodes={0: 1, 1: 1, 2: 1},
            fault_domain_by_node={node_id: node_id for node_id in config.node_resources},
            domain_availability={node_id: 0.999 for node_id in config.node_resources},
            node_conditional_availability={node_id: 0.99 for node_id in config.node_resources},
            maximum_vnf_unavailability={0: 0.005},
        ),
    )

    assert result.code == "OK"
    assert sum(
        count > 0
        for (function_id, _), count in result.plan.instance_counts.items()
        if function_id == 0
    ) >= 2


def test_teacher_plan_can_be_encoded_and_decoded_without_change() -> None:
    config = load_phase_a_config(load_config("configs/debug.yaml"))
    spec = ActionSpec.from_phase_a_config(config)
    decoder = SafeDeploymentDecoder(
        config, spec, function_memory_mb={0: 256, 1: 512, 2: 768}
    )
    decoder_input = DecoderInput(
        effective_node_up={node_id: True for node_id in config.node_resources},
        locked_instance_counts={},
        required_replica_nodes={0: 1, 1: 1, 2: 1},
    )
    original = decoder.decode(np.full(spec.action_dim, 0.35), decoder_input)

    encoded = decoder.encode_plan(original.plan, decoder_input)
    decoded = decoder.decode(encoded.scores, decoder_input)

    assert encoded.code == "OK"
    assert decoded.code == "OK"
    assert decoded.plan == original.plan
    assert np.all((encoded.scores > 0.0) & (encoded.scores < 1.0))


def test_teacher_enumeration_discards_partial_results_when_limit_is_hit() -> None:
    config = load_phase_a_config(load_config("configs/debug.yaml"))
    spec = ActionSpec.from_phase_a_config(config)
    decoder = SafeDeploymentDecoder(
        config, spec, function_memory_mb={0: 256, 1: 512, 2: 768}
    )
    result = decoder.enumerate_safe_plans(
        DecoderInput(
            effective_node_up={node_id: True for node_id in config.node_resources},
            locked_instance_counts={},
            required_replica_nodes={0: 1, 1: 1, 2: 1},
        ),
        max_generated_patterns=1,
        retention_slots_by_pair={pair: spec.retention_slot_options[0] for pair in spec.pairs},
    )

    assert result.code == "DECODER_SEARCH_LIMIT"
    assert result.plans == ()
    assert result.generated_patterns == 1


def test_teacher_enumeration_reports_proven_empty_safe_set() -> None:
    config = load_phase_a_config(load_config("configs/debug.yaml"))
    spec = ActionSpec.from_phase_a_config(config)
    decoder = SafeDeploymentDecoder(
        config, spec, function_memory_mb={0: 256, 1: 512, 2: 768}
    )
    result = decoder.enumerate_safe_plans(
        DecoderInput(
            effective_node_up={node_id: False for node_id in config.node_resources},
            locked_instance_counts={},
            required_replica_nodes={0: 1, 1: 1, 2: 1},
        ),
        max_generated_patterns=10,
        retention_slots_by_pair={},
    )

    assert result.code == "NO_SAFE_FEASIBLE_DEPLOYMENT"
    assert result.plans == ()
