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
