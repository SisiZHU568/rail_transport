"""测试无界扩散动作适配和状态/动作规格绑定。"""

import numpy as np
import pytest

from src.config import load_config
from src.orchestration_config import load_phase_a_config
from src.phase_d_policy_specs import (
    NormalizationState,
    ObservationFeature,
    ObservationSpec,
    PolicyAdapter,
    PolicySpecification,
)
from src.safe_deployment_decoder import ActionSpec


def test_policy_adapter_uses_sigmoid_without_clipping() -> None:
    adapter = PolicyAdapter(action_dim=3)
    scores = adapter.to_scores(np.array([-100.0, 0.0, 100.0]))

    assert 0.0 <= scores[0] < 1e-20
    assert scores[1] == pytest.approx(0.5)
    assert 1.0 - 1e-12 <= scores[2] <= 1.0
    with pytest.raises(ValueError, match="INVALID_POLICY_SCORE"):
        adapter.to_scores(np.array([0.0, np.nan, 1.0]))


def test_observation_spec_hash_excludes_running_statistics() -> None:
    observation = ObservationSpec(
        features=(
            ObservationFeature("episode_progress", 1, "fixed_scale", 1.0),
            ObservationFeature("node_health", 6, "none", 1.0),
        ),
        ordered_function_ids=(0, 1, 2),
        ordered_node_ids=(0, 1, 2, 3, 4, 5),
        schema_version="phase-d-observation-v1",
    )
    first = PolicySpecification(
        observation,
        ActionSpec.from_phase_a_config(load_phase_a_config(load_config("configs/debug.yaml"))),
        NormalizationState(1, (0.0,) * 7, (1.0,) * 7, 1),
    )
    second = PolicySpecification(
        observation,
        first.action_spec,
        NormalizationState(1, (2.0,) * 7, (3.0,) * 7, 99),
    )

    assert first.observation_spec_hash == second.observation_spec_hash
    assert first.action_spec_hash == second.action_spec_hash
    assert first.normalization_state != second.normalization_state


def test_policy_spec_rejects_normalization_dimension_mismatch() -> None:
    observation = ObservationSpec(
        (ObservationFeature("x", 2, "none", 1.0),), (0,), (0,), "v1"
    )
    action = ActionSpec.from_phase_a_config(load_phase_a_config(load_config("configs/debug.yaml")))

    with pytest.raises(ValueError, match="normalization"):
        PolicySpecification(observation, action, NormalizationState(1, (0.0,), (1.0,), 1))
