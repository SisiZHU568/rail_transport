import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from src.dppo import DPPOAgent, DPPOConfig, DPPORolloutBuffer, DPPORolloutTransition
from src.dppo_diffusion import ConditionalDiffusionMLP, CosineNoiseSchedule
from src.phase_e_online_checkpoint import (
    PhaseEOnlineMetadata,
    load_phase_e_online_checkpoint,
    save_phase_e_online_checkpoint,
)


def _trained_agent() -> DPPOAgent:
    torch.manual_seed(4)
    agent = DPPOAgent(
        ConditionalDiffusionMLP(4, 3, (8, 8)),
        CosineNoiseSchedule(3),
        DPPOConfig(
            diffusion_steps=3,
            fine_tuned_steps=2,
            value_hidden_dims=(8,),
            batch_size=2,
            update_epochs=1,
            target_kl=10.0,
            seed=91,
        ),
        device="cpu",
    )
    buffer = DPPORolloutBuffer()
    for index in range(2):
        state = np.full(4, index / 10, dtype=np.float32)
        sample = agent.sample_action(state, seed=100 + index)
        chain = sample.actions[:, 0].detach().cpu().numpy()
        buffer.append(
            DPPORolloutTransition(
                state=state,
                raw_action=chain[-1],
                denoising_actions=chain,
                old_log_probabilities=(
                    sample.log_probabilities[:, 0].detach().cpu().numpy()
                ),
                reward=-0.2,
                value=float(agent.value(state)),
                terminated=index == 1,
            )
        )
    agent.update(buffer, next_value=0.0)
    return agent


def _assert_state_dict_equal(left: torch.nn.Module, right: torch.nn.Module) -> None:
    assert left.state_dict().keys() == right.state_dict().keys()
    assert all(
        torch.equal(value, right.state_dict()[name])
        for name, value in left.state_dict().items()
    )


def _assert_numpy_rng_equal(left: tuple, right: tuple) -> None:
    assert left[0] == right[0]
    assert np.array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def test_phase_e_online_checkpoint_restores_complete_training_state(
    tmp_path: Path,
) -> None:
    source = _trained_agent()
    metadata = PhaseEOnlineMetadata(
        format_version="phase-e-online-v1",
        observation_spec_hash="observation",
        action_spec_hash="action",
        config_sha256="config",
        next_frame_index=16,
        completed_update_count=1,
        best_validation_reward=-0.25,
    )
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state().clone()
    path = tmp_path / "online.pt"

    save_phase_e_online_checkpoint(path, source, metadata)
    random.random()
    np.random.random()
    torch.rand(1)
    restored = _trained_agent()
    loaded = load_phase_e_online_checkpoint(
        path,
        restored,
        expected_observation_spec_hash="observation",
        expected_action_spec_hash="action",
        expected_config_sha256="config",
    )

    assert loaded.metadata == metadata
    _assert_state_dict_equal(restored.frozen_policy, source.frozen_policy)
    _assert_state_dict_equal(restored.trainable_policy, source.trainable_policy)
    _assert_state_dict_equal(restored.value_network, source.value_network)
    assert restored.policy_optimizer.state_dict()["state"]
    assert restored.value_optimizer.state_dict()["state"]
    assert random.getstate() == python_rng
    _assert_numpy_rng_equal(np.random.get_state(), numpy_rng)
    assert torch.equal(torch.get_rng_state(), torch_rng)


@pytest.mark.parametrize(
    ("field", "expected"),
    (
        ("observation_spec_hash", "other-observation"),
        ("action_spec_hash", "other-action"),
        ("config_sha256", "other-config"),
    ),
)
def test_phase_e_online_checkpoint_rejects_binding_mismatch_before_mutation(
    tmp_path: Path,
    field: str,
    expected: str,
) -> None:
    source = _trained_agent()
    metadata = PhaseEOnlineMetadata(
        "phase-e-online-v1", "observation", "action", "config", 16, 1, None
    )
    path = tmp_path / "online.pt"
    save_phase_e_online_checkpoint(path, source, metadata)
    restored = _trained_agent()
    before = {
        name: value.clone() for name, value in restored.trainable_policy.state_dict().items()
    }
    arguments = {
        "expected_observation_spec_hash": "observation",
        "expected_action_spec_hash": "action",
        "expected_config_sha256": "config",
    }
    arguments[f"expected_{field}"] = expected

    with pytest.raises(ValueError, match="CHECKPOINT_SPEC_MISMATCH"):
        load_phase_e_online_checkpoint(path, restored, **arguments)

    assert all(
        torch.equal(value, restored.trainable_policy.state_dict()[name])
        for name, value in before.items()
    )


def test_phase_e_online_metadata_rejects_invalid_values() -> None:
    valid = PhaseEOnlineMetadata(
        "phase-e-online-v1", "observation", "action", "config", 0, 0, None
    )
    with pytest.raises(ValueError):
        replace(valid, format_version="dppo-online-checkpoint-v2")
    with pytest.raises(ValueError):
        replace(valid, next_frame_index=-1)
    with pytest.raises(ValueError):
        replace(valid, best_validation_reward=float("nan"))
