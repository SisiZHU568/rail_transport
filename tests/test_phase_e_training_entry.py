from pathlib import Path

import numpy as np
import pytest

from src.phase_e_training_entry import load_teacher_dataset
from run_phase_e_pretraining import parse_arguments


def test_teacher_dataset_binds_specs_and_keeps_v_and_u(tmp_path: Path) -> None:
    path = tmp_path / "teachers.npz"
    np.savez(
        path,
        observations=np.zeros((3, 5), dtype=np.float32),
        unbounded_actions=np.zeros((3, 4), dtype=np.float32),
        scores=np.full((3, 4), 0.5, dtype=np.float32),
        observation_spec_hash=np.array("obs"),
        action_spec_hash=np.array("act"),
    )

    dataset = load_teacher_dataset(
        path, expected_observation_hash="obs", expected_action_hash="act",
        state_dim=5, action_dim=4,
    )

    assert dataset.observations.shape == (3, 5)
    assert np.allclose(1.0 / (1.0 + np.exp(-dataset.unbounded_actions)), dataset.scores)


def test_teacher_dataset_rejects_spec_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    np.savez(
        path,
        observations=np.zeros((1, 2), dtype=np.float32),
        unbounded_actions=np.zeros((1, 2), dtype=np.float32),
        scores=np.full((1, 2), 0.5, dtype=np.float32),
        observation_spec_hash=np.array("old"),
        action_spec_hash=np.array("act"),
    )
    with pytest.raises(ValueError, match="CHECKPOINT_SPEC_MISMATCH"):
        load_teacher_dataset(path, expected_observation_hash="new", expected_action_hash="act", state_dim=2, action_dim=2)


def test_pretraining_cli_requires_dataset_and_output() -> None:
    with pytest.raises(SystemExit):
        parse_arguments([])
    args = parse_arguments([
        "--dataset", "teachers.npz", "--output", "model.pt",
        "--observation-spec-hash", "obs", "--action-spec-hash", "act",
        "--steps", "3",
    ])
    assert args.dataset == "teachers.npz"
    assert args.output == "model.pt"
    assert args.steps == 3
