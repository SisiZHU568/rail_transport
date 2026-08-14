from pathlib import Path

import numpy as np
import pytest

from src.phase_e_training_entry import (
    TeacherDatasetRecord,
    load_teacher_dataset,
    save_teacher_dataset,
)
from run_phase_e_pretraining import parse_arguments


def test_teacher_dataset_binds_specs_and_keeps_v_and_u(tmp_path: Path) -> None:
    path = tmp_path / "teachers.npz"
    np.savez(
        path,
        observations=np.zeros((3, 5), dtype=np.float32),
        unbounded_actions=np.zeros((3, 4), dtype=np.float32),
        scores=np.full((3, 4), 0.5, dtype=np.float32),
        effective_action_masks=np.ones((3, 4), dtype=np.bool_),
        source_trajectory_ids=np.array(["t0", "t0", "t1"]),
        state_group_ids=np.array(["g0", "g1", "g2"]),
        state_hashes=np.array(["s0", "s1", "s2"]),
        plan_hashes=np.array(["p0", "p1", "p2"]),
        teacher_types=np.array(["COST", "RELIABILITY", "BALANCE"]),
        plan_jsons=np.array(["{}", "{}", "{}"]),
        predicted_metrics=np.zeros((3, 5), dtype=np.float64),
        preflight_codes=np.array(["OK", "OK", "OK"]),
        dataset_schema_version=np.array("phase-e-teacher-v1"),
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
        effective_action_masks=np.ones((1, 2), dtype=np.bool_),
        source_trajectory_ids=np.array(["t0"]),
        state_group_ids=np.array(["g0"]),
        state_hashes=np.array(["s0"]),
        plan_hashes=np.array(["p0"]),
        teacher_types=np.array(["COST"]),
        plan_jsons=np.array(["{}"]),
        predicted_metrics=np.zeros((1, 5), dtype=np.float64),
        preflight_codes=np.array(["OK"]),
        dataset_schema_version=np.array("phase-e-teacher-v1"),
        observation_spec_hash=np.array("old"),
        action_spec_hash=np.array("act"),
    )
    with pytest.raises(ValueError, match="CHECKPOINT_SPEC_MISMATCH"):
        load_teacher_dataset(path, expected_observation_hash="new", expected_action_hash="act", state_dim=2, action_dim=2)


def test_teacher_dataset_writer_preserves_audit_fields(tmp_path: Path) -> None:
    path = tmp_path / "teachers.npz"
    record = TeacherDatasetRecord(
        source_trajectory_id="train-0",
        state_group_id="train-0:0",
        state_hash="state-sha",
        plan_hash="plan-sha",
        observation=np.array([0.1, 0.2], dtype=np.float32),
        unbounded_action=np.zeros(4, dtype=np.float32),
        scores=np.full(4, 0.5, dtype=np.float32),
        effective_action_mask=np.array([True, True, True, False]),
        teacher_types=("COST", "BALANCE"),
        plan_json='{"instance_counts":[]}',
        predicted_metrics=(1.0, 2.0, 0.1, 0.3, 0.02),
        preflight_code="OK",
    )

    save_teacher_dataset(
        path,
        (record,),
        observation_spec_hash="obs",
        action_spec_hash="act",
    )
    loaded = load_teacher_dataset(
        path,
        expected_observation_hash="obs",
        expected_action_hash="act",
        state_dim=2,
        action_dim=4,
    )

    assert loaded.source_trajectory_ids.tolist() == ["train-0"]
    assert loaded.teacher_types.tolist() == ["COST|BALANCE"]
    assert loaded.effective_action_masks.tolist() == [[True, True, True, False]]
    assert loaded.predicted_metrics.tolist() == [[1.0, 2.0, 0.1, 0.3, 0.02]]


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
