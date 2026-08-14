import pytest

from src.phase_e_artifacts import (
    CheckpointSpecification,
    TrajectoryManifest,
    validate_manifest_isolation,
    validate_checkpoint_specification,
)


def test_trajectory_manifests_must_be_pairwise_isolated() -> None:
    manifests = (
        TrajectoryManifest("train", (("train-0", 100),)),
        TrajectoryManifest("calibration", (("cal-0", 200),)),
        TrajectoryManifest("validation", (("val-0", 300),)),
        TrajectoryManifest("test", (("test-0", 400),)),
    )
    validate_manifest_isolation(manifests)
    assert all(len(item.sha256) == 64 for item in manifests)

    with pytest.raises(ValueError, match="轨迹清单不隔离"):
        validate_manifest_isolation((manifests[0], TrajectoryManifest("test", (("x", 100),))))


def test_checkpoint_rejects_observation_or_action_spec_mismatch() -> None:
    saved = CheckpointSpecification("obs-a", "act-a", "model-a", 3)
    validate_checkpoint_specification(saved, saved)
    with pytest.raises(ValueError, match="CHECKPOINT_SPEC_MISMATCH"):
        validate_checkpoint_specification(saved, CheckpointSpecification("obs-b", "act-a", "model-a", 3))
