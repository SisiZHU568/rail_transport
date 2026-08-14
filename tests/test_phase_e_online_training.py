import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from run_phase_e_online_training import _config_sha256, main, parse_arguments
from src.config import load_config
from src.dppo_diffusion import ConditionalDiffusionMLP
from src.fast_resource_optimizer import FastResourceOptimizer
from src.phase_e_environment import ArrivalBatch
from src.phase_e_runtime import build_phase_e_runtime
from src.phase_e_training_entry import TeacherDatasetRecord, save_teacher_dataset


def _phase_e_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    raw = load_config("configs/debug.yaml")
    runtime = build_phase_e_runtime(raw, total_slow_frames=1)
    captured = {}
    settings = raw["phase_e_runtime"]

    def policy(context):
        observation = runtime.observation_adapter.encode(context)
        captured["observation"] = observation
        return np.zeros(runtime.controller.action_spec.action_dim)

    result = runtime.environment.run_slow_frame(
        start_slot=0,
        policy=policy,
        arrivals_by_slot={
            0: (
                ArrivalBatch(
                    "artifact-bootstrap",
                    0,
                    float(settings["smoke_arrival_equivalent_bits"]),
                    float(settings["smoke_deadline_seconds"]),
                ),
            )
        },
        network_for_slot=lambda slot: runtime.network_for_slot(slot, frame_index=0),
        training_mode=True,
    )
    assert result.code == "OK"
    observation = captured["observation"]
    action_dim = runtime.controller.action_spec.action_dim
    dataset = tmp_path / "teachers.npz"
    save_teacher_dataset(
        dataset,
        (
            TeacherDatasetRecord(
                source_trajectory_id="bootstrap",
                state_group_id="bootstrap:0",
                state_hash="state",
                plan_hash="plan",
                observation=np.asarray(observation.values, dtype=np.float32),
                unbounded_action=np.zeros(action_dim, dtype=np.float32),
                scores=np.full(action_dim, 0.5, dtype=np.float32),
                effective_action_mask=np.ones(action_dim, dtype=np.bool_),
                teacher_types=("BALANCE",),
                plan_json="{}",
                predicted_metrics=(0.0, 0.0, 0.0, 0.0, 0.0),
                preflight_code="OK",
            ),
        ),
        observation_spec_hash=observation.spec.sha256,
        action_spec_hash=runtime.controller.action_spec.sha256,
    )
    model = ConditionalDiffusionMLP(
        observation.spec.dimension,
        action_dim,
        tuple(raw["dppo"]["diffusion"]["hidden_dims"]),
    )
    checkpoint = tmp_path / "pretrained.pt"
    torch.save(
        {
            "format_version": "phase-e-pretrained-v1",
            "observation_spec_hash": observation.spec.sha256,
            "action_spec_hash": runtime.controller.action_spec.sha256,
            "state_dim": observation.spec.dimension,
            "action_dim": action_dim,
            "model_hidden_dims": model.hidden_dims,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "optimizer_steps": 1,
        },
        checkpoint,
    )
    return checkpoint, dataset


def test_formal_cli_requires_bound_artifacts_and_complete_rollouts() -> None:
    with pytest.raises(SystemExit):
        parse_arguments([])
    with pytest.raises(SystemExit, match="rollout"):
        main(
            [
                "--pretrained-checkpoint", "missing.pt",
                "--teacher-dataset", "missing.npz",
                "--output", "output",
                "--frames", "17",
                "--device", "cpu",
            ]
        )


def test_online_config_fingerprint_binds_cli_training_overrides() -> None:
    raw = load_config("configs/debug.yaml")

    baseline = _config_sha256(raw, clip_ratio=0.01, device="cpu")

    assert baseline != _config_sha256(raw, clip_ratio=0.02, device="cpu")
    assert baseline != _config_sha256(raw, clip_ratio=0.01, device="cuda")


def test_formal_training_runs_two_updates_and_resumes_at_update_boundary(
    tmp_path: Path,
) -> None:
    pretrained, teachers = _phase_e_artifacts(tmp_path)
    output = tmp_path / "online"
    common = [
        "--pretrained-checkpoint", str(pretrained),
        "--teacher-dataset", str(teachers),
        "--output", str(output),
        "--device", "cpu",
    ]

    main([*common, "--frames", "16"])
    main(
        [
            *common,
            "--frames", "32",
            "--resume-checkpoint", str(output / "dppo_phase_e_last.pt"),
        ]
    )

    last = output / "dppo_phase_e_last.pt"
    best = output / "dppo_phase_e_best.pt"
    history = output / "training_history.csv"
    assert last.is_file()
    assert best.is_file()
    assert history.is_file()
    assert not (output / "failure_audit.json").exists()
    with history.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert [int(row["completed_update_count"]) for row in rows] == [1, 2]
    payload = torch.load(last, map_location="cpu", weights_only=False)
    assert payload["metadata"]["next_frame_index"] == 32
    assert payload["metadata"]["completed_update_count"] == 2


def test_formal_training_audits_internal_failure_without_success_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pretrained, teachers = _phase_e_artifacts(tmp_path)
    output = tmp_path / "failed-online"
    monkeypatch.setattr(
        FastResourceOptimizer,
        "solve",
        lambda self, *args, **kwargs: self._failure_result(
            is_dcp=True,
            primary_status="forced_failure",
        ),
    )

    with pytest.raises(SystemExit, match="FAST_SOLVER_FAILURE"):
        main(
            [
                "--pretrained-checkpoint", str(pretrained),
                "--teacher-dataset", str(teachers),
                "--output", str(output),
                "--frames", "16",
                "--device", "cpu",
            ]
        )

    audit = json.loads((output / "failure_audit.json").read_text(encoding="utf-8"))
    assert audit["failure_code"] == "FAST_SOLVER_FAILURE"
    assert audit["completed_update_count"] == 0
    assert not (output / "training_history.csv").exists()
    assert not (output / "dppo_phase_e_last.pt").exists()
