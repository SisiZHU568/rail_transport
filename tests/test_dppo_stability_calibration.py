"""测试 DPPO 稳定性短校准命令的公平性、可复现性与失败产物。"""

from __future__ import annotations

import csv
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_dppo_stability_calibration as calibration
from src.config import load_config
from src.dppo_stability import load_stability_profile


def _calibration_config() -> dict:
    config = load_config("configs/debug.yaml")
    config["dppo"]["stability"]["calibration_iterations"] = 2
    config["dppo"]["stability"]["calibration_episodes_per_iteration"] = 2
    return config


def _install_fast_boundaries(monkeypatch, *, fail_candidate: float | None = None):
    """只替换昂贵环境/轨迹边界；资格、选择和持久化仍走真实代码。"""

    seeds_by_candidate: dict[float, list[int]] = {}
    configs_by_candidate: dict[float, object] = {}
    agents: list[object] = []
    loaded_models: list[object] = []
    checkpoint_paths: list[Path] = []

    dimensions = SimpleNamespace(
        state_dim=3,
        action_dim=2,
        # 测试配置允许最多 3 个副本，因此假环境也必须提供至少 3 个计算节点。
        mec_count=2,
        compute_node_count=3,
        function_count=1,
    )

    class FakeEnvironment:
        def __init__(self) -> None:
            self.dimensions = dimensions

    class FakeModel:
        def __init__(self) -> None:
            self.initial_state = {"weight": (1.0, 2.0)}

    class FakeAgent:
        def __init__(self, model, schedule, agent_config, *, device) -> None:
            self.model = model
            self.schedule = schedule
            self.config = agent_config
            self.device = device
            self.policy_optimizer = object()
            self.value_optimizer = object()
            self.initial_state = deepcopy(model.initial_state)
            self.update_count = 0
            agents.append(self)

        def update(self, _buffer):
            self.update_count += 1
            candidate = self.config.clip_ratio
            if fail_candidate == candidate:
                raise OverflowError("synthetic overflow")
            return {
                # 与当前正式校准目标区间 [0.60, 0.85] 保持一致。
                "clip_fraction": 0.70,
                "approximate_kl": candidate + self.update_count / 1000.0,
                "maximum_approximate_kl": candidate + self.update_count / 100.0,
                "optimizer_step_count": 2.0,
            }

    def fake_build_config(config, *, clip_ratio):
        built = SimpleNamespace(
            clip_ratio=clip_ratio,
            diffusion_steps=int(config["dppo"]["diffusion"]["steps"]),
        )
        configs_by_candidate[clip_ratio] = built
        return built

    def fake_load_checkpoint(path, *, expected, device):
        checkpoint_paths.append(Path(path))
        model = FakeModel()
        loaded_models.append(model)
        return SimpleNamespace(model=model, metadata=expected)

    def fake_collect_rollout(_environment, agent, _buffer, seed):
        seeds_by_candidate.setdefault(agent.config.clip_ratio, []).append(seed)
        return {"transition_count": 1.0}

    monkeypatch.setattr(calibration, "build_dppo_environment", lambda _config: FakeEnvironment())
    monkeypatch.setattr(calibration, "build_dppo_agent_config", fake_build_config)
    monkeypatch.setattr(calibration, "load_dppo_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(calibration, "DPPOAgent", FakeAgent)
    monkeypatch.setattr(calibration, "CosineNoiseSchedule", lambda steps: ("schedule", steps))
    monkeypatch.setattr(calibration, "collect_rollout", fake_collect_rollout)
    return SimpleNamespace(
        seeds_by_candidate=seeds_by_candidate,
        configs_by_candidate=configs_by_candidate,
        agents=agents,
        loaded_models=loaded_models,
        checkpoint_paths=checkpoint_paths,
    )


def test_calibration_uses_shared_seeds_candidate_override_and_fresh_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _calibration_config()
    checkpoint = tmp_path / "pretrained.pt"
    checkpoint.write_bytes(b"same checkpoint")
    output = tmp_path / "calibration"
    observed = _install_fast_boundaries(monkeypatch)

    profile = calibration.calibrate_dppo_stability(
        config=config,
        pretrained_checkpoint=checkpoint,
        output_root=output,
        device="cpu",
    )

    candidates = (0.10, 0.01, 0.001)
    expected_seeds = (20000, 20001, 20002, 20003)
    assert tuple(observed.seeds_by_candidate) == candidates
    assert all(
        tuple(observed.seeds_by_candidate[candidate]) == expected_seeds
        for candidate in candidates
    )
    assert tuple(observed.configs_by_candidate) == candidates
    assert all(
        observed.configs_by_candidate[candidate].clip_ratio == candidate
        for candidate in candidates
    )
    assert len({id(agent) for agent in observed.agents}) == len(candidates)
    assert len({id(model) for model in observed.loaded_models}) == len(candidates)
    assert all(agent.initial_state == {"weight": (1.0, 2.0)} for agent in observed.agents)
    assert len({id(agent.policy_optimizer) for agent in observed.agents}) == len(candidates)
    assert len({id(agent.value_optimizer) for agent in observed.agents}) == len(candidates)
    assert observed.checkpoint_paths == [checkpoint, checkpoint, checkpoint]
    assert profile.episode_seeds == expected_seeds
    assert profile.selected_clip_ratio == 0.10


def test_csv_is_descending_utf8_sig_and_matches_profile(tmp_path: Path, monkeypatch) -> None:
    config = _calibration_config()
    checkpoint = tmp_path / "pretrained.pt"
    checkpoint.write_bytes(b"csv checkpoint")
    output = tmp_path / "calibration"
    _install_fast_boundaries(monkeypatch)

    profile = calibration.calibrate_dppo_stability(
        config=config,
        pretrained_checkpoint=checkpoint,
        output_root=output,
        device="cpu",
    )

    csv_path = output / "candidate_metrics.csv"
    assert csv_path.read_bytes().startswith(b"\xef\xbb\xbf")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert [float(row["candidate"]) for row in rows] == [0.10, 0.01, 0.001]
    assert {
        "candidate",
        "mean_clip_fraction",
        "mean_approximate_kl",
        "maximum_approximate_kl",
        "optimizer_step_count",
        "finite",
        "qualified",
        "failure_reasons",
        "execution_error",
    } <= set(rows[0])
    by_candidate = {result.clip_ratio: result for result in profile.candidate_results}
    for row in rows:
        result = by_candidate[float(row["candidate"])]
        assert float(row["mean_clip_fraction"]) == result.mean_clip_fraction
        assert float(row["mean_approximate_kl"]) == result.mean_approximate_kl
        assert float(row["maximum_approximate_kl"]) == result.maximum_approximate_kl
        assert int(row["optimizer_step_count"]) == result.optimizer_step_count
        assert json.loads(row["finite"]) is result.finite
        assert json.loads(row["qualified"]) is result.qualified
    assert load_stability_profile(output / "stability_profile.json") == profile


def test_numerical_error_is_recorded_and_other_candidates_still_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _calibration_config()
    checkpoint = tmp_path / "pretrained.pt"
    checkpoint.write_bytes(b"error checkpoint")
    output = tmp_path / "calibration"
    observed = _install_fast_boundaries(monkeypatch, fail_candidate=0.10)

    profile = calibration.calibrate_dppo_stability(
        config=config,
        pretrained_checkpoint=checkpoint,
        output_root=output,
        device="cpu",
    )

    failed = next(result for result in profile.candidate_results if result.clip_ratio == 0.10)
    assert failed.finite is False
    assert failed.qualified is False
    assert failed.mean_clip_fraction is None
    assert tuple(observed.seeds_by_candidate) == (0.10, 0.01, 0.001)
    with (output / "candidate_metrics.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))
    failed_row = next(row for row in rows if float(row["candidate"]) == 0.10)
    assert "OverflowError: synthetic overflow" == failed_row["execution_error"]
    assert (output / "stability_profile.json").is_file()


def test_keyboard_interrupt_is_not_swallowed(tmp_path: Path, monkeypatch) -> None:
    config = _calibration_config()
    checkpoint = tmp_path / "pretrained.pt"
    checkpoint.write_bytes(b"interrupt checkpoint")
    _install_fast_boundaries(monkeypatch)
    monkeypatch.setattr(
        calibration,
        "collect_rollout",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        calibration.calibrate_dppo_stability(
            config=config,
            pretrained_checkpoint=checkpoint,
            output_root=tmp_path / "output",
            device="cpu",
        )


def test_same_inputs_produce_identical_candidate_results(tmp_path: Path, monkeypatch) -> None:
    config = _calibration_config()
    checkpoint = tmp_path / "pretrained.pt"
    checkpoint.write_bytes(b"reproducible checkpoint")
    _install_fast_boundaries(monkeypatch)

    first = calibration.calibrate_dppo_stability(
        config=deepcopy(config),
        pretrained_checkpoint=checkpoint,
        output_root=tmp_path / "first",
        device="cpu",
    )
    second = calibration.calibrate_dppo_stability(
        config=deepcopy(config),
        pretrained_checkpoint=checkpoint,
        output_root=tmp_path / "second",
        device="cpu",
    )

    assert first.selected_clip_ratio == second.selected_clip_ratio
    assert first.candidate_results == second.candidate_results
    assert first.config_hash == second.config_hash
    assert first.pretrained_checkpoint_sha256 == second.pretrained_checkpoint_sha256


def test_unqualified_main_exits_nonzero_after_both_artifacts_exist(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _calibration_config()
    config["dppo"]["stability"]["target_clip_fraction_min"] = 0.8
    config["dppo"]["stability"]["target_clip_fraction_max"] = 0.9
    checkpoint = tmp_path / "pretrained.pt"
    checkpoint.write_bytes(b"unqualified checkpoint")
    output = tmp_path / "output"
    _install_fast_boundaries(monkeypatch)
    monkeypatch.setattr(calibration, "load_config", lambda _path: config)

    with pytest.raises(SystemExit) as raised:
        calibration.main(
            [
                "--pretrained-checkpoint",
                str(checkpoint),
                "--output-root",
                str(output),
                "--device",
                "cpu",
            ]
        )

    assert raised.value.code != 0
    assert (output / "candidate_metrics.csv").is_file()
    assert (output / "stability_profile.json").is_file()
    assert load_stability_profile(output / "stability_profile.json").qualified is False


def test_cli_defaults_config_and_allows_device_override() -> None:
    parsed = calibration.parse_arguments(
        [
            "--pretrained-checkpoint",
            "input.pt",
            "--output-root",
            "output",
            "--device",
            "cuda",
        ]
    )

    assert Path(parsed.config).as_posix().endswith("configs/debug.yaml")
    assert parsed.pretrained_checkpoint == "input.pt"
    assert parsed.output_root == "output"
    assert parsed.device == "cuda"
