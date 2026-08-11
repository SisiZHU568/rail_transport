"""测试 DPPO 稳定性配置、校准判定与可审计配置文件。"""

from __future__ import annotations

from dataclasses import replace
import json
import math
import os

import pytest

from src.config import load_config
from src.dppo_stability import (
    DPPOCalibrationCandidateResult,
    DPPOStabilityProfile,
    STABILITY_PROFILE_SCHEMA_VERSION,
    evaluate_calibration_candidate,
    load_stability_profile,
    parse_stability_profile_json,
    save_stability_profile,
    select_stability_profile,
    sha256_file,
    stability_profile_json,
    stability_profile_sha256,
    validate_stability_profile,
)
from src.dppo_training_config import (
    DPPOStabilitySettings,
    build_dppo_agent_config,
    load_dppo_stability_settings,
)


def _settings() -> DPPOStabilitySettings:
    return DPPOStabilitySettings(
        training_sampling_min_std=0.01,
        probability_min_std=0.10,
        evaluation_sampling_min_std=0.001,
        target_kl=1.0,
        target_clip_fraction_min=0.10,
        target_clip_fraction_max=0.20,
        clip_ratio_candidates=(0.10, 0.01, 0.001),
        calibration_iterations=3,
        calibration_episodes_per_iteration=2,
        calibration_seed_start=20000,
    )


def _result(clip_ratio: float, *, clip_fraction: float = 0.15, kl: float = 0.2):
    return evaluate_calibration_candidate(
        clip_ratio=clip_ratio,
        mean_clip_fraction=clip_fraction,
        mean_approximate_kl=kl,
        maximum_approximate_kl=kl,
        optimizer_step_count=2,
        settings=_settings(),
    )


def _profile(tmp_path) -> tuple[DPPOStabilityProfile, object]:
    checkpoint = tmp_path / "pretrained.pt"
    checkpoint.write_bytes(b"checkpoint")
    profile = select_stability_profile(
        config_hash="debug-config-v1",
        pretrained_checkpoint_sha256=sha256_file(checkpoint),
        settings=_settings(),
        episode_seeds=(20000, 20001, 20002, 20003, 20004, 20005),
        candidate_results=(_result(0.01), _result(0.10), _result(0.001)),
    )
    return profile, checkpoint


def test_load_settings_and_build_agent_config_map_every_algorithm_field() -> None:
    config = load_config("configs/debug.yaml")

    settings = load_dppo_stability_settings(config)
    agent = build_dppo_agent_config(config, clip_ratio=0.01)

    assert settings == _settings()
    assert agent.gamma == config["dppo"]["training"]["gamma"]
    assert agent.gae_lambda == config["dppo"]["training"]["gae_lambda"]
    assert agent.clip_ratio == 0.01
    assert agent.clip_ratio != config["dppo"]["training"]["clip_ratio"]
    assert agent.denoising_discount == config["dppo"]["training"]["denoising_discount"]
    assert agent.policy_learning_rate == config["dppo"]["training"]["policy_learning_rate"]
    assert agent.value_learning_rate == config["dppo"]["training"]["value_learning_rate"]
    assert agent.batch_size == config["dppo"]["training"]["batch_size"]
    assert agent.update_epochs == config["dppo"]["training"]["update_epochs"]
    assert agent.gradient_clip_norm == config["dppo"]["training"]["gradient_clip_norm"]
    assert agent.diffusion_steps == config["dppo"]["diffusion"]["steps"]
    assert agent.fine_tuned_steps == config["dppo"]["diffusion"]["fine_tuned_steps"]
    assert agent.value_hidden_dims == tuple(config["dppo"]["training"]["value_hidden_dims"])
    assert agent.seed == config["dppo"]["training"]["seed"]
    assert agent.training_sampling_min_std == settings.training_sampling_min_std
    assert agent.probability_min_std == settings.probability_min_std
    assert agent.evaluation_sampling_min_std == settings.evaluation_sampling_min_std
    assert agent.target_kl == settings.target_kl
    assert agent.normalize_advantages is True


@pytest.mark.parametrize("invalid", [None, "256,256", {"width": 256}])
def test_build_agent_config_rejects_non_sequence_value_hidden_dims(invalid) -> None:
    config = load_config("configs/debug.yaml")
    config["dppo"]["training"]["value_hidden_dims"] = invalid

    with pytest.raises(ValueError, match=r"dppo\.training\.value_hidden_dims"):
        build_dppo_agent_config(config, clip_ratio=0.10)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("training_sampling_min_std",), 0.0),
        (("probability_min_std",), math.inf),
        (("evaluation_sampling_min_std",), True),
        (("target_kl",), math.nan),
        (("target_clip_fraction_min",), -0.01),
        (("target_clip_fraction_max",), 1.01),
        (("target_clip_fraction_min",), 0.21),
        (("clip_ratio_candidates",), []),
        (("clip_ratio_candidates",), [0.1, 0.1]),
        (("clip_ratio_candidates",), [1.0]),
        (("calibration_iterations",), True),
        (("calibration_episodes_per_iteration",), 0),
        (("calibration_seed_start",), -1),
    ],
)
def test_settings_parser_rejects_invalid_boundaries(path, value) -> None:
    config = load_config("configs/debug.yaml")
    stability = dict(config["dppo"]["stability"])
    stability[path[0]] = value
    config["dppo"] = dict(config["dppo"])
    config["dppo"]["stability"] = stability

    with pytest.raises(ValueError):
        load_dppo_stability_settings(config)


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"maximum_approximate_kl": 1.0}, "KL"),
        ({"mean_clip_fraction": 0.09}, "clip fraction"),
        ({"mean_clip_fraction": 0.21}, "clip fraction"),
        ({"optimizer_step_count": 0}, "优化器"),
        ({"mean_approximate_kl": math.nan}, "有限"),
    ],
)
def test_candidate_qualification_reports_every_failure_branch(changes, reason) -> None:
    values = {
        "clip_ratio": 0.1,
        "mean_clip_fraction": 0.15,
        "mean_approximate_kl": 0.2,
        "maximum_approximate_kl": 0.3,
        "optimizer_step_count": 2,
        "settings": _settings(),
    }
    values.update(changes)

    result = evaluate_calibration_candidate(**values)

    assert result.qualified is False
    assert result.failure_reasons
    assert any(reason in item for item in result.failure_reasons)


def test_candidate_boundary_is_inclusive_but_target_kl_is_strict() -> None:
    low = _result(0.10, clip_fraction=0.10)
    high = _result(0.01, clip_fraction=0.20)

    assert low.qualified is True
    assert high.qualified is True
    assert low.failure_reasons == ()


def test_candidate_dataclass_rejects_raw_nan_instead_of_leaking_invalid_json() -> None:
    with pytest.raises(ValueError, match="mean_clip_fraction"):
        DPPOCalibrationCandidateResult(
            clip_ratio=0.10,
            mean_clip_fraction=math.nan,
            mean_approximate_kl=0.2,
            maximum_approximate_kl=0.3,
            optimizer_step_count=1,
            finite=False,
            qualified=False,
            failure_reasons=("非有限",),
        )


def test_profile_selects_largest_qualified_candidate_not_input_order() -> None:
    profile = select_stability_profile(
        config_hash="config-v1",
        pretrained_checkpoint_sha256="a" * 64,
        settings=_settings(),
        episode_seeds=(20000, 20001),
        candidate_results=(_result(0.01), _result(0.001), _result(0.10)),
    )

    assert profile.qualified is True
    assert profile.selected_clip_ratio == 0.10
    assert profile.candidate_values == _settings().clip_ratio_candidates


def test_profile_without_qualified_candidate_records_failure() -> None:
    profile = select_stability_profile(
        config_hash="config-v1",
        pretrained_checkpoint_sha256="a" * 64,
        settings=_settings(),
        episode_seeds=(20000, 20001),
        candidate_results=(
            _result(0.10, clip_fraction=0.01),
            _result(0.01, clip_fraction=0.01),
            _result(0.001, clip_fraction=0.01),
        ),
    )

    assert profile.qualified is False
    assert profile.selected_clip_ratio is None
    assert profile.failure_reasons


def test_nonfinite_candidate_can_be_saved_in_an_unqualified_profile(tmp_path) -> None:
    nonfinite = evaluate_calibration_candidate(
        clip_ratio=0.10,
        mean_clip_fraction=math.nan,
        mean_approximate_kl=math.inf,
        maximum_approximate_kl=-math.inf,
        optimizer_step_count=1,
        settings=_settings(),
    )
    profile = select_stability_profile(
        config_hash="config-v1",
        pretrained_checkpoint_sha256="a" * 64,
        settings=_settings(),
        episode_seeds=(20000, 20001),
        candidate_results=(
            nonfinite,
            _result(0.01, clip_fraction=0.01),
            _result(0.001, clip_fraction=0.01),
        ),
    )
    output = tmp_path / "failed-stability.json"

    save_stability_profile(output, profile)

    assert profile.qualified is False
    assert nonfinite.finite is False
    assert nonfinite.mean_clip_fraction is None
    assert nonfinite.mean_approximate_kl is None
    assert nonfinite.maximum_approximate_kl is None
    assert load_stability_profile(output) == profile
    assert "NaN" not in output.read_text(encoding="utf-8")
    assert "Infinity" not in output.read_text(encoding="utf-8")


def test_profile_recomputes_forged_candidate_qualification() -> None:
    forged = replace(_result(0.10, clip_fraction=0.01), qualified=True, failure_reasons=())

    profile = select_stability_profile(
        config_hash="config-v1",
        pretrained_checkpoint_sha256="a" * 64,
        settings=_settings(),
        episode_seeds=(20000, 20001),
        candidate_results=(forged, _result(0.01), _result(0.001)),
    )

    assert profile.selected_clip_ratio == 0.01
    recomputed = next(item for item in profile.candidate_results if item.clip_ratio == 0.10)
    assert recomputed.qualified is False


@pytest.mark.parametrize(
    "candidate_results",
    [
        (_result(0.10), _result(0.01)),
        (_result(0.10), _result(0.10), _result(0.001)),
        (_result(0.10), _result(0.01), _result(0.02)),
    ],
)
def test_profile_requires_exactly_one_result_per_configured_candidate(candidate_results) -> None:
    with pytest.raises(ValueError, match="候选"):
        select_stability_profile(
            config_hash="config-v1",
            pretrained_checkpoint_sha256="a" * 64,
            settings=_settings(),
            episode_seeds=(20000, 20001),
            candidate_results=candidate_results,
        )


def test_profile_json_round_trip_digest_and_file_io_are_canonical(tmp_path) -> None:
    profile, _ = _profile(tmp_path)
    serialized = stability_profile_json(profile)
    output = tmp_path / "nested" / "stability.json"

    save_stability_profile(output, profile)

    assert parse_stability_profile_json(serialized) == profile
    assert load_stability_profile(output) == profile
    assert output.read_text(encoding="utf-8") == serialized + "\n"
    assert stability_profile_sha256(profile) == stability_profile_sha256(
        parse_stability_profile_json(serialized.replace("\n", "\r\n"))
    )


def test_save_profile_writes_exactly_one_binary_lf_without_crlf(tmp_path) -> None:
    profile, _ = _profile(tmp_path)
    output = tmp_path / "stability.json"

    save_stability_profile(output, profile)

    payload = output.read_bytes()
    assert b"\r\n" not in payload
    assert payload.endswith(b"\n")
    assert not payload.endswith(b"\n\n")


def test_save_profile_replace_failure_preserves_old_file_and_cleans_temp(
    tmp_path,
    monkeypatch,
) -> None:
    profile, _ = _profile(tmp_path)
    output = tmp_path / "stability.json"
    output.write_bytes(b"old-profile\n")
    original_entries = set(tmp_path.iterdir())

    def fail_replace(source, destination) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        save_stability_profile(output, profile)

    assert output.read_bytes() == b"old-profile\n"
    assert set(tmp_path.iterdir()) == original_entries


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("maximum_approximate_kl", 99.0),
        ("mean_clip_fraction", 0.99),
    ],
)
def test_parser_rejects_metrics_tampered_without_updating_derived_fields(
    tmp_path,
    field,
    value,
) -> None:
    profile, _ = _profile(tmp_path)
    data = json.loads(stability_profile_json(profile))
    selected = next(
        item
        for item in data["candidate_results"]
        if item["clip_ratio"] == data["selected_clip_ratio"]
    )
    selected[field] = value

    with pytest.raises(ValueError, match="完整性"):
        parse_stability_profile_json(json.dumps(data))


def test_parser_rejects_selected_ratio_that_is_not_largest_qualified(tmp_path) -> None:
    profile, _ = _profile(tmp_path)
    data = json.loads(stability_profile_json(profile))
    data["selected_clip_ratio"] = 0.01

    with pytest.raises(ValueError, match="完整性"):
        parse_stability_profile_json(json.dumps(data))


def test_parser_rejects_forged_candidate_qualified_flag(tmp_path) -> None:
    profile, _ = _profile(tmp_path)
    data = json.loads(stability_profile_json(profile))
    candidate = next(
        item for item in data["candidate_results"] if item["clip_ratio"] == 0.01
    )
    candidate["qualified"] = False

    with pytest.raises(ValueError, match="完整性"):
        parse_stability_profile_json(json.dumps(data))


def test_parser_rejects_forged_candidate_finite_flag(tmp_path) -> None:
    profile, _ = _profile(tmp_path)
    data = json.loads(stability_profile_json(profile))
    data["candidate_results"][0]["finite"] = False

    with pytest.raises(ValueError):
        parse_stability_profile_json(json.dumps(data))


def test_parser_rejects_forged_candidate_failure_reasons(tmp_path) -> None:
    profile = select_stability_profile(
        config_hash="config-v1",
        pretrained_checkpoint_sha256="a" * 64,
        settings=_settings(),
        episode_seeds=(20000, 20001),
        candidate_results=(
            _result(0.10, clip_fraction=0.01),
            _result(0.01, clip_fraction=0.01),
            _result(0.001, clip_fraction=0.01),
        ),
    )
    data = json.loads(stability_profile_json(profile))
    data["candidate_results"][0]["failure_reasons"] = ["伪造失败原因"]

    with pytest.raises(ValueError, match="完整性"):
        parse_stability_profile_json(json.dumps(data, ensure_ascii=False))


def test_validate_profile_recomputes_integrity_after_in_memory_tampering(tmp_path) -> None:
    profile, checkpoint = _profile(tmp_path)
    selected = next(
        item
        for item in profile.candidate_results
        if item.clip_ratio == profile.selected_clip_ratio
    )
    # 模拟绕过 frozen 数据类的恶意内存修改，验证训练入口仍会独立重算。
    object.__setattr__(selected, "maximum_approximate_kl", 99.0)

    with pytest.raises(ValueError, match="完整性"):
        validate_stability_profile(
            profile,
            config_hash=profile.config_hash,
            pretrained_checkpoint_path=checkpoint,
            settings=_settings(),
        )


def test_validate_profile_rechecks_candidate_set_after_in_memory_tampering(tmp_path) -> None:
    profile, checkpoint = _profile(tmp_path)
    object.__setattr__(profile, "candidate_results", profile.candidate_results[:-1])

    with pytest.raises(ValueError, match="完整性"):
        validate_stability_profile(
            profile,
            config_hash=profile.config_hash,
            pretrained_checkpoint_path=checkpoint,
            settings=_settings(),
        )


@pytest.mark.parametrize(
    "tampered_seeds",
    [(), [20000, 20001], (20000, True), (20000, -1), (20000, 20000)],
)
def test_validate_profile_rechecks_episode_seeds_before_indexing(
    tmp_path,
    tampered_seeds,
) -> None:
    profile, checkpoint = _profile(tmp_path)
    object.__setattr__(profile, "episode_seeds", tampered_seeds)

    with pytest.raises(ValueError, match="episode_seeds"):
        validate_stability_profile(
            profile,
            config_hash=profile.config_hash,
            pretrained_checkpoint_path=checkpoint,
            settings=_settings(),
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"config_hash": "wrong"}, "配置哈希"),
        ({"checkpoint_contents": b"changed"}, "检查点"),
        ({"qualified": False}, "未通过"),
        ({"settings": replace(_settings(), target_kl=0.5)}, "target_kl"),
    ],
)
def test_validate_profile_rejects_mismatches(tmp_path, change, message) -> None:
    profile, checkpoint = _profile(tmp_path)
    if "checkpoint_contents" in change:
        checkpoint.write_bytes(change["checkpoint_contents"])
    if "qualified" in change:
        profile = select_stability_profile(
            config_hash=profile.config_hash,
            pretrained_checkpoint_sha256=profile.pretrained_checkpoint_sha256,
            settings=_settings(),
            episode_seeds=profile.episode_seeds,
            candidate_results=(
                _result(0.10, clip_fraction=0.01),
                _result(0.01, clip_fraction=0.01),
                _result(0.001, clip_fraction=0.01),
            ),
        )

    with pytest.raises(ValueError, match=message):
        validate_stability_profile(
            profile,
            config_hash=change.get("config_hash", profile.config_hash),
            pretrained_checkpoint_path=checkpoint,
            settings=change.get("settings", _settings()),
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data.update(schema_version="unknown"),
        lambda data: data.update(target_kl=float("nan")),
        lambda data: data.update(candidate_values="not-a-list"),
        lambda data: data.pop("candidate_results"),
        lambda data: data.update(pretrained_checkpoint_sha256="ABC"),
    ],
)
def test_parser_rejects_unknown_schema_nan_and_structural_errors(tmp_path, mutation) -> None:
    profile, _ = _profile(tmp_path)
    data = json.loads(stability_profile_json(profile))
    mutation(data)

    with pytest.raises(ValueError):
        parse_stability_profile_json(json.dumps(data, allow_nan=True))


@pytest.mark.parametrize(
    "changes",
    [
        {"config_hash": ""},
        {"pretrained_checkpoint_sha256": "A" * 64},
        {"candidate_values": (0.1, 0.1)},
        {"episode_seeds": (20000, 20000)},
        {"qualified": True, "selected_clip_ratio": None},
    ],
)
def test_profile_dataclass_rejects_invalid_identity_and_contradictions(tmp_path, changes) -> None:
    profile, _ = _profile(tmp_path)

    with pytest.raises(ValueError):
        replace(profile, **changes)


def test_schema_version_is_explicit() -> None:
    assert STABILITY_PROFILE_SCHEMA_VERSION == "dppo-stability-v1"
