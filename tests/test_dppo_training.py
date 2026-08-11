"""测试 DDPO 与慢时间尺度环境连接后的在线训练闭环。"""

import csv
from dataclasses import replace
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import run_dppo_training
from run_dppo_training import collect_rollout, train_dppo
from src.config import load_config
from src.dppo import DPPOAgent, DPPOConfig, DPPORolloutBuffer
from src.dppo_checkpoint import (
    DPPOCheckpointMetadata,
    load_dppo_online_checkpoint,
    save_dppo_online_checkpoint,
)
from src.dppo_diffusion import ConditionalDiffusionMLP, CosineNoiseSchedule
from src.dppo_projection import ProjectionResult
from src.dppo_scenario import build_dppo_environment
from src.dppo_stability import (
    DPPOStabilityProfile,
    evaluate_calibration_candidate,
    save_stability_profile,
    select_stability_profile,
    sha256_file,
    stability_profile_sha256,
)
from src.dppo_training_config import (
    DPPOStabilitySettings,
    load_dppo_stability_settings,
)


class AlwaysFailProjector:
    """稳定制造不可投影动作，验证负奖励样本不会被训练循环丢弃。"""

    def project(self, **_: object) -> ProjectionResult:
        return ProjectionResult(
            function_intents=None,
            raw_feasible=False,
            success=False,
            reasons=("在线训练测试注入的投影失败",),
            changed_assignment_count=0,
            requested_assignment_count=1,
            change_ratio=0.0,
            fault_domains={},
        )


def _environment():
    """使用真实场景工厂，但让测试中的拒绝分支无需运行快层。"""

    environment = build_dppo_environment(load_config("configs/debug.yaml"))
    environment.projector = AlwaysFailProjector()
    return environment


def _agent(environment, *, clip_ratio: float = 0.1) -> DPPOAgent:
    """构造小型网络，减少在线更新测试耗时。"""

    torch.manual_seed(17)
    diffusion_steps = 4
    model = ConditionalDiffusionMLP(
        environment.dimensions.state_dim,
        environment.dimensions.action_dim,
        (16, 16),
    )
    return DPPOAgent(
        model,
        CosineNoiseSchedule(diffusion_steps),
        DPPOConfig(
            diffusion_steps=diffusion_steps,
            fine_tuned_steps=2,
            value_hidden_dims=(16, 16),
            batch_size=8,
            update_epochs=1,
            seed=71,
            clip_ratio=clip_ratio,
        ),
        device="cpu",
    )


def _metadata(environment, agent: DPPOAgent) -> DPPOCheckpointMetadata:
    """构造与当前测试场景和双层策略一致的检查点元数据。"""

    dimensions = environment.dimensions
    return DPPOCheckpointMetadata(
        state_schema_version="dppo-v2-flat",
        action_schema_version="joint-sfc-continuous-v2",
        state_dim=dimensions.state_dim,
        action_dim=dimensions.action_dim,
        mec_count=dimensions.mec_count,
        compute_node_count=dimensions.compute_node_count,
        function_count=dimensions.function_count,
        diffusion_steps=agent.config.diffusion_steps,
        fine_tuned_steps=agent.config.fine_tuned_steps,
        maximum_retention_seconds=20.0,
        minimum_replicas=2,
        maximum_replicas=3,
        config_hash="online-training-test",
    )


def _profile(
    metadata: DPPOCheckpointMetadata,
    *,
    pretrained_sha256: str = "a" * 64,
) -> DPPOStabilityProfile:
    """通过正式候选评估和选择入口生成合格训练门禁。"""

    settings = DPPOStabilitySettings(
        training_sampling_min_std=0.01,
        probability_min_std=0.10,
        evaluation_sampling_min_std=0.001,
        target_kl=1.0,
        target_clip_fraction_min=0.10,
        target_clip_fraction_max=0.20,
        clip_ratio_candidates=(0.10,),
        calibration_iterations=1,
        calibration_episodes_per_iteration=1,
        calibration_seed_start=20000,
    )
    result = evaluate_calibration_candidate(
        clip_ratio=0.10,
        mean_clip_fraction=0.15,
        mean_approximate_kl=0.2,
        maximum_approximate_kl=0.3,
        optimizer_step_count=1,
        settings=settings,
    )
    return select_stability_profile(
        config_hash=metadata.config_hash,
        pretrained_checkpoint_sha256=pretrained_sha256,
        settings=settings,
        episode_seeds=(20000,),
        candidate_results=(result,),
    )


def _config_profile(
    config: dict,
    pretrained_path: Path,
    *,
    mode: str = "qualified",
) -> DPPOStabilityProfile:
    settings = load_dppo_stability_settings(config)
    profile_settings = (
        replace(settings, target_kl=0.5) if mode == "settings" else settings
    )
    maximum_kl = profile_settings.target_kl if mode == "unqualified" else 0.2
    qualifying_clip_fraction = (
        profile_settings.target_clip_fraction_min
        + profile_settings.target_clip_fraction_max
    ) / 2.0
    results = tuple(
        evaluate_calibration_candidate(
            clip_ratio=value,
            mean_clip_fraction=qualifying_clip_fraction,
            mean_approximate_kl=0.1,
            maximum_approximate_kl=maximum_kl,
            optimizer_step_count=1,
            settings=profile_settings,
        )
        for value in profile_settings.clip_ratio_candidates
    )
    seed_count = (
        profile_settings.calibration_iterations
        * profile_settings.calibration_episodes_per_iteration
    )
    return select_stability_profile(
        config_hash=(
            "wrong-config"
            if mode == "config_hash"
            else run_dppo_training.compute_config_hash(config)
        ),
        pretrained_checkpoint_sha256=(
            "a" * 64 if mode == "pretrained_sha" else sha256_file(pretrained_path)
        ),
        settings=profile_settings,
        episode_seeds=tuple(
            range(
                profile_settings.calibration_seed_start,
                profile_settings.calibration_seed_start + seed_count,
            )
        ),
        candidate_results=results,
    )


def _assert_same_state_dict(
    first: dict[str, torch.Tensor],
    second: dict[str, torch.Tensor],
) -> None:
    """逐参数验证两个网络状态完全相同。"""

    assert first.keys() == second.keys()
    assert all(torch.equal(first[key], second[key]) for key in first)


def test_environment_reports_clipped_decoder_and_training_audit_fields() -> None:
    """环境必须明确证明解码器收到截断动作，并直接暴露训练审计指标。"""

    environment = _environment()
    environment.reset(seed=9)
    raw_action = np.full(
        environment.dimensions.action_dim,
        2.0,
        dtype=np.float32,
    )

    _, reward, _, _, info = environment.step(raw_action)

    expected_decoded = environment.action_space.decode(info["clipped_action"])
    assert np.max(info["clipped_action"]) == 1.0
    assert info["decoded_action"] == expected_decoded
    assert info["raw_feasible"] is False
    assert info["projection_success"] is False
    assert info["projection_change_ratio"] == 0.0
    assert info["fast_repair_attempts"] == 0
    assert info["fast_repair_successes"] == 0
    assert info["fast_repair_failures"] == 0
    assert reward < 0.0


def test_collect_rollout_keeps_rejected_action_on_policy() -> None:
    """投影失败动作仍必须连同完整去噪链和负奖励进入 on-policy Buffer。"""

    environment = _environment()
    agent = _agent(environment)
    buffer = DPPORolloutBuffer()

    metrics = collect_rollout(environment, agent, buffer, seed=9)

    assert len(buffer) > 0
    assert all(transition.reward < 0.0 for transition in buffer.transitions)
    assert all(
        transition.denoising_actions.shape
        == (agent.config.diffusion_steps + 1, environment.dimensions.action_dim)
        for transition in buffer.transitions
    )
    assert all(
        np.array_equal(
            transition.raw_action,
            transition.denoising_actions[-1],
        )
        for transition in buffer.transitions
    )
    assert metrics["raw_feasibility_rate"] == 0.0
    assert metrics["projection_rejection_rate"] == 1.0
    assert math.isfinite(metrics["mean_reward"])

    losses = agent.update(buffer)
    assert math.isfinite(losses["policy_loss"])
    assert math.isfinite(losses["value_loss"])
    assert losses["optimizer_step_count"] >= 1.0
    assert 0.0 <= losses["clip_fraction"] <= 1.0


def test_online_checkpoint_restores_both_policy_layers_and_value_network(
    tmp_path: Path,
) -> None:
    """在线检查点必须完整恢复冻结前段、可训练末段和价值网络。"""

    environment = _environment()
    agent = _agent(environment)
    metadata = _metadata(environment, agent)
    profile = _profile(metadata)
    with torch.no_grad():
        next(agent.trainable_policy.parameters()).add_(0.25)
        next(agent.value_network.parameters()).sub_(0.10)
    checkpoint_path = tmp_path / "checkpoints" / "last.pt"

    save_dppo_online_checkpoint(
        checkpoint_path,
        agent,
        metadata,
        iteration=3,
        best_mean_reward=-0.75,
        stability_profile=profile,
    )
    loaded = load_dppo_online_checkpoint(
        checkpoint_path,
        expected=metadata,
        device="cpu",
    )

    assert loaded.iteration == 3
    assert loaded.best_mean_reward == -0.75
    assert loaded.agent.config == agent.config
    assert loaded.stability_profile == profile
    assert loaded.stability_profile_sha256 == stability_profile_sha256(profile)
    _assert_same_state_dict(
        agent.frozen_policy.state_dict(),
        loaded.agent.frozen_policy.state_dict(),
    )
    _assert_same_state_dict(
        agent.trainable_policy.state_dict(),
        loaded.agent.trainable_policy.state_dict(),
    )
    _assert_same_state_dict(
        agent.value_network.state_dict(),
        loaded.agent.value_network.state_dict(),
    )


def test_one_iteration_training_writes_only_under_supplied_output_root(
    tmp_path: Path,
) -> None:
    """烟雾训练只允许在调用方指定目录保存历史和两个检查点。"""

    environment = _environment()
    agent = _agent(environment)
    metadata = _metadata(environment, agent)
    profile = _profile(metadata)
    output_root = tmp_path / "online-output"

    history = train_dppo(
        environment,
        agent,
        metadata,
        iterations=1,
        episodes_per_iteration=1,
        seed=90,
        output_root=output_root,
        stability_profile=profile,
    )

    expected_paths = {
        output_root / "dppo_online_last.pt",
        output_root / "dppo_online_best.pt",
        output_root / "training_history.csv",
    }
    assert expected_paths <= set(output_root.iterdir())
    assert len(history) == 1
    assert all(
        isinstance(value, str) or math.isfinite(float(value))
        for value in history[0].values()
    )
    with (output_root / "training_history.csv").open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 1
    assert {
        "mean_reward",
        "policy_loss",
        "value_loss",
        "raw_feasibility_rate",
        "projection_change_ratio",
        "projection_rejection_rate",
        "repair_success_rate",
        "gradient_norm",
        "maximum_approximate_kl",
        "optimizer_step_count",
        "kl_early_stopped",
        "selected_clip_ratio",
        "stability_profile_sha256",
    } <= rows[0].keys()
    assert rows[0]["stability_profile_sha256"] == stability_profile_sha256(profile)
    loaded = load_dppo_online_checkpoint(
        output_root / "dppo_online_last.pt",
        expected=metadata,
        device="cpu",
    )
    assert loaded.iteration == 0
    assert loaded.stability_profile == profile
    best_loaded = load_dppo_online_checkpoint(
        output_root / "dppo_online_best.pt",
        expected=metadata,
        device="cpu",
    )
    assert best_loaded.stability_profile == profile
    assert best_loaded.stability_profile_sha256 == stability_profile_sha256(profile)


def test_training_rejects_profile_binding_before_creating_output(tmp_path: Path) -> None:
    environment = _environment()
    agent = _agent(environment)
    metadata = _metadata(environment, agent)
    output_root = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match="config_hash"):
        train_dppo(
            environment,
            agent,
            metadata,
            iterations=1,
            episodes_per_iteration=1,
            seed=90,
            output_root=output_root,
            stability_profile=replace(_profile(metadata), config_hash="wrong"),
        )

    assert not output_root.exists()


def test_training_rejects_agent_profile_binding_before_creating_output(
    tmp_path: Path,
) -> None:
    environment = _environment()
    agent = _agent(environment, clip_ratio=0.2)
    metadata = _metadata(environment, agent)
    output_root = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match="clip_ratio"):
        train_dppo(
            environment,
            agent,
            metadata,
            iterations=1,
            episodes_per_iteration=1,
            seed=90,
            output_root=output_root,
            stability_profile=_profile(metadata),
        )

    assert not output_root.exists()


def test_training_cli_requires_stability_profile() -> None:
    with pytest.raises(SystemExit):
        run_dppo_training.parse_arguments(
            ["--pretrained-checkpoint", "pretrained.pt", "--output-root", "out"]
        )


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("unqualified", "未通过"),
        ("config_hash", "配置哈希"),
        ("pretrained_sha", "SHA256"),
        ("settings", "target_kl"),
    ],
)
def test_main_rejects_invalid_profile_before_creating_output(
    tmp_path: Path,
    monkeypatch,
    mode: str,
    message: str,
) -> None:
    config = load_config("configs/debug.yaml")
    pretrained_path = tmp_path / "pretrained.pt"
    pretrained_path.write_bytes(b"pretrained")
    profile_path = tmp_path / "profile.json"
    save_stability_profile(
        profile_path,
        _config_profile(config, pretrained_path, mode=mode),
    )
    output_root = tmp_path / "must-not-exist"

    monkeypatch.setattr(
        run_dppo_training,
        "build_dppo_environment",
        lambda _config: pytest.fail("profile 校验失败前不得创建环境"),
    )

    with pytest.raises(ValueError, match=message):
        run_dppo_training.main(
            [
                "--pretrained-checkpoint",
                str(pretrained_path),
                "--stability-profile",
                str(profile_path),
                "--output-root",
                str(output_root),
                "--device",
                "cpu",
            ]
        )

    assert not output_root.exists()


def test_main_uses_qualified_profile_clip_ratio_instead_of_yaml(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """正式训练只能消费 profile 选择值，不能回退到 YAML clip_ratio。"""

    config = load_config("configs/debug.yaml")
    pretrained_path = tmp_path / "pretrained.pt"
    pretrained_path.write_bytes(b"pretrained")
    config_hash = run_dppo_training.compute_config_hash(config)
    settings = load_dppo_stability_settings(config)
    qualifying_clip_fraction = (
        settings.target_clip_fraction_min
        + settings.target_clip_fraction_max
    ) / 2.0
    results = tuple(
        evaluate_calibration_candidate(
            clip_ratio=value,
            mean_clip_fraction=qualifying_clip_fraction,
            mean_approximate_kl=0.2,
            maximum_approximate_kl=0.3,
            optimizer_step_count=1,
            settings=settings,
        )
        for value in settings.clip_ratio_candidates
    )
    seed_count = (
        settings.calibration_iterations
        * settings.calibration_episodes_per_iteration
    )
    profile = select_stability_profile(
        config_hash=config_hash,
        pretrained_checkpoint_sha256=sha256_file(pretrained_path),
        settings=settings,
        episode_seeds=tuple(
            range(
                settings.calibration_seed_start,
                settings.calibration_seed_start + seed_count,
            )
        ),
        candidate_results=results,
    )
    profile_path = tmp_path / "profile.json"
    save_stability_profile(profile_path, profile)
    built_config = SimpleNamespace(clip_ratio=profile.selected_clip_ratio)
    observed: list[tuple[dict, float]] = []
    environment = SimpleNamespace(dimensions=SimpleNamespace(state_dim=3, action_dim=2))
    agent = SimpleNamespace(frozen_denoising_steps=2, trainable_denoising_steps=2)

    monkeypatch.setattr(run_dppo_training, "load_config", lambda _path: config)
    monkeypatch.setattr(run_dppo_training, "build_dppo_environment", lambda _config: environment)
    monkeypatch.setattr(
        run_dppo_training,
        "_checkpoint_metadata",
        lambda *_args: SimpleNamespace(
            diffusion_steps=4,
            config_hash=config_hash,
        ),
    )
    monkeypatch.setattr(
        run_dppo_training,
        "load_dppo_checkpoint",
        lambda *_args, **_kwargs: SimpleNamespace(model=object()),
    )
    monkeypatch.setattr(run_dppo_training, "CosineNoiseSchedule", lambda steps: steps)
    monkeypatch.setattr(run_dppo_training, "DPPOAgent", lambda *_args, **_kwargs: agent)
    monkeypatch.setattr(run_dppo_training, "train_dppo", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(
        run_dppo_training,
        "load_dppo_online_checkpoint",
        lambda *_args, **_kwargs: SimpleNamespace(
            iteration=0,
            best_mean_reward=0.0,
            stability_profile_sha256=stability_profile_sha256(profile),
        ),
    )

    def fake_builder(actual_config, *, clip_ratio):
        observed.append((actual_config, clip_ratio))
        return built_config

    monkeypatch.setattr(run_dppo_training, "build_dppo_agent_config", fake_builder)

    run_dppo_training.main(
        [
            "--pretrained-checkpoint",
            str(pretrained_path),
            "--stability-profile",
            str(profile_path),
            "--output-root",
            str(tmp_path / "output"),
            "--device",
            "cpu",
        ]
    )

    assert not hasattr(run_dppo_training, "_agent_config")
    assert observed == [(config, profile.selected_clip_ratio)]
