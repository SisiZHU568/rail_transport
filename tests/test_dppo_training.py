"""测试 DDPO 与慢时间尺度环境连接后的在线训练闭环。"""

import csv
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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


def _agent(environment) -> DPPOAgent:
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
        ),
        device="cpu",
    )


def _metadata(environment, agent: DPPOAgent) -> DPPOCheckpointMetadata:
    """构造与当前测试场景和双层策略一致的检查点元数据。"""

    dimensions = environment.dimensions
    return DPPOCheckpointMetadata(
        state_schema_version="dppo-v1-flat",
        action_schema_version="joint-sfc-continuous-v1",
        state_dim=dimensions.state_dim,
        action_dim=dimensions.action_dim,
        mec_count=dimensions.mec_count,
        compute_node_count=dimensions.compute_node_count,
        function_count=dimensions.function_count,
        diffusion_steps=agent.config.diffusion_steps,
        fine_tuned_steps=agent.config.fine_tuned_steps,
        maximum_retention_seconds=20.0,
        replica_threshold=0.0,
        config_hash="online-training-test",
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


def test_online_checkpoint_restores_both_policy_layers_and_value_network(
    tmp_path: Path,
) -> None:
    """在线检查点必须完整恢复冻结前段、可训练末段和价值网络。"""

    environment = _environment()
    agent = _agent(environment)
    metadata = _metadata(environment, agent)
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
    )
    loaded = load_dppo_online_checkpoint(
        checkpoint_path,
        expected=metadata,
        device="cpu",
    )

    assert loaded.iteration == 3
    assert loaded.best_mean_reward == -0.75
    assert loaded.agent.config == agent.config
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
    output_root = tmp_path / "online-output"

    history = train_dppo(
        environment,
        agent,
        metadata,
        iterations=1,
        episodes_per_iteration=1,
        seed=90,
        output_root=output_root,
    )

    expected_paths = {
        output_root / "dppo_online_last.pt",
        output_root / "dppo_online_best.pt",
        output_root / "training_history.csv",
    }
    assert expected_paths <= set(output_root.iterdir())
    assert len(history) == 1
    assert all(math.isfinite(float(value)) for value in history[0].values())
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
    } <= rows[0].keys()
    loaded = load_dppo_online_checkpoint(
        output_root / "dppo_online_last.pt",
        expected=metadata,
        device="cpu",
    )
    assert loaded.iteration == 0


def test_main_uses_unified_agent_config_builder_with_legacy_clip_ratio(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """接入 profile 前，旧训练入口仍显式把 YAML clip_ratio 交给唯一 builder。"""

    config = load_config("configs/debug.yaml")
    built_config = object()
    observed: list[tuple[dict, float]] = []
    environment = SimpleNamespace(dimensions=SimpleNamespace(state_dim=3, action_dim=2))
    agent = SimpleNamespace(frozen_denoising_steps=2, trainable_denoising_steps=2)

    monkeypatch.setattr(run_dppo_training, "load_config", lambda _path: config)
    monkeypatch.setattr(run_dppo_training, "build_dppo_environment", lambda _config: environment)
    monkeypatch.setattr(
        run_dppo_training,
        "_checkpoint_metadata",
        lambda *_args: SimpleNamespace(diffusion_steps=4),
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
        lambda *_args, **_kwargs: SimpleNamespace(iteration=0, best_mean_reward=0.0),
    )

    def fake_builder(actual_config, *, clip_ratio):
        observed.append((actual_config, clip_ratio))
        return built_config

    monkeypatch.setattr(run_dppo_training, "build_dppo_agent_config", fake_builder)

    run_dppo_training.main(
        [
            "--pretrained-checkpoint",
            str(tmp_path / "pretrained.pt"),
            "--output-root",
            str(tmp_path / "output"),
            "--device",
            "cpu",
        ]
    )

    assert not hasattr(run_dppo_training, "_agent_config")
    assert observed == [(config, float(config["dppo"]["training"]["clip_ratio"]))]
