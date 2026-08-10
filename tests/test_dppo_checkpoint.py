"""测试扩散预训练、版本化检查点和设备兼容性。"""

from dataclasses import replace

import pytest
import torch

from src.dppo_checkpoint import (
    DPPOCheckpointMetadata,
    evaluate_diffusion_loss,
    load_dppo_checkpoint,
    pretrain_diffusion_epoch,
    resolve_torch_device,
    save_dppo_checkpoint,
    seed_torch_for_pretraining,
)
from src.dppo_diffusion import (
    ConditionalDiffusionMLP,
    CosineNoiseSchedule,
    sample_denoising_chain,
)


def _metadata() -> DPPOCheckpointMetadata:
    """返回小规模检查点使用的完整兼容性元数据。"""

    return DPPOCheckpointMetadata(
        state_schema_version="dppo-v1-flat",
        action_schema_version="joint-sfc-continuous-v1",
        state_dim=58,
        action_dim=14,
        mec_count=3,
        compute_node_count=4,
        function_count=2,
        diffusion_steps=20,
        fine_tuned_steps=5,
        maximum_retention_seconds=20.0,
        replica_threshold=0.0,
        config_hash="test-hash",
    )


def _synthetic_expert_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """构造两批可重复的状态和 [-1,1] 专家动作。"""

    generator = torch.Generator(device="cpu").manual_seed(101)
    states = torch.randn((8, 58), generator=generator)
    actions = torch.tanh(torch.randn((8, 14), generator=generator))
    return states, actions


def test_two_minibatches_train_and_checkpoint_round_trip(tmp_path) -> None:
    """训练、保存和加载后，相同种子的去噪动作必须完全一致。"""

    torch.manual_seed(103)
    states, actions = _synthetic_expert_batch()
    model = ConditionalDiffusionMLP(58, 14, (32, 32))
    schedule = CosineNoiseSchedule(steps=20)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    train_loss = pretrain_diffusion_epoch(
        model,
        schedule,
        optimizer,
        states,
        actions,
        batch_size=4,
        seed=107,
        device="cpu",
    )
    validation_loss = evaluate_diffusion_loss(
        model,
        schedule,
        states[:4],
        actions[:4],
        batch_size=4,
        seed=109,
        device="cpu",
    )
    checkpoint_path = tmp_path / "tiny.pt"
    save_dppo_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        _metadata(),
        epoch=0,
    )
    loaded = load_dppo_checkpoint(
        checkpoint_path,
        expected=_metadata(),
        device="cpu",
    )
    original_sample = sample_denoising_chain(
        model,
        schedule,
        states[:2],
        seed=113,
    )
    loaded_sample = sample_denoising_chain(
        loaded.model,
        schedule,
        states[:2],
        seed=113,
    )

    assert torch.isfinite(torch.tensor(train_loss))
    assert torch.isfinite(torch.tensor(validation_loss))
    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter in model.named_parameters()
    )
    assert loaded.epoch == 0
    assert loaded.metadata == _metadata()
    assert loaded.optimizer.state_dict()["state"]
    assert torch.equal(original_sample.actions, loaded_sample.actions)
    assert torch.equal(
        original_sample.log_probabilities,
        loaded_sample.log_probabilities,
    )


def test_checkpoint_rejects_every_incompatible_metadata_field(tmp_path) -> None:
    """任何场景、算法或配置字段变化都不能静默复用旧检查点。"""

    metadata = _metadata()
    model = ConditionalDiffusionMLP(58, 14, (32, 32))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "tiny.pt"
    save_dppo_checkpoint(checkpoint_path, model, optimizer, metadata, epoch=0)
    incompatible_values = {
        "state_schema_version": "dppo-v2",
        "action_schema_version": "joint-sfc-v2",
        "state_dim": 78,
        "action_dim": 27,
        "mec_count": 5,
        "compute_node_count": 6,
        "function_count": 3,
        "diffusion_steps": 10,
        "fine_tuned_steps": 4,
        "maximum_retention_seconds": 30.0,
        "replica_threshold": 0.25,
        "config_hash": "different-hash",
    }

    for field_name, incompatible_value in incompatible_values.items():
        with pytest.raises(ValueError, match=field_name):
            load_dppo_checkpoint(
                checkpoint_path,
                expected=replace(
                    metadata,
                    **{field_name: incompatible_value},
                ),
                device="cpu",
            )


def test_checkpoint_restores_saved_torch_rng_state(tmp_path) -> None:
    """恢复训练时应从保存时的 PyTorch 随机状态继续。"""

    torch.manual_seed(127)
    model = ConditionalDiffusionMLP(58, 14, (16, 16))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "rng.pt"
    saved_rng_state = torch.get_rng_state().clone()
    save_dppo_checkpoint(checkpoint_path, model, optimizer, _metadata(), epoch=2)
    torch.manual_seed(131)

    loaded = load_dppo_checkpoint(
        checkpoint_path,
        expected=_metadata(),
        device="cpu",
    )

    assert loaded.epoch == 2
    assert torch.equal(loaded.rng_state, saved_rng_state)
    assert torch.equal(torch.get_rng_state(), saved_rng_state)


def test_unavailable_cuda_is_rejected_explicitly() -> None:
    """用户请求不可用的 CUDA 时必须明确报错，不能静默退回 CPU。"""

    if torch.cuda.is_available():
        assert resolve_torch_device("cuda").type == "cuda"
    else:
        with pytest.raises(ValueError, match="CUDA"):
            resolve_torch_device("cuda")
    assert resolve_torch_device("cpu").type == "cpu"


def test_pretraining_seed_reproduces_model_initialization() -> None:
    """配置中的预训练种子必须控制网络初始参数，而不只控制批次噪声。"""

    seed_torch_for_pretraining(137, "cpu")
    first = ConditionalDiffusionMLP(10, 4, (16, 16))
    first_parameters = tuple(
        parameter.detach().clone() for parameter in first.parameters()
    )
    seed_torch_for_pretraining(137, "cpu")
    second = ConditionalDiffusionMLP(10, 4, (16, 16))

    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(
            first_parameters,
            second.parameters(),
            strict=True,
        )
    )


def test_pretraining_cli_requires_dataset_and_output_roots() -> None:
    """预训练命令必须显式指定输入数据和输出检查点目录。"""

    from run_dppo_pretraining import parse_arguments

    with pytest.raises(SystemExit):
        parse_arguments([])

    arguments = parse_arguments(
        [
            "--dataset-root",
            "temporary-dataset",
            "--output-root",
            "temporary-checkpoint",
            "--epochs",
            "1",
            "--device",
            "cpu",
        ]
    )
    assert arguments.dataset_root == "temporary-dataset"
    assert arguments.output_root == "temporary-checkpoint"
    assert arguments.epochs == 1
    assert arguments.device == "cpu"
