"""使用版本化专家数据预训练 DPPO 条件扩散动作网络。"""

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from src.config import load_config
from src.dppo_checkpoint import (
    DPPOCheckpointMetadata,
    evaluate_diffusion_loss,
    load_dppo_checkpoint,
    pretrain_diffusion_epoch,
    resolve_torch_device,
    save_dppo_checkpoint,
    seed_torch_for_pretraining,
)
from src.dppo_dataset import (
    EXPERT_DATASET_SCHEMA_VERSION,
    ExpertDatasetMetadata,
    ExpertTransitionRecord,
    compute_config_hash,
    load_expert_dataset,
)
from src.dppo_diffusion import ConditionalDiffusionMLP, CosineNoiseSchedule
from src.dppo_scenario import build_dppo_scenario
from src.dppo_slow_timescale_env import DPPOSlowTimescaleEnvironment


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """解析参数，并强制显式给出数据集和检查点输出目录。"""

    parser = argparse.ArgumentParser(
        description="在专家仿真数据上预训练 DPPO 条件扩散网络。"
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "configs" / "debug.yaml"),
        help="YAML 配置文件路径。",
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="第 10 批生成的专家数据集目录。",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="预训练检查点输出目录。",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="覆盖配置中的预训练轮数。",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
        help="覆盖配置中的训练设备。",
    )
    return parser.parse_args(arguments)


def _records_to_tensors(
    records: tuple[ExpertTransitionRecord, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """把同构行为克隆记录转换为连续 float32 批张量。"""

    if not records:
        raise ValueError("预训练数据分区不能为空。")
    states = torch.from_numpy(
        np.stack([record.state for record in records]).astype(np.float32)
    )
    actions = torch.from_numpy(
        np.stack([record.expert_action for record in records]).astype(np.float32)
    )
    return states, actions


def _checkpoint_metadata(
    config: dict[str, Any],
    scenario: DPPOSlowTimescaleEnvironment,
    config_hash: str,
) -> DPPOCheckpointMetadata:
    """从唯一场景维度和配置构造检查点兼容性元数据。"""

    dimensions = scenario.dimensions
    action_config = config["dppo"]["action"]
    diffusion_config = config["dppo"]["diffusion"]
    return DPPOCheckpointMetadata(
        state_schema_version=str(config["dppo"]["training"]["state_schema_version"]),
        action_schema_version=str(action_config["schema_version"]),
        state_dim=dimensions.state_dim,
        action_dim=dimensions.action_dim,
        mec_count=dimensions.mec_count,
        compute_node_count=dimensions.compute_node_count,
        function_count=dimensions.function_count,
        diffusion_steps=int(diffusion_config["steps"]),
        fine_tuned_steps=int(diffusion_config["fine_tuned_steps"]),
        maximum_retention_seconds=float(
            action_config["maximum_retention_seconds"]
        ),
        minimum_replicas=int(action_config["minimum_replicas"]),
        maximum_replicas=int(action_config["maximum_replicas"]),
        config_hash=config_hash,
    )


def main(arguments: Sequence[str] | None = None) -> None:
    """加载兼容数据集，训练扩散网络并保存可恢复检查点。"""

    parsed = parse_arguments(arguments)
    config = load_config(parsed.config)
    pretraining_config = config["dppo"]["pretraining"]
    epochs = (
        int(pretraining_config["epochs"])
        if parsed.epochs is None
        else parsed.epochs
    )
    if epochs <= 0:
        raise ValueError("epochs 必须是正整数。")
    device_name = (
        str(config["dppo"]["training"]["device"])
        if parsed.device is None
        else parsed.device
    )
    device = resolve_torch_device(device_name)
    config_hash = compute_config_hash(config)
    scenario = build_dppo_scenario(config)
    checkpoint_metadata = _checkpoint_metadata(config, scenario, config_hash)
    expected_dataset_metadata = ExpertDatasetMetadata(
        schema_version=EXPERT_DATASET_SCHEMA_VERSION,
        state_dim=scenario.dimensions.state_dim,
        action_dim=scenario.dimensions.action_dim,
        config_hash=config_hash,
        state_schema_version=checkpoint_metadata.state_schema_version,
        action_schema_version=checkpoint_metadata.action_schema_version,
    )
    dataset = load_expert_dataset(
        parsed.dataset_root,
        expected_metadata=expected_dataset_metadata,
    )
    train_states, train_actions = _records_to_tensors(
        dataset.partitions["train"]
    )
    validation_states, validation_actions = _records_to_tensors(
        dataset.partitions["validation"]
    )

    diffusion_config = config["dppo"]["diffusion"]
    base_seed = int(pretraining_config["seed"])
    seed_torch_for_pretraining(base_seed, device)
    model = ConditionalDiffusionMLP(
        scenario.dimensions.state_dim,
        scenario.dimensions.action_dim,
        tuple(int(width) for width in diffusion_config["hidden_dims"]),
    ).to(device)
    schedule = CosineNoiseSchedule(int(diffusion_config["steps"]))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(pretraining_config["learning_rate"]),
    )
    batch_size = int(pretraining_config["batch_size"])
    for epoch in range(epochs):
        train_loss = pretrain_diffusion_epoch(
            model,
            schedule,
            optimizer,
            train_states,
            train_actions,
            batch_size=batch_size,
            seed=base_seed + epoch,
            device=device,
            gradient_clip_norm=float(pretraining_config["gradient_clip_norm"]),
        )
        validation_loss = evaluate_diffusion_loss(
            model,
            schedule,
            validation_states,
            validation_actions,
            batch_size=batch_size,
            seed=base_seed + 100000 + epoch,
            device=device,
        )
        print(
            f"epoch={epoch + 1}/{epochs} "
            f"train_loss={train_loss:.6f} "
            f"validation_loss={validation_loss:.6f}"
        )

    checkpoint_path = Path(parsed.output_root) / "dppo_pretrained.pt"
    save_dppo_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        checkpoint_metadata,
        epoch=epochs - 1,
    )
    # 写盘后立即重新加载，确保交付的文件本身可恢复且配置完全一致。
    restored = load_dppo_checkpoint(
        checkpoint_path,
        expected=checkpoint_metadata,
        device=device,
    )
    print(f"检查点：{checkpoint_path.resolve()}")
    print(f"恢复 epoch：{restored.epoch}")
    print(f"配置哈希：{restored.metadata.config_hash}")


if __name__ == "__main__":
    main()
