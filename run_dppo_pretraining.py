"""使用版本化专家数据预训练 DPPO 条件扩散动作网络。"""

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from src.config import load_config
from src.dppo_checkpoint import (
    DPPOCheckpointMetadata,
    evaluate_diffusion_loss,
    load_dppo_checkpoint,
    pretrain_diffusion_step,
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


@dataclass(frozen=True)
class PretrainingHistoryRecord:
    """保存一次定期验证对应的真实更新步和损失。"""

    optimizer_step: int
    training_loss: float
    validation_loss: float
    is_best: bool


def write_pretraining_history_csv(
    path: str | Path,
    history: Sequence[PretrainingHistoryRecord],
) -> None:
    """写出便于人工 review 和论文绘图的预训练损失记录。"""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            ("optimizer_step", "training_loss", "validation_loss", "is_best")
        )
        for row in history:
            writer.writerow(
                (
                    row.optimizer_step,
                    row.training_loss,
                    row.validation_loss,
                    row.is_best,
                )
            )


def train_exact_pretraining_steps(
    model: ConditionalDiffusionMLP,
    schedule: CosineNoiseSchedule,
    optimizer: torch.optim.Optimizer,
    train_states: torch.Tensor,
    train_actions: torch.Tensor,
    validation_states: torch.Tensor,
    validation_actions: torch.Tensor,
    *,
    metadata: DPPOCheckpointMetadata,
    checkpoint_path: str | Path,
    optimizer_steps: int,
    validation_interval_steps: int,
    batch_size: int,
    seed: int,
    device: str | torch.device,
    gradient_clip_norm: float | None,
) -> tuple[PretrainingHistoryRecord, ...]:
    """精确训练指定更新次数，并把验证损失最低的模型保存为检查点。"""

    for name, value in (
        ("optimizer_steps", optimizer_steps),
        ("validation_interval_steps", validation_interval_steps),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} 必须是正整数。")

    history: list[PretrainingHistoryRecord] = []
    interval_losses: list[float] = []
    best_validation_loss = math.inf
    # 固定验证噪声，使第 10、20……步的损失只反映模型变化。
    validation_seed = seed + 100000
    for zero_based_step in range(optimizer_steps):
        training_loss = pretrain_diffusion_step(
            model,
            schedule,
            optimizer,
            train_states,
            train_actions,
            optimizer_step=zero_based_step,
            batch_size=batch_size,
            seed=seed,
            device=device,
            gradient_clip_norm=gradient_clip_norm,
        )
        interval_losses.append(training_loss)
        completed_steps = zero_based_step + 1
        should_validate = (
            completed_steps % validation_interval_steps == 0
            or completed_steps == optimizer_steps
        )
        if not should_validate:
            continue

        validation_loss = evaluate_diffusion_loss(
            model,
            schedule,
            validation_states,
            validation_actions,
            batch_size=batch_size,
            seed=validation_seed,
            device=device,
        )
        mean_training_loss = sum(interval_losses) / len(interval_losses)
        interval_losses.clear()
        is_best = validation_loss < best_validation_loss
        if is_best:
            best_validation_loss = validation_loss
            # 检查点 v1 的 epoch 字段暂存零基更新步，避免破坏既有加载接口。
            save_dppo_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                metadata,
                epoch=completed_steps - 1,
            )
        history.append(
            PretrainingHistoryRecord(
                optimizer_step=completed_steps,
                training_loss=mean_training_loss,
                validation_loss=validation_loss,
                is_best=is_best,
            )
        )
        print(
            f"optimizer_step={completed_steps}/{optimizer_steps} "
            f"train_loss={mean_training_loss:.6f} "
            f"validation_loss={validation_loss:.6f} "
            f"is_best={is_best}"
        )
    return tuple(history)


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
        "--optimizer-steps",
        type=int,
        default=None,
        help="覆盖配置中的预训练优化器更新次数。",
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
    optimizer_steps = (
        int(pretraining_config["optimizer_steps"])
        if parsed.optimizer_steps is None
        else parsed.optimizer_steps
    )
    if optimizer_steps <= 0:
        raise ValueError("optimizer_steps 必须是正整数。")
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
    checkpoint_path = Path(parsed.output_root) / "dppo_pretrained.pt"
    history = train_exact_pretraining_steps(
        model,
        schedule,
        optimizer,
        train_states,
        train_actions,
        validation_states,
        validation_actions,
        metadata=checkpoint_metadata,
        checkpoint_path=checkpoint_path,
        optimizer_steps=optimizer_steps,
        validation_interval_steps=int(
            pretraining_config["validation_interval_steps"]
        ),
        batch_size=batch_size,
        seed=base_seed,
        device=device,
        gradient_clip_norm=float(pretraining_config["gradient_clip_norm"]),
    )
    history_path = Path(parsed.output_root) / "pretraining_history.csv"
    write_pretraining_history_csv(history_path, history)
    # 写盘后立即重新加载，确保交付的文件本身可恢复且配置完全一致。
    restored = load_dppo_checkpoint(
        checkpoint_path,
        expected=checkpoint_metadata,
        device=device,
    )
    print(f"检查点：{checkpoint_path.resolve()}")
    best_record = min(history, key=lambda row: row.validation_loss)
    print(f"实际完成 optimizer steps：{optimizer_steps}")
    print(f"最佳 optimizer step：{best_record.optimizer_step}")
    print(f"最佳 validation loss：{best_record.validation_loss:.6f}")
    print(f"恢复 optimizer step：{restored.epoch + 1}")
    print(f"训练记录：{history_path.resolve()}")
    print(f"配置哈希：{restored.metadata.config_hash}")


if __name__ == "__main__":
    main()
