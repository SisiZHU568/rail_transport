"""从配置化仿真教师生成可审计的 DPPO 专家数据集。"""

import argparse
from pathlib import Path
from typing import Sequence

from src.config import load_config
from src.dppo_dataset import (
    EXPERT_DATASET_SCHEMA_VERSION,
    ExpertDatasetMetadata,
    collect_expert_records,
    compute_config_hash,
    save_expert_dataset,
    split_records_by_episode,
)


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行参数，并强制调用方明确指定输出目录。"""

    parser = argparse.ArgumentParser(
        description="生成按 Episode 隔离切分的 DPPO 专家仿真数据。"
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "configs" / "debug.yaml"),
        help="YAML 配置文件路径。",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="覆盖配置中的 Episode 数量。",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="数据集输出目录；必须显式提供，避免测试覆盖正式结果。",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=None,
        help="覆盖配置中的第一个 Episode 随机种子。",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    """收集教师记录、按 Episode 切分并写入 NPZ 与 JSON。"""

    parsed = parse_arguments(arguments)
    config = load_config(parsed.config)
    dataset_config = config["dppo"]["dataset"]
    episode_count = (
        int(dataset_config["episodes"])
        if parsed.episodes is None
        else parsed.episodes
    )
    seed_start = (
        int(dataset_config["seed_start"])
        if parsed.seed_start is None
        else parsed.seed_start
    )
    records = collect_expert_records(
        config,
        episode_count=episode_count,
        seed_start=seed_start,
    )
    if not records:
        raise RuntimeError("数据收集没有产生任何记录。")
    partitions = split_records_by_episode(
        records,
        fractions=(
            float(dataset_config["train_fraction"]),
            float(dataset_config["validation_fraction"]),
            float(dataset_config["test_fraction"]),
        ),
        seed=int(dataset_config["split_seed"]),
    )
    metadata = ExpertDatasetMetadata(
        schema_version=EXPERT_DATASET_SCHEMA_VERSION,
        state_dim=int(records[0].state.shape[0]),
        action_dim=int(records[0].expert_action.shape[0]),
        config_hash=compute_config_hash(config),
        action_schema_version=str(config["dppo"]["action"]["schema_version"]),
    )
    save_expert_dataset(parsed.output_root, partitions, metadata)

    accepted_counts = {
        name: sum(record.final_feasible for record in split)
        for name, split in partitions.items()
    }
    diagnostic_count = sum(not record.final_feasible for record in records)
    print("DPPO 专家数据集生成完成")
    print(f"输出目录：{Path(parsed.output_root).resolve()}")
    print(f"Episode 数量：{episode_count}")
    print(f"慢步记录数：{len(records)}")
    print(f"行为克隆样本：{accepted_counts}")
    print(f"诊断记录数：{diagnostic_count}")
    print(f"配置哈希：{metadata.config_hash}")


if __name__ == "__main__":
    main()
