"""只读比较随机扩散模型和预训练扩散模型的专家动作拟合能力。"""

import argparse
import csv
from dataclasses import asdict, dataclass, fields
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.dppo_action_space import DPPOActionSpace
from src.config import load_config
from src.dppo_checkpoint import load_dppo_checkpoint, resolve_torch_device
from src.dppo_dataset import (
    EXPERT_DATASET_SCHEMA_VERSION,
    ExpertDatasetMetadata,
    ExpertTransitionRecord,
    compute_config_hash,
    load_expert_dataset,
)
from src.dppo_diffusion import (
    ConditionalDiffusionMLP,
    CosineNoiseSchedule,
    sample_denoising_chain,
)
from src.dppo_scenario import build_dppo_scenario
from src.dppo_teacher import build_simulation_teacher
from src.dppo_pretraining_diagnostics import (
    GeneratedActionMetric,
    evaluate_generated_action,
    summarize_diagnostic_rows,
)
from src.dppo_stability import sha256_file
from src.dppo_training_config import load_dppo_stability_settings
from src.slow_timescale_execution_core import ProjectionInputs
from run_dppo_pretraining import _checkpoint_metadata


@dataclass(frozen=True)
class ReplayDiagnosticContext:
    """保存专家记录对应的执行前状态与投影资源快照。"""

    partition: str
    record: ExpertTransitionRecord
    state: np.ndarray
    projection_inputs: ProjectionInputs


@dataclass(frozen=True)
class FairActionPair:
    """保存两个模型在完全相同采样种子下生成的最终动作。"""

    random_action: np.ndarray
    pretrained_action: np.ndarray


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """解析只读诊断参数，强制显式指定输入和输出目录。"""

    parser = argparse.ArgumentParser(
        description="比较随机模型与预训练 DPPO 扩散模型的专家动作拟合能力。"
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "configs" / "debug.yaml"),
        help="YAML 配置文件路径。",
    )
    parser.add_argument("--dataset-root", required=True, help="v2 专家数据集目录。")
    parser.add_argument(
        "--pretrained-checkpoint",
        required=True,
        help="待诊断的 DPPO 预训练检查点。",
    )
    parser.add_argument(
        "--output-root", required=True, help="诊断 CSV 和 JSON 输出目录。"
    )
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cuda", help="模型采样设备。"
    )
    parser.add_argument(
        "--seed", type=int, default=73000, help="固定诊断采样基础种子。"
    )
    return parser.parse_args(arguments)


def replay_diagnostic_contexts(
    config: dict[str, object],
    partitioned_records: Mapping[str, tuple[ExpertTransitionRecord, ...]],
) -> tuple[ReplayDiagnosticContext, ...]:
    """按 Episode 重放教师轨迹，恢复每条标签生成前的真实资源状态。"""

    normalized: list[tuple[str, ExpertTransitionRecord]] = []
    for partition, records in partitioned_records.items():
        if partition not in {"validation", "test"}:
            raise ValueError("partition 必须是 validation 或 test。")
        normalized.extend((partition, record) for record in records)
    grouped: dict[tuple[str, int], list[ExpertTransitionRecord]] = {}
    for partition, record in normalized:
        if not isinstance(record, ExpertTransitionRecord):
            raise TypeError("records 必须包含 ExpertTransitionRecord。")
        grouped.setdefault((partition, record.episode_seed), []).append(record)

    contexts: list[ReplayDiagnosticContext] = []
    for (partition, episode_seed), records in sorted(grouped.items()):
        ordered = sorted(records, key=lambda item: item.slow_step)
        if len({record.teacher_name for record in ordered}) != 1:
            raise ValueError(f"episode_seed={episode_seed} 的 teacher_name 不一致。")
        scenario = build_dppo_scenario(config)
        teacher = build_simulation_teacher(ordered[0].teacher_name, scenario)
        state = scenario.reset(seed=episode_seed)
        target_by_step = {record.slow_step: record for record in ordered}
        if len(target_by_step) != len(ordered):
            raise ValueError(f"episode_seed={episode_seed} 包含重复 slow_step。")
        maximum_step = max(target_by_step)
        for slow_step in range(maximum_step + 1):
            record = target_by_step.get(slow_step)
            if record is not None:
                if not np.allclose(
                    state,
                    record.state,
                    rtol=1e-6,
                    atol=1e-6,
                ):
                    raise ValueError(
                        f"episode_seed={episode_seed}, slow_step={slow_step} 的重放状态不一致。"
                    )
                # 必须在执行教师动作前保存快照，否则会读到未来资源状态。
                projection_inputs = (
                    scenario.execution_core.current_projection_inputs()
                )
                state_copy = np.array(state, dtype=np.float32, copy=True)
                state_copy.setflags(write=False)
                contexts.append(
                    ReplayDiagnosticContext(
                        partition=partition,
                        record=record,
                        state=state_copy,
                        projection_inputs=projection_inputs,
                    )
                )
            proposal = teacher.propose(scenario.current_public_snapshot())
            next_state, _, terminated, truncated, _ = scenario.step(
                proposal.relaxed_action
            )
            state = next_state
            if (terminated or truncated) and slow_step < maximum_step:
                raise ValueError(
                    f"episode_seed={episode_seed} 在 slow_step={slow_step} 提前结束。"
                )
    return tuple(
        sorted(
            contexts,
            key=lambda item: (
                item.partition,
                item.record.episode_seed,
                item.record.slow_step,
            ),
        )
    )


def sample_fair_actions(
    random_model: ConditionalDiffusionMLP,
    pretrained_model: ConditionalDiffusionMLP,
    schedule: CosineNoiseSchedule,
    state: np.ndarray,
    *,
    seed: int,
    minimum_sampling_standard_deviation: float,
    action_space: DPPOActionSpace,
    device: str | torch.device,
) -> FairActionPair:
    """使用同一初始噪声和逐步噪声公平采样两个同结构模型。"""

    resolved_device = torch.device(device)
    if random_model.state_dim != pretrained_model.state_dim:
        raise ValueError("两个模型的 state_dim 必须一致。")
    if random_model.action_dim != pretrained_model.action_dim:
        raise ValueError("两个模型的 action_dim 必须一致。")
    if action_space.action_dim != random_model.action_dim:
        raise ValueError("action_space 与模型 action_dim 不一致。")
    # 重放状态故意设为只读；先复制再交给 PyTorch，避免共享只读 NumPy 内存。
    writable_state = np.array(state, dtype=np.float32, copy=True)
    states = torch.as_tensor(
        writable_state,
        dtype=next(random_model.parameters()).dtype,
        device=resolved_device,
    ).reshape(1, -1)
    random_was_training = random_model.training
    pretrained_was_training = pretrained_model.training
    random_model.to(resolved_device).eval()
    pretrained_model.to(resolved_device).eval()
    try:
        random_sample = sample_denoising_chain(
            random_model,
            schedule,
            states,
            seed=seed,
            minimum_sampling_standard_deviation=(
                minimum_sampling_standard_deviation
            ),
        )
        pretrained_sample = sample_denoising_chain(
            pretrained_model,
            schedule,
            states,
            seed=seed,
            minimum_sampling_standard_deviation=(
                minimum_sampling_standard_deviation
            ),
        )
    finally:
        random_model.train(random_was_training)
        pretrained_model.train(pretrained_was_training)
    random_action = action_space.clip(
        random_sample.actions[-1, 0].detach().cpu().numpy()
    )
    pretrained_action = action_space.clip(
        pretrained_sample.actions[-1, 0].detach().cpu().numpy()
    )
    return FairActionPair(
        random_action=random_action,
        pretrained_action=pretrained_action,
    )


def _atomic_write_text(path: Path, text: str) -> None:
    """先在同目录写完并同步临时文件，再原子替换目标文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _csv_text(rows: tuple[GeneratedActionMetric, ...]) -> str:
    field_names = [field.name for field in fields(GeneratedActionMetric)]
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=field_names, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        payload = asdict(row)
        payload["projection_reasons"] = json.dumps(
            payload["projection_reasons"], ensure_ascii=False, separators=(",", ":")
        )
        payload["generated_replica_counts"] = json.dumps(
            payload["generated_replica_counts"], separators=(",", ":")
        )
        writer.writerow(payload)
    return stream.getvalue()


def write_diagnostic_outputs(
    output_root: str | Path,
    rows: tuple[GeneratedActionMetric, ...],
    summary: Mapping[str, object],
) -> None:
    """以稳定 UTF-8 格式写出逐记录 CSV 和严格 JSON。"""

    normalized_rows = tuple(rows)
    if not normalized_rows:
        raise ValueError("rows 不能为空。")
    json_text = json.dumps(
        dict(summary),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    root = Path(output_root)
    _atomic_write_text(root / "per_record_metrics.csv", _csv_text(normalized_rows))
    _atomic_write_text(root / "summary.json", json_text)


def _diagnostic_seed(base_seed: int, record: ExpertTransitionRecord) -> int:
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("诊断 seed 必须是非负整数。")
    value = base_seed + record.episode_seed * 1000 + record.slow_step
    if value > torch.iinfo(torch.int64).max:
        raise OverflowError("诊断采样 seed 超出 PyTorch int64 范围。")
    return value


def diagnostic_sampling_min_std(config: Mapping[str, Any]) -> float:
    """通过项目唯一的稳定性配置入口读取评估采样标准差下限。"""

    return load_dppo_stability_settings(config).evaluation_sampling_min_std


def run_diagnostic(
    *,
    config: dict[str, Any],
    dataset_root: str | Path,
    pretrained_checkpoint: str | Path,
    output_root: str | Path,
    device: str,
    seed: int,
) -> dict[str, object]:
    """执行兼容性门禁、因果重放、公平采样和双指标汇总。"""

    resolved_device = resolve_torch_device(device)
    scenario = build_dppo_scenario(config)
    config_hash = compute_config_hash(config)
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
        dataset_root,
        expected_metadata=expected_dataset_metadata,
    )
    selected = {
        name: dataset.partitions[name] for name in ("validation", "test")
    }
    if any(not records for records in selected.values()):
        raise ValueError("validation 和 test 分区都必须包含可行专家记录。")
    if any(
        not (record.expert_action_feasible and record.final_feasible)
        for records in selected.values()
        for record in records
    ):
        raise ValueError("诊断分区包含不可用于行为克隆的记录。")

    restored = load_dppo_checkpoint(
        pretrained_checkpoint,
        expected=checkpoint_metadata,
        device=resolved_device,
    )
    pretrained_model = restored.model
    with torch.random.fork_rng(
        devices=[resolved_device] if resolved_device.type == "cuda" else []
    ):
        torch.manual_seed(int(config["dppo"]["pretraining"]["seed"]))
        random_model = ConditionalDiffusionMLP(
            checkpoint_metadata.state_dim,
            checkpoint_metadata.action_dim,
            pretrained_model.hidden_dims,
        ).to(resolved_device)
    schedule = CosineNoiseSchedule(checkpoint_metadata.diffusion_steps)
    minimum_std = diagnostic_sampling_min_std(config)
    contexts = replay_diagnostic_contexts(config, selected)
    rows: list[GeneratedActionMetric] = []
    for context in contexts:
        pair = sample_fair_actions(
            random_model,
            pretrained_model,
            schedule,
            context.state,
            seed=_diagnostic_seed(seed, context.record),
            minimum_sampling_standard_deviation=minimum_std,
            action_space=scenario.action_space,
            device=resolved_device,
        )
        for model_name, action in (
            ("random", pair.random_action),
            ("pretrained", pair.pretrained_action),
        ):
            inputs = context.projection_inputs
            projection = scenario.projector.project(
                decoded_action=scenario.action_space.decode(action),
                operational_node_ids=inputs.operational_node_ids,
                free_cpu=inputs.free_cpu,
                free_memory_mb=inputs.free_memory_mb,
                fault_domains=inputs.fault_domains,
            )
            rows.append(
                evaluate_generated_action(
                    partition=context.partition,
                    model_name=model_name,
                    episode_seed=context.record.episode_seed,
                    slow_step=context.record.slow_step,
                    teacher_name=context.record.teacher_name,
                    generated_action=action,
                    expert_action=context.record.expert_action,
                    action_space=scenario.action_space,
                    projection_result=projection,
                )
            )
    row_tuple = tuple(rows)
    summary = summarize_diagnostic_rows(row_tuple)
    summary.update(
        {
            "schema_version": "dppo-pretraining-diagnostics-v1",
            "diagnostic_seed": seed,
            "device": str(resolved_device),
            "dataset_schema_version": dataset.metadata.schema_version,
            "state_schema_version": dataset.metadata.state_schema_version,
            "action_schema_version": dataset.metadata.action_schema_version,
            "config_hash": config_hash,
            "pretrained_checkpoint_sha256": sha256_file(pretrained_checkpoint),
            "partition_counts": {
                name: len(records) for name, records in selected.items()
            },
        }
    )
    write_diagnostic_outputs(output_root, row_tuple, summary)
    return summary


def main(arguments: Sequence[str] | None = None) -> None:
    parsed = parse_arguments(arguments)
    summary = run_diagnostic(
        config=load_config(parsed.config),
        dataset_root=parsed.dataset_root,
        pretrained_checkpoint=parsed.pretrained_checkpoint,
        output_root=parsed.output_root,
        device=parsed.device,
        seed=parsed.seed,
    )
    random_summary = summary["models"]["random"]
    pretrained_summary = summary["models"]["pretrained"]
    print(f"record_count={summary['record_count']}")
    print(
        "mean_action_mse="
        f"random:{random_summary['mean_action_mse']:.6f},"
        f"pretrained:{pretrained_summary['mean_action_mse']:.6f}"
    )
    print(
        "raw_feasibility_rate="
        f"random:{random_summary['raw_feasibility_rate']:.6f},"
        f"pretrained:{pretrained_summary['raw_feasibility_rate']:.6f}"
    )
    print(f"conclusion={summary['conclusion']}")


if __name__ == "__main__":
    main()
