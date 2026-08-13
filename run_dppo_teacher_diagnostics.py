"""重放 held-out 专家标签并比较三类仿真教师的执行效果。"""

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.config import load_config
from src.dppo_dataset import (
    EXPERT_DATASET_SCHEMA_VERSION,
    ExpertDatasetMetadata,
    ExpertTransitionRecord,
    compute_config_hash,
    load_expert_dataset,
)
from src.dppo_scenario import build_dppo_scenario
from src.dppo_teacher import build_simulation_teacher
from src.dppo_teacher_diagnostics import (
    TeacherTransitionMetric,
    summarize_teacher_candidate_coverage,
    summarize_teacher_metrics,
    write_teacher_diagnostic_outputs,
)


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="比较 cost、reliability、balanced 三类专家标签的 held-out 执行效果。"
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "configs" / "debug.yaml"),
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args(arguments)


def _rejection_reasons(info: dict[str, object]) -> tuple[str, ...]:
    explicit = info.get("rejection_reasons", ())
    if explicit:
        return tuple(str(reason) for reason in explicit)
    reasons: list[str] = []
    if not bool(info["projection_success"]):
        reasons.append("projection_rejected")
    failures = int(info["fast_repair_failures"])
    if failures > 0:
        reasons.append(f"fast_repair_failures={failures}")
    return tuple(reasons)


def replay_expert_transition(
    config: dict[str, Any],
    partition: str,
    record: ExpertTransitionRecord,
) -> TeacherTransitionMetric:
    """从原教师轨迹独立推进到目标慢步，再执行保存的专家动作。"""

    if partition not in {"validation", "test"}:
        raise ValueError("partition 只能是 validation 或 test。")
    scenario = build_dppo_scenario(config)
    teacher = build_simulation_teacher(record.teacher_name, scenario)
    state = scenario.reset(seed=record.episode_seed)
    for slow_step in range(record.slow_step):
        proposal = teacher.propose(scenario.current_public_snapshot())
        state, _, terminated, truncated, _ = scenario.step(proposal.relaxed_action)
        if terminated or truncated:
            raise ValueError(
                f"episode_seed={record.episode_seed} 在目标 slow_step 前提前结束。"
            )
    if not np.allclose(state, record.state, rtol=1e-6, atol=1e-6):
        raise ValueError(
            f"episode_seed={record.episode_seed}, slow_step={record.slow_step} "
            "的重放状态不一致。"
        )
    _, reward, _, _, info = scenario.step(record.expert_action)
    solver_ran = str(info["fast_solver_status"]) != "not_run"
    solver_time = float(info["fast_solver_time_seconds"]) if solver_ran else 0.0
    return TeacherTransitionMetric(
        partition=partition,
        teacher_name=record.teacher_name,
        episode_seed=record.episode_seed,
        slow_step=record.slow_step,
        reward=float(reward),
        raw_feasible=bool(info["raw_feasible"]),
        projection_change_ratio=float(info["projection_change_ratio"]),
        projection_rejected=not bool(info["projection_success"]),
        repair_attempts=int(info["fast_repair_attempts"]),
        repair_successes=int(info["fast_repair_successes"]),
        repair_failures=int(info["fast_repair_failures"]),
        fast_solver_ran=solver_ran,
        fast_solver_time_seconds=solver_time,
        rejection_reasons=_rejection_reasons(info),
    )


def run_teacher_diagnostics(
    *,
    config: dict[str, Any],
    dataset_root: str | Path,
    output_root: str | Path,
) -> dict[str, object]:
    """校验数据身份，重放 held-out 标签并写出可审计结果。"""

    scenario = build_dppo_scenario(config)
    expected_metadata = ExpertDatasetMetadata(
        schema_version=EXPERT_DATASET_SCHEMA_VERSION,
        state_dim=scenario.dimensions.state_dim,
        action_dim=scenario.dimensions.action_dim,
        config_hash=compute_config_hash(config),
        state_schema_version=str(config["dppo"]["training"]["state_schema_version"]),
        action_schema_version=str(config["dppo"]["action"]["schema_version"]),
    )
    dataset = load_expert_dataset(dataset_root, expected_metadata=expected_metadata)
    rows = tuple(
        replay_expert_transition(config, partition, record)
        for partition in ("validation", "test")
        for record in dataset.partitions[partition]
    )
    if not rows:
        raise ValueError("validation 和 test 分区没有可诊断的专家标签。")
    summary = summarize_teacher_metrics(rows, minimum_sample_count=3)
    held_out_diagnostics = tuple(
        item.record
        for item in dataset.diagnostics
        if item.partition in {"validation", "test"}
    )
    accepted_records = tuple(
        record
        for partition in ("validation", "test")
        for record in dataset.partitions[partition]
    )
    summary.update(
        {
            "config_hash": expected_metadata.config_hash,
            "dataset_schema_version": expected_metadata.schema_version,
            "partition_counts": {
                "validation": len(dataset.partitions["validation"]),
                "test": len(dataset.partitions["test"]),
            },
            "candidate_coverage": summarize_teacher_candidate_coverage(
                accepted_records,
                held_out_diagnostics,
            ),
        }
    )
    write_teacher_diagnostic_outputs(output_root, rows, summary)
    return summary


def main(arguments: Sequence[str] | None = None) -> None:
    parsed = parse_arguments(arguments)
    summary = run_teacher_diagnostics(
        config=load_config(parsed.config),
        dataset_root=parsed.dataset_root,
        output_root=parsed.output_root,
    )
    print(json.dumps(summary["teachers"], ensure_ascii=False, indent=2))
    print(f"输出目录：{Path(parsed.output_root).resolve()}")


if __name__ == "__main__":
    main()
