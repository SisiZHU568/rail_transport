"""测试 DPPO 专家仿真数据的记录、切分、持久化与因果收集。"""

from copy import deepcopy
import json

import numpy as np
import pytest

from src.config import load_config
from src.dppo_dataset import (
    EXPERT_DATASET_SCHEMA_VERSION,
    ExpertDatasetMetadata,
    ExpertTransitionRecord,
    collect_expert_records,
    compute_config_hash,
    load_expert_dataset,
    save_expert_dataset,
    split_records_by_episode,
)
from src.dppo_scenario import build_dppo_scenario
from src.dppo_teacher import build_simulation_teacher


def _record(
    seed: int,
    *,
    slow_step: int = 0,
    teacher_proposal_raw_feasible: bool | None = None,
    expert_action_feasible: bool | None = None,
    final_feasible: bool = True,
    reasons: tuple[str, ...] = (),
) -> ExpertTransitionRecord:
    """构造小规模记录，避免持久化单元测试运行真实仿真。"""

    return ExpertTransitionRecord(
        state=np.full(58, seed + slow_step, dtype=np.float32),
        expert_action=np.full(14, seed - slow_step, dtype=np.float32),
        teacher_name="cost",
        episode_seed=seed,
        slow_step=slow_step,
        teacher_proposal_raw_feasible=(
            final_feasible
            if teacher_proposal_raw_feasible is None
            else teacher_proposal_raw_feasible
        ),
        expert_action_feasible=(
            final_feasible
            if expert_action_feasible is None
            else expert_action_feasible
        ),
        projection_change_ratio=0.0 if final_feasible else 0.5,
        final_feasible=final_feasible,
        run_cost=1.0,
        route_cost=2.0,
        cold_start_cost=3.0,
        rejection_reasons=reasons,
    )


def _metadata(config_hash: str = "test-hash") -> ExpertDatasetMetadata:
    """返回与小规模测试数组匹配的版本元数据。"""

    return ExpertDatasetMetadata(
        schema_version=EXPERT_DATASET_SCHEMA_VERSION,
        state_dim=58,
        action_dim=14,
        config_hash=config_hash,
    )


def test_episode_seeds_never_cross_dataset_partitions() -> None:
    """同一个 Episode 的多个慢步不能被拆到不同数据集。"""

    records = [
        _record(seed, slow_step=slow_step)
        for seed in range(6)
        for slow_step in range(2)
    ]

    partitions = split_records_by_episode(
        records,
        fractions=(0.5, 0.25, 0.25),
        seed=7,
    )
    seed_sets = {
        name: {record.episode_seed for record in split}
        for name, split in partitions.items()
    }

    assert seed_sets["train"].isdisjoint(seed_sets["validation"])
    assert seed_sets["train"].isdisjoint(seed_sets["test"])
    assert seed_sets["validation"].isdisjoint(seed_sets["test"])
    assert set.union(*seed_sets.values()) == set(range(6))


def test_three_episodes_keep_all_positive_partitions_nonempty() -> None:
    """三条 smoke Episode 应分别覆盖训练、验证和测试分区。"""

    partitions = split_records_by_episode(
        [_record(seed) for seed in range(3)],
        fractions=(0.70, 0.15, 0.15),
        seed=7,
    )

    assert all(len(records) == 1 for records in partitions.values())


def test_rejected_records_stay_in_diagnostics_not_behavior_cloning(
    tmp_path,
) -> None:
    """最终不可行标签必须可审计，但不能进入扩散模型预训练样本。"""

    accepted = _record(10)
    rejected = _record(
        11,
        final_feasible=False,
        reasons=("sla_violations=1",),
    )
    infeasible_label = _record(
        12,
        teacher_proposal_raw_feasible=False,
        expert_action_feasible=False,
        final_feasible=True,
        reasons=("expert_action_not_raw_feasible",),
    )
    partitions = {
        "train": (accepted, rejected, infeasible_label),
        "validation": (),
        "test": (),
    }

    save_expert_dataset(tmp_path, partitions, _metadata())
    loaded = load_expert_dataset(tmp_path)

    assert loaded.partitions["train"] == (accepted,)
    assert loaded.partitions["validation"] == ()
    assert loaded.partitions["test"] == ()
    assert len(loaded.diagnostics) == 2
    assert loaded.diagnostics[0].record == rejected
    assert loaded.diagnostics[0].partition == "train"
    assert loaded.diagnostics[0].record.rejection_reasons == (
        "sla_violations=1",
    )
    assert loaded.diagnostics[1].record.rejection_reasons == (
        "expert_action_not_raw_feasible",
    )
    assert loaded.diagnostics[1].record == infeasible_label
    assert loaded.diagnostics[1].record.expert_action_feasible is False


def test_save_load_preserves_arrays_and_schema_versions(tmp_path) -> None:
    """NPZ/JSON 往返不能改变 float32 数组、顺序或模式版本。"""

    first = _record(20, slow_step=0)
    second = _record(20, slow_step=1)
    metadata = _metadata(config_hash="stable-config-hash")
    partitions = {
        "train": (first, second),
        "validation": (),
        "test": (),
    }

    save_expert_dataset(tmp_path, partitions, metadata)
    loaded = load_expert_dataset(tmp_path, expected_metadata=metadata)

    assert loaded.metadata == metadata
    assert loaded.metadata.action_schema_version == "joint-sfc-continuous-v2"
    assert loaded.partitions["train"] == (first, second)
    for expected, actual in zip(
        (first, second),
        loaded.partitions["train"],
        strict=True,
    ):
        assert actual.state.dtype == np.float32
        assert actual.expert_action.dtype == np.float32
        np.testing.assert_array_equal(actual.state, expected.state)
        np.testing.assert_array_equal(actual.expert_action, expected.expert_action)
        assert (
            actual.teacher_proposal_raw_feasible
            is expected.teacher_proposal_raw_feasible
        )
        assert actual.expert_action_feasible is expected.expert_action_feasible


def test_save_rejects_record_dimension_mismatch(tmp_path) -> None:
    """错误状态或动作维度必须在写盘前暴露，不能生成损坏数据集。"""

    bad_record = ExpertTransitionRecord(
        state=np.zeros(57, dtype=np.float32),
        expert_action=np.zeros(14, dtype=np.float32),
        teacher_name="cost",
        episode_seed=1,
        slow_step=0,
        teacher_proposal_raw_feasible=True,
        expert_action_feasible=True,
        projection_change_ratio=0.0,
        final_feasible=True,
        run_cost=0.0,
        route_cost=0.0,
        cold_start_cost=0.0,
    )

    with pytest.raises(ValueError, match="state_dim"):
        save_expert_dataset(
            tmp_path,
            {"train": (bad_record,), "validation": (), "test": ()},
            _metadata(),
        )


def test_load_rejects_arrays_that_disagree_with_metadata(tmp_path) -> None:
    """元数据被误改后，加载器必须发现数组维度不再匹配。"""

    partitions = {
        "train": (_record(1),),
        "validation": (),
        "test": (),
    }
    save_expert_dataset(tmp_path, partitions, _metadata())
    metadata_path = tmp_path / "metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["state_dim"] = 59
    metadata_path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="state_dim"):
        load_expert_dataset(tmp_path)


def test_config_hash_is_deterministic_and_sensitive_to_configuration() -> None:
    """键顺序不能改变哈希，但实验参数变化必须改变哈希。"""

    first = {"b": {"x": 2}, "a": [1, 2]}
    reordered = {"a": [1, 2], "b": {"x": 2}}
    changed = {"a": [1, 3], "b": {"x": 2}}

    assert compute_config_hash(first) == compute_config_hash(reordered)
    assert compute_config_hash(first) != compute_config_hash(changed)


def test_collect_one_cost_teacher_step_uses_online_environment() -> None:
    """收集器应输出真实投影和快层执行后的单步诊断，而不是伪造标签。"""

    config = deepcopy(load_config("configs/debug.yaml"))
    config["dppo"]["dataset"]["teacher_names"] = ["cost"]
    config["dppo"]["dataset"]["max_slow_steps_per_episode"] = 1

    records = collect_expert_records(
        config,
        episode_count=1,
        seed_start=7100,
    )

    assert len(records) == 1
    record = records[0]
    assert record.episode_seed == 7100
    assert record.slow_step == 0
    assert record.teacher_name == "cost"
    assert record.state.shape == (78,)
    assert record.expert_action.shape == (27,)
    assert np.isfinite(record.state).all()
    assert np.isfinite(record.expert_action).all()


def test_collector_reencodes_projected_intent_as_feasible_v2_label() -> None:
    """教师原建议可不可行与最终监督标签是否可行必须分开记录。"""

    config = deepcopy(load_config("configs/debug.yaml"))
    config["dppo"]["dataset"]["teacher_names"] = ["cost"]
    config["dppo"]["dataset"]["max_slow_steps_per_episode"] = 2

    records = collect_expert_records(
        config,
        episode_count=1,
        seed_start=42000,
    )

    accepted = next(record for record in records if record.final_feasible)
    assert EXPERT_DATASET_SCHEMA_VERSION == "dppo-expert-v2"
    assert accepted.teacher_proposal_raw_feasible is False
    assert accepted.expert_action_feasible is True
    assert accepted.projection_change_ratio > 0.0

    # 用同一 seed 重放到该慢步，直接核对标签的排名语义：投影选中的节点
    # 必须位于最前面，其余节点继续保持教师原始顺序。
    scenario = build_dppo_scenario(config)
    scenario.reset(seed=42000)
    teacher = build_simulation_teacher("cost", scenario)
    for slow_step in range(accepted.slow_step + 1):
        proposal = teacher.propose(scenario.current_public_snapshot())
        _, _, _, _, info = scenario.step(proposal.relaxed_action)
        if slow_step != accepted.slow_step:
            continue
        projection = info["projection_result"]
        corrected = scenario.action_space.decode(accepted.expert_action)
        original_by_function = {
            action.function_id: action
            for action in proposal.decoded_action.function_actions
        }
        corrected_by_function = {
            action.function_id: action for action in corrected.function_actions
        }
        for intent in projection.function_intents:
            selected = tuple(intent.preferred_node_ids)
            corrected_ranking = corrected_by_function[intent.function_id].ranked_node_ids
            assert corrected_ranking[: len(selected)] == selected
            assert corrected_ranking[len(selected) :] == tuple(
                node_id
                for node_id in original_by_function[
                    intent.function_id
                ].ranked_node_ids
                if node_id not in set(selected)
            )


def test_dataset_cli_requires_explicit_output_root() -> None:
    """命令行不得把测试数据静默写进正式结果目录。"""

    from run_dppo_dataset_generation import parse_arguments

    with pytest.raises(SystemExit):
        parse_arguments(["--episodes", "1"])

    arguments = parse_arguments(
        [
            "--episodes",
            "3",
            "--output-root",
            "temporary-output",
            "--seed-start",
            "7000",
        ]
    )
    assert arguments.episodes == 3
    assert arguments.output_root == "temporary-output"
    assert arguments.seed_start == 7000
