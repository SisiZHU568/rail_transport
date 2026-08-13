"""测试 DPPO 预训练诊断的动作拟合、语义和可行性指标。"""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from run_dppo_pretraining_diagnostics import (
    diagnostic_sampling_min_std,
    parse_arguments,
    replay_diagnostic_contexts,
    sample_fair_actions,
    write_diagnostic_outputs,
)
from src.config import load_config
from src.dppo_action_space import DPPOActionSpace, DecodedFunctionAction
from src.dppo_dataset import collect_expert_records
from src.dppo_diffusion import ConditionalDiffusionMLP, CosineNoiseSchedule
from src.dppo_pretraining_diagnostics import (
    GeneratedActionMetric,
    evaluate_generated_action,
    summarize_diagnostic_rows,
)
from src.dppo_projection import ProjectionResult
from src.scenario_dimensions import ScenarioDimensions


def _action_space() -> DPPOActionSpace:
    dimensions = ScenarioDimensions((0, 1, 2), 3, (10, 20, 30))
    return DPPOActionSpace(
        dimensions,
        maximum_retention_seconds=20.0,
        minimum_replicas=2,
        maximum_replicas=3,
    )


def _encoded_actions() -> tuple[DPPOActionSpace, np.ndarray, np.ndarray]:
    space = _action_space()
    expert = space.encode_teacher_action(
        (
            DecodedFunctionAction(10, (0, 1, 2, 3), 2, 10.0, 5.0),
            DecodedFunctionAction(20, (1, 0, 2, 3), 3, 12.0, 6.0),
            DecodedFunctionAction(30, (2, 1, 0, 3), 2, 14.0, 7.0),
        )
    )
    generated = space.encode_teacher_action(
        (
            # 第一个 VNF 的副本数、节点前缀和顺序全部正确。
            DecodedFunctionAction(10, (0, 1, 3, 2), 2, 11.0, 4.0),
            # 第二个 VNF 的节点集合相同但前缀顺序不同。
            DecodedFunctionAction(20, (0, 1, 2, 3), 3, 10.0, 9.0),
            # 第三个 VNF 同时使用不同节点集合和不同副本数。
            DecodedFunctionAction(30, (3, 2, 1, 0), 3, 18.0, 3.0),
        )
    )
    return space, expert, generated


def _projection_result() -> ProjectionResult:
    return ProjectionResult(
        function_intents=(),
        raw_feasible=False,
        success=True,
        reasons=(),
        changed_assignment_count=1,
        requested_assignment_count=6,
        change_ratio=1.0 / 6.0,
        fault_domains={},
    )


def test_evaluate_generated_action_reports_continuous_and_decoded_metrics() -> None:
    space, expert, generated = _encoded_actions()

    row = evaluate_generated_action(
        partition="validation",
        model_name="pretrained",
        episode_seed=60000,
        slow_step=1,
        teacher_name="cost",
        generated_action=generated,
        expert_action=expert,
        action_space=space,
        projection_result=_projection_result(),
    )

    assert row.action_mse == pytest.approx(
        np.mean((generated.astype(np.float64) - expert.astype(np.float64)) ** 2)
    )
    assert row.action_mae == pytest.approx(
        np.mean(np.abs(generated.astype(np.float64) - expert.astype(np.float64)))
    )
    assert row.replica_accuracy == pytest.approx(2.0 / 3.0)
    assert row.node_prefix_exact_rate == pytest.approx(1.0 / 3.0)
    assert row.node_prefix_set_rate == pytest.approx(2.0 / 3.0)
    assert row.primary_retention_mae_seconds == pytest.approx(7.0 / 3.0)
    assert row.backup_retention_mae_seconds == pytest.approx(8.0 / 3.0)
    assert row.raw_feasible is False
    assert row.projection_success is True
    assert row.projection_change_ratio == pytest.approx(1.0 / 6.0)
    assert row.generated_replica_counts == (2, 3, 3)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("partition", "train", "partition"),
        ("model_name", "online", "model_name"),
        ("generated_action", np.asarray([np.nan]), "generated_action"),
    ],
)
def test_evaluate_generated_action_rejects_invalid_inputs(
    field: str,
    value: object,
    message: str,
) -> None:
    space, expert, generated = _encoded_actions()
    arguments = {
        "partition": "test",
        "model_name": "random",
        "episode_seed": 60001,
        "slow_step": 0,
        "teacher_name": "reliability",
        "generated_action": generated,
        "expert_action": expert,
        "action_space": space,
        "projection_result": _projection_result(),
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=message):
        evaluate_generated_action(**arguments)


def _metric(
    model_name: str,
    seed: int,
    *,
    mse: float,
    replica_accuracy: float = 0.5,
    prefix_rate: float = 0.5,
    raw_feasible: bool = False,
) -> GeneratedActionMetric:
    return GeneratedActionMetric(
        partition="validation" if seed == 1 else "test",
        model_name=model_name,
        episode_seed=seed,
        slow_step=0,
        teacher_name="cost",
        action_mse=mse,
        action_mae=mse / 2.0,
        replica_accuracy=replica_accuracy,
        node_prefix_exact_rate=prefix_rate,
        node_prefix_set_rate=prefix_rate,
        primary_retention_mae_seconds=2.0,
        backup_retention_mae_seconds=3.0,
        raw_feasible=raw_feasible,
        projection_success=True,
        projection_change_ratio=0.25,
        projection_reasons=(),
        generated_replica_counts=(2, 3),
    )


def test_summary_pairs_models_and_does_not_overclaim_weak_improvement() -> None:
    rows = (
        _metric("random", 1, mse=0.4),
        _metric("pretrained", 1, mse=0.3),
        _metric("random", 2, mse=0.6),
        _metric("pretrained", 2, mse=0.5),
    )

    summary = summarize_diagnostic_rows(rows)

    assert summary["record_count"] == 2
    assert summary["models"]["random"]["mean_action_mse"] == pytest.approx(0.5)
    assert summary["models"]["pretrained"]["mean_action_mse"] == pytest.approx(0.4)
    assert summary["comparison"]["relative_mean_mse_reduction"] == pytest.approx(0.2)
    assert summary["comparison"]["record_mse_improvement_rate"] == pytest.approx(1.0)
    # 连续误差虽然下降，但部署语义没有改善，不能宣称已经学会可行部署。
    assert summary["conclusion"] == "continuous_only_mapping_problem"


def test_summary_reports_continuous_and_deployment_signal_together() -> None:
    rows = (
        _metric("random", 1, mse=0.5, replica_accuracy=0.3),
        _metric(
            "pretrained",
            1,
            mse=0.3,
            replica_accuracy=0.8,
            raw_feasible=True,
        ),
        _metric("random", 2, mse=0.7, replica_accuracy=0.4),
        _metric("pretrained", 2, mse=0.5, replica_accuracy=0.7),
    )

    summary = summarize_diagnostic_rows(rows)

    assert summary["comparison"]["replica_accuracy_delta"] == pytest.approx(0.4)
    assert summary["comparison"]["raw_feasibility_rate_delta"] == pytest.approx(0.5)
    assert summary["conclusion"] == "continuous_and_deployment_signal_learned"


def test_summary_rejects_unpaired_model_rows() -> None:
    with pytest.raises(ValueError, match="成对"):
        summarize_diagnostic_rows((_metric("random", 1, mse=0.5),))


def test_replay_recovers_pre_execution_state_and_projection_inputs() -> None:
    config = deepcopy(load_config("configs/debug.yaml"))
    config["dppo"]["dataset"]["teacher_names"] = ["cost"]
    config["dppo"]["dataset"]["max_slow_steps_per_episode"] = 2
    records = collect_expert_records(config, episode_count=1, seed_start=42000)

    contexts = replay_diagnostic_contexts(config, {"validation": records})

    assert len(contexts) == len(records)
    for context, record in zip(contexts, records, strict=True):
        assert context.partition == "validation"
        assert context.record == record
        np.testing.assert_allclose(
            context.state,
            record.state,
            rtol=1e-6,
            atol=1e-6,
        )
        assert context.projection_inputs.operational_node_ids


def test_replay_rejects_a_saved_state_that_cannot_be_reproduced() -> None:
    config = deepcopy(load_config("configs/debug.yaml"))
    config["dppo"]["dataset"]["teacher_names"] = ["cost"]
    config["dppo"]["dataset"]["max_slow_steps_per_episode"] = 1
    record = collect_expert_records(config, episode_count=1, seed_start=42001)[0]
    changed_state = record.state.copy()
    changed_state[0] += 0.1
    changed = replace(record, state=changed_state)

    with pytest.raises(ValueError, match="42001.*slow_step=0"):
        replay_diagnostic_contexts(config, {"test": (changed,)})


def test_fair_sampling_is_reproducible_and_equal_for_identical_models() -> None:
    space = _action_space()
    torch.manual_seed(91)
    random_model = ConditionalDiffusionMLP(
        state_dim=5,
        action_dim=space.action_dim,
        hidden_dims=(8,),
    )
    pretrained_model = deepcopy(random_model)
    schedule = CosineNoiseSchedule(steps=3)
    states = np.linspace(-1.0, 1.0, 5, dtype=np.float32)

    first = sample_fair_actions(
        random_model,
        pretrained_model,
        schedule,
        states,
        seed=73001,
        minimum_sampling_standard_deviation=0.001,
        action_space=space,
        device="cpu",
    )
    second = sample_fair_actions(
        random_model,
        pretrained_model,
        schedule,
        states,
        seed=73001,
        minimum_sampling_standard_deviation=0.001,
        action_space=space,
        device="cpu",
    )

    np.testing.assert_array_equal(first.random_action, first.pretrained_action)
    np.testing.assert_array_equal(first.random_action, second.random_action)
    np.testing.assert_array_equal(first.pretrained_action, second.pretrained_action)
    assert first.random_action.shape == (space.action_dim,)
    assert np.isfinite(first.random_action).all()
    assert np.max(np.abs(first.random_action)) <= 1.0


def test_fair_sampling_accepts_read_only_replay_state_without_warning(
    recwarn,
) -> None:
    space = _action_space()
    model = ConditionalDiffusionMLP(5, space.action_dim, (8,))
    state = np.zeros(5, dtype=np.float32)
    state.setflags(write=False)

    sample_fair_actions(
        model,
        deepcopy(model),
        CosineNoiseSchedule(2),
        state,
        seed=73002,
        minimum_sampling_standard_deviation=0.001,
        action_space=space,
        device="cpu",
    )

    assert not recwarn.list


def test_diagnostic_cli_requires_dataset_checkpoint_and_output() -> None:
    with pytest.raises(SystemExit):
        parse_arguments([])
    with pytest.raises(SystemExit):
        parse_arguments(["--dataset-root", "dataset"])

    arguments = parse_arguments(
        [
            "--dataset-root",
            "dataset",
            "--pretrained-checkpoint",
            "model.pt",
            "--output-root",
            "output",
            "--device",
            "cuda",
            "--seed",
            "73000",
        ]
    )

    assert arguments.dataset_root == "dataset"
    assert arguments.pretrained_checkpoint == "model.pt"
    assert arguments.output_root == "output"
    assert arguments.device == "cuda"
    assert arguments.seed == 73000


def test_diagnostic_sampling_floor_uses_stability_configuration() -> None:
    config = deepcopy(load_config("configs/debug.yaml"))
    config["dppo"]["stability"]["evaluation_sampling_min_std"] = 0.0025

    assert diagnostic_sampling_min_std(config) == pytest.approx(0.0025)


def test_output_writer_uses_stable_csv_and_strict_json(tmp_path) -> None:
    rows = (
        _metric("random", 1, mse=0.5),
        _metric("pretrained", 1, mse=0.3, raw_feasible=True),
    )
    summary = summarize_diagnostic_rows(rows)
    summary["schema_version"] = "dppo-pretraining-diagnostics-v1"

    write_diagnostic_outputs(tmp_path, rows, summary)

    csv_text = (tmp_path / "per_record_metrics.csv").read_text(encoding="utf-8")
    json_text = (tmp_path / "summary.json").read_text(encoding="utf-8")
    assert csv_text.splitlines()[0].startswith("partition,model_name,episode_seed")
    assert len(csv_text.splitlines()) == 3
    assert "NaN" not in json_text and "Infinity" not in json_text
    assert __import__("json").loads(json_text)["record_count"] == 1


def test_atomic_writer_preserves_existing_file_when_replace_fails(
    tmp_path,
    monkeypatch,
) -> None:
    import os

    target = tmp_path / "summary.json"
    target.write_bytes(b"old-summary")
    rows = (
        _metric("random", 1, mse=0.5),
        _metric("pretrained", 1, mse=0.3),
    )

    def fail_replace(source, destination) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        write_diagnostic_outputs(
            tmp_path,
            rows,
            summarize_diagnostic_rows(rows),
        )

    assert target.read_bytes() == b"old-summary"
