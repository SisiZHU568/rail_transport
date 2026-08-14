"""测试固定验证集的微平均、公共噪声和无权重检查点选择。"""

from src.phase_e_validation import (
    TrajectoryManifest,
    TrajectoryManifestEntry,
    ValidationCheckpointDescriptor,
    ValidationCheckpointResult,
    ValidationRunCounters,
    ValidationSelectionTolerance,
    aggregate_validation_runs,
    derive_policy_noise_seed,
    run_fixed_validation,
    select_validation_checkpoint,
    validate_manifest_isolation,
)
import pytest


def _result(
    name: str,
    update: int,
    *,
    violations: int,
    risk: int,
    deficit: float,
    workload: float,
    cost: float,
    failures: int = 0,
) -> ValidationCheckpointResult:
    counters = ValidationRunCounters(
        trajectory_count=2,
        new_violation_count=violations,
        risk_batch_count=risk,
        service_deficit_equivalent_bits=deficit,
        queue_workload_equivalent_bits=workload,
        raw_resource_cost=cost,
        internal_failure_count=failures,
    )
    return ValidationCheckpointResult(name, update, counters)


def test_validation_runs_use_micro_average_and_mean_trajectory_cost() -> None:
    aggregate = aggregate_validation_runs((
        ValidationRunCounters(1, 1, 2, 20.0, 100.0, 4.0, 0),
        ValidationRunCounters(2, 2, 8, 10.0, 100.0, 14.0, 0),
    ))

    assert aggregate.violation_rate == 0.3
    assert aggregate.deficit_rate == 0.15
    assert aggregate.mean_raw_cost_per_trajectory == 6.0


def test_checkpoint_selection_filters_failures_then_uses_quantized_lexicographic_key() -> None:
    tolerance = ValidationSelectionTolerance(1e-3, 1e-3, 0.1)
    failed_but_better = _result(
        "failed.pt", 1, violations=0, risk=10, deficit=0, workload=10,
        cost=1.0, failures=1,
    )
    later = _result(
        "later.pt", 9, violations=1, risk=1000, deficit=10, workload=1000,
        cost=20.04,
    )
    earlier = _result(
        "earlier.pt", 4, violations=1, risk=1000, deficit=10, workload=1000,
        cost=20.01,
    )

    selection = select_validation_checkpoint(
        (failed_but_better, later, earlier), tolerance
    )

    assert selection.code == "OK"
    assert selection.selected is earlier


def test_checkpoint_selection_returns_explicit_no_valid_state() -> None:
    invalid = _result(
        "bad.pt", 1, violations=0, risk=0, deficit=0, workload=0,
        cost=0.0, failures=2,
    )

    selection = select_validation_checkpoint(
        (invalid,), ValidationSelectionTolerance(1e-3, 1e-3, 0.1)
    )

    assert selection.code == "NO_VALID_CHECKPOINT"
    assert selection.selected is None


def test_policy_noise_seed_is_derived_per_trajectory_repeat_and_decision() -> None:
    seed = derive_policy_noise_seed("validation-07", 2, 11, 47000)

    assert seed == derive_policy_noise_seed("validation-07", 2, 11, 47000)
    assert seed != derive_policy_noise_seed("validation-07", 2, 12, 47000)
    assert seed != derive_policy_noise_seed("validation-07", 3, 11, 47000)
    assert 0 <= seed < 2**63


def test_trajectory_manifests_have_stable_hash_and_disjoint_ids_and_seeds() -> None:
    train = TrajectoryManifest(
        "train", (TrajectoryManifestEntry("train-1", 100),), 47000, 2
    )
    validation = TrajectoryManifest(
        "validation", (TrajectoryManifestEntry("validation-1", 200),), 47000, 2
    )

    validate_manifest_isolation((train, validation))

    assert train.sha256 == TrajectoryManifest(
        "train", (TrajectoryManifestEntry("train-1", 100),), 47000, 2
    ).sha256


def test_trajectory_manifests_reject_overlapping_base_seed() -> None:
    train = TrajectoryManifest(
        "train", (TrajectoryManifestEntry("train-1", 100),), 47000, 1
    )
    test = TrajectoryManifest(
        "test", (TrajectoryManifestEntry("test-1", 100),), 57000, 1
    )

    with pytest.raises(ValueError, match="基础随机种子"):
        validate_manifest_isolation((train, test))


def test_fixed_validation_reuses_decision_noise_and_selects_checkpoint() -> None:
    manifest = TrajectoryManifest(
        "validation",
        (
            TrajectoryManifestEntry("validation-1", 200),
            TrajectoryManifestEntry("validation-2", 201),
        ),
        47000,
        2,
    )
    observed_seeds: dict[str, list[int]] = {"early.pt": [], "late.pt": []}

    def evaluate(checkpoint_id, entry, repeat_id, policy_seeds):
        observed_seeds[checkpoint_id].extend(
            [policy_seeds.for_slow_decision(0), policy_seeds.for_slow_decision(3)]
        )
        violation = 0 if checkpoint_id == "late.pt" else 1
        return ValidationRunCounters(1, violation, 10, 1.0, 10.0, 3.0, 0)

    report = run_fixed_validation(
        (
            ValidationCheckpointDescriptor("early.pt", 2),
            ValidationCheckpointDescriptor("late.pt", 5),
        ),
        manifest,
        ValidationSelectionTolerance(1e-3, 1e-3, 0.1),
        evaluate,
    )

    assert observed_seeds["early.pt"] == observed_seeds["late.pt"]
    assert report.manifest_sha256 == manifest.sha256
    assert report.selection.selected.checkpoint_id == "late.pt"
