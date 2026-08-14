import pytest

from src.phase_e_training_workflow import (
    FrameRewardInput,
    TransactionalRollout,
    ValidationResult,
    compute_frame_reward,
    select_validation_checkpoint,
)


def test_frame_reward_has_one_weight_and_expected_denominators() -> None:
    result = compute_frame_reward(
        FrameRewardInput(
            at_risk_batches=10,
            new_violations=2,
            queue_equivalent_bits=100.0,
            deficit_equivalent_bits=25.0,
            raw_cost=40.0,
            reference_cost=100.0,
        ),
        alpha=0.8,
    )

    # V=.2, D=.25, S=V+D-VD=.4, C=.4，因此 reward=-.4。
    assert result.service_loss == pytest.approx(0.4)
    assert result.reward == pytest.approx(-0.4)
    assert -1.0 <= result.reward <= 0.0


def test_internal_failure_discards_only_current_uncommitted_rollout() -> None:
    rollout = TransactionalRollout(required_slow_frames=2)
    rollout.stage("frame-0")
    assert rollout.commit_if_ready() == ()
    rollout.stage("frame-1")
    assert rollout.commit_if_ready() == ("frame-0", "frame-1")

    rollout.stage("frame-2")
    discarded = rollout.abort("FAST_SOLVER_FAILURE")
    assert discarded == ("frame-2",)
    assert rollout.committed == ("frame-0", "frame-1")


def test_checkpoint_selection_rejects_internal_failures_and_quantizes_metrics() -> None:
    candidates = (
        ValidationResult("bad", 0, 0.0, 0.0, 0.0, 1),
        ValidationResult("later", 20, 0.0104, 0.20, 50.0, 0),
        ValidationResult("earlier", 10, 0.0101, 0.20, 50.0, 0),
    )
    chosen = select_validation_checkpoint(
        candidates,
        violation_precision=0.001,
        deficit_precision=0.001,
        cost_precision=0.01,
    )

    assert chosen.checkpoint_id == "earlier"
    with pytest.raises(ValueError, match="NO_VALID_CHECKPOINT"):
        select_validation_checkpoint(
            candidates[:1],
            violation_precision=0.001,
            deficit_precision=0.001,
            cost_precision=0.01,
        )
