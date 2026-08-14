"""测试 GPU/CPU 通用的 DPPO 慢帧采样与事务式更新。"""

import numpy as np
import torch

from src.dppo import DPPOAgent, DPPOBehaviorCloningBatch, DPPOConfig
from src.dppo_diffusion import ConditionalDiffusionMLP, CosineNoiseSchedule
from src.phase_e_environment import PhaseESlowFrameResult
from src.phase_e_online_trainer import PhaseEOnlineTrainer
from src.phase_e_training_workflow import FrameReward


def _agent() -> DPPOAgent:
    torch.manual_seed(3)
    policy = ConditionalDiffusionMLP(4, 6, (16, 16))
    config = DPPOConfig(
        diffusion_steps=3,
        fine_tuned_steps=2,
        value_hidden_dims=(16,),
        batch_size=2,
        update_epochs=1,
        target_kl=10.0,
    )
    return DPPOAgent(
        policy, CosineNoiseSchedule(3), config, device="cpu"
    )


def _success(reward: float) -> PhaseESlowFrameResult:
    details = FrameReward(0.0, 0.0, 0.0, 0.0, reward)
    return PhaseESlowFrameResult("OK", details, (), 0.0, 0.0, 0.0, 0, 0)


def test_online_trainer_uses_full_unbounded_action_and_updates_complete_rollout() -> None:
    agent = _agent()
    trainer = PhaseEOnlineTrainer(agent, rollout_length_slow_frames=2)
    state0 = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    state1 = state0 + 0.1
    next_state = state1 + 0.1

    decision0 = trainer.sample_decision(state0, seed=100)
    decision1 = trainer.sample_decision(state1, seed=101)
    trainer.record_result(decision0, _success(-0.4), terminated=False)
    trainer.record_result(decision1, _success(-0.2), terminated=False)
    metrics = trainer.update_if_ready(next_state)

    assert decision0.unbounded_action.shape == (agent.action_dim,)
    assert np.all((decision0.scores > 0.0) & (decision0.scores < 1.0))
    assert metrics is not None
    assert metrics["optimizer_step_count"] >= 1.0
    assert trainer.pending_count == 0
    assert trainer.completed_update_count == 1


def test_internal_failure_discards_only_current_uncommitted_rollout() -> None:
    trainer = PhaseEOnlineTrainer(_agent(), rollout_length_slow_frames=2)
    decision = trainer.sample_decision(np.zeros(4, dtype=np.float32), seed=9)
    trainer.record_result(decision, _success(-0.1), terminated=False)

    discarded = trainer.record_result(
        trainer.sample_decision(np.ones(4, dtype=np.float32), seed=10),
        PhaseESlowFrameResult(
            "FAST_SOLVER_FAILURE", None, (), 1.0, 1.0, 0.0, 0, 1
        ),
        terminated=True,
    )

    assert discarded == 1
    assert trainer.pending_count == 0
    assert trainer.internal_failure_count == 1


def test_online_trainer_applies_early_decaying_teacher_bc() -> None:
    agent = _agent()
    states = np.stack(
        [np.full(agent.state_dim, index / 10, dtype=np.float32) for index in range(4)]
    )
    actions = np.stack(
        [np.full(agent.action_dim, -0.2 + index / 20, dtype=np.float32) for index in range(4)]
    )
    trainer = PhaseEOnlineTrainer(
        agent,
        rollout_length_slow_frames=2,
        teacher_batch=DPPOBehaviorCloningBatch(states, actions),
        total_online_updates=10,
    )
    state = np.zeros(agent.state_dim, dtype=np.float32)
    for index in range(2):
        decision = trainer.sample_decision(state + index * 0.1, seed=50 + index)
        trainer.record_result(decision, _success(-0.1), terminated=False)

    metrics = trainer.update_if_ready(state + 0.3)

    assert metrics is not None
    assert metrics["behavior_cloning_weight"] == 0.05
    assert metrics["behavior_cloning_loss"] >= 0.0
