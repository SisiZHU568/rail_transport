"""阶段 E 在线 DPPO：完整无界动作采样、事务 rollout 与 GPU 更新。"""

from dataclasses import dataclass

import numpy as np

from src.dppo import (
    DPPOAgent,
    DPPOBehaviorCloningBatch,
    DPPORolloutBuffer,
    DPPORolloutTransition,
)
from src.phase_d_policy_specs import PolicyAdapter
from src.phase_e_environment import PhaseESlowFrameResult
from src.phase_e_learning import online_bc_weight


@dataclass(frozen=True)
class PhaseEPolicyDecision:
    state: np.ndarray
    unbounded_action: np.ndarray
    scores: np.ndarray
    denoising_actions: np.ndarray
    old_log_probabilities: np.ndarray
    value: float

    def __post_init__(self) -> None:
        for name in (
            "state", "unbounded_action", "scores",
            "denoising_actions", "old_log_probabilities",
        ):
            # 有界评分保留 float64，避免接近边界的 sigmoid 值被 float32
            # 提前舍入成精确 0/1；网络轨迹仍保持原生 float32。
            dtype = np.float64 if name == "scores" else np.float32
            array = np.asarray(getattr(self, name), dtype=dtype).copy()
            if not np.isfinite(array).all():
                raise ValueError(f"{name} 必须只包含有限数。")
            array.setflags(write=False)
            object.__setattr__(self, name, array)


class PhaseEOnlineTrainer:
    """只把完整、无内部失败的固定长度慢帧段交给 DPPO 更新。"""

    def __init__(
        self,
        agent: DPPOAgent,
        *,
        rollout_length_slow_frames: int,
        teacher_batch: DPPOBehaviorCloningBatch | None = None,
        total_online_updates: int = 0,
    ) -> None:
        if rollout_length_slow_frames <= 0:
            raise ValueError("rollout_length_slow_frames 必须为正。")
        self.agent = agent
        self.adapter = PolicyAdapter(agent.action_dim)
        self.rollout_length_slow_frames = rollout_length_slow_frames
        if teacher_batch is not None and total_online_updates <= 0:
            raise ValueError("启用在线教师 BC 时 total_online_updates 必须为正。")
        if teacher_batch is None and total_online_updates != 0:
            raise ValueError("未提供 teacher_batch 时不能设置 total_online_updates。")
        self.teacher_batch = teacher_batch
        self.total_online_updates = total_online_updates
        self._pending: list[DPPORolloutTransition] = []
        self.completed_update_count = 0
        self.internal_failure_count = 0
        self.last_internal_failure_code: str | None = None

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def sample_decision(self, state: np.ndarray, *, seed: int) -> PhaseEPolicyDecision:
        """采样完整反向扩散链；PPO 对数概率始终针对无界变量 v。"""

        state_array = np.asarray(state, dtype=np.float32)
        sample = self.agent.sample_action(state_array, seed=seed)
        chain = sample.actions[:, 0].detach().cpu().numpy()
        unbounded = chain[-1]
        scores = self.adapter.to_scores(unbounded)
        old_log_probabilities = (
            sample.log_probabilities[:, 0].detach().cpu().numpy()
        )
        return PhaseEPolicyDecision(
            state=state_array,
            unbounded_action=unbounded,
            scores=scores,
            denoising_actions=chain,
            old_log_probabilities=old_log_probabilities,
            value=float(self.agent.value(state_array)),
        )

    def record_result(
        self,
        decision: PhaseEPolicyDecision,
        result: PhaseESlowFrameResult,
        *,
        terminated: bool,
    ) -> int:
        """成功转移进入暂存段；内部失败丢弃自上次更新后的整个未提交段。"""

        if result.code != "OK" or result.reward is None:
            discarded = len(self._pending)
            self._pending.clear()
            self.internal_failure_count += 1
            self.last_internal_failure_code = result.code
            return discarded
        if len(self._pending) >= self.rollout_length_slow_frames:
            raise RuntimeError("完整 rollout 尚未更新，不能继续追加新转移。")
        self._pending.append(
            DPPORolloutTransition(
                state=decision.state,
                raw_action=decision.unbounded_action,
                denoising_actions=decision.denoising_actions,
                old_log_probabilities=decision.old_log_probabilities,
                reward=result.reward.reward,
                value=decision.value,
                terminated=terminated,
            )
        )
        return 0

    def update_if_ready(self, next_state: np.ndarray) -> dict[str, float] | None:
        """在下一状态可用后进行 bootstrap；不足固定长度时保持暂存。"""

        if len(self._pending) < self.rollout_length_slow_frames:
            return None
        if len(self._pending) != self.rollout_length_slow_frames:
            raise RuntimeError("事务 rollout 长度不变量被破坏。")
        buffer = DPPORolloutBuffer()
        for transition in self._pending:
            buffer.append(transition)
        next_value = float(self.agent.value(np.asarray(next_state, dtype=np.float32)))
        bc_weight = 0.0
        if self.teacher_batch is not None:
            bc_weight = online_bc_weight(
                self.completed_update_count,
                total_updates=self.total_online_updates,
            )
        metrics = self.agent.update(
            buffer,
            next_value=next_value,
            teacher_batch=self.teacher_batch,
            behavior_cloning_weight=bc_weight,
        )
        self._pending.clear()
        self.completed_update_count += 1
        return metrics
