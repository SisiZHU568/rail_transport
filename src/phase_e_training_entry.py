"""阶段 E 训练入口共享的数据集读取与规格校验。"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TeacherDataset:
    observations: np.ndarray
    unbounded_actions: np.ndarray
    scores: np.ndarray


def load_teacher_dataset(
    path: str | Path,
    *,
    expected_observation_hash: str,
    expected_action_hash: str,
    state_dim: int,
    action_dim: int,
) -> TeacherDataset:
    """只接受同时保存 v、u 且与当前状态/动作规格一致的数据集。"""
    with np.load(Path(path), allow_pickle=False) as data:
        if (
            str(data["observation_spec_hash"].item()) != expected_observation_hash
            or str(data["action_spec_hash"].item()) != expected_action_hash
        ):
            raise ValueError("CHECKPOINT_SPEC_MISMATCH")
        observations = np.asarray(data["observations"], dtype=np.float32)
        unbounded = np.asarray(data["unbounded_actions"], dtype=np.float32)
        scores = np.asarray(data["scores"], dtype=np.float32)
    if observations.ndim != 2 or observations.shape[1] != state_dim:
        raise ValueError("教师 observations 维度与 ObservationSpec 不一致。")
    if unbounded.shape != (observations.shape[0], action_dim) or scores.shape != unbounded.shape:
        raise ValueError("教师 v/u 维度与 ActionSpec 不一致。")
    if not np.isfinite(observations).all() or not np.isfinite(unbounded).all():
        raise ValueError("教师数据必须只包含有限数。")
    expected_scores = 1.0 / (1.0 + np.exp(-unbounded.astype(np.float64)))
    if not np.allclose(scores, expected_scores, rtol=1e-5, atol=1e-6):
        raise ValueError("教师评分 u 与无界动作 v 不一致。")
    for array in (observations, unbounded, scores):
        array.setflags(write=False)
    return TeacherDataset(observations, unbounded, scores)
