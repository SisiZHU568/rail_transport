"""阶段 E 训练入口共享的数据集读取与规格校验。"""

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import numpy as np


@dataclass(frozen=True)
class TeacherDataset:
    observations: np.ndarray
    unbounded_actions: np.ndarray
    scores: np.ndarray
    effective_action_masks: np.ndarray
    source_trajectory_ids: np.ndarray
    state_group_ids: np.ndarray
    state_hashes: np.ndarray
    plan_hashes: np.ndarray
    teacher_types: np.ndarray
    plan_jsons: np.ndarray
    predicted_metrics: np.ndarray


@dataclass(frozen=True)
class TeacherDatasetRecord:
    source_trajectory_id: str
    state_group_id: str
    state_hash: str
    plan_hash: str
    observation: np.ndarray
    unbounded_action: np.ndarray
    scores: np.ndarray
    effective_action_mask: np.ndarray
    teacher_types: tuple[str, ...]
    plan_json: str
    predicted_metrics: tuple[float, float, float, float, float]
    preflight_code: str

    def __post_init__(self) -> None:
        strings = (
            self.source_trajectory_id,
            self.state_group_id,
            self.state_hash,
            self.plan_hash,
            self.plan_json,
        )
        if any(not value for value in strings):
            raise ValueError("教师记录的轨迹、状态和计划标识不能为空。")
        observation = np.asarray(self.observation, dtype=np.float32)
        unbounded = np.asarray(self.unbounded_action, dtype=np.float32)
        scores = np.asarray(self.scores, dtype=np.float32)
        mask = np.asarray(self.effective_action_mask, dtype=np.bool_)
        if observation.ndim != 1 or unbounded.ndim != 1:
            raise ValueError("教师观察和动作必须是一维向量。")
        if scores.shape != unbounded.shape or mask.shape != unbounded.shape:
            raise ValueError("教师 v/u/有效动作掩码维度必须一致。")
        if not np.isfinite(observation).all() or not np.isfinite(unbounded).all():
            raise ValueError("教师记录只能包含有限数。")
        expected = 1.0 / (1.0 + np.exp(-unbounded.astype(np.float64)))
        if not np.allclose(scores, expected, rtol=1e-5, atol=1e-6):
            raise ValueError("教师记录中的 u 必须等于 sigmoid(v)。")
        allowed_types = {"COST", "RELIABILITY", "BALANCE"}
        if not self.teacher_types or not set(self.teacher_types) <= allowed_types:
            raise ValueError("教师记录包含未知教师类型。")
        if self.preflight_code != "OK":
            raise ValueError("只有通过生命周期预审的教师记录才能入库。")
        if len(self.predicted_metrics) != 5 or any(
            not np.isfinite(value) for value in self.predicted_metrics
        ):
            raise ValueError("教师记录必须保存五个有限代理指标。")
        for name, array in (
            ("observation", observation),
            ("unbounded_action", unbounded),
            ("scores", scores),
            ("effective_action_mask", mask),
        ):
            array = array.copy()
            array.setflags(write=False)
            object.__setattr__(self, name, array)


def save_teacher_dataset(
    path: str | Path,
    records: tuple[TeacherDatasetRecord, ...],
    *,
    observation_spec_hash: str,
    action_spec_hash: str,
) -> None:
    """用原子替换写入正式教师数据集，避免中断后留下半个 NPZ。"""

    if not records:
        raise ValueError("教师数据集不能为空。")
    state_dim = records[0].observation.size
    action_dim = records[0].unbounded_action.size
    if any(
        record.observation.size != state_dim
        or record.unbounded_action.size != action_dim
        for record in records
    ):
        raise ValueError("教师数据集中的状态和动作维度必须统一。")
    group_owners: dict[str, str] = {}
    for record in records:
        owner = group_owners.setdefault(
            record.state_group_id, record.source_trajectory_id
        )
        if owner != record.source_trajectory_id:
            raise ValueError("同一 state_group_id 不能跨来源轨迹。")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", suffix=".npz", dir=target.parent, delete=False
        ) as temporary:
            temporary_name = temporary.name
            np.savez_compressed(
                temporary,
                observations=np.stack([item.observation for item in records]),
                unbounded_actions=np.stack(
                    [item.unbounded_action for item in records]
                ),
                scores=np.stack([item.scores for item in records]),
                effective_action_masks=np.stack(
                    [item.effective_action_mask for item in records]
                ),
                source_trajectory_ids=np.array(
                    [item.source_trajectory_id for item in records]
                ),
                state_group_ids=np.array([item.state_group_id for item in records]),
                state_hashes=np.array([item.state_hash for item in records]),
                plan_hashes=np.array([item.plan_hash for item in records]),
                teacher_types=np.array(
                    ["|".join(item.teacher_types) for item in records]
                ),
                plan_jsons=np.array([item.plan_json for item in records]),
                predicted_metrics=np.asarray(
                    [item.predicted_metrics for item in records], dtype=np.float64
                ),
                preflight_codes=np.array([item.preflight_code for item in records]),
                dataset_schema_version=np.array("phase-e-teacher-v1"),
                observation_spec_hash=np.array(observation_spec_hash),
                action_spec_hash=np.array(action_spec_hash),
            )
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, target)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


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
        required = {
            "observations", "unbounded_actions", "scores",
            "effective_action_masks", "source_trajectory_ids",
            "state_group_ids", "state_hashes", "plan_hashes",
            "teacher_types", "plan_jsons", "predicted_metrics",
            "preflight_codes", "dataset_schema_version",
            "observation_spec_hash", "action_spec_hash",
        }
        if not required <= set(data.files):
            raise ValueError("教师数据集缺少正式审计字段。")
        if str(data["dataset_schema_version"].item()) != "phase-e-teacher-v1":
            raise ValueError("教师数据集 schema 版本不受支持。")
        if (
            str(data["observation_spec_hash"].item()) != expected_observation_hash
            or str(data["action_spec_hash"].item()) != expected_action_hash
        ):
            raise ValueError("CHECKPOINT_SPEC_MISMATCH")
        observations = np.asarray(data["observations"], dtype=np.float32)
        unbounded = np.asarray(data["unbounded_actions"], dtype=np.float32)
        scores = np.asarray(data["scores"], dtype=np.float32)
        masks = np.asarray(data["effective_action_masks"], dtype=np.bool_)
        source_ids = np.asarray(data["source_trajectory_ids"]).astype(str)
        group_ids = np.asarray(data["state_group_ids"]).astype(str)
        state_hashes = np.asarray(data["state_hashes"]).astype(str)
        plan_hashes = np.asarray(data["plan_hashes"]).astype(str)
        teacher_types = np.asarray(data["teacher_types"]).astype(str)
        plan_jsons = np.asarray(data["plan_jsons"]).astype(str)
        metrics = np.asarray(data["predicted_metrics"], dtype=np.float64)
        preflight_codes = np.asarray(data["preflight_codes"]).astype(str)
    if observations.ndim != 2 or observations.shape[1] != state_dim:
        raise ValueError("教师 observations 维度与 ObservationSpec 不一致。")
    if unbounded.shape != (observations.shape[0], action_dim) or scores.shape != unbounded.shape:
        raise ValueError("教师 v/u 维度与 ActionSpec 不一致。")
    row_count = observations.shape[0]
    if masks.shape != unbounded.shape or metrics.shape != (row_count, 5):
        raise ValueError("教师掩码或代理指标维度无效。")
    metadata = (
        source_ids, group_ids, state_hashes, plan_hashes, teacher_types,
        plan_jsons, preflight_codes,
    )
    if any(array.shape != (row_count,) for array in metadata):
        raise ValueError("教师逐样本审计字段数量不一致。")
    if np.any(preflight_codes != "OK"):
        raise ValueError("教师数据集包含未通过预审的样本。")
    if not np.isfinite(observations).all() or not np.isfinite(unbounded).all():
        raise ValueError("教师数据必须只包含有限数。")
    if not np.isfinite(metrics).all():
        raise ValueError("教师代理指标必须只包含有限数。")
    expected_scores = 1.0 / (1.0 + np.exp(-unbounded.astype(np.float64)))
    if not np.allclose(scores, expected_scores, rtol=1e-5, atol=1e-6):
        raise ValueError("教师评分 u 与无界动作 v 不一致。")
    for array in (
        observations, unbounded, scores, masks, metrics, *metadata
    ):
        array.setflags(write=False)
    return TeacherDataset(
        observations, unbounded, scores, masks, source_ids, group_ids,
        state_hashes, plan_hashes, teacher_types, plan_jsons, metrics,
    )
