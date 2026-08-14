"""阶段 E 的轨迹清单隔离和检查点结构规格绑定。"""

from dataclasses import dataclass
import hashlib
import json


@dataclass(frozen=True)
class TrajectoryManifest:
    purpose: str
    trajectories: tuple[tuple[str, int], ...]

    @property
    def sha256(self) -> str:
        payload = {"purpose": self.purpose, "trajectories": self.trajectories}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def validate_manifest_isolation(manifests: tuple[TrajectoryManifest, ...]) -> None:
    seen_ids: set[str] = set()
    seen_seeds: set[int] = set()
    for manifest in manifests:
        ids = {trajectory_id for trajectory_id, _ in manifest.trajectories}
        seeds = {seed for _, seed in manifest.trajectories}
        if seen_ids & ids or seen_seeds & seeds:
            raise ValueError("轨迹清单不隔离。")
        seen_ids.update(ids)
        seen_seeds.update(seeds)


@dataclass(frozen=True)
class CheckpointSpecification:
    observation_spec_hash: str
    action_spec_hash: str
    model_architecture_hash: str
    normalization_state_version: int


def validate_checkpoint_specification(
    saved: CheckpointSpecification,
    expected: CheckpointSpecification,
) -> None:
    if saved != expected:
        raise ValueError("CHECKPOINT_SPEC_MISMATCH")
