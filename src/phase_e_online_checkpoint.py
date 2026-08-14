"""Phase E 在线训练 checkpoint 的严格绑定、原子保存与完整恢复。"""

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from src.dppo import DPPOAgent


_FORMAT_VERSION = "phase-e-online-v1"


@dataclass(frozen=True)
class PhaseEOnlineMetadata:
    format_version: str
    observation_spec_hash: str
    action_spec_hash: str
    config_sha256: str
    next_frame_index: int
    completed_update_count: int
    best_validation_reward: float | None

    def __post_init__(self) -> None:
        if self.format_version != _FORMAT_VERSION:
            raise ValueError("在线 checkpoint 格式必须是 phase-e-online-v1。")
        for name in (
            "observation_spec_hash",
            "action_spec_hash",
            "config_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} 必须是非空字符串。")
        for name in ("next_frame_index", "completed_update_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} 必须是非负整数。")
        if (
            self.best_validation_reward is not None
            and not math.isfinite(self.best_validation_reward)
        ):
            raise ValueError("best_validation_reward 必须为空或有限数。")


@dataclass(frozen=True)
class LoadedPhaseEOnlineCheckpoint:
    metadata: PhaseEOnlineMetadata


def _payload(agent: DPPOAgent, metadata: PhaseEOnlineMetadata) -> dict:
    return {
        "format_version": _FORMAT_VERSION,
        "metadata": asdict(metadata),
        "frozen_policy_state_dict": agent.frozen_policy.state_dict(),
        "trainable_policy_state_dict": agent.trainable_policy.state_dict(),
        "value_network_state_dict": agent.value_network.state_dict(),
        "policy_optimizer_state_dict": agent.policy_optimizer.state_dict(),
        "value_optimizer_state_dict": agent.value_optimizer.state_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else ()
        ),
    }


def save_phase_e_online_checkpoint(
    path: str | Path,
    agent: DPPOAgent,
    metadata: PhaseEOnlineMetadata,
) -> Path:
    """把完整更新边界原子保存到目标路径。"""

    if not isinstance(agent, DPPOAgent):
        raise TypeError("agent 必须是 DPPOAgent。")
    if not isinstance(metadata, PhaseEOnlineMetadata):
        raise TypeError("metadata 必须是 PhaseEOnlineMetadata。")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        torch.save(_payload(agent, metadata), temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def _load_metadata(payload: dict) -> PhaseEOnlineMetadata:
    if payload.get("format_version") != _FORMAT_VERSION:
        raise ValueError("CHECKPOINT_SPEC_MISMATCH: checkpoint 格式不匹配。")
    raw = payload.get("metadata")
    if not isinstance(raw, dict):
        raise ValueError("CHECKPOINT_SPEC_MISMATCH: checkpoint metadata 缺失。")
    try:
        return PhaseEOnlineMetadata(**raw)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "CHECKPOINT_SPEC_MISMATCH: checkpoint metadata 无效。"
        ) from error


def read_phase_e_online_metadata(path: str | Path) -> PhaseEOnlineMetadata:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("CHECKPOINT_SPEC_MISMATCH: checkpoint payload 无效。")
    return _load_metadata(payload)


def load_phase_e_online_checkpoint(
    path: str | Path,
    agent: DPPOAgent,
    *,
    expected_observation_spec_hash: str,
    expected_action_spec_hash: str,
    expected_config_sha256: str,
) -> LoadedPhaseEOnlineCheckpoint:
    """校验全部规格绑定后，把完整训练状态恢复到已有 agent。"""

    if not isinstance(agent, DPPOAgent):
        raise TypeError("agent 必须是 DPPOAgent。")
    payload = torch.load(Path(path), map_location=agent.device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("CHECKPOINT_SPEC_MISMATCH: checkpoint payload 无效。")
    metadata = _load_metadata(payload)
    bindings = (
        (metadata.observation_spec_hash, expected_observation_spec_hash),
        (metadata.action_spec_hash, expected_action_spec_hash),
        (metadata.config_sha256, expected_config_sha256),
    )
    if any(actual != expected for actual, expected in bindings):
        raise ValueError("CHECKPOINT_SPEC_MISMATCH: checkpoint 绑定不匹配。")

    agent.frozen_policy.load_state_dict(payload["frozen_policy_state_dict"])
    agent.trainable_policy.load_state_dict(payload["trainable_policy_state_dict"])
    agent.value_network.load_state_dict(payload["value_network_state_dict"])
    agent.policy_optimizer.load_state_dict(payload["policy_optimizer_state_dict"])
    agent.value_optimizer.load_state_dict(payload["value_optimizer_state_dict"])
    random.setstate(payload["python_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    cuda_states = payload.get("cuda_rng_states", ())
    if cuda_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_states)
    return LoadedPhaseEOnlineCheckpoint(metadata)
