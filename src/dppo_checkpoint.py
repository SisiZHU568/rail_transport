"""实现 DPPO 扩散预训练循环和可严格校验的版本化检查点。"""

from dataclasses import asdict, dataclass, fields
import math
from pathlib import Path
from typing import Any

import torch

from src.dppo import DPPOAgent, DPPOConfig
from src.dppo_diffusion import (
    ConditionalDiffusionMLP,
    CosineNoiseSchedule,
    diffusion_noise_loss,
)
from src.dppo_stability import (
    DPPOStabilityProfile,
    parse_stability_profile_json,
    stability_profile_json,
    stability_profile_sha256,
)


@dataclass(frozen=True)
class DPPOCheckpointMetadata:
    """保存决定检查点能否安全复用的全部实验模式字段。"""

    state_schema_version: str
    action_schema_version: str
    state_dim: int
    action_dim: int
    mec_count: int
    compute_node_count: int
    function_count: int
    diffusion_steps: int
    fine_tuned_steps: int
    maximum_retention_seconds: float
    replica_threshold: float
    config_hash: str

    def __post_init__(self) -> None:
        """拒绝无法构成合法 DPPO 场景或动作编码的元数据。"""

        for name in (
            "state_schema_version",
            "action_schema_version",
            "config_hash",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} 必须是非空字符串。")
        for name in (
            "state_dim",
            "action_dim",
            "mec_count",
            "compute_node_count",
            "function_count",
            "diffusion_steps",
            "fine_tuned_steps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} 必须是正整数。")
        if self.fine_tuned_steps > self.diffusion_steps:
            raise ValueError("fine_tuned_steps 不能大于 diffusion_steps。")
        retention = float(self.maximum_retention_seconds)
        if not math.isfinite(retention) or retention <= 0.0:
            raise ValueError("maximum_retention_seconds 必须是正有限数。")
        threshold = float(self.replica_threshold)
        if not math.isfinite(threshold) or not -1.0 < threshold <= 1.0:
            raise ValueError("replica_threshold 必须位于 (-1, 1]。")
        object.__setattr__(self, "maximum_retention_seconds", retention)
        object.__setattr__(self, "replica_threshold", threshold)


@dataclass(frozen=True)
class LoadedDPPOCheckpoint:
    """保存恢复后的网络、优化器、元数据和训练进度。"""

    model: ConditionalDiffusionMLP
    optimizer: torch.optim.Adam
    metadata: DPPOCheckpointMetadata
    epoch: int
    rng_state: torch.Tensor


@dataclass(frozen=True)
class LoadedDPPOOnlineCheckpoint:
    """保存恢复后的双层策略、价值网络及在线训练进度。"""

    agent: DPPOAgent
    metadata: DPPOCheckpointMetadata
    iteration: int
    best_mean_reward: float
    rng_state: torch.Tensor
    stability_profile: DPPOStabilityProfile
    stability_profile_sha256: str


def _validate_profile_config_binding(
    profile: DPPOStabilityProfile,
    config: DPPOConfig,
    metadata: DPPOCheckpointMetadata,
) -> None:
    """验证正式训练稳定性门禁与配置、场景身份保持严格绑定。"""

    if not isinstance(profile, DPPOStabilityProfile):
        raise TypeError("stability_profile 必须是 DPPOStabilityProfile。")
    # 规范化往返会重新执行 profile 完整性校验，不能信任被内存篡改的 frozen 对象。
    parse_stability_profile_json(stability_profile_json(profile))
    if not profile.qualified or profile.selected_clip_ratio is None:
        raise ValueError("稳定性配置未通过校准，不能用于正式训练检查点。")
    if profile.config_hash != metadata.config_hash:
        raise ValueError("稳定性配置 config_hash 与检查点元数据不一致。")
    expected = {
        "clip_ratio": profile.selected_clip_ratio,
        "training_sampling_min_std": profile.training_sampling_min_std,
        "probability_min_std": profile.probability_min_std,
        "evaluation_sampling_min_std": profile.evaluation_sampling_min_std,
        "target_kl": profile.target_kl,
    }
    for name, expected_value in expected.items():
        if getattr(config, name) != expected_value:
            raise ValueError(f"智能体配置 {name} 与稳定性配置不一致。")


def validate_dppo_stability_binding(
    agent: DPPOAgent,
    metadata: DPPOCheckpointMetadata,
    stability_profile: DPPOStabilityProfile,
) -> None:
    """在产生训练输出前统一验证 agent、metadata 与 profile 的绑定。"""

    if not isinstance(agent, DPPOAgent):
        raise TypeError("agent 必须是 DPPOAgent。")
    if not isinstance(metadata, DPPOCheckpointMetadata):
        raise TypeError("metadata 必须是 DPPOCheckpointMetadata。")
    _validate_profile_config_binding(stability_profile, agent.config, metadata)


def validate_checkpoint_metadata(
    actual: DPPOCheckpointMetadata,
    expected: DPPOCheckpointMetadata,
) -> None:
    """逐字段检查兼容性，并在错误中明确指出不匹配字段。"""

    if not isinstance(actual, DPPOCheckpointMetadata):
        raise TypeError("actual 必须是 DPPOCheckpointMetadata。")
    if not isinstance(expected, DPPOCheckpointMetadata):
        raise TypeError("expected 必须是 DPPOCheckpointMetadata。")
    for field in fields(DPPOCheckpointMetadata):
        if getattr(actual, field.name) != getattr(expected, field.name):
            raise ValueError(f"检查点 {field.name} 与当前配置不一致。")


def resolve_torch_device(device: str | torch.device) -> torch.device:
    """解析训练设备；请求不可用 CUDA 时明确失败而非静默降级。"""

    try:
        resolved = torch.device(device)
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"无效的 PyTorch 设备：{device!r}。") from error
    if resolved.type not in {"cpu", "cuda"}:
        raise ValueError("预训练设备只能是 cpu 或 cuda。")
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("请求了 CUDA，但当前 PyTorch 环境没有可用 CUDA 设备。")
    return resolved


def seed_torch_for_pretraining(
    seed: int,
    device: str | torch.device,
) -> torch.device:
    """在创建模型前统一设置 CPU 和可选 CUDA 的预训练随机种子。"""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("预训练 seed 必须是非负整数。")
    resolved = resolve_torch_device(device)
    torch.manual_seed(seed)
    if resolved.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return resolved


def _validate_training_tensors(
    model: ConditionalDiffusionMLP,
    states: torch.Tensor,
    clean_actions: torch.Tensor,
    batch_size: int,
    seed: int,
) -> int:
    """统一检查预训练和验证所需数据形状与超参数。"""

    if not isinstance(states, torch.Tensor) or states.shape[1:] != (
        model.state_dim,
    ):
        raise ValueError(f"states 的 state_dim 必须为 {model.state_dim}。")
    if not isinstance(clean_actions, torch.Tensor) or clean_actions.shape[1:] != (
        model.action_dim,
    ):
        raise ValueError(
            f"clean_actions 的 action_dim 必须为 {model.action_dim}。"
        )
    if states.ndim != 2 or clean_actions.ndim != 2:
        raise ValueError("states 和 clean_actions 必须是二维张量。")
    if states.shape[0] != clean_actions.shape[0] or states.shape[0] == 0:
        raise ValueError("states 和 clean_actions 必须具有相同的非零样本数。")
    if not states.is_floating_point() or not clean_actions.is_floating_point():
        raise ValueError("states 和 clean_actions 必须是浮点张量。")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size 必须是正整数。")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed 必须是非负整数。")
    return int(states.shape[0])


def _select_batch(
    values: torch.Tensor,
    indices: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """在数据原设备取批次，再送入目标训练设备。"""

    source_indices = indices.to(values.device)
    return values.index_select(0, source_indices).to(
        device=device,
        dtype=torch.float32,
    )


def pretrain_diffusion_epoch(
    model: ConditionalDiffusionMLP,
    schedule: CosineNoiseSchedule,
    optimizer: torch.optim.Optimizer,
    states: torch.Tensor,
    clean_actions: torch.Tensor,
    *,
    batch_size: int,
    seed: int,
    device: str | torch.device,
    gradient_clip_norm: float | None = None,
) -> float:
    """确定性打乱专家样本并完成一轮扩散噪声预测训练。"""

    resolved_device = resolve_torch_device(device)
    sample_count = _validate_training_tensors(
        model,
        states,
        clean_actions,
        batch_size,
        seed,
    )
    if gradient_clip_norm is not None and (
        not math.isfinite(float(gradient_clip_norm))
        or float(gradient_clip_norm) <= 0.0
    ):
        raise ValueError("gradient_clip_norm 必须是正有限数或 None。")
    model.to(resolved_device)
    model.train()
    shuffle_generator = torch.Generator(device="cpu").manual_seed(seed)
    noise_generator = torch.Generator(device=resolved_device).manual_seed(seed + 1)
    permutation = torch.randperm(sample_count, generator=shuffle_generator)
    total_loss = 0.0
    total_examples = 0
    for start in range(0, sample_count, batch_size):
        indices = permutation[start : start + batch_size]
        batch_states = _select_batch(states, indices, resolved_device)
        batch_actions = _select_batch(clean_actions, indices, resolved_device)
        optimizer.zero_grad(set_to_none=True)
        loss = diffusion_noise_loss(
            model,
            schedule,
            batch_states,
            batch_actions,
            noise_generator,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("扩散预训练损失出现非有限值。")
        loss.backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(gradient_clip_norm),
                error_if_nonfinite=True,
            )
        optimizer.step()
        examples = int(batch_states.shape[0])
        total_loss += float(loss.detach().cpu()) * examples
        total_examples += examples
    return total_loss / total_examples


def evaluate_diffusion_loss(
    model: ConditionalDiffusionMLP,
    schedule: CosineNoiseSchedule,
    states: torch.Tensor,
    clean_actions: torch.Tensor,
    *,
    batch_size: int,
    seed: int,
    device: str | torch.device,
) -> float:
    """使用固定随机种子计算一轮可复现的验证噪声损失。"""

    resolved_device = resolve_torch_device(device)
    sample_count = _validate_training_tensors(
        model,
        states,
        clean_actions,
        batch_size,
        seed,
    )
    model.to(resolved_device)
    was_training = model.training
    model.eval()
    noise_generator = torch.Generator(device=resolved_device).manual_seed(seed)
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for start in range(0, sample_count, batch_size):
            indices = torch.arange(start, min(start + batch_size, sample_count))
            batch_states = _select_batch(states, indices, resolved_device)
            batch_actions = _select_batch(clean_actions, indices, resolved_device)
            loss = diffusion_noise_loss(
                model,
                schedule,
                batch_states,
                batch_actions,
                noise_generator,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("扩散验证损失出现非有限值。")
            examples = int(batch_states.shape[0])
            total_loss += float(loss.cpu()) * examples
            total_examples += examples
    model.train(was_training)
    return total_loss / total_examples


def save_dppo_checkpoint(
    checkpoint_path: str | Path,
    model: ConditionalDiffusionMLP,
    optimizer: torch.optim.Optimizer,
    metadata: DPPOCheckpointMetadata,
    *,
    epoch: int,
) -> None:
    """保存扩散网络、Adam 状态、完整元数据和随机状态。"""

    if not isinstance(model, ConditionalDiffusionMLP):
        raise TypeError("model 必须是 ConditionalDiffusionMLP。")
    if not isinstance(optimizer, torch.optim.Adam):
        raise TypeError("首版 DPPO 检查点只支持 torch.optim.Adam。")
    if not isinstance(metadata, DPPOCheckpointMetadata):
        raise TypeError("metadata 必须是 DPPOCheckpointMetadata。")
    if model.state_dim != metadata.state_dim or model.action_dim != metadata.action_dim:
        raise ValueError("模型维度与检查点元数据不一致。")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch 必须是非负整数。")
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": "dppo-checkpoint-v1",
        "metadata": asdict(metadata),
        "model_hidden_dims": tuple(model.hidden_dims),
        "model_state_dict": model.state_dict(),
        "optimizer_class": "Adam",
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }
    torch.save(payload, path)


def load_dppo_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected: DPPOCheckpointMetadata,
    device: str | torch.device,
) -> LoadedDPPOCheckpoint:
    """校验元数据后恢复模型、Adam、epoch 和 PyTorch 随机状态。"""

    resolved_device = resolve_torch_device(device)
    payload = torch.load(
        Path(checkpoint_path),
        map_location=resolved_device,
        weights_only=True,
    )
    if payload.get("format_version") != "dppo-checkpoint-v1":
        raise ValueError("检查点 format_version 不受支持。")
    actual = DPPOCheckpointMetadata(**payload["metadata"])
    validate_checkpoint_metadata(actual, expected)
    model = ConditionalDiffusionMLP(
        actual.state_dim,
        actual.action_dim,
        tuple(int(width) for width in payload["model_hidden_dims"]),
    ).to(resolved_device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if payload.get("optimizer_class") != "Adam":
        raise ValueError("检查点优化器不是受支持的 Adam。")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    epoch = payload["epoch"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("检查点 epoch 必须是非负整数。")
    rng_state = payload["torch_rng_state"].cpu()
    torch.set_rng_state(rng_state)
    if resolved_device.type == "cuda" and payload["cuda_rng_states"]:
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in payload["cuda_rng_states"]]
        )
    return LoadedDPPOCheckpoint(
        model=model,
        optimizer=optimizer,
        metadata=actual,
        epoch=epoch,
        rng_state=rng_state.clone(),
    )


def save_dppo_online_checkpoint(
    checkpoint_path: str | Path,
    agent: DPPOAgent,
    metadata: DPPOCheckpointMetadata,
    *,
    iteration: int,
    best_mean_reward: float,
    stability_profile: DPPOStabilityProfile,
) -> None:
    """保存可继续训练的完整 DDPO 双层策略、价值网络和优化器。"""

    if not isinstance(agent, DPPOAgent):
        raise TypeError("agent 必须是 DPPOAgent。")
    if not isinstance(metadata, DPPOCheckpointMetadata):
        raise TypeError("metadata 必须是 DPPOCheckpointMetadata。")
    validate_dppo_stability_binding(agent, metadata, stability_profile)
    if agent.state_dim != metadata.state_dim or agent.action_dim != metadata.action_dim:
        raise ValueError("智能体维度与在线检查点元数据不一致。")
    if (
        agent.config.diffusion_steps != metadata.diffusion_steps
        or agent.config.fine_tuned_steps != metadata.fine_tuned_steps
    ):
        raise ValueError("智能体去噪步数与在线检查点元数据不一致。")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("iteration 必须是非负整数。")
    reward = float(best_mean_reward)
    if not math.isfinite(reward):
        raise ValueError("best_mean_reward 必须是有限数。")

    profile_json = stability_profile_json(stability_profile)
    profile_digest = stability_profile_sha256(stability_profile)

    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": "dppo-online-checkpoint-v2",
        "metadata": asdict(metadata),
        "config": asdict(agent.config),
        "stability_profile_json": profile_json,
        "stability_profile_sha256": profile_digest,
        "model_hidden_dims": tuple(agent.frozen_policy.hidden_dims),
        # 固定前段和可训练末段必须分别保存，恢复后不能混成一个网络。
        "frozen_policy_state_dict": agent.frozen_policy.state_dict(),
        "trainable_policy_state_dict": agent.trainable_policy.state_dict(),
        "value_network_state_dict": agent.value_network.state_dict(),
        "policy_optimizer_state_dict": agent.policy_optimizer.state_dict(),
        "value_optimizer_state_dict": agent.value_optimizer.state_dict(),
        "iteration": iteration,
        "best_mean_reward": reward,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }
    torch.save(payload, path)


def load_dppo_online_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected: DPPOCheckpointMetadata,
    device: str | torch.device,
) -> LoadedDPPOOnlineCheckpoint:
    """校验元数据后恢复可继续训练和独立评估的完整 DDPO 智能体。"""

    resolved_device = resolve_torch_device(device)
    payload = torch.load(
        Path(checkpoint_path),
        map_location=resolved_device,
        weights_only=True,
    )
    format_version = payload.get("format_version")
    if format_version == "dppo-online-checkpoint-v1":
        # v1 没有校准 profile，无法证明恢复出的智能体符合正式训练稳定性门禁。
        raise ValueError("在线检查点 v1 缺少正式训练稳定性绑定。")
    if format_version != "dppo-online-checkpoint-v2":
        raise ValueError("在线检查点 format_version 不受支持。")
    actual = DPPOCheckpointMetadata(**payload["metadata"])
    validate_checkpoint_metadata(actual, expected)
    config = DPPOConfig(**payload["config"])
    if (
        config.diffusion_steps != actual.diffusion_steps
        or config.fine_tuned_steps != actual.fine_tuned_steps
    ):
        raise ValueError("在线检查点配置与元数据的去噪步数不一致。")
    serialized_profile = payload.get("stability_profile_json")
    if not isinstance(serialized_profile, str):
        raise ValueError("在线检查点缺少稳定性配置 JSON。")
    profile = parse_stability_profile_json(serialized_profile)
    if serialized_profile != stability_profile_json(profile):
        raise ValueError("在线检查点稳定性配置 JSON 不是规范格式。")
    stored_profile_digest = payload.get("stability_profile_sha256")
    actual_profile_digest = stability_profile_sha256(profile)
    if stored_profile_digest != actual_profile_digest:
        raise ValueError("在线检查点稳定性配置 SHA256 不匹配。")
    _validate_profile_config_binding(profile, config, actual)

    frozen_policy = ConditionalDiffusionMLP(
        actual.state_dim,
        actual.action_dim,
        tuple(int(width) for width in payload["model_hidden_dims"]),
    ).to(resolved_device)
    frozen_policy.load_state_dict(payload["frozen_policy_state_dict"], strict=True)
    agent = DPPOAgent(
        frozen_policy,
        CosineNoiseSchedule(actual.diffusion_steps),
        config,
        device=resolved_device,
    )
    agent.frozen_policy.load_state_dict(
        payload["frozen_policy_state_dict"],
        strict=True,
    )
    agent.trainable_policy.load_state_dict(
        payload["trainable_policy_state_dict"],
        strict=True,
    )
    agent.value_network.load_state_dict(
        payload["value_network_state_dict"],
        strict=True,
    )
    agent.policy_optimizer.load_state_dict(payload["policy_optimizer_state_dict"])
    agent.value_optimizer.load_state_dict(payload["value_optimizer_state_dict"])

    iteration = payload["iteration"]
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("在线检查点 iteration 必须是非负整数。")
    best_mean_reward = float(payload["best_mean_reward"])
    if not math.isfinite(best_mean_reward):
        raise ValueError("在线检查点 best_mean_reward 必须是有限数。")
    rng_state = payload["torch_rng_state"].cpu()
    torch.set_rng_state(rng_state)
    if resolved_device.type == "cuda" and payload["cuda_rng_states"]:
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in payload["cuda_rng_states"]]
        )
    return LoadedDPPOOnlineCheckpoint(
        agent=agent,
        metadata=actual,
        iteration=iteration,
        best_mean_reward=best_mean_reward,
        rng_state=rng_state.clone(),
        stability_profile=profile,
        stability_profile_sha256=actual_profile_digest,
    )
