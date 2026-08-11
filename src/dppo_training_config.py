"""把 DPPO 的 YAML 配置集中转换为算法层不可变配置。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

from src.dppo import DPPOConfig


_STABILITY_KEYS = frozenset(
    {
        "training_sampling_min_std",
        "probability_min_std",
        "evaluation_sampling_min_std",
        "target_kl",
        "target_clip_fraction_min",
        "target_clip_fraction_max",
        "clip_ratio_candidates",
        "calibration_iterations",
        "calibration_episodes_per_iteration",
        "calibration_seed_start",
    }
)


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"配置项 {key} 必须是字典。")
    return value


def _finite_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是有限数。")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} 必须是有限数。")
    return converted


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} 必须是正整数。")
    return value


def _nonnegative_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} 必须是非负整数。")
    return value


@dataclass(frozen=True)
class DPPOStabilitySettings:
    """保存校准、训练概率计算和独立评估共同依赖的稳定性边界。"""

    training_sampling_min_std: float
    probability_min_std: float
    evaluation_sampling_min_std: float
    target_kl: float
    target_clip_fraction_min: float
    target_clip_fraction_max: float
    clip_ratio_candidates: tuple[float, ...]
    calibration_iterations: int
    calibration_episodes_per_iteration: int
    calibration_seed_start: int

    def __post_init__(self) -> None:
        positive_values = {
            "training_sampling_min_std": self.training_sampling_min_std,
            "probability_min_std": self.probability_min_std,
            "evaluation_sampling_min_std": self.evaluation_sampling_min_std,
            "target_kl": self.target_kl,
        }
        converted: dict[str, float] = {}
        for name, value in positive_values.items():
            number = _finite_number(name, value)
            if number <= 0.0:
                raise ValueError(f"{name} 必须是正有限数。")
            converted[name] = number

        clip_min = _finite_number(
            "target_clip_fraction_min",
            self.target_clip_fraction_min,
        )
        clip_max = _finite_number(
            "target_clip_fraction_max",
            self.target_clip_fraction_max,
        )
        if not 0.0 <= clip_min <= clip_max <= 1.0:
            raise ValueError("目标 clip fraction 区间必须满足 0 <= min <= max <= 1。")

        if not isinstance(self.clip_ratio_candidates, tuple) or not self.clip_ratio_candidates:
            raise ValueError("clip_ratio_candidates 必须是非空元组。")
        candidates = tuple(
            _finite_number("clip_ratio_candidates 中的候选值", value)
            for value in self.clip_ratio_candidates
        )
        if any(not 0.0 < value < 1.0 for value in candidates):
            raise ValueError("clip_ratio_candidates 中每项必须位于 (0, 1)。")
        if len(candidates) != len(set(candidates)):
            raise ValueError("clip_ratio_candidates 不能包含重复值。")

        iterations = _positive_integer(
            "calibration_iterations",
            self.calibration_iterations,
        )
        episodes = _positive_integer(
            "calibration_episodes_per_iteration",
            self.calibration_episodes_per_iteration,
        )
        seed_start = _nonnegative_integer(
            "calibration_seed_start",
            self.calibration_seed_start,
        )

        # frozen 数据类也应保存统一的 float/tuple 表示，避免 JSON 摘要受输入类型影响。
        for name, value in converted.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "target_clip_fraction_min", clip_min)
        object.__setattr__(self, "target_clip_fraction_max", clip_max)
        object.__setattr__(self, "clip_ratio_candidates", candidates)
        object.__setattr__(self, "calibration_iterations", iterations)
        object.__setattr__(self, "calibration_episodes_per_iteration", episodes)
        object.__setattr__(self, "calibration_seed_start", seed_start)


def load_dppo_stability_settings(
    config: Mapping[str, Any],
) -> DPPOStabilitySettings:
    """严格读取 ``dppo.stability``，缺项和未知项都立即报错。"""

    if not isinstance(config, Mapping):
        raise ValueError("config 必须是字典。")
    dppo = _mapping(config, "dppo")
    stability = _mapping(dppo, "stability")
    actual_keys = set(stability)
    if actual_keys != _STABILITY_KEYS:
        missing = sorted(_STABILITY_KEYS - actual_keys)
        unknown = sorted(actual_keys - _STABILITY_KEYS)
        raise ValueError(
            "dppo.stability 字段不完整或包含未知项："
            f"缺少={missing}，未知={unknown}。"
        )

    raw_candidates = stability["clip_ratio_candidates"]
    if not isinstance(raw_candidates, (list, tuple)):
        raise ValueError("clip_ratio_candidates 必须是非空列表。")

    return DPPOStabilitySettings(
        training_sampling_min_std=stability["training_sampling_min_std"],
        probability_min_std=stability["probability_min_std"],
        evaluation_sampling_min_std=stability["evaluation_sampling_min_std"],
        target_kl=stability["target_kl"],
        target_clip_fraction_min=stability["target_clip_fraction_min"],
        target_clip_fraction_max=stability["target_clip_fraction_max"],
        clip_ratio_candidates=tuple(raw_candidates),
        calibration_iterations=stability["calibration_iterations"],
        calibration_episodes_per_iteration=stability[
            "calibration_episodes_per_iteration"
        ],
        calibration_seed_start=stability["calibration_seed_start"],
    )


def build_dppo_agent_config(
    config: Mapping[str, Any],
    *,
    clip_ratio: float,
) -> DPPOConfig:
    """建立唯一 YAML→``DPPOConfig`` 映射，并由调用方指定校准后的裁剪率。"""

    if not isinstance(config, Mapping):
        raise ValueError("config 必须是字典。")
    dppo = _mapping(config, "dppo")
    training = _mapping(dppo, "training")
    diffusion = _mapping(dppo, "diffusion")
    settings = load_dppo_stability_settings(config)
    raw_hidden_dims = training.get("value_hidden_dims")
    if (
        not isinstance(raw_hidden_dims, (list, tuple))
        or not raw_hidden_dims
        or any(
            isinstance(width, bool)
            or not isinstance(width, int)
            or width <= 0
            for width in raw_hidden_dims
        )
    ):
        raise ValueError(
            "dppo.training.value_hidden_dims 必须是非空正整数列表或元组。"
        )
    hidden_dims = tuple(raw_hidden_dims)

    # clip_ratio 故意不读取 training.clip_ratio；后续训练和校准都必须显式选择候选值。
    try:
        return DPPOConfig(
            gamma=training["gamma"],
            gae_lambda=training["gae_lambda"],
            clip_ratio=clip_ratio,
            denoising_discount=training["denoising_discount"],
            policy_learning_rate=training["policy_learning_rate"],
            value_learning_rate=training["value_learning_rate"],
            batch_size=training["batch_size"],
            update_epochs=training["update_epochs"],
            gradient_clip_norm=training["gradient_clip_norm"],
            diffusion_steps=diffusion["steps"],
            fine_tuned_steps=diffusion["fine_tuned_steps"],
            value_hidden_dims=hidden_dims,
            seed=training["seed"],
            training_sampling_min_std=settings.training_sampling_min_std,
            probability_min_std=settings.probability_min_std,
            evaluation_sampling_min_std=settings.evaluation_sampling_min_std,
            target_kl=settings.target_kl,
            normalize_advantages=training["normalize_advantages"],
        )
    except KeyError as error:
        raise ValueError(f"DPPO 训练配置缺少字段：{error.args[0]}。") from error
