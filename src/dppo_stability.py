"""定义 DPPO 校准结果和可审计稳定性配置文件。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from src.dppo_training_config import DPPOStabilitySettings


STABILITY_PROFILE_SCHEMA_VERSION = "dppo-stability-v1"
_LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _finite_float(name: str, value: Any) -> float:
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


def _string_tuple(name: str, value: Any) -> tuple[str, ...]:
    if not isinstance(value, tuple) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{name} 必须是非空字符串组成的元组。")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} 不能包含重复项。")
    return value


def _optional_metric(name: str, value: Any) -> float | None:
    if value is None:
        return None
    return _finite_float(name, value)


@dataclass(frozen=True)
class DPPOCalibrationCandidateResult:
    """保存一个裁剪率候选的原始统计量和集中计算出的资格结论。"""

    clip_ratio: float
    mean_clip_fraction: float | None
    mean_approximate_kl: float | None
    maximum_approximate_kl: float | None
    optimizer_step_count: int
    finite: bool
    qualified: bool
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        clip_ratio = _finite_float("clip_ratio", self.clip_ratio)
        if not 0.0 < clip_ratio < 1.0:
            raise ValueError("clip_ratio 必须位于 (0, 1)。")
        metrics = (
            _optional_metric("mean_clip_fraction", self.mean_clip_fraction),
            _optional_metric("mean_approximate_kl", self.mean_approximate_kl),
            _optional_metric("maximum_approximate_kl", self.maximum_approximate_kl),
        )
        step_count = _nonnegative_integer(
            "optimizer_step_count",
            self.optimizer_step_count,
        )
        if not isinstance(self.finite, bool) or not isinstance(self.qualified, bool):
            raise ValueError("finite 和 qualified 必须是 bool。")
        all_metrics_finite = all(value is not None for value in metrics)
        if self.finite != all_metrics_finite:
            raise ValueError("finite 必须准确反映三个运行指标是否均为有限数。")
        reasons = _string_tuple("failure_reasons", self.failure_reasons)
        if self.qualified and (
            not self.finite or step_count < 1 or bool(reasons)
        ):
            raise ValueError("qualified=True 与候选统计量或失败原因矛盾。")

        object.__setattr__(self, "clip_ratio", clip_ratio)
        object.__setattr__(self, "mean_clip_fraction", metrics[0])
        object.__setattr__(self, "mean_approximate_kl", metrics[1])
        object.__setattr__(self, "maximum_approximate_kl", metrics[2])
        object.__setattr__(self, "optimizer_step_count", step_count)
        object.__setattr__(self, "failure_reasons", reasons)


def _normalize_metric(name: str, value: Any) -> tuple[float | None, str | None]:
    """把非有限运行指标转成 JSON 的 null，同时保留具体失败字段。"""

    try:
        return _finite_float(name, value), None
    except (TypeError, ValueError):
        return None, f"{name} 必须是有限数。"


def evaluate_calibration_candidate(
    *,
    clip_ratio: float,
    mean_clip_fraction: float,
    mean_approximate_kl: float,
    maximum_approximate_kl: float,
    optimizer_step_count: int,
    settings: DPPOStabilitySettings,
) -> DPPOCalibrationCandidateResult:
    """只依据原始统计量计算资格，调用者传入的旧结论不会参与判断。"""

    if not isinstance(settings, DPPOStabilitySettings):
        raise ValueError("settings 必须是 DPPOStabilitySettings。")
    normalized_metrics: dict[str, float | None] = {}
    failure_reasons: list[str] = []
    for name, raw_value in (
        ("mean_clip_fraction", mean_clip_fraction),
        ("mean_approximate_kl", mean_approximate_kl),
        ("maximum_approximate_kl", maximum_approximate_kl),
    ):
        normalized, reason = _normalize_metric(name, raw_value)
        normalized_metrics[name] = normalized
        if reason is not None:
            failure_reasons.append(reason)

    step_count = _nonnegative_integer(
        "optimizer_step_count",
        optimizer_step_count,
    )
    maximum_kl = normalized_metrics["maximum_approximate_kl"]
    if maximum_kl is not None and maximum_kl >= settings.target_kl:
        failure_reasons.append(
            "最大 approximate KL 必须严格小于 target KL。"
        )
    mean_clip = normalized_metrics["mean_clip_fraction"]
    if mean_clip is not None and not (
        settings.target_clip_fraction_min
        <= mean_clip
        <= settings.target_clip_fraction_max
    ):
        failure_reasons.append("mean clip fraction 不在目标闭区间内。")
    if step_count < 1:
        failure_reasons.append("优化器步数必须至少为 1。")

    finite = all(value is not None for value in normalized_metrics.values())
    reasons = tuple(failure_reasons)
    return DPPOCalibrationCandidateResult(
        clip_ratio=clip_ratio,
        mean_clip_fraction=normalized_metrics["mean_clip_fraction"],
        mean_approximate_kl=normalized_metrics["mean_approximate_kl"],
        maximum_approximate_kl=normalized_metrics["maximum_approximate_kl"],
        optimizer_step_count=step_count,
        finite=finite,
        qualified=not reasons,
        failure_reasons=reasons,
    )


@dataclass(frozen=True)
class DPPOStabilityProfile:
    """封装训练前校准的输入身份、候选统计量和最终裁剪率。"""

    schema_version: str
    qualified: bool
    selected_clip_ratio: float | None
    config_hash: str
    pretrained_checkpoint_sha256: str
    training_sampling_min_std: float
    probability_min_std: float
    evaluation_sampling_min_std: float
    target_kl: float
    target_clip_fraction_min: float
    target_clip_fraction_max: float
    candidate_values: tuple[float, ...]
    episode_seeds: tuple[int, ...]
    iterations_per_candidate: int
    episodes_per_iteration: int
    candidate_results: tuple[DPPOCalibrationCandidateResult, ...]
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != STABILITY_PROFILE_SCHEMA_VERSION:
            raise ValueError(f"未知稳定性配置 schema：{self.schema_version!r}。")
        if not isinstance(self.qualified, bool):
            raise ValueError("qualified 必须是 bool。")
        if not isinstance(self.config_hash, str) or not self.config_hash.strip():
            raise ValueError("config_hash 必须是非空字符串。")
        if (
            not isinstance(self.pretrained_checkpoint_sha256, str)
            or _LOWER_SHA256.fullmatch(self.pretrained_checkpoint_sha256) is None
        ):
            raise ValueError("pretrained_checkpoint_sha256 必须是 64 位小写十六进制。")

        positive_names = (
            "training_sampling_min_std",
            "probability_min_std",
            "evaluation_sampling_min_std",
            "target_kl",
        )
        normalized_positive: dict[str, float] = {}
        for name in positive_names:
            value = _finite_float(name, getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"{name} 必须是正有限数。")
            normalized_positive[name] = value
        clip_min = _finite_float(
            "target_clip_fraction_min",
            self.target_clip_fraction_min,
        )
        clip_max = _finite_float(
            "target_clip_fraction_max",
            self.target_clip_fraction_max,
        )
        if not 0.0 <= clip_min <= clip_max <= 1.0:
            raise ValueError("目标 clip fraction 区间必须满足 0 <= min <= max <= 1。")

        if not isinstance(self.candidate_values, tuple) or not self.candidate_values:
            raise ValueError("candidate_values 必须是非空元组。")
        candidates = tuple(
            _finite_float("candidate_values 中的候选值", value)
            for value in self.candidate_values
        )
        if any(not 0.0 < value < 1.0 for value in candidates):
            raise ValueError("candidate_values 中每项必须位于 (0, 1)。")
        if len(candidates) != len(set(candidates)):
            raise ValueError("candidate_values 不能重复。")

        if not isinstance(self.episode_seeds, tuple) or not self.episode_seeds:
            raise ValueError("episode_seeds 必须是非空元组。")
        seeds = tuple(
            _nonnegative_integer("episode_seeds 中的种子", seed)
            for seed in self.episode_seeds
        )
        if len(seeds) != len(set(seeds)):
            raise ValueError("episode_seeds 不能重复。")
        iterations = _positive_integer(
            "iterations_per_candidate",
            self.iterations_per_candidate,
        )
        episodes = _positive_integer(
            "episodes_per_iteration",
            self.episodes_per_iteration,
        )

        if not isinstance(self.candidate_results, tuple) or any(
            not isinstance(result, DPPOCalibrationCandidateResult)
            for result in self.candidate_results
        ):
            raise ValueError("candidate_results 必须是候选结果元组。")
        result_values = tuple(result.clip_ratio for result in self.candidate_results)
        if len(result_values) != len(set(result_values)) or set(result_values) != set(
            candidates
        ):
            raise ValueError("candidate_results 必须与 candidate_values 一一对应。")

        selected = self.selected_clip_ratio
        if selected is not None:
            selected = _finite_float("selected_clip_ratio", selected)
        reasons = _string_tuple("failure_reasons", self.failure_reasons)
        if self.qualified:
            if selected is None or selected not in candidates or reasons:
                raise ValueError("qualified=True 时必须选择候选且不能有失败原因。")
            selected_result = next(
                result for result in self.candidate_results
                if result.clip_ratio == selected
            )
            if not selected_result.qualified:
                raise ValueError("selected_clip_ratio 对应候选并未通过校准。")
        elif selected is not None or not reasons:
            raise ValueError("qualified=False 时 selected 必须为空且失败原因不能为空。")

        for name, value in normalized_positive.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "target_clip_fraction_min", clip_min)
        object.__setattr__(self, "target_clip_fraction_max", clip_max)
        object.__setattr__(self, "candidate_values", candidates)
        object.__setattr__(self, "episode_seeds", seeds)
        object.__setattr__(self, "iterations_per_candidate", iterations)
        object.__setattr__(self, "episodes_per_iteration", episodes)
        object.__setattr__(self, "selected_clip_ratio", selected)
        object.__setattr__(self, "failure_reasons", reasons)

        # qualified、失败原因和最终选择都只是由原始指标推导出的缓存，
        # profile 即使来自本地文件也不能信任这些持久化结论。
        _verify_profile_integrity(self)


def _verify_profile_integrity(profile: DPPOStabilityProfile) -> None:
    """重算候选和 profile 派生结论，拒绝指标与结论不一致的篡改。"""

    if not isinstance(profile.candidate_results, tuple) or any(
        not isinstance(result, DPPOCalibrationCandidateResult)
        for result in profile.candidate_results
    ):
        raise ValueError("稳定性配置完整性校验失败：候选结果结构无效。")
    result_values = tuple(
        result.clip_ratio for result in profile.candidate_results
    )
    if (
        len(result_values) != len(set(result_values))
        or set(result_values) != set(profile.candidate_values)
    ):
        raise ValueError(
            "稳定性配置完整性校验失败：候选结果与候选配置不再一一对应。"
        )

    # DPPOStabilitySettings 的 seed_start 不参与候选资格公式；这里取首个已校验种子，
    # 只为复用唯一的候选评估入口，避免复制两套阈值判断逻辑。
    settings = DPPOStabilitySettings(
        training_sampling_min_std=profile.training_sampling_min_std,
        probability_min_std=profile.probability_min_std,
        evaluation_sampling_min_std=profile.evaluation_sampling_min_std,
        target_kl=profile.target_kl,
        target_clip_fraction_min=profile.target_clip_fraction_min,
        target_clip_fraction_max=profile.target_clip_fraction_max,
        clip_ratio_candidates=profile.candidate_values,
        calibration_iterations=profile.iterations_per_candidate,
        calibration_episodes_per_iteration=profile.episodes_per_iteration,
        calibration_seed_start=profile.episode_seeds[0],
    )
    recomputed_results = tuple(
        evaluate_calibration_candidate(
            clip_ratio=result.clip_ratio,
            mean_clip_fraction=result.mean_clip_fraction,
            mean_approximate_kl=result.mean_approximate_kl,
            maximum_approximate_kl=result.maximum_approximate_kl,
            optimizer_step_count=result.optimizer_step_count,
            settings=settings,
        )
        for result in profile.candidate_results
    )
    for persisted, recomputed in zip(
        profile.candidate_results,
        recomputed_results,
        strict=True,
    ):
        if persisted != recomputed:
            raise ValueError(
                "稳定性配置完整性校验失败："
                f"候选 clip_ratio={persisted.clip_ratio} 的派生字段与原始指标不一致。"
            )

    qualified_values = tuple(
        result.clip_ratio for result in recomputed_results if result.qualified
    )
    expected_selected = max(qualified_values) if qualified_values else None
    expected_qualified = expected_selected is not None
    expected_reasons = (
        () if expected_qualified else ("没有候选项通过稳定性校准。",)
    )
    if (
        profile.qualified != expected_qualified
        or profile.selected_clip_ratio != expected_selected
        or profile.failure_reasons != expected_reasons
    ):
        raise ValueError(
            "稳定性配置完整性校验失败："
            "profile 的 qualified、selected_clip_ratio 或 failure_reasons 与候选重算结果不一致。"
        )


def select_stability_profile(
    *,
    config_hash: str,
    pretrained_checkpoint_sha256: str,
    settings: DPPOStabilitySettings,
    episode_seeds: Sequence[int],
    candidate_results: Sequence[DPPOCalibrationCandidateResult],
) -> DPPOStabilityProfile:
    """重算全部候选资格，并选择通过校准的最大裁剪率。"""

    if not isinstance(settings, DPPOStabilitySettings):
        raise ValueError("settings 必须是 DPPOStabilitySettings。")
    raw_results = tuple(candidate_results)
    if any(not isinstance(item, DPPOCalibrationCandidateResult) for item in raw_results):
        raise ValueError("候选结果类型无效。")
    raw_values = tuple(item.clip_ratio for item in raw_results)
    if len(raw_values) != len(set(raw_values)) or set(raw_values) != set(
        settings.clip_ratio_candidates
    ):
        raise ValueError("候选结果必须与配置候选一一对应且不能重复。")

    # 先按配置顺序建立规范结果，再从原始指标重算，完全忽略外部 qualified 标志。
    by_value = {result.clip_ratio: result for result in raw_results}
    recomputed = tuple(
        evaluate_calibration_candidate(
            clip_ratio=value,
            mean_clip_fraction=by_value[value].mean_clip_fraction,
            mean_approximate_kl=by_value[value].mean_approximate_kl,
            maximum_approximate_kl=by_value[value].maximum_approximate_kl,
            optimizer_step_count=by_value[value].optimizer_step_count,
            settings=settings,
        )
        for value in settings.clip_ratio_candidates
    )
    qualified_values = tuple(
        result.clip_ratio for result in recomputed if result.qualified
    )
    selected = max(qualified_values) if qualified_values else None
    profile_reasons = () if selected is not None else ("没有候选项通过稳定性校准。",)

    return DPPOStabilityProfile(
        schema_version=STABILITY_PROFILE_SCHEMA_VERSION,
        qualified=selected is not None,
        selected_clip_ratio=selected,
        config_hash=config_hash,
        pretrained_checkpoint_sha256=pretrained_checkpoint_sha256,
        training_sampling_min_std=settings.training_sampling_min_std,
        probability_min_std=settings.probability_min_std,
        evaluation_sampling_min_std=settings.evaluation_sampling_min_std,
        target_kl=settings.target_kl,
        target_clip_fraction_min=settings.target_clip_fraction_min,
        target_clip_fraction_max=settings.target_clip_fraction_max,
        candidate_values=settings.clip_ratio_candidates,
        episode_seeds=tuple(episode_seeds),
        iterations_per_candidate=settings.calibration_iterations,
        episodes_per_iteration=settings.calibration_episodes_per_iteration,
        candidate_results=recomputed,
        failure_reasons=profile_reasons,
    )


def sha256_file(path: str | Path) -> str:
    """分块读取文件，避免大型检查点一次性进入内存。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stability_profile_json(profile: DPPOStabilityProfile) -> str:
    """返回不受平台换行风格影响的规范 JSON 文本。"""

    if not isinstance(profile, DPPOStabilityProfile):
        raise ValueError("profile 必须是 DPPOStabilityProfile。")
    return json.dumps(
        asdict(profile),
        sort_keys=True,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON 不允许非有限数：{value}。")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 包含重复字段：{key}。")
        result[key] = value
    return result


def _exact_keys(name: str, value: Any, expected: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} 必须是 JSON 对象。")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} 结构错误：缺少={sorted(expected - actual)}，"
            f"未知={sorted(actual - expected)}。"
        )
    return value


_PROFILE_FIELDS = {
    "schema_version",
    "qualified",
    "selected_clip_ratio",
    "config_hash",
    "pretrained_checkpoint_sha256",
    "training_sampling_min_std",
    "probability_min_std",
    "evaluation_sampling_min_std",
    "target_kl",
    "target_clip_fraction_min",
    "target_clip_fraction_max",
    "candidate_values",
    "episode_seeds",
    "iterations_per_candidate",
    "episodes_per_iteration",
    "candidate_results",
    "failure_reasons",
}
_CANDIDATE_FIELDS = {
    "clip_ratio",
    "mean_clip_fraction",
    "mean_approximate_kl",
    "maximum_approximate_kl",
    "optimizer_step_count",
    "finite",
    "qualified",
    "failure_reasons",
}


def _json_list(name: str, value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} 必须是 JSON 数组。")
    return value


def parse_stability_profile_json(serialized: str) -> DPPOStabilityProfile:
    """严格恢复 profile；未知字段、重复键和非法类型均不会被默默修复。"""

    if not isinstance(serialized, str):
        raise ValueError("serialized 必须是字符串。")
    try:
        raw = json.loads(
            serialized,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"稳定性配置 JSON 无效：{error}。") from error
    data = _exact_keys("稳定性配置", raw, _PROFILE_FIELDS)

    raw_results = _json_list("candidate_results", data["candidate_results"])
    results: list[DPPOCalibrationCandidateResult] = []
    for index, raw_result in enumerate(raw_results):
        item = _exact_keys(
            f"candidate_results[{index}]",
            raw_result,
            _CANDIDATE_FIELDS,
        )
        reasons = _json_list(
            f"candidate_results[{index}].failure_reasons",
            item["failure_reasons"],
        )
        results.append(
            DPPOCalibrationCandidateResult(
                clip_ratio=item["clip_ratio"],
                mean_clip_fraction=item["mean_clip_fraction"],
                mean_approximate_kl=item["mean_approximate_kl"],
                maximum_approximate_kl=item["maximum_approximate_kl"],
                optimizer_step_count=item["optimizer_step_count"],
                finite=item["finite"],
                qualified=item["qualified"],
                failure_reasons=tuple(reasons),
            )
        )

    return DPPOStabilityProfile(
        schema_version=data["schema_version"],
        qualified=data["qualified"],
        selected_clip_ratio=data["selected_clip_ratio"],
        config_hash=data["config_hash"],
        pretrained_checkpoint_sha256=data["pretrained_checkpoint_sha256"],
        training_sampling_min_std=data["training_sampling_min_std"],
        probability_min_std=data["probability_min_std"],
        evaluation_sampling_min_std=data["evaluation_sampling_min_std"],
        target_kl=data["target_kl"],
        target_clip_fraction_min=data["target_clip_fraction_min"],
        target_clip_fraction_max=data["target_clip_fraction_max"],
        candidate_values=tuple(_json_list("candidate_values", data["candidate_values"])),
        episode_seeds=tuple(_json_list("episode_seeds", data["episode_seeds"])),
        iterations_per_candidate=data["iterations_per_candidate"],
        episodes_per_iteration=data["episodes_per_iteration"],
        candidate_results=tuple(results),
        failure_reasons=tuple(
            _json_list("failure_reasons", data["failure_reasons"])
        ),
    )


def save_stability_profile(path: str | Path, profile: DPPOStabilityProfile) -> None:
    """以 UTF-8 和单个结尾换行保存规范 profile。"""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(stability_profile_json(profile) + "\n", encoding="utf-8")


def load_stability_profile(path: str | Path) -> DPPOStabilityProfile:
    return parse_stability_profile_json(Path(path).read_text(encoding="utf-8"))


def stability_profile_sha256(profile: DPPOStabilityProfile) -> str:
    """只摘要规范 JSON 字节，不把文件系统结尾换行纳入身份。"""

    return hashlib.sha256(stability_profile_json(profile).encode("utf-8")).hexdigest()


def validate_stability_profile(
    profile: DPPOStabilityProfile,
    *,
    config_hash: str,
    pretrained_checkpoint_path: str | Path,
    settings: DPPOStabilitySettings,
) -> None:
    """在训练前验证 profile 身份及全部稳定性参数是否仍与当前配置一致。"""

    if not isinstance(profile, DPPOStabilityProfile):
        raise ValueError("稳定性配置类型无效。")
    # 再次独立重算，防止调用者通过 object.__setattr__ 绕过 frozen 数据类。
    _verify_profile_integrity(profile)
    if not profile.qualified:
        raise ValueError("稳定性配置未通过校准，不能用于训练。")
    if profile.schema_version != STABILITY_PROFILE_SCHEMA_VERSION:
        raise ValueError("稳定性配置 schema 不匹配。")
    if profile.config_hash != config_hash:
        raise ValueError("稳定性配置的配置哈希不匹配。")
    actual_checkpoint_sha = sha256_file(pretrained_checkpoint_path)
    if profile.pretrained_checkpoint_sha256 != actual_checkpoint_sha:
        raise ValueError("稳定性配置的预训练检查点 SHA256 不匹配。")
    if (
        profile.selected_clip_ratio is None
        or profile.selected_clip_ratio not in profile.candidate_values
    ):
        raise ValueError("稳定性配置选择的 clip_ratio 不属于候选集合。")

    expected_fields = {
        "training_sampling_min_std": settings.training_sampling_min_std,
        "probability_min_std": settings.probability_min_std,
        "evaluation_sampling_min_std": settings.evaluation_sampling_min_std,
        "target_kl": settings.target_kl,
        "target_clip_fraction_min": settings.target_clip_fraction_min,
        "target_clip_fraction_max": settings.target_clip_fraction_max,
    }
    for name, expected in expected_fields.items():
        if getattr(profile, name) != expected:
            raise ValueError(f"稳定性配置字段 {name} 与当前 settings 不匹配。")
    if profile.candidate_values != settings.clip_ratio_candidates:
        raise ValueError("稳定性配置的候选集合与当前 settings 不匹配。")
    if profile.iterations_per_candidate != settings.calibration_iterations:
        raise ValueError("稳定性配置的 calibration_iterations 不匹配。")
    if profile.episodes_per_iteration != settings.calibration_episodes_per_iteration:
        raise ValueError("稳定性配置的 calibration_episodes_per_iteration 不匹配。")

    expected_seed_count = (
        settings.calibration_iterations
        * settings.calibration_episodes_per_iteration
    )
    expected_seeds = tuple(
        range(
            settings.calibration_seed_start,
            settings.calibration_seed_start + expected_seed_count,
        )
    )
    if profile.episode_seeds != expected_seeds:
        raise ValueError("稳定性配置的校准 Episode 种子与当前 settings 不匹配。")
