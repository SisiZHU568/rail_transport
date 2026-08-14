"""阶段 E 固定验证的公共噪声、微平均指标与检查点选择。"""

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Callable


def _nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} 必须是非负整数。")
    return value


def _nonnegative_float(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是非负有限数。")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{name} 必须是非负有限数。")
    return converted


@dataclass(frozen=True)
class TrajectoryManifestEntry:
    trajectory_id: str
    base_random_seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.trajectory_id, str) or not self.trajectory_id:
            raise ValueError("trajectory_id 不能为空。")
        _nonnegative_int("base_random_seed", self.base_random_seed)


@dataclass(frozen=True)
class TrajectoryManifest:
    """冻结一组外部随机轨迹及逐决策扩散噪声规则。"""

    split: str
    entries: tuple[TrajectoryManifestEntry, ...]
    base_policy_noise_seed: int
    diffusion_repeat_count: int

    def __post_init__(self) -> None:
        if self.split not in {"train", "calibration", "validation", "test"}:
            raise ValueError("未知轨迹清单分区。")
        if not self.entries or any(
            not isinstance(item, TrajectoryManifestEntry) for item in self.entries
        ):
            raise ValueError("轨迹清单必须包含有效条目。")
        ids = [item.trajectory_id for item in self.entries]
        seeds = [item.base_random_seed for item in self.entries]
        if len(set(ids)) != len(ids) or len(set(seeds)) != len(seeds):
            raise ValueError("单个轨迹清单中的 ID 和基础随机种子必须唯一。")
        _nonnegative_int("base_policy_noise_seed", self.base_policy_noise_seed)
        if (
            isinstance(self.diffusion_repeat_count, bool)
            or not isinstance(self.diffusion_repeat_count, int)
            or self.diffusion_repeat_count <= 0
        ):
            raise ValueError("diffusion_repeat_count 必须为正整数。")

    @property
    def sha256(self) -> str:
        payload = {
            "schema": "phase-e-trajectory-manifest-v1",
            "split": self.split,
            "entries": [item.__dict__ for item in self.entries],
            "base_policy_noise_seed": self.base_policy_noise_seed,
            "diffusion_repeat_count": self.diffusion_repeat_count,
        }
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


def validate_manifest_isolation(
    manifests: tuple[TrajectoryManifest, ...],
) -> None:
    """训练、校准、固定验证和最终测试不得复用轨迹或外部随机种子。"""

    if len(manifests) < 2:
        raise ValueError("至少需要两个清单才能审计数据隔离。")
    if len({item.split for item in manifests}) != len(manifests):
        raise ValueError("轨迹清单分区不能重复。")
    seen_ids: set[str] = set()
    seen_seeds: set[int] = set()
    for manifest in manifests:
        if not isinstance(manifest, TrajectoryManifest):
            raise TypeError("manifests 只能包含 TrajectoryManifest。")
        ids = {item.trajectory_id for item in manifest.entries}
        seeds = {item.base_random_seed for item in manifest.entries}
        if seen_ids & ids:
            raise ValueError("不同清单的轨迹 ID 不能重叠。")
        if seen_seeds & seeds:
            raise ValueError("不同清单的基础随机种子不能重叠。")
        seen_ids.update(ids)
        seen_seeds.update(seeds)


@dataclass(frozen=True)
class ValidationRunCounters:
    """保存可跨轨迹直接相加的原始计数，避免先求率再错误平均。"""

    trajectory_count: int
    new_violation_count: int
    risk_batch_count: int
    service_deficit_equivalent_bits: float
    queue_workload_equivalent_bits: float
    raw_resource_cost: float
    internal_failure_count: int

    def __post_init__(self) -> None:
        for name in (
            "trajectory_count",
            "new_violation_count",
            "risk_batch_count",
            "internal_failure_count",
        ):
            object.__setattr__(self, name, _nonnegative_int(name, getattr(self, name)))
        for name in (
            "service_deficit_equivalent_bits",
            "queue_workload_equivalent_bits",
            "raw_resource_cost",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_float(name, getattr(self, name)),
            )
        if self.trajectory_count == 0:
            raise ValueError("trajectory_count 必须为正。")
        if self.new_violation_count > self.risk_batch_count:
            raise ValueError("新增 SLA 违约数不能超过风险批次数。")
        if self.service_deficit_equivalent_bits > self.queue_workload_equivalent_bits:
            raise ValueError("服务缺口不能超过对应队列工作量。")


@dataclass(frozen=True)
class ValidationMetrics:
    violation_rate: float
    deficit_rate: float
    mean_raw_cost_per_trajectory: float
    internal_failure_count: int


def aggregate_validation_runs(
    runs: tuple[ValidationRunCounters, ...],
) -> ValidationMetrics:
    """跨轨迹和扩散重复做微平均，成本按每条轨迹总成本取均值。"""

    if not runs:
        raise ValueError("固定验证至少需要一条运行记录。")
    trajectory_count = sum(item.trajectory_count for item in runs)
    risk_count = sum(item.risk_batch_count for item in runs)
    workload = sum(item.queue_workload_equivalent_bits for item in runs)
    return ValidationMetrics(
        violation_rate=(
            sum(item.new_violation_count for item in runs) / risk_count
            if risk_count
            else 0.0
        ),
        deficit_rate=(
            sum(item.service_deficit_equivalent_bits for item in runs) / workload
            if workload > 0.0
            else 0.0
        ),
        mean_raw_cost_per_trajectory=(
            sum(item.raw_resource_cost for item in runs) / trajectory_count
        ),
        internal_failure_count=sum(item.internal_failure_count for item in runs),
    )


@dataclass(frozen=True)
class ValidationCheckpointResult:
    checkpoint_id: str
    update_index: int
    counters: ValidationRunCounters

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint_id, str) or not self.checkpoint_id:
            raise ValueError("checkpoint_id 不能为空。")
        _nonnegative_int("update_index", self.update_index)
        if not isinstance(self.counters, ValidationRunCounters):
            raise TypeError("counters 必须是 ValidationRunCounters。")

    @property
    def metrics(self) -> ValidationMetrics:
        return aggregate_validation_runs((self.counters,))


@dataclass(frozen=True)
class ValidationSelectionTolerance:
    violation_rate: float
    deficit_rate: float
    raw_cost: float

    def __post_init__(self) -> None:
        for name in ("violation_rate", "deficit_rate", "raw_cost"):
            value = _nonnegative_float(name, getattr(self, name))
            if value == 0.0:
                raise ValueError("验证选择精度必须为正。")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class ValidationCheckpointSelection:
    code: str
    selected: ValidationCheckpointResult | None


@dataclass(frozen=True)
class ValidationCheckpointDescriptor:
    checkpoint_id: str
    update_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint_id, str) or not self.checkpoint_id:
            raise ValueError("checkpoint_id 不能为空。")
        _nonnegative_int("update_index", self.update_index)


@dataclass(frozen=True)
class FixedValidationReport:
    manifest_sha256: str
    checkpoint_results: tuple[ValidationCheckpointResult, ...]
    selection: ValidationCheckpointSelection


@dataclass(frozen=True)
class PolicyNoiseSeedContext:
    trajectory_id: str
    diffusion_repeat_id: int
    base_policy_noise_seed: int

    def for_slow_decision(self, slow_decision_index: int) -> int:
        return derive_policy_noise_seed(
            self.trajectory_id,
            self.diffusion_repeat_id,
            slow_decision_index,
            self.base_policy_noise_seed,
        )


def _quantize(value: float, precision: float) -> int:
    """按预注册精度四舍五入为整数，避免极小浮点差异支配排序。"""

    return math.floor(value / precision + 0.5)


def select_validation_checkpoint(
    candidates: tuple[ValidationCheckpointResult, ...],
    tolerance: ValidationSelectionTolerance,
) -> ValidationCheckpointSelection:
    """先硬过滤内部失败，再按 SLA、缺口、成本和更早更新词典序选择。"""

    if not candidates:
        raise ValueError("候选检查点不能为空。")
    if not isinstance(tolerance, ValidationSelectionTolerance):
        raise TypeError("tolerance 必须是 ValidationSelectionTolerance。")
    valid = tuple(
        item for item in candidates if item.metrics.internal_failure_count == 0
    )
    if not valid:
        return ValidationCheckpointSelection("NO_VALID_CHECKPOINT", None)

    def key(item: ValidationCheckpointResult) -> tuple[int, int, int, int]:
        metrics = item.metrics
        return (
            _quantize(metrics.violation_rate, tolerance.violation_rate),
            _quantize(metrics.deficit_rate, tolerance.deficit_rate),
            _quantize(metrics.mean_raw_cost_per_trajectory, tolerance.raw_cost),
            item.update_index,
        )

    return ValidationCheckpointSelection("OK", min(valid, key=key))


def derive_policy_noise_seed(
    trajectory_id: str,
    diffusion_repeat_id: int,
    slow_decision_index: int,
    base_policy_noise_seed: int,
) -> int:
    """按决策位置派生公共扩散噪声，避免不同排空长度导致随机流错位。"""

    if not isinstance(trajectory_id, str) or not trajectory_id:
        raise ValueError("trajectory_id 不能为空。")
    for name, value in (
        ("diffusion_repeat_id", diffusion_repeat_id),
        ("slow_decision_index", slow_decision_index),
        ("base_policy_noise_seed", base_policy_noise_seed),
    ):
        _nonnegative_int(name, value)
    payload = json.dumps(
        [
            trajectory_id,
            diffusion_repeat_id,
            slow_decision_index,
            base_policy_noise_seed,
        ],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


ValidationEvaluator = Callable[
    [str, TrajectoryManifestEntry, int, PolicyNoiseSeedContext],
    ValidationRunCounters,
]


def run_fixed_validation(
    checkpoints: tuple[ValidationCheckpointDescriptor, ...],
    manifest: TrajectoryManifest,
    tolerance: ValidationSelectionTolerance,
    evaluator: ValidationEvaluator,
) -> FixedValidationReport:
    """在同一轨迹与逐决策噪声规则上评估所有候选，再统一选择。"""

    if not checkpoints:
        raise ValueError("固定验证至少需要一个检查点。")
    if len({item.checkpoint_id for item in checkpoints}) != len(checkpoints):
        raise ValueError("固定验证检查点 ID 不能重复。")
    if manifest.split != "validation":
        raise ValueError("固定验证只能使用 validation 清单。")
    results: list[ValidationCheckpointResult] = []
    for checkpoint in checkpoints:
        runs: list[ValidationRunCounters] = []
        for entry in manifest.entries:
            for repeat_id in range(manifest.diffusion_repeat_count):
                # 每次环境运行必须从相同初始状态开始；上下文按慢决策索引
                # 独立派生噪声，排空长度不同也不会让后续随机流错位。
                seed_context = PolicyNoiseSeedContext(
                    entry.trajectory_id,
                    repeat_id,
                    manifest.base_policy_noise_seed,
                )
                run = evaluator(
                    checkpoint.checkpoint_id,
                    entry,
                    repeat_id,
                    seed_context,
                )
                if not isinstance(run, ValidationRunCounters):
                    raise TypeError("验证 evaluator 必须返回 ValidationRunCounters。")
                if run.trajectory_count != 1:
                    raise ValueError("每次 evaluator 调用必须只代表一条轨迹重复。")
                runs.append(run)
        counters = ValidationRunCounters(
            trajectory_count=sum(item.trajectory_count for item in runs),
            new_violation_count=sum(item.new_violation_count for item in runs),
            risk_batch_count=sum(item.risk_batch_count for item in runs),
            service_deficit_equivalent_bits=sum(
                item.service_deficit_equivalent_bits for item in runs
            ),
            queue_workload_equivalent_bits=sum(
                item.queue_workload_equivalent_bits for item in runs
            ),
            raw_resource_cost=sum(item.raw_resource_cost for item in runs),
            internal_failure_count=sum(
                item.internal_failure_count for item in runs
            ),
        )
        results.append(
            ValidationCheckpointResult(
                checkpoint.checkpoint_id,
                checkpoint.update_index,
                counters,
            )
        )
    checkpoint_results = tuple(results)
    return FixedValidationReport(
        manifest.sha256,
        checkpoint_results,
        select_validation_checkpoint(checkpoint_results, tolerance),
    )
