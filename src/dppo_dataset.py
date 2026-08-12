"""生成、切分并持久化带版本信息的 DPPO 专家仿真数据。"""

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.dppo_action_space import DecodedFunctionAction
from src.dppo_scenario import build_dppo_scenario
from src.dppo_state_encoder import DPPO_STATE_SCHEMA_VERSION
from src.dppo_teacher import build_simulation_teacher


EXPERT_DATASET_SCHEMA_VERSION = "dppo-expert-v2"
_PARTITION_NAMES = ("train", "validation", "test")


@dataclass(frozen=True, eq=False)
class ExpertTransitionRecord:
    """保存一个教师动作及其真实投影和快层执行结果。"""

    state: np.ndarray
    expert_action: np.ndarray
    teacher_name: str
    episode_seed: int
    slow_step: int
    teacher_proposal_raw_feasible: bool
    expert_action_feasible: bool
    projection_change_ratio: float
    final_feasible: bool
    run_cost: float
    route_cost: float
    cold_start_cost: float
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """复制并冻结数组，同时拒绝会污染训练的非法记录。"""

        state = self._validated_vector(self.state, "state")
        action = self._validated_vector(self.expert_action, "expert_action")
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "expert_action", action)
        if not isinstance(self.teacher_name, str) or not self.teacher_name:
            raise ValueError("teacher_name 必须是非空字符串。")
        for name in ("episode_seed", "slow_step"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} 必须是非负整数。")
        if not isinstance(self.teacher_proposal_raw_feasible, bool):
            raise ValueError("teacher_proposal_raw_feasible 必须是布尔值。")
        if not isinstance(self.expert_action_feasible, bool):
            raise ValueError("expert_action_feasible 必须是布尔值。")
        if not isinstance(self.final_feasible, bool):
            raise ValueError("final_feasible 必须是布尔值。")
        ratio = float(self.projection_change_ratio)
        if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
            raise ValueError("projection_change_ratio 必须位于 [0, 1]。")
        object.__setattr__(self, "projection_change_ratio", ratio)
        for name in ("run_cost", "route_cost", "cold_start_cost"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} 必须是非负有限数。")
            object.__setattr__(self, name, value)
        reasons = tuple(self.rejection_reasons)
        if any(not isinstance(reason, str) or not reason for reason in reasons):
            raise ValueError("rejection_reasons 必须由非空字符串组成。")
        if self.final_feasible and self.expert_action_feasible and reasons:
            raise ValueError("可用于行为克隆的记录不能包含拒绝原因。")
        object.__setattr__(self, "rejection_reasons", reasons)

    @staticmethod
    def _validated_vector(value: object, name: str) -> np.ndarray:
        """把状态或动作统一成只读 float32 一维数组。"""

        try:
            vector = np.asarray(value, dtype=np.float32).copy()
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} 必须是一维数值数组。") from error
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError(f"{name} 必须是一维有限数值数组。")
        vector.setflags(write=False)
        return vector

    def __eq__(self, other: object) -> bool:
        """为包含 NumPy 数组的冻结记录提供无歧义的值比较。"""

        if not isinstance(other, ExpertTransitionRecord):
            return NotImplemented
        scalar_fields = (
            "teacher_name",
            "episode_seed",
            "slow_step",
            "teacher_proposal_raw_feasible",
            "expert_action_feasible",
            "projection_change_ratio",
            "final_feasible",
            "run_cost",
            "route_cost",
            "cold_start_cost",
            "rejection_reasons",
        )
        return (
            np.array_equal(self.state, other.state)
            and np.array_equal(self.expert_action, other.expert_action)
            and all(
                getattr(self, name) == getattr(other, name)
                for name in scalar_fields
            )
        )


@dataclass(frozen=True)
class ExpertDatasetMetadata:
    """保存防止数据、配置和模型模式混用的元数据。"""

    schema_version: str
    state_dim: int
    action_dim: int
    config_hash: str
    state_schema_version: str = DPPO_STATE_SCHEMA_VERSION
    action_schema_version: str = "joint-sfc-continuous-v2"

    def __post_init__(self) -> None:
        """在写盘前验证版本、维度和配置指纹。"""

        for name in (
            "schema_version",
            "config_hash",
            "state_schema_version",
            "action_schema_version",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} 必须是非空字符串。")
        for name in ("state_dim", "action_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} 必须是正整数。")


@dataclass(frozen=True)
class DiagnosticRecord:
    """为不可用于行为克隆的记录补充分区来源。"""

    partition: str
    record: ExpertTransitionRecord


@dataclass(frozen=True)
class LoadedExpertDataset:
    """保存加载后的三个行为克隆分区和全部拒绝诊断。"""

    metadata: ExpertDatasetMetadata
    partitions: Mapping[str, tuple[ExpertTransitionRecord, ...]]
    diagnostics: tuple[DiagnosticRecord, ...]


def compute_config_hash(config: Mapping[str, Any]) -> str:
    """对规范化 JSON 配置计算 SHA-256，键的原始顺序不影响结果。"""

    try:
        serialized = json.dumps(
            config,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("配置必须能够无损序列化为有限 JSON。") from error
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _validated_fractions(fractions: tuple[float, float, float]) -> np.ndarray:
    """验证训练、验证、测试比例并返回浮点数组。"""

    if not isinstance(fractions, tuple) or len(fractions) != 3:
        raise ValueError("fractions 必须依次包含训练、验证和测试三个比例。")
    values = np.asarray(fractions, dtype=np.float64)
    if not np.isfinite(values).all() or not ((0.0 < values) & (values < 1.0)).all():
        raise ValueError("每个数据集比例必须位于 (0, 1)。")
    if not math.isclose(float(values.sum()), 1.0, abs_tol=1e-12):
        raise ValueError("数据集比例之和必须为 1。")
    return values


def _partition_counts(episode_count: int, fractions: np.ndarray) -> np.ndarray:
    """使用最大余数法分配 Episode，并在可能时避免空分区。"""

    raw_counts = fractions * episode_count
    counts = np.floor(raw_counts).astype(np.int64)
    remaining = episode_count - int(counts.sum())
    remainder_order = np.argsort(-(raw_counts - counts), kind="stable")
    for index in remainder_order[:remaining]:
        counts[index] += 1
    if episode_count >= len(_PARTITION_NAMES):
        for empty_index in np.flatnonzero(counts == 0):
            donor_index = int(np.argmax(counts))
            if counts[donor_index] <= 1:
                break
            counts[donor_index] -= 1
            counts[empty_index] += 1
    return counts


def split_records_by_episode(
    records: list[ExpertTransitionRecord] | tuple[ExpertTransitionRecord, ...],
    *,
    fractions: tuple[float, float, float],
    seed: int,
) -> dict[str, tuple[ExpertTransitionRecord, ...]]:
    """按 Episode 随机切分，保证同一轨迹的慢步不会跨分区。"""

    supplied = tuple(records)
    if any(not isinstance(record, ExpertTransitionRecord) for record in supplied):
        raise TypeError("records 必须只包含 ExpertTransitionRecord。")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("切分 seed 必须是非负整数。")
    fraction_values = _validated_fractions(fractions)
    episode_seeds = np.asarray(
        sorted({record.episode_seed for record in supplied}),
        dtype=np.int64,
    )
    shuffled = np.random.default_rng(seed).permutation(episode_seeds)
    counts = _partition_counts(len(shuffled), fraction_values)
    seed_sets: dict[str, set[int]] = {}
    cursor = 0
    for name, count in zip(_PARTITION_NAMES, counts, strict=True):
        next_cursor = cursor + int(count)
        seed_sets[name] = {int(value) for value in shuffled[cursor:next_cursor]}
        cursor = next_cursor
    return {
        name: tuple(
            record for record in supplied if record.episode_seed in seed_sets[name]
        )
        for name in _PARTITION_NAMES
    }


def _validate_partitions(
    partitions: Mapping[str, tuple[ExpertTransitionRecord, ...]],
    metadata: ExpertDatasetMetadata,
) -> dict[str, tuple[ExpertTransitionRecord, ...]]:
    """核对分区名称、记录维度和 Episode 隔离。"""

    if set(partitions) != set(_PARTITION_NAMES):
        raise ValueError("partitions 必须且只能包含 train、validation、test。")
    normalized = {name: tuple(partitions[name]) for name in _PARTITION_NAMES}
    seed_sets: dict[str, set[int]] = {}
    for name, records in normalized.items():
        seed_sets[name] = set()
        for record in records:
            if not isinstance(record, ExpertTransitionRecord):
                raise TypeError("数据分区只能包含 ExpertTransitionRecord。")
            if record.state.shape != (metadata.state_dim,):
                raise ValueError(
                    f"记录 state_dim={record.state.shape[0]}，"
                    f"但元数据要求 state_dim={metadata.state_dim}。"
                )
            if record.expert_action.shape != (metadata.action_dim,):
                raise ValueError(
                    f"记录 action_dim={record.expert_action.shape[0]}，"
                    f"但元数据要求 action_dim={metadata.action_dim}。"
                )
            seed_sets[name].add(record.episode_seed)
    for first_index, first_name in enumerate(_PARTITION_NAMES):
        for second_name in _PARTITION_NAMES[first_index + 1 :]:
            if not seed_sets[first_name].isdisjoint(seed_sets[second_name]):
                raise ValueError("同一个 Episode seed 不能出现在多个数据分区。")
    return normalized


def _records_to_arrays(
    records: tuple[ExpertTransitionRecord, ...],
    metadata: ExpertDatasetMetadata,
) -> dict[str, np.ndarray]:
    """把同构记录转换为不需要 pickle 的 NPZ 数组。"""

    if records:
        states = np.stack([record.state for record in records]).astype(np.float32)
        actions = np.stack([record.expert_action for record in records]).astype(
            np.float32
        )
    else:
        states = np.empty((0, metadata.state_dim), dtype=np.float32)
        actions = np.empty((0, metadata.action_dim), dtype=np.float32)
    return {
        "states": states,
        "expert_actions": actions,
        "teacher_names": np.asarray(
            [record.teacher_name for record in records], dtype=np.str_
        ),
        "episode_seeds": np.asarray(
            [record.episode_seed for record in records], dtype=np.int64
        ),
        "slow_steps": np.asarray(
            [record.slow_step for record in records], dtype=np.int64
        ),
        "teacher_proposal_raw_feasible": np.asarray(
            [record.teacher_proposal_raw_feasible for record in records],
            dtype=np.bool_,
        ),
        "expert_action_feasible": np.asarray(
            [record.expert_action_feasible for record in records], dtype=np.bool_
        ),
        "projection_change_ratios": np.asarray(
            [record.projection_change_ratio for record in records], dtype=np.float64
        ),
        "final_feasible": np.asarray(
            [record.final_feasible for record in records], dtype=np.bool_
        ),
        "run_costs": np.asarray([record.run_cost for record in records]),
        "route_costs": np.asarray([record.route_cost for record in records]),
        "cold_start_costs": np.asarray(
            [record.cold_start_cost for record in records]
        ),
        "rejection_reasons_json": np.asarray(
            [
                json.dumps(record.rejection_reasons, ensure_ascii=False)
                for record in records
            ],
            dtype=np.str_,
        ),
    }


def _arrays_to_records(path: Path) -> tuple[ExpertTransitionRecord, ...]:
    """从单个 NPZ 文件恢复记录，并拒绝对象数组。"""

    with np.load(path, allow_pickle=False) as arrays:
        count = int(arrays["states"].shape[0])
        return tuple(
            ExpertTransitionRecord(
                state=arrays["states"][index],
                expert_action=arrays["expert_actions"][index],
                teacher_name=str(arrays["teacher_names"][index]),
                episode_seed=int(arrays["episode_seeds"][index]),
                slow_step=int(arrays["slow_steps"][index]),
                teacher_proposal_raw_feasible=bool(
                    arrays["teacher_proposal_raw_feasible"][index]
                ),
                expert_action_feasible=bool(
                    arrays["expert_action_feasible"][index]
                ),
                projection_change_ratio=float(
                    arrays["projection_change_ratios"][index]
                ),
                final_feasible=bool(arrays["final_feasible"][index]),
                run_cost=float(arrays["run_costs"][index]),
                route_cost=float(arrays["route_costs"][index]),
                cold_start_cost=float(arrays["cold_start_costs"][index]),
                rejection_reasons=tuple(
                    json.loads(str(arrays["rejection_reasons_json"][index]))
                ),
            )
            for index in range(count)
        )


def save_expert_dataset(
    output_root: str | Path,
    partitions: Mapping[str, tuple[ExpertTransitionRecord, ...]],
    metadata: ExpertDatasetMetadata,
) -> None:
    """保存三个可行行为克隆分区和一份不可行诊断分区。"""

    if not isinstance(metadata, ExpertDatasetMetadata):
        raise TypeError("metadata 必须是 ExpertDatasetMetadata。")
    normalized = _validate_partitions(partitions, metadata)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    diagnostics: list[tuple[str, ExpertTransitionRecord]] = []
    behavior_counts: dict[str, int] = {}
    for name in _PARTITION_NAMES:
        # 行为克隆只学习“标签本身可行且执行结果也可行”的样本。
        accepted = tuple(
            record
            for record in normalized[name]
            if record.expert_action_feasible and record.final_feasible
        )
        diagnostics.extend(
            (name, record)
            for record in normalized[name]
            if not (record.expert_action_feasible and record.final_feasible)
        )
        behavior_counts[name] = len(accepted)
        np.savez_compressed(root / f"{name}.npz", **_records_to_arrays(accepted, metadata))
    diagnostic_records = tuple(record for _, record in diagnostics)
    diagnostic_arrays = _records_to_arrays(diagnostic_records, metadata)
    diagnostic_arrays["partitions"] = np.asarray(
        [name for name, _ in diagnostics], dtype=np.str_
    )
    np.savez_compressed(root / "diagnostics.npz", **diagnostic_arrays)
    metadata_payload = asdict(metadata)
    metadata_payload["behavior_cloning_counts"] = behavior_counts
    metadata_payload["diagnostic_count"] = len(diagnostics)
    (root / "metadata.json").write_text(
        json.dumps(metadata_payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _metadata_from_json(path: Path) -> ExpertDatasetMetadata:
    """只读取元数据类声明的字段，计数属于附加审计信息。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    return ExpertDatasetMetadata(
        **{field.name: payload[field.name] for field in fields(ExpertDatasetMetadata)}
    )


def _validate_metadata(
    actual: ExpertDatasetMetadata,
    expected: ExpertDatasetMetadata,
) -> None:
    """逐字段报告不兼容项，便于定位配置或模式混用。"""

    for field in fields(ExpertDatasetMetadata):
        if getattr(actual, field.name) != getattr(expected, field.name):
            raise ValueError(f"数据集 {field.name} 与当前配置不一致。")


def load_expert_dataset(
    output_root: str | Path,
    *,
    expected_metadata: ExpertDatasetMetadata | None = None,
) -> LoadedExpertDataset:
    """加载行为克隆分区和诊断记录，并可选执行元数据兼容检查。"""

    root = Path(output_root)
    metadata = _metadata_from_json(root / "metadata.json")
    if expected_metadata is not None:
        _validate_metadata(metadata, expected_metadata)
    partitions = {
        name: _arrays_to_records(root / f"{name}.npz")
        for name in _PARTITION_NAMES
    }
    # 加载时再次核对数组形状，防止元数据或 NPZ 被单独替换后静默混用。
    partitions = _validate_partitions(partitions, metadata)
    diagnostic_records = _arrays_to_records(root / "diagnostics.npz")
    _validate_partitions(
        {"train": diagnostic_records, "validation": (), "test": ()},
        metadata,
    )
    with np.load(root / "diagnostics.npz", allow_pickle=False) as arrays:
        partition_names = tuple(str(value) for value in arrays["partitions"])
    diagnostics = tuple(
        DiagnosticRecord(partition=name, record=record)
        for name, record in zip(
            partition_names,
            diagnostic_records,
            strict=True,
        )
    )
    return LoadedExpertDataset(
        metadata=metadata,
        partitions=partitions,
        diagnostics=diagnostics,
    )


def _rejection_reasons(info: Mapping[str, object]) -> tuple[str, ...]:
    """把投影和最终执行违约转换成稳定、可检索的诊断文本。"""

    projection = info["projection_result"]
    reasons = list(projection.reasons)
    metrics = info["final_metrics"]
    for name in (
        "sla_violations",
        "constraint_rejected_batches",
        "fast_repair_failures",
    ):
        value = int(getattr(metrics, name))
        if value > 0:
            reasons.append(f"{name}={value}")
    if not reasons:
        reasons.append("final_feasibility_check_failed")
    return tuple(reasons)


def _reencode_projected_expert_action(
    scenario: Any,
    proposal: Any,
    projection: Any,
    projection_inputs: Any,
) -> tuple[np.ndarray, bool]:
    """把投影后的节点前缀重新编码为可用于行为克隆的完整连续动作。

    投影意图只保存实际副本节点；其余节点仍按教师原排名补齐。这样只修正
    违反硬约束的节点选择，不会额外改变教师的副本数和保留时间设计。
    """

    if not projection.success or projection.function_intents is None:
        return proposal.relaxed_action, False

    original_by_function = {
        action.function_id: action
        for action in proposal.decoded_action.function_actions
    }
    corrected_actions: list[DecodedFunctionAction] = []
    for intent in projection.function_intents:
        original = original_by_function[intent.function_id]
        selected = tuple(intent.preferred_node_ids)
        selected_set = set(selected)
        complete_ranking = selected + tuple(
            node_id
            for node_id in original.ranked_node_ids
            if node_id not in selected_set
        )
        corrected_actions.append(
            DecodedFunctionAction(
                function_id=intent.function_id,
                ranked_node_ids=complete_ranking,
                replica_count=intent.replica_count,
                primary_retention_seconds=intent.primary_retention_seconds,
                backup_retention_seconds=intent.backup_retention_seconds,
            )
        )

    corrected = scenario.action_space.encode_teacher_action(corrected_actions)
    verification = scenario.projector.project(
        decoded_action=scenario.action_space.decode(corrected),
        operational_node_ids=projection_inputs.operational_node_ids,
        free_cpu=projection_inputs.free_cpu,
        free_memory_mb=projection_inputs.free_memory_mb,
        fault_domains=projection_inputs.fault_domains,
    )
    return corrected, bool(verification.raw_feasible)


def collect_expert_records(
    config: dict[str, Any],
    *,
    episode_count: int,
    seed_start: int,
) -> tuple[ExpertTransitionRecord, ...]:
    """使用在线 DPPO 的同一投影与快层执行闭环收集专家记录。"""

    if isinstance(episode_count, bool) or not isinstance(episode_count, int):
        raise ValueError("episode_count 必须是正整数。")
    if episode_count <= 0:
        raise ValueError("episode_count 必须是正整数。")
    if isinstance(seed_start, bool) or not isinstance(seed_start, int) or seed_start < 0:
        raise ValueError("seed_start 必须是非负整数。")
    dataset_config = config["dppo"]["dataset"]
    teacher_names = tuple(dataset_config["teacher_names"])
    if not teacher_names or any(not isinstance(name, str) for name in teacher_names):
        raise ValueError("dppo.dataset.teacher_names 必须是非空字符串列表。")
    maximum_steps = dataset_config["max_slow_steps_per_episode"]
    if (
        isinstance(maximum_steps, bool)
        or not isinstance(maximum_steps, int)
        or maximum_steps <= 0
    ):
        raise ValueError("max_slow_steps_per_episode 必须是正整数。")

    scenario = build_dppo_scenario(config)
    records: list[ExpertTransitionRecord] = []
    for episode_index in range(episode_count):
        episode_seed = seed_start + episode_index
        teacher_name = teacher_names[episode_index % len(teacher_names)]
        teacher = build_simulation_teacher(teacher_name, scenario)
        state = scenario.reset(seed=episode_seed)
        for slow_step in range(maximum_steps):
            proposal = teacher.propose(scenario.current_public_snapshot())
            # 必须在执行窗口前保存投影输入；执行后资源与故障状态已经变化，
            # 不能再用未来快照验证当前教师标签。
            projection_inputs = scenario.execution_core.current_projection_inputs()
            next_state, _, terminated, truncated, info = scenario.step(
                proposal.relaxed_action
            )
            projection = info["projection_result"]
            expert_action, expert_action_feasible = (
                _reencode_projected_expert_action(
                    scenario,
                    proposal,
                    projection,
                    projection_inputs,
                )
            )
            breakdown = info["reward_breakdown"]
            final_feasible = bool(
                projection.success and breakdown.violation_penalty == 0.0
            )
            records.append(
                ExpertTransitionRecord(
                    state=state,
                    expert_action=expert_action,
                    teacher_name=teacher_name,
                    episode_seed=episode_seed,
                    slow_step=slow_step,
                    teacher_proposal_raw_feasible=projection.raw_feasible,
                    expert_action_feasible=expert_action_feasible,
                    projection_change_ratio=projection.change_ratio,
                    final_feasible=final_feasible,
                    run_cost=breakdown.run_cost,
                    route_cost=breakdown.route_cost,
                    cold_start_cost=breakdown.cold_start_cost,
                    rejection_reasons=(
                        ("expert_action_not_raw_feasible",)
                        if not expert_action_feasible
                        else (() if final_feasible else _rejection_reasons(info))
                    ),
                )
            )
            state = next_state
            if terminated or truncated:
                break
    return tuple(records)
