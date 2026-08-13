"""配置驱动的 DPPO 动作规格和确定性全局可完成部署解码。"""

from dataclasses import dataclass
import hashlib
import json
import math
from time import perf_counter
from types import MappingProxyType
from typing import Mapping

import numpy as np

from src.orchestration_config import PhaseAConfig


@dataclass(frozen=True)
class ActionSpecEntry:
    function_id: int
    node_id: int
    action_type: str


@dataclass(frozen=True)
class ActionSpec:
    """动作顺序显式保存，禁止依赖字典遍历顺序。"""

    pairs: tuple[tuple[int, int], ...]
    entries: tuple[ActionSpecEntry, ...]
    retention_slot_options: tuple[int, ...]
    sha256: str

    @classmethod
    def from_phase_a_config(cls, config: PhaseAConfig) -> "ActionSpec":
        pairs = tuple(sorted(config.deployment_pairs))
        entries = tuple(
            entry
            for function_id, node_id in pairs
            for entry in (
                ActionSpecEntry(function_id, node_id, "instance_count_score"),
                ActionSpecEntry(function_id, node_id, "retention_time_score"),
            )
        )
        payload = {
            "pairs": pairs,
            "entries": [entry.__dict__ for entry in entries],
            "retention_slot_options": config.retention_slot_options,
            "schema": "phase-d-action-v1",
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return cls(pairs, entries, config.retention_slot_options, digest)

    @property
    def action_dim(self) -> int:
        return len(self.entries)


@dataclass(frozen=True)
class DecoderInput:
    effective_node_up: Mapping[int, bool]
    locked_instance_counts: Mapping[tuple[int, int], int]
    required_replica_nodes: Mapping[int, int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "effective_node_up", MappingProxyType(dict(self.effective_node_up))
        )
        object.__setattr__(
            self,
            "locked_instance_counts",
            MappingProxyType(dict(self.locked_instance_counts)),
        )
        object.__setattr__(
            self,
            "required_replica_nodes",
            MappingProxyType(dict(self.required_replica_nodes)),
        )


@dataclass(frozen=True)
class DeploymentPlan:
    instance_counts: Mapping[tuple[int, int], int]
    retention_slots: Mapping[tuple[int, int], int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "instance_counts", MappingProxyType(dict(self.instance_counts))
        )
        object.__setattr__(
            self, "retention_slots", MappingProxyType(dict(self.retention_slots))
        )


@dataclass(frozen=True)
class DecoderResult:
    code: str
    plan: DeploymentPlan | None
    effective_action_mask: tuple[bool, ...]
    used_fallback: bool
    expanded_states: int
    pruned_states: int
    elapsed_seconds: float


class SafeDeploymentDecoder:
    """按固定 pair 顺序解码，并用后续可完成性检查屏蔽候选。"""

    def __init__(
        self,
        config: PhaseAConfig,
        action_spec: ActionSpec,
        *,
        function_memory_mb: Mapping[int, float],
        max_expanded_states: int = 200_000,
    ) -> None:
        if action_spec != ActionSpec.from_phase_a_config(config):
            raise ValueError("ActionSpec 与阶段 A 配置不一致。")
        if max_expanded_states <= 0:
            raise ValueError("max_expanded_states 必须为正。")
        self.config = config
        self.action_spec = action_spec
        self.function_memory_mb = MappingProxyType(dict(function_memory_mb))
        self.max_expanded_states = max_expanded_states
        self._expanded = 0
        self._pruned = 0
        self._limit_hit = False

    def _memory_used(
        self,
        assignments: Mapping[tuple[int, int], int],
        decoder_input: DecoderInput,
    ) -> dict[int, float]:
        used = {node_id: 0.0 for node_id in self.config.node_resources}
        for pair, target in assignments.items():
            function_id, node_id = pair
            actual = max(target, decoder_input.locked_instance_counts.get(pair, 0))
            used[node_id] += actual * self.function_memory_mb[function_id]
        return used

    def _partial_possible(
        self,
        assignments: Mapping[tuple[int, int], int],
        next_index: int,
        decoder_input: DecoderInput,
    ) -> bool:
        used = self._memory_used(assignments, decoder_input)
        if any(
            used[node_id] > resource.memory_capacity_mb + 1e-9
            for node_id, resource in self.config.node_resources.items()
        ):
            return False
        remaining_pairs = set(self.action_spec.pairs[next_index:])
        functions = sorted({function_id for function_id, _ in self.action_spec.pairs})
        for function_id in functions:
            selected = sum(
                value > 0
                for (candidate_function, _), value in assignments.items()
                if candidate_function == function_id
            )
            possible_nodes = {
                node_id
                for candidate_function, node_id in remaining_pairs
                if candidate_function == function_id
                and decoder_input.effective_node_up.get(node_id, False)
                and used[node_id] + self.function_memory_mb[function_id]
                <= self.config.node_resources[node_id].memory_capacity_mb + 1e-9
            }
            if selected + len(possible_nodes) < decoder_input.required_replica_nodes.get(
                function_id, 1
            ):
                return False
        return True

    def _complete(
        self,
        index: int,
        assignments: dict[tuple[int, int], int],
        decoder_input: DecoderInput,
    ) -> dict[tuple[int, int], int] | None:
        self._expanded += 1
        if self._expanded > self.max_expanded_states:
            self._limit_hit = True
            return None
        if not self._partial_possible(assignments, index, decoder_input):
            self._pruned += 1
            return None
        if index == len(self.action_spec.pairs):
            return dict(assignments)
        pair = self.action_spec.pairs[index]
        function_id, node_id = pair
        allowed = self.config.allowed_instance_counts(function_id, node_id)
        if not decoder_input.effective_node_up.get(node_id, False):
            allowed = (0,)
        # 可完成性证明只需要找到一个规范解，优先零以保留资源。
        for value in allowed:
            assignments[pair] = value
            result = self._complete(index + 1, assignments, decoder_input)
            if result is not None:
                return result
            if self._limit_hit:
                return None
        assignments.pop(pair, None)
        return None

    @staticmethod
    def _score_index(score: float, size: int) -> int:
        if size <= 1:
            return 0
        return min(size - 1, max(0, int(math.floor(score * size))))

    def decode(self, policy_scores: np.ndarray, decoder_input: DecoderInput) -> DecoderResult:
        started = perf_counter()
        scores = np.asarray(policy_scores, dtype=np.float64)
        if scores.shape != (self.action_spec.action_dim,) or not np.isfinite(scores).all():
            return DecoderResult(
                "INVALID_POLICY_SCORE", None,
                tuple(index % 2 == 0 for index in range(self.action_spec.action_dim)),
                False, 0, 0, perf_counter() - started,
            )
        if np.any(scores < 0.0) or np.any(scores > 1.0):
            return DecoderResult(
                "INVALID_POLICY_SCORE", None,
                tuple(index % 2 == 0 for index in range(self.action_spec.action_dim)),
                False, 0, 0, perf_counter() - started,
            )
        self._expanded = self._pruned = 0
        self._limit_hit = False
        initial = self._complete(0, {}, decoder_input)
        if initial is None:
            code = "DECODER_SEARCH_LIMIT" if self._limit_hit else "NO_SAFE_FEASIBLE_DEPLOYMENT"
            return DecoderResult(
                code, None,
                tuple(index % 2 == 0 for index in range(self.action_spec.action_dim)),
                False, self._expanded, self._pruned, perf_counter() - started,
            )

        assignments: dict[tuple[int, int], int] = {}
        for index, pair in enumerate(self.action_spec.pairs):
            function_id, node_id = pair
            candidates = self.config.allowed_instance_counts(function_id, node_id)
            if not decoder_input.effective_node_up.get(node_id, False):
                candidates = (0,)
            feasible: list[int] = []
            for candidate in candidates:
                trial = {**assignments, pair: candidate}
                if self._complete(index + 1, trial, decoder_input) is not None:
                    feasible.append(candidate)
                if self._limit_hit:
                    return DecoderResult(
                        "DECODER_SEARCH_LIMIT", None,
                        tuple(i % 2 == 0 for i in range(self.action_spec.action_dim)),
                        False, self._expanded, self._pruned, perf_counter() - started,
                    )
            if not feasible:
                return DecoderResult(
                    "DECODER_INTERNAL_FAILURE", None,
                    tuple(i % 2 == 0 for i in range(self.action_spec.action_dim)),
                    False, self._expanded, self._pruned, perf_counter() - started,
                )
            assignments[pair] = feasible[
                self._score_index(float(scores[2 * index]), len(feasible))
            ]

        retention: dict[tuple[int, int], int] = {}
        mask: list[bool] = []
        options = self.action_spec.retention_slot_options
        for index, pair in enumerate(self.action_spec.pairs):
            count = assignments[pair]
            mask.extend((True, count > 0))
            retention[pair] = (
                options[self._score_index(float(scores[2 * index + 1]), len(options))]
                if count > 0 else 0
            )
        return DecoderResult(
            "OK", DeploymentPlan(assignments, retention), tuple(mask), False,
            self._expanded, self._pruned, perf_counter() - started,
        )
