"""配置驱动的 DPPO 动作规格和确定性全局可完成部署解码。"""

from dataclasses import dataclass, field
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
    fault_domain_by_node: Mapping[int, int] = field(default_factory=dict)
    minimum_fault_domains: Mapping[int, int] = field(default_factory=dict)
    domain_availability: Mapping[int, float] = field(default_factory=dict)
    node_conditional_availability: Mapping[int, float] = field(default_factory=dict)
    maximum_vnf_unavailability: Mapping[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "effective_node_up", MappingProxyType(dict(self.effective_node_up))
        )
        object.__setattr__(
            self,
            "locked_instance_counts",
            MappingProxyType(dict(self.locked_instance_counts)),
        )
        for field_name in (
            "required_replica_nodes",
            "fault_domain_by_node",
            "minimum_fault_domains",
            "domain_availability",
            "node_conditional_availability",
            "maximum_vnf_unavailability",
        ):
            object.__setattr__(
                self, field_name, MappingProxyType(dict(getattr(self, field_name)))
            )
        availability_values = (
            *self.domain_availability.values(),
            *self.node_conditional_availability.values(),
            *self.maximum_vnf_unavailability.values(),
        )
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in availability_values):
            raise ValueError("可靠性概率必须位于 [0, 1]。")


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


@dataclass(frozen=True)
class PlanEncodingResult:
    """教师计划反向编码结果；非 OK 时不返回可用于训练的部分评分。"""

    code: str
    scores: np.ndarray


@dataclass(frozen=True)
class PlanEnumerationResult:
    """完整安全计划枚举结果；达到上限时主动丢弃已生成的部分集合。"""

    code: str
    plans: tuple[DeploymentPlan, ...]
    generated_patterns: int
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
            selected_domains = {
                decoder_input.fault_domain_by_node.get(node_id, node_id)
                for (candidate_function, node_id), value in assignments.items()
                if candidate_function == function_id and value > 0
            }
            possible_domains = {
                decoder_input.fault_domain_by_node.get(node_id, node_id)
                for node_id in possible_nodes
            }
            if len(selected_domains | possible_domains) < decoder_input.minimum_fault_domains.get(
                function_id, 1
            ):
                return False
            # 当前 VNF 的全部位置已确定时立即检查可靠性，避免枚举后续 VNF 后才剪枝。
            if not any(pair[0] == function_id for pair in remaining_pairs):
                if not self._function_is_safe(function_id, assignments, decoder_input):
                    return False
        if next_index == len(self.action_spec.pairs):
            return self._complete_is_safe(assignments, decoder_input)
        return True

    @staticmethod
    def _function_is_safe(
        function_id: int,
        assignments: Mapping[tuple[int, int], int],
        decoder_input: DecoderInput,
    ) -> bool:
        nodes = [
            node_id
            for (candidate_function, node_id), count in assignments.items()
            if candidate_function == function_id and count > 0
        ]
        domains: dict[int, list[int]] = {}
        for node_id in nodes:
            domain_id = decoder_input.fault_domain_by_node.get(node_id, node_id)
            domains.setdefault(domain_id, []).append(node_id)
        if len(domains) < decoder_input.minimum_fault_domains.get(function_id, 1):
            return False
        unavailable = 1.0
        for domain_id, domain_nodes in domains.items():
            domain_up = decoder_input.domain_availability.get(domain_id, 1.0)
            all_nodes_down = math.prod(
                1.0 - decoder_input.node_conditional_availability.get(node_id, 1.0)
                for node_id in domain_nodes
            )
            unavailable *= (1.0 - domain_up) + domain_up * all_nodes_down
        return unavailable <= decoder_input.maximum_vnf_unavailability.get(
            function_id, 1.0
        ) + 1e-12

    def _complete_is_safe(
        self,
        assignments: Mapping[tuple[int, int], int],
        decoder_input: DecoderInput,
    ) -> bool:
        """按故障域并集界充分条件检查最终目标部署。"""

        functions = sorted({function_id for function_id, _ in self.action_spec.pairs})
        return all(
            self._function_is_safe(function_id, assignments, decoder_input)
            for function_id in functions
        )

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

    def enumerate_safe_plans(
        self,
        decoder_input: DecoderInput,
        *,
        max_generated_patterns: int,
        retention_slots_by_pair: Mapping[tuple[int, int], int],
    ) -> PlanEnumerationResult:
        """确定性枚举并集界安全集合，绝不把搜索上限前的部分结果当作完整集合。"""

        started = perf_counter()
        if max_generated_patterns <= 0:
            raise ValueError("max_generated_patterns 必须为正整数。")
        self._expanded = self._pruned = 0
        self._limit_hit = False
        plans: list[DeploymentPlan] = []
        generated_patterns = 0

        def visit(index: int, assignments: dict[tuple[int, int], int]) -> None:
            nonlocal generated_patterns
            self._expanded += 1
            if self._expanded > self.max_expanded_states:
                self._limit_hit = True
                return
            if not self._partial_possible(assignments, index, decoder_input):
                self._pruned += 1
                return
            if index == len(self.action_spec.pairs):
                if generated_patterns >= max_generated_patterns:
                    self._limit_hit = True
                    return
                retention: dict[tuple[int, int], int] = {}
                for pair in self.action_spec.pairs:
                    if assignments[pair] == 0:
                        retention[pair] = 0
                        continue
                    value = retention_slots_by_pair.get(pair)
                    if value not in self.action_spec.retention_slot_options:
                        raise ValueError(f"已部署组合 {pair} 缺少合法保留时间档位。")
                    retention[pair] = int(value)
                plans.append(DeploymentPlan(dict(assignments), retention))
                generated_patterns += 1
                return
            pair = self.action_spec.pairs[index]
            function_id, node_id = pair
            candidates = self.config.allowed_instance_counts(function_id, node_id)
            if not decoder_input.effective_node_up.get(node_id, False):
                candidates = (0,)
            for candidate in candidates:
                assignments[pair] = candidate
                visit(index + 1, assignments)
                if self._limit_hit:
                    return
            assignments.pop(pair, None)

        visit(0, {})
        code = (
            "DECODER_SEARCH_LIMIT"
            if self._limit_hit
            else "OK" if plans else "NO_SAFE_FEASIBLE_DEPLOYMENT"
        )
        return PlanEnumerationResult(
            code=code,
            plans=tuple(plans) if code == "OK" else (),
            generated_patterns=generated_patterns,
            expanded_states=self._expanded,
            pruned_states=self._pruned,
            elapsed_seconds=perf_counter() - started,
        )

    def encode_plan(
        self,
        plan: DeploymentPlan | None,
        decoder_input: DecoderInput,
    ) -> PlanEncodingResult:
        """把教师计划编码为每个可行候选区间的中点，并用同一解码器完成复验。"""

        empty = np.empty(0, dtype=np.float64)
        empty.setflags(write=False)
        if plan is None or set(plan.instance_counts) != set(self.action_spec.pairs):
            return PlanEncodingResult("INVALID_DEPLOYMENT_PLAN", empty)
        self._expanded = self._pruned = 0
        self._limit_hit = False
        scores = np.full(self.action_spec.action_dim, 0.5, dtype=np.float64)
        assignments: dict[tuple[int, int], int] = {}
        for index, pair in enumerate(self.action_spec.pairs):
            function_id, node_id = pair
            candidates = self.config.allowed_instance_counts(function_id, node_id)
            if not decoder_input.effective_node_up.get(node_id, False):
                candidates = (0,)
            feasible: list[int] = []
            for candidate in candidates:
                if self._complete(
                    index + 1, {**assignments, pair: candidate}, decoder_input
                ) is not None:
                    feasible.append(candidate)
                if self._limit_hit:
                    return PlanEncodingResult("DECODER_SEARCH_LIMIT", empty)
            target = plan.instance_counts[pair]
            if target not in feasible:
                return PlanEncodingResult("INVALID_DEPLOYMENT_PLAN", empty)
            target_index = feasible.index(target)
            scores[2 * index] = (target_index + 0.5) / len(feasible)
            assignments[pair] = target

            retention = plan.retention_slots.get(pair, 0)
            if target == 0:
                if retention != 0:
                    return PlanEncodingResult("INVALID_DEPLOYMENT_PLAN", empty)
                continue
            options = self.action_spec.retention_slot_options
            if retention not in options:
                return PlanEncodingResult("INVALID_DEPLOYMENT_PLAN", empty)
            retention_index = options.index(retention)
            scores[2 * index + 1] = (retention_index + 0.5) / len(options)

        # 教师样本入库前必须通过同一状态、同一规格和同一解码器的往返校验。
        verification = self.decode(scores, decoder_input)
        if verification.code != "OK" or verification.plan != plan:
            return PlanEncodingResult("PLAN_ROUNDTRIP_MISMATCH", empty)
        scores.setflags(write=False)
        return PlanEncodingResult("OK", scores)

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
