# DPPO Main Algorithm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the active branch's retired value-based controller with a configuration-driven, constraint-projected DPPO pipeline covering simulated expert data, diffusion pre-training, online PPO fine-tuning, and independent evaluation.

**Architecture:** A configuration-derived `ScenarioDimensions` object fixes all ordering and tensor shapes. A conditional diffusion actor generates a relaxed joint SFC action, a deterministic projector produces `SFCDeploymentIntent`, and the shared fast executor performs exact audit and repair. Expert teachers pre-train the diffusion policy before an on-policy DPPO rollout buffer performs clipped PPO updates over the final denoising steps.

**Tech Stack:** Python 3.12, NumPy, PyTorch, PyYAML, pytest, existing rail mobility/failure/network/reliability/fast-execution modules.

**Design reference:** `docs/superpowers/specs/2026-08-10-dppo-main-algorithm-design.md`

---

## File structure locked by this plan

New focused modules:

- `src/scenario_dimensions.py`: configuration-derived node/function ordering and dimensions.
- `src/dppo_action_space.py`: relaxed action layout, clipping, decoding, and teacher encoding.
- `src/sfc_deployment_intent.py`: algorithm-independent per-function deployment intent.
- `src/dppo_projection.py`: deterministic necessary-condition projection and diagnostics.
- `src/dppo_intent_adapter.py`: attach window lifetime metadata to projected function intents.
- `src/continuous_retention.py`: stateful primary/backup keep-alive expiry tracking.
- `src/dppo_state_encoder.py`: configuration-driven DPPO state and history encoding.
- `src/slow_timescale_execution_core.py`: algorithm-independent causal trace and fast-slot execution.
- `src/dppo_slow_timescale_env.py`: causal slow-window DPPO environment.
- `src/dppo_scenario.py`: configuration-only construction of DPPO scenarios.
- `src/dppo_teacher.py`: cost, reliability, and balanced simulation teachers.
- `src/dppo_dataset.py`: expert-record schema, grouped splits, persistence, and loading.
- `src/dppo_diffusion.py`: cosine schedule and conditional MLP diffusion actor.
- `src/dppo_checkpoint.py`: versioned pre-training and online checkpoint metadata.
- `src/dppo.py`: value network, rollout buffer, GAE, and clipped DPPO update.
- `src/dppo_evaluation.py`: episode, feasibility, reliability, retention, and cost reports.
- `run_dppo_dataset_generation.py`, `run_dppo_pretraining.py`, `run_dppo_training.py`, `run_dppo_evaluation.py`: isolated CLI entry points.

Retained and modified shared modules:

- `src/fast_slot_executor.py`: accept explicit per-function deployment intent.
- `src/rl_reward.py`: retain only cost reward and shared window metrics.
- `src/config.py`: validate new scenario and DPPO configuration.
- `configs/debug.yaml`: declare functions, SFC, scale, action, diffusion, data, and training parameters.

Retired modules are deleted only after their replacements or their consumers are removed. The archived remote branch `codex/ddqn-state-action-redesign` at `9cc5a6e` remains the recovery source.

---

### Task 1: Remove directly named retired-algorithm assets

**Files:**
- Create: `tests/test_active_algorithm_scope.py`
- Modify: `configs/debug.yaml`
- Modify: `src/paired_statistics.py`
- Modify: `tests/test_paired_statistics.py`
- Modify: `src/ttl_retention.py`
- Delete: `src/ddqn.py`
- Delete: `src/ddqn_evaluation.py`
- Delete: `tests/test_ddqn.py`
- Delete: `tests/test_ddqn_evaluation.py`
- Delete: `run_ddqn_analysis.py`
- Delete: `run_ddqn_evaluation.py`
- Delete: `run_ddqn_statistical_analysis.py`
- Delete: `run_ddqn_training.py`
- Delete: `docs/superpowers/specs/2026-08-07-ddqn-state-action-redesign-design.md`
- Delete: `docs/superpowers/plans/2026-08-07-ddqn-state-action-redesign.md`
- Delete: all 16 tracked `results/figures/ddqn_*` and `results/tables/ddqn_*` files listed by `git ls-files`

- [x] **Step 1: Write the failing active-scope test**

```python
from pathlib import Path

from src.config import load_config


RETIRED_TOKEN = "d" + "dqn"


def test_active_tree_has_no_retired_named_files() -> None:
    root = Path(__file__).resolve().parents[1]
    checked_roots = (root / "src", root / "tests", root / "results")
    offenders = [
        path.relative_to(root).as_posix()
        for checked_root in checked_roots
        if checked_root.exists()
        for path in checked_root.rglob("*")
        if path.is_file() and RETIRED_TOKEN in path.name.lower()
    ]
    assert offenders == []


def test_config_has_no_retired_sections() -> None:
    config = load_config("configs/debug.yaml")
    assert all(RETIRED_TOKEN not in key.lower() for key in config)
```

- [x] **Step 2: Run the scope test and verify RED**

Run:

```powershell
$env:PYTHONUTF8='1'; $env:PYTHONPATH='.'; D:\Anaconda3\python.exe -X utf8 -m pytest -q -p no:cacheprovider tests/test_active_algorithm_scope.py
```

Expected: FAIL listing the tracked retired algorithm files and configuration keys.

- [x] **Step 3: Delete only the verified targets and remove configuration sections**

Use `git ls-files` to reprint the exact target list immediately before deletion. Delete the text files with `apply_patch`; delete each tracked binary PNG with `Remove-Item -LiteralPath` using its full verified path. Remove the complete retired-algorithm training, evaluation, and statistics YAML mappings without touching shared topology, failure, reliability, or cost mappings. Make `paired_statistics.py` algorithm-neutral by requiring an explicit `reference_policy` and building conclusion text from that value; update its tests to use neutral policy names. Replace the old algorithm name in the shared TTL module docstring with “legacy discrete controller” without changing TTL behavior.

- [x] **Step 4: Verify GREEN and run remaining public tests**

```powershell
$env:PYTHONUTF8='1'; $env:PYTHONPATH='.'; D:\Anaconda3\python.exe -X utf8 -m pytest -q -p no:cacheprovider tests/test_active_algorithm_scope.py tests/test_network.py tests/test_fast_slot_executor.py tests/test_reliability.py
```

Expected: PASS. Record removed paths and note that they remain recoverable from commit `9cc5a6e`.

- [x] **Step 5: Commit cleanup**

```powershell
git add -A
git commit -m "refactor: remove retired learning algorithm assets"
```

---

### Task 2: Make scenario size and VNF definitions configuration-driven

**Files:**
- Create: `src/scenario_dimensions.py`
- Create: `tests/test_scenario_dimensions.py`
- Modify: `configs/debug.yaml`
- Modify: `src/config.py`
- Modify: `src/rl_scenario.py`
- Modify: `src/topology.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_topology.py`

- [x] **Step 1: Write failing dimension and config tests**

```python
import pytest

from src.scenario_dimensions import ScenarioDimensions


@pytest.mark.parametrize(
    ("mec_ids", "function_ids", "state_dim", "action_dim"),
    [
        ((0, 1, 2), (0, 1), 58, 14),
        ((0, 1, 2, 3, 4), (0, 1, 2), 78, 27),
        (tuple(range(8)), (0, 1, 2), 102, 36),
    ],
)
def test_dimensions_follow_configured_scale(
    mec_ids: tuple[int, ...],
    function_ids: tuple[int, ...],
    state_dim: int,
    action_dim: int,
) -> None:
    dimensions = ScenarioDimensions(
        mec_node_ids=mec_ids,
        cloud_node_id=len(mec_ids),
        function_ids=function_ids,
    )
    assert dimensions.state_dim == state_dim
    assert dimensions.action_dim == action_dim
```

The first expected state is `8*3 + 4*2 + 26 = 58`; the action is `2*(3+4) = 14`.

- [x] **Step 2: Run and verify RED**

```powershell
$env:PYTHONUTF8='1'; $env:PYTHONPATH='.'; D:\Anaconda3\python.exe -X utf8 -m pytest -q -p no:cacheprovider tests/test_scenario_dimensions.py tests/test_config.py
```

Expected: import failure for `src.scenario_dimensions` and missing `rl_scenario.functions`, `rl_scenario.sfc`, and `dppo` configuration.

- [x] **Step 3: Implement the immutable dimension object and config builders**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioDimensions:
    mec_node_ids: tuple[int, ...]
    cloud_node_id: int
    function_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        compute_ids = self.mec_node_ids + (self.cloud_node_id,)
        if len(set(compute_ids)) != len(compute_ids):
            raise ValueError("计算节点 ID 必须唯一。")
        if len(compute_ids) < 3:
            raise ValueError("支持三副本时至少需要三个计算节点。")
        if not self.function_ids or len(set(self.function_ids)) != len(self.function_ids):
            raise ValueError("VNF ID 必须非空且唯一。")

    @property
    def compute_node_ids(self) -> tuple[int, ...]:
        return self.mec_node_ids + (self.cloud_node_id,)

    @property
    def state_dim(self) -> int:
        return 8 * len(self.mec_node_ids) + 4 * len(self.function_ids) + 26

    @property
    def action_dim(self) -> int:
        return len(self.function_ids) * (len(self.compute_node_ids) + 3)
```

Move every function and SFC field currently constructed in Python into `rl_scenario.functions` and `rl_scenario.sfc`. `build_rl_functions(config)` and `build_rl_sfc(config)` must preserve configured list order and reject duplicate IDs or a mismatched SFC chain.

- [x] **Step 4: Run dimensions, config, entity, and topology tests**

```powershell
$env:PYTHONUTF8='1'; $env:PYTHONPATH='.'; D:\Anaconda3\python.exe -X utf8 -m pytest -q -p no:cacheprovider tests/test_scenario_dimensions.py tests/test_config.py tests/test_entities.py tests/test_topology.py
```

Expected: PASS for 3, 5, and 8 MEC dimension cases.

- [x] **Step 5: Commit configuration-driven scale**

```powershell
git add src/scenario_dimensions.py src/config.py src/rl_scenario.py src/topology.py configs/debug.yaml tests/test_scenario_dimensions.py tests/test_config.py tests/test_topology.py
git commit -m "feat: derive DPPO dimensions from scenario config"
```

---

### Task 3: Implement the continuous joint SFC action layout

**Files:**
- Create: `src/dppo_action_space.py`
- Create: `tests/test_dppo_action_space.py`
- Modify: `src/config.py`
- Modify: `src/rl_scenario.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_scenario_dimensions.py`
- Modify: `docs/superpowers/specs/2026-08-10-dppo-main-algorithm-design.md`

- [x] **Step 1: Write failing shape, threshold, ranking, and retention tests**

```python
import numpy as np

from src.dppo_action_space import DPPOActionSpace
from src.scenario_dimensions import ScenarioDimensions


def test_decode_default_joint_action() -> None:
    dimensions = ScenarioDimensions(tuple(range(5)), 5, (0, 1, 2))
    space = DPPOActionSpace(dimensions, maximum_retention_seconds=20.0)
    action = np.zeros(dimensions.action_dim, dtype=np.float32)
    action[:6] = np.asarray([0.2, 0.9, -0.1, 0.7, 0.1, 0.3])
    decoded = space.decode(action)
    assert decoded.function_actions[0].replica_count == 3
    assert decoded.function_actions[0].ranked_node_ids[:3] == (1, 3, 5)
    assert decoded.function_actions[0].primary_retention_seconds == 10.0
    assert decoded.function_actions[0].backup_retention_seconds == 10.0
```

- [x] **Step 2: Run and verify RED**

```powershell
$env:PYTHONUTF8='1'; $env:PYTHONPATH='.'; D:\Anaconda3\python.exe -X utf8 -m pytest -q -p no:cacheprovider tests/test_dppo_action_space.py
```

Expected: missing module failure.

- [x] **Step 3: Implement dynamic slices and deterministic decoding**

Define frozen `DecodedFunctionAction` and `DecodedDPPOAction`. In `DPPOActionSpace.decode`, reshape the first `F*N` values to `(F,N)`, read the next `F` replica scores and final `2F` retention values. Use stable `sorted(node_ids, key=lambda node_id: (-score, node_id))`, threshold `>= replica_threshold` for three replicas, and `(x+1)/2*maximum_retention_seconds` for retention. Reject non-finite values and wrong shapes; clip only finite values to `[-1,1]`.

Lock the public interface to:

```python
@dataclass(frozen=True)
class DecodedFunctionAction:
    function_id: int
    ranked_node_ids: tuple[int, ...]
    replica_count: int
    primary_retention_seconds: float
    backup_retention_seconds: float


@dataclass(frozen=True)
class DecodedDPPOAction:
    function_actions: tuple[DecodedFunctionAction, ...]
```

- [x] **Step 4: Run action tests at multiple scales**

```powershell
$env:PYTHONUTF8='1'; $env:PYTHONPATH='.'; D:\Anaconda3\python.exe -X utf8 -m pytest -q -p no:cacheprovider tests/test_dppo_action_space.py tests/test_scenario_dimensions.py
```

Expected: PASS and no fixed `18`, `21`, `27`, or `36` action slices in source.

- [x] **Step 5: Commit action space**

```powershell
git add src/dppo_action_space.py src/config.py src/rl_scenario.py tests/test_dppo_action_space.py tests/test_config.py tests/test_scenario_dimensions.py docs/superpowers/specs/2026-08-10-dppo-main-algorithm-design.md docs/superpowers/plans/2026-08-10-dppo-main-algorithm.md
git commit -m "feat: add configuration-driven DPPO action space"
```

---

### Task 4: Add deployment intent and deterministic feasibility projection

**Files:**
- Create: `src/sfc_deployment_intent.py`
- Create: `src/dppo_projection.py`
- Create: `src/dppo_intent_adapter.py`
- Create: `tests/test_dppo_projection.py`

- [x] **Step 1: Write failing projection tests**

```python
import numpy as np

from src.dppo_action_space import DPPOActionSpace
from src.dppo_projection import DPPOProjector, ProjectionResourceDemand
from src.scenario_dimensions import ScenarioDimensions


def test_projection_skips_failed_nodes_and_spreads_fault_domains() -> None:
    dimensions = ScenarioDimensions(tuple(range(5)), 5, (0,))
    action_space = DPPOActionSpace(dimensions, maximum_retention_seconds=20.0)
    raw_action = np.zeros(dimensions.action_dim, dtype=np.float32)
    raw_action[:6] = np.asarray([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
    projector = DPPOProjector(
        dimensions=dimensions,
        resource_demands={
            0: ProjectionResourceDemand(cpu=1.0, memory_mb=64.0),
        },
        minimum_distinct_fault_domains=2,
    )
    result = projector.project(
        decoded_action=action_space.decode(raw_action),
        operational_node_ids=frozenset({1, 2, 3, 4, 5}),
        free_cpu={node_id: 100.0 for node_id in range(6)},
        free_memory_mb={node_id: 4096.0 for node_id in range(6)},
        fault_domains={0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 3},
    )
    assert result.function_intents is not None
    intent = result.function_intents[0]
    assert 0 not in intent.preferred_node_ids
    assert len(intent.preferred_node_ids) == intent.replica_count
    assert len({result.fault_domains[node_id] for node_id in intent.preferred_node_ids}) >= 2
```

- [x] **Step 2: Run and verify RED**

Run `pytest -q -p no:cacheprovider tests/test_dppo_projection.py`; expect missing projector failure.

- [x] **Step 3: Implement intent and projection result types**

Create `FunctionDeploymentIntent`, `SFCDeploymentIntent`, `ProjectionResourceDemand`, `ProjectionResult`, and `DPPOIntentAdapter`. The projector must traverse the actor's complete node ranking, select unique operational nodes, prefer unused fault domains until `minimum_distinct_fault_domains` is met, then fill remaining replicas. It must return `function_intents=None` with machine-readable reasons when CPU, memory, node count, or fault-domain necessary conditions cannot be met. It must report changed assignments divided by requested assignment count as `change_ratio`. The adapter is the only code that attaches `decision_slot`, `valid_until_slot`, and `source_algorithm="dppo"` to a successful projection.

Use these exact result contracts so the environment and evaluator consume one representation:

```python
@dataclass(frozen=True)
class FunctionDeploymentIntent:
    function_id: int
    preferred_node_ids: tuple[int, ...]
    replica_count: int
    primary_retention_seconds: float
    backup_retention_seconds: float


@dataclass(frozen=True)
class SFCDeploymentIntent:
    decision_slot: int
    valid_until_slot: int
    function_intents: tuple[FunctionDeploymentIntent, ...]
    source_algorithm: str


@dataclass(frozen=True)
class ProjectionResult:
    function_intents: tuple[FunctionDeploymentIntent, ...] | None
    raw_feasible: bool
    success: bool
    reasons: tuple[str, ...]
    changed_assignment_count: int
    requested_assignment_count: int
    change_ratio: float
    fault_domains: Mapping[int, int]


class DPPOIntentAdapter:
    def to_intent(
        self,
        projection: ProjectionResult,
        *,
        decision_slot: int,
        valid_until_slot: int,
    ) -> SFCDeploymentIntent:
        if not projection.success or projection.function_intents is None:
            raise ValueError("A failed projection cannot become an execution intent.")
        return SFCDeploymentIntent(
            decision_slot=decision_slot,
            valid_until_slot=valid_until_slot,
            function_intents=projection.function_intents,
            source_algorithm="dppo",
        )
```

- [x] **Step 4: Run projection and constraint tests**

```powershell
$env:PYTHONUTF8='1'; $env:PYTHONPATH='.'; D:\Anaconda3\python.exe -X utf8 -m pytest -q -p no:cacheprovider tests/test_dppo_projection.py tests/test_constraint_audit.py
```

- [x] **Step 5: Commit projector**

```powershell
git add src/sfc_deployment_intent.py src/dppo_projection.py src/dppo_intent_adapter.py tests/test_dppo_projection.py
git commit -m "feat: project relaxed DPPO actions to SFC intents"
```

---

### Task 5: Implement continuous primary/backup retention

**Files:**
- Create: `src/continuous_retention.py`
- Create: `tests/test_continuous_retention.py`

- [x] **Step 1: Write failing expiry and refresh tests**

```python
from src.continuous_retention import ContinuousRetentionTracker


def test_primary_and_backup_expire_at_independent_slots() -> None:
    tracker = ContinuousRetentionTracker(slot_seconds=1.0)
    tracker.apply_intent(slot=2, function_id=0, primary_node_id=1,
                         backup_node_ids=(2,), primary_seconds=4.0,
                         backup_seconds=1.0)
    assert tracker.hot_node_ids(0, slot=3) == frozenset({1, 2})
    assert tracker.hot_node_ids(0, slot=4) == frozenset({1})
    assert tracker.hot_node_ids(0, slot=7) == frozenset()
```

- [x] **Step 2: Run and verify RED**

Run `pytest -q -p no:cacheprovider tests/test_continuous_retention.py`; expect missing class failure.

- [x] **Step 3: Implement expiry tracker**

Store expiry as `dict[(function_id,node_id), int]`. Convert positive seconds with `ceil(seconds/slot_seconds)`, expire when `slot >= expiry_slot`, refresh selected replicas with `max(old_expiry,new_expiry)`, remove failed nodes immediately, and clear all state on reset. Reject negative, non-finite retention and non-positive slot length.

```python
class ContinuousRetentionTracker:
    def apply_intent(
        self,
        *,
        slot: int,
        function_id: int,
        primary_node_id: int,
        backup_node_ids: tuple[int, ...],
        primary_seconds: float,
        backup_seconds: float,
    ) -> None:
        primary_expiry = slot + math.ceil(primary_seconds / self.slot_seconds)
        self._refresh(function_id, primary_node_id, primary_expiry)
        for node_id in backup_node_ids:
            backup_expiry = slot + math.ceil(backup_seconds / self.slot_seconds)
            self._refresh(function_id, node_id, backup_expiry)

    def hot_node_ids(self, function_id: int, *, slot: int) -> frozenset[int]:
        return frozenset(
            node_id
            for (stored_function_id, node_id), expiry in self._expiry_slots.items()
            if stored_function_id == function_id and slot < expiry
        )

    def remove_failed_nodes(self, failed_node_ids: Collection[int]) -> None:
        self._expiry_slots = {
            key: expiry
            for key, expiry in self._expiry_slots.items()
            if key[1] not in failed_node_ids
        }

    def reset(self) -> None:
        self._expiry_slots.clear()
```

- [x] **Step 4: Run retention and cold-start tests**

```powershell
$env:PYTHONUTF8='1'; $env:PYTHONPATH='.'; D:\Anaconda3\python.exe -X utf8 -m pytest -q -p no:cacheprovider tests/test_continuous_retention.py tests/test_cold_start.py tests/test_ttl_retention.py
```

- [x] **Step 5: Commit retention**

```powershell
git add src/continuous_retention.py tests/test_continuous_retention.py
git commit -m "feat: track continuous DPPO replica retention"
```

---

### Task 6: Add the DPPO state encoder before retiring old consumers

**Files:**
- Create: `src/dppo_state_encoder.py`
- Create: `tests/test_dppo_state_encoder.py`
- Retain temporarily: `src/rl_agent_action_space.py`, `src/rl_state_encoder.py`, and
  their tests until Tasks 7-8 remove their remaining consumers. Deleting them here
  would make the intermediate branch unimportable.

- [x] **Step 1: Write failing dynamic-state and history tests**

Construct snapshots for `(M=3,F=2)`, `(M=5,F=3)`, and `(M=8,F=3)`. Assert `encoded.shape == (dimensions.state_dim,)`, all values are finite in `[0,1]`, and the last 11 values encode mean replica count, three-replica ratio, primary/backup retention, projection-change ratio, hot/cloud ratios, success, SLA violation, cost, and repair-failure rate in that exact order.

```python
@pytest.mark.parametrize("mec_count,function_count", [(3, 2), (5, 3), (8, 3)])
def test_state_shape_is_derived_from_configuration(
    mec_count: int,
    function_count: int,
) -> None:
    dimensions = ScenarioDimensions(
        tuple(range(mec_count)), mec_count, tuple(range(function_count))
    )
    encoder = DPPOStateEncoder(dimensions)
    encoded = encoder.encode(
        DPPOStateSnapshot.zeros(dimensions),
        DPPOHistoryObservation.zeros(),
    )
    assert encoded.shape == (dimensions.state_dim,)
    assert np.isfinite(encoded).all()
    assert ((0.0 <= encoded) & (encoded <= 1.0)).all()
    assert encoder.feature_names[-11:] == DPPO_HISTORY_FEATURE_NAMES
```

- [x] **Step 2: Run and verify RED**

Run `pytest -q -p no:cacheprovider tests/test_dppo_state_encoder.py`; expect missing encoder failure.

- [x] **Step 3: Implement DPPO-only state types and record deferred retirement**

Copy only the validated mobility, node, VNF, and SFC feature formulas into `DPPOStateSnapshot`, `DPPOHistoryObservation`, `DPPOHistoryEncoder`, and `DPPOStateEncoder`. Build feature names from `ScenarioDimensions` order. Do not import any retired action type. The import audit found that the shared fast executor and retired environment still consume the old action/state modules; keep those files unchanged in this task and delete them atomically in Task 8 after Tasks 7-8 remove every consumer.

```python
DPPO_HISTORY_FEATURE_NAMES = (
    "previous_mean_replica_count",
    "previous_three_replica_ratio",
    "previous_primary_retention_ratio",
    "previous_backup_retention_ratio",
    "previous_projection_change_ratio",
    "current_hot_replica_ratio",
    "current_cloud_replica_ratio",
    "previous_success_rate",
    "previous_sla_violation_rate",
    "previous_normalized_cost",
    "previous_repair_failure_rate",
)


class DPPOHistoryEncoder:
    def encode(self, history: DPPOHistoryObservation) -> list[float]:
        return [float(getattr(history, name)) for name in DPPO_HISTORY_FEATURE_NAMES]


class DPPOStateEncoder:
    def encode(
        self,
        snapshot: DPPOStateSnapshot,
        history: DPPOHistoryObservation,
    ) -> np.ndarray:
        values = self._encode_current_snapshot(snapshot) + self.history_encoder.encode(history)
        encoded = np.asarray(values, dtype=np.float32)
        if encoded.shape != (self.dimensions.state_dim,):
            raise ValueError("Encoded state dimension does not match configuration.")
        return encoded
```

- [x] **Step 4: Run state, action, projection, and active-scope tests**

```powershell
$env:PYTHONUTF8='1'; $env:PYTHONPATH='.'; D:\Anaconda3\python.exe -X utf8 -m pytest -q -p no:cacheprovider tests/test_dppo_state_encoder.py tests/test_dppo_action_space.py tests/test_dppo_projection.py tests/test_active_algorithm_scope.py
```

- [x] **Step 5: Commit DPPO state encoder**

```powershell
git add src/dppo_state_encoder.py tests/test_dppo_state_encoder.py docs/superpowers/plans/2026-08-10-dppo-main-algorithm.md
git commit -m "feat: add configuration-driven DPPO state encoder"
```

---

### Task 7: Let the shared fast executor consume explicit per-function intents

**Files:**
- Modify: `src/constraint_audit.py`
- Modify: `src/continuous_retention.py`
- Modify: `src/fast_optimizer.py`
- Modify: `src/fast_slot_executor.py`
- Modify: `src/sfc_deployment_intent.py`
- Modify: `src/two_timescale_control.py`
- Modify: `src/two_timescale_simulator.py`
- Modify: `tests/test_fast_slot_executor.py`
- Modify: `tests/test_two_timescale_simulator.py`
- Create: `tests/test_fast_slot_executor_intent.py`

- [x] **Step 1: Write failing mixed-replica execution test**

Create an `SFCDeploymentIntent` whose three functions request `(2,3,2)` replicas on explicit node rankings and independent retention times. Execute one slot and assert the initial candidate map uses those per-function counts, the final audit is returned, and continuous retention determines hot nodes.

```python
from dataclasses import replace

from src.sfc_deployment_intent import FunctionDeploymentIntent, SFCDeploymentIntent
from tests.test_fast_slot_executor import build_slot_input


def test_executor_uses_per_function_replica_counts() -> None:
    function_intents = tuple(
        FunctionDeploymentIntent(
            function_id=function_id,
            preferred_node_ids=ranking,
            replica_count=replica_count,
            primary_retention_seconds=primary_seconds,
            backup_retention_seconds=backup_seconds,
        )
        for function_id, ranking, replica_count, primary_seconds, backup_seconds in zip(
            (0, 1, 2),
            ((0, 1, 2), (1, 2, 3), (2, 3, 4)),
            (2, 3, 2),
            (4.0, 6.0, 8.0),
            (1.0, 2.0, 3.0),
            strict=True,
        )
    )
    intent = SFCDeploymentIntent(0, 4, function_intents, "test")
    result = FastSlotExecutor().execute(
        replace(build_slot_input(), deployment_intent=intent)
    )
    assert tuple(len(result.initial_candidate_map[f]) for f in (0, 1, 2)) == (2, 3, 2)
    assert result.final_audit is not None
    assert result.retained_hot_node_ids_by_function[0] == frozenset({0, 1})
```

- [x] **Step 2: Run and verify RED**

Run `pytest -q -p no:cacheprovider tests/test_fast_slot_executor_intent.py`; expect `FastSlotInput` not accepting `deployment_intent`.

- [x] **Step 3: Add an explicit intent path without algorithm branching**

`FastSlotInput` receives a required `deployment_intent`. Remove planner selection from the learning path. The executor constructs the initial candidate map directly from each `FunctionDeploymentIntent`, applies the retention tracker, audits, invokes the existing optimizer only when needed, and returns raw/projected/final maps. Rule-based simulators must construct the same intent before calling the executor; the executor must not inspect a controller name.

```python
@dataclass(frozen=True)
class FastSlotInput:
    # Keep every existing exogenous input field unchanged.
    deployment_intent: SFCDeploymentIntent


def _candidate_map_from_intent(
    intent: SFCDeploymentIntent,
) -> dict[int, tuple[int, ...]]:
    return {
        function_intent.function_id: function_intent.preferred_node_ids[
            : function_intent.replica_count
        ]
        for function_intent in intent.function_intents
    }
```

- [x] **Step 4: Run executor, optimizer, simulator, and reliability tests**

```powershell
$env:PYTHONUTF8='1'; $env:PYTHONPATH='.'; D:\Anaconda3\python.exe -X utf8 -m pytest -q -p no:cacheprovider tests/test_fast_slot_executor.py tests/test_fast_slot_executor_intent.py tests/test_fast_optimizer.py tests/test_two_timescale_simulator.py
```

- [x] **Step 5: Commit shared intent execution**

```powershell
git add src/fast_slot_executor.py src/two_timescale_control.py src/two_timescale_simulator.py tests/test_fast_slot_executor.py tests/test_fast_slot_executor_intent.py tests/test_two_timescale_simulator.py
git commit -m "refactor: execute explicit per-function SFC intents"
```

---

### Task 8: Build the DPPO slow environment and remove the retired environment

**Files:**
- Create: `src/deployment_policies.py`
- Create: `src/slow_timescale_execution_core.py`
- Create: `src/dppo_slow_timescale_env.py`
- Create: `src/dppo_scenario.py`
- Create: `tests/test_slow_timescale_execution_core.py`
- Create: `tests/test_dppo_slow_timescale_env.py`
- Modify: `run_rl_environment_demo.py`
- Modify: `configs/debug.yaml`
- Modify: `src/fast_optimizer.py`
- Modify: `src/fast_slot_executor.py`
- Modify: `src/rl_reward.py`
- Modify: `src/two_timescale_control.py`
- Modify: `tests/test_rl_reward.py`
- Modify: `tests/test_active_algorithm_scope.py`
- Modify: `tests/test_fast_optimizer.py`
- Modify: `tests/test_fast_slot_executor.py`
- Modify: `tests/test_scenario_dimensions.py`
- Modify: `tests/test_two_timescale_control.py`
- Delete: `src/slow_timescale_rl_env.py`
- Delete: `src/rl_scenario.py`
- Delete: `tests/test_slow_timescale_rl_env.py`
- Delete: `src/rl_agent_action_space.py`
- Delete: `tests/test_rl_agent_action_space.py`
- Delete: `src/rl_state_encoder.py`
- Delete: `tests/test_rl_state_encoder.py`

- [x] **Step 1: Write failing causal DPPO environment tests**

Tests must assert the shared core owns the pre-generated exogenous trace and exposes only the observed prefix, reset returns `dimensions.state_dim`, step accepts exactly `dimensions.action_dim`, future request changes do not alter current state, one slow step calls the fast executor once per fast slot, projection failure advances with a violation, and `info` contains raw action, clipped action, decoded action, projection result, final metrics, and next observation.

```python
def test_step_exposes_raw_projected_and_final_results(dppo_environment) -> None:
    state = dppo_environment.reset(seed=123)
    action = np.zeros(dppo_environment.dimensions.action_dim, dtype=np.float32)
    next_state, reward, terminated, truncated, info = dppo_environment.step(action)
    assert state.shape == next_state.shape == (dppo_environment.dimensions.state_dim,)
    assert np.isfinite(reward)
    assert not (terminated and truncated)
    assert {
        "raw_action", "clipped_action", "decoded_action", "projection_result",
        "final_metrics", "next_observation",
    } <= info.keys()
```

- [x] **Step 2: Run and verify RED**

Run `pytest -q -p no:cacheprovider tests/test_dppo_slow_timescale_env.py`; expect missing environment failure.

- [x] **Step 3: Implement the DPPO-only environment atomically**

Move the approved pre-generated exogenous trace, observed-request-prefix boundary, time advancement, fast-slot loop, and window metric aggregation into `SlowTimescaleExecutionCore`. The DPPO environment owns action clipping, decoding, projection, intent adaptation, reward, and learner-facing `info`, while the core remains algorithm-independent. On projection success convert with `DPPOIntentAdapter` and execute every slot; on failure do not call the executor and ask the core to advance the rejected window. Retain only `RLWindowCostMetrics`, `RLCostRewardBreakdown`, `RLWindowMetrics`, and `calculate_cost_reward`; remove legacy weighted reward classes and tests. Replace the demo with a deterministic zero-action DPPO smoke step. Delete retired environment/scenario/action/state files only after all imports use the DPPO replacements. Task 6 deliberately defers the old action/state deletion to this atomic migration because the intermediate shared fast layer and retired environment still import those modules.

Extend `test_active_algorithm_scope.py` with a content scan over `src/**/*.py`, `tests/**/*.py`, `configs/*.yaml`, and root `run_*.py`. Build both the retired short token and the retired full algorithm name from string fragments inside the test so the test does not flag itself. Assert neither value occurs in an active file; the two approved design/plan archive-recovery notes are outside this scan.

```python
class DPPOSlowTimescaleEnvironment:
    def step(
        self,
        raw_action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        clipped_action = self.action_space.clip(raw_action)
        decoded_action = self.action_space.decode(clipped_action)
        projection_inputs = self._current_projection_inputs()
        projection = self.projector.project(
            decoded_action=decoded_action,
            **projection_inputs,
        )
        if not projection.success:
            execution = self.execution_core.advance_rejected_window(projection.reasons)
        else:
            intent = self.intent_adapter.to_intent(
                projection,
                decision_slot=self.execution_core.current_slot,
                valid_until_slot=self.execution_core.next_window_end_slot,
            )
            execution = self.execution_core.execute_window(intent)
        return self._build_step_result(raw_action, clipped_action, projection, execution)
```

- [x] **Step 4: Run environment and shared integration tests**

```powershell
$env:PYTHONUTF8='1'; $env:PYTHONPATH='.'; D:\Anaconda3\python.exe -X utf8 -m pytest -q -p no:cacheprovider tests/test_slow_timescale_execution_core.py tests/test_dppo_slow_timescale_env.py tests/test_rl_reward.py tests/test_fast_slot_executor_intent.py tests/test_config.py tests/test_active_algorithm_scope.py
```

- [x] **Step 5: Commit DPPO environment**

```powershell
git add -A
git commit -m "feat: add causal DPPO slow-timescale environment"
```

---

### Task 9: Generate diverse simulation teacher actions

**Files:**
- Create: `src/dppo_teacher.py`
- Create: `tests/test_dppo_teacher.py`

- [ ] **Step 1: Write failing teacher tests**

For a fixed public observation, verify cost teacher prefers near operational nodes and two replicas, reliability teacher uses three replicas and distinct domains, balanced teacher satisfies reliability before cost ranking, repeated calls are identical, and changing hidden future requests does not change the teacher action.

```python
@pytest.mark.parametrize(
    ("teacher_name", "expected_replicas"),
    [("cost", 2), ("reliability", 3), ("balanced", 3)],
)
def test_teacher_is_deterministic_and_uses_declared_replica_policy(
    teacher_name: str,
    expected_replicas: int,
) -> None:
    scenario = build_dppo_scenario(load_config("configs/debug.yaml"))
    public_snapshot = scenario.execution_core.current_public_snapshot()
    teacher = build_simulation_teacher(teacher_name, scenario)
    first = teacher.propose(public_snapshot)
    second = teacher.propose(public_snapshot)
    assert np.array_equal(first.relaxed_action, second.relaxed_action)
    assert all(
        item.replica_count == expected_replicas
        for item in first.decoded_action.function_actions
    )
```

- [ ] **Step 2: Run and verify RED**

Run `pytest -q -p no:cacheprovider tests/test_dppo_teacher.py`; expect missing teacher module.

- [ ] **Step 3: Implement three deterministic teachers**

Teachers receive only `DPPOStateSnapshot`, topology, function definitions, and public configuration. Score each operational node from normalized delay, free resources, reliability, and cloud cost using fixed teacher-specific lexicographic ordering rather than tunable reward weights. Encode complete node ranking with `1 - 2*j/(N-1)`, replicas as `-1/+1`, and retention through the inverse action scaling. Never call environment step to select the best action after observing a future window.

```python
@dataclass(frozen=True)
class TeacherProposal:
    teacher_name: str
    relaxed_action: np.ndarray
    decoded_action: DecodedDPPOAction


def ranking_scores(node_ids: Sequence[int]) -> dict[int, float]:
    denominator = max(1, len(node_ids) - 1)
    return {
        node_id: 1.0 - 2.0 * rank / denominator
        for rank, node_id in enumerate(node_ids)
    }


def build_simulation_teacher(name: str, scenario: DPPOScenario) -> SimulationTeacher:
    teacher_types = {
        "cost": CostTeacher,
        "reliability": ReliabilityTeacher,
        "balanced": BalancedTeacher,
    }
    return teacher_types[name](scenario.teacher_context)
```

- [ ] **Step 4: Run teacher, action, and no-leakage tests**

Run `pytest -q -p no:cacheprovider tests/test_dppo_teacher.py tests/test_dppo_action_space.py tests/test_workload_prediction.py`; expect PASS.

- [ ] **Step 5: Commit teachers**

```powershell
git add src/dppo_teacher.py tests/test_dppo_teacher.py
git commit -m "feat: add causal simulation teachers for DPPO"
```

---

### Task 10: Persist and split expert simulation data by Episode

**Files:**
- Create: `src/dppo_dataset.py`
- Create: `tests/test_dppo_dataset.py`
- Create: `run_dppo_dataset_generation.py`
- Modify: `configs/debug.yaml`

- [ ] **Step 1: Write failing schema and split tests**

Create records for six Episode seeds, split them, and assert no seed appears in two partitions, state/action shapes match metadata, rejected records remain in diagnostics but not behavior-cloning samples, and save/load preserves arrays and schema versions exactly.

```python
def test_episode_seeds_never_cross_dataset_partitions(tmp_path) -> None:
    records = [
        ExpertTransitionRecord(
            state=np.zeros(58, dtype=np.float32),
            expert_action=np.zeros(14, dtype=np.float32),
            teacher_name="cost",
            episode_seed=seed,
            slow_step=0,
            raw_feasible=True,
            projection_change_ratio=0.0,
            final_feasible=True,
            run_cost=1.0,
            route_cost=0.0,
            cold_start_cost=0.0,
        )
        for seed in range(6)
    ]
    partitions = split_records_by_episode(records, fractions=(0.5, 0.25, 0.25), seed=7)
    seed_sets = {
        name: {record.episode_seed for record in split}
        for name, split in partitions.items()
    }
    assert seed_sets["train"].isdisjoint(seed_sets["validation"])
    assert seed_sets["train"].isdisjoint(seed_sets["test"])
    assert seed_sets["validation"].isdisjoint(seed_sets["test"])
```

- [ ] **Step 2: Run and verify RED**

Run `pytest -q -p no:cacheprovider tests/test_dppo_dataset.py`; expect missing dataset module.

- [ ] **Step 3: Implement records, grouped split, NPZ persistence, and CLI**

Use frozen `ExpertTransitionRecord` and `ExpertDatasetMetadata`. For each teacher proposal, run the same projector, intent adapter, constraint audit, and fast executor used by online DPPO; only `final_feasible=true` records enter behavior cloning, while rejected proposals enter diagnostics with explicit reasons. Store numeric arrays in compressed NPZ and metadata/config hash in UTF-8 JSON beside it. Split unique Episode seeds deterministically using configured train/validation/test fractions that sum to one. The CLI accepts `--episodes`, `--output-root`, and `--seed-start`; it must never default to a formal result path during tests.

```python
@dataclass(frozen=True)
class ExpertTransitionRecord:
    state: np.ndarray
    expert_action: np.ndarray
    teacher_name: str
    episode_seed: int
    slow_step: int
    raw_feasible: bool
    projection_change_ratio: float
    final_feasible: bool
    run_cost: float
    route_cost: float
    cold_start_cost: float


@dataclass(frozen=True)
class ExpertDatasetMetadata:
    schema_version: str
    state_dim: int
    action_dim: int
    config_hash: str
```

- [ ] **Step 4: Run dataset generation smoke test in temporary storage**

```powershell
$env:PYTHONUTF8='1'; $env:PYTHONPATH='.'; D:\Anaconda3\python.exe -X utf8 -m pytest -q -p no:cacheprovider tests/test_dppo_dataset.py
$smokeRoot=Join-Path $env:TEMP 'dppo-dataset-smoke'; D:\Anaconda3\python.exe -X utf8 run_dppo_dataset_generation.py --episodes 3 --output-root $smokeRoot --seed-start 7000
```

Expected: train/validation/test or diagnostics artifacts exist under the temporary root with no repository changes.

- [ ] **Step 5: Commit dataset pipeline**

```powershell
git add src/dppo_dataset.py tests/test_dppo_dataset.py run_dppo_dataset_generation.py configs/debug.yaml
git commit -m "feat: generate versioned DPPO expert datasets"
```

---

### Task 11: Implement conditional diffusion and cosine scheduling

**Files:**
- Create: `src/dppo_diffusion.py`
- Create: `tests/test_dppo_diffusion.py`

- [ ] **Step 1: Write failing diffusion tests**

Assert cosine betas are finite in `(0,1)`, `q_sample` preserves `(batch,action_dim)`, conditional MLP predicts the same shape, noise loss is finite, seeded reverse sampling is reproducible, and every reverse Gaussian log probability is finite.

```python
def test_seeded_denoising_chain_is_reproducible() -> None:
    schedule = CosineNoiseSchedule(steps=20)
    model = ConditionalDiffusionMLP(state_dim=58, action_dim=14, hidden_dims=(64, 64))
    state = torch.zeros((2, 58))
    first = sample_denoising_chain(model, schedule, state, seed=31)
    second = sample_denoising_chain(model, schedule, state, seed=31)
    assert first.actions.shape == (21, 2, 14)
    assert torch.equal(first.actions, second.actions)
    assert torch.isfinite(first.log_probabilities).all()
```

- [ ] **Step 2: Run and verify RED**

Run `pytest -q -p no:cacheprovider tests/test_dppo_diffusion.py`; expect missing module.

- [ ] **Step 3: Implement schedule and conditional MLP**

Implement `CosineNoiseSchedule`, sinusoidal timestep embedding, `ConditionalDiffusionMLP`, `diffusion_noise_loss`, and `sample_denoising_chain`. Derive all linear sizes from `state_dim`, `action_dim`, and configured hidden dimensions. Return a frozen `DenoisingSample` containing all `a_k`, means, standard deviations, and log probabilities needed by DPPO.

```python
@dataclass(frozen=True)
class DenoisingSample:
    actions: torch.Tensor
    means: torch.Tensor
    standard_deviations: torch.Tensor
    log_probabilities: torch.Tensor


def diffusion_noise_loss(
    model: ConditionalDiffusionMLP,
    schedule: CosineNoiseSchedule,
    states: torch.Tensor,
    clean_actions: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    timesteps = torch.randint(
        0, schedule.steps, (states.shape[0],), generator=generator, device=states.device
    )
    noise = torch.randn(clean_actions.shape, generator=generator, device=states.device)
    noisy_actions = schedule.q_sample(clean_actions, timesteps, noise)
    predicted_noise = model(noisy_actions, timesteps, states)
    return torch.nn.functional.mse_loss(predicted_noise, noise)
```

- [ ] **Step 4: Run diffusion tests on CPU**

```powershell
$env:PYTHONUTF8='1'; $env:PYTHONPATH='.'; D:\Anaconda3\python.exe -X utf8 -m pytest -q -p no:cacheprovider tests/test_dppo_diffusion.py
```

- [ ] **Step 5: Commit diffusion actor**

```powershell
git add src/dppo_diffusion.py tests/test_dppo_diffusion.py
git commit -m "feat: add conditional diffusion policy network"
```

---

### Task 12: Add versioned diffusion pre-training and checkpoints

**Files:**
- Create: `src/dppo_checkpoint.py`
- Create: `tests/test_dppo_checkpoint.py`
- Create: `run_dppo_pretraining.py`
- Modify: `configs/debug.yaml`

- [ ] **Step 1: Write failing pre-training checkpoint tests**

Train two mini-batches from a tiny synthetic expert dataset, save, reload, and assert identical seeded samples. Verify mismatched state dimension, action dimension, topology count, function count, diffusion steps, or configuration hash raises a specific `ValueError`.

```python
def test_checkpoint_rejects_incompatible_dimensions(tmp_path) -> None:
    metadata = DPPOCheckpointMetadata(
        state_schema_version="dppo-v1-flat",
        action_schema_version="joint-sfc-continuous-v1",
        state_dim=58,
        action_dim=14,
        mec_count=3,
        compute_node_count=4,
        function_count=2,
        diffusion_steps=20,
        fine_tuned_steps=5,
        maximum_retention_seconds=20.0,
        replica_threshold=0.0,
        config_hash="test-hash",
    )
    model = ConditionalDiffusionMLP(58, 14, (32, 32))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "tiny.pt"
    save_dppo_checkpoint(checkpoint_path, model, optimizer, metadata, epoch=0)
    with pytest.raises(ValueError, match="state_dim"):
        load_dppo_checkpoint(
            checkpoint_path,
            expected=replace(metadata, state_dim=78),
            device="cpu",
        )
```

- [ ] **Step 2: Run and verify RED**

Run `pytest -q -p no:cacheprovider tests/test_dppo_checkpoint.py`; expect missing checkpoint module.

- [ ] **Step 3: Implement metadata, trainer, and CLI**

`DPPOCheckpointMetadata` stores `dppo-v1-flat`, `joint-sfc-continuous-v1`, dimensions/counts, 20 diffusion steps, 5 fine-tuned steps, retention maximum, replica threshold, and config hash. Save model, optimizer, metadata, epoch, and RNG state. CLI accepts `--dataset-root`, `--output-root`, `--epochs`, and `--device {cpu,cuda}`; reject unavailable CUDA explicitly.

```python
@dataclass(frozen=True)
class DPPOCheckpointMetadata:
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


def validate_checkpoint_metadata(
    actual: DPPOCheckpointMetadata,
    expected: DPPOCheckpointMetadata,
) -> None:
    for field in fields(DPPOCheckpointMetadata):
        if getattr(actual, field.name) != getattr(expected, field.name):
            raise ValueError(f"Checkpoint {field.name} does not match configuration.")
```

- [ ] **Step 4: Run checkpoint test and one-epoch temporary pre-training**

Run tests, then use the Task 10 smoke dataset with `--epochs 1 --device cpu`. Expected: finite train/validation loss and reloadable checkpoint outside the repository.

- [ ] **Step 5: Commit pre-training**

```powershell
git add src/dppo_checkpoint.py tests/test_dppo_checkpoint.py run_dppo_pretraining.py configs/debug.yaml
git commit -m "feat: pretrain and checkpoint DPPO diffusion policy"
```

---

### Task 13: Implement on-policy rollouts, GAE, and clipped DPPO updates

**Files:**
- Create: `src/dppo.py`
- Create: `tests/test_dppo.py`

- [ ] **Step 1: Write failing rollout and optimization tests**

Test grouped environment transitions store full denoising chains, GAE matches a hand-calculated three-step example, terminal bootstrap is zero, probability ratios use old log probabilities, clipping bounds the surrogate, early 15 denoising steps stay frozen, final 5 steps change, and `clear()` empties the rollout buffer after update.

```python
def test_gae_matches_three_step_manual_example() -> None:
    advantages, returns = compute_gae(
        rewards=np.asarray([1.0, 2.0, 3.0]),
        values=np.asarray([0.5, 1.0, 1.5]),
        terminated=np.asarray([False, False, True]),
        next_value=0.0,
        gamma=1.0,
        gae_lambda=1.0,
    )
    assert np.allclose(returns, [6.0, 5.0, 3.0])
    assert np.allclose(advantages, [5.5, 4.0, 1.5])
```

- [ ] **Step 2: Run and verify RED**

Run `pytest -q -p no:cacheprovider tests/test_dppo.py`; expect missing module.

- [ ] **Step 3: Implement DPPO core**

Create `DPPOConfig`, `ValueNetwork`, `DPPORolloutBuffer`, `compute_gae`, and `DPPOAgent`. Keep a frozen pre-trained diffusion network for early steps and a trainable copy for final steps. Compute environment advantages once per outer transition, multiply them by `denoising_discount**k`, and apply the clipped PPO objective to trainable denoising log-probabilities. Apply value loss, gradient clipping, finite checks, and deterministic minibatch seeding.

```python
@dataclass(frozen=True)
class DPPORolloutTransition:
    state: np.ndarray
    raw_action: np.ndarray
    denoising_actions: np.ndarray
    old_log_probabilities: np.ndarray
    reward: float
    value: float
    terminated: bool


class DPPORolloutBuffer:
    def __init__(self) -> None:
        self.transitions: list[DPPORolloutTransition] = []

    def append(self, transition: DPPORolloutTransition) -> None:
        self.transitions.append(transition)

    def clear(self) -> None:
        self.transitions.clear()

    def __len__(self) -> int:
        return len(self.transitions)
```

- [ ] **Step 4: Run DPPO and diffusion tests**

Run `pytest -q -p no:cacheprovider tests/test_dppo.py tests/test_dppo_diffusion.py tests/test_dppo_checkpoint.py`; expect PASS.

- [ ] **Step 5: Commit DPPO optimizer**

```powershell
git add src/dppo.py tests/test_dppo.py
git commit -m "feat: add two-layer DPPO policy updates"
```

---

### Task 14: Integrate online DPPO training with the slow environment

**Files:**
- Create: `run_dppo_training.py`
- Create: `tests/test_dppo_training.py`
- Modify: `src/dppo_slow_timescale_env.py`
- Modify: `configs/debug.yaml`

- [ ] **Step 1: Write failing one-rollout training test**

Use a real small environment and pre-trained actor. Assert sampled raw action and chain are stored, clipped action reaches the decoder, reward is finite, projection failure remains a negative on-policy transition, update returns finite policy/value losses, and output stays under the supplied temporary root.

```python
def test_collect_rollout_keeps_rejected_action_on_policy(
    small_environment,
    dppo_agent,
) -> None:
    buffer = DPPORolloutBuffer()
    metrics = collect_rollout(small_environment, dppo_agent, buffer, seed=9)
    assert len(buffer) > 0
    assert any(transition.reward < 0.0 for transition in buffer.transitions)
    assert np.isfinite(metrics["mean_reward"])
    losses = dppo_agent.update(buffer)
    assert np.isfinite(losses["policy_loss"])
    assert np.isfinite(losses["value_loss"])
```

- [ ] **Step 2: Run and verify RED**

Run `pytest -q -p no:cacheprovider tests/test_dppo_training.py`; expect missing training helper.

- [ ] **Step 3: Implement training helper and CLI**

Expose a testable `collect_rollout(environment, agent, rollout_buffer, seed)` and keep CLI orchestration in `main()`. Add `--iterations`, `--episodes-per-iteration`, `--pretrained-checkpoint`, `--output-root`, and `--device`. Save last/best checkpoints, CSV history, reward, policy loss, value loss, raw feasibility, projection rate, repair rate, and gradient norm.

```python
def collect_rollout(
    environment: DPPOSlowTimescaleEnvironment,
    agent: DPPOAgent,
    rollout_buffer: DPPORolloutBuffer,
    seed: int,
) -> dict[str, float]:
    state = environment.reset(seed=seed)
    while True:
        sample = agent.sample_action(state, seed=seed + len(rollout_buffer))
        next_state, reward, terminated, truncated, info = environment.step(
            sample.actions[-1, 0].detach().cpu().numpy()
        )
        rollout_buffer.append(DPPORolloutTransition(
            state=state.copy(),
            raw_action=sample.actions[-1, 0].detach().cpu().numpy(),
            denoising_actions=sample.actions[:, 0].detach().cpu().numpy(),
            old_log_probabilities=sample.log_probabilities[:, 0].detach().cpu().numpy(),
            reward=float(reward),
            value=float(agent.value(state)),
            terminated=bool(terminated),
        ))
        state = next_state
        if terminated or truncated:
            break
    rewards = [transition.reward for transition in rollout_buffer.transitions]
    return {"mean_reward": float(np.mean(rewards))}
```

- [ ] **Step 4: Run one-iteration CPU smoke training**

Use temporary dataset/checkpoint/output roots. Expected: state/action dimensions printed from configuration, finite metrics, successful save/reload, and no repository result changes.

- [ ] **Step 5: Commit online training**

```powershell
git add run_dppo_training.py tests/test_dppo_training.py src/dppo_slow_timescale_env.py configs/debug.yaml
git commit -m "feat: train DPPO in the slow-timescale environment"
```

---

### Task 15: Add independent DPPO evaluation and structured reports

**Files:**
- Create: `src/dppo_evaluation.py`
- Create: `tests/test_dppo_evaluation.py`
- Create: `run_dppo_evaluation.py`

- [ ] **Step 1: Write failing evaluation-schema tests**

Require episode and summary rows to include success/SLA, mean/peak/P95/P99 delay, reliability, cold starts, active memory, cloud rate, per-VNF replica counts, primary/backup retention, raw feasibility, projection change, repair attempts/success/failure, rejection, run/route/cold/total costs, sample count, standard deviation, and confidence interval.

```python
REQUIRED_EVALUATION_COLUMNS = {
    "success_rate", "sla_success_rate", "mean_delay_ms", "peak_delay_ms",
    "p95_delay_ms", "p99_delay_ms", "mean_reliability", "cold_start_count",
    "mean_active_memory_mb", "cloud_replica_ratio", "mean_primary_retention_s",
    "mean_backup_retention_s", "raw_feasibility_rate", "projection_change_ratio",
    "repair_attempts", "repair_successes", "repair_failures", "rejection_count",
    "run_cost", "route_cost", "cold_start_cost", "total_cost",
}


def test_evaluation_schema_contains_required_metrics() -> None:
    evaluation_rows = [{key: 0.0 for key in REQUIRED_EVALUATION_COLUMNS}]
    assert REQUIRED_EVALUATION_COLUMNS <= evaluation_rows[0].keys()
    assert {"sample_count", "standard_deviation", "confidence_interval_95"} <= (
        summarize_evaluation(evaluation_rows)[0].keys()
    )
```

- [ ] **Step 2: Run and verify RED**

Run `pytest -q -p no:cacheprovider tests/test_dppo_evaluation.py`; expect missing evaluation module.

- [ ] **Step 3: Implement evaluator and isolated CLI**

Evaluate fixed Episode seeds without gradient, aggregate raw fast-slot delay samples before percentiles, and produce UTF-8-SIG CSV. CLI accepts `--checkpoint`, `--episodes`, `--seed-start`, `--output-root`, and `--device`; output action/retention distributions separately from performance summaries.

```python
@dataclass(frozen=True)
class DPPOEpisodeEvaluation:
    episode_seed: int
    metrics: Mapping[str, float]
    fast_slot_delay_samples_ms: tuple[float, ...]
    replica_counts_by_function: Mapping[int, float]


def evaluate_dppo_episode(
    environment: DPPOSlowTimescaleEnvironment,
    agent: DPPOAgent,
    *,
    episode_seed: int,
) -> DPPOEpisodeEvaluation:
    with torch.no_grad():
        return _run_evaluation_episode(environment, agent, episode_seed)
```

- [ ] **Step 4: Run tests and two-Episode temporary evaluation**

Expected: structured CSV/PNG artifacts under the temporary root and no legacy action-ratio fields.

- [ ] **Step 5: Commit evaluation**

```powershell
git add src/dppo_evaluation.py tests/test_dppo_evaluation.py run_dppo_evaluation.py
git commit -m "feat: report structured DPPO experiments"
```

---

### Task 16: Complete active-tree, full-suite, syntax, and smoke verification

**Files:**
- Modify only files exposed by verification failures.

- [ ] **Step 1: Verify no retired algorithm remains in active implementation paths**

Run the scope test and a case-insensitive source/config/run-script scan. The only allowed archive reference is the recovery note inside the approved DPPO design and plan documents.

- [ ] **Step 2: Run complete tests**

```powershell
$env:PYTHONUTF8='1'; $env:PYTHONPATH='.'; D:\Anaconda3\python.exe -X utf8 -m pytest -q -p no:cacheprovider
```

Expected: all non-retired tests pass with zero failures.

- [ ] **Step 3: Parse all Python files without leaving bytecode in the repository**

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; Get-ChildItem -Path src,tests -Recurse -Filter '*.py' -File | ForEach-Object { D:\Anaconda3\python.exe -X utf8 -m py_compile $_.FullName; if($LASTEXITCODE -ne 0){ exit $LASTEXITCODE } }
```

Expected: exit code 0.

- [ ] **Step 4: Run the complete temporary DPPO smoke chain**

Generate a small expert dataset, pre-train one epoch, train one online iteration, reload the checkpoint, and evaluate two Episodes, all below `$env:TEMP\dppo-main-smoke`. Verify no formal result path changed and every reported reward/loss is finite.

- [ ] **Step 5: Inspect diff, commit verification fixes only if needed, and push**

```powershell
git status --short
git diff --check
git log --oneline --decorate -20
```

If verification required tracked corrections, commit them as `fix: complete DPPO main algorithm verification`. Do not create an empty commit. Push `codex/dppo-main-algorithm`; formal long training remains a separate user-approved phase.
