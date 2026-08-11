# Configurable DPPO Replica Bounds Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hard-coded DPPO 2/3-replica threshold with configuration-driven inclusive replica bounds while preserving one continuous replica score per VNF.

**Architecture:** `DPPOActionSpace` owns the only integer-to-continuous quantization rule. Configuration, teachers, projection, checkpoint metadata, and run scripts consume the same `minimum_replicas` and `maximum_replicas`; the action dimension and DPPO network do not change. The action schema becomes `joint-sfc-continuous-v2`, so old checkpoints fail closed instead of being interpreted with new semantics.

**Tech Stack:** Python 3.12, NumPy, PyTorch checkpoint metadata, PyYAML configuration, pytest.

---

## File map

- `configs/debug.yaml`: declare the active replica interval and v2 action schema.
- `src/config.py`: validate positive integer bounds and the configured compute-node limit.
- `src/dppo_action_space.py`: quantize one continuous score into any configured integer interval and encode teachers at bin centers.
- `src/dppo_teacher.py`: derive cost/reliability/balanced teacher counts from the action space.
- `src/dppo_projection.py`: remove the `(2, 3)` validation and accept any positive count no larger than the scenario node count.
- `src/sfc_deployment_intent.py`: remove the downstream 1/2/3 guard while retaining positive-integer and node-count consistency checks.
- `src/dppo_scenario.py`: construct the action space from the two new configuration fields.
- `src/dppo_slow_timescale_env.py`, `src/dppo_state_encoder.py`: normalize replica history by the configured maximum and version the changed state meaning as `dppo-v2-flat`.
- `src/dppo_checkpoint.py`: bind the replica interval into checkpoint compatibility metadata.
- `src/dppo_dataset.py`: mark newly generated expert actions with the v2 action schema so v1 datasets are not reused silently.
- `run_dppo_pretraining.py`, `run_dppo_training.py`, `run_dppo_evaluation.py`: construct the new metadata fields.
- `tests/test_config.py`, `tests/test_dppo_action_space.py`, `tests/test_dppo_teacher.py`, `tests/test_dppo_projection.py`, `tests/test_dppo_checkpoint.py`: cover the new public behavior.
- Directly affected pretraining/training tests: update fixtures to v2 metadata without expanding their scope.

### Task 1: Configuration and action quantization

**Files:**
- Modify: `configs/debug.yaml`
- Modify: `src/config.py`
- Modify: `src/dppo_action_space.py`
- Test: `tests/test_config.py`
- Test: `tests/test_dppo_action_space.py`

- [ ] **Step 1: Write failing configuration tests**

Replace the threshold-specific test with explicit bound validation:

```python
@pytest.mark.parametrize(
    ("field", "value"),
    (("minimum_replicas", True), ("minimum_replicas", 0),
     ("maximum_replicas", 0), ("maximum_replicas", 2.5)),
)
def test_validate_config_rejects_invalid_replica_bounds(field, value):
    config = deepcopy(load_config("configs/debug.yaml"))
    config["dppo"]["action"][field] = value
    with pytest.raises(ValueError, match=field):
        validate_config(config)


def test_validate_config_rejects_reversed_replica_bounds():
    config = deepcopy(load_config("configs/debug.yaml"))
    config["dppo"]["action"]["minimum_replicas"] = 4
    config["dppo"]["action"]["maximum_replicas"] = 3
    with pytest.raises(ValueError, match="minimum_replicas.*maximum_replicas"):
        validate_config(config)


def test_validate_config_rejects_replica_maximum_above_compute_nodes():
    config = deepcopy(load_config("configs/debug.yaml"))
    config["dppo"]["action"]["maximum_replicas"] = 99
    with pytest.raises(ValueError, match="compute_node_count"):
        validate_config(config)
```

- [ ] **Step 2: Write failing action-space tests**

Add a four-bin decode test and fixed-range round trip:

```python
def test_replica_score_uniformly_decodes_configured_interval():
    space = DPPOActionSpace(
        ScenarioDimensions(tuple(range(5)), 5, (0, 1, 2, 3)),
        maximum_retention_seconds=20.0,
        minimum_replicas=2,
        maximum_replicas=5,
    )
    action = np.zeros(space.action_dim, dtype=np.float32)
    action[space.replica_score_slice] = np.asarray(
        [-1.0, -0.25, 0.25, 1.0], dtype=np.float32
    )
    assert [item.replica_count for item in space.decode(action).function_actions] == [2, 3, 4, 5]


def test_fixed_replica_interval_round_trips_teacher_action():
    space = DPPOActionSpace(
        ScenarioDimensions(tuple(range(5)), 5, (0,)),
        maximum_retention_seconds=20.0,
        minimum_replicas=4,
        maximum_replicas=4,
    )
    teacher = DecodedFunctionAction(
        function_id=0,
        ranked_node_ids=space.dimensions.compute_node_ids,
        replica_count=4,
        primary_retention_seconds=10.0,
        backup_retention_seconds=5.0,
    )
    encoded = space.encode_teacher_action((teacher,))
    assert space.decode(encoded).function_actions[0].replica_count == 4
```

- [ ] **Step 3: Run the focused tests and confirm RED**

Run:

```powershell
$env:PYTHONUTF8='1'
D:\Anaconda3\python.exe -m pytest tests/test_config.py tests/test_dppo_action_space.py -q --basetemp=..\.pytest_tmp\replica_bounds_red1
```

Expected: failures mention missing `minimum_replicas` / `maximum_replicas` constructor arguments and the old threshold validation.

- [ ] **Step 4: Implement configuration validation**

In `src/config.py`, replace `replica_threshold` validation with positive integer checks and compute the configured DPPO deployment-node count from trackside MECs and the cloud. The onboard node is not part of `ScenarioDimensions.compute_node_ids`, so it must not increase this limit:

```python
minimum_replicas = _require_positive_integer(action, "minimum_replicas")
maximum_replicas = _require_positive_integer(action, "maximum_replicas")
if minimum_replicas > maximum_replicas:
    raise ValueError("minimum_replicas 不能大于 maximum_replicas。")
compute_node_count = int(config["topology"]["mec_count"])
compute_node_count += int(bool(config["topology"]["include_cloud"]))
if maximum_replicas > compute_node_count:
    raise ValueError("maximum_replicas 不能大于 compute_node_count。")
```

Update `configs/debug.yaml`:

```yaml
schema_version: joint-sfc-continuous-v2
minimum_replicas: 2
maximum_replicas: 3
```

Delete `replica_threshold`.

- [ ] **Step 5: Implement interval quantization**

Change `DPPOActionSpace.__init__` to accept and expose `minimum_replicas` and `maximum_replicas`. Add these helpers with Chinese comments explaining that action dimension stays fixed:

```python
def _decode_replica_count(self, score: float) -> int:
    candidate_count = self.maximum_replicas - self.minimum_replicas + 1
    normalized = (float(score) + 1.0) / 2.0
    index = min(int(math.floor(normalized * candidate_count)), candidate_count - 1)
    return self.minimum_replicas + index

def _encode_replica_count(self, replica_count: int) -> float:
    if isinstance(replica_count, bool) or not isinstance(replica_count, int):
        raise ValueError("教师动作的副本数必须是整数。")
    if not self.minimum_replicas <= replica_count <= self.maximum_replicas:
        raise ValueError("教师动作的副本数必须位于配置上下限内。")
    candidate_count = self.maximum_replicas - self.minimum_replicas + 1
    index = replica_count - self.minimum_replicas
    return -1.0 + 2.0 * (index + 0.5) / candidate_count
```

Use the helpers from `decode` and `encode_teacher_action`; remove all threshold fields and `(2, 3)` checks from this file.

- [ ] **Step 6: Run the focused tests and confirm GREEN**

Run the same two test files. Expected: all pass.

- [ ] **Step 7: Commit Task 1**

```powershell
D:\Git\cmd\git.exe add configs/debug.yaml src/config.py src/dppo_action_space.py tests/test_config.py tests/test_dppo_action_space.py
D:\Git\cmd\git.exe commit -m "feat: configure DPPO replica count bounds"
```

### Task 2: Teachers, projection, and scenario wiring

**Files:**
- Modify: `src/dppo_teacher.py`
- Modify: `src/dppo_projection.py`
- Modify: `src/sfc_deployment_intent.py`
- Modify: `src/dppo_scenario.py`
- Test: `tests/test_dppo_teacher.py`
- Test: `tests/test_dppo_projection.py`

- [ ] **Step 1: Write failing teacher behavior tests**

Build a scenario with `[2, 5]` and assert public teacher behavior:

```python
def test_teachers_derive_replica_count_from_configured_bounds():
    config = load_config(DEBUG_CONFIG)
    config["dppo"]["action"]["minimum_replicas"] = 2
    config["dppo"]["action"]["maximum_replicas"] = 5
    scenario = build_dppo_scenario(config)
    snapshot = scenario.current_public_snapshot()
    expected = {"cost": 2, "balanced": 3, "reliability": 5}
    for name, replica_count in expected.items():
        proposal = build_simulation_teacher(name, scenario).propose(snapshot)
        assert {
            item.replica_count for item in proposal.decoded_action.function_actions
        } == {replica_count}
```

- [ ] **Step 2: Write a failing projection test above three replicas**

```python
def test_projector_accepts_four_replica_action_when_nodes_are_available():
    dimensions = ScenarioDimensions(tuple(range(5)), 5, (0,))
    decoded = _decoded_action(
        (0,), dimensions.compute_node_ids, replica_count=4
    )
    projector = DPPOProjector(
        dimensions=dimensions,
        resource_demands={
            0: ProjectionResourceDemand(cpu=1.0, memory_mb=64.0),
        },
        minimum_distinct_fault_domains=2,
    )
    free_cpu, free_memory_mb = _uniform_resources(
        dimensions.compute_node_ids
    )
    result = projector.project(
        decoded_action=decoded,
        operational_node_ids=frozenset(dimensions.compute_node_ids),
        free_cpu=free_cpu,
        free_memory_mb=free_memory_mb,
        fault_domains={
            node_id: node_id for node_id in dimensions.compute_node_ids
        },
    )
    assert result.success is True
    assert result.function_intents[0].replica_count == 4
```

- [ ] **Step 3: Run focused tests and confirm RED**

Run:

```powershell
$env:PYTHONUTF8='1'
D:\Anaconda3\python.exe -m pytest tests/test_dppo_teacher.py tests/test_dppo_projection.py -q --basetemp=..\.pytest_tmp\replica_bounds_red2
```

Expected: teachers still return hard-coded 2/3 and projection rejects four replicas.

- [ ] **Step 4: Make teachers configuration-driven**

Replace class constants with one overridable method:

```python
def _replica_count(self) -> int:
    return self.context.action_space.minimum_replicas
```

`ReliabilityTeacher` returns `maximum_replicas`; `BalancedTeacher` returns
`min(minimum_replicas + 1, maximum_replicas)`. Pass the local count into `_diverse_prefix` so ranking and emitted actions use the same value.

- [ ] **Step 5: Remove the projection 2/3 guard and wire the scenario**

In `src/dppo_projection.py`, validate only the scenario-independent legal range:

```python
if (
    isinstance(action.replica_count, bool)
    or not isinstance(action.replica_count, int)
    or not 1 <= action.replica_count <= self.dimensions.compute_node_count
):
    raise ValueError("replica_count 必须是未超过计算节点数的正整数。")
```

In `src/dppo_scenario.py`, pass both bounds to `DPPOActionSpace` and remove the threshold argument.

- [ ] **Step 6: Run focused tests and confirm GREEN**

Run the same two test files. Expected: all pass.

- [ ] **Step 7: Commit Task 2**

```powershell
D:\Git\cmd\git.exe add src/dppo_teacher.py src/dppo_projection.py src/dppo_scenario.py tests/test_dppo_teacher.py tests/test_dppo_projection.py
D:\Git\cmd\git.exe commit -m "feat: propagate configurable replica bounds"
```

### Task 3: Checkpoint and script compatibility

**Files:**
- Modify: `src/dppo_checkpoint.py`
- Modify: `run_dppo_pretraining.py`
- Modify: `run_dppo_training.py`
- Modify: `run_dppo_evaluation.py`
- Modify: checkpoint fixtures in directly affected tests
- Test: `tests/test_dppo_checkpoint.py`
- Test: `tests/test_dppo_dataset.py`

- [ ] **Step 1: Write failing metadata tests**

Update `_metadata()` to v2 and add bounds:

```python
return DPPOCheckpointMetadata(
    state_schema_version="dppo-v1-flat",
    action_schema_version="joint-sfc-continuous-v2",
    state_dim=78,
    action_dim=27,
    mec_count=5,
    compute_node_count=7,
    function_count=3,
    diffusion_steps=20,
    fine_tuned_steps=5,
    maximum_retention_seconds=20.0,
    minimum_replicas=2,
    maximum_replicas=5,
    config_hash="test-config",
)
```

Add a compatibility test that changes only `maximum_replicas` and expects checkpoint loading to reject the metadata mismatch.

- [ ] **Step 2: Run the checkpoint test and confirm RED**

Run:

```powershell
$env:PYTHONUTF8='1'
D:\Anaconda3\python.exe -m pytest tests/test_dppo_checkpoint.py -q --basetemp=..\.pytest_tmp\replica_bounds_red3
```

Expected: `DPPOCheckpointMetadata` does not accept the two new fields.

- [ ] **Step 3: Replace threshold metadata with bounds**

In `DPPOCheckpointMetadata`, replace `replica_threshold` with:

```python
minimum_replicas: int
maximum_replicas: int
```

Validate both as positive non-boolean integers and require
`minimum_replicas <= maximum_replicas <= compute_node_count`. Do not add v1 conversion code; strict dataclass field parsing must reject old metadata.

- [ ] **Step 4: Update all metadata constructors and test fixtures**

Each run script must read exactly:

```python
minimum_replicas=int(action_config["minimum_replicas"]),
maximum_replicas=int(action_config["maximum_replicas"]),
```

Update action schema literals in test fixtures to `joint-sfc-continuous-v2`. Remove all `replica_threshold` fixture fields.

- [ ] **Step 5: Run directly affected checkpoint and training tests**

Run:

```powershell
$env:PYTHONUTF8='1'
D:\Anaconda3\python.exe -m pytest tests/test_dppo_checkpoint.py tests/test_dppo_training.py -q --basetemp=..\.pytest_tmp\replica_bounds_green3
```

Expected: all pass.

- [ ] **Step 6: Commit Task 3**

```powershell
D:\Git\cmd\git.exe add src/dppo_checkpoint.py run_dppo_pretraining.py run_dppo_training.py run_dppo_evaluation.py tests
D:\Git\cmd\git.exe commit -m "feat: bind replica bounds to DPPO checkpoints"
```

### Task 4: Focused verification and documentation cleanup

**Files:**
- Modify only directly affected README/spec wording if active documentation still describes threshold-based 2/3 encoding.

- [ ] **Step 1: Check for retired active-code semantics**

Run:

```powershell
D:\Git\cmd\git.exe grep -n "replica_threshold" -- src configs run_dppo_*.py tests
D:\Git\cmd\git.exe grep -n "replica_count not in (2, 3)" -- src tests
```

Expected: no active source/config/test matches. Historical design and implementation plan documents may retain the old term only when explaining migration.

- [ ] **Step 2: Run the complete focused suite**

Run:

```powershell
$env:PYTHONUTF8='1'
D:\Anaconda3\python.exe -m pytest tests/test_config.py tests/test_dppo_action_space.py tests/test_dppo_teacher.py tests/test_dppo_projection.py tests/test_dppo_checkpoint.py tests/test_dppo_training.py -q --basetemp=..\.pytest_tmp\replica_bounds_final
```

Expected: all tests pass. Do not run the unrelated full project suite.

- [ ] **Step 3: Verify syntax and diff hygiene**

Run:

```powershell
D:\Anaconda3\python.exe -m py_compile src/config.py src/dppo_action_space.py src/dppo_teacher.py src/dppo_projection.py src/dppo_scenario.py src/dppo_checkpoint.py run_dppo_pretraining.py run_dppo_training.py run_dppo_evaluation.py
D:\Git\cmd\git.exe diff --check
D:\Git\cmd\git.exe status --short
```

Expected: syntax and diff checks succeed; status lists only intended files.

- [ ] **Step 4: Commit any final documentation-only cleanup**

If Step 1 required wording changes:

```powershell
D:\Git\cmd\git.exe add README.md docs
D:\Git\cmd\git.exe commit -m "docs: explain configurable DPPO replicas"
```

- [ ] **Step 5: Push the completed branch**

```powershell
D:\Git\cmd\git.exe push origin codex/dppo-main-algorithm
```

Expected: GitHub branch advances to the final local commit.
