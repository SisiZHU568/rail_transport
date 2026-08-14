# Phase E Online PPO Training Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the native Phase E environment complete 32 real slow frames and two PPO updates, then persist resumable online checkpoints, history, and failure audits.

**Architecture:** Normalize floating-point queue tails only at the queue commit boundary so flow remains conserved before CLARABEL sees the next snapshot. Extract reusable Phase E runtime construction from the smoke entry, add a dedicated `phase-e-online-v1` checkpoint module, and keep the formal CLI as orchestration over existing environment and trainer APIs.

**Tech Stack:** Python 3.12, PyTorch, NumPy, CVXPY/CLARABEL, pytest, CSV/JSON.

---

### Task 1: Unify Queue Residual and Completion Tolerances

**Files:**
- Modify: `src/queue_manager.py:200-206, 297-323, 485-634`
- Modify: `tests/test_queue_manager.py`

- [ ] **Step 1: Add a failing intermediate-stage tail-conservation test**

Append a test that constructs a manager with `flow_absolute_tolerance_bits=10.0`, admits a 100-bit batch, and commits a 95-bit uplink operation. Assert that the produced stage-0 fragment carries all 100 bits and no uplink residual remains:

```python
def test_commit_absorbs_numeric_tail_into_last_successful_operation() -> None:
    manager = QueueStateManager(
        StageFlowConfig((1.0,)),
        slot_seconds=1.0,
        flow_absolute_tolerance_bits=10.0,
    )
    manager.admit_batch("batch", 0, 0.0, 20.0, 100.0)
    result = manager.commit_allocation(
        plan(
            manager,
            (
                AllocationOperation(
                    QueueKey.uplink(0), "uplink", 95.0,
                    destination_node_id=0,
                ),
            ),
        ),
        context(queue_version=manager.snapshot().version, current_slot=0),
    )
    assert result.accepted
    assert result.snapshot.uplink_fragments == ()
    assert sum(item.input_equivalent_bits for item in result.snapshot.stage_fragments) == 100.0
```

- [ ] **Step 2: Run the new test and verify RED**

Run:

```powershell
$env:PYTHONUTF8='1'
D:\anaconda3\python.exe -m pytest -q tests/test_queue_manager.py::test_commit_absorbs_numeric_tail_into_last_successful_operation
```

Expected: FAIL because the current implementation retains a 5-bit residual and produces only 95 bits.

- [ ] **Step 3: Absorb only a consumed fragment's within-tolerance tail**

In `commit_allocation`, replace the fixed `1e-9` decisions with the group tolerance. Immediately after computing `amount`, consume the whole fragment when the remainder is positive but within tolerance:

```python
amount = min(remaining, quotas[index])
tail = remaining - amount
if amount > 0.0 and 0.0 < tail <= tolerance:
    amount = remaining
```

Use `remaining > tolerance` to decide whether to emit a residual. Do not delete an untouched fragment and do not relax the existing final `quota > tolerance` audit.

- [ ] **Step 4: Verify the intermediate-stage test is GREEN**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Add a failing final-stage completion-normalization test**

Create a final-stage fragment of 100 bits, commit a 95-bit execute operation with 10-bit tolerance, advance one slot, and assert the batch is completed at exactly 100 bits:

```python
def test_final_stage_tail_completes_batch_with_shared_tolerance() -> None:
    manager = QueueStateManager(
        StageFlowConfig((1.0,)),
        initial_batches=(BatchRecord("batch", 0, 0.0, 20.0, 100.0),),
        initial_stage_fragments=(
            QueueFragment("final", "batch", 0, 0, 0, None, 100.0, 0),
        ),
        flow_absolute_tolerance_bits=10.0,
    )
    result = manager.commit_allocation(
        plan(
            manager,
            (AllocationOperation(QueueKey.stage(0, 0, None), "execute", 95.0),),
        ),
        context(queue_version=manager.snapshot().version, current_slot=0),
    )
    assert result.accepted
    completed = manager.begin_slot(1).batches[0]
    assert completed.completed_input_equivalent_bits == 100.0
    assert completed.completion_slot == 1
```

- [ ] **Step 6: Run the completion test and verify RED**

Run the single new test. Expected: FAIL before the shared completion tolerance is applied.

- [ ] **Step 7: Apply `_flow_tolerance` to batch completion**

In `begin_slot`, calculate `completion_tolerance = self._flow_tolerance(batch.total_input_equivalent_bits)` and replace the hard-coded `math.isclose(..., rel_tol=1e-12, abs_tol=1e-9)` check with:

```python
if abs(completed - batch.total_input_equivalent_bits) <= completion_tolerance:
    completed = batch.total_input_equivalent_bits
    completion_slot = completion_slot_by_batch.get(batch.batch_id, completion_slot)
```

Keep the over-completion guard, but compare against the same tolerance.

- [ ] **Step 8: Run all queue and Phase B tests**

Run:

```powershell
D:\anaconda3\python.exe -m pytest -q tests/test_queue_manager.py tests/test_phase_b_orchestration.py tests/test_sla_drain.py
```

Expected: all pass.

- [ ] **Step 9: Commit Task 1**

```powershell
git add src/queue_manager.py tests/test_queue_manager.py
git commit -m "fix: normalize queue numeric tails"
```

---

### Task 2: Prove the Real Fast Layer Remains Stable Through Two Rollouts

**Files:**
- Modify: `tests/test_phase_e_environment.py`
- Modify: `tests/test_phase_e_online_trainer.py`

- [ ] **Step 1: Add a real-environment regression for the fixed behavior**

Extract the existing environment construction in `tests/test_phase_e_environment.py` into `_environment_fixture()`. Add a deterministic 32-frame loop that admits one 200,000-bit batch per frame, uses a seeded `PhaseEOnlineTrainer`, records every successful frame, and calls `update_if_ready` before the next decision. Assert:

```python
assert failure_codes == []
assert trainer.completed_update_count == 2
assert trainer.internal_failure_count == 0
assert all(item < 10.0 for item in maximum_unserved_numeric_tails)
```

Use a small DPPO test model (`diffusion_steps=3`, `update_epochs=1`, `batch_size=8`) so the test exercises real PPO logic without production-network cost.

- [ ] **Step 2: Run the 32-frame regression**

Run the new test after Task 1. Expected: PASS with two optimizer updates and no internal failure. The RED evidence for the production change is provided by Task 1's two focused tests; this end-to-end test protects the integrated behavior.

- [ ] **Step 3: Add an assertion that a failed slow frame cannot enter PPO**

Extend the test or `tests/test_phase_e_online_trainer.py` to assert that injecting `FAST_SOLVER_FAILURE` clears only the current pending segment and leaves `completed_update_count` unchanged.

- [ ] **Step 4: Run focused Phase C-E tests**

```powershell
D:\anaconda3\python.exe -m pytest -q tests/test_fast_resource_optimizer.py tests/test_phase_c_orchestration.py tests/test_phase_e_environment.py tests/test_phase_e_online_trainer.py
```

Expected: all pass.

- [ ] **Step 5: Commit Task 2**

```powershell
git add tests/test_phase_e_environment.py tests/test_phase_e_online_trainer.py
git commit -m "test: exercise two real phase E rollouts"
```

---

### Task 3: Add Resumable Phase E Online Checkpoints

**Files:**
- Create: `src/phase_e_online_checkpoint.py`
- Create: `tests/test_phase_e_online_checkpoint.py`

- [ ] **Step 1: Write failing checkpoint round-trip tests**

Define tests for `PhaseEOnlineMetadata`, `save_phase_e_online_checkpoint`, and `load_phase_e_online_checkpoint`. The round trip must restore:

```python
assert loaded.metadata.format_version == "phase-e-online-v1"
assert loaded.metadata.next_frame_index == 16
assert loaded.metadata.completed_update_count == 1
assert_state_dict_equal(restored.trainable_policy, original.trainable_policy)
assert_state_dict_equal(restored.value_network, original.value_network)
assert restored.policy_optimizer.state_dict()["state"]
assert restored.value_optimizer.state_dict()["state"]
```

Also test rejection of observation hash, action hash, and config fingerprint mismatches.

- [ ] **Step 2: Run the new file and verify RED**

```powershell
D:\anaconda3\python.exe -m pytest -q tests/test_phase_e_online_checkpoint.py
```

Expected: collection failure because the module does not exist.

- [ ] **Step 3: Implement immutable metadata and strict validation**

Create:

```python
@dataclass(frozen=True)
class PhaseEOnlineMetadata:
    format_version: str
    observation_spec_hash: str
    action_spec_hash: str
    config_sha256: str
    next_frame_index: int
    completed_update_count: int
    best_validation_reward: float | None
```

Reject any format other than `phase-e-online-v1`, blank hashes, negative counters, and non-finite validation rewards.

- [ ] **Step 4: Implement atomic save**

Serialize frozen/trainable policies, value network, both optimizers, metadata, and RNG states. Write beside the target and atomically replace it:

```python
temporary = target.with_name(f".{target.name}.tmp")
torch.save(payload, temporary)
os.replace(temporary, target)
```

Store `random.getstate()`, `np.random.get_state()`, `torch.get_rng_state()`, and CUDA RNG states when CUDA is available.

- [ ] **Step 5: Implement strict load into an existing `DPPOAgent`**

Load with `map_location=agent.device`, validate all bindings before mutating the agent, then restore network, optimizer, and RNG states. Return a frozen `LoadedPhaseEOnlineCheckpoint` containing metadata.

- [ ] **Step 6: Verify GREEN and run checkpoint regressions**

```powershell
D:\anaconda3\python.exe -m pytest -q tests/test_phase_e_online_checkpoint.py tests/test_dppo_checkpoint.py tests/test_phase_e_training_entry.py
```

Expected: all pass.

- [ ] **Step 7: Commit Task 3**

```powershell
git add src/phase_e_online_checkpoint.py tests/test_phase_e_online_checkpoint.py
git commit -m "feat: persist phase E online training state"
```

---

### Task 4: Extract Reusable Phase E Runtime Construction

**Files:**
- Create: `src/phase_e_runtime.py`
- Modify: `run_phase_e_online_smoke.py`
- Create: `tests/test_phase_e_runtime.py`

- [ ] **Step 1: Write a failing runtime-builder test**

Specify a `PhaseERuntime` dataclass containing config, controller, environment, observation adapter, network links, and training seed. Assert `build_phase_e_runtime(load_config(...), total_slow_frames=32)` produces matching observation/action dimensions and a network snapshot for every fast slot.

- [ ] **Step 2: Run the test and verify RED**

Expected: import failure for `src.phase_e_runtime`.

- [ ] **Step 3: Move construction without changing behavior**

Move `_network_links` and lines 83-146 of `run_phase_e_online_smoke.py` into `build_phase_e_runtime`. Keep all reliability bounds, flow tolerance conversion, topology construction, and reference-cost wiring identical.

- [ ] **Step 4: Make smoke use the shared runtime**

Replace the duplicated setup with:

```python
raw = load_config(args.config)
runtime = build_phase_e_runtime(raw, total_slow_frames=args.frames)
```

Keep random initialization and smoke-only output behavior unchanged.

- [ ] **Step 5: Run runtime and smoke tests**

```powershell
D:\anaconda3\python.exe -m pytest -q tests/test_phase_e_runtime.py tests/test_phase_e_training_entry.py tests/test_phase_e_environment.py
D:\anaconda3\python.exe run_phase_e_online_smoke.py --frames 2 --device cpu
```

Expected: tests pass and both frames return `code=OK`.

- [ ] **Step 6: Commit Task 4**

```powershell
git add src/phase_e_runtime.py run_phase_e_online_smoke.py tests/test_phase_e_runtime.py
git commit -m "refactor: share phase E runtime construction"
```

---

### Task 5: Implement the Formal Online Training Runner

**Files:**
- Create: `run_phase_e_online_training.py`
- Create: `tests/test_phase_e_online_training.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing CLI and artifact-contract tests**

Test that the CLI requires config, pretrained checkpoint, teacher dataset, output directory, and a positive frame count divisible by rollout length. Test that random initialization is unavailable. Expected artifacts are:

```python
assert (output / "dppo_phase_e_last.pt").is_file()
assert (output / "dppo_phase_e_best.pt").is_file()
assert (output / "training_history.csv").is_file()
assert not (output / "failure_audit.json").exists()
```

- [ ] **Step 2: Verify RED**

Run the new test file. Expected: import failure for `run_phase_e_online_training`.

- [ ] **Step 3: Implement CLI validation and startup binding**

Arguments:

```text
--config --pretrained-checkpoint --teacher-dataset --output
--frames --device --resume-checkpoint
```

Load the runtime first to derive observation/action hashes, then load both artifacts with strict bindings. Compute `config_sha256` from canonical JSON (`sort_keys=True`, compact separators).

- [ ] **Step 4: Implement the transactional training loop**

Reuse the smoke policy callback and environment call, but aggregate 16 frame results per update. After `update_if_ready` returns metrics, append one CSV row and save `last`. Save `best` when mean rollout reward improves. Mark only the final requested frame as terminated.

- [ ] **Step 5: Implement failure audit and exit behavior**

On a non-OK frame, write JSON containing failure code, frame index, start slot, pending transitions discarded, completed updates, and last successful checkpoint. Do not write a history row or success checkpoint for the failed segment; return a non-zero exit.

- [ ] **Step 6: Implement resume**

When `--resume-checkpoint` is present, restore the agent and start at `metadata.next_frame_index`. Require `trainer.pending_count == 0`; continue update counts and append to the existing CSV only when its last row matches the checkpoint.

- [ ] **Step 7: Add a 32-frame end-to-end acceptance test**

Create a small valid Phase E teacher dataset and pretraining checkpoint in `tmp_path` using existing save/load APIs. Run the formal entry on CPU for 32 frames and assert two history rows, two reloadable checkpoints, zero failure audit, and `completed_update_count == 2`.

- [ ] **Step 8: Document the new command**

Replace README references to deleted `run_dppo_training.py` with the Phase E artifact pipeline and formal command. State that existing v1/v2 checkpoints are intentionally incompatible.

- [ ] **Step 9: Run focused and full verification**

```powershell
$env:PYTHONUTF8='1'
D:\anaconda3\python.exe -m pytest -q tests/test_queue_manager.py tests/test_fast_resource_optimizer.py tests/test_phase_e_environment.py tests/test_phase_e_online_trainer.py tests/test_phase_e_online_checkpoint.py tests/test_phase_e_runtime.py tests/test_phase_e_online_training.py
D:\anaconda3\python.exe -m pytest -q
D:\anaconda3\python.exe -m compileall -q src tests run_phase_e_online_training.py
git diff --check
```

Expected: 32-frame acceptance completes two PPO updates; full suite has zero failures.

- [ ] **Step 10: Commit Task 5**

```powershell
git add run_phase_e_online_training.py tests/test_phase_e_online_training.py README.md
git commit -m "feat: run resumable phase E online training"
```

---

### Task 6: Final Review and Delivery

**Files:**
- Review all files changed by Tasks 1-5.

- [ ] **Step 1: Run the exact 32-frame command with specification-bound artifacts**

Use artifacts generated by the Phase E dataset/pretraining entry and an isolated output directory. Confirm two updates and both checkpoints reload.

- [ ] **Step 2: Run full verification again**

Run the full pytest, compileall, and diff-check commands from Task 5 after the acceptance run.

- [ ] **Step 3: Request specification and code-quality reviews**

Review against `docs/superpowers/specs/2026-08-14-phase-e-online-training-readiness-design.md`. Fix and re-review every Critical or Important issue before proceeding.

- [ ] **Step 4: Push the branch and update PR #1**

Push `codex/dppo-main-algorithm` and update the PR test plan with the final test count and 32-frame result.
