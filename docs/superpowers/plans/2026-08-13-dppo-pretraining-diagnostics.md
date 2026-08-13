# DPPO Pretraining Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a read-only, reproducible diagnostic that compares a randomly initialized diffusion policy with a pretrained policy on held-out expert records, covering both action fitting and raw deployment feasibility.

**Architecture:** A focused library module owns immutable metric rows, decoded-action comparison, aggregation, and directional classification. A separate CLI owns configuration/checkpoint gates, causal scenario replay, equal-seed sampling, projection checks, and atomic CSV/JSON output. Existing training, reward, projection, and fast-scheduler code remains unchanged.

**Tech Stack:** Python 3.12, NumPy, PyTorch 2.11 CUDA, existing DPPO dataset/action-space/checkpoint/projector APIs, pytest

---

### Task 1: Define per-record fitting and semantic metrics

**Files:**
- Create: `src/dppo_pretraining_diagnostics.py`
- Create: `tests/test_dppo_pretraining_diagnostics.py`

- [ ] **Step 1: Write failing tests for metric rows**

Create tests that construct a small real `DPPOActionSpace`, decode one expert action and one generated action, and require a public function with this behavior:

```python
row = evaluate_generated_action(
    partition="validation",
    model_name="pretrained",
    episode_seed=60000,
    slow_step=1,
    teacher_name="cost",
    generated_action=generated,
    expert_action=expert,
    action_space=action_space,
    projection_result=projection_result,
)

assert row.action_mse == pytest.approx(np.mean((generated - expert) ** 2))
assert row.action_mae == pytest.approx(np.mean(np.abs(generated - expert)))
assert row.replica_accuracy == pytest.approx(2 / 3)
assert row.node_prefix_exact_rate == pytest.approx(1 / 3)
assert row.node_prefix_set_rate == pytest.approx(2 / 3)
assert row.raw_feasible is False
assert row.projection_change_ratio == pytest.approx(1 / 6)
```

Also require rejection of non-finite actions, wrong shapes, unsupported partition/model names, and mismatched function order.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
D:\Anaconda3\envs\rail-dppo-gpu\python.exe -m pytest tests/test_dppo_pretraining_diagnostics.py -q --tb=short --basetemp=.pytest-tmp
```

Expected: collection fails because `src.dppo_pretraining_diagnostics` does not exist.

- [ ] **Step 3: Implement the immutable metric row and evaluator**

Add `GeneratedActionMetric` and `evaluate_generated_action`. The evaluator must:

- copy all actions as finite `float32` vectors;
- compute continuous MSE/MAE in `float64`;
- decode generated and expert actions with the supplied action space;
- compare each VNF by function ID;
- define the deployment prefix length from the expert replica count;
- compute ordered prefix, unordered prefix, replica, primary-retention, and backup-retention metrics;
- preserve projection raw-feasible/success/change ratio/reasons;
- store generated replica counts as a tuple for later aggregation.

Include concise Chinese comments explaining why the expert replica count defines the comparison prefix.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the same command. Expected: all Task 1 tests pass.

### Task 2: Aggregate paired random/pretrained results

**Files:**
- Modify: `src/dppo_pretraining_diagnostics.py`
- Modify: `tests/test_dppo_pretraining_diagnostics.py`

- [ ] **Step 1: Write failing aggregation tests**

Add paired rows for two records and require:

```python
summary = summarize_diagnostic_rows(rows)

assert summary["record_count"] == 2
assert summary["models"]["random"]["mean_action_mse"] == pytest.approx(0.5)
assert summary["models"]["pretrained"]["mean_action_mse"] == pytest.approx(0.4)
assert summary["comparison"]["relative_mean_mse_reduction"] == pytest.approx(0.2)
assert summary["comparison"]["record_mse_improvement_rate"] == pytest.approx(0.5)
assert summary["conclusion"] == "continuous_signal_not_learned"
```

Add a second case meeting both thresholds and improving at least one semantic/feasibility metric; require `continuous_and_deployment_signal_learned`. Require explicit rejection when a record is missing either the random or pretrained row.

- [ ] **Step 2: Verify RED**

Run the focused test file. Expected: failures because aggregation does not exist.

- [ ] **Step 3: Implement deterministic aggregation and classification**

Implement `summarize_diagnostic_rows(rows)` returning only JSON-safe finite scalars, strings, lists, and dictionaries. For each model report:

- mean/median action MSE and MAE;
- mean replica/prefix/retention metrics;
- raw feasibility and projection success rates;
- mean projection change ratio;
- sorted projection failure counts;
- sorted replica-count distribution.

For comparison report relative MSE reduction, per-record MSE improvement rate, and pretrained-minus-random deltas for semantic and feasibility metrics. Use the approved 10% and greater-than-50% continuous-signal thresholds. Classify into:

- `continuous_signal_not_learned`;
- `continuous_only_mapping_problem`;
- `continuous_and_deployment_signal_learned`.

- [ ] **Step 4: Verify GREEN**

Run the focused test file. Expected: all Task 1–2 tests pass.

### Task 3: Add causal replay and fair sampling

**Files:**
- Create: `run_dppo_pretraining_diagnostics.py`
- Modify: `tests/test_dppo_pretraining_diagnostics.py`

- [ ] **Step 1: Write failing replay and sampling tests**

Using `configs/debug.yaml` and a two-step cost-teacher record collected by the real dataset collector, require:

```python
contexts = replay_diagnostic_contexts(config, records)
assert len(contexts) == len(records)
np.testing.assert_allclose(contexts[0].state, records[0].state, rtol=1e-6, atol=1e-6)
assert contexts[0].projection_inputs.operational_node_ids
```

Change one saved state element and require a `ValueError` containing the episode seed and slow step. Add a small equal-architecture pair of models and require `sample_fair_actions` to use the same seed, return clipped finite actions, reproduce exactly on repeated calls, and produce identical outputs when model parameters are identical.

- [ ] **Step 2: Verify RED**

Run the focused test file. Expected: import or attribute failures because the CLI functions do not exist.

- [ ] **Step 3: Implement replay and fair sampling**

Add:

```python
@dataclass(frozen=True)
class ReplayDiagnosticContext:
    partition: str
    record: ExpertTransitionRecord
    state: np.ndarray
    projection_inputs: ProjectionInputs
```

Implement `replay_diagnostic_contexts(config, partitioned_records)` by grouping on `(partition, episode_seed)`, rebuilding the configured teacher, resetting with the episode seed, capturing `current_projection_inputs()` before the teacher action is executed, and comparing the replay state to the stored state. Do not reuse an execution-after state for projection.

Implement `sample_fair_actions(random_model, pretrained_model, schedule, states, seed, minimum_sampling_standard_deviation, action_space, device)`. Put both models in evaluation mode, call the existing full-chain sampler with the same seed, select the final action, clip through the action space, and restore prior train/eval modes.

- [ ] **Step 4: Verify GREEN**

Run the focused test file. Expected: all Task 1–3 tests pass.

### Task 4: Add the read-only CLI and stable outputs

**Files:**
- Modify: `run_dppo_pretraining_diagnostics.py`
- Modify: `tests/test_dppo_pretraining_diagnostics.py`

- [ ] **Step 1: Write failing CLI and output tests**

Require `parse_arguments` to fail unless `--dataset-root`, `--pretrained-checkpoint`, and `--output-root` are explicit. Require a small real run to write:

```text
per_record_metrics.csv
summary.json
```

Assert CSV has two rows per held-out record with stable headers; JSON contains schema/config/checkpoint information, model aggregates, comparison, and conclusion; `json.loads` succeeds and recursively contains no NaN/Infinity. Monkeypatch `os.replace` to fail and require any pre-existing output file to remain byte-for-byte unchanged.

- [ ] **Step 2: Verify RED**

Run the focused test file. Expected: CLI/output assertions fail.

- [ ] **Step 3: Implement compatibility gates, model construction, evaluation, and atomic output**

The CLI must:

1. load config and build the current scenario;
2. construct exact expected dataset/checkpoint metadata;
3. load validation and test records and require both non-empty and behavior-cloning eligible;
4. load the pretrained checkpoint on the requested device;
5. construct the random model with the same architecture under `torch.random.fork_rng` and the configured pretraining seed;
6. replay contexts, sample paired actions, project both decoded actions against each pre-execution snapshot, and call the pure evaluator;
7. write stable CSV and strict sorted JSON using same-directory temporary files, `flush`, `fsync`, and `os.replace`;
8. print record count, random/pretrained MSE, raw feasibility, and conclusion.

Use a seed formula that cannot collide across the current two slow steps:

```python
sample_seed = diagnostic_seed + record.episode_seed * 1000 + record.slow_step
```

Reject an integer overflow beyond PyTorch generator seed range instead of wrapping silently.

- [ ] **Step 4: Verify the focused suite and core dependencies**

Run:

```powershell
D:\Anaconda3\envs\rail-dppo-gpu\python.exe -m pytest tests/test_dppo_pretraining_diagnostics.py tests/test_dppo_dataset.py tests/test_dppo_checkpoint.py tests/test_dppo_diffusion.py -q --tb=short --basetemp=.pytest-tmp
```

Expected: all selected tests pass.

### Task 5: Run the real medium-dataset diagnostic and report

**Files:**
- Create at runtime: `results/dppo/gpu_feasible_medium_v2/pretraining_diagnostics/per_record_metrics.csv`
- Create at runtime: `results/dppo/gpu_feasible_medium_v2/pretraining_diagnostics/summary.json`
- Create: `docs/superpowers/reports/2026-08-13-dppo-pretraining-diagnostics-results.md`

- [ ] **Step 1: Run the diagnostic on GPU**

Run:

```powershell
D:\Anaconda3\envs\rail-dppo-gpu\python.exe run_dppo_pretraining_diagnostics.py `
  --config configs/debug.yaml `
  --dataset-root results/dppo/gpu_feasible_medium_v2/dataset `
  --pretrained-checkpoint results/dppo/gpu_feasible_medium_v2/pretraining/dppo_pretrained.pt `
  --output-root results/dppo/gpu_feasible_medium_v2/pretraining_diagnostics `
  --device cuda `
  --seed 73000
```

Expected: 46 per-model rows for 23 held-out records, finite JSON metrics, and one of the three approved conclusions.

- [ ] **Step 2: Review the result without starting new training**

Write a concise Chinese report containing:

- random versus pretrained mean/median MSE and MAE;
- record improvement rate;
- replica and node-prefix deltas;
- retention-error deltas;
- raw feasibility, projection success, change-ratio deltas;
- teacher distribution and sample-size limitation;
- the approved directional conclusion and the next single development decision.

- [ ] **Step 3: Fresh verification**

Run the Task 4 selected tests again, validate `summary.json` with strict JSON parsing, run `git diff --check`, and confirm only the two new source files, one new test file, plan, and report are tracked.

- [ ] **Step 4: Commit and push**

Commit source/tests/plan/report to `codex/dppo-main-algorithm`. Runtime CSV, JSON, datasets, and checkpoints remain ignored. Push and verify local and remote commit IDs match.
