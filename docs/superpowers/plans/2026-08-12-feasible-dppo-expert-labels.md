# Feasible DPPO Expert Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Store projection-corrected, raw-feasible continuous actions as DPPO behavior-cloning labels without changing the reward function or online scheduler.

**Architecture:** The dataset collector captures the causal projection inputs, executes the teacher proposal through the unchanged environment, converts a successful projected intent back into a complete ranked DPPO action, and verifies that reconstructed label against the captured inputs. Dataset schema v2 separately records teacher-proposal feasibility and saved-label feasibility.

**Tech Stack:** Python 3.12, NumPy, pytest, existing DPPO action space and projector

---

### Task 1: Define the v2 record behavior

**Files:**
- Modify: `tests/test_dppo_dataset.py`
- Modify: `src/dppo_dataset.py`

- [x] **Step 1: Write failing record and collection tests**

Add tests proving that:

```python
assert record.teacher_proposal_raw_feasible is False
assert record.expert_action_feasible is True
```

and that decoding `record.expert_action` produces the projected node prefix while preserving the teacher's remaining node order.

- [x] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
D:\Anaconda3\envs\rail-dppo-gpu\python.exe -m pytest tests/test_dppo_dataset.py -q
```

Expected: failures because the v2 fields and corrected label behavior do not exist.

- [x] **Step 3: Implement the minimum corrected-label conversion**

In `src/dppo_dataset.py`:

- change the schema version to `dppo-expert-v2`;
- replace `raw_feasible` with `teacher_proposal_raw_feasible`;
- add `expert_action_feasible`;
- reconstruct each full ranking as projected nodes followed by unselected nodes in original teacher order;
- encode and re-project the reconstructed label against the pre-execution projection inputs;
- keep the original proposal only when projection failed, and mark it unsuitable for behavior cloning.

- [x] **Step 4: Run the focused tests and verify GREEN**

Run the same test command. Expected: all dataset tests pass.

### Task 2: Persist and filter v2 labels

**Files:**
- Modify: `tests/test_dppo_dataset.py`
- Modify: `src/dppo_dataset.py`

- [x] **Step 1: Write failing persistence and filtering tests**

Require NPZ round-trip fields:

```python
teacher_proposal_raw_feasible
expert_action_feasible
```

and require behavior-cloning partitions to accept a record only when both `expert_action_feasible` and `final_feasible` are true.

- [x] **Step 2: Verify RED**

Run `tests/test_dppo_dataset.py`. Expected: v2 persistence/filter assertions fail.

- [x] **Step 3: Implement v2 persistence and filtering**

Update array serialization, loading, equality and behavior-cloning acceptance. Keep rejected records in diagnostics with explicit reasons.

- [x] **Step 4: Verify GREEN and the small dependency set**

Run:

```powershell
D:\Anaconda3\envs\rail-dppo-gpu\python.exe -m pytest tests/test_dppo_dataset.py tests/test_dppo_checkpoint.py tests/test_config.py -q
```

Expected: all selected tests pass.

### Task 3: Generate and audit a small v2 dataset

**Files:**
- Create at runtime: `results/dppo/feasible_label_smoke_v2/`

- [x] **Step 1: Generate six episodes**

Run the existing dataset CLI with six episodes, seed start `57000`, and the isolated output root.

- [x] **Step 2: Audit the generated arrays**

Assert:

- schema is `dppo-expert-v2`;
- every behavior-cloning row has `expert_action_feasible=True` and `final_feasible=True`;
- at least one accepted row has `teacher_proposal_raw_feasible=False`, proving that corrected labels are being used;
- no behavior-cloning partition stores an infeasible expert label.

- [x] **Step 3: Verify unchanged algorithm scope**

Run `git diff --check` and confirm that `src/rl_reward.py`, `src/dppo.py`, `src/dppo_projection.py`, and `src/fast_convex_scheduler.py` are unchanged.

- [x] **Step 4: Commit and push**

Commit only the plan, source and tests. Runtime datasets remain ignored.
