# DPPO Teacher Execution Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure CLARABEL latency during online rollouts and compare cost, reliability, and balanced expert labels on held-out execution outcomes before scaling training.

**Architecture:** Extend the existing rollout metrics with four read-only solver timing aggregates. Add a separate diagnostic command that replays only validation/test expert transitions through the existing environment, groups observable outcomes by teacher name, and writes CSV/JSON artifacts without changing policy or reward behavior.

**Tech Stack:** Python 3.12, NumPy, PyTorch, CSV/JSON, pytest, existing DPPO environment and CLARABEL scheduler

---

### Task 1: Add read-only CLARABEL timing to online history

**Files:**
- Modify: `run_dppo_training.py`
- Modify: `tests/test_dppo_training.py`

- [ ] Write failing tests where two rollout steps report solver status/time `optimal/0.02`, `not_run/0.0`, and assert run count 1, total/mean/max 0.02.
- [ ] Run `pytest tests/test_dppo_training.py -q` and verify the new field assertions fail.
- [ ] In `collect_rollout`, collect time only when `fast_solver_status != "not_run"`; return count, total and max. In `_aggregate_rollout_metrics`, sum count/total, take max, and derive mean with a zero-count guard.
- [ ] Append `fast_solver_run_count`, `fast_solver_mean_time_seconds`, `fast_solver_max_time_seconds`, and `fast_solver_total_time_seconds` to `TRAINING_HISTORY_COLUMNS` and each iteration row. Add Chinese comments that these metrics are observational and do not enter reward.
- [ ] Re-run `tests/test_dppo_training.py` and verify GREEN.

### Task 2: Implement held-out teacher execution diagnostics

**Files:**
- Create: `src/dppo_teacher_diagnostics.py`
- Create: `run_dppo_teacher_diagnostics.py`
- Create: `tests/test_dppo_teacher_diagnostics.py`

- [ ] Write failing public-API tests for grouping held-out transition metrics, zero-sample teachers, insufficient-sample conclusions, strict finite validation, and canonical CSV/JSON fields.
- [ ] Run `pytest tests/test_dppo_teacher_diagnostics.py -q` and verify RED because the module/command does not exist.
- [ ] Implement immutable `TeacherTransitionMetric` and `TeacherExecutionSummary` records plus a pure `summarize_teacher_metrics` function. Always emit cost/reliability/balanced summaries and apply the design's sample-count gate.
- [ ] Implement CLI loading with `ExpertDatasetMetadata` compatibility checks. Replay only validation/test records grouped by partition and Episode seed; reset the environment once per Episode, advance in slow-step order, compare the current state to the stored state, then execute the stored expert action.
- [ ] Write `teacher_transition_metrics.csv` and strict UTF-8 `teacher_summary.json` atomically. Include partition, teacher, seed, slow step, reward, feasibility/projection/repair and solver timing fields.
- [ ] Re-run the new test file and verify GREEN.

### Task 3: Run core verification and real diagnostics

**Files:**
- Create: `docs/superpowers/reports/2026-08-13-dppo-teacher-execution-diagnostics-results.md`

- [ ] Run `tests/test_dppo_training.py tests/test_dppo_teacher_diagnostics.py tests/test_dppo_dataset.py -q`.
- [ ] Run the diagnostic on `results/dppo/gpu_exact_200_v2/dataset`, writing to `results/dppo/gpu_exact_200_v2/teacher_diagnostics`.
- [ ] Read the per-teacher sample counts, rewards, feasibility, repair and solver timing. Apply the predeclared rules and state one concrete change required before scaling.
- [ ] Write the concise report, run `git diff --check`, commit, verify clean status, and push `codex/dppo-main-algorithm`.
