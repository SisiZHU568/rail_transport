# DPPO Small-Scale Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run 20 DPPO iterations with 2 episodes per iteration on the verified GPU pipeline and determine whether reward shows improvement, instability, or an initial convergence plateau.

**Architecture:** Reuse the GPU smoke pretrained checkpoint and its bound stability profile. The DPPO networks run on CUDA while each fast-timescale convex allocation remains on CPU CLARABEL. Write all new runtime artifacts to an isolated ignored result directory and analyze the existing training CSV without changing production code.

**Tech Stack:** Python 3.12, PyTorch 2.11 CUDA 12.8, CVXPY 1.6.7, CLARABEL, pytest, PowerShell

---

### Task 1: Verify the bound training inputs

**Files:**
- Read: `results/dppo/gpu_smoke_v2/pretraining/dppo_pretrained.pt`
- Read: `results/dppo/gpu_smoke_v2/calibration/stability_profile.json`
- Test: `tests/test_config.py`
- Test: `tests/test_fast_convex_scheduler.py`
- Test: `tests/test_dppo_slow_timescale_env.py`

- [ ] **Step 1: Run the core checks in the GPU environment**

Run:

```powershell
D:\Anaconda3\envs\rail-dppo-gpu\python.exe -m pytest tests/test_config.py tests/test_fast_convex_scheduler.py tests/test_dppo_slow_timescale_env.py -q
```

Expected: 29 tests pass.

- [ ] **Step 2: Verify the checkpoint/profile binding**

Load both files with the public stability-profile parser and assert that the profile is qualified, selects `clip_ratio=0.1`, matches the checkpoint SHA, and uses the current configuration hash.

Expected: the validation command exits with code 0.

### Task 2: Run 20 × 2 online training

**Files:**
- Create at runtime: `results/dppo/gpu_pilot_v2/dppo_online_best.pt`
- Create at runtime: `results/dppo/gpu_pilot_v2/dppo_online_last.pt`
- Create at runtime: `results/dppo/gpu_pilot_v2/training_history.csv`

- [ ] **Step 1: Start the isolated training run**

Run:

```powershell
D:\Anaconda3\envs\rail-dppo-gpu\python.exe run_dppo_training.py `
  --config configs/debug.yaml `
  --iterations 20 `
  --episodes-per-iteration 2 `
  --pretrained-checkpoint results/dppo/gpu_smoke_v2/pretraining/dppo_pretrained.pt `
  --stability-profile results/dppo/gpu_smoke_v2/calibration/stability_profile.json `
  --output-root results/dppo/gpu_pilot_v2 `
  --device cuda
```

Expected: all 20 iterations finish and the process exits with code 0.

- [ ] **Step 2: Confirm required runtime artifacts**

Assert that both checkpoints and `training_history.csv` exist, that the CSV has exactly 20 rows, every tracked numeric field is finite, optimizer steps occurred, and the last checkpoint contains CUDA RNG state plus the expected stability-profile digest.

Expected: artifact validation exits with code 0.

### Task 3: Analyze reward and constraint stability

**Files:**
- Read: `results/dppo/gpu_pilot_v2/training_history.csv`

- [ ] **Step 1: Calculate convergence indicators**

Calculate and report:

- raw mean reward for each iteration;
- 5-iteration moving mean reward;
- mean reward of iterations 1–5 and 16–20;
- reward standard deviation in iterations 1–5 and 16–20;
- KL early-stop count, maximum KL, mean repair success rate, and mean raw feasibility rate.

Expected: all values are finite and computed from 20 rows.

- [ ] **Step 2: Classify the result conservatively**

Use one of three conclusions:

- `出现学习趋势` when the last-five reward mean improves over the first-five mean but the moving mean has not stabilized;
- `已进入初步平台` when the last moving means change only slightly while constraint indicators remain stable;
- `尚未收敛或不稳定` when rewards do not improve, oscillate strongly, or numerical/constraint metrics deteriorate.

Do not claim formal convergence from only 40 episodes.

- [ ] **Step 3: Verify Git scope**

Run:

```powershell
D:\Git\cmd\git.exe status --short
```

Expected: runtime results remain ignored and no production file is modified.

