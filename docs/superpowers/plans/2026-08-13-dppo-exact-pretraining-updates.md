# DPPO Exact Pretraining Updates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make DPPO diffusion pretraining execute an exact configured number of optimizer updates, validate at fixed intervals, and save the checkpoint with the lowest reproducible validation loss.

**Architecture:** Add one deterministic single-update primitive to `src/dppo_checkpoint.py`, then let the existing pretraining command orchestrate exactly 200 calls, fixed-seed validation, best-checkpoint replacement, and CSV history. Keep the current checkpoint format and all online PPO, reward, projection, and fast-timescale code unchanged.

**Tech Stack:** Python 3.12, PyTorch, NumPy, CSV, pytest, YAML configuration

---

### Task 1: Replace epoch configuration with exact optimizer-step configuration

**Files:**
- Modify: `configs/debug.yaml`
- Modify: `src/config.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_dppo_checkpoint.py`

- [ ] **Step 1: Write failing configuration and CLI tests**

Add assertions that `dppo.pretraining.optimizer_steps` and `validation_interval_steps` are positive integers, that `epochs` is no longer the configured counter, and that the CLI accepts `--optimizer-steps` instead of `--epochs`.

```python
def test_dppo_pretraining_requires_positive_optimizer_steps(tmp_path) -> None:
    config = _debug_config_dict()
    config["dppo"]["pretraining"]["optimizer_steps"] = 0
    with pytest.raises(ValueError, match="dppo.pretraining.optimizer_steps"):
        _load_written_config(tmp_path, config)


def test_pretraining_cli_accepts_optimizer_step_override() -> None:
    arguments = parse_arguments([
        "--dataset-root", "dataset",
        "--output-root", "output",
        "--optimizer-steps", "7",
        "--device", "cpu",
    ])
    assert arguments.optimizer_steps == 7
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
& 'D:\Anaconda3\envs\rail-dppo-gpu\python.exe' -m pytest tests/test_config.py tests/test_dppo_checkpoint.py -q
```

Expected: failures because the new configuration keys and CLI parameter are not implemented.

- [ ] **Step 3: Implement the configuration change**

Change the YAML block to:

```yaml
pretraining:
  # 使用真实 optimizer.step 次数计量训练量，数据规模变化时实验含义不变。
  optimizer_steps: 200
  # 固定间隔检查验证集，并保存验证损失最低的模型。
  validation_interval_steps: 10
  batch_size: 64
```

Update `validate_config` to call `_require_positive_integer` for both new keys. Replace `--epochs` with `--optimizer-steps` in `parse_arguments`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the same command and expect all focused tests to pass.

### Task 2: Add one deterministic diffusion optimizer update

**Files:**
- Modify: `src/dppo_checkpoint.py`
- Modify: `tests/test_dppo_checkpoint.py`

- [ ] **Step 1: Write failing single-update behavior tests**

Add a `CountingAdam` test optimizer and verify five calls produce exactly five optimizer updates. Test a sample count larger than the batch size so that the deterministic batch calculation completes one full pass before starting a reshuffled pass.

```python
class CountingAdam(torch.optim.Adam):
    def __init__(self, parameters, **kwargs):
        super().__init__(parameters, **kwargs)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


for optimizer_step in range(5):
    loss = pretrain_diffusion_step(
        model, schedule, optimizer, states, actions,
        optimizer_step=optimizer_step,
        batch_size=4,
        seed=12000,
        device="cpu",
    )
assert optimizer.step_calls == 5
assert all(math.isfinite(loss) for loss in losses)
```

- [ ] **Step 2: Run the new tests and verify RED**

Run only the new test names. Expected: import failure because `pretrain_diffusion_step` does not exist.

- [ ] **Step 3: Implement `pretrain_diffusion_step`**

The function must:

```python
batches_per_pass = math.ceil(sample_count / batch_size)
pass_index, batch_index = divmod(optimizer_step, batches_per_pass)
shuffle_generator = torch.Generator(device="cpu").manual_seed(seed + pass_index)
permutation = torch.randperm(sample_count, generator=shuffle_generator)
start = batch_index * batch_size
indices = permutation[start : start + batch_size]
noise_generator = torch.Generator(device=resolved_device).manual_seed(
    seed + 1_000_000 + optimizer_step
)
```

Then perform exactly one `zero_grad`, noise-loss calculation, backward pass, optional gradient clipping, and `optimizer.step()`. Add Chinese comments explaining the pass index and why training noise changes per update. Preserve `pretrain_diffusion_epoch` for existing callers and tests.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run `tests/test_dppo_checkpoint.py` and expect it to pass.

### Task 3: Validate periodically and retain the best checkpoint

**Files:**
- Modify: `run_dppo_pretraining.py`
- Modify: `tests/test_dppo_checkpoint.py`

- [ ] **Step 1: Write failing orchestration tests**

Create a tiny real diffusion model and dataset, execute four optimizer steps with a validation interval of two, and assert:

```python
assert [row.optimizer_step for row in history] == [2, 4]
assert loaded.epoch + 1 == min(history, key=lambda row: row.validation_loss).optimizer_step
assert history_path.read_text(encoding="utf-8").splitlines()[0] == (
    "optimizer_step,training_loss,validation_loss,is_best"
)
```

Also verify the validation seed passed at steps 2 and 4 is identical.

- [ ] **Step 2: Run the new tests and verify RED**

Run only the newly added orchestration tests. Expected: failure because step-based orchestration and history do not exist.

- [ ] **Step 3: Implement step-based orchestration**

Add an immutable history record:

```python
@dataclass(frozen=True)
class PretrainingHistoryRecord:
    optimizer_step: int
    training_loss: float
    validation_loss: float
    is_best: bool
```

In the training loop, call `pretrain_diffusion_step` exactly `optimizer_steps` times. At every validation interval and at the final step, average only the training losses since the previous validation, evaluate using constant seed `base_seed + 100000`, and save `dppo_pretrained.pt` only when validation loss strictly decreases. Save `epoch=completed_steps - 1` to retain checkpoint-v1 compatibility.

Write `pretraining_history.csv` with the four fields above. Print completed steps, best step, best validation loss, checkpoint path, and configuration hash. Replace the restored progress label with `恢复 optimizer step`.

- [ ] **Step 4: Run core tests and verify GREEN**

Run:

```powershell
& 'D:\Anaconda3\envs\rail-dppo-gpu\python.exe' -m pytest tests/test_config.py tests/test_dppo_checkpoint.py tests/test_dppo_pretraining_diagnostics.py -q
```

Expected: all tests pass with no warnings introduced by this change.

### Task 4: Run 200 GPU updates and repeat the pretraining diagnostic

**Files:**
- Create: `docs/superpowers/reports/2026-08-13-dppo-200-step-pretraining-results.md`

- [ ] **Step 1: Run exact-step pretraining on the existing medium v2 dataset**

Use the existing dataset and output directory discovered from the previous medium run. Run with `--device cuda --optimizer-steps 200`. Confirm the output states 200 completed updates and that `pretraining_history.csv` contains 20 validation records.

- [ ] **Step 2: Run the existing pretraining diagnostic only**

Compare the original random model with the newly saved best `dppo_pretrained.pt`. Do not start online PPO.

- [ ] **Step 3: Write the concise result report**

Record the best optimizer step, best validation loss, mean/median MSE and MAE changes, improved-record percentage, replica and prefix metrics, raw feasibility, and whether the predefined 10%/50% thresholds passed. State the next action directly from the threshold result.

- [ ] **Step 4: Verify, commit, and push**

Run the three core test files once more, run `git diff --check`, commit code and report, verify the worktree is clean, and push `codex/dppo-main-algorithm` to `origin`.
