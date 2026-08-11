# Official DPPO Core Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use the official DPPO policy-loss and denoising-MDP formulation in the existing rail Serverless SFC agent without importing the upstream robotics stack.

**Architecture:** Add one small, attributed algorithm module derived from the official `PPODiffusion.loss`, then call it from the existing `DPPOAgent.update`. Keep state encoding, dynamic joint-action sizing, action projection, fast-layer execution, rollout storage and the training CLI in the rail project.

**Tech Stack:** Python 3.12, PyTorch, NumPy, pytest, YAML configuration

---

### Task 1: Add provenance and official policy-loss core

**Files:**
- Create: `third_party/irom_dppo/LICENSE`
- Create: `third_party/irom_dppo/NOTICE.md`
- Create: `src/dppo_official_core.py`
- Create: `tests/test_dppo_official_core.py`

- [ ] **Step 1: Write failing schedule and loss tests**

```python
def test_official_clip_schedule_grows_from_base_to_maximum():
    schedule = official_clip_schedule(
        fine_tuned_steps=5,
        maximum_clip_ratio=0.20,
        base_clip_ratio=0.001,
        growth_rate=3.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert schedule.shape == (5,)
    assert schedule[0].item() == pytest.approx(0.001)
    assert schedule[-1].item() == pytest.approx(0.20)
    assert torch.all(schedule[1:] > schedule[:-1])


def test_official_policy_loss_is_finite_and_backpropagates():
    new_logp = torch.tensor([[0.0, 0.05, -0.03]], requires_grad=True)
    old_logp = torch.zeros_like(new_logp)
    advantages = torch.tensor([1.5])
    result = official_dppo_policy_loss(
        new_logp,
        old_logp,
        advantages,
        gamma_denoising=0.99,
        maximum_clip_ratio=0.20,
        base_clip_ratio=0.001,
        growth_rate=3.0,
    )
    assert torch.isfinite(result.policy_loss)
    result.policy_loss.backward()
    assert new_logp.grad is not None
    assert torch.isfinite(new_logp.grad).all()
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest tests/test_dppo_official_core.py -q`

Expected: collection fails because `src.dppo_official_core` does not exist.

- [ ] **Step 3: Implement only the upstream-derived formulas**

```python
@dataclass(frozen=True)
class OfficialDPPOLoss:
    policy_loss: torch.Tensor
    approximate_kl: torch.Tensor
    clip_fraction: torch.Tensor
    mean_ratio: torch.Tensor
    clip_ratios: torch.Tensor


def official_clip_schedule(...):
    indices = torch.arange(fine_tuned_steps, device=device, dtype=dtype)
    if fine_tuned_steps == 1:
        return torch.full_like(indices, maximum_clip_ratio)
    progress = indices / (fine_tuned_steps - 1)
    return base_clip_ratio + (maximum_clip_ratio - base_clip_ratio) * (
        torch.exp(growth_rate * progress) - 1.0
    ) / math.expm1(growth_rate)
```

Implement the clipped surrogate exactly from upstream `diffusion_ppo.py`, while adapting
its `(B*K,)` indexing to this project's equivalent `(B, K)` tensor layout. Add Chinese
comments at the denoising discount and step-dependent clipping calculations.

- [ ] **Step 4: Add license and source notice**

Copy the upstream MIT license verbatim. `NOTICE.md` must name the repository URL, the three
referenced upstream files and state that the code was adapted to a rail SFC environment.

- [ ] **Step 5: Run the core tests and commit**

Run: `python -m pytest tests/test_dppo_official_core.py -q`

Expected: all tests pass.

Commit: `feat: add official DPPO policy core`

### Task 2: Bind official hyperparameters to project configuration

**Files:**
- Modify: `src/dppo.py`
- Modify: `src/dppo_training_config.py`
- Modify: `configs/debug.yaml`
- Modify: `tests/test_dppo_official_core.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write failing configuration tests**

```python
def test_official_clip_settings_reach_dppo_config(debug_config):
    config = build_dppo_config(debug_config)
    assert config.clip_ratio_base == pytest.approx(0.001)
    assert config.clip_ratio_rate == pytest.approx(3.0)
```

Also assert that a base ratio above `clip_ratio`, or a non-positive rate, raises `ValueError`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m pytest tests/test_config.py tests/test_dppo_official_core.py -q`

Expected: failures report missing `clip_ratio_base` and `clip_ratio_rate` fields.

- [ ] **Step 3: Append and validate the two configuration fields**

```python
@dataclass(frozen=True)
class DPPOConfig:
    # Existing fields remain in their current order for compatibility.
    clip_ratio_base: float = 0.001
    clip_ratio_rate: float = 3.0
```

Require `0 < clip_ratio_base <= clip_ratio` and `clip_ratio_rate > 0`. Add both values under
`dppo.training` in `configs/debug.yaml` and map them in `build_dppo_config`.

- [ ] **Step 4: Run the focused tests and commit**

Run: `python -m pytest tests/test_config.py tests/test_dppo_official_core.py -q`

Expected: all focused tests pass.

Commit: `feat: configure official DPPO clipping schedule`

### Task 3: Use the official loss inside DPPOAgent

**Files:**
- Modify: `src/dppo.py`
- Modify: `tests/test_dppo_official_core.py`
- Modify: `tests/test_dppo.py`

- [ ] **Step 1: Write a failing agent-update test**

```python
def test_agent_update_uses_step_dependent_official_clipping(monkeypatch, tiny_agent, rollout):
    captured = {}

    def capture_loss(new_logp, old_logp, advantages, **settings):
        captured.update(settings)
        return real_official_loss(new_logp, old_logp, advantages, **settings)

    monkeypatch.setattr(dppo_module, "official_dppo_policy_loss", capture_loss)
    metrics = tiny_agent.update(rollout)
    assert captured["base_clip_ratio"] == pytest.approx(0.001)
    assert captured["growth_rate"] == pytest.approx(3.0)
    assert metrics["policy_optimizer_steps"] >= 1
```

- [ ] **Step 2: Run the agent tests and verify RED**

Run: `python -m pytest tests/test_dppo_official_core.py tests/test_dppo.py -q`

Expected: the capture is never called because the agent still uses the fixed-ratio surrogate.

- [ ] **Step 3: Replace only the policy-loss call**

In `DPPOAgent.update`, keep GAE, full-trajectory advantage normalization, stable KL early stop,
gradient clipping and value update unchanged. Replace `clipped_policy_surrogate(...)` with:

```python
official_loss = official_dppo_policy_loss(
    new_log_probabilities,
    batch_old_log_probabilities,
    batch_advantages,
    gamma_denoising=self.config.denoising_discount,
    maximum_clip_ratio=self.config.clip_ratio,
    base_clip_ratio=self.config.clip_ratio_base,
    growth_rate=self.config.clip_ratio_rate,
)
policy_loss = official_loss.policy_loss
```

Use `official_loss.clip_fraction` for the reported batch metric. Do not change environment,
projection or fast-layer code.

- [ ] **Step 4: Run the affected tests and commit**

Run: `python -m pytest tests/test_dppo_official_core.py tests/test_dppo.py -q`

Expected: all affected tests pass.

Commit: `feat: train rail policy with official DPPO loss`

### Task 4: Add one real rail smoke test and handoff documentation

**Files:**
- Modify: `tests/test_dppo_official_core.py`
- Modify: `README.md`

- [ ] **Step 1: Write the end-to-end smoke test**

The test must use the existing real debug scenario builders, reset
`DPPOSlowTimescaleEnvironment`, sample one joint action, execute `env.step`, append the
resulting `DPPORolloutTransition`, and call `agent.update`. Assertions are limited to:

```python
assert raw_action.shape == (env.action_space.dimension,)
assert np.isfinite(reward)
assert metrics["policy_optimizer_steps"] >= 1
assert np.isfinite(metrics["policy_loss"])
```

- [ ] **Step 2: Run the smoke test before any compatibility edit**

Run: `python -m pytest tests/test_dppo_official_core.py -q`

Expected: the new test either passes with the existing adapter or fails at the exact tensor
boundary that needs the minimal compatibility correction.

- [ ] **Step 3: Make only the compatibility correction proven necessary by RED**

If the test exposes a tensor-layout mismatch, add a small adapter that maps state to
`(B, 1, Do)` and action to `(B, 1, Da)` at the official-core boundary, then removes the
singleton action-chunk dimension before `env.step`. Do not modify the rail environment API.

- [ ] **Step 4: Document usage and run the final focused suite**

Add a README section that states the upstream source, `To=1`, `Ta=1`, dynamic `Do/Da`, and
the command used for the core smoke test.

Run:

```text
python -m pytest tests/test_dppo_official_core.py tests/test_dppo.py tests/test_config.py tests/test_dppo_training.py -q
```

Expected: all selected tests pass without warnings or collection errors.

- [ ] **Step 5: Commit and push**

Commit: `docs: explain official DPPO rail adaptation`

Push the current `codex/dppo-main-algorithm` branch to the configured GitHub remote and report
the commit SHA, focused test result and review-oriented file summary to the user.
