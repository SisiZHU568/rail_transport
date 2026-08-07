# DDQN Slow-Layer State, Action, and Feasible Execution Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the approved 78-dimensional DDQN observation, 12 structured slow-layer actions, history-only workload prediction, hybrid edge-cloud execution, simplified cost reward, and a shared audit-repair-execute closure.

**Architecture:** DDQN selects only `(replica_count, retention_policy, cloud_policy)` at each slow boundary. A shared fast-slot executor generates a placement, audits it, repairs it without changing the slow decision, executes only feasible requests, and returns costs and metrics to both the rule simulator and RL environment.

**Tech Stack:** Python 3.11, NumPy, PyTorch, dataclasses, PyYAML, Matplotlib, pytest, Git.

---

## Execution rules

- Work only in the existing isolated worktree on branch `codex/ddqn-state-action-redesign`.
- Follow TDD for every behavior change: failing test, observed failure, minimal implementation, passing test, commit.
- Add Chinese comments to new public types, formulas, safety branches, action-mask logic, and non-obvious normalization.
- Preserve `SINGLE` as a non-DDQN baseline; do not add it to the 12-action catalog.
- Do not run the formal 300+ Episode training in this plan. The final task runs only a five-Episode smoke training into a temporary output directory.
- Do not modify files under `sources/`.

## File responsibility map

### New files

- `src/workload_prediction.py`: history-only slow-window workload forecasting.
- `src/rl_state_encoder.py`: the stable 78-dimensional state schema and validation.
- `src/fast_slot_executor.py`: shared plan-audit-repair-execute-cost closure.
- `tests/test_workload_prediction.py`: forecast and leakage regression tests.
- `tests/test_rl_state_encoder.py`: state dimension, order, normalization, and reset tests.
- `tests/test_fast_slot_executor.py`: shared closure, rejection, cloud, and cost tests.

### Major modifications

- `src/rl_agent_action_space.py`: structured action catalog, codec, names, and action mask.
- `src/two_timescale_control.py`: retention/cloud enums in slow decisions and legacy mappings.
- `src/topology.py`: optional cloud compute node and unified node lookup.
- `src/network.py`: common transfer interface and hybrid MEC-cloud network.
- `src/failure_process.py`: cloud-aware operational-state lookup.
- `src/risk_aware_failure_process.py`: cloud-aware stochastic failure state.
- `src/reliability.py`: exact reliability over trackside and cloud nodes.
- `src/constraint_audit.py`: cloud-inclusive node capacity registry.
- `src/fast_optimizer.py`: cloud permission and cost-aware deterministic repair.
- `src/sfc_execution.py`: common network protocol and routing cost output.
- `src/rl_reward.py`: raw cost aggregation and one violation penalty.
- `src/slow_timescale_rl_env.py`: 12-action slow windows, state encoder, shared executor, masks, and metrics.
- `src/two_timescale_simulator.py`: shared executor integration.
- `src/ddqn.py`: masked exploration/targets and checkpoint schema metadata.
- `src/ddqn_evaluation.py`: structured action and repair/cloud metrics.
- `src/rl_scenario.py`: construct all new dependencies from configuration.
- `run_ddqn_training.py`: direct 12-action training and smoke-output override.
- `run_ddqn_evaluation.py`: new checkpoint schema and metrics output.
- `run_rl_environment_demo.py`: structured action demonstration.
- `configs/debug.yaml`: cloud fault domain, backhaul, costs, predictor, normalization, and max replicas.

## Common test command setup

Use this PowerShell prefix for every pytest command so Windows temporary files remain inside the writable project area:

```powershell
$testTemp='C:\Users\Administrator\.codex\.chatgpt-projects\g-p-6a6ff4453abc819184ee35997a150260\tmp\pytest-ddqn-redesign'
New-Item -ItemType Directory -Force -Path $testTemp | Out-Null
$env:TEMP=$testTemp
$env:TMP=$testTemp
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONPATH='.'
```

---

### Task 1: Replace the three-action adapter with the 12-action catalog

**Files:**
- Modify: `src/rl_agent_action_space.py`
- Modify: `tests/test_rl_agent_action_space.py`

- [ ] **Step 1: Write the failing catalog and round-trip tests**

Replace the old environment-action mapping tests with tests that make the approved Cartesian product explicit:

```python
import pytest

from src.rl_agent_action_space import (
    CloudPolicy,
    RetentionPolicy,
    decode_ddqn_action,
    encode_ddqn_action,
    get_ddqn_action_count,
)


@pytest.mark.parametrize(
    ("action_id", "replicas", "retention", "cloud"),
    [
        (0, 2, RetentionPolicy.ON_DEMAND, CloudPolicy.EDGE_ONLY),
        (5, 2, RetentionPolicy.ALL_WARM, CloudPolicy.CLOUD_ALLOWED),
        (6, 3, RetentionPolicy.ON_DEMAND, CloudPolicy.EDGE_ONLY),
        (11, 3, RetentionPolicy.ALL_WARM, CloudPolicy.CLOUD_ALLOWED),
    ],
)
def test_structured_action_decoding(
    action_id: int,
    replicas: int,
    retention: RetentionPolicy,
    cloud: CloudPolicy,
) -> None:
    action = decode_ddqn_action(action_id)
    assert action.replica_count == replicas
    assert action.retention_policy is retention
    assert action.cloud_policy is cloud
    assert encode_ddqn_action(
        replica_count=replicas,
        retention_policy=retention,
        cloud_policy=cloud,
    ) == action_id


def test_action_catalog_has_twelve_unique_entries() -> None:
    assert get_ddqn_action_count() == 12
    assert len({
        decode_ddqn_action(action_id)
        for action_id in range(12)
    }) == 12


def test_invalid_action_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="0～11"):
        decode_ddqn_action(12)
```

- [ ] **Step 2: Run the focused test and verify the expected failure**

Run:

```powershell
D:\Anaconda3\python.exe -m pytest -q -p no:cacheprovider -c NUL tests/test_rl_agent_action_space.py
```

Expected: FAIL because `RetentionPolicy`, `CloudPolicy`, `decode_ddqn_action`, and `encode_ddqn_action` do not exist.

- [ ] **Step 3: Implement the complete structured action codec**

Replace the old import of `SlowControlAction` and the three-action translation maps with:

```python
from dataclasses import dataclass
from enum import IntEnum


class RetentionPolicy(IntEnum):
    """慢层允许的容器保留等级。"""

    ON_DEMAND = 0
    PRIMARY_WARM = 1
    ALL_WARM = 2


class CloudPolicy(IntEnum):
    """慢层是否允许快层使用中心云。"""

    EDGE_ONLY = 0
    CLOUD_ALLOWED = 1


@dataclass(frozen=True)
class StructuredSlowAction:
    """一个可解释的 DDQN 慢层组合动作。"""

    action_id: int
    replica_count: int
    retention_policy: RetentionPolicy
    cloud_policy: CloudPolicy


def get_ddqn_action_count() -> int:
    return 12


def encode_ddqn_action(
    replica_count: int,
    retention_policy: RetentionPolicy,
    cloud_policy: CloudPolicy,
) -> int:
    if replica_count not in (2, 3):
        raise ValueError("DDQN副本数只能是2或3。")
    retention = RetentionPolicy(retention_policy)
    cloud = CloudPolicy(cloud_policy)
    return (
        (replica_count - 2) * 3 + int(retention)
    ) * 2 + int(cloud)


def decode_ddqn_action(
    action: int,
) -> StructuredSlowAction:
    action_id = int(action)
    if not 0 <= action_id < get_ddqn_action_count():
        raise ValueError("非法DDQN动作，合法范围为0～11。")
    replica_count = 2 + action_id // 6
    remainder = action_id % 6
    retention = RetentionPolicy(remainder // 2)
    cloud = CloudPolicy(remainder % 2)
    return StructuredSlowAction(
        action_id=action_id,
        replica_count=replica_count,
        retention_policy=retention,
        cloud_policy=cloud,
    )


DDQN_ACTION_NAMES = tuple(
    (
        f"R{action.replica_count}_"
        f"{action.retention_policy.name}_"
        f"{action.cloud_policy.name}"
    )
    for action in (
        decode_ddqn_action(index)
        for index in range(get_ddqn_action_count())
    )
)
```

- [ ] **Step 4: Run the focused tests**

Run the command from Step 2.

Expected: `tests/test_rl_agent_action_space.py` passes.

- [ ] **Step 5: Commit the action catalog**

```powershell
git add src/rl_agent_action_space.py tests/test_rl_agent_action_space.py
git commit -m "feat: define structured DDQN actions"
```

---

### Task 2: Add the history-only workload predictor

**Files:**
- Create: `src/workload_prediction.py`
- Create: `tests/test_workload_prediction.py`

- [ ] **Step 1: Write failing prediction and leakage tests**

```python
import pytest

from src.workload_prediction import HistoricalWorkloadPredictor


def test_forecast_uses_only_recent_history() -> None:
    predictor = HistoricalWorkloadPredictor(
        lookback_slots=3,
        baseline_request_rate=1.0,
    )
    result = predictor.predict([9, 1, 2, 3])
    assert result.mean_requests == pytest.approx(2.0)
    assert result.peak_requests == 3.0
    assert result.normalized_trend == pytest.approx(2.0 / 3.0)


def test_future_suffix_cannot_change_current_forecast() -> None:
    predictor = HistoricalWorkloadPredictor(3, 1.0)
    history = [0, 1, 3]
    first = predictor.predict(history)
    second = predictor.predict(history)
    assert first == second


def test_empty_history_uses_configured_baseline() -> None:
    result = HistoricalWorkloadPredictor(4, 2.0).predict([])
    assert result.mean_requests == 2.0
    assert result.peak_requests == 2.0
    assert result.normalized_trend == 0.0
```

- [ ] **Step 2: Run the new tests and observe the import failure**

```powershell
D:\Anaconda3\python.exe -m pytest -q -p no:cacheprovider -c NUL tests/test_workload_prediction.py
```

Expected: FAIL because `src.workload_prediction` does not exist.

- [ ] **Step 3: Implement the predictor without reading a workload object**

```python
from dataclasses import dataclass
import math
from collections.abc import Sequence


@dataclass(frozen=True)
class WorkloadForecast:
    mean_requests: float
    peak_requests: float
    normalized_trend: float


class HistoricalWorkloadPredictor:
    """只接收已经观测到的请求前缀，禁止访问未来轨迹。"""

    def __init__(
        self,
        lookback_slots: int,
        baseline_request_rate: float,
    ) -> None:
        if lookback_slots <= 0:
            raise ValueError("回看时隙数必须大于0。")
        if not math.isfinite(baseline_request_rate) or baseline_request_rate < 0:
            raise ValueError("基准请求率必须是非负有限值。")
        self.lookback_slots = lookback_slots
        self.baseline_request_rate = float(baseline_request_rate)

    def predict(
        self,
        observed_request_counts: Sequence[int],
    ) -> WorkloadForecast:
        if any(count < 0 for count in observed_request_counts):
            raise ValueError("历史请求数不能小于0。")
        if not observed_request_counts:
            return WorkloadForecast(
                mean_requests=self.baseline_request_rate,
                peak_requests=self.baseline_request_rate,
                normalized_trend=0.0,
            )
        window = list(observed_request_counts[-self.lookback_slots:])
        mean_requests = sum(window) / len(window)
        peak_requests = float(max(window))
        if len(window) == 1:
            trend = 0.0
        else:
            denominator = max(peak_requests, 1.0)
            trend = (window[-1] - window[0]) / denominator
        return WorkloadForecast(
            mean_requests=float(mean_requests),
            peak_requests=peak_requests,
            normalized_trend=float(max(-1.0, min(trend, 1.0))),
        )
```

- [ ] **Step 4: Run predictor tests**

Expected: all new predictor tests pass.

- [ ] **Step 5: Commit the predictor**

```powershell
git add src/workload_prediction.py tests/test_workload_prediction.py
git commit -m "feat: add history-only workload prediction"
```

---

### Task 3: Add the stable 78-dimensional state encoder

**Files:**
- Create: `src/rl_state_encoder.py`
- Create: `tests/test_rl_state_encoder.py`

- [ ] **Step 1: Write failing state schema tests**

Build five trackside node observations, one cloud observation, and three VNF profiles. Assert:

```python
def test_current_schema_has_seventy_eight_features() -> None:
    encoder, snapshot = build_encoder_and_snapshot()
    state = encoder.encode(snapshot)
    assert encoder.state_dim == 78
    assert len(encoder.feature_names) == 78
    assert state.shape == (78,)
    assert np.isfinite(state).all()
    assert ((0.0 <= state) & (state <= 1.0)).all()


def test_node_ids_are_one_hot_not_scalar_ordinals() -> None:
    encoder, snapshot = build_encoder_and_snapshot()
    state = encoder.encode(snapshot)
    current_names = encoder.feature_names[:5]
    assert current_names == tuple(
        f"current_mec_{node_id}" for node_id in range(5)
    )
    assert state[:5].sum() == 1.0


def test_reset_history_is_all_zero() -> None:
    encoder, snapshot = build_encoder_and_snapshot(
        current_action=None,
        has_history=False,
    )
    state = encoder.encode(snapshot)
    history_start = 67
    assert np.all(state[history_start:] == 0.0)
```

- [ ] **Step 2: Run the new state tests and observe the import failure**

```powershell
D:\Anaconda3\python.exe -m pytest -q -p no:cacheprovider -c NUL tests/test_rl_state_encoder.py
```

Expected: FAIL because `src.rl_state_encoder` does not exist.

- [ ] **Step 3: Implement explicit snapshot types and feature order**

Create these public input records so the encoder never reads simulator internals:

```python
@dataclass(frozen=True)
class NodeStateObservation:
    node_id: int
    free_cpu_ratio: float
    free_memory_ratio: float
    base_availability: float
    predicted_failure_probability: float
    operational: bool
    normalized_delay_from_serving: float


@dataclass(frozen=True)
class RLStateSnapshot:
    serving_mec: int
    next_mec: int
    route_progress: float
    normalized_remaining_dwell: float
    normalized_mean_requests: float
    normalized_peak_requests: float
    normalized_load_trend: float
    global_failure_risk: float
    node_observations: tuple[NodeStateObservation, ...]
    current_action: StructuredSlowAction | None
    hot_replica_ratio: float
    cloud_replica_ratio: float
    has_history: bool
    previous_success_rate: float
    previous_sla_violation_rate: float
    previous_normalized_total_cost: float
    previous_repair_failure_rate: float
```

Implement `RLStateEncoder` with constructor inputs `trackside_node_ids`, `cloud_node_id`, `functions`, `sfc`, and the normalization denominators. Generate feature names in exactly this order:

```python
feature_names = (
    current_mec_one_hot                 # N
    + next_mec_one_hot                  # N
    + mobility_and_demand_names         # 6
    + six_names_per_node                # 6(N+1)
    + four_names_per_function           # 4M
    + ("sfc_reliability_target",
       "sfc_normalized_deadline",
       "sfc_normalized_input_size")     # 3
    + history_names                     # 11
)
```

Encode load trend from `[-1,1]` to `[0,1]` with `(trend + 1) / 2`. When `has_history` is false, append eleven zeros instead of a fabricated previous action. Reject missing/duplicate node observations, wrong node order, non-finite values, or a final shape different from `state_dim`.

- [ ] **Step 4: Run state tests**

Expected: all state tests pass with `state_dim == 78` for `N=5`, `M=3`.

- [ ] **Step 5: Commit the state encoder**

```powershell
git add src/rl_state_encoder.py tests/test_rl_state_encoder.py
git commit -m "feat: add structured RL state encoder"
```

---

### Task 4: Introduce the simplified cost reward without breaking the old environment yet

**Files:**
- Modify: `src/rl_reward.py`
- Modify: `tests/test_rl_reward.py`

- [ ] **Step 1: Add failing tests for the approved formula**

```python
from src.rl_reward import RLWindowCostMetrics, calculate_cost_reward


def test_reward_is_normalized_cost_plus_violation() -> None:
    result = calculate_cost_reward(
        metrics=RLWindowCostMetrics(
            run_cost=20.0,
            route_cost=10.0,
            cold_start_cost=5.0,
            has_violation=True,
        ),
        maximum_window_cost=100.0,
    )
    assert result.total_cost == 35.0
    assert result.normalized_total_cost == pytest.approx(0.35)
    assert result.violation_penalty == 1.0
    assert result.reward == pytest.approx(-1.35)


def test_no_request_warm_cost_is_still_charged() -> None:
    result = calculate_cost_reward(
        RLWindowCostMetrics(12.0, 0.0, 0.0, False),
        maximum_window_cost=100.0,
    )
    assert result.reward == pytest.approx(-0.12)


def test_total_cost_is_clipped_before_penalty() -> None:
    result = calculate_cost_reward(
        RLWindowCostMetrics(200.0, 0.0, 0.0, True),
        maximum_window_cost=100.0,
    )
    assert result.reward == -2.0
```

- [ ] **Step 2: Run the reward tests and verify missing symbols**

```powershell
D:\Anaconda3\python.exe -m pytest -q -p no:cacheprovider -c NUL tests/test_rl_reward.py
```

Expected: FAIL because the new cost records and function do not exist.

- [ ] **Step 3: Add the new reward API alongside the legacy API**

```python
@dataclass(frozen=True)
class RLWindowCostMetrics:
    run_cost: float
    route_cost: float
    cold_start_cost: float
    has_violation: bool

    def __post_init__(self) -> None:
        values = (self.run_cost, self.route_cost, self.cold_start_cost)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("窗口成本必须是非负有限值。")


@dataclass(frozen=True)
class RLCostRewardBreakdown:
    run_cost: float
    route_cost: float
    cold_start_cost: float
    total_cost: float
    normalized_total_cost: float
    violation_penalty: float
    reward: float


def calculate_cost_reward(
    metrics: RLWindowCostMetrics,
    maximum_window_cost: float,
) -> RLCostRewardBreakdown:
    if not math.isfinite(maximum_window_cost) or maximum_window_cost <= 0:
        raise ValueError("窗口最大成本必须是正有限值。")
    total_cost = metrics.run_cost + metrics.route_cost + metrics.cold_start_cost
    normalized = min(total_cost / maximum_window_cost, 1.0)
    penalty = 1.0 if metrics.has_violation else 0.0
    return RLCostRewardBreakdown(
        run_cost=metrics.run_cost,
        route_cost=metrics.route_cost,
        cold_start_cost=metrics.cold_start_cost,
        total_cost=total_cost,
        normalized_total_cost=normalized,
        violation_penalty=penalty,
        reward=-(normalized + penalty),
    )
```

Keep the legacy weighted function temporarily so the old RL environment remains green until Task 13 switches atomically.

- [ ] **Step 4: Run reward tests and the current RL environment tests**

```powershell
D:\Anaconda3\python.exe -m pytest -q -p no:cacheprovider -c NUL tests/test_rl_reward.py tests/test_slow_timescale_rl_env.py
```

Expected: both files pass.

- [ ] **Step 5: Commit the reward API**

```powershell
git add src/rl_reward.py tests/test_rl_reward.py
git commit -m "feat: add simplified DDQN cost reward"
```

---

### Task 5: Add the center-cloud node to the compute topology

**Files:**
- Modify: `src/topology.py`
- Modify: `src/reliability.py`
- Modify: `src/constraint_audit.py`
- Modify: `configs/debug.yaml`
- Modify: `tests/test_topology.py`
- Modify: `tests/test_reliability.py`
- Modify: `tests/test_constraint_audit.py`

- [ ] **Step 1: Write failing cloud registry and audit tests**

Add assertions that the debug topology still has five trackside sites but now has six compute nodes:

```python
def test_cloud_is_compute_node_not_trackside_site() -> None:
    config = load_config("configs/debug.yaml")
    topology = build_linear_topology(config)
    assert topology.mec_count == 5
    assert len(topology.compute_nodes) == 6
    assert topology.cloud_node is not None
    assert topology.cloud_node.node_type is NodeType.CLOUD
    assert topology.get_node(topology.cloud_node.node_id) is topology.cloud_node
    with pytest.raises(KeyError):
        topology.get_site(topology.cloud_node.node_id)
```

Add a constraint-audit case whose candidate map contains the cloud node and assert it is not reported as unknown.

- [ ] **Step 2: Run the three focused files and observe failure**

```powershell
D:\Anaconda3\python.exe -m pytest -q -p no:cacheprovider -c NUL tests/test_topology.py tests/test_reliability.py tests/test_constraint_audit.py
```

Expected: FAIL because `cloud_node`, `compute_nodes`, and `get_node` do not exist.

- [ ] **Step 3: Extend `LinearRailTopology` without mixing cloud into mobility sites**

Use this interface:

```python
class LinearRailTopology:
    def __init__(
        self,
        sites: list[TracksideSite],
        cloud_node: EdgeNode | None = None,
    ) -> None:
        if not sites:
            raise ValueError("铁路拓扑至少需要一个轨旁MEC。")
        self.sites = sorted(sites, key=lambda site: site.position_m)
        trackside_ids = [site.node.node_id for site in self.sites]
        positions = [site.position_m for site in self.sites]
        if len(trackside_ids) != len(set(trackside_ids)):
            raise ValueError("轨旁MEC的node_id不能重复。")
        if len(positions) != len(set(positions)):
            raise ValueError("两个轨旁MEC不能位于相同位置。")
        if cloud_node is not None and cloud_node.node_type is not NodeType.CLOUD:
            raise ValueError("cloud_node必须是中心云节点。")
        trackside_ids = {site.node.node_id for site in self.sites}
        if cloud_node is not None and cloud_node.node_id in trackside_ids:
            raise ValueError("中心云与轨旁MEC的节点编号不能重复。")
        self.cloud_node = cloud_node

    @property
    def compute_nodes(self) -> tuple[EdgeNode, ...]:
        nodes = [site.node for site in self.sites]
        if self.cloud_node is not None:
            nodes.append(self.cloud_node)
        return tuple(nodes)

    def get_node(self, node_id: int) -> EdgeNode:
        for node in self.compute_nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(f"没有找到 node_id={node_id} 的计算节点。")
```

In `build_linear_topology`, create the cloud only when `topology.include_cloud` is true. Use node ID `mec_count` and `node_resources.cloud_fault_domain_id`.

Update reliability lookup and auditor node maps from `topology.get_site(node_id).node` / `topology.sites` to `topology.get_node(node_id)` / `topology.compute_nodes` where compute candidates are intended. Keep mobility and physical track order on `topology.sites`.

Update `configs/debug.yaml`:

```yaml
serverless:
  max_replicas: 3

node_resources:
  cloud_fault_domain_id: 3

reliability:
  fault_domains:
    - domain_id: 0
      availability: 0.995
    - domain_id: 1
      availability: 0.995
    - domain_id: 2
      availability: 0.995
    - domain_id: 3
      availability: 0.999
```

- [ ] **Step 4: Run the focused topology/reliability/audit tests**

Expected: all pass, and existing route endpoints remain unchanged.

- [ ] **Step 5: Commit the cloud topology**

```powershell
git add src/topology.py src/reliability.py src/constraint_audit.py configs/debug.yaml tests/test_topology.py tests/test_reliability.py tests/test_constraint_audit.py
git commit -m "feat: register center cloud compute node"
```

---

### Task 6: Include the cloud in infrastructure failure state

**Files:**
- Modify: `src/failure_process.py`
- Modify: `src/risk_aware_failure_process.py`
- Modify: `tests/test_failure_process.py`
- Modify: `tests/test_risk_aware_failure_process.py`

- [ ] **Step 1: Write failing cloud operational-state tests**

```python
def test_failure_state_contains_every_compute_node() -> None:
    config = load_config("configs/debug.yaml")
    topology = build_linear_topology(config)
    process = build_windowed_markov_failure_process(
        config=config,
        topology=topology,
        random_seed=123,
    )
    process.reset()
    state = process.state_for_slot(0)
    assert set(state.node_local_up) == {
        node.node_id for node in topology.compute_nodes
    }
    assert topology.cloud_node is not None
    assert isinstance(
        state.is_node_operational(
            topology.cloud_node.node_id,
            topology,
        ),
        bool,
    )
```

- [ ] **Step 2: Run failure-process tests and observe the missing cloud**

```powershell
D:\Anaconda3\python.exe -m pytest -q -p no:cacheprovider -c NUL tests/test_failure_process.py tests/test_risk_aware_failure_process.py
```

Expected: FAIL because builders iterate only over `topology.sites` and `is_node_operational` calls `get_site`.

- [ ] **Step 3: Replace compute-state iteration with `topology.compute_nodes`**

Make these exact semantic changes:

- `InfrastructureState.is_node_operational` obtains `fault_domain` from `topology.get_node(node_id)`.
- Markov and windowed builders create node failure/recovery dictionaries for every `topology.compute_nodes` entry.
- Domain dictionaries include every domain referenced by compute nodes.
- Scripted trackside failure configuration remains optional for the cloud; when no cloud-specific failure is scripted, its stochastic probability comes from cloud reliability.

- [ ] **Step 4: Run failure tests and reliability tests**

```powershell
D:\Anaconda3\python.exe -m pytest -q -p no:cacheprovider -c NUL tests/test_failure_process.py tests/test_risk_aware_failure_process.py tests/test_reliability.py
```

Expected: all pass.

- [ ] **Step 5: Commit cloud failure-state support**

```powershell
git add src/failure_process.py src/risk_aware_failure_process.py tests/test_failure_process.py tests/test_risk_aware_failure_process.py
git commit -m "feat: model center cloud failures"
```

---

### Task 7: Add a hybrid edge-cloud transfer interface and routing cost

**Files:**
- Modify: `src/network.py`
- Modify: `src/sfc_execution.py`
- Modify: `src/adaptive_runtime_reliability.py`
- Modify: `src/runtime_reliability.py`
- Modify: `src/simulator.py`
- Modify: `src/two_timescale_simulator.py`
- Modify: `tests/test_network.py`
- Modify: `tests/test_sfc_execution.py`

- [ ] **Step 1: Write failing cloud delay and cost tests**

```python
def test_edge_to_cloud_uses_backhaul_parameters() -> None:
    network = build_test_hybrid_network()
    delay = network.transfer_delay_ms(
        data_size_mb=2.0,
        source_node_id=0,
        destination_node_id=5,
    )
    cost = network.transfer_cost(
        data_size_mb=2.0,
        source_node_id=0,
        destination_node_id=5,
    )
    assert delay == pytest.approx(72.0)  # 2*8/500*1000 + 40
    assert cost == pytest.approx(0.4)    # 2 MB * 0.2


def test_sfc_execution_reports_routing_cost() -> None:
    result = execute_test_sfc_on_nodes([0, 5])
    assert result.total_routing_cost > 0.0
```

- [ ] **Step 2: Run network and SFC execution tests**

```powershell
D:\Anaconda3\python.exe -m pytest -q -p no:cacheprovider -c NUL tests/test_network.py tests/test_sfc_execution.py
```

Expected: FAIL because there is no hybrid network or routing-cost API.

- [ ] **Step 3: Implement a common protocol and hybrid network**

Add:

```python
class TransferNetworkProtocol(Protocol):
    def transfer_delay_ms(
        self,
        data_size_mb: float,
        source_node_id: int,
        destination_node_id: int,
    ) -> float:
        raise NotImplementedError

    def transfer_cost(
        self,
        data_size_mb: float,
        source_node_id: int,
        destination_node_id: int,
    ) -> float:
        raise NotImplementedError
```

Extend `LinearMECNetwork` with `edge_data_cost_per_mb_hop` and `transfer_cost = data_size_mb * hop_count * price`.

Add `HybridRailNetwork` that delegates MEC-MEC traffic to `LinearMECNetwork` and handles any MEC-cloud leg with:

```python
delay_ms = (
    data_size_mb * 8.0 / cloud_backhaul_bandwidth_mbps * 1000.0
    + cloud_one_way_propagation_delay_ms
)
cost = data_size_mb * cloud_data_cost_per_mb
```

Same-node transfers have zero delay and cost. Cloud-cloud transfers also have zero delay and cost. Unknown node IDs raise `KeyError`.

Add these configuration values:

```yaml
network:
  edge_data_cost_per_mb_hop: 0.01
  cloud_backhaul_bandwidth_mbps: 500.0
  cloud_one_way_propagation_delay_ms: 40.0
  cloud_data_cost_per_mb: 0.20
```

Change network type hints in the listed simulators/executors to `TransferNetworkProtocol`. Extend `SFCExecutionResult` with `total_routing_cost` and call `network.transfer_cost` for every input, inter-function, and return leg in the same places where delay is accumulated.

- [ ] **Step 4: Run network, SFC, simulator, and reliability regression tests**

```powershell
D:\Anaconda3\python.exe -m pytest -q -p no:cacheprovider -c NUL tests/test_network.py tests/test_sfc_execution.py tests/test_simulator.py tests/test_runtime_reliability.py tests/test_adaptive_runtime_reliability.py
```

Expected: all pass.

- [ ] **Step 5: Commit the hybrid network**

```powershell
git add src/network.py src/sfc_execution.py src/adaptive_runtime_reliability.py src/runtime_reliability.py src/simulator.py src/two_timescale_simulator.py configs/debug.yaml tests/test_network.py tests/test_sfc_execution.py
git commit -m "feat: add hybrid edge cloud routing"
```

---

### Task 8: Migrate slow decisions to retention and cloud policies

**Files:**
- Modify: `src/two_timescale_control.py`
- Modify: `src/redundancy_placement.py`
- Modify: `tests/test_two_timescale_control.py`
- Modify: `tests/test_redundancy_placement.py`
- Modify: `tests/test_fast_optimizer.py`

- [ ] **Step 1: Write failing slow-decision and legacy-mapping tests**

```python
def test_slow_decision_exposes_structured_policy() -> None:
    decision = SlowTimescaleDecision(
        decision_slot=0,
        valid_until_slot=9,
        replica_count=3,
        retention_policy=RetentionPolicy.PRIMARY_WARM,
        cloud_policy=CloudPolicy.CLOUD_ALLOWED,
        reason="测试",
    )
    assert decision.use_redundancy is True
    assert decision.replica_count == 3


def test_legacy_modes_have_explicit_mapping() -> None:
    assert legacy_mode_to_policy(StandbyMode.COLD) == (
        2,
        RetentionPolicy.PRIMARY_WARM,
        CloudPolicy.EDGE_ONLY,
    )
    assert legacy_mode_to_policy(StandbyMode.DYNAMIC) == (
        2,
        RetentionPolicy.PRIMARY_WARM,
        CloudPolicy.EDGE_ONLY,
    )
```

Also add a planner-builder test that `build_reliability_aware_replica_planner(config, replica_count=3)` returns plans with exactly three distinct nodes per VNF.

- [ ] **Step 2: Run control/planner tests and observe constructor failures**

```powershell
D:\Anaconda3\python.exe -m pytest -q -p no:cacheprovider -c NUL tests/test_two_timescale_control.py tests/test_redundancy_placement.py tests/test_fast_optimizer.py
```

Expected: FAIL because the new policy fields and replica-count override do not exist.

- [ ] **Step 3: Implement the new slow-decision contract**

`SlowTimescaleDecision` becomes:

```python
@dataclass(frozen=True)
class SlowTimescaleDecision:
    decision_slot: int
    valid_until_slot: int
    replica_count: int
    retention_policy: RetentionPolicy
    cloud_policy: CloudPolicy
    reason: str

    @property
    def use_redundancy(self) -> bool:
        return self.replica_count > 1
```

Keep `StandbyMode` only for fixed legacy inputs and add `legacy_mode_to_policy`. Map `DYNAMIC` to `PRIMARY_WARM`; handover activation remains a fast-layer behavior. Update rule/fixed controllers and every test constructor in the same commit.

Change the planner builder signature to:

```python
def build_reliability_aware_replica_planner(
    config: dict[str, Any],
    replica_count: int | None = None,
) -> ReliabilityAwareReplicaPlanner:
    selected_count = (
        int(config["reliability"]["replica_count"])
        if replica_count is None
        else int(replica_count)
    )
    return ReliabilityAwareReplicaPlanner(
        replica_count=selected_count,
        minimum_distinct_fault_domains=int(
            config["reliability"]["minimum_distinct_fault_domains"]
        ),
    )
```

- [ ] **Step 4: Run control, planner, optimizer, and simulator tests**

```powershell
D:\Anaconda3\python.exe -m pytest -q -p no:cacheprovider -c NUL tests/test_two_timescale_control.py tests/test_redundancy_placement.py tests/test_fast_optimizer.py tests/test_two_timescale_simulator.py
```

Expected: all pass after updating callers to the new fields.

- [ ] **Step 5: Commit slow policy migration**

```powershell
git add src/two_timescale_control.py src/redundancy_placement.py tests/test_two_timescale_control.py tests/test_redundancy_placement.py tests/test_fast_optimizer.py tests/test_two_timescale_simulator.py
git commit -m "refactor: structure slow timescale decisions"
```

---

### Task 9: Add legal-action masks and cloud-aware cost repair

**Files:**
- Modify: `src/rl_agent_action_space.py`
- Modify: `src/two_timescale_control.py`
- Modify: `src/fast_optimizer.py`
- Modify: `tests/test_rl_agent_action_space.py`
- Modify: `tests/test_two_timescale_control.py`
- Modify: `tests/test_fast_optimizer.py`

- [ ] **Step 1: Write failing mask and cloud-permission tests**

Add an immutable context:

```python
context = ActionFeasibilityContext(
    operational_edge_node_ids=frozenset({0, 1}),
    operational_cloud_node_id=5,
    node_free_memory_mb={0: 4000.0, 1: 4000.0, 5: 64000.0},
    node_fault_domains={0: 0, 1: 0, 5: 3},
    total_function_memory_mb=1536.0,
    minimum_distinct_fault_domains=2,
)
mask = build_valid_action_mask(context)
assert mask.shape == (12,)
assert mask.dtype == np.bool_
assert not bool(mask[0])           # edge-only R2 cannot span two domains
assert bool(mask[1])               # cloud-allowed R2 can use domain 3
```

Add optimizer tests:

- `EDGE_ONLY` never returns the cloud node.
- `CLOUD_ALLOWED` may return the cloud when all edge repairs are infeasible.
- The repaired candidate map still has exactly the requested 2 or 3 nodes per VNF.

- [ ] **Step 2: Run action and optimizer tests and observe failures**

```powershell
D:\Anaconda3\python.exe -m pytest -q -p no:cacheprovider -c NUL tests/test_rl_agent_action_space.py tests/test_fast_optimizer.py
```

Expected: FAIL because mask context and cloud filtering are not implemented.

- [ ] **Step 3: Implement necessary-condition masking**

Add:

```python
@dataclass(frozen=True)
class ActionFeasibilityContext:
    operational_edge_node_ids: frozenset[int]
    operational_cloud_node_id: int | None
    node_free_memory_mb: dict[int, float]
    node_fault_domains: dict[int, int]
    total_function_memory_mb: float
    minimum_distinct_fault_domains: int


def build_valid_action_mask(
    context: ActionFeasibilityContext,
) -> np.ndarray:
    mask = np.zeros(get_ddqn_action_count(), dtype=np.bool_)
    for action_id in range(get_ddqn_action_count()):
        action = decode_ddqn_action(action_id)
        candidates = set(context.operational_edge_node_ids)
        if (
            action.cloud_policy is CloudPolicy.CLOUD_ALLOWED
            and context.operational_cloud_node_id is not None
        ):
            candidates.add(context.operational_cloud_node_id)
        enough_nodes = len(candidates) >= action.replica_count
        enough_domains = len({
            context.node_fault_domains[node_id]
            for node_id in candidates
        }) >= context.minimum_distinct_fault_domains
        enough_hot_memory = True
        if action.retention_policy is RetentionPolicy.ALL_WARM:
            required = (
                action.replica_count
                * context.total_function_memory_mb
            )
            available = sum(
                context.node_free_memory_mb.get(node_id, 0.0)
                for node_id in candidates
            )
            enough_hot_memory = available >= required
        mask[action_id] = (
            enough_nodes and enough_domains and enough_hot_memory
        )
    return mask
```

In `build_fast_decision_for_plan`, apply retention semantics exactly:

- `ON_DEMAND`: no proactive hot nodes;
- `PRIMARY_WARM`: primary hot, plus the first backup during handover activation;
- `ALL_WARM`: every planned replica hot;
- any selected execution node not in the hot set is a cold start, including the primary under `ON_DEMAND`.

Modify `FastFeasibilityOptimizer` to receive `TransferNetworkProtocol` and cost rates. Candidate nodes are trackside only for `EDGE_ONLY`; append `topology.cloud_node` for `CLOUD_ALLOWED`. Replace geographic-position scoring with deterministic estimated total cost followed by changed-function count and flattened node IDs. The estimate must include node-specific run price, `network.transfer_cost`, and cold-start price.

- [ ] **Step 4: Run action/control/optimizer tests**

Expected: masks and cloud restrictions pass; all repaired plans retain the exact slow replica count.

- [ ] **Step 5: Commit masks and cloud-aware repair**

```powershell
git add src/rl_agent_action_space.py src/two_timescale_control.py src/fast_optimizer.py tests/test_rl_agent_action_space.py tests/test_two_timescale_control.py tests/test_fast_optimizer.py
git commit -m "feat: mask actions and repair with cloud policy"
```

---

### Task 10: Extract the shared fast-slot executor

**Files:**
- Create: `src/fast_slot_executor.py`
- Create: `tests/test_fast_slot_executor.py`
- Modify: `src/rl_reward.py`

- [ ] **Step 1: Write failing shared-closure tests**

Create a small two-function fixture and assert:

```python
def test_infeasible_final_audit_never_executes_sfc(monkeypatch) -> None:
    calls = 0

    def forbidden_execute(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("不可行方案不应进入SFC执行器")

    monkeypatch.setattr(
        "src.fast_slot_executor.execute_sfc_batch",
        forbidden_execute,
    )
    result = build_executor(unrepairable=True).execute(build_slot_input())
    assert calls == 0
    assert result.request_success is False
    assert result.constraint_rejected is True


def test_shared_executor_returns_three_cost_components() -> None:
    result = build_executor(unrepairable=False).execute(build_slot_input())
    assert result.final_audit.all_constraints_met is True
    assert result.run_cost >= 0.0
    assert result.route_cost >= 0.0
    assert result.cold_start_cost >= 0.0
    assert result.total_cost == pytest.approx(
        result.run_cost + result.route_cost + result.cold_start_cost
    )
```

- [ ] **Step 2: Run the new executor tests and observe import failure**

```powershell
D:\Anaconda3\python.exe -m pytest -q -p no:cacheprovider -c NUL tests/test_fast_slot_executor.py
```

Expected: FAIL because the shared executor does not exist.

- [ ] **Step 3: Implement one input record, one result record, and one executor**

Create these public contracts:

```python
@dataclass(frozen=True)
class RuntimeCostRates:
    edge_cpu_cost_per_unit: float
    edge_memory_cost_per_mb_second: float
    cloud_cpu_cost_per_unit: float
    cloud_memory_cost_per_mb_second: float
    cold_start_cost_per_ms: float


@dataclass(frozen=True)
class FastSlotInput:
    train_state: TrainState
    request_count: int
    infrastructure_state: InfrastructureState
    slow_decision: SlowTimescaleDecision
    previous_candidate_map: dict[int, tuple[int, ...]] | None


@dataclass(frozen=True)
class FastSlotExecutionResult:
    function_replica_node_ids: dict[int, tuple[int, ...]]
    function_hot_node_ids: dict[int, tuple[int, ...]]
    selected_execution_node_ids: tuple[int, ...]
    initial_audit: SlotConstraintAudit
    final_audit: SlotConstraintAudit
    fast_repair_attempted: bool
    fast_repair_succeeded: bool | None
    fast_repair_reason: str
    request_success: bool | None
    deadline_met: bool | None
    end_to_end_delay_ms: float | None
    cold_start_delay_ms: float
    active_memory_mb: float
    plan_change_count: int
    used_cloud: bool
    constraint_rejected: bool
    run_cost: float
    route_cost: float
    cold_start_cost: float

    @property
    def total_cost(self) -> float:
        return self.run_cost + self.route_cost + self.cold_start_cost
```

`FastSlotExecutor.execute` performs, in order: planner lookup by replica count, initial fast decision, initial audit, optimizer, final audit check, optional `execute_sfc_batch`, delay/SLA calculation, active-pair memory calculation, and the three raw costs. Use the final audit's CPU/memory demands and node type to apply edge/cloud run rates. Return a normal rejected result rather than raising when no repair exists.

Give the executor one explicit constructor so both callers assemble exactly the same closure:

```python
class FastSlotExecutor:
    def __init__(
        self,
        topology: LinearRailTopology,
        network: TransferNetworkProtocol,
        functions: list[ServerlessFunction],
        sfc: SFCType,
        replica_planners: dict[int, ReplicaPlannerProtocol],
        constraint_auditor: SlotConstraintAuditor,
        fast_optimizer: FastFeasibilityOptimizer,
        input_size_mb_per_request: float,
        slot_seconds: float,
        handover_hot_window_s: float,
        failover_delay_ms_per_function: float,
        cost_rates: RuntimeCostRates,
        return_result_to_source: bool = True,
    ) -> None:
        """保存快时隙闭环所需的唯一一组依赖。"""
```

Validate that planner keys are exactly `{1, 2, 3}`. `execute()` selects `replica_planners[slow_decision.replica_count]`, builds the replica plan and `FastTimescaleState`, and then calls `build_fast_decision_for_plan`; callers must not pre-build or alter the fast decision. In `rl_scenario.py`, construct the mapping as `{1: SingleReplicaPlanner(), 2: two_replica_planner, 3: three_replica_planner}`, where the two reliability-aware planners share the same placement rules but receive different fixed replica counts.

Import the single shared `ReplicaPlannerProtocol` and `SingleReplicaPlanner` from `src.runtime_reliability`; do not create another local protocol in the executor. When Task 11 removes the simulator-local protocol, update its type import to the same shared definition.

- [ ] **Step 4: Run executor, auditor, optimizer, and SFC tests**

```powershell
D:\Anaconda3\python.exe -m pytest -q -p no:cacheprovider -c NUL tests/test_fast_slot_executor.py tests/test_constraint_audit.py tests/test_fast_optimizer.py tests/test_sfc_execution.py
```

Expected: all pass.

- [ ] **Step 5: Commit the shared executor**

```powershell
git add src/fast_slot_executor.py src/rl_reward.py tests/test_fast_slot_executor.py
git commit -m "feat: add shared fast slot executor"
```

---

### Task 11: Make the rule simulator use the shared executor

**Files:**
- Modify: `src/two_timescale_simulator.py`
- Modify: `src/two_timescale_monte_carlo.py`
- Modify: `run_two_timescale_comparison.py`
- Modify: `tests/test_two_timescale_simulator.py`
- Modify: `tests/test_two_timescale_monte_carlo.py`

- [ ] **Step 1: Add a failing shared-executor delegation test**

Inject a spy executor into `TwoTimescaleRuntimeSimulator` and assert it is called once per slot and its final plan/costs are copied to the slot record:

```python
assert spy_executor.call_count == len(result.records)
assert result.records[0].total_run_cost == pytest.approx(
    spy_executor.results[0].run_cost
)
assert result.records[0].cloud_used is spy_executor.results[0].used_cloud
```

- [ ] **Step 2: Run simulator tests and observe the missing injection point**

```powershell
D:\Anaconda3\python.exe -m pytest -q -p no:cacheprovider -c NUL tests/test_two_timescale_simulator.py tests/test_two_timescale_monte_carlo.py
```

Expected: FAIL because the simulator still owns duplicate audit/repair/execute logic.

- [ ] **Step 3: Refactor `run()` to delegate each slot**

Add a `fast_slot_executor: FastSlotExecutor` constructor dependency. Remove simulator-local `_build_candidate_map`, `_cold_activated_pairs`, and `_calculate_active_memory` after all callers use the shared executor. Keep slow-state/control timing in the simulator.

Extend `TwoTimescaleSlotRecord` and summary with:

```python
retention_policy: str
cloud_policy: str
cloud_used: bool
total_run_cost: float
total_route_cost: float
total_cold_start_cost: float
total_system_cost: float
```

Preserve prior repair and constraint metrics, sourcing them from `FastSlotExecutionResult`.

- [ ] **Step 4: Run simulator and Monte Carlo tests**

Expected: all pass and paired seeds remain deterministic.

- [ ] **Step 5: Commit simulator delegation**

```powershell
git add src/two_timescale_simulator.py src/two_timescale_monte_carlo.py run_two_timescale_comparison.py tests/test_two_timescale_simulator.py tests/test_two_timescale_monte_carlo.py
git commit -m "refactor: share fast execution in simulator"
```

---

### Task 12: Add masked DDQN learning and checkpoint schema metadata

**Files:**
- Modify: `src/ddqn.py`
- Modify: `tests/test_ddqn.py`

- [ ] **Step 1: Write failing exploration, target-mask, and metadata tests**

```python
def test_exploration_never_selects_masked_action() -> None:
    agent = build_agent(state_dim=78, action_count=12)
    mask = np.zeros(12, dtype=np.bool_)
    mask[[2, 9]] = True
    actions = {
        agent.select_action(np.zeros(78, np.float32), 1.0, mask)
        for _ in range(100)
    }
    assert actions <= {2, 9}


def test_replay_batch_contains_next_action_masks() -> None:
    buffer = ReplayBuffer(capacity=10, state_dim=78, action_count=12)
    buffer.add(
        state=np.zeros(78, np.float32),
        action=2,
        reward=-0.1,
        next_state=np.zeros(78, np.float32),
        done=False,
        next_action_mask=np.ones(12, dtype=np.bool_),
    )
    batch = buffer.sample(1, np.random.default_rng(1))
    assert batch.next_action_masks.shape == (1, 12)


def test_old_schema_checkpoint_is_rejected(tmp_path: Path) -> None:
    agent = build_agent(state_dim=78, action_count=12)
    path = tmp_path / "old.pt"
    torch.save({"state_dim": 78, "action_count": 12}, path)
    with pytest.raises(ValueError, match="状态模式版本"):
        agent.load(path)
```

- [ ] **Step 2: Run DDQN tests and observe signature failures**

```powershell
D:\Anaconda3\python.exe -m pytest -q -p no:cacheprovider -c NUL tests/test_ddqn.py
```

Expected: FAIL because replay masks and schema metadata are absent.

- [ ] **Step 3: Implement masked selection, masked Double-DQN targets, and metadata**

Add:

```python
@dataclass(frozen=True)
class DDQNCheckpointMetadata:
    state_schema_version: str
    action_schema_version: str
    mec_count: int
    function_count: int
```

Require the metadata at construction time so an agent cannot save an unversioned model:

```python
def __init__(
    self,
    state_dim: int,
    action_count: int,
    config: DDQNConfig,
    metadata: DDQNCheckpointMetadata,
    device: str | torch.device | None = None,
) -> None:
    """创建带状态/动作模式版本信息的DDQN智能体。"""
```

Validate that `state_dim == 78`, `action_count == 12`, and that the supplied metadata describes the current five-MEC/three-function scenario before building the networks.

`ReplayBuffer` receives `action_count`, allocates `next_action_masks[capacity, action_count]`, validates boolean shape, and returns it in `ReplayBatch`.

Change selection to:

```python
def select_action(
    self,
    state: np.ndarray,
    epsilon: float,
    action_mask: np.ndarray,
) -> int:
    valid_ids = np.flatnonzero(action_mask)
    if len(valid_ids) == 0:
        raise ValueError("当前状态没有合法DDQN动作。")
    if self.rng.random() < epsilon:
        return int(self.rng.choice(valid_ids))
    q_values = self.online_network(state_tensor)
    mask_tensor = torch.from_numpy(action_mask).to(self.device)
    masked = q_values.masked_fill(~mask_tensor.unsqueeze(0), float("-inf"))
    return int(masked.argmax(dim=1).item())
```

In `learn`, mask online-network next Q values before `argmax`; the target network evaluates only that selected valid action. Terminal transitions store an all-true next mask because their bootstrapped term is multiplied by zero.

Save `asdict(self.metadata)` and require exact equality of both schema versions, MEC count, function count, state dimension, and action count before loading weights.

- [ ] **Step 4: Run DDQN tests**

Expected: output shape is `(batch, 12)`, masks are honored, loss is finite, and old checkpoints are rejected.

- [ ] **Step 5: Commit masked DDQN core**

```powershell
git add src/ddqn.py tests/test_ddqn.py
git commit -m "feat: mask DDQN actions and version checkpoints"
```

---

### Task 13: Rebuild the slow-timescale RL environment around the shared components

**Files:**
- Modify: `src/slow_timescale_rl_env.py`
- Modify: `src/rl_reward.py`
- Modify: `src/rl_scenario.py`
- Modify: `tests/test_slow_timescale_rl_env.py`

- [ ] **Step 1: Replace old behavior tests with approved-environment tests**

Add tests for:

```python
def test_reset_returns_v2_state_and_action_mask() -> None:
    env = build_test_environment()
    state, info = env.reset(seed=123)
    assert state.shape == (78,)
    assert env.state_dim == 78
    assert env.action_count == 12
    assert info["action_mask"].shape == (12,)


def test_future_trace_change_does_not_change_current_state() -> None:
    first = build_test_environment(request_trace=[0, 1, 9, 9])
    second = build_test_environment(request_trace=[0, 1, 0, 0])
    first_state, _ = first.reset(seed=1)
    second_state, _ = second.reset(seed=1)
    np.testing.assert_array_equal(first_state, second_state)


def test_step_uses_shared_executor_and_simplified_reward() -> None:
    env, spy = build_environment_with_spy_executor()
    env.reset(seed=1)
    next_state, reward, terminated, truncated, info = env.step(3)
    assert spy.call_count == info["window_length"]
    assert reward == info["reward_breakdown"].reward
    assert "fast_repair_attempts" in info["metrics"].__dict__
    assert next_state.shape == (78,)
```

Also test that a masked action raises a clear error and `advance_without_action()` records a violation without creating a replay action.

- [ ] **Step 2: Run the environment tests and observe old dimensions/actions**

```powershell
D:\Anaconda3\python.exe -m pytest -q -p no:cacheprovider -c NUL tests/test_slow_timescale_rl_env.py
```

Expected: FAIL because the old environment still returns 15 dimensions and four legacy actions.

- [ ] **Step 3: Refactor the environment atomically**

Required constructor dependencies:

```python
state_encoder: RLStateEncoder
workload_predictor: HistoricalWorkloadPredictor
fast_slot_executor: FastSlotExecutor
maximum_window_cost: float
minimum_distinct_fault_domains: int
```

Required behavior:

1. Keep the pre-generated execution trace for repeatable failures and actual requests, but never pass future trace entries to the predictor.
2. Maintain `_observed_request_counts`; append a count only after that fast slot has executed.
3. Build node free-resource observations from the last final audit, or full capacity at reset. For each trackside or cloud node, calculate `free_cpu = max(0.0, node.cpu_capacity - last_audit.node_cpu_demand.get(node_id, 0.0))` and the analogous free-memory value before normalization.
4. `get_valid_action_mask()` builds `ActionFeasibilityContext` from the current infrastructure state.
5. `step(action_id)` decodes the 12-action tuple, creates `SlowTimescaleDecision`, calls the shared executor for every slot, and aggregates cost/repair/cloud/reliability metrics.
6. Call `calculate_cost_reward` with the three aggregated costs and `has_violation = sla_violations > 0`.
7. Return the next action mask in `info["next_observation"]["action_mask"]`.
8. Remove `_predict_request_rate`, `_build_candidate_map`, `_hot_nodes_for_action`, and all direct `execute_sfc_batch` logic.

Use the same explicit observation formulas for every compute node, including the cloud:

```python
node_failure_probability = min(
    1.0,
    max(
        0.0,
        1.0
        - node.reliability
        * (1.0 - global_failure_risk),
    ),
)
normalized_node_delay = min(
    1.0,
    network.transfer_delay_ms(
        1.0,
        serving_node_id,
        node.node_id,
    ) / maximum_network_delay_ms,
)
```

`rl_scenario.py` owns construction of the `{1, 2, 3}` replica-planner mapping and passes that mapping only to `FastSlotExecutor`. The environment receives the completed executor and never selects a planner directly.

Extend `RLWindowMetrics` with defaults for raw costs, repair attempts/success/failure, constraint rejection, cloud use, and exact reliability statistics. After all callers migrate, delete `RLRewardWeights`, `calculate_rl_reward`, and the old weighted breakdown tests.

- [ ] **Step 4: Run environment, reward, executor, and scenario tests**

```powershell
D:\Anaconda3\python.exe -m pytest -q -p no:cacheprovider -c NUL tests/test_slow_timescale_rl_env.py tests/test_rl_reward.py tests/test_fast_slot_executor.py tests/test_config.py
```

Expected: all pass, no source file imports `SlowControlAction`, and state dimension is 78.

- [ ] **Step 5: Commit RL environment integration**

```powershell
git add src/slow_timescale_rl_env.py src/rl_reward.py src/rl_scenario.py tests/test_slow_timescale_rl_env.py tests/test_rl_reward.py tests/test_config.py
git commit -m "feat: rebuild DDQN slow environment"
```

---

### Task 14: Update training, evaluation, configuration, and demos

**Files:**
- Modify: `src/ddqn_evaluation.py`
- Modify: `run_ddqn_training.py`
- Modify: `run_ddqn_evaluation.py`
- Modify: `run_rl_environment_demo.py`
- Modify: `configs/debug.yaml`
- Modify: `tests/test_ddqn_evaluation.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write failing evaluation and training-integration tests**

Update evaluation tests to require:

```python
assert record.fast_repair_attempts >= 0
assert record.constraint_rejected_batches >= 0
assert 0.0 <= record.cloud_usage_rate <= 1.0
assert len(record.action_counts) == 12

rows = build_action_distribution_rows([record])
assert {row["action_id"] for row in rows} == set(range(12))
assert all("replica_count" in row for row in rows)
assert all("retention_policy" in row for row in rows)
assert all("cloud_policy" in row for row in rows)
```

Add a training helper test that a one-step transition passes `next_action_mask` to `ReplayBuffer.add` and never calls the removed environment-action adapter.

- [ ] **Step 2: Run evaluation/config tests and observe old columns**

```powershell
D:\Anaconda3\python.exe -m pytest -q -p no:cacheprovider -c NUL tests/test_ddqn_evaluation.py tests/test_config.py
```

Expected: FAIL because output still assumes single/cold/hot/dynamic ratios.

- [ ] **Step 3: Update all entry points and output schemas**

Training loop changes:

```python
action_mask = info["action_mask"]
action = agent.select_action(state, epsilon, action_mask)
next_state, reward, terminated, truncated, step_info = environment.step(action)
next_mask = step_info["next_observation"]["action_mask"]
replay_buffer.add(
    state=state,
    action=action,
    reward=reward,
    next_state=next_state,
    done=terminated or truncated,
    next_action_mask=(
        np.ones(12, dtype=np.bool_)
        if terminated or truncated
        else next_mask
    ),
)
```

If the current mask is all false, call `environment.advance_without_action()` and do not add an action transition.

Add command-line options:

```text
--episodes INTEGER
--output-root PATH
--device {cpu,cuda}
```

The defaults retain current formal paths and configured Episode count. Smoke runs use `--output-root` so formal models are untouched.

Replace legacy action-ratio columns with:

- one row per action ID/name/count/ratio;
- marginal `replica_2_ratio`, `replica_3_ratio`;
- `on_demand_ratio`, `primary_warm_ratio`, `all_warm_ratio`;
- `edge_only_ratio`, `cloud_allowed_ratio`;
- request success and SLA-violation rates;
- mean, peak, p95, and p99 end-to-end delay;
- mean and peak active warm-instance memory plus cold-start count;
- minimum and mean exact SFC reliability;
- observed cloud execution rate;
- repair attempts/success/failure and constraint rejection;
- run/route/cold/total costs.

Add configuration:

```yaml
rl_environment:
  workload_prediction:
    lookback_slots: 10
    baseline_request_rate: 1.0
  state_normalization:
    maximum_request_rate: 5.0
    maximum_network_delay_ms: 200.0
    maximum_input_size_mb: 10.0
  cost_rates:
    edge_cpu_cost_per_unit: 0.01
    edge_memory_cost_per_mb_second: 0.001
    cloud_cpu_cost_per_unit: 0.05
    cloud_memory_cost_per_mb_second: 0.005
    cold_start_cost_per_ms: 0.10
    maximum_window_cost: 10000.0
```

Construct `DDQNCheckpointMetadata` with state schema `ddqn-v2-78`, action schema `structured-slow-v1`, MEC count 5, and function count 3.

- [ ] **Step 4: Run training/evaluation unit tests and a one-Episode programmatic run**

```powershell
D:\Anaconda3\python.exe -m pytest -q -p no:cacheprovider -c NUL tests/test_ddqn.py tests/test_ddqn_evaluation.py tests/test_slow_timescale_rl_env.py tests/test_config.py
```

Expected: all pass and no output code references `single_ratio`, `cold_ratio`, `hot_ratio`, or `dynamic_ratio`.

- [ ] **Step 5: Commit entry-point and metric migration**

```powershell
git add src/ddqn_evaluation.py run_ddqn_training.py run_ddqn_evaluation.py run_rl_environment_demo.py configs/debug.yaml tests/test_ddqn_evaluation.py tests/test_config.py
git commit -m "feat: report structured DDQN experiments"
```

---

### Task 15: Full verification and temporary smoke training

**Files:**
- Verify: all Python source and tests
- Temporary output only: `$env:TEMP\ddqn-redesign-smoke`

- [ ] **Step 1: Run the complete test suite**

```powershell
D:\Anaconda3\python.exe -m pytest -q -p no:cacheprovider -c NUL tests
```

Expected: all tests pass; the count is at least the 194-test baseline plus the newly added tests.

- [ ] **Step 2: Parse every Python file**

```powershell
Get-ChildItem -Path src,tests -Recurse -Filter '*.py' -File | ForEach-Object {
    D:\Anaconda3\python.exe -m py_compile $_.FullName
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
```

Expected: exit code 0 and no syntax errors.

- [ ] **Step 3: Run a five-Episode smoke training outside formal result paths**

```powershell
$smokeRoot=Join-Path $env:TEMP 'ddqn-redesign-smoke'
D:\Anaconda3\python.exe run_ddqn_training.py --episodes 5 --output-root $smokeRoot --device cpu
```

Expected:

- Q-network output count reported as 12;
- state dimension reported as 78;
- every selected action is allowed by its mask;
- rewards and losses are finite;
- checkpoint save and reload succeed;
- no file under the formal `results/models` or DDQN figure/table paths changes.

- [ ] **Step 4: Run a smoke evaluation using the temporary checkpoint**

Use the evaluation script's model/output overrides added in Task 14:

```powershell
D:\Anaconda3\python.exe run_ddqn_evaluation.py --model-root $smokeRoot --output-root $smokeRoot\evaluation --episodes 2 --device cpu
```

Expected: evaluation completes and writes structured action, cloud, repair, reliability, and cost metrics.

- [ ] **Step 5: Inspect repository cleanliness and intended diff**

```powershell
git status --short
git diff --check
git log --oneline --decorate -15
```

Expected: only intended tracked changes remain; temporary smoke artifacts are outside the repository; `git diff --check` reports no whitespace errors.

- [ ] **Step 6: Commit any verification-only corrections, if the preceding checks required tracked fixes**

When and only when verification exposed a tracked correction:

```powershell
git add src tests configs/debug.yaml run_ddqn_training.py run_ddqn_evaluation.py run_rl_environment_demo.py run_two_timescale_comparison.py
git commit -m "fix: complete DDQN redesign verification"
```

If `git status --short` is empty, do not create an empty commit.

## Completion boundary

This plan ends after code, tests, smoke training, code review, and GitHub push. Formal training that overwrites the old DDQN model and same-named DDQN figures/tables is a separate user-approved phase after review. That later formal phase must include predicted-high-risk and simultaneous multi-node/multi-fault-domain failure scenarios, and must report the 2/3-replica selection distribution; otherwise the three-replica action cannot be claimed as an experimentally supported contribution.
