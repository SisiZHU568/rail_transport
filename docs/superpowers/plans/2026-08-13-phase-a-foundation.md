# Phase A Configuration, CTMC, and Instance Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 完成双时间尺度主算法阶段 A：配置驱动的部署边界、精确 CTMC 故障快照、温/启动中实例批次生命周期和原子部署提交，并彻底删除车载计算候选语义。

**Architecture:** 新增只读配置模型和独立生命周期状态所有者；`failure_process.py` 只持有故障域/节点的隐藏 CTMC 状态与独立随机流，输出不可变 `FailureSnapshot`；`instance_lifecycle.py` 只持有实例批次并通过预演/原子提交更新状态。阶段 A 不接入队列、凸优化、DPPO 新动作或奖励，后续阶段只消费这里定义的快照接口。

**Tech Stack:** Python 3.12、dataclasses、PyYAML、pytest；不新增第三方依赖。

---

## 文件结构

- Create: `src/orchestration_config.py` — 阶段 A 只读配置对象、CTMC 率与稳态可用度。
- Modify: `src/config.py` — 在 YAML 读取入口调用阶段 A 严格校验。
- Modify: `configs/debug.yaml` — 连续时间故障率、生命周期参数、实例上下限；删除车载计算字段。
- Modify: `src/entities.py` — 删除未使用的 `NodeType.ONBOARD` 计算节点类型。
- Modify: `src/topology.py`, `src/reliability.py` — 仅构建轨旁 MEC 和中心云，节点/故障域稳态可用度来自同一 CTMC 率源。
- Modify: `src/failure_process.py` — 用 `FailureSnapshot` 和精确 CTMC 过程替换旧的按时隙概率模型。
- Create: `src/instance_lifecycle.py` — 实例批次、快照、预演、故障销毁与原子部署提交。
- Create: `src/cost_ledger.py` — 部署、冷启动和实例—秒运行费的唯一账本。
- Create: `src/orchestration_core.py` — 固定阶段 A 的单时隙调用顺序并统一提交计费事件，不持有故障/生命周期内部状态。
- Modify: `src/slow_timescale_execution_core.py`, `src/fast_slot_executor.py`, `src/simulator.py`, `src/risk_aware_failure_process.py`, `src/dppo_scenario.py` — 消费新故障快照与连续时间率接口。
- Create: `tests/test_orchestration_config.py`, `tests/test_instance_lifecycle.py`, `tests/test_phase_a_orchestration.py` — 新模型核心测试。
- Modify: `tests/test_config.py`, `tests/test_topology.py`, `tests/test_entities.py`, `tests/test_failure_process.py` 及直接依赖故障快照的测试。

### Task 1: 锁定阶段 A 配置并删除车载计算候选

**Files:**
- Create: `src/orchestration_config.py`
- Modify: `src/config.py`
- Modify: `configs/debug.yaml`
- Modify: `src/entities.py`
- Modify: `src/topology.py`
- Modify: `src/reliability.py`
- Create: `tests/test_orchestration_config.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_entities.py`
- Modify: `tests/test_topology.py`

- [ ] **Step 1: 写配置与车载删除的失败测试**

```python
# tests/test_orchestration_config.py
from copy import deepcopy

import pytest

from src.config import load_config, validate_config
from src.orchestration_config import load_phase_a_config


def test_debug_config_has_no_onboard_compute_fields() -> None:
    config = load_config("configs/debug.yaml")
    assert "include_onboard" not in config["topology"]
    assert all(not key.startswith("onboard_") for key in config["node_resources"])


def test_phase_a_config_uses_explicit_ctmc_rates_and_instance_ranges() -> None:
    model = load_phase_a_config(load_config("configs/debug.yaml"))
    assert model.fast_slot_seconds == pytest.approx(1.0)
    assert model.failure_base_seed == 42
    assert model.deployment_pairs[(0, 0)].cold_start_seconds == pytest.approx(0.3)
    assert set(model.domain_rates) == {0, 1, 2, 3}
    assert set(model.node_rates) == {0, 1, 2, 3, 4, 5}
    assert all(rate.failure_rate_per_second >= 0.0 for rate in model.node_rates.values())
    assert model.allowed_instance_counts(0, 0) == (0, 1, 2, 3, 4)
    assert model.allowed_instance_counts(0, 5) == (0, 1, 2, 3, 4)
    assert model.retention_slot_options == (0, 5, 10, 15, 20)


@pytest.mark.parametrize(
    ("failure_rate", "recovery_rate"),
    ((-0.1, 1.0), (1.0, -0.1), (0.0, 0.0)),
)
def test_phase_a_config_rejects_invalid_ctmc_rates(
    failure_rate: float,
    recovery_rate: float,
) -> None:
    config = deepcopy(load_config("configs/debug.yaml"))
    config["runtime_failure"]["domain_rates"][0].update(
        failure_rate_per_second=failure_rate,
        recovery_rate_per_second=recovery_rate,
    )
    with pytest.raises(ValueError, match="domain_rates"):
        validate_config(config)
```

```python
# tests/test_entities.py
from src.entities import NodeType


def test_onboard_is_not_a_compute_node_type() -> None:
    assert {item.value for item in NodeType} == {"trackside", "cloud"}
```

- [ ] **Step 2: 运行测试并确认 RED**

Run:

```powershell
python -m pytest tests/test_orchestration_config.py tests/test_config.py tests/test_entities.py tests/test_topology.py -q
```

Expected: 新模块不存在、旧 YAML 仍含 `include_onboard`，测试失败。

- [ ] **Step 3: 实现不可变阶段 A 配置对象与严格校验**

```python
# src/orchestration_config.py
from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class CTMCRates:
    failure_rate_per_second: float
    recovery_rate_per_second: float

    def __post_init__(self) -> None:
        values = (self.failure_rate_per_second, self.recovery_rate_per_second)
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("CTMC 失效率和恢复率必须是非负有限数。")
        if sum(values) <= 0.0:
            raise ValueError("CTMC 失效率与恢复率不能同时为零。")

    @property
    def steady_availability(self) -> float:
        return self.recovery_rate_per_second / (
            self.failure_rate_per_second + self.recovery_rate_per_second
        )


@dataclass(frozen=True)
class LifecycleConfig:
    schema_version: str
    price_version: str
    minimum_retention_seconds: float
    maximum_retention_seconds: float
    retention_step_seconds: float


@dataclass(frozen=True)
class DeploymentPairConfig:
    function_id: int
    node_id: int
    minimum_instances: int
    maximum_instances: int
    cold_start_seconds: float
    single_instance_max_cpu_cycles_per_second: float
    deployment_cost_per_instance: float
    cold_start_cost_per_instance: float
    running_cost_per_instance_second: float

    @property
    def allowed_instance_counts(self) -> tuple[int, ...]:
        return (0, *range(self.minimum_instances, self.maximum_instances + 1))


@dataclass(frozen=True)
class NodeResourceConfig:
    node_id: int
    memory_capacity_mb: float
    cpu_capacity_cycles_per_second: float
    core_count: int


@dataclass(frozen=True)
class PhaseAConfig:
    fast_slot_seconds: float
    failure_base_seed: int
    domain_rates: dict[int, CTMCRates]
    node_rates: dict[int, CTMCRates]
    node_resources: dict[int, NodeResourceConfig]
    deployment_pairs: dict[tuple[int, int], DeploymentPairConfig]
    lifecycle: LifecycleConfig

    @property
    def retention_slot_options(self) -> tuple[int, ...]:
        start = self.lifecycle.minimum_retention_seconds
        stop = self.lifecycle.maximum_retention_seconds
        step = self.lifecycle.retention_step_seconds
        seconds = {stop}
        index = 0
        while start + index * step <= stop:
            seconds.add(start + index * step)
            index += 1
        return tuple(sorted({math.ceil(value / self.fast_slot_seconds) for value in seconds}))

    def allowed_instance_counts(self, function_id: int, node_id: int) -> tuple[int, ...]:
        return self.deployment_pairs[(function_id, node_id)].allowed_instance_counts


def _rate_map(items: object, field: str) -> dict[int, CTMCRates]:
    if not isinstance(items, list) or not items:
        raise ValueError(f"runtime_failure.{field} 必须是非空列表。")
    result: dict[int, CTMCRates] = {}
    for item in items:
        if not isinstance(item, dict) or isinstance(item.get("id"), bool):
            raise ValueError(f"runtime_failure.{field} 条目格式错误。")
        entity_id = item.get("id")
        if not isinstance(entity_id, int) or entity_id < 0 or entity_id in result:
            raise ValueError(f"runtime_failure.{field} 的 id 必须唯一且非负。")
        try:
            result[entity_id] = CTMCRates(
                failure_rate_per_second=float(item["failure_rate_per_second"]),
                recovery_rate_per_second=float(item["recovery_rate_per_second"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"runtime_failure.{field} 包含无效 CTMC 率。") from error
    return result


def load_phase_a_config(config: dict[str, Any]) -> PhaseAConfig:
    simulation = config["simulation"]
    runtime = config["runtime_failure"]
    lifecycle = config["instance_lifecycle"]
    slot_seconds = float(simulation["fast_slot_seconds"])
    if not math.isfinite(slot_seconds) or slot_seconds <= 0.0:
        raise ValueError("simulation.fast_slot_seconds 必须是正有限数。")
    node_resources = _parse_node_resources(config)
    deployment_pairs = _parse_deployment_pairs(config, node_resources)
    result = PhaseAConfig(
        fast_slot_seconds=slot_seconds,
        failure_base_seed=int(runtime["base_seed"]),
        domain_rates=_rate_map(runtime.get("domain_rates"), "domain_rates"),
        node_rates=_rate_map(runtime.get("node_rates"), "node_rates"),
        node_resources=node_resources,
        deployment_pairs=deployment_pairs,
        lifecycle=LifecycleConfig(
            schema_version=str(lifecycle["schema_version"]),
            price_version=str(lifecycle["price_version"]),
            minimum_retention_seconds=float(lifecycle["minimum_retention_seconds"]),
            maximum_retention_seconds=float(lifecycle["maximum_retention_seconds"]),
            retention_step_seconds=float(lifecycle["retention_step_seconds"]),
        ),
    )
    _validate_phase_a_references(config, result)
    return result
```

`_parse_node_resources` 展开配置中轨旁 MEC 的同构资源和中心云资源，得到规范节点 ID；`_parse_deployment_pairs` 将每条 `(function_id, node_ids)` 声明展开为唯一 `(function_id,node_id)` 键，并验证整数范围、有限正容量、非负价格和引用；`_validate_phase_a_references` 要求基础种子为非负整数，域率、节点率、节点资源和拓扑节点集合完全一致，每个 SFC 函数至少有一个允许部署节点，保留时间满足 `0 <= min <= max` 且 `step > 0`，并要求每个组合的 `cold_start_seconds * 1000` 与同一 VNF 的既有 `cold_start_time_ms` 一致。任何重复组合或未知 ID 都以带字段路径的 `ValueError` 拒绝。

完成以下入口和实体修改：

```python
# src/config.py: validate_config() 结尾
from src.orchestration_config import load_phase_a_config

# 原有校验之后，阶段 A 配置成为同一加载事务的一部分。
load_phase_a_config(config)
```

```python
# src/entities.py
class NodeType(str, Enum):
    TRACKSIDE = "trackside"
    CLOUD = "cloud"
```

YAML 中彻底删除 `topology.include_onboard`、三个 `node_resources.onboard_*`、`mec_reliability`、`cloud_reliability` 和 `reliability.fault_domains`。`serverless.max_replicas` 与 `dppo.action.minimum_replicas/maximum_replicas` 在阶段 A 仍由尚未替换的旧动作模块读取，暂不删除；阶段 D 在替换旧动作空间时移除，它们不得参与 `PhaseAConfig` 的新生命周期判断。新增的规范形状固定为：

```yaml
node_resources:
  mec_cpu_capacity: 100.0                  # 旧执行器在阶段 C 替换前继续读取
  mec_memory_mb: 8192.0
  mec_cpu_capacity_cycles_per_second: 20000000000.0
  mec_core_count: 8
  cloud_cpu_capacity: 1000.0               # 旧执行器在阶段 C 替换前继续读取
  cloud_memory_mb: 65536.0
  cloud_cpu_capacity_cycles_per_second: 100000000000.0
  cloud_core_count: 32
  cloud_fault_domain_id: 3

runtime_failure:
  base_seed: 42
  domain_rates:
    - {id: 0, failure_rate_per_second: 0.000005, recovery_rate_per_second: 0.001}
    - {id: 1, failure_rate_per_second: 0.000005, recovery_rate_per_second: 0.001}
    - {id: 2, failure_rate_per_second: 0.000005, recovery_rate_per_second: 0.001}
    - {id: 3, failure_rate_per_second: 0.000001, recovery_rate_per_second: 0.001}
  node_rates:
    - {id: 0, failure_rate_per_second: 0.00002, recovery_rate_per_second: 0.001}
    - {id: 1, failure_rate_per_second: 0.00002, recovery_rate_per_second: 0.001}
    - {id: 2, failure_rate_per_second: 0.00002, recovery_rate_per_second: 0.001}
    - {id: 3, failure_rate_per_second: 0.00002, recovery_rate_per_second: 0.001}
    - {id: 4, failure_rate_per_second: 0.00002, recovery_rate_per_second: 0.001}
    - {id: 5, failure_rate_per_second: 0.000001, recovery_rate_per_second: 0.001}
  failover_delay_ms_per_function: 20.0

instance_lifecycle:
  schema_version: phase-a-v1
  price_version: debug-price-v1
  minimum_retention_seconds: 0.0
  maximum_retention_seconds: 20.0
  retention_step_seconds: 5.0
  allowed_deployments:
    - function_id: 0
      node_ids: [0, 1, 2, 3, 4, 5]
      minimum_instances: 1
      maximum_instances: 4
      cold_start_seconds: 0.3
      single_instance_max_cpu_cycles_per_second: 2000000000.0
      deployment_cost_per_instance: 1.0
      cold_start_cost_per_instance: 0.3
      running_cost_per_instance_second: 0.001
    - function_id: 1
      node_ids: [0, 1, 2, 3, 4, 5]
      minimum_instances: 1
      maximum_instances: 4
      cold_start_seconds: 0.5
      single_instance_max_cpu_cycles_per_second: 2000000000.0
      deployment_cost_per_instance: 1.0
      cold_start_cost_per_instance: 0.5
      running_cost_per_instance_second: 0.001
    - function_id: 2
      node_ids: [0, 1, 2, 3, 4, 5]
      minimum_instances: 1
      maximum_instances: 4
      cold_start_seconds: 0.8
      single_instance_max_cpu_cycles_per_second: 2000000000.0
      deployment_cost_per_instance: 1.0
      cold_start_cost_per_instance: 0.8
      running_cost_per_instance_second: 0.001
```

`src/topology.py` 从 `node_rates[node_id].steady_availability` 设置节点局部可靠性；`src/reliability.py` 从 `domain_rates[domain_id].steady_availability` 建立故障域可靠性模型。两者不再读取另一套 availability 字段。

- [ ] **Step 4: 运行配置核心测试并确认 GREEN**

Run:

```powershell
python -m pytest tests/test_orchestration_config.py tests/test_config.py tests/test_entities.py tests/test_topology.py -q
```

Expected: 全部通过，且活动配置中不存在车载计算字段。

- [ ] **Step 5: 提交配置边界**

```powershell
git add src/orchestration_config.py src/config.py src/entities.py src/topology.py src/reliability.py configs/debug.yaml tests/test_orchestration_config.py tests/test_config.py tests/test_entities.py tests/test_topology.py tests/test_reliability.py
git commit -m "refactor: define phase A model configuration"
```

### Task 2: 用精确 CTMC 和独立随机流生成故障快照

**Files:**
- Modify: `src/failure_process.py`
- Modify: `tests/test_failure_process.py`

- [ ] **Step 1: 写 CTMC 数学、稳态初始化和有效事件的失败测试**

```python
# tests/test_failure_process.py
import math

import pytest

from src.failure_process import CTMCTransition, MarkovFailureProcess
from src.orchestration_config import CTMCRates


def test_exact_ctmc_discretization_preserves_stationary_availability() -> None:
    rates = CTMCRates(0.2, 0.8)
    transition = CTMCTransition.from_rates(rates, slot_seconds=3.0)
    expected_change = 1.0 - math.exp(-(0.2 + 0.8) * 3.0)
    assert transition.failure_probability == pytest.approx(0.2 * expected_change)
    assert transition.recovery_probability == pytest.approx(0.8 * expected_change)
    assert transition.recovery_probability / (
        transition.failure_probability + transition.recovery_probability
    ) == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("rates", "expected_up"),
    ((CTMCRates(0.0, 1.0), True), (CTMCRates(1.0, 0.0), False)),
)
def test_stationary_initialization_handles_zero_rate_boundaries(
    rates: CTMCRates,
    expected_up: bool,
) -> None:
    assert MarkovFailureProcess.stationary_initial_state(rates, draw=0.5) is expected_up


def test_domain_recovery_does_not_restore_locally_failed_node(test_topology) -> None:
    process = MarkovFailureProcess.for_test_script(
        topology=test_topology,
        domain_states=[{0: False}, {0: True}],
        node_states=[{0: False}, {0: False}],
    )
    first = process.state_for_slot(0)
    second = process.state_for_slot(1)
    assert 0 not in second.newly_available_node_ids
    assert second.is_node_operational(0, test_topology) is False
    assert first.version + 1 == second.version
```

- [ ] **Step 2: 运行测试并确认 RED**

Run:

```powershell
python -m pytest tests/test_failure_process.py -q
```

Expected: `CTMCTransition`、新快照字段或稳态初始化不存在，测试失败。

- [ ] **Step 3: 实现精确边界采样 CTMC**

```python
# src/failure_process.py（核心定义）
from dataclasses import dataclass
import hashlib
import math
import random

from src.orchestration_config import CTMCRates


@dataclass(frozen=True)
class CTMCTransition:
    failure_probability: float
    recovery_probability: float

    @classmethod
    def from_rates(cls, rates: CTMCRates, slot_seconds: float) -> "CTMCTransition":
        total = rates.failure_rate_per_second + rates.recovery_rate_per_second
        change = -math.expm1(-total * slot_seconds)
        return cls(
            failure_probability=rates.failure_rate_per_second / total * change,
            recovery_probability=rates.recovery_rate_per_second / total * change,
        )


@dataclass(frozen=True)
class FailureSnapshot:
    version: int
    time_slot: int
    domain_hidden_up: tuple[tuple[int, bool], ...]
    node_hidden_up: tuple[tuple[int, bool], ...]
    effective_node_up: tuple[tuple[int, bool], ...]
    newly_unavailable_node_ids: tuple[int, ...]
    newly_available_node_ids: tuple[int, ...]

    def is_node_operational(self, node_id: int, topology) -> bool:
        del topology  # 有效状态已在快照生成时固化。
        return dict(self.effective_node_up)[node_id]


def _stream_seed(base_seed: int, kind: str, entity_id: int) -> int:
    payload = f"{base_seed}:{kind}:{entity_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
```

`MarkovFailureProcess.reset()` 必须为每个域、每个节点分别创建 `random.Random(_stream_seed(...))`，并用各自稳态可用度初始化隐藏状态。`state_for_slot()` 只允许按 0、1、2…顺序访问；从时隙 1 开始先更新每条隐藏链，再由“域隐藏状态 AND 节点隐藏状态”生成有效状态和两个事件集合。这里的精确性仅指时隙边界状态转移矩阵，不模拟时隙内短暂故障。

- [ ] **Step 4: 运行故障测试并确认 GREEN**

Run:

```powershell
python -m pytest tests/test_failure_process.py -q
```

Expected: 精确概率、稳态初始化、重置复现、域/节点有效事件全部通过。

- [ ] **Step 5: 提交 CTMC 故障模型**

```powershell
git add src/failure_process.py tests/test_failure_process.py
git commit -m "refactor: add exact CTMC failure snapshots"
```

### Task 3: 将现有调用点迁移到唯一 `FailureSnapshot`

**Files:**
- Modify: `src/slow_timescale_execution_core.py`
- Modify: `src/fast_slot_executor.py`
- Modify: `src/simulator.py`
- Modify: `src/risk_aware_failure_process.py`
- Modify: `src/dppo_scenario.py`
- Modify: `run_monte_carlo_experiment.py`
- Modify: `run_runtime_reliability_demo.py`
- Modify: `run_two_timescale_monte_carlo.py`
- Modify: `tests/test_fast_slot_executor.py`
- Modify: `tests/test_risk_aware_failure_process.py`

- [ ] **Step 1: 增加旧快照名称已删除的范围测试**

```python
# tests/test_active_algorithm_scope.py
def test_active_python_tree_uses_only_failure_snapshot() -> None:
    root = Path(__file__).resolve().parents[1]
    offenders = []
    retired = "Infrastructure" + "State"
    candidates = (
        tuple((root / "src").rglob("*.py"))
        + tuple((root / "tests").rglob("*.py"))
        + tuple(root.glob("run_*.py"))
    )
    for path in candidates:
        if path.name == "test_active_algorithm_scope.py":
            continue
        if retired in path.read_text(encoding="utf-8"):
            offenders.append(path.relative_to(root).as_posix())
    assert offenders == []
```

- [ ] **Step 2: 运行迁移测试并确认 RED**

Run:

```powershell
python -m pytest tests/test_active_algorithm_scope.py tests/test_slow_timescale_execution_core.py tests/test_fast_slot_executor.py tests/test_simulator.py tests/test_risk_aware_failure_process.py -q
```

Expected: 旧名称仍被引用，范围测试失败。

- [ ] **Step 3: 机械迁移类型和测试夹具**

`src/slow_timescale_execution_core.py` 的 `EpisodeTraceSlot.infrastructure_state`、`src/fast_slot_executor.py` 的 `FastSlotInput.infrastructure_state` 和 `src/simulator.py` 的故障输入统一改为 `FailureSnapshot`。公共数据类只保留新类型：

```python
from src.failure_process import FailureSnapshot


@dataclass(frozen=True)
class FastSlotInput:
    train_state: TrainState
    request_count: int
    infrastructure_state: FailureSnapshot
    slow_decision: SlowTimescaleDecision | None
    previous_candidate_map: dict[int, tuple[int, ...]] | None
    deployment_intent: SFCDeploymentIntent | None = None
```

`risk_aware_failure_process.py` 不再持有第二套概率型 Markov 类。保留其公开构建函数名以减少调用点噪声，但函数内部根据当前时隙风险窗口对基础 **failure rate** 乘确定性倍数，再调用同一个 `CTMCTransition.from_rates()`；恢复率不变，域/节点随机流仍由实体 ID 分离。它返回同一 `FailureSnapshot`，稳态初始化使用基础率。测试需要人工状态时，通过 `ScriptedFailureProcess.state_for_slot()` 获取快照，不手写可变字典。迁移完成后删除旧 `InfrastructureState` 与旧概率参数类，不保留别名或兼容开关。

- [ ] **Step 4: 运行直接依赖回归并确认 GREEN**

Run:

```powershell
python -m pytest tests/test_active_algorithm_scope.py tests/test_failure_process.py tests/test_slow_timescale_execution_core.py tests/test_fast_slot_executor.py tests/test_simulator.py tests/test_risk_aware_failure_process.py tests/test_dppo_slow_timescale_env.py -q
```

Expected: 全部通过，活动源码和测试不存在旧快照名称。

- [ ] **Step 5: 提交唯一故障快照迁移**

```powershell
git add src tests
git commit -m "refactor: migrate runtime to failure snapshots"
```

### Task 4: 实现实例批次生命周期和原子部署提交

**Files:**
- Create: `src/instance_lifecycle.py`
- Create: `tests/test_instance_lifecycle.py`

- [ ] **Step 1: 写生命周期边界与原子性的失败测试**

```python
# tests/test_instance_lifecycle.py
import pytest

from src.instance_lifecycle import (
    DeploymentTarget,
    InstanceLifecycleManager,
    LifecycleDeploymentPlan,
    LifecycleStatus,
)


def test_zero_cold_start_is_warm_immediately(manager, failure_snapshot) -> None:
    plan = LifecycleDeploymentPlan(
        expected_lifecycle_version=manager.snapshot().version,
        expected_failure_version=failure_snapshot.version,
        current_slot=4,
        targets=(DeploymentTarget(0, 0, 2, 3),),
    )
    result = manager.commit_deployment(plan, failure_snapshot)
    assert result.accepted is True
    assert result.snapshot.warm_count(0, 0) == 2
    assert result.snapshot.starting_count(0, 0) == 0


def test_ready_and_retention_boundaries_use_greater_or_equal(manager) -> None:
    manager.seed_batch(
        function_id=0,
        node_id=0,
        count=2,
        status=LifecycleStatus.STARTING,
        ready_slot=5,
        retention_deadline_slot=8,
    )
    at_ready = manager.advance_to_slot(5)
    assert at_ready.warm_count(0, 0) == 2
    assert at_ready.locked_count(0, 0, current_slot=7) == 2
    assert at_ready.locked_count(0, 0, current_slot=8) == 0


def test_partial_refresh_splits_batch_without_extending_excess(manager, snapshot) -> None:
    manager.seed_warm_batch(0, 0, count=5, deadline_slot=10)
    result = manager.commit_deployment(
        LifecycleDeploymentPlan(
            expected_lifecycle_version=manager.snapshot().version,
            expected_failure_version=snapshot.version,
            current_slot=5,
            targets=(DeploymentTarget(0, 0, 3, 20),),
        ),
        snapshot,
    )
    assert [(b.count, b.retention_deadline_slot) for b in result.snapshot.batches] == [
        (2, 10),
        (3, 25),
    ]


def test_stale_or_over_memory_plan_has_no_partial_effect(manager, snapshot) -> None:
    before = manager.snapshot()
    stale = LifecycleDeploymentPlan(
        expected_lifecycle_version=before.version + 1,
        expected_failure_version=snapshot.version,
        current_slot=0,
        targets=(DeploymentTarget(0, 0, 99, 1),),
    )
    result = manager.commit_deployment(stale, snapshot)
    assert result.code == "STALE_SNAPSHOT"
    assert manager.snapshot() == before
```

继续在同一文件覆盖：STARTING 计入目标数与内存、温/启动中刷新期限公式、优先复用到期实例、故障销毁全部批次、恢复节点为空、健康/内存/允许实例集合拒绝、确定性拆分 ID、预演不修改真实状态。

- [ ] **Step 2: 运行生命周期测试并确认 RED**

Run:

```powershell
python -m pytest tests/test_instance_lifecycle.py -q
```

Expected: `src.instance_lifecycle` 不存在，测试收集失败。

- [ ] **Step 3: 实现批次、快照和提交接口**

```python
# src/instance_lifecycle.py（公共接口骨架）
from dataclasses import dataclass
from enum import Enum

from src.failure_process import FailureSnapshot


class LifecycleStatus(str, Enum):
    STARTING = "starting"
    WARM = "warm"


@dataclass(frozen=True, order=True)
class InstanceBatch:
    batch_id: str
    function_id: int
    node_id: int
    status: LifecycleStatus
    count: int
    ready_slot: int
    retention_deadline_slot: int


@dataclass(frozen=True)
class LifecycleSnapshot:
    version: int
    current_slot: int
    batches: tuple[InstanceBatch, ...]
    memory_used_mb_by_node: tuple[tuple[int, float], ...]

    def warm_count(self, function_id: int, node_id: int) -> int:
        return sum(
            batch.count
            for batch in self.batches
            if batch.function_id == function_id
            and batch.node_id == node_id
            and batch.status is LifecycleStatus.WARM
        )

    def starting_count(self, function_id: int, node_id: int) -> int:
        return sum(
            batch.count
            for batch in self.batches
            if batch.function_id == function_id
            and batch.node_id == node_id
            and batch.status is LifecycleStatus.STARTING
        )


@dataclass(frozen=True)
class DeploymentTarget:
    function_id: int
    node_id: int
    target_count: int
    retention_slots: int


@dataclass(frozen=True)
class LifecycleDeploymentPlan:
    expected_lifecycle_version: int
    expected_failure_version: int
    current_slot: int
    targets: tuple[DeploymentTarget, ...]
```

`InstanceLifecycleManager.commit_deployment()` 必须先复制批次到局部列表，按固定 `(function_id,node_id,batch_id)` 顺序执行：删除已到期且超出 `max(locked,target)` 的数量、优先复用剩余实例、只刷新目标需要的部分、创建差额；随后审计健康、实例档位和内存。全部通过才一次替换真实批次并递增版本。创建/拆分 ID 使用父批次 ID、提交版本、固定序号的 SHA-256，不使用随机数。

- [ ] **Step 4: 运行生命周期测试并确认 GREEN**

Run:

```powershell
python -m pytest tests/test_instance_lifecycle.py -q
```

Expected: 所有边界、故障、拆分和原子性测试通过。

- [ ] **Step 5: 提交生命周期状态所有者**

```powershell
git add src/instance_lifecycle.py tests/test_instance_lifecycle.py
git commit -m "feat: add atomic instance lifecycle state"
```

### Task 5: 固定阶段 A 事件顺序并完成核心回归

**Files:**
- Create: `src/cost_ledger.py`
- Create: `src/orchestration_core.py`
- Create: `tests/test_phase_a_orchestration.py`
- Create: `tests/test_cost_ledger.py`
- Modify: `README.md`

- [ ] **Step 1: 写冷启动完成、故障销毁和恢复为空的集成失败测试**

```python
# tests/test_phase_a_orchestration.py
def test_phase_a_slot_order_completes_then_applies_failure(coordinator) -> None:
    coordinator.lifecycle.seed_starting_batch(
        function_id=0,
        node_id=0,
        count=2,
        ready_slot=3,
        deadline_slot=9,
    )
    result = coordinator.begin_slot(3)
    assert 0 in result.failure_snapshot.newly_unavailable_node_ids
    assert result.lifecycle_snapshot.warm_count(0, 0) == 0
    assert result.lifecycle_snapshot.starting_count(0, 0) == 0


def test_recovered_node_stays_empty_until_a_new_plan(coordinator) -> None:
    coordinator.begin_slot(3)
    recovered = coordinator.begin_slot(4)
    assert 0 in recovered.failure_snapshot.newly_available_node_ids
    assert all(batch.node_id != 0 for batch in recovered.lifecycle_snapshot.batches)
```

```python
# tests/test_cost_ledger.py
import pytest

from src.cost_ledger import CostLedger, CostLedgerEntry


def test_running_cost_uses_instance_seconds_not_slots() -> None:
    one_second = CostLedger.running_entries(
        slot=0,
        slot_seconds=1.0,
        active_counts={(0, 0): 2},
        running_prices={(0, 0): 0.5},
        price_version="debug-v1",
    )
    half_second_twice = (
        CostLedger.running_entries(
            slot=0,
            slot_seconds=0.5,
            active_counts={(0, 0): 2},
            running_prices={(0, 0): 0.5},
            price_version="debug-v1",
        )
        + CostLedger.running_entries(
            slot=1,
            slot_seconds=0.5,
            active_counts={(0, 0): 2},
            running_prices={(0, 0): 0.5},
            price_version="debug-v1",
        )
    )
    assert sum(item.amount for item in one_second) == pytest.approx(1.0)
    assert sum(item.amount for item in half_second_twice) == pytest.approx(1.0)


def test_ledger_append_is_atomic_and_retention_has_no_charge() -> None:
    ledger = CostLedger()
    entry = CostLedgerEntry(
        slot=3,
        cost_type="deployment",
        function_id=0,
        node_id=0,
        instance_batch_id="batch-1",
        count=2,
        physical_quantity=2.0,
        unit_price=1.5,
        amount=3.0,
        price_version="debug-v1",
    )
    ledger.append_all((entry,))
    assert ledger.entries == (entry,)
    assert all(item.cost_type != "retention" for item in ledger.entries)
```

- [ ] **Step 2: 运行集成测试并确认 RED**

Run:

```powershell
python -m pytest tests/test_phase_a_orchestration.py tests/test_cost_ledger.py -q
```

Expected: `PhaseASlotCoordinator` 和 `CostLedger` 不存在，测试失败。

- [ ] **Step 3: 实现唯一成本账本和编排调用顺序**

```python
# src/cost_ledger.py
from dataclasses import dataclass


@dataclass(frozen=True)
class CostLedgerEntry:
    slot: int
    cost_type: str
    function_id: int
    node_id: int
    instance_batch_id: str
    count: int
    physical_quantity: float
    unit_price: float
    amount: float
    price_version: str


class CostLedger:
    def __init__(self) -> None:
        self._entries: tuple[CostLedgerEntry, ...] = ()

    @property
    def entries(self) -> tuple[CostLedgerEntry, ...]:
        return self._entries

    def append_all(self, entries: tuple[CostLedgerEntry, ...]) -> None:
        # 先完整校验，再一次性替换；任何非法记录都不能部分追加。
        if any(entry.amount < 0.0 or entry.cost_type == "retention" for entry in entries):
            raise ValueError("成本记录非法或发生保留费重复计费。")
        self._entries = (*self._entries, *entries)

    @staticmethod
    def running_entries(
        *,
        slot: int,
        slot_seconds: float,
        active_counts: dict[tuple[int, int], int],
        running_prices: dict[tuple[int, int], float],
        price_version: str,
    ) -> tuple[CostLedgerEntry, ...]:
        return tuple(
            CostLedgerEntry(
                slot=slot,
                cost_type="running",
                function_id=function_id,
                node_id=node_id,
                instance_batch_id="",
                count=count,
                physical_quantity=count * slot_seconds,
                unit_price=running_prices[(function_id, node_id)],
                amount=count * slot_seconds * running_prices[(function_id, node_id)],
                price_version=price_version,
            )
            for (function_id, node_id), count in sorted(active_counts.items())
            if count > 0
        )
```

生命周期计划提交返回不可变 `billing_events`：每个新建批次一条 `deployment`，非零/零冷启动都按配置生成一条 `cold`；复用实例不生成这两种事件，故障不生成退款。编排层在生命周期提交成功后把事件转为账本记录；在故障处理与慢计划提交完成后，根据当前 `STARTING + WARM` 数量为区间 `[tΔt,(t+1)Δt)` 追加 `running`，不产生独立 `retention` 金额。

```python
# src/orchestration_core.py
from dataclasses import dataclass

from src.failure_process import FailureSnapshot, MarkovFailureProcess
from src.instance_lifecycle import InstanceLifecycleManager, LifecycleSnapshot


@dataclass(frozen=True)
class PhaseASlotResult:
    failure_snapshot: FailureSnapshot
    lifecycle_snapshot: LifecycleSnapshot


class PhaseASlotCoordinator:
    def __init__(
        self,
        failure_process: MarkovFailureProcess,
        lifecycle: InstanceLifecycleManager,
    ) -> None:
        self.failure_process = failure_process
        self.lifecycle = lifecycle

    def begin_slot(self, current_slot: int) -> PhaseASlotResult:
        # 时隙边界先把 ready_slot 已到的实例变温，再读取本时隙故障。
        self.lifecycle.advance_to_slot(current_slot)
        failure = self.failure_process.state_for_slot(current_slot)
        lifecycle = self.lifecycle.apply_failure_snapshot(failure)
        return PhaseASlotResult(failure, lifecycle)
```

README 只补充当前已完成阶段 A 的边界和运行核心测试命令，不描述尚未实现的 B–E 为已完成。

- [ ] **Step 4: 运行阶段 A 核心套件**

Run:

```powershell
python -m pytest tests/test_orchestration_config.py tests/test_config.py tests/test_entities.py tests/test_topology.py tests/test_reliability.py tests/test_failure_process.py tests/test_risk_aware_failure_process.py tests/test_instance_lifecycle.py tests/test_cost_ledger.py tests/test_phase_a_orchestration.py tests/test_active_algorithm_scope.py tests/test_slow_timescale_execution_core.py tests/test_fast_slot_executor.py tests/test_simulator.py tests/test_dppo_slow_timescale_env.py -q
```

Expected: 全部通过；不运行与阶段 A 无关的训练、教师、评估和大规模仿真测试。

- [ ] **Step 5: 执行静态范围检查**

Run:

```powershell
git grep -n -E "include_onboard|onboard_cpu_capacity|onboard_memory_mb|onboard_reliability|InfrastructureState|domain_recovery_probability_per_slot|node_recovery_probability_per_slot" -- src tests configs
git diff --check
git status --short
```

Expected: 第一条无输出；`git diff --check` 成功；工作区只包含阶段 A 预期文件。

- [ ] **Step 6: 提交并推送阶段 A**

```powershell
git add src/cost_ledger.py src/orchestration_core.py tests/test_cost_ledger.py tests/test_phase_a_orchestration.py README.md
git commit -m "feat: complete phase A orchestration foundation"
git push origin codex/dppo-main-algorithm
```

完成后停止，不提前实现阶段 B 队列、阶段 C 凸优化、阶段 D 安全解码或阶段 E 训练；先向用户报告阶段 A 的文件、作用和核心测试结果，供 review。
