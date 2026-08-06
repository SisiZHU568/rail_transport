# Fast Feasibility Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在执行 SFC 前检查并修复 CPU、内存、副本数量、运行节点和精确可靠性约束，无法修复时拒绝请求并输出可复现实验指标。

**Architecture:** 将现有审计逻辑提取为独立 `SlotConstraintAuditor`，将重复快层路由提取为纯函数，再由 `FastFeasibilityOptimizer` 在慢层副本数量和备用模式内进行确定性离散搜索。模拟器只负责编排“初始决策—初审—修复—复审—执行或拒绝”，全部成本和统计使用最终方案。

**Tech Stack:** Python 3.10+、dataclasses、itertools、NumPy、pytest、现有 `FaultDomainReliabilityModel` 与 `LinearRailTopology`。

---

## 执行约定

所有测试命令在仓库根目录执行。Windows 环境先运行：

```powershell
$testTemp = 'C:\Users\Administrator\.codex\.chatgpt-projects\g-p-6a6ff4453abc819184ee35997a150260\tmp\pytest-rail-transport'
New-Item -ItemType Directory -Force -Path $testTemp | Out-Null
$env:TEMP = $testTemp
$env:TMP = $testTemp
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONPATH = '.'
```

使用 `-c NUL -p no:cacheprovider`，避免 Windows 读取 UTF-8 `pytest.ini` 时的编码问题，也避免在仓库中生成缓存文件。

每个任务都遵循 RED—GREEN—REFACTOR：先写一个会因缺少目标行为而失败的测试，确认失败原因正确，再写最少实现。每个任务通过后单独提交并推送到 `codex/fast-feasibility-repair`。

## 文件职责

- Create: `src/constraint_audit.py`：唯一的时隙资源、副本与可靠性审计实现。
- Create: `src/fast_optimizer.py`：确定性快层可行性修复器与结果对象。
- Delete: `src/fast_optimizor.py`：删除未被引用的空文件并修正拼写。
- Modify: `src/entities.py`：补充副本数量违规字段。
- Modify: `src/two_timescale_control.py`：提供公共快层路由函数，两个现有控制器复用它。
- Modify: `src/two_timescale_simulator.py`：接入执行前修复、最终方案执行和汇总指标。
- Modify: `src/two_timescale_monte_carlo.py`：传播新增汇总指标。
- Modify: `run_two_timescale_comparison.py`：打印并保存修复指标。
- Modify: `run_two_timescale_monte_carlo.py`：打印并保存多随机种子修复指标。
- Create: `tests/test_constraint_audit.py`：独立审计单元测试。
- Create: `tests/test_fast_optimizer.py`：修复器搜索、失败与确定性测试。
- Modify: `tests/test_entities.py`、`tests/test_two_timescale_control.py`、`tests/test_two_timescale_simulator.py`、`tests/test_two_timescale_monte_carlo.py`：兼容性和集成测试。

### Task 1: 提取唯一的约束审计器

**Files:**
- Create: `src/constraint_audit.py`
- Modify: `src/entities.py:306-339`
- Modify: `tests/test_entities.py:222-249`
- Create: `tests/test_constraint_audit.py`

- [ ] **Step 1: 写副本数量和资源审计的失败测试**

创建 `tests/test_constraint_audit.py`，先加入以下完整夹具和两个测试：

```python
from src.constraint_audit import SlotConstraintAuditor
from src.entities import (
    EdgeNode,
    NodeType,
    ServerlessFunction,
    ServicePriority,
    SFCType,
)
from src.reliability import FaultDomainReliabilityModel
from src.topology import LinearRailTopology, TracksideSite


def build_auditor() -> SlotConstraintAuditor:
    topology = LinearRailTopology(
        sites=[
            TracksideSite(
                node=EdgeNode(
                    node_id=node_id,
                    name=f"MEC-{node_id + 1}",
                    node_type=NodeType.TRACKSIDE,
                    cpu_capacity=100.0,
                    memory_capacity_mb=1000.0,
                    reliability=0.99,
                    fault_domain=node_id,
                ),
                position_m=float(node_id * 1000),
                coverage_radius_m=1200.0,
            )
            for node_id in range(3)
        ]
    )
    functions = [
        ServerlessFunction(
            function_id=function_id,
            name=f"测试函数{function_id}",
            memory_mb=600.0,
            cpu_cycles_per_request=60.0,
            image_size_mb=10.0,
            warm_exec_time_ms=10.0,
            cold_start_time_ms=100.0,
            output_ratio=1.0,
        )
        for function_id in (0, 1)
    ]
    sfc = SFCType(
        sfc_id=0,
        name="审计测试SFC",
        function_ids=[0, 1],
        deadline_ms=500.0,
        reliability_target=0.90,
        priority=ServicePriority.CRITICAL,
    )
    reliability_model = FaultDomainReliabilityModel(
        topology=topology,
        fault_domain_availability={
            0: 0.999,
            1: 0.999,
            2: 0.999,
        },
        minimum_distinct_fault_domains=1,
    )
    return SlotConstraintAuditor(
        functions=functions,
        sfc=sfc,
        topology=topology,
        reliability_model=reliability_model,
    )


def test_expected_replica_count_is_a_hard_constraint() -> None:
    audit = build_auditor().audit(
        request_count=1,
        expected_replica_count=2,
        candidate_map={0: (0,), 1: (1,)},
        selected_execution_node_ids=(0, 1),
        request_success=True,
        function_hot_node_ids={0: (0,), 1: (1,)},
        cold_activated_pairs=set(),
    )

    assert audit.replica_count_violation_function_ids == (0, 1)
    assert audit.replica_plan_valid is False
    assert audit.all_constraints_met is False
    assert any("副本数量" in reason for reason in audit.violation_reasons)


def test_same_node_resource_demand_is_accumulated() -> None:
    auditor = build_auditor()
    audit = auditor.audit(
        request_count=1,
        expected_replica_count=1,
        candidate_map={0: (0,), 1: (0,)},
        selected_execution_node_ids=(0, 0),
        request_success=True,
        function_hot_node_ids={0: (0,), 1: (0,)},
        cold_activated_pairs=set(),
    )

    assert audit.node_cpu_demand == {0: 120.0}
    assert audit.node_memory_demand_mb == {0: 1200.0}
    assert audit.cpu_violation_node_ids == (0,)
    assert audit.memory_violation_node_ids == (0,)
    assert auditor.resource_constraints_met(
        request_count=1,
        selected_execution_node_ids=(0, 0),
        request_success=True,
        function_hot_node_ids={0: (0,), 1: (0,)},
        cold_activated_pairs=set(),
    ) is False


def test_capacity_equal_to_demand_is_still_feasible() -> None:
    auditor = build_auditor()
    auditor.node_map[0].cpu_capacity = 120.0
    auditor.node_map[0].memory_capacity_mb = 1200.0

    audit = auditor.audit(
        request_count=1,
        expected_replica_count=1,
        candidate_map={0: (0,), 1: (0,)},
        selected_execution_node_ids=(0, 0),
        request_success=True,
        function_hot_node_ids={0: (0,), 1: (0,)},
        cold_activated_pairs=set(),
    )

    assert audit.cpu_violation_node_ids == ()
    assert audit.memory_violation_node_ids == ()
    assert audit.resource_constraints_met is True


def test_missing_function_is_reported_as_invalid_plan() -> None:
    audit = build_auditor().audit(
        request_count=1,
        expected_replica_count=1,
        candidate_map={0: (0,)},
        selected_execution_node_ids=(0,),
        request_success=True,
        function_hot_node_ids={0: (0,)},
        cold_activated_pairs=set(),
    )

    assert audit.missing_function_ids == (1,)
    assert audit.replica_plan_valid is False
    assert audit.exact_sfc_reliability is None
```

- [ ] **Step 2: 运行测试并确认按预期失败**

Run:

```powershell
D:\Anaconda3\python.exe -m pytest -q -p no:cacheprovider -c NUL tests/test_constraint_audit.py
```

Expected: collection 失败，原因是 `src.constraint_audit` 尚不存在。这个失败证明测试确实依赖待实现的新模块。

- [ ] **Step 3: 给审计结果增加副本数量违规字段**

在 `SlotConstraintAudit` 的 `violation_reasons` 后增加默认空元组，使旧构造位置在迁移期间仍可运行：

```python
    # 实际副本数与慢层许可数量不一致的函数编号。
    replica_count_violation_function_ids: tuple[int, ...] = ()
```

在 `tests/test_entities.py::test_constraint_audit_preserves_diagnostic_details` 的构造参数中加入：

```python
        replica_count_violation_function_ids=(0,),
```

并增加断言：

```python
    assert audit.replica_count_violation_function_ids == (0,)
```

- [ ] **Step 4: 创建独立审计器并复用同一资源公式**

创建 `src/constraint_audit.py`。把 `src/two_timescale_simulator.py:538-800` 的现有审计主体移动到 `SlotConstraintAuditor.audit`，并按以下接口组织。资源计算必须只存在于 `_calculate_resource_demands`，`audit` 和修复器的提前剪枝都复用它：

```python
"""快时隙资源、副本计划和可靠性硬约束审计。"""

from src.entities import (
    ServerlessFunction,
    SFCType,
    SlotConstraintAudit,
)
from src.reliability import FaultDomainReliabilityModel
from src.topology import LinearRailTopology


class SlotConstraintAuditor:
    """只计算约束结果，不修改部署或请求状态。"""

    def __init__(
        self,
        functions: list[ServerlessFunction],
        sfc: SFCType,
        topology: LinearRailTopology,
        reliability_model: FaultDomainReliabilityModel,
    ) -> None:
        self.functions = list(functions)
        self.function_map = {
            function.function_id: function
            for function in functions
        }
        if len(self.function_map) != len(functions):
            raise ValueError("Serverless函数编号不能重复。")
        if any(
            function_id not in self.function_map
            for function_id in sfc.function_ids
        ):
            raise KeyError("SFC引用了未提供的Serverless函数。")

        self.sfc = sfc
        self.topology = topology
        self.reliability_model = reliability_model
        self.node_map = {
            site.node.node_id: site.node
            for site in topology.sites
        }

    def _calculate_resource_demands(
        self,
        request_count: int,
        selected_execution_node_ids: tuple[int, ...],
        request_success: bool | None,
        function_hot_node_ids: dict[int, tuple[int, ...]],
        cold_activated_pairs: set[tuple[int, int]],
    ) -> tuple[dict[int, float], dict[int, float]]:
        if request_count < 0:
            raise ValueError("请求数量不能小于0。")

        active_pairs = {
            (function_id, node_id)
            for function_id, node_ids in function_hot_node_ids.items()
            for node_id in node_ids
        }
        active_pairs.update(cold_activated_pairs)

        node_memory_demand_mb: dict[int, float] = {}
        for function_id, node_id in active_pairs:
            node_memory_demand_mb[node_id] = (
                node_memory_demand_mb.get(node_id, 0.0)
                + self.function_map[function_id].memory_mb
            )

        node_cpu_demand: dict[int, float] = {}
        if request_success is True:
            for function_id, node_id in zip(
                self.sfc.function_ids,
                selected_execution_node_ids,
            ):
                node_cpu_demand[node_id] = (
                    node_cpu_demand.get(node_id, 0.0)
                    + self.function_map[function_id].cpu_demand(
                        request_count
                    )
                )

        return node_cpu_demand, node_memory_demand_mb

    def resource_constraints_met(
        self,
        request_count: int,
        selected_execution_node_ids: tuple[int, ...],
        request_success: bool | None,
        function_hot_node_ids: dict[int, tuple[int, ...]],
        cold_activated_pairs: set[tuple[int, int]],
    ) -> bool:
        """供搜索剪枝使用；公式与完整审计完全相同。"""
        cpu, memory = self._calculate_resource_demands(
            request_count=request_count,
            selected_execution_node_ids=selected_execution_node_ids,
            request_success=request_success,
            function_hot_node_ids=function_hot_node_ids,
            cold_activated_pairs=cold_activated_pairs,
        )
        referenced_node_ids = set(cpu) | set(memory)
        if any(node_id not in self.node_map for node_id in referenced_node_ids):
            return False
        return all(
            self.node_map[node_id].has_sufficient_capacity(
                cpu_demand=cpu.get(node_id, 0.0),
                memory_demand_mb=memory.get(node_id, 0.0),
            )
            for node_id in referenced_node_ids
        )

    def audit(
        self,
        request_count: int,
        expected_replica_count: int,
        candidate_map: dict[int, tuple[int, ...]],
        selected_execution_node_ids: tuple[int, ...],
        request_success: bool | None,
        function_hot_node_ids: dict[int, tuple[int, ...]],
        cold_activated_pairs: set[tuple[int, int]],
    ) -> SlotConstraintAudit:
        if expected_replica_count <= 0:
            raise ValueError("期望副本数量必须大于0。")

        required_ids = set(self.sfc.function_ids)
        supplied_ids = set(candidate_map)
        missing_ids = tuple(sorted(required_ids - supplied_ids))
        extra_ids = tuple(sorted(supplied_ids - required_ids))
        invalid_node_ids: set[int] = set()
        count_violation_ids: list[int] = []
        reasons: list[str] = []
        replica_plan_valid = not (missing_ids or extra_ids)

        if missing_ids:
            reasons.append(f"副本计划缺少函数{list(missing_ids)}。")
        if extra_ids:
            reasons.append(f"副本计划包含额外函数{list(extra_ids)}。")

        for function_id in self.sfc.function_ids:
            node_ids = candidate_map.get(function_id, ())
            if len(node_ids) != expected_replica_count:
                count_violation_ids.append(function_id)
                replica_plan_valid = False
                reasons.append(
                    f"函数{function_id}的副本数量{len(node_ids)}"
                    f"不等于慢层要求{expected_replica_count}。"
                )
            if len(node_ids) != len(set(node_ids)):
                replica_plan_valid = False
                reasons.append(f"函数{function_id}存在重复副本节点。")
            for node_id in node_ids:
                if node_id not in self.node_map:
                    invalid_node_ids.add(node_id)
                    replica_plan_valid = False

        if invalid_node_ids:
            reasons.append(
                f"副本计划引用未知节点{sorted(invalid_node_ids)}。"
            )

        cpu, memory = self._calculate_resource_demands(
            request_count=request_count,
            selected_execution_node_ids=selected_execution_node_ids,
            request_success=request_success,
            function_hot_node_ids=function_hot_node_ids,
            cold_activated_pairs=cold_activated_pairs,
        )
        cpu_violations = tuple(sorted(
            node_id
            for node_id, demand in cpu.items()
            if node_id not in self.node_map
            or demand > self.node_map[node_id].cpu_capacity
        ))
        memory_violations = tuple(sorted(
            node_id
            for node_id, demand in memory.items()
            if node_id not in self.node_map
            or demand > self.node_map[node_id].memory_capacity_mb
        ))

        for node_id in cpu_violations:
            if node_id in self.node_map:
                reasons.append(
                    f"节点{node_id}的CPU需求{cpu[node_id]:.3f}超过容量"
                    f"{self.node_map[node_id].cpu_capacity:.3f}。"
                )
        for node_id in memory_violations:
            if node_id in self.node_map:
                reasons.append(
                    f"节点{node_id}的内存需求{memory[node_id]:.3f}MB超过容量"
                    f"{self.node_map[node_id].memory_capacity_mb:.3f}MB。"
                )

        exact_reliability: float | None = None
        reliability_target_met = False
        if replica_plan_valid:
            reliability = self.reliability_model.evaluate_sfc(
                sfc=self.sfc,
                function_replica_node_ids=candidate_map,
            )
            exact_reliability = reliability.exact_shared_failure_availability
            reliability_target_met = reliability.target_met
            if not reliability_target_met:
                reasons.append(
                    f"精确SFC可靠性{exact_reliability:.6f}未达到目标"
                    f"{self.sfc.reliability_target:.6f}，"
                    "或副本未满足故障域隔离要求。"
                )

        resource_met = not (cpu_violations or memory_violations)
        return SlotConstraintAudit(
            node_cpu_demand=cpu,
            node_memory_demand_mb=memory,
            cpu_violation_node_ids=cpu_violations,
            memory_violation_node_ids=memory_violations,
            invalid_replica_node_ids=tuple(sorted(invalid_node_ids)),
            missing_function_ids=missing_ids,
            exact_sfc_reliability=exact_reliability,
            reliability_target=self.sfc.reliability_target,
            reliability_target_met=reliability_target_met,
            resource_constraints_met=resource_met,
            replica_plan_valid=replica_plan_valid,
            all_constraints_met=(
                resource_met
                and replica_plan_valid
                and reliability_target_met
            ),
            violation_reasons=tuple(reasons),
            replica_count_violation_function_ids=tuple(
                sorted(count_violation_ids)
            ),
        )
```

- [ ] **Step 5: 运行新审计测试和实体测试**

Run:

```powershell
D:\Anaconda3\python.exe -m pytest -q -p no:cacheprovider -c NUL tests/test_constraint_audit.py tests/test_entities.py
```

Expected: 两个文件全部通过，且无 warning/error。

- [ ] **Step 6: 提交并推送 Task 1**

```powershell
git add src/entities.py src/constraint_audit.py tests/test_entities.py tests/test_constraint_audit.py
git commit -m "feat: extract slot constraint auditor"
git push origin codex/fast-feasibility-repair
```

### Task 2: 提取公共快层路由函数

**Files:**
- Modify: `src/two_timescale_control.py:244-658,829-939`
- Modify: `tests/test_two_timescale_control.py`

- [ ] **Step 1: 写“修复后的新主节点需要冷启动”失败测试**

在 `tests/test_two_timescale_control.py` 导入 `build_fast_decision_for_plan`，并加入：

```python
def test_repaired_primary_that_was_not_hot_requires_cold_start() -> None:
    state = FastTimescaleState(
        time_slot=1,
        serving_mec=0,
        remaining_dwell_time_s=10.0,
        request_count=1,
        function_ids=(0,),
        candidate_node_ids={0: (1,)},
        operational_node_ids=frozenset({0, 1}),
    )

    decision = build_fast_decision_for_plan(
        state=state,
        standby_mode=StandbyMode.SINGLE,
        backup_activation_triggered=False,
        previously_hot_node_ids={0: (0,)},
    )

    assert decision.selected_execution_node_ids == (1,)
    assert decision.cold_start_function_ids == (0,)
    assert decision.request_success is True
```

- [ ] **Step 2: 运行目标测试并确认失败**

Run:

```powershell
D:\Anaconda3\python.exe -m pytest -q -p no:cacheprovider -c NUL tests/test_two_timescale_control.py::test_repaired_primary_that_was_not_hot_requires_cold_start
```

Expected: collection 失败，原因是公共函数尚未定义。

- [ ] **Step 3: 实现公共纯函数**

在 `FastTimescaleDecision` 后加入下面的公共函数。中文注释必须保留，便于 review 主备和冷启动语义：

```python
def build_fast_decision_for_plan(
    state: FastTimescaleState,
    standby_mode: StandbyMode,
    backup_activation_triggered: bool,
    previously_hot_node_ids: (
        dict[int, tuple[int, ...]] | None
    ) = None,
) -> FastTimescaleDecision:
    """根据给定副本方案生成无副作用的快层路由结果。"""

    function_hot_node_ids: dict[int, tuple[int, ...]] = {}
    selected_execution_node_ids: list[int] = []
    failover_function_ids: list[int] = []
    cold_start_function_ids: list[int] = []
    unavailable_function_ids: list[int] = []

    for function_id in state.function_ids:
        all_candidate_nodes = state.candidate_node_ids[function_id]
        usable_candidate_nodes = (
            (all_candidate_nodes[0],)
            if standby_mode is StandbyMode.SINGLE
            else all_candidate_nodes
        )
        primary_node_id = usable_candidate_nodes[0]

        # HOT 或已经触发备用激活时，计划中的全部副本都占用活动内存。
        if (
            standby_mode is StandbyMode.HOT
            or backup_activation_triggered
        ):
            hot_node_ids = tuple(usable_candidate_nodes)
        else:
            hot_node_ids = (primary_node_id,)
        function_hot_node_ids[function_id] = hot_node_ids

        if state.request_count == 0:
            continue

        operational_candidates = [
            node_id
            for node_id in usable_candidate_nodes
            if node_id in state.operational_node_ids
        ]
        if not operational_candidates:
            unavailable_function_ids.append(function_id)
            continue

        selected_node_id = operational_candidates[0]
        selected_execution_node_ids.append(selected_node_id)
        if selected_node_id != primary_node_id:
            failover_function_ids.append(function_id)

        # 普通控制器使用当前计划的温实例；修复器使用修复前真实温实例。
        known_hot_node_ids = (
            hot_node_ids
            if previously_hot_node_ids is None
            else previously_hot_node_ids.get(function_id, ())
        )
        if selected_node_id not in known_hot_node_ids:
            cold_start_function_ids.append(function_id)

    if state.request_count == 0:
        request_success: bool | None = None
    elif unavailable_function_ids:
        request_success = False
        selected_execution_node_ids = []
    else:
        request_success = True

    return FastTimescaleDecision(
        function_hot_node_ids=function_hot_node_ids,
        selected_execution_node_ids=tuple(selected_execution_node_ids),
        backup_activation_triggered=backup_activation_triggered,
        failover_function_ids=tuple(failover_function_ids),
        cold_start_function_ids=tuple(cold_start_function_ids),
        unavailable_function_ids=tuple(unavailable_function_ids),
        request_success=request_success,
    )
```

- [ ] **Step 4: 让两个控制器复用公共函数**

把规则控制器 `get_fast_decision` 在算出 `backup_activation_triggered` 后的重复路由代码替换为：

```python
        return build_fast_decision_for_plan(
            state=state,
            standby_mode=slow_decision.standby_mode,
            backup_activation_triggered=backup_activation_triggered,
        )
```

把固定控制器 `get_fast_decision` 在算出自己的 `backup_activation_triggered` 后替换为：

```python
        return build_fast_decision_for_plan(
            state=state,
            standby_mode=self.standby_mode,
            backup_activation_triggered=backup_activation_triggered,
        )
```

两个控制器仍各自负责判断是否进入切换激活窗口，公共函数只负责实例状态和路由，不能读取控制器内部状态。

- [ ] **Step 5: 运行全部快层控制测试**

Run:

```powershell
D:\Anaconda3\python.exe -m pytest -q -p no:cacheprovider -c NUL tests/test_two_timescale_control.py
```

Expected: 新测试和所有现有 SINGLE/COLD/HOT/故障接管测试全部通过。

- [ ] **Step 6: 提交并推送 Task 2**

```powershell
git add src/two_timescale_control.py tests/test_two_timescale_control.py
git commit -m "refactor: share fast decision routing"
git push origin codex/fast-feasibility-repair
```

### Task 3: 实现确定性快层可行性修复器

**Files:**
- Create: `src/fast_optimizer.py`
- Delete: `src/fast_optimizor.py`
- Create: `tests/test_fast_optimizer.py`

- [ ] **Step 1: 写无须修复、资源修复、可靠性修复和不可修复测试**

创建 `tests/test_fast_optimizer.py`，使用以下完整导入、夹具和执行辅助函数：

```python
from src.constraint_audit import SlotConstraintAuditor
from src.entities import (
    EdgeNode,
    NodeType,
    ServerlessFunction,
    ServicePriority,
    SFCType,
)
from src.fast_optimizer import (
    FastFeasibilityOptimizer,
    FastOptimizationResult,
)
from src.reliability import FaultDomainReliabilityModel
from src.topology import LinearRailTopology, TracksideSite
from src.two_timescale_control import (
    FastTimescaleState,
    SlowTimescaleDecision,
    StandbyMode,
    build_fast_decision_for_plan,
)


def optimize_plan(
    *,
    auditor: SlotConstraintAuditor,
    topology: LinearRailTopology,
    state: FastTimescaleState,
    slow_decision: SlowTimescaleDecision,
) -> FastOptimizationResult:
    initial_decision = build_fast_decision_for_plan(
        state=state,
        standby_mode=slow_decision.standby_mode,
        backup_activation_triggered=False,
    )
    cold_pairs = {
        (function_id, initial_decision.selected_execution_node_ids[index])
        for index, function_id in enumerate(state.function_ids)
        if function_id in initial_decision.cold_start_function_ids
        and initial_decision.request_success is True
    }
    initial_audit = auditor.audit(
        request_count=state.request_count,
        expected_replica_count=slow_decision.replica_count,
        candidate_map=state.candidate_node_ids,
        selected_execution_node_ids=(
            initial_decision.selected_execution_node_ids
        ),
        request_success=initial_decision.request_success,
        function_hot_node_ids=initial_decision.function_hot_node_ids,
        cold_activated_pairs=cold_pairs,
    )
    optimizer = FastFeasibilityOptimizer(
        functions=auditor.functions,
        sfc=auditor.sfc,
        topology=topology,
        auditor=auditor,
        return_result_to_source=True,
    )
    return optimizer.optimize(
        state=state,
        slow_decision=slow_decision,
        initial_decision=initial_decision,
        initial_audit=initial_audit,
    )
```

继续加入完整的场景构造函数。低可靠性资源测试使用一个故障域要求，可靠性修复测试使用两个故障域要求：

```python
def optimize_plan_for_test(
    *,
    replica_count: int,
    standby_mode: StandbyMode,
    candidate_map: dict[int, tuple[int, ...]],
    request_count: int,
    minimum_distinct_fault_domains: int,
    fault_domains: tuple[int, ...] = (0, 1, 2, 3, 4),
    operational_node_ids: set[int] | None = None,
    function_memory_mb: float = 600.0,
    cpu_per_request: float = 60.0,
) -> tuple[FastOptimizationResult, dict[int, int]]:
    node_fault_domains = {
        node_id: fault_domain
        for node_id, fault_domain in enumerate(fault_domains)
    }
    topology = LinearRailTopology(
        sites=[
            TracksideSite(
                node=EdgeNode(
                    node_id=node_id,
                    name=f"MEC-{node_id + 1}",
                    node_type=NodeType.TRACKSIDE,
                    cpu_capacity=100.0,
                    memory_capacity_mb=1000.0,
                    reliability=0.99,
                    fault_domain=node_fault_domains[node_id],
                ),
                position_m=float(node_id * 1000),
                coverage_radius_m=1200.0,
            )
            for node_id in range(len(fault_domains))
        ]
    )
    functions = [
        ServerlessFunction(
            function_id=function_id,
            name=f"测试函数{function_id}",
            memory_mb=function_memory_mb,
            cpu_cycles_per_request=cpu_per_request,
            image_size_mb=10.0,
            warm_exec_time_ms=10.0,
            cold_start_time_ms=100.0,
            output_ratio=1.0,
        )
        for function_id in (0, 1)
    ]
    sfc = SFCType(
        sfc_id=0,
        name="修复测试SFC",
        function_ids=[0, 1],
        deadline_ms=500.0,
        reliability_target=0.90,
        priority=ServicePriority.CRITICAL,
    )
    reliability_model = FaultDomainReliabilityModel(
        topology=topology,
        fault_domain_availability={
            fault_domain: 0.999
            for fault_domain in set(fault_domains)
        },
        minimum_distinct_fault_domains=(
            minimum_distinct_fault_domains
        ),
    )
    auditor = SlotConstraintAuditor(
        functions=functions,
        sfc=sfc,
        topology=topology,
        reliability_model=reliability_model,
    )
    state = FastTimescaleState(
        time_slot=0,
        serving_mec=0,
        remaining_dwell_time_s=10.0,
        request_count=request_count,
        function_ids=(0, 1),
        candidate_node_ids=candidate_map,
        operational_node_ids=frozenset(
            set(range(len(fault_domains)))
            if operational_node_ids is None
            else operational_node_ids
        ),
    )
    slow_decision = SlowTimescaleDecision(
        decision_slot=0,
        valid_until_slot=9,
        use_redundancy=replica_count > 1,
        replica_count=replica_count,
        standby_mode=standby_mode,
        reason="测试慢动作",
    )
    return (
        optimize_plan(
            auditor=auditor,
            topology=topology,
            state=state,
            slow_decision=slow_decision,
        ),
        node_fault_domains,
    )


def build_overload_result() -> FastOptimizationResult:
    result, _ = optimize_plan_for_test(
        replica_count=1,
        standby_mode=StandbyMode.SINGLE,
        candidate_map={0: (0,), 1: (0,)},
        request_count=1,
        minimum_distinct_fault_domains=1,
    )
    return result


def test_feasible_initial_plan_is_returned_without_search() -> None:
    result, _ = optimize_plan_for_test(
        replica_count=1,
        standby_mode=StandbyMode.SINGLE,
        candidate_map={0: (0,), 1: (1,)},
        request_count=1,
        minimum_distinct_fault_domains=1,
    )
    assert result.attempted is False
    assert result.succeeded is None
    assert result.evaluated_candidate_count == 0
    assert result.function_replica_node_ids == {0: (0,), 1: (1,)}


def test_cpu_and_memory_overload_is_repaired_by_spreading_functions() -> None:
    result, _ = optimize_plan_for_test(
        replica_count=1,
        standby_mode=StandbyMode.SINGLE,
        candidate_map={0: (0,), 1: (0,)},
        request_count=1,
        minimum_distinct_fault_domains=1,
    )
    assert result.initial_audit.resource_constraints_met is False
    assert result.succeeded is True
    assert result.final_audit.all_constraints_met is True
    assert result.function_replica_node_ids == {0: (0,), 1: (1,)}


def test_same_domain_replicas_are_moved_across_fault_domains() -> None:
    result, node_fault_domains = optimize_plan_for_test(
        replica_count=2,
        standby_mode=StandbyMode.COLD,
        candidate_map={0: (0, 1), 1: (0, 1)},
        request_count=1,
        minimum_distinct_fault_domains=2,
        fault_domains=(0, 0, 1, 1, 2),
        function_memory_mb=100.0,
        cpu_per_request=10.0,
    )
    assert result.initial_audit.reliability_target_met is False
    assert result.succeeded is True
    assert result.final_audit.reliability_target_met is True
    for node_ids in result.function_replica_node_ids.values():
        assert len({node_fault_domains[node_id] for node_id in node_ids}) == 2


def test_unavailable_initial_path_is_repaired_to_operational_nodes() -> None:
    result, _ = optimize_plan_for_test(
        replica_count=1,
        standby_mode=StandbyMode.SINGLE,
        candidate_map={0: (0,), 1: (0,)},
        request_count=1,
        minimum_distinct_fault_domains=1,
        operational_node_ids={1, 2, 3, 4},
        function_memory_mb=100.0,
        cpu_per_request=10.0,
    )
    assert result.initial_audit.all_constraints_met is True
    assert result.succeeded is True
    assert result.decision.request_success is True
    assert set(result.decision.selected_execution_node_ids) <= {1, 2, 3, 4}


def test_single_replica_cannot_bypass_two_domain_requirement() -> None:
    result, _ = optimize_plan_for_test(
        replica_count=1,
        standby_mode=StandbyMode.SINGLE,
        candidate_map={0: (0,), 1: (1,)},
        request_count=1,
        minimum_distinct_fault_domains=2,
    )
    assert result.attempted is True
    assert result.succeeded is False
    assert result.decision.request_success is False
    assert result.decision.selected_execution_node_ids == ()
    assert "副本数量" in result.reason


def test_no_request_remains_none_when_repair_is_impossible() -> None:
    result, _ = optimize_plan_for_test(
        replica_count=1,
        standby_mode=StandbyMode.SINGLE,
        candidate_map={0: (0,), 1: (1,)},
        request_count=0,
        minimum_distinct_fault_domains=2,
    )
    assert result.succeeded is False
    assert result.decision.request_success is None


def test_too_few_operational_nodes_returns_normal_failure() -> None:
    result, _ = optimize_plan_for_test(
        replica_count=2,
        standby_mode=StandbyMode.HOT,
        candidate_map={0: (0, 1), 1: (0, 1)},
        request_count=1,
        minimum_distinct_fault_domains=2,
        operational_node_ids={0},
        function_memory_mb=100.0,
        cpu_per_request=10.0,
    )
    assert result.succeeded is False
    assert result.evaluated_candidate_count == 0
    assert "正常MEC" in result.reason


def test_every_node_resource_shortage_returns_failure() -> None:
    result, _ = optimize_plan_for_test(
        replica_count=1,
        standby_mode=StandbyMode.SINGLE,
        candidate_map={0: (0,), 1: (1,)},
        request_count=1,
        minimum_distinct_fault_domains=1,
        function_memory_mb=1200.0,
        cpu_per_request=10.0,
    )
    assert result.succeeded is False
    assert result.evaluated_candidate_count > 0
    assert result.decision.request_success is False


def test_repair_is_deterministic() -> None:
    first = build_overload_result()
    second = build_overload_result()
    assert first.function_replica_node_ids == second.function_replica_node_ids
    assert first.reason == second.reason
    assert first.evaluated_candidate_count == second.evaluated_candidate_count
```

- [ ] **Step 2: 运行新测试并确认失败**

Run:

```powershell
D:\Anaconda3\python.exe -m pytest -q -p no:cacheprovider -c NUL tests/test_fast_optimizer.py
```

Expected: collection 失败，原因是 `src.fast_optimizer` 尚不存在。

- [ ] **Step 3: 实现结果对象和失败语义**

创建 `src/fast_optimizer.py`，先定义结果对象、构造验证、冷启动实例映射和拒绝结果：

```python
"""慢层模板约束下的确定性快层可行性修复。"""

from dataclasses import dataclass, replace
from itertools import permutations, product

from src.constraint_audit import SlotConstraintAuditor
from src.entities import ServerlessFunction, SFCType, SlotConstraintAudit
from src.topology import LinearRailTopology
from src.two_timescale_control import (
    FastTimescaleDecision,
    FastTimescaleState,
    SlowTimescaleDecision,
    build_fast_decision_for_plan,
)


@dataclass(frozen=True)
class FastOptimizationResult:
    attempted: bool
    succeeded: bool | None
    function_replica_node_ids: dict[int, tuple[int, ...]]
    decision: FastTimescaleDecision
    initial_audit: SlotConstraintAudit
    final_audit: SlotConstraintAudit
    reason: str
    evaluated_candidate_count: int


class FastFeasibilityOptimizer:
    """搜索改动最少且满足全部硬约束的快层方案。"""

    def __init__(
        self,
        functions: list[ServerlessFunction],
        sfc: SFCType,
        topology: LinearRailTopology,
        auditor: SlotConstraintAuditor,
        return_result_to_source: bool = True,
    ) -> None:
        self.functions = list(functions)
        self.sfc = sfc
        self.topology = topology
        self.auditor = auditor
        self.return_result_to_source = return_result_to_source
        self.node_ids = tuple(
            sorted(site.node.node_id for site in topology.sites)
        )
        self.node_positions = {
            site.node.node_id: site.position_m
            for site in topology.sites
        }

    def _cold_activated_pairs(
        self,
        decision: FastTimescaleDecision,
    ) -> set[tuple[int, int]]:
        if decision.request_success is not True:
            return set()
        selected_by_function = dict(zip(
            self.sfc.function_ids,
            decision.selected_execution_node_ids,
        ))
        return {
            (function_id, selected_by_function[function_id])
            for function_id in decision.cold_start_function_ids
        }

    def _rejected_decision(
        self,
        state: FastTimescaleState,
        initial_decision: FastTimescaleDecision,
    ) -> FastTimescaleDecision:
        if state.request_count == 0:
            return initial_decision
        return replace(
            initial_decision,
            selected_execution_node_ids=(),
            failover_function_ids=(),
            cold_start_function_ids=(),
            request_success=False,
        )

    def _failure_result(
        self,
        state: FastTimescaleState,
        initial_decision: FastTimescaleDecision,
        initial_audit: SlotConstraintAudit,
        reason: str,
        evaluated_candidate_count: int,
    ) -> FastOptimizationResult:
        return FastOptimizationResult(
            attempted=True,
            succeeded=False,
            function_replica_node_ids=dict(state.candidate_node_ids),
            decision=self._rejected_decision(state, initial_decision),
            initial_audit=initial_audit,
            final_audit=initial_audit,
            reason=reason,
            evaluated_candidate_count=evaluated_candidate_count,
        )
```

- [ ] **Step 4: 实现稳定评分和完整搜索**

继续在类中加入以下方法。候选只使用正常节点，副本数量严格等于慢动作；先按稳定评分排序，再依次用共享审计器检查：

```python
    def _score(
        self,
        state: FastTimescaleState,
        candidate_map: dict[int, tuple[int, ...]],
        decision: FastTimescaleDecision,
    ) -> tuple[int, int, int, float, tuple[int, ...]]:
        initial_map = state.candidate_node_ids
        changed_function_count = sum(
            candidate_map[function_id] != initial_map[function_id]
            for function_id in state.function_ids
        )
        replaced_replica_count = sum(
            len(
                set(initial_map[function_id])
                - set(candidate_map[function_id])
            )
            for function_id in state.function_ids
        )
        path = [state.serving_mec]
        path.extend(decision.selected_execution_node_ids)
        if self.return_result_to_source and decision.selected_execution_node_ids:
            path.append(state.serving_mec)
        distance = sum(
            abs(
                self.node_positions[left]
                - self.node_positions[right]
            )
            for left, right in zip(path, path[1:])
        )
        flattened_ids = tuple(
            node_id
            for function_id in state.function_ids
            for node_id in candidate_map[function_id]
        )
        return (
            int(changed_function_count),
            int(replaced_replica_count),
            len(decision.cold_start_function_ids),
            float(distance),
            flattened_ids,
        )

    def optimize(
        self,
        state: FastTimescaleState,
        slow_decision: SlowTimescaleDecision,
        initial_decision: FastTimescaleDecision,
        initial_audit: SlotConstraintAudit,
    ) -> FastOptimizationResult:
        if slow_decision.replica_count <= 0:
            raise ValueError("慢层副本数量必须大于0。")

        execution_ready = (
            state.request_count == 0
            or initial_decision.request_success is True
        )
        if initial_audit.all_constraints_met and execution_ready:
            return FastOptimizationResult(
                attempted=False,
                succeeded=None,
                function_replica_node_ids=dict(state.candidate_node_ids),
                decision=initial_decision,
                initial_audit=initial_audit,
                final_audit=initial_audit,
                reason="初始方案满足全部硬约束，无需修复。",
                evaluated_candidate_count=0,
            )

        required_count = slow_decision.replica_count
        minimum_domains = (
            self.auditor.reliability_model.minimum_distinct_fault_domains
        )
        if required_count < minimum_domains:
            return self._failure_result(
                state=state,
                initial_decision=initial_decision,
                initial_audit=initial_audit,
                reason=(
                    f"慢层只允许{required_count}个副本，"
                    f"但可靠性模型至少要求{minimum_domains}个故障域，"
                    "快层不能擅自增加副本数量。"
                ),
                evaluated_candidate_count=0,
            )

        operational_ids = tuple(sorted(
            node_id
            for node_id in state.operational_node_ids
            if node_id in self.node_ids
        ))
        if len(operational_ids) < required_count:
            return self._failure_result(
                state=state,
                initial_decision=initial_decision,
                initial_audit=initial_audit,
                reason=(
                    f"当前只有{len(operational_ids)}个正常MEC，"
                    f"少于慢层要求的{required_count}个副本。"
                ),
                evaluated_candidate_count=0,
            )

        per_function_options = tuple(
            permutations(operational_ids, required_count)
        )
        ranked_candidates: list[tuple[
            tuple[int, int, int, float, tuple[int, ...]],
            dict[int, tuple[int, ...]],
            FastTimescaleDecision,
        ]] = []

        for node_tuples in product(
            per_function_options,
            repeat=len(state.function_ids),
        ):
            candidate_map = {
                function_id: tuple(node_ids)
                for function_id, node_ids in zip(
                    state.function_ids,
                    node_tuples,
                )
            }
            if candidate_map == state.candidate_node_ids:
                continue
            repaired_state = replace(
                state,
                candidate_node_ids=candidate_map,
            )
            decision = build_fast_decision_for_plan(
                state=repaired_state,
                standby_mode=slow_decision.standby_mode,
                backup_activation_triggered=(
                    initial_decision.backup_activation_triggered
                ),
                previously_hot_node_ids=(
                    initial_decision.function_hot_node_ids
                ),
            )
            ranked_candidates.append((
                self._score(state, candidate_map, decision),
                candidate_map,
                decision,
            ))

        evaluated_count = 0
        for _, candidate_map, decision in sorted(
            ranked_candidates,
            key=lambda item: item[0],
        ):
            evaluated_count += 1
            cold_pairs = self._cold_activated_pairs(decision)
            if not self.auditor.resource_constraints_met(
                request_count=state.request_count,
                selected_execution_node_ids=(
                    decision.selected_execution_node_ids
                ),
                request_success=decision.request_success,
                function_hot_node_ids=decision.function_hot_node_ids,
                cold_activated_pairs=cold_pairs,
            ):
                continue
            audit = self.auditor.audit(
                request_count=state.request_count,
                expected_replica_count=required_count,
                candidate_map=candidate_map,
                selected_execution_node_ids=(
                    decision.selected_execution_node_ids
                ),
                request_success=decision.request_success,
                function_hot_node_ids=decision.function_hot_node_ids,
                cold_activated_pairs=cold_pairs,
            )
            execution_ready = (
                state.request_count == 0
                or decision.request_success is True
            )
            if audit.all_constraints_met and execution_ready:
                changed_ids = [
                    function_id
                    for function_id in state.function_ids
                    if candidate_map[function_id]
                    != state.candidate_node_ids[function_id]
                ]
                return FastOptimizationResult(
                    attempted=True,
                    succeeded=True,
                    function_replica_node_ids=dict(candidate_map),
                    decision=decision,
                    initial_audit=initial_audit,
                    final_audit=audit,
                    reason=f"已修复函数{changed_ids}的副本位置。",
                    evaluated_candidate_count=evaluated_count,
                )

        return self._failure_result(
            state=state,
            initial_decision=initial_decision,
            initial_audit=initial_audit,
            reason="已检查全部候选方案，没有找到满足硬约束的方案。",
            evaluated_candidate_count=evaluated_count,
        )
```

- [ ] **Step 5: 删除空的错误拼写文件并运行修复器测试**

删除未被任何导入引用且内容为空的 `src/fast_optimizor.py`，然后运行：

```powershell
D:\Anaconda3\python.exe -m pytest -q -p no:cacheprovider -c NUL tests/test_fast_optimizer.py tests/test_constraint_audit.py tests/test_two_timescale_control.py
```

Expected: 修复器、审计器和公共路由测试全部通过；资源超限方案被分散，单副本不能绕过双故障域要求，相同输入结果一致。

- [ ] **Step 6: 提交并推送 Task 3**

```powershell
git add src/fast_optimizer.py src/fast_optimizor.py tests/test_fast_optimizer.py
git commit -m "feat: add deterministic fast feasibility optimizer"
git push origin codex/fast-feasibility-repair
```

### Task 4: 在模拟器执行前接入修复闭环

**Files:**
- Modify: `src/two_timescale_simulator.py:180-209,212-258,280-399,538-800,803-1208`
- Modify: `tests/test_two_timescale_simulator.py:82-187,330-432`

- [ ] **Step 1: 把现有“只记录”集成测试改为“修复后执行”失败测试**

将 `test_same_node_load_is_accumulated_and_overload_is_recorded` 改名为 `test_resource_overload_is_repaired_before_execution`。使用 `StandbyMode.HOT`，使慢层允许两个跨域副本；保持两个函数每个 5000 MB、60 CPU，并断言：

```python
    record = result.records[0]
    assert record.initial_constraint_audit.resource_constraints_met is False
    assert record.fast_repair_attempted is True
    assert record.fast_repair_succeeded is True
    assert record.constraint_audit.all_constraints_met is True
    assert record.request_success is True
    assert all(
        demand <= 8192.0
        for demand in record.constraint_audit.node_memory_demand_mb.values()
    )
```

再加入不可修复活动请求测试：

```python
def test_unrepairable_single_replica_request_is_rejected() -> None:
    controller = FixedModeTwoTimescaleController(
        standby_mode=StandbyMode.SINGLE,
        handover_hot_window_s=1.0,
    )
    result = build_test_simulator(
        controller=controller,
        risk=0.01,
        request_trace=[1],
        down_nodes_by_slot={},
    ).run()

    record = result.records[0]
    assert record.fast_repair_attempted is True
    assert record.fast_repair_succeeded is False
    assert record.request_success is False
    assert record.selected_execution_node_ids == ()
    assert record.end_to_end_delay_ms is None


def test_unrepairable_no_request_slot_is_not_counted_as_request_failure() -> None:
    controller = FixedModeTwoTimescaleController(
        standby_mode=StandbyMode.SINGLE,
        handover_hot_window_s=1.0,
    )
    result = build_test_simulator(
        controller=controller,
        risk=0.01,
        request_trace=[0],
        down_nodes_by_slot={},
    ).run()

    record = result.records[0]
    assert record.fast_repair_succeeded is False
    assert record.request_success is None
    assert result.summary.failed_batches == 0
```

- [ ] **Step 2: 运行三个集成测试并确认失败**

Run:

```powershell
D:\Anaconda3\python.exe -m pytest -q -p no:cacheprovider -c NUL tests/test_two_timescale_simulator.py -k "resource_overload_is_repaired or unrepairable"
```

Expected: 新字段不存在或原资源超限仍未被修复，测试按预期失败。

- [ ] **Step 3: 在时隙记录中加入修复前后信息**

将原“只记录、不改变结果”的注释替换为以下字段：

```python
    # 修复前的审计用于解释为什么触发快层搜索。
    initial_constraint_audit: SlotConstraintAudit

    # None 表示原方案可行，没有执行搜索。
    fast_repair_attempted: bool
    fast_repair_succeeded: bool | None
    fast_repair_reason: str
    fast_repair_evaluated_candidate_count: int

    # 最终采用方案的审计；修复失败时保留初始审计作为拒绝依据。
    constraint_audit: SlotConstraintAudit
```

- [ ] **Step 4: 在构造函数创建共享审计器和修复器**

导入：

```python
from src.constraint_audit import SlotConstraintAuditor
from src.fast_optimizer import FastFeasibilityOptimizer
```

在保存 `reliability_model` 后创建：

```python
        self.constraint_auditor = SlotConstraintAuditor(
            functions=self.functions,
            sfc=self.sfc,
            topology=self.topology,
            reliability_model=self.reliability_model,
        )
        self.fast_optimizer = FastFeasibilityOptimizer(
            functions=self.functions,
            sfc=self.sfc,
            topology=self.topology,
            auditor=self.constraint_auditor,
            return_result_to_source=self.return_result_to_source,
        )
```

注意：`self.return_result_to_source` 必须在创建修复器之前赋值。删除模拟器内完整的 `_audit_slot_constraints`，避免出现第二套公式。

- [ ] **Step 5: 增加预测冷启动实例辅助方法**

在 `_calculate_active_memory` 前增加：

```python
    def _cold_activated_pairs(
        self,
        decision: FastTimescaleDecision,
    ) -> set[tuple[int, int]]:
        """把冷启动函数编号映射到最终执行节点。"""
        if decision.request_success is not True:
            return set()
        selected_by_function = dict(zip(
            self.sfc.function_ids,
            decision.selected_execution_node_ids,
        ))
        return {
            (function_id, selected_by_function[function_id])
            for function_id in decision.cold_start_function_ids
        }
```

- [ ] **Step 6: 将 run() 重排为“先审计修复、后执行”**

在初始 `fast_decision` 生成后、任何 `execute_sfc_batch` 调用前加入：

```python
            initial_cold_pairs = self._cold_activated_pairs(
                fast_decision
            )
            initial_audit = self.constraint_auditor.audit(
                request_count=request_count,
                expected_replica_count=slow_decision.replica_count,
                candidate_map=candidate_map,
                selected_execution_node_ids=(
                    fast_decision.selected_execution_node_ids
                ),
                request_success=fast_decision.request_success,
                function_hot_node_ids=(
                    fast_decision.function_hot_node_ids
                ),
                cold_activated_pairs=initial_cold_pairs,
            )
            optimization = self.fast_optimizer.optimize(
                state=fast_state,
                slow_decision=slow_decision,
                initial_decision=fast_decision,
                initial_audit=initial_audit,
            )

            # 从这里开始，后续执行、成本和记录只使用最终方案。
            candidate_map = dict(
                optimization.function_replica_node_ids
            )
            fast_decision = optimization.decision
            constraint_audit = optimization.final_audit
            cold_activated_pairs = self._cold_activated_pairs(
                fast_decision
            )
            plan_change_count = self._count_plan_changes(
                previous_map=previous_candidate_map,
                current_map=candidate_map,
            )
```

删除原来在优化前计算 `plan_change_count` 的代码。保留现有执行时延计算，但删除执行之后向 `cold_activated_pairs` 逐项追加的循环，因为冷启动实例已经由最终决策提前计算。只有 `fast_decision.request_success is True` 时调用 `execute_sfc_batch`。

修复失败时最终决策为 False/None，因此不会执行；活动内存计算使用空的请求冷启动集合，只统计实际维持的温实例。

- [ ] **Step 7: 写入新增记录字段并使用最终方案**

创建 `TwoTimescaleSlotRecord` 时加入：

```python
                    initial_constraint_audit=initial_audit,
                    fast_repair_attempted=optimization.attempted,
                    fast_repair_succeeded=optimization.succeeded,
                    fast_repair_reason=optimization.reason,
                    fast_repair_evaluated_candidate_count=(
                        optimization.evaluated_candidate_count
                    ),
                    constraint_audit=constraint_audit,
```

确认以下现有字段全部来自最终 `candidate_map` 或最终 `fast_decision`：

```python
function_replica_node_ids
function_hot_node_ids
selected_execution_node_ids
failover_function_ids
cold_start_function_ids
request_success
active_memory_mb
replica_plan_changed_function_count
```

- [ ] **Step 8: 运行模拟器目标测试**

Run:

```powershell
D:\Anaconda3\python.exe -m pytest -q -p no:cacheprovider -c NUL tests/test_two_timescale_simulator.py tests/test_failure_risk_prediction.py
```

Expected: 可修复资源超限请求成功；单副本高可靠请求被拒绝；无请求保持 None；旧的故障接管和成本测试仍通过。

- [ ] **Step 9: 提交并推送 Task 4**

```powershell
git add src/two_timescale_simulator.py tests/test_two_timescale_simulator.py tests/test_failure_risk_prediction.py
git commit -m "feat: enforce feasibility before SFC execution"
git push origin codex/fast-feasibility-repair
```

### Task 5: 增加修复汇总指标和实验输出

**Files:**
- Modify: `src/two_timescale_simulator.py:212-258,1210-1504`
- Modify: `src/two_timescale_monte_carlo.py:50-112,197-347`
- Modify: `run_two_timescale_comparison.py:304-565`
- Modify: `run_two_timescale_monte_carlo.py:372-581`
- Modify: `tests/test_two_timescale_simulator.py`
- Modify: `tests/test_two_timescale_monte_carlo.py`

- [ ] **Step 1: 写汇总统计失败测试**

在 `tests/test_two_timescale_simulator.py` 加入：

```python
def test_summary_counts_repairs_and_constraint_rejections() -> None:
    controller = FixedModeTwoTimescaleController(
        standby_mode=StandbyMode.SINGLE,
        handover_hot_window_s=1.0,
    )
    result = build_test_simulator(
        controller=controller,
        risk=0.01,
        request_trace=[1],
        down_nodes_by_slot={},
    ).run()

    summary = result.summary
    assert summary.fast_repair_attempts == 1
    assert summary.fast_repair_successes == 0
    assert summary.fast_repair_failures == 1
    assert summary.fast_repair_success_rate == 0.0
    assert summary.constraint_rejected_batches == 1
    assert summary.failed_batches == 1
    assert summary.sla_violations == 1
```

在 `tests/test_two_timescale_monte_carlo.py` 的假仿真汇总中加入这些数值，并断言原始记录和方案统计都包含 `fast_repair_success_rate` 与 `constraint_rejected_batches`。

- [ ] **Step 2: 运行汇总测试并确认失败**

Run:

```powershell
D:\Anaconda3\python.exe -m pytest -q -p no:cacheprovider -c NUL tests/test_two_timescale_simulator.py::test_summary_counts_repairs_and_constraint_rejections tests/test_two_timescale_monte_carlo.py
```

Expected: 新汇总属性尚不存在，测试失败。

- [ ] **Step 3: 在 TwoTimescaleSummary 计算五个指标**

在数据类中加入：

```python
    fast_repair_attempts: int
    fast_repair_successes: int
    fast_repair_failures: int
    fast_repair_success_rate: float
    constraint_rejected_batches: int
```

在 `_build_summary` 中计算：

```python
        fast_repair_attempts = sum(
            int(record.fast_repair_attempted)
            for record in records
        )
        fast_repair_successes = sum(
            int(record.fast_repair_succeeded is True)
            for record in records
        )
        fast_repair_failures = sum(
            int(record.fast_repair_succeeded is False)
            for record in records
        )
        fast_repair_success_rate = (
            fast_repair_successes / fast_repair_attempts
            if fast_repair_attempts > 0
            else 0.0
        )
        constraint_rejected_batches = sum(
            int(
                record.request_count > 0
                and record.fast_repair_succeeded is False
            )
            for record in records
        )
```

把五个值作为命名参数传入 `TwoTimescaleSummary` 构造函数。不要修改原 `sla_violations = failed_batches + deadline_violations`，约束拒绝已经通过失败批次自动计入 SLA。

- [ ] **Step 4: 传播到蒙特卡洛原始记录和统计结果**

给 `TwoTimescaleMonteCarloRunRecord` 增加：

```python
    fast_repair_attempts: int
    fast_repair_successes: int
    fast_repair_failures: int
    fast_repair_success_rate: float
    constraint_rejected_batches: int
```

给 `TwoTimescaleMonteCarloScenarioSummary` 增加同名 `MetricSummary` 字段。在构造原始记录时从 `summary` 复制五项；在方案汇总时分别调用 `metric("字段名")`。

- [ ] **Step 5: 更新单次对比输出和 CSV**

在 `print_result` 增加：

```python
    print(f"  快层修复尝试次数：{summary.fast_repair_attempts}")
    print(f"  快层修复成功次数：{summary.fast_repair_successes}")
    print(f"  快层修复成功率：{summary.fast_repair_success_rate:.6f}")
    print(f"  约束拒绝批次数：{summary.constraint_rejected_batches}")
```

在 `save_summary_csv.fieldnames` 的 SLA 字段后加入五个字段。`writer.writerow` 已经通过 `getattr` 自动读取，不增加第二份字段映射。

在 `print_two_timescale_events` 的 `important` 条件加入 `record.fast_repair_attempted`，并打印 `fast_repair_succeeded` 与 `fast_repair_reason`。

- [ ] **Step 6: 更新蒙特卡洛打印和 CSV**

在 `print_scenario_summary` 增加 `fast_repair_success_rate` 与 `constraint_rejected_batches` 的 `print_metric` 调用。在 `save_summary_csv.metric_mapping` 增加：

```python
            "fast_repair_success_rate": (
                summary.fast_repair_success_rate
            ),
            "constraint_rejected_batches": (
                summary.constraint_rejected_batches
            ),
```

原始 CSV 从 dataclass 字段自动生成，因此不手写重复列名。

- [ ] **Step 7: 运行统计和输出相关测试**

Run:

```powershell
D:\Anaconda3\python.exe -m pytest -q -p no:cacheprovider -c NUL tests/test_two_timescale_simulator.py tests/test_two_timescale_monte_carlo.py
```

Expected: 修复尝试、成功、失败、成功率和拒绝批次统计通过；约束拒绝只产生一次 SLA 惩罚。

- [ ] **Step 8: 提交并推送 Task 5**

```powershell
git add src/two_timescale_simulator.py src/two_timescale_monte_carlo.py run_two_timescale_comparison.py run_two_timescale_monte_carlo.py tests/test_two_timescale_simulator.py tests/test_two_timescale_monte_carlo.py
git commit -m "feat: report fast feasibility repair metrics"
git push origin codex/fast-feasibility-repair
```

### Task 6: 全量回归、轻量实验和 review 材料

**Files:**
- Verify: all modified Python files
- Verify: `docs/superpowers/specs/2026-08-06-fast-feasibility-repair-design.md`

- [ ] **Step 1: 运行完整测试集**

Run:

```powershell
D:\Anaconda3\python.exe -m pytest -q -p no:cacheprovider -c NUL tests
```

Expected: 现有 177 个测试与本轮新增测试全部通过，0 failed，0 errors。

- [ ] **Step 2: 检查所有 Python 文件语法**

Run:

```powershell
D:\Anaconda3\python.exe -c "import ast,pathlib; files=list(pathlib.Path('.').rglob('*.py')); [ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p)) for p in files]; print(f'parsed {len(files)} Python files')"
```

Expected: 打印解析文件总数，无异常。

- [ ] **Step 3: 运行单次对比入口的无文件烟雾场景**

Run:

```powershell
D:\Anaconda3\python.exe -c "from src.config import load_config; from run_two_timescale_comparison import build_simulator; from src.two_timescale_control import build_rule_based_two_timescale_controller; c=load_config('configs/debug.yaml'); r=build_simulator(c,build_rule_based_two_timescale_controller(c)).run(); print(len(r.records), r.summary.fast_repair_attempts, r.summary.constraint_rejected_batches)"
```

Expected: 输出非零时隙数量和两个整数，且不创建或覆盖结果表。

- [ ] **Step 4: 运行一组配对蒙特卡洛轻量场景**

Run:

```powershell
D:\Anaconda3\python.exe -c "from src.config import load_config; from run_two_timescale_monte_carlo import build_simulator; from src.two_timescale_monte_carlo import run_two_timescale_monte_carlo; c=load_config('configs/debug.yaml'); builders={'fixed_single':lambda seed:build_simulator(c,'fixed_single',seed),'two_timescale':lambda seed:build_simulator(c,'two_timescale',seed)}; r=run_two_timescale_monte_carlo(builders,1,5000,0.95); print([(s.scenario_name,s.fast_repair_success_rate.mean,s.constraint_rejected_batches.mean) for s in r.scenario_summaries])"
```

Expected: 两种方案各输出一组修复成功率和约束拒绝均值，不写入仓库中的 `results/`。

- [ ] **Step 5: 检查变更范围和中文注释**

Run:

```powershell
git diff --check origin/main...HEAD
git status --short --branch
git log --oneline --decorate origin/main..HEAD
```

Expected: `git diff --check` 无输出；工作区干净；提交历史只包含设计、审计器、公共路由、快层修复、模拟器接入和指标输出。

人工 review 以下注释必须存在并与代码一致：

1. 快层为什么不能改变慢层副本数量；
2. 候选评分五级顺序；
3. 无请求时为什么保持 `request_success=None`；
4. 修复失败时为什么不计请求冷启动和 CPU；
5. 为什么审计器是资源与可靠性公式的唯一来源。

- [ ] **Step 6: 最终代码审查并修复发现的问题**

使用 `requesting-code-review` 技能检查：设计覆盖、测试覆盖、错误处理、确定性、统计是否重复计数、是否存在与本阶段无关的改动。任何修复都先增加或调整能够复现问题的测试，再修改生产代码并重新执行 Step 1–5。

- [ ] **Step 7: 提交最终审查修正并推送**

如果 Step 6 产生修正：

```powershell
git add src tests run_two_timescale_comparison.py run_two_timescale_monte_carlo.py
git commit -m "fix: address fast feasibility repair review"
git push origin codex/fast-feasibility-repair
```

如果没有文件变化，不创建空提交；确认远端分支已经包含每个任务的提交。

- [ ] **Step 8: 整理给用户的 review 清单**

最终说明按以下顺序组织：

1. 修改的每个文件及作用；
2. 一条可修复资源超限请求如何从初审失败变为复审成功；
3. 单副本无法满足双故障域要求时为什么必须拒绝；
4. 修复前后审计字段如何阅读；
5. 新增汇总指标如何解释；
6. 测试总数、语法文件数和轻量实验输出；
7. 当前仍未实现的中心云、凸优化和依赖舍入；
8. GitHub 分支和各提交编号。
