# DPPO Fast Convex Scheduler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 CVXPY + CLARABEL 为 DPPO 主算法实现快时间尺度多路径请求调度，并保证快层只使用慢层已经部署的副本、失败时直接拒绝且不调用旧枚举优化器。

**Architecture:** `FastConvexScheduler` 负责生成固定部署下的完整 SFC 路径、构建连续分流模型并做容量感知确定性取整；`SlotConstraintAuditor` 统一执行部署和整数多路径的最终硬约束检查；`FastSlotExecutor` 仅在 DPPO 显式部署意图分支调用新调度器并逐批执行，历史规则分支继续使用旧优化器。配置由 `dppo.fast_scheduler` 唯一注入，CLARABEL 失败不切换任何备用求解器。

**Tech Stack:** Python 3.12、NumPy、CVXPY、CLARABEL、pytest、PyYAML

---

## 实施原则

- 只运行本功能直接相关的核心测试，不做全量基线回归。
- 每个功能先写失败测试，再写最小实现，再提交一次小型 commit。
- 新增的关键数学变量、取整步骤和失败分支必须写面向初学者的中文注释。
- DPPO 训练继续使用配置中的 `cuda`；CLARABEL 固定在 CPU 上运行，两者互不替代。
- 新调度器不迁移、不增加、不删除慢层部署副本。
- 只有严格的 `optimal` 状态可以进入取整；`optimal_inaccurate`、`infeasible`、异常和非有限解全部拒绝。
- 无请求时不调用 CVXPY，`request_success` 保持 `None`。

## Task 1：安装检查和快层配置入口

**Files:**

- Modify: `configs/debug.yaml`
- Modify: `src/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1：确认当前 Python 环境能导入 CVXPY 和 CLARABEL**

Run:

```powershell
D:\Anaconda3\python.exe -c "import cvxpy as cp; print(cp.__version__); print(cp.installed_solvers())"
```

Expected: 输出 CVXPY 版本，且列表包含 `CLARABEL`。

若缺少依赖，只安装本任务需要的求解依赖：

```powershell
D:\Anaconda3\python.exe -m pip install "cvxpy>=1.6,<2"
```

重新运行上面的导入检查。不得安装或启用 HiGHS 作为替代。

- [ ] **Step 2：为配置校验编写失败测试**

在 `tests/test_config.py` 增加核心用例：

```python
def test_load_debug_config_contains_clarabel_fast_scheduler() -> None:
    config = load_config(DEBUG_CONFIG_PATH)

    assert config["dppo"]["fast_scheduler"] == {
        "solver": "CLARABEL",
        "max_iterations": 200,
        "feasibility_tolerance": 1.0e-7,
    }


@pytest.mark.parametrize(
    ("key", "invalid_value"),
    [
        ("solver", "HIGHS"),
        ("max_iterations", 0),
        ("feasibility_tolerance", 0.0),
    ],
)
def test_validate_config_rejects_invalid_fast_scheduler_setting(
    key: str,
    invalid_value: object,
) -> None:
    config = load_raw_debug_config()
    config["dppo"]["fast_scheduler"][key] = invalid_value

    with pytest.raises(ValueError, match=f"dppo.fast_scheduler.{key}"):
        validate_config(config)
```

- [ ] **Step 3：运行配置 RED 测试**

Run:

```powershell
$env:PYTHONUTF8='1'
D:\Anaconda3\python.exe -m pytest tests/test_config.py -q --basetemp="C:\Users\Administrator\.codex\.chatgpt-projects\g-p-6a6ff4453abc819184ee35997a150260\.pytest_tmp\fast_scheduler_config_red"
```

Expected: 新测试因配置块缺失或未校验而失败。

- [ ] **Step 4：加入唯一配置块和严格校验**

在 `configs/debug.yaml` 的 `dppo` 下新增：

```yaml
  fast_scheduler:
    # 快层数学求解固定使用 CLARABEL；DPPO 神经网络仍由 training.device 控制。
    solver: CLARABEL
    max_iterations: 200
    feasibility_tolerance: 1.0e-7
```

在 `src/config.py::validate_config` 中：

- 用 `_require_mapping(dppo, "fast_scheduler")` 强制该块存在；
- `solver` 必须严格等于 `CLARABEL`；
- `max_iterations` 必须为正整数且拒绝布尔值；
- `feasibility_tolerance` 必须为正有限数且拒绝布尔值；
- 错误信息包含完整配置键，方便用户定位 YAML。

- [ ] **Step 5：运行配置 GREEN 测试并提交**

```powershell
$env:PYTHONUTF8='1'
D:\Anaconda3\python.exe -m pytest tests/test_config.py -q --basetemp="C:\Users\Administrator\.codex\.chatgpt-projects\g-p-6a6ff4453abc819184ee35997a150260\.pytest_tmp\fast_scheduler_config_green"
```

Expected: 配置核心测试全部通过。

```powershell
D:\Git\cmd\git.exe add configs/debug.yaml src/config.py tests/test_config.py
D:\Git\cmd\git.exe commit -m "feat: configure DPPO fast convex scheduler"
```

## Task 2：实现独立的凸优化调度器

**Files:**

- Create: `src/fast_convex_scheduler.py`
- Create: `tests/test_fast_convex_scheduler.py`

- [ ] **Step 1：先写数据对象和输入校验测试**

在 `tests/test_fast_convex_scheduler.py` 建立小型两函数、三节点场景，覆盖：

1. `FastScheduledBatch` 拒绝非正请求数和空路径；调度器拒绝路径长度与 SFC 不匹配；
2. 调度结果中的比例、批次数和路径使用不可变元组；
3. 请求数为 0 时不调用 `cvxpy.Problem.solve`，返回成功的空批次，状态为 `not_run`；
4. `candidate_map` 必须完整覆盖 SFC，路径节点只能来自慢层候选集合；
5. 故障节点不会进入合法路径。

- [ ] **Step 2：运行数据对象 RED 测试**

```powershell
$env:PYTHONUTF8='1'
D:\Anaconda3\python.exe -m pytest tests/test_fast_convex_scheduler.py -q --basetemp="C:\Users\Administrator\.codex\.chatgpt-projects\g-p-6a6ff4453abc819184ee35997a150260\.pytest_tmp\fast_scheduler_model_red"
```

Expected: 因 `src.fast_convex_scheduler` 不存在而在收集阶段失败。

- [ ] **Step 3：实现公共数据对象和构造校验**

在 `src/fast_convex_scheduler.py` 新增：

```python
@dataclass(frozen=True)
class FastScheduledBatch:
    request_count: int
    execution_node_ids: tuple[int, ...]


@dataclass(frozen=True)
class FastConvexSchedulingResult:
    succeeded: bool
    solver_status: str
    objective_value: float | None
    solve_time_seconds: float
    path_node_ids: tuple[tuple[int, ...], ...]
    path_fractions: tuple[float, ...]
    scheduled_batches: tuple[FastScheduledBatch, ...]
    reason: str
```

实现 `FastConvexScheduler.__init__`，保存：

- functions、sfc、topology、network；
- 与执行器相同的三类成本单价和时隙长度；
- `solver_name="CLARABEL"`、最大迭代数和可行性容差；
- `return_result_to_source` 和单请求输入数据量。

所有数值参数先检查类型、有限性和范围；求解器名字只接受 `CLARABEL`。

- [ ] **Step 4：先写路径和成本模型测试**

增加核心测试：

1. 所有合法路径严格按节点元组排序；
2. 超过 SFC deadline 的路径在建模前被排除；
3. 单路径预计成本等于“CPU 运行 + 网络传输 + 实际需冷启动实例”的同单位成本之和；
4. 相同节点承载多个 VNF 时，CPU 系数按函数累加；
5. 网络接口抛出不可达错误时，该路径被排除，而不是终止整个路径生成。

路径预计时延和网络成本调用既有 `execute_sfc_request()` 或共享网络接口计算，禁止复制另一套数据传输公式。

- [ ] **Step 5：实现固定部署下的路径生成和系数计算**

新增私有方法，职责清晰分开：

```python
def _build_eligible_paths(...)
def _estimate_path(...)
def _build_cpu_coefficients(...)
```

规则：

- 对每个函数仅使用 `candidate_map[function_id]` 中同时属于 `operational_node_ids` 的节点；
- 用 `itertools.product` 生成执行路径，但不能生成或修改部署方案；
- 使用当前温实例集合判断函数—节点对是否产生冷启动；
- 预计端到端时延超过 deadline 的路径直接删除；
- 对路径元组排序，以确保取整可复现。

- [ ] **Step 6：先写求解、取整和严格失败测试**

增加核心测试：

1. 低负载全部请求选择最低成本路径；
2. 最低成本节点容量不足时，请求分流到两条完整路径；
3. 取整后 `sum(batch.request_count) == request_count`；
4. 相同输入连续调用两次得到相同批次顺序和请求数；
5. floor 后的余数按“小数余量降序、路径元组升序”分配；
6. 取整后任一节点 CPU 或激活内存超限则整体失败；
7. `infeasible`、`optimal_inaccurate`、求解异常、NaN/Infinity 解全部返回失败；
8. 所有失败用例只调用一次配置的 CLARABEL，不尝试其他 solver。

对状态测试使用 monkeypatch 包装 `cvxpy.Problem.solve`，不依赖私有 CVXPY 实现。

- [ ] **Step 7：实现凸模型**

在 `FastConvexScheduler.schedule(...)` 中：

```python
x = cp.Variable(len(paths), nonneg=True)
constraints = [cp.sum(x) == 1.0, x <= 1.0]
for node_id in compute_node_ids:
    constraints.append(
        request_count * cpu_coefficients[node_id] @ x
        <= node_cpu_capacity[node_id]
    )
problem = cp.Problem(cp.Minimize(path_costs @ x), constraints)
problem.solve(
    solver="CLARABEL",
    max_iter=max_iterations,
    tol_feas=feasibility_tolerance,
    tol_gap_abs=feasibility_tolerance,
    verbose=False,
)
```

实现时添加中文注释说明：

- `x[p]` 是请求比例而非新部署动作；
- 容量约束中乘请求数是从“每请求需求”换算到当前时隙总需求；
- 目标函数不使用额外权重。

只接受 `problem.status == cp.OPTIMAL`；记录 `solver_stats.solve_time`，缺失时使用外层单调时钟耗时。

- [ ] **Step 8：实现容量感知确定性取整**

新增 `_round_request_counts(...)`：

1. 计算 `floor(request_count * fraction)`；
2. 汇总当前整数 CPU 和被激活的冷实例内存；
3. 按 `(-fractional_remainder, path_tuple)` 排序余数候选；
4. 给候选路径增加一个请求前检查 CPU、内存和 SLA；
5. 无法分完全部余数时返回失败，不二次搜索、不调用旧优化器；
6. 删除请求数为 0 的批次，并按路径元组稳定输出。

注意：连续比例只指导整数取整，代码注释和结果字段不得把它描述为整数精确最优解。

- [ ] **Step 9：运行独立调度器 GREEN 测试并提交**

```powershell
$env:PYTHONUTF8='1'
D:\Anaconda3\python.exe -m pytest tests/test_fast_convex_scheduler.py -q --basetemp="C:\Users\Administrator\.codex\.chatgpt-projects\g-p-6a6ff4453abc819184ee35997a150260\.pytest_tmp\fast_scheduler_model_green"
D:\Git\cmd\git.exe diff --check
D:\Git\cmd\git.exe add src/fast_convex_scheduler.py tests/test_fast_convex_scheduler.py
D:\Git\cmd\git.exe commit -m "feat: add CLARABEL fast request scheduler"
```

Expected: 新文件核心测试全部通过，diff check 无错误。

## Task 3：扩展共享约束审计器支持固定部署和多路径批次

**Files:**

- Modify: `src/constraint_audit.py`
- Modify: `tests/test_constraint_audit.py`

- [ ] **Step 1：先写部署审计 RED 测试**

为新的 `audit_deployment(...)` 编写核心测试：

1. 正确副本数量、已知节点、温实例内存和可靠性均满足时通过；
2. 任一部署节点故障时失败；
3. 副本数量与慢层意图不一致时失败；
4. 温实例内存超过节点容量时失败；
5. 故障域可靠性未达标时失败；
6. 部署审计不产生请求 CPU 需求。

- [ ] **Step 2：先写多路径调度审计 RED 测试**

为新的 `audit_scheduled_batches(...)` 编写核心测试：

1. 两个路径批次的请求数必须累加到输入总请求数；
2. 所有路径必须与 SFC 函数数相同，且每个节点属于对应函数的慢层部署；
3. 多路径在同一节点的 CPU 需求按“路径请求数 × 函数单请求需求”累计；
4. 冷启动内存以函数—节点对去重后累计；
5. 任一 CPU、内存或 SLA 违规使 `all_constraints_met=False`；
6. 无请求时只接受空批次。

- [ ] **Step 3：运行审计 RED 测试**

```powershell
$env:PYTHONUTF8='1'
D:\Anaconda3\python.exe -m pytest tests/test_constraint_audit.py -q --basetemp="C:\Users\Administrator\.codex\.chatgpt-projects\g-p-6a6ff4453abc819184ee35997a150260\.pytest_tmp\fast_scheduler_audit_red"
```

Expected: 新方法不存在或不支持多路径而失败。

- [ ] **Step 4：复用现有公式实现两个审计入口**

在 `SlotConstraintAuditor` 中加入：

```python
def audit_deployment(...)
def audit_scheduled_batches(...)
```

实现要求：

- 抽取并复用现有 `audit()` 的副本结构、可靠性和资源检查逻辑；
- 旧 `audit()` 保持兼容，避免历史规则分支受影响；
- 调度审计的 CPU 按所有整数批次累计；
- 活动内存用 `set[(function_id, node_id)]` 去重；
- 部署审计与调度审计返回统一 `SlotConstraintAudit`，最终是否可执行只读取该对象；
- 错误原因保留中文并指出具体函数或节点。

- [ ] **Step 5：运行审计 GREEN 和旧单路径核心回归并提交**

```powershell
$env:PYTHONUTF8='1'
D:\Anaconda3\python.exe -m pytest tests/test_constraint_audit.py tests/test_fast_optimizer.py -q --basetemp="C:\Users\Administrator\.codex\.chatgpt-projects\g-p-6a6ff4453abc819184ee35997a150260\.pytest_tmp\fast_scheduler_audit_green"
D:\Git\cmd\git.exe diff --check
D:\Git\cmd\git.exe add src/constraint_audit.py tests/test_constraint_audit.py
D:\Git\cmd\git.exe commit -m "feat: audit multi-path fast schedules"
```

Expected: 新多路径测试和旧单路径优化器核心测试通过。

## Task 4：接入 DPPO 快时隙执行闭环

**Files:**

- Modify: `src/fast_slot_executor.py`
- Modify: `src/dppo_scenario.py`
- Modify: `src/dppo_slow_timescale_env.py`
- Modify: `tests/test_fast_slot_executor_intent.py`
- Modify: `tests/test_dppo_slow_timescale_env.py`

- [ ] **Step 1：先写“无旧算法回退”RED 测试**

在 `tests/test_fast_slot_executor_intent.py` 增加测试替身：

```python
class ForbiddenLegacyOptimizer:
    def optimize(self, *args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("DPPO explicit intent must not call legacy optimizer")
```

核心用例：

1. 有 `deployment_intent` 时只调用新调度器；
2. 新调度器返回失败时直接得到 `constraint_rejected=True`；
3. 失败路径没有执行 SFC、没有写入温热保留状态、没有调用旧优化器；
4. 显式意图却未注入新调度器时给出清晰配置错误，不隐式回退；
5. `slow_decision` 历史入口仍调用旧优化器且不调用新调度器。

- [ ] **Step 2：先写多路径执行和汇总 RED 测试**

新增核心用例：

1. 两个 `FastScheduledBatch` 分别调用 `execute_sfc_batch`，请求数与路径准确传递；
2. 总路由成本、冷启动成本和运行成本按真实批次汇总；
3. 端到端时延按请求数加权平均；
4. 只有所有批次执行和最终调度审计都成功，`request_success` 才为 True；
5. 冷启动使用 `(function_id, node_id)` 对，避免同一函数在不同路径节点丢失位置；
6. 无请求时结果为 `request_success=None`、solver 状态 `not_run`，且不调用求解器和 SFC 执行器；
7. 结果暴露 solver 状态、连续目标值、耗时和整数路径分配。

- [ ] **Step 3：运行执行器 RED 测试**

```powershell
$env:PYTHONUTF8='1'
D:\Anaconda3\python.exe -m pytest tests/test_fast_slot_executor_intent.py -q --basetemp="C:\Users\Administrator\.codex\.chatgpt-projects\g-p-6a6ff4453abc819184ee35997a150260\.pytest_tmp\fast_scheduler_executor_red"
```

Expected: 当前显式意图仍调用旧优化器且只支持单路径，因此新测试失败。

- [ ] **Step 4：扩展执行结果字段**

在 `FastSlotExecutionResult` 末尾增加有兼容默认值的字段：

```python
fast_solver_status: str = "not_used"
fast_solver_objective_value: float | None = None
fast_solver_time_seconds: float = 0.0
scheduled_request_counts: tuple[int, ...] = ()
scheduled_execution_node_ids: tuple[tuple[int, ...], ...] = ()
```

保留已有 `selected_execution_node_ids`，在多路径成功时填第一条非空批次路径，供旧日志读取；论文与新环境信息必须读取完整 `scheduled_*` 字段，不能据此误认为只有一条路径。

- [ ] **Step 5：在执行器中分离 DPPO 和历史分支**

构造器增加：

```python
fast_convex_scheduler: FastConvexScheduler | None = None
```

执行流程修改为：

```python
if slot_input.deployment_intent is not None:
    if self.fast_convex_scheduler is None:
        raise RuntimeError("DPPO explicit intent requires FastConvexScheduler.")
    # 部署审计 -> CLARABEL 调度 -> 整数调度审计 -> 分批执行
else:
    # 原有 FastFeasibilityOptimizer 历史路径保持不变
```

DPPO 分支必须满足：

- `final_candidate_map` 始终等于慢层意图产生的 `candidate_map`；
- 部署审计失败时不调用求解器；
- 求解失败时不调用 `fast_optimizer.optimize()`；
- 整数调度审计是执行前最后门禁；
- 只在全部成功后写入 retention tracker；
- 多批次冷启动按函数—节点对去重计费；
- 单批次 SLA 不满足时整个时隙失败；
- 代码用中文注释解释“历史分支保留”不等于“DPPO 回退”。

- [ ] **Step 6：从场景配置创建并注入唯一调度器**

在 `src/dppo_scenario.py::build_dppo_scenario` 中：

- 读取 `config["dppo"]["fast_scheduler"]`；
- 用现有 functions、sfc、topology、network、cost_rates、slot_seconds 创建唯一 `FastConvexScheduler`；
- 将其传给 `FastSlotExecutor`；
- 保留旧 `FastFeasibilityOptimizer` 只服务兼容历史入口，不作为 DPPO 失败回退。

- [ ] **Step 7：把求解指标传到 DPPO 环境 info**

在 `src/dppo_slow_timescale_env.py` 找到快时隙结果汇总位置，增加：

- `fast_solver_statuses`；
- `fast_solver_objective_values`；
- `fast_solver_time_seconds`；
- `fast_scheduled_path_counts`。

只做信息透传，不修改主奖励函数，避免重新引入多项人工权重。

- [ ] **Step 8：运行执行闭环 GREEN 测试并提交**

```powershell
$env:PYTHONUTF8='1'
D:\Anaconda3\python.exe -m pytest tests/test_fast_slot_executor_intent.py tests/test_dppo_slow_timescale_env.py -q --basetemp="C:\Users\Administrator\.codex\.chatgpt-projects\g-p-6a6ff4453abc819184ee35997a150260\.pytest_tmp\fast_scheduler_executor_green"
D:\Git\cmd\git.exe diff --check
D:\Git\cmd\git.exe add src/fast_slot_executor.py src/dppo_scenario.py src/dppo_slow_timescale_env.py tests/test_fast_slot_executor_intent.py tests/test_dppo_slow_timescale_env.py
D:\Git\cmd\git.exe commit -m "feat: execute DPPO requests through convex fast layer"
```

Expected: DPPO 显式意图、多路径执行和无回退测试全部通过。

## Task 5：核心闭环验收、说明和上传

**Files:**

- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-08-11-fast-convex-scheduler-design.md`（仅当实现接口与设计文档有已验证的小偏差时）
- Test: `tests/test_fast_convex_scheduler.py`
- Test: `tests/test_constraint_audit.py`
- Test: `tests/test_fast_slot_executor_intent.py`
- Test: `tests/test_dppo_slow_timescale_env.py`

- [ ] **Step 1：增加一个真实慢窗口核心测试**

在 `tests/test_dppo_slow_timescale_env.py` 增加小规模真实闭环：

- 使用 debug 场景的缩短慢窗口；
- 给环境一个合法的 DPPO 连续动作并生成 `SFCDeploymentIntent`；
- 至少一个有请求快时隙真实调用 CVXPY + CLARABEL；
- 断言部署节点没有被快层改变；
- 断言整数批次请求总数等于到达请求数；
- 断言 info 中可读取 `optimal`、目标值、耗时和路径数；
- 若配置故意制造不可行，断言直接拒绝且奖励按现有失败语义产生。

测试只跑 1 个缩短慢窗口，不启动完整训练。

- [ ] **Step 2：在 README 增加最短运行说明**

写明：

- DPPO 是慢层部署，CLARABEL 是快层请求调度；
- GPU 只加速 DPPO，CLARABEL 使用 CPU；
- 安装检查命令；
- 配置键位置；
- 失败不会调用旧枚举器；
- 连续目标值与最终整数执行成本含义不同。

- [ ] **Step 3：运行唯一验收测试组**

```powershell
$env:PYTHONUTF8='1'
D:\Anaconda3\python.exe -m pytest tests/test_config.py tests/test_fast_convex_scheduler.py tests/test_constraint_audit.py tests/test_fast_slot_executor_intent.py tests/test_dppo_slow_timescale_env.py -q --basetemp="C:\Users\Administrator\.codex\.chatgpt-projects\g-p-6a6ff4453abc819184ee35997a150260\.pytest_tmp\fast_scheduler_acceptance"
```

Expected: 全部通过。不要为了本阶段运行无关基线或完整实验套件。

- [ ] **Step 4：做最小静态检查**

```powershell
D:\Anaconda3\python.exe -m py_compile src/fast_convex_scheduler.py src/constraint_audit.py src/fast_slot_executor.py src/dppo_scenario.py src/dppo_slow_timescale_env.py
D:\Git\cmd\git.exe diff --check
D:\Git\cmd\git.exe status --short
```

Expected: 编译和 diff check 成功；状态只包含本计划文件。

- [ ] **Step 5：提交说明并上传分支**

```powershell
D:\Git\cmd\git.exe add README.md docs/superpowers/specs/2026-08-11-fast-convex-scheduler-design.md tests/test_dppo_slow_timescale_env.py
D:\Git\cmd\git.exe commit -m "docs: explain DPPO convex fast scheduling"
D:\Git\cmd\git.exe push origin codex/dppo-main-algorithm
```

若设计文档没有产生实现偏差，不要为了提交而修改它；只提交真实变更。

## 最终 Review 清单

- [ ] DPPO 显式部署意图从未调用 `FastFeasibilityOptimizer.optimize()`。
- [ ] 快层输出中的每个节点都属于对应 VNF 的慢层部署集合。
- [ ] 一个时隙可拆分到多条完整 SFC 路径。
- [ ] 目标函数只有运行、网络和冷启动三类同单位成本，没有新增奖励权重。
- [ ] 部署、CPU、内存、故障、可靠性和 SLA 均由硬约束或最终审计拒绝。
- [ ] 取整后请求总数严格守恒且相同输入结果稳定。
- [ ] `optimal_inaccurate`、异常、不可行和非有限解直接拒绝。
- [ ] 无请求时不调用求解器，失败时不执行部分请求。
- [ ] 求解状态、目标值、耗时和整数路径分配可从环境结果读取。
- [ ] 新代码关键位置有简明中文注释。
- [ ] 核心验收测试通过，分支已推送到 GitHub。
