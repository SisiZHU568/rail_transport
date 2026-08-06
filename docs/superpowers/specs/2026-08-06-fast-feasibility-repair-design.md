# 快时间尺度可行性修复设计

## 1. 背景

上一阶段已经为每个快时隙增加了 `SlotConstraintAudit`，能够记录 CPU、内存、副本计划和 SFC 可靠性是否满足约束。不过，当前模拟器采用“只记录、不拦截”的方式：即使方案资源超限或可靠性不足，只要现有路由找到了可用副本，请求仍可能被统计为成功。

本阶段把约束审计接入请求执行流程，形成以下闭环：

1. 快层控制器先生成初始方案；
2. 在执行请求前检查硬约束；
3. 初始方案不可行时，尝试修复副本位置和执行节点；
4. 修复后的方案再次接受同一套约束检查；
5. 只有最终方案可行时才执行 SFC；
6. 无法修复时拒绝当前请求批次，并保存明确原因。

## 2. 目标与非目标

### 2.1 本阶段目标

- 将 CPU、内存、副本计划和可靠性作为执行前的硬约束；
- 实现独立、确定性的快层可行性修复器；
- 快层可以调整副本所在 MEC 和实际执行节点；
- 快层严格遵守慢层给出的副本数量和备用模式；
- 同时保存修复前、修复后的约束审计；
- 统计修复尝试次数、成功率和约束拒绝批次数；
- 保证相同输入得到相同输出，便于实验复现；
- 为后续凸优化、分数解和依赖舍入提供可对照的可行性基线。

### 2.2 本阶段不实现

- 不引入中心云外包；
- 不改变 DDQN 的状态、动作或奖励；
- 不实现相对熵正则、分式规划、内点法或依赖舍入；
- 不允许快层增加或减少慢层决定的副本数量；
- 不把本阶段的确定性搜索宣称为最终论文优化算法；
- 不扩大到车载节点或中心云，候选资源仍为当前 `LinearRailTopology` 中的轨旁 MEC。

## 3. 双时间尺度职责边界

慢层继续决定：

- 是否使用冗余；
- 每个函数使用一个还是两个副本；
- 备用采用 `SINGLE`、`COLD` 或 `HOT` 模式；
- 决策在多少个快时隙内有效。

快层可以决定：

- 在慢层指定的副本数量内，将副本放到哪些正常工作的 MEC；
- 哪个副本作为当前主执行节点；
- 是否需要故障接管或冷启动；
- 当前请求批次能否在全部硬约束下执行。

快层不能通过临时增加副本来掩盖慢层动作不可行的问题。例如，慢层选择单副本但没有任何单副本方案能达到可靠性目标时，快层必须返回修复失败。这样才能在实验中准确识别慢层策略的不足。

## 4. 总体数据流

```text
慢层决策
   ↓
副本规划器生成初始副本位置
   ↓
现有快层控制器生成初始执行决策
   ↓
执行前约束审计 ──可行──→ 执行 SFC
   │
   └─不可行
       ↓
快层可行性修复器搜索替代副本位置
       ↓
使用公共路由函数生成修复后的执行决策
       ↓
再次约束审计 ──可行──→ 执行 SFC
       │
       └─不可行或没有候选方案──→ 拒绝请求并记录原因
```

无请求时仍检查部署可行性，并允许修复副本计划。但是，无请求时 `request_success` 必须保持 `None`；部署不可行不能被统计成请求失败。

## 5. 组件设计

### 5.1 `src/constraint_audit.py`

新增 `SlotConstraintAuditor`，从 `TwoTimescaleRuntimeSimulator` 中接管现有 `_audit_slot_constraints` 逻辑。构造函数持有以下只读依赖：

- `functions`；
- `sfc`；
- `topology`；
- `reliability_model`。

公开方法为：

```python
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
```

该类只计算并返回审计结果，不修改部署、决策或请求状态。模拟器和修复器必须复用该类，不能分别实现两套容量或可靠性公式。

`SlotConstraintAudit` 同时新增 `replica_count_violation_function_ids`。只要任一函数的实际副本数不等于 `expected_replica_count`，`replica_plan_valid` 和 `all_constraints_met` 都必须为 `False`，并生成中文违规原因。这一检查保证快层不会误用超过慢层许可数量的副本。

### 5.2 `src/two_timescale_control.py`

新增纯函数 `build_fast_decision_for_plan`，负责根据一个副本方案生成完整的 `FastTimescaleDecision`：

```python
def build_fast_decision_for_plan(
    state: FastTimescaleState,
    standby_mode: StandbyMode,
    backup_activation_triggered: bool,
    previously_hot_node_ids: (
        dict[int, tuple[int, ...]] | None
    ) = None,
) -> FastTimescaleDecision:
```

规则控制器和固定基线控制器先按照各自规则决定 `backup_activation_triggered`，再调用该函数。这样可以删除两个控制器中重复的路由代码，并确保修复器使用完全相同的主备、故障接管和冷启动语义。

`previously_hot_node_ids` 为空时保持当前控制器行为；修复器传入初始决策的温实例映射。如果修复后的执行节点原来不是温实例，该函数必须把对应函数记录为冷启动。

### 5.3 `src/fast_optimizer.py`

现有空文件 `src/fast_optimizor.py` 的拼写不正确，而且尚未被任何代码引用。本阶段将删除该空文件，并使用正确名称创建 `src/fast_optimizer.py`。

新增不可变结果对象：

```python
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
```

字段语义：

- `attempted=False`、`succeeded=None`：初始方案可行，不需要搜索；
- `attempted=True`、`succeeded=True`：搜索并找到可行方案；
- `attempted=True`、`succeeded=False`：搜索完成但没有可行方案；
- 修复失败时继续返回初始副本方案和初始审计，表示没有采用任何不可行的替代方案；
- 修复失败且存在请求时，结果中的 `decision` 是初始决策的“拒绝副本”：保留温实例信息，清空执行路径，并把 `request_success` 设为 `False`；
- 修复失败且没有请求时，结果中的 `decision.request_success` 保持 `None`；
- `reason` 使用中文，说明无需修复、修改了哪些函数，或者为什么失败；
- `evaluated_candidate_count` 用于观察搜索开销，不包含只做一次的初始方案检查。

新增 `FastFeasibilityOptimizer`。它持有拓扑、审计器等依赖，公开方法为：

```python
def optimize(
    self,
    state: FastTimescaleState,
    slow_decision: SlowTimescaleDecision,
    initial_decision: FastTimescaleDecision,
    initial_audit: SlotConstraintAudit,
) -> FastOptimizationResult:
```

初始副本映射由 `state.candidate_node_ids` 提供，避免同一份输入以两个参数重复传递。

### 5.4 `src/two_timescale_simulator.py`

`TwoTimescaleRuntimeSimulator` 构造时创建或接收同一个 `SlotConstraintAuditor` 和 `FastFeasibilityOptimizer`。每个时隙按照以下顺序运行：

1. 取得初始副本映射；
2. 生成初始快层决策；
3. 根据冷启动函数预测本时隙会被激活的实例；
4. 在执行 SFC 前完成初始审计；
5. 调用修复器；
6. 使用最终副本映射和最终快层决策重新计算活动内存；
7. 最终审计可行时执行 SFC；
8. 最终审计不可行时不调用 `execute_sfc_batch`；
9. 有请求时把最终 `request_success` 记录为 `False`，无请求时保持 `None`。

副本重配置数量、活动内存、冷启动、请求时延和综合成本都必须在修复完成后，根据最终方案重新计算，不能继续使用初始方案的中间结果。

修复失败时，`initial_audit` 和 `final_audit` 都保留初始方案的“如果执行将产生的需求”，用于解释拒绝原因；实际运行记录不再计入请求触发的冷启动实例和请求执行 CPU，只保留原本维持的温实例内存。这样既能说明为什么拒绝，也不会把没有真正发生的执行开销计入成本。

## 6. 确定性修复算法

### 6.1 初始方案优先

修复器首先检查初始审计：

- `all_constraints_met=True`，并且无请求或初始决策已经形成完整执行路径时原样返回；
- 有请求但 `initial_decision.request_success is not True` 时，即使静态资源和可靠性审计合格，也必须搜索正常工作的替代执行节点；
- 不执行候选搜索；
- `attempted=False`；
- 原有副本位置、路由和冷启动结果保持不变。

### 6.2 候选节点与副本数量

- 候选节点只来自本时隙 `operational_node_ids`；
- 每个函数必须恰好使用 `slow_decision.replica_count` 个不同节点；
- `SINGLE` 模式只生成一个节点的有序元组；
- `COLD` 和 `HOT` 模式按照慢层给出的两个副本生成主备有序元组；
- 元组第一个节点是主副本，顺序变化表示主备角色发生变化；
- 任何未知、失效或重复节点都不能进入修复候选方案。

### 6.3 搜索顺序

候选方案使用稳定的字典序评分，按以下优先级从小到大搜索：

1. 发生副本方案变化的函数数量；
2. 被替换的副本节点数量；
3. 本时隙新增的执行路径冷启动数量；
4. 列车当前接入 MEC、各函数执行节点及返回路径之间的总地理距离；
5. 按 SFC 函数顺序展开后的节点编号元组。

第五项只用于打破完全相同的评分，保证相同输入得到相同结果。

“被替换的副本节点数量”按每个函数的旧节点集合减去新节点集合后累加。只交换同一组节点的主备顺序时该数量为零，但第一项仍会把该函数识别为方案变化。总地理距离只有在 `return_result_to_source=True` 时才包含最终结果返回当前接入 MEC 的距离。

当前正式调试拓扑包含 5 个轨旁 MEC、每个函数最多 2 个副本。本阶段允许对这些小规模离散候选进行完整、确定性的搜索。后续拓扑扩大时，由数学优化器替换搜索实现，但保留相同输入输出接口。

### 6.4 提前剪枝与最终检查

构造候选方案时，以下情况立即剪枝：

- 某个函数无法选出慢层要求数量的正常节点；
- 当前已经确定的执行节点 CPU 累计需求超过节点容量；
- 当前已经确定的活动实例内存累计需求超过节点容量。

完整方案仍必须交给 `SlotConstraintAuditor` 做最终检查，包括：

- CPU 容量；
- 活动实例内存容量；
- 副本计划结构；
- 精确共享故障域 SFC 可靠性；
- 故障域多样性要求。

找到评分最小的可行方案后停止。没有可行方案时返回修复失败，不选择“违反程度较小但仍不可行”的方案。

## 7. 冷启动与活动内存

- `HOT` 模式下，修复方案中的全部副本计入活动内存；
- `COLD` 模式下，通常只有主副本计入活动内存；
- 切换窗口已触发备用激活时，主备都计入活动内存；
- 修复后的执行节点若不在初始温实例集合中，需要记录冷启动；
- 因执行而冷启动的实例同时计入本时隙活动内存；
- 后台创建但不在请求执行路径上的新热备实例计入内存，但其启动过程暂不叠加到用户端到端时延；
- 这一限制在代码注释和最终 review 说明中明确保留，后续生命周期模型再补充后台启动成本。

## 8. 时隙记录与汇总指标

`TwoTimescaleSlotRecord` 新增：

```python
initial_constraint_audit: SlotConstraintAudit
fast_repair_attempted: bool
fast_repair_succeeded: bool | None
fast_repair_reason: str
fast_repair_evaluated_candidate_count: int
```

现有 `constraint_audit` 字段继续保留，但含义明确为最终采用方案的审计结果。现有读取 `record.constraint_audit` 的代码不需要改名。

`TwoTimescaleSummary` 新增：

```python
fast_repair_attempts: int
fast_repair_successes: int
fast_repair_failures: int
fast_repair_success_rate: float
constraint_rejected_batches: int
```

统计规则：

- 修复尝试次数统计所有时隙，包括没有请求但部署不可行的时隙；
- 没有修复尝试时成功率定义为 `0.0`，避免除零和 `NaN`；
- `constraint_rejected_batches` 只统计 `request_count > 0` 且最终因硬约束不可行而被拒绝的时隙；
- 被拒绝批次通过现有失败记录自动进入 SLA 违约和惩罚，不额外重复计费。

实验入口 `run_two_timescale_comparison.py` 和 `run_two_timescale_monte_carlo.py` 同步输出新增汇总指标，使其能够直接用于论文表格。

## 9. 错误处理与输入验证

- 副本数量小于 1 时抛出 `ValueError`；
- 直接调用审计器时，SFC 函数集合与副本映射不一致会返回结构违规；
- 模拟器收到函数集合不完整的规划器输出时仍抛出 `ValueError`，因为这种输入无法构造 `FastTimescaleState`，属于规划器编程错误而不是可修复的运行态资源不足；
- 可用节点数少于副本数量时返回正常的“修复失败”结果，不抛出异常；
- 修复器生成的方案若未通过最终审计，必须继续搜索，不能直接交给模拟器；
- 搜索耗尽属于预期业务结果，使用中文 `reason` 说明，不使用异常表示；
- 编程错误或非法构造参数仍使用异常暴露，不能伪装成“没有可行方案”。

## 10. 测试设计

### 10.1 `tests/test_constraint_audit.py`

- 将现有审计测试迁移或补充为独立单元测试；
- 验证 CPU 和内存累加；
- 验证容量恰好相等时可行；
- 验证无效副本计划；
- 验证精确可靠性和故障域多样性。

### 10.2 `tests/test_two_timescale_control.py`

- 公共路由函数保持 `SINGLE`、`COLD`、`HOT` 的现有行为；
- 主节点失效时选择备用；
- 冷备用接管产生冷启动；
- 传入初始温实例映射后，新执行节点正确产生冷启动；
- 两类现有控制器重构后输出不变。

### 10.3 `tests/test_fast_optimizer.py`

- 初始方案可行时不搜索；
- 同节点 CPU 超限时把执行分散到不同 MEC；
- 热实例内存超限时更换副本节点；
- 双副本从同一故障域调整到不同故障域；
- 单副本无法满足可靠性时返回失败；
- 正常节点数量少于副本数量时返回失败；
- 全部节点资源不足时返回失败；
- 相同输入重复运行得到相同方案和原因；
- 选中的方案满足慢层副本数量和备用模式。

### 10.4 `tests/test_two_timescale_simulator.py`

- 原方案可行时执行路径与现有结果一致；
- 资源超限但能够修复时请求成功；
- 修复前审计不可行、最终审计可行；
- 无法修复且有请求时不执行 SFC，并记录请求失败；
- 无请求且无法修复时 `request_success is None`；
- 修复统计、约束拒绝统计和 SLA 统计一致；
- 冷启动、活动内存和副本重配置成本使用最终方案计算。

### 10.5 全量验证

- 运行全部现有与新增测试；
- 解析全部 Python 文件，确认语法正确；
- 运行双时间尺度对比入口；
- 运行双时间尺度蒙特卡洛入口的轻量场景；
- 检查输出表包含新增修复指标；
- 检查 Git 工作区只包含本阶段相关修改。

## 11. 研究方案对应关系

本阶段对应研究方案中的快时间尺度在线资源分配与可行性保障部分：

- 将资源容量和高可靠性从观测指标升级为硬约束；
- 允许在慢层模板限定下动态调整跨边缘副本位置和执行路由；
- 形成“慢层模板—快层修复—执行反馈”的双尺度闭环；
- 输出可行性修复率和约束拒绝率，为后续算法对比提供指标。

它尚未覆盖研究方案中的中心云外包、连续松弛、相对熵正则、分式优化和依赖舍入。这些内容将在当前闭环稳定后分阶段实现。

## 12. 验收标准

本阶段只有同时满足以下条件才算完成：

1. 初始方案可行时行为保持不变；
2. 可修复的资源或可靠性违规能够得到最终可行方案；
3. 不可修复的活动请求被拒绝且计入失败与 SLA 违约；
4. 无请求时不会产生虚假请求失败；
5. 修复前后审计和中文原因均可从时隙记录读取；
6. 汇总与实验输出包含修复和约束拒绝指标；
7. 快层始终遵守慢层的副本数量和备用模式；
8. 所有测试、语法检查和轻量实验入口通过；
9. 代码包含面向初学者的必要中文注释；
10. 修改提交并推送到 `codex/fast-feasibility-repair` 分支，供用户 review。
