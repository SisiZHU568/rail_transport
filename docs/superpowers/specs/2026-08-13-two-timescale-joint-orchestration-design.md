# 双时间尺度 DPPO—凸优化联合编排重构设计

日期：2026-08-13

状态：首轮书面 review 修订已纳入，等待复审
目标分支：`codex/dppo-main-algorithm`

## 1. 文档地位与研究边界

本文档定义论文主算法下一版的唯一有效实现语义。凡现有文档或代码中与以下内容有关的描述，均由本文档取代：

- 通用动作投影器或启发式部署回退；
- 独立 `replica_count` 动作；
- 快层路径比例优化；
- 快层调用冷实例或临时冷启动；
- 车载节点作为 VNF 部署或计算候选；
- 将无效保留时间维度从联合扩散策略的 PPO 概率中直接删除。

研究对象是高可靠轨道边缘 Serverless SFC 双时间尺度动态编排：

- 慢层 DPPO 决定每个 VNF—节点的目标容器实例数和保留时间；
- 快层连续凸优化只使用健康温实例，联合分配上行带宽与发射功率、有线流量、VNF 执行流量和 CPU 执行时间；
- 跨时隙队列与 EDF 审计器负责严格流水线因果性和批次级真实 SLA；
- 理论可靠性采用稳态点可用度，实际运行评价采用持续性 CTMC 故障—恢复轨迹。

首版不研究：

- 车载 VNF 部署或本地计算；
- 有线回传链路故障或回传发射功率；
- 已训练模型的零样本跨规模泛化；
- GNN、Transformer 或注意力式可变规模策略；
- CP-SAT、MILP 或启发式部署回退；
- 冷实例被快层临时激活；
- 将最终测试集用于模型、超参数或检查点选择。

## 2. 实施原则与阶段顺序

按 A→B→C→D→E 纵向重构：

| 阶段 | 内容 | 完成标准 |
|---|---|---|
| A | 配置、CTMC 故障、实例生命周期、删除车载部署 | 精确 CTMC 离散化；稳态初始化；生命周期边界正确；车载计算语义消失 |
| B | 阶段队列、在途流量、EDF 与 SLA 审计 | 流量守恒；阶段因果；EDF 可复现；跨时隙 SLA 正确 |
| C | 两阶段词典序凸优化 | 无线指数锥和计算幂锥通过 DCP；两次 CLARABEL 稳定；一级目标不退化 |
| D | 安全解码、状态动作接口、检查点规格 | 删除旧动作和投影器；只生成结构安全部署；无解与搜索故障严格区分 |
| E | 教师预训练、奖励、验证与在线训练 | 无未来信息；稳定性校准；固定验证选模；小规模闭环通过后再扩大训练 |

每阶段必须形成独立、可运行、可测试的提交。新逻辑接通并通过核心测试后，在同一提交删除其替代的旧语义；不保留 `_old` 文件、兼容开关或双套接口。Git 历史承担回退功能。

正式实施前建立基线：记录 Git 提交与工作区状态，运行现有测试并保存通过/失败清单，固定最小仿真配置和随机种子，保存关键输出用于发现非预期退化。新旧模型语义不同，不要求指标数值相同。

## 3. 总体架构

### 3.1 单一状态所有者

系统采用“单一状态所有者＋不可变快照＋显式结果提交”。

| 模块 | 持有状态 | 主要输出 |
|---|---|---|
| 配置层 | 无运行状态 | 只读模型配置、结构规格 |
| 故障过程层 | 域/节点隐藏 CTMC 状态、随机流 | `FailureSnapshot` |
| 生命周期层 | 实例批次、启动状态、保留期限、内存 | `LifecycleSnapshot` |
| 队列与 EDF 层 | 批次、阶段队列、在途事件、截止时间 | `QueueSnapshot`、SLA 结果 |
| 慢层解码器 | 无环境状态 | `DeploymentPlan` 或明确状态码 |
| 快层求解器 | 无环境状态 | `FastAllocationPlan` 或明确状态码 |
| 可靠性审计器 | 可靠性统计 | 结构、温实例和实际可靠性结果 |
| 仿真编排层 | 时钟、慢帧累计指标 | 调用模块并提交实际状态变更 |

求解器和审计器只能读取快照，不能访问完整环境对象或原地修改输入。只有仿真编排层可以调用状态模块的提交接口。

### 3.2 每快时隙事件顺序

1. 到达当前时刻的在途流量进入相应阶段队列；
2. `current_slot >= ready_slot` 的冷启动实例转为温状态；
3. `current_slot >= retention_deadline_slot` 的实例标记为可缩容，但不自动删除；
4. 从公共随机轨迹读取故障域、节点、负载、信道和移动状态；
5. 销毁刚失效节点上的全部温实例和启动中实例；恢复节点保持为空；
6. 新业务批次进入上行队列；
7. 若为慢帧边界，DPPO 读取统一观察快照并生成连续评分；
8. 安全解码器生成部署计划，生命周期层原子提交，新增实例进入冷启动；
9. 生成只读温实例、队列、故障和网络资源快照；
10. 快层依次执行两次 CLARABEL 凸优化；
11. EDF 按聚合服务量映射具体批次；
12. 原子更新阶段队列和在途事件，所有输出最早在下一快时隙可用；
13. 检查完成、首次超时、未服务量、可靠性、成本和求解耗时；
14. 慢帧末汇总 DPPO 奖励和实验指标。

同一时隙内，上行完成、VNF 执行完成和同节点 VNF 转移都不能立即被下一阶段消费。

## 4. 阶段 A：配置、故障与实例生命周期

### 4.1 配置

配置必须显式包含：

- 快时隙长度和每慢帧快时隙数；
- 故障域与节点的 `failure_rate_per_second`、`recovery_rate_per_second`；
- 节点故障域、内存、总 CPU 周期容量和核心数；
- VNF 单实例内存、单实例最大频率/容量、冷启动时延；
- 每个允许 VNF—节点组合的实例数范围；
- 保留时间最小值、最大值和步长；
- 故障、业务、信道、策略噪声等独立随机流的基础种子。

实例数可选集合为：

\[
n_{k,i,T}\in\{0\}\cup
\{n^{min}_{k,i},\ldots,n^{max}_{k,i}\}.
\]

其中 0 表示不部署，正整数下限表示一旦部署时的最小容器数。

配置要求：

\[
\lambda_j\ge0,\quad \mu_j\ge0,\quad \lambda_j+\mu_j>0.
\]

所有时间、bit、CPU 周期、频率、内存和价格字段必须带有明确单位并进行范围与引用校验。

彻底删除 `include_onboard`、车载计算容量、车载部署内存以及车载 VNF 候选。保留列车作为请求产生、位置、移动、无线信道和发射能耗实体。所有任务先由列车上传至当前服务 MEC，SFC 只在轨旁 MEC 或中心云执行。

### 4.2 精确 CTMC 离散化

若连续时间故障率和恢复率为 \(\lambda_j,\mu_j\)，快时隙长度为 \(\Delta t\)，则：

\[
p_j^{fail}=
\frac{\lambda_j}{\lambda_j+\mu_j}
\left(1-e^{-(\lambda_j+\mu_j)\Delta t}\right),
\]

\[
p_j^{recover}=
\frac{\mu_j}{\lambda_j+\mu_j}
\left(1-e^{-(\lambda_j+\mu_j)\Delta t}\right).
\]

这保证离散链的稳态可用度严格为：

\[
a_j=\frac{\mu_j}{\lambda_j+\mu_j}.
\]

故障域和节点分别维护独立隐藏马尔可夫状态。节点有效健康状态为：

\[
H_{i,t}=G_{g(i),t}\land N_{i,t}.
\]

即使故障域失效，节点自己的隐藏链仍独立演化，但有效状态始终不可用。初始状态按各自稳态分布采样。

`newly_unavailable_nodes` 和 `newly_available_nodes` 必须由有效状态边沿 \(H_{i,t-1}\to H_{i,t}\) 计算。故障域恢复时，只有节点隐藏状态也正常的节点才算恢复。生命周期层只消费有效状态变化；隐藏状态变化可作为诊断信息保留。

每个故障域、节点、业务到达和随机信道各有独立随机流。算法对比复用相同故障、负载、移动和信道轨迹，即公共随机数实验。

### 4.3 实例批次与时间语义

生命周期层使用实例批次：

```text
(function_id, node_id, batch_id,
 count, state, ready_slot, retention_deadline_slot)
```

活动状态只有 `STARTING` 和 `WARM`。删除或故障销毁后从活动集合移除。

边界语义：

- `current_slot >= ready_slot`：`STARTING` 转为 `WARM`；
- `current_slot >= retention_deadline_slot`：承诺到期、允许缩容；
- `current_slot < retention_deadline_slot`：仍被锁定；
- 冷启动时延为零：提交计划时立即建立温实例。

冷启动完成时隙为：

\[
ready\_slot=current\_slot+
\left\lceil\frac{d^{cold}_{k,i}}{\Delta t}\right\rceil.
\]

保留时长统一记为 \(L_{k,i,T}\)（秒），换算后的整数快时隙数为：

\[
L^{slot}_{k,i,T}=\left\lceil\frac{L_{k,i,T}}{\Delta t}\right\rceil.
\]

期限统一刷新公式为：

\[
deadline\leftarrow
\max\left(deadline,
\max(current\_slot,ready\_slot)+L^{slot}_{k,i,T}\right).
\]

期限只能延长，不能缩短。新实例从变温时刻开始计算保留承诺。

### 4.4 目标数、实际数与原子提交

DPPO 输出的 \(n_{k,i,T}^{target}\) 是目标活动实例数，包含 `STARTING + WARM`。当前已有活动实例数记为 \(n_{k,i,T}^{existing}\)，其中未到期锁定数统一记为 \(m_{k,i,T}^{locked}\)。提交后的实际数量可因锁定实例而暂时超过目标：

\[
a_{k,i,T}^{post}=\max
\left(m_{k,i,T}^{locked},n_{k,i,T}^{target}\right).
\]

生命周期层必须优先复用现有实例，不能把仍被目标需要的已到期实例先删除再重新冷启动。删除数和新建数严格定义为：

\[
n_{k,i,T}^{delete}=\max\left(
0,
n_{k,i,T}^{existing}-
\max(m_{k,i,T}^{locked},n_{k,i,T}^{target})
\right),
\]

\[
n_{k,i,T}^{new}=\max\left(
0,
n_{k,i,T}^{target}-n_{k,i,T}^{existing}
\right).
\]

需要保留的现有实例按规范实例批次顺序选择；只有超出
\(\max(m^{locked},n^{target})\) 的已到期实例才能删除。

计划提交顺序：

1. 找出未到期锁定实例；
2. 优先选择并复用目标仍需要的现有实例；
3. 按 \(n^{delete}\) 删除已到期且确实超出目标的实例；
4. 按 \(n^{new}\) 创建现有实例无法满足的差额；
5. 对目标范围内需要继续保留的实例刷新期限；
6. 超出目标的锁定实例不刷新；
7. 新实例进入 `STARTING`，零冷启动时立即 `WARM`。

一个批次只被部分刷新时必须确定性拆分：减少原批次 `count`，为刷新部分生成确定性子批次编号，并分别保存新旧截止时间。不得为了方便刷新整个批次。

`DeploymentPlan` 携带 `expected_lifecycle_version` 与故障/观察快照版本。生命周期层在临时副本完成删除、拆分、刷新、创建和内存审计，全部成功后一次性替换真实状态。明确错误包括：

- `STALE_SNAPSHOT`
- `INVALID_DEPLOYMENT_PLAN`
- `MEMORY_CAPACITY_EXCEEDED`
- `UNHEALTHY_TARGET_NODE`

不得部分提交、投影或自动迁移。

### 4.5 故障数据与实例规则

节点有效状态从可用变为不可用时，立即销毁该节点所有启动中和温实例，释放内存；已产生费用不退还。节点恢复后为空节点，只能由未来慢层计划重新部署。

首版假设 VNF 无状态或状态已外置。逻辑队列由可靠控制面/网络缓冲持有，节点故障不会删除尚未处理的数据；有线网络可靠，在途流量继续按原传播时间到达。

### 4.6 阶段 A 核心测试

- 精确 CTMC 转移概率、稳态分布及 \(\lambda=0\)、\(\mu=0\) 边界；
- 相同种子产生相同故障轨迹；
- 有效状态边沿正确反映故障域变化；
- 冷启动和保留期限恰好位于边界时隙；
- 零冷启动立即变温；
- 启动中实例计入目标数与内存；
- 锁定实例不能提前删除或缩短期限；
- 批次部分刷新可确定性拆分；
- 故障清空实例，恢复节点为空；
- 快照过期或非法计划原子拒绝；
- 实例数只能取 0 或允许的正整数；
- 配置和候选中不存在车载计算节点。

## 5. 阶段 B：跨时隙队列、在途流量与 EDF

### 5.1 队列状态与片段真值

队列层持有上行队列、以 `(stage_id, node_id)` 为键的 VNF 阶段队列，以及在途事件集合。

片段只保存原始输入等效量作为唯一真值：

```text
fragment_id
batch_id
service_id
stage_id
location
routing_target_node  # 可选；未绑定时为 null
input_equivalent_bits
available_slot
```

阶段队列的规范键为：

```text
(stage_id, location, routing_target_node)
```

其中 `routing_target_node=null` 表示尚未绑定执行目标；非空表示该子队列已经绑定到指定健康温节点。已绑定流不能在聚合约束或 EDF 提交时改指向其他目标。只有目标节点失效，编排层才在下一时隙通过显式状态提交清除绑定，使片段重新进入未绑定子队列。

在途状态使用不可变 `InTransitRecord`：

```text
transit_id
fragment
source_node_id
destination_node_id
link_id
departure_slot
arrival_slot
input_equivalent_bits
```

`fragment.input_equivalent_bits` 与记录顶层等效量必须在容差内一致；顶层字段用于快速守恒审计，不构成第二份可修改真值。

设第 \(r\) 个 VNF 前的累计输出比例：

\[
\gamma_r=\prod_{j<r}\rho_j,
\qquad \rho_j>0,\quad\gamma_r>0.
\]

物理数据量由等效量派生：

\[
q_r^{physical}=\gamma_r q_r^{equivalent}.
\]

求解器返回物理服务量后，再除以 \(\gamma_r\) 转换为 EDF 扣减量。所有守恒判断使用统一的绝对与相对容差，避免重复拆分造成双真值漂移。

### 5.2 阶段语义与流水线

队列中的 `stage_id=r` 表示下一步需要执行 VNF \(r\)：

- 上行完成：下一时隙进入第一 VNF 阶段；
- 执行 VNF \(r\)：下一时隙生成阶段 \(r+1\) 片段；
- 执行最终 VNF：生成下一时隙生效的完成事件；
- 有线转发：只改变节点，不改变 `stage_id`；
- 同节点相邻 VNF：仍至少等待一个时隙。

求解器只能消费 `available_slot <= current_slot` 的片段。同一份数据在一个时隙内至多执行一种推进操作：上行、一条有线链路或一个 VNF 阶段。

阶段队列按未绑定/目标绑定子队列分别满足可用量约束。未绑定流可以选择任一健康温目标；已绑定目标 \(d\) 的流只能继续沿目标 \(d\) 的允许链路转发或在 \(d\) 执行：

\[
z_{r,i,t}+\sum_{d,e\in\delta^+(i)}x_{r,d,e,t}
\le Q_{r,i,t}^{physical}.
\]

有线传播到达时隙：

\[
arrival\_slot=current\_slot+
\max\left(1,
\left\lceil\frac{d_e^{prop}}{\Delta t}\right\rceil
\right).
\]

上行片段使用实际上传时隙的当前服务 MEC 作为入口。同一批次跨多个时隙上传时，可以进入不同 MEC。

### 5.3 确定性 EDF

每个批次保存到达时刻、绝对截止时刻、完成时刻、是否已记录违约及最终时延。绝对截止时间在整个 SFC 过程中不重置。

队列内片段固定按下列键升序：

```text
(absolute_deadline, arrival_time, batch_id, fragment_id)
```

提交采用片段优先的嵌套顺序：外层遍历 EDF 片段，内层按固定操作键
`(operation_type, routing_target_node, destination_node_id, link_id)`
遍历执行或传输出口。紧急片段未处理完前，晚截止片段不得绕过。一个片段可被多个出口确定性拆分，但总扣减必须等于计划服务量，且不得超过对应的未绑定或目标绑定子队列。已绑定片段的出口键必须保持同一个 `routing_target_node`。

EDF 只在同一阶段队列内部映射聚合服务，不能跨队列重新分配求解器决定的资源。

### 5.4 故障节点队列

节点故障仅表示 MEC 计算服务不可用：

- 原逻辑队列保留；
- 禁止在故障节点执行 VNF；
- 有线容量允许时可以转发到其他健康温节点；
- 不允许瞬时、无成本的数据迁移。

### 5.5 SLA 与排空期

若完成事件在 `completion_slot` 生效，则按时条件为：

\[
completion\_slot\cdot\Delta t
\le absolute\_deadline\_time.
\]

检查时刻恰好等于截止时间不算违约；严格超过且未完成时记录一次违约。已违约批次继续接受 EDF 服务，并记录最终完成时延和超时时长。

正常业务期结束后停止新到达并进入排空期，故障、恢复、生命周期和调度继续运行，直到：

- 所有已到达批次均完成；或
- 达到最大排空时限。

排空时限后未完成的批次记为 `censored_unfinished_batch`，单独报告数量、剩余等效工作量及其是否已经违约。

### 5.6 原子提交

`FastAllocationPlan` 携带队列、生命周期、故障和网络快照版本及快时隙编号。队列层在临时副本中完成 EDF 扣减、片段拆分、在途事件、输出比例转换、完成量和 SLA 更新，全部通过后原子替换。

明确错误包括：

- `STALE_SNAPSHOT`
- `FLOW_EXCEEDS_AVAILABLE_QUEUE`
- `INVALID_STAGE_TRANSITION`
- `FLOW_CONSERVATION_VIOLATION`
- `INVALID_ALLOCATION_PLAN`

### 5.7 阶段 B 核心测试

- 上行、VNF 和有线流量守恒；
- 输出比例与输入等效量转换，大量拆分后仍一致；
- 数据不能在一个时隙跨越多个阶段；
- 同节点相邻 VNF 仍延迟一时隙；
- 有线传播时隙和移动入口 MEC 正确；
- EDF 排序、部分服务、多出口和确定性拆分；
- 故障节点队列保留、禁止执行但允许转发；
- 截止边界、迟到完成和违约只计一次；
- 排空期、截断批次和成功率分母正确；
- 版本过期或守恒失败时原子拒绝；
- 输入快照不被原地修改。

阶段 B 只建立队列、EDF 和审计器。旧调度器暂时独立保留，到阶段 C 新求解器接通时一并删除。

## 6. 阶段 C：两阶段词典序连续凸优化

### 6.1 输入和版本

无状态求解器只接收：

- `QueueSnapshot`
- `LifecycleSnapshot`
- `FailureSnapshot`
- `NetworkSnapshot`
- 只读模型配置

`NetworkSnapshot` 包括当前服务 MEC、单列车信道增益、上行带宽、最大发射功率、有线容量和传播时延、资源价格及版本。

求解器只把四个输入版本原样写入计划。最终版本匹配由编排层和提交接口完成，求解器不能自行判断环境是否变化。

### 6.2 温实例和目标节点多商品流

VNF \(r\) 只能在有效健康且存在温实例的节点执行。`STARTING` 实例不提供容量。

有线流量带有目标执行节点商品标记：

\[
x_{r,d,e,t}\ge0,\qquad d\in D^{warm}_{r,t}.
\]

最短路只使用当前 `NetworkSnapshot` 中容量严格为正的有线链路。容量为零的链路不生成变量；不可到达目标不生成对应商品。配置要求每条物理链路传播代价严格为正：

\[
d_e^{prop}>0.
\]

为防止极小传播代价或相同最短传播距离产生平势边，势函数使用词典序二元值：

\[
\Phi_{r,d,t}(i)=
\left(d^{shortest}_{i,d},h^{shortest}_{i,d}\right),
\]

其中 \(h^{shortest}\) 是在最短传播距离路径中的最小跳数。只允许
\(\Phi(j)<_{lex}\Phi(i)\) 的链路。若距离严格下降则直接允许；距离相等时必须跳数严格下降。

原距离势函数仍记为：

\[
\phi_{r,d,t}(i)=d^{shortest}_{i,d}.
\]

该规则既防止稳定拓扑环路，又允许已有温实例的过载节点把流量转向其他温节点。

未绑定片段第一次选择目标 \(d\) 后携带 `routing_target_node=d`，进入相应绑定子队列或在途记录。后续优化不得把它重指向其他目标。目标节点失效时，编排层在下一时隙显式清除目标并重新路由。没有任何健康温实例时，该阶段只能保留服务缺口，不触发冷启动或回退。

共享链路容量：

\[
\sum_{r,d}x_{r,d,e,t}\le C_{e,t}\Delta t.
\]

### 6.3 单列车多业务无线模型

首版是一台列车发射机、多个业务类别。变量为：

\[
q^{UL}_{s,t},\quad b^{UL}_{s,t},\quad p^{UL}_{s,t}.
\]

约束：

\[
q^{UL}_{s,t}+\xi^{UL}_{s,t}=Q^{UL}_{s,t},
\]

\[
\sum_s b^{UL}_{s,t}\le B^{UL}_t,\qquad
\sum_s p^{UL}_{s,t}\le P^{max}_t,
\]

\[
q^{UL}_{s,t}\le
\Delta t\,b^{UL}_{s,t}
\log_2\left(1+
\frac{h_t p^{UL}_{s,t}}{N_0b^{UL}_{s,t}}
\right).
\]

其中 \(p^{UL}\) 是整个快时隙内的平均发射功率，\(N_0\) 是噪声功率谱密度，\(h_t\) 在时隙内固定。代码使用相对熵的指数锥表达式，避免直接除以带宽。

无线能耗：

\[
E^{UL}_t=\Delta t\sum_s p^{UL}_{s,t}.
\]

未来若扩展多列车，必须按发射机分别建立功率预算和信道增益，不能复用当前共享总功率约束。

### 6.4 有线流量

有线链路只优化物理传输 bit，不设置传输功率或独立速率：

\[
c_{r,d,e,t}=\frac{x_{r,d,e,t}}{\Delta t}.
\]

链路输出按传播时隙进入在途集合。

### 6.5 工作量—执行时间透视模型

节点 \(i\) 执行 VNF \(r\) 的物理数据量为 \(z_{r,i,t}\)，CPU 工作量为：

\[
w_{r,i,t}=C_rz_{r,i,t}.
\]

优化执行占用时间 \(\tau_{r,i,t}\ge0\)，满足：

\[
w_{r,i,t}\le F^{inst,max}_{r,i}\tau_{r,i,t},
\]

\[
\tau_{r,i,t}\le m^{warm}_{r,i,t}\Delta t,
\]

\[
\sum_r\tau_{r,i,t}\le m_i^{core}\Delta t,
\]

\[
\sum_rw_{r,i,t}\le F_i^{max}\Delta t.
\]

计算能耗闭包透视：

\[
E^{comp}_{r,i,t}\ge
\kappa_i\frac{w_{r,i,t}^3}{\tau_{r,i,t}^2}.
\]

配置必须满足 \(\kappa_i>0\)。用 `PowCone3D(E/κ, τ, w; 1/3)` 的等价形式表达。当 \(w=\tau=0\) 时能耗为零；\(w>0,\tau=0\) 不可行。

为消除 \(w=0\) 时任意正 \(\tau\) 的退化，二级目标包含真实 CPU 占用费：

\[
C_t^{CPU}=\sum_{r,i}\pi_i^{CPU}\tau_{r,i,t},
\qquad \pi_i^{CPU}>0.
\]

求解后使用固定物理时间容差 \(\varepsilon_\tau\) 派生：

\[
f^{active}_{r,i,t}=
\begin{cases}
0,&\tau_{r,i,t}\le\varepsilon_\tau,\\
w_{r,i,t}/\tau_{r,i,t},&\tau_{r,i,t}>\varepsilon_\tau,
\end{cases}
\quad
f^{aggregate}_{r,i,t}=\frac{w_{r,i,t}}{\Delta t},
\quad
\alpha_{r,i,t}=\frac{f^{aggregate}_{r,i,t}}{F_i^{max}}.
\]

CPU 比例和频率不是独立决策变量。

### 6.6 队列和服务缺口

阶段队列拆成未绑定量 \(Q^{unbound}_{r,i,t}\) 和按目标 \(d\) 绑定的量 \(Q^{bound}_{r,d,i,t}\)。对未绑定流：

\[
z^{unbound}_{r,i,t}
+\sum_{d,e\in\delta^+(i)}x^{unbound}_{r,d,e,t}
+\xi^{unbound}_{r,i,t}
=Q^{unbound,physical}_{r,i,t}.
\]

其中 \(z^{unbound}_{r,i,t}\) 只有节点 \(i\) 自身存在该 VNF 的健康温实例时才生成；它等价于在执行时把目标确定为 \(d=i\)。

对已经绑定目标 \(d\) 的流：

\[
\mathbf 1[i=d]z^{bound}_{r,d,i,t}
+\sum_{e\in\delta^+(i)}x^{bound}_{r,d,e,t}
+\xi^{bound}_{r,d,i,t}
=Q^{bound,physical}_{r,d,i,t}.
\]

绑定流只能使用商品 \(d\) 的下降势链路；\(i\ne d\) 时不生成执行变量，\(i=d\) 时不允许改绑到其他目标。总执行量和链路流量分别由未绑定与绑定分量求和后进入计算容量和共享链路容量约束。

所有未绑定和绑定子队列的服务缺口分别转换为输入等效 bit：

\[
\xi_{r,i,t}^{equivalent}
=\frac{\xi_{r,i,t}}{\gamma_r}.
\]

上行物理量与等效量相同。在途数据不属于本时隙可服务队列，不进入缺口分母。

### 6.7 两阶段词典序目标

一级目标最小化紧迫度加权服务缺口：

\[
J^*_{1,t}=\min
\sum_q\omega_{q,t}\xi_{q,t}^{equivalent},
\]

\[
\omega_{q,t}=
\frac{\Delta t}{\max(s_{q,t},\Delta t)}\in(0,1].
\]

其中 \(s_{q,t}\) 是队列中最早绝对截止时间的剩余松弛时间。

二级目标在保持一级近最优的条件下最小化真实资源成本：

\[
\min C_t^{energy}+C_t^{CPU}+C_t^{wired}+C_t^{cloud},
\]

\[
J_{1,t}\le J^*_{1,t}+\varepsilon_{lex},
\]

\[
\varepsilon_{lex}=\max
\left(\varepsilon_{abs},
\varepsilon_{rel}\max(1,|J^*_{1,t}|)\right).
\]

成本边界：

- `energy`：无线与计算能耗折算费用；
- `CPU`：处理器占用费用；
- `wired`：按实际传输 bit 计费；
- `cloud`：云平台额外使用费。

若云价格已包含某项 CPU 或能耗费用，不得重复计费。所有成本统一成货币单位，不使用大惩罚系数。

### 6.8 数值尺度与结果审计

求解器内部固定使用适合数值计算的尺度，例如 Mbit、Gcycle、GHz、秒、J/mJ 和统一货币单位；结果再转换回物理单位。残差同时报告归一化尺度和物理尺度。

只有两阶段均通过 DCP、CLARABEL 状态可接受、变量有限、所有约束残差达标且二级解未破坏一级目标时，才返回计划。

计划记录：两阶段状态与耗时、\(J_1^*\) 和二级 \(J_1\)、服务缺口、成本分解、链路/节点利用率、功率、带宽、派生频率、CPU 比例和最大残差。

数值失败返回 `FAST_SOLVER_FAILURE`，不提交资源服务量，不触发冷启动或启发式回退。运行模式决定后续行为：

- 评估模式：继续推进时钟和 SLA 审计，并统计求解器失败率；
- 训练模式：保存诊断后截断当前尚未提交的事务式 rollout，不把故障转移写入 PPO 缓冲区。

### 6.9 阶段 C 替换和测试

新求解器接通后，同一提交删除旧路径比例优化、固定路径成本替代资源分配、快层冷启动回退及对应配置和测试。

核心测试覆盖：DCP、零队列/零信道边界、温/冷实例容量、单实例与节点总容量、多商品无环流、一级优先、二级容差、可手算场景、容量不足仍有有限缺口、归一化尺度、状态/耗时/残差、版本过期、求解失败无回退和快照不可变。

## 7. 两级可靠性模型

### 7.1 解码安全条件

令 \(A_k\) 表示 VNF \(k\) 不可用。由并集界：

\[
P(SFC\ unavailable)\le\sum_{k=1}^{K}P(A_k).
\]

解码器要求：

\[
P(A_k)\le\frac{1-R^{min}}{K},\qquad\forall k.
\]

该条件不要求不同 VNF 故障事件独立，但只是充分条件。

设故障域 \(g\) 稳态可用度为 \(a_g\)，节点 \(i\) 在域正常条件下的稳态可用度为 \(a_i\)，VNF \(k\) 在域 \(g\) 的部署节点集合为 \(S_{k,g}\)。在域之间独立、域内节点条件独立的首版假设下：

\[
P(A_k)=
\prod_{g:S_{k,g}\ne\varnothing}
\left[(1-a_g)+a_g
\prod_{i\in S_{k,g}}(1-a_i)\right].
\]

同一节点多个实例只增加处理容量，对节点级可靠性只贡献一次。

### 7.2 精确理论审计

在共享故障状态 \(\omega\) 下精确枚举：

\[
R_{SFC,t}^{exact}=
\sum_{\omega\in\Omega}P(\omega)
\mathbf{1}
\left[
\forall k,\exists i:
m^{warm}_{k,i,t}>0,
G_{g(i)}(\omega)=1,
N_i(\omega)=1
\right].
\]

在“不同故障域 CTMC 相互独立、域正常条件下不同节点局部 CTMC 相互独立”的首版假设下，该枚举是精确计算，并保留由多个 VNF 共享节点、多个节点共享故障域产生的相关性。它不代表任意相关故障模型。首版假设有线链路可靠；未来加入链路故障时，状态检查还需验证相邻 VNF 之间存在可用路径。

### 7.3 三类指标

- 结构可靠性 \(R_T^{struct}\)：按冷启动完成后的目标支持集合计算，用于安全解码；锁定但目标为零的超额实例不贡献结构可靠性。
- 温实例快照可靠性 \(R_t^{warm}\)：按当前实际温实例结构、但使用稳态故障概率计算。
- 实际服务成功率：由 CTMC 轨迹、冷启动、容量、队列和 EDF 的按时完成结果统计。

当前故障状态不替代概率可靠性。安全集合为空只返回 `NO_SAFE_FEASIBLE_DEPLOYMENT`，不能声称物理上绝对无解。论文报告：

\[
\Delta R_t=R_{SFC,t}^{exact}-R^{min}
\]

以及并集界造成的资源冗余或可行解损失。

## 8. 阶段 D：配置驱动状态、动作和安全解码

### 8.1 规模和动作规格

允许部署组合：

\[
P=\{(k,i):VNF\ k\ allows\ node\ i\}.
\]

车载节点不在 \(P\) 中。动作维度：

\[
D_a=2|P|,
\]

所有组合都合法时退化为 \(2KI\)。动作采用规范交错顺序：

```text
(function_id, node_id, instance_count_score)
(function_id, node_id, retention_time_score)
```

`ActionSpec` 明确记录实体顺序、索引、允许实例集合、保留时间档位、版本和 SHA-256 哈希，不依赖字典遍历。

不同场景规模自动生成不同网络尺寸，代码无需修改，但重新初始化并训练模型。首版不宣称检查点跨规模直接复用。

### 8.2 观察规格

`ObservationSpec` 明确每个特征的名称、公式、顺序、维度、固定物理尺度、窗口、EWMA 系数、空集合默认值和截断范围。

所有实例批次摘要共用一个明确公式。对带实例数权重的集合 \(\{(x_b,c_b)\}\)：

\[
count=\sum_bc_b,
\quad x_{min}=\min_{b:c_b>0}x_b,
\quad x_{max}=\max_{b:c_b>0}x_b,
\quad \bar x=\frac{\sum_bc_bx_b}{\sum_bc_b}.
\]

空集合的 `count/min/max/weighted_mean` 全部为 0。冷启动剩余时间使用
\(x_b=\max(0,ready\_slot-current\_slot)\)；锁定剩余期限使用
\(x_b=\max(0,deadline-current\_slot)\)，且只统计 `current_slot < deadline` 的批次。

业务 \(s\) 的到达率只使用慢帧历史 EWMA：

\[
\widehat\lambda_{s,T}=\beta_{arrival}\widehat\lambda_{s,T-1}
+(1-\beta_{arrival})\frac{A_{s,T-1}^{eq}}{H},
\]

初值为 0，\(H=N_{slow}\Delta t\)，\(\beta_{arrival}\in[0,1)\) 从配置读取。不能使用当前慢帧尚未完整观测的到达量。

SLA 摘要固定为：风险批次数、累计已违约批次数，以及尚未完成且未违约批次的最小剩余松弛时间；空集合松弛为 0。上帧资源利用率统一为“实际使用量/该帧有效可用容量”，分母为零时利用率为 0 并同时提供 `capacity_available` 二值特征。上一慢帧服务缺口、成本和违约数分别使用固定队列量尺度、\(C^{ref}\) 和批次数尺度归一化。

所有数量特征按 `ObservationSpec` 保存的正尺度除法后裁剪到预设区间；松弛时间保留正负号并裁剪到 `[-maximum_sla_slots, maximum_sla_slots]` 后再缩放到 `[-1,1]`。这些尺度属于结构规格，不随训练数据改变。

特征至少包括：

- 实验进度
  \(t/(T_{normal}+T_{drain}^{max})\)（截断到 `[0,1]`）、排空期标记 `is_drain_phase` 与慢帧内位置 \((t\bmod N_{slow})/N_{slow}\)；
- 列车位置、速度、当前/下一服务 MEC 的固定 one-hot；
- 节点域/节点隐藏状态和有效健康状态、内存、CPU、核心数、可用度；
- 实际、温、启动中和锁定实例数；
- 冷启动剩余时隙的最小值、最大值和按实例数加权均值；
- 锁定数量、剩余期限最小值、最大值和加权均值；
- 按上述固定公式计算的上一慢帧到达率 EWMA；
- 阶段队列、最小剩余松弛、已违约和未违约批次数；
- 上帧执行、转发、服务缺口、成本和资源利用率；
- 结构可靠性和温实例可靠性；
- VNF 输出比例、累计比例、CPU 周期需求和 SLA 参数。

输入隐藏状态等价于完全监测假设：控制器能在每个时隙边界检测域和节点状态。

规格哈希只包含结构和固定归一化方法。运行均值、方差和样本数属于独立 `NormalizationState`，不会改变规格哈希。评估时冻结；恢复训练时从检查点继续更新。

### 8.3 统一观察版本

`ObservationSnapshot` 携带：

- `queue_version`
- `lifecycle_version`
- `failure_version`
- `network_version`
- `observation_bundle_version`

部署计划原样保存并在提交前校验整个观察包。结构哈希不受随机种子、回合数、输出目录、日志频率和测试轨迹编号影响。

### 8.4 PolicyAdapter、连续评分和保留语义

扩散网络最后一步产生无界有限向量 \(v\in\mathbb{R}^{D_a}\)。独立的确定性 `PolicyAdapter` 使用 sigmoid 得到评分：

\[
u=\sigma(v)\in(0,1)^{D_a}.
\]

网络和 PPO 对无界扩散链 \(v\) 计算联合概率；环境解码只读取适配后的 \(u\)。`PolicyAdapter` 不执行裁剪或可行性修复。仅当网络输出或适配结果非有限时返回 `INVALID_POLICY_SCORE`。

教师反向编码生成的评分必须位于开区间 \((0,1)\)：每个离散候选使用其量化区间中心，无效保留评分为 0.5。扩散 BC 的无界目标由
\(v^{teacher}=logit(u^{teacher})\) 唯一得到，因此训练目标与在线 `sigmoid` 适配严格互逆，不对 0 或 1 做数值裁剪。

因此解码评分为：

\[
u^n_{k,i,T},u^L_{k,i,T}\in[0,1].
\]

对测试或外部接口直接构造的评分，越界同样返回 `INVALID_POLICY_SCORE`。解码器不修复分数。

实例数评分在安全候选集合中确定性量化。保留时间评分映射到由最小值、最大值、步长生成并去重排序的整数时隙档位。目标实例数为零时，保留评分不产生环境效果，也不刷新锁定超额实例。

### 8.5 安全模式生成

每个 VNF 先枚举支持集合：

\[
S_k=\{i:n_{k,i}>0\}.
\]

支持集合必须满足节点健康、允许部署、最小副本节点数、故障域分散和 VNF 并集界预算；再为支持节点枚举合法正整数实例数，其余取零。

“精确搜索”是指在并集界定义的安全集合内完整精确搜索，不代表枚举全部物理可靠部署。

内存按提交后的实际实例计算：

\[
\sum_k\max(m^{locked}_{k,i},n_{k,i})M_k\le M_i^{max}.
\]

不能先扣全部锁定内存再重复计费目标实例。

### 8.6 顺序解码与全局可完成性

动作位置固定按 `(VNF优先级, 节点优先级, function_id, node_id)`。每个候选值都必须：

1. 固定当前值；
2. 过滤与已解码结果矛盾的模式；
3. 对剩余 VNF 进行完整组合搜索；
4. 仅在仍存在完整安全部署时保留。

候选整数升序。若集合为 \(A\)，则：

\[
index=\left\lfloor u^n(|A|-1)+0.5\right\rfloor.
\]

\(|A|=1\) 时选择唯一值。显式使用上述舍入，不使用银行家舍入。

初始安全集合经完整枚举为空时返回 `NO_SAFE_FEASIBLE_DEPLOYMENT`。若初始已证明非空，而顺序解码意外得到空候选，则返回 `DECODER_INTERNAL_FAILURE`。

### 8.7 分支定界与缓存

搜索 VNF 顺序固定为 `(兼容模式数, -VNF优先级, function_id)`。剪枝包括内存超限、剩余内存不足、VNF 无兼容模式、剩余节点/故障域不足。

缓存键包括配置/规格哈希、健康掩码、可靠性目标、搜索位置、各节点剩余内存、锁定实例状态和允许模式集合规范标识。

确定性上限：

- `max_generated_patterns`
- `max_expanded_states`
- `max_cache_entries`

不能以墙钟超时决定结果。达到限制返回 `DECODER_SEARCH_LIMIT`，且不得使用已生成的部分集合。数值错误或不变量破坏返回 `DECODER_INTERNAL_FAILURE`。

诊断记录各 VNF 模式数、展开/剪枝数、缓存命中率、每个动作位置的原始/可完成候选数、耗时和最终状态码。

### 8.8 计划、掩码与检查点

成功计划包含实例数、保留时间、观察包版本、规格哈希和搜索诊断。`effective_action_mask` 标记目标为零时无环境效果的保留维度。

检查点保存有序实体、允许组合、完整规格和哈希、模型结构、扩散参数、归一化状态及版本、优化器与训练进度。结构不匹配返回 `CHECKPOINT_SPEC_MISMATCH`，禁止截断、填充、重排或部分加载。

### 8.9 阶段 D 删除与测试

新接口接通后删除独立 `replica_count`、旧动作维度、部署模板、通用投影器、启发式回退、旧状态排列和兼容解析。

核心测试覆盖配置驱动维度、禁止组合、稳定顺序、可靠支持集合、锁定内存、后续可完成性、确定性计划、保留映射、错误状态区分、确定性搜索限制、检查点规格拒绝和旧逻辑清除。

## 9. 阶段 E：奖励、教师与训练

### 9.1 慢帧奖励

本慢帧首次违约率：

\[
V_T=
\begin{cases}
N_T^{new\_violation}/|B_T^{risk}|,&|B_T^{risk}|>0,\\
0,&\text{otherwise}.
\end{cases}
\]

\(B_T^{risk}\) 是本慢帧内曾处于“尚未完成且尚未记录违约”的批次集合，包括以前到达的批次。旧违约不重复计分。

队列服务缺口率：

\[
D_T=
\begin{cases}
\dfrac{\sum_{t\in T}\sum_q\xi_{q,t}^{equivalent}}
{\sum_{t\in T}\sum_qQ_{q,t}^{equivalent}},&
\sum Q>0,\\
0,&\text{otherwise}.
\end{cases}
\]

其中 \(q\) 包含上行和全部阶段—节点队列，不含在途量。

总资源成本：

\[
C_T^{raw}=C_T^{deployment}+C_T^{cold}+C_T^{retention}
+C_T^{energy}+C_T^{CPU}+C_T^{wired}+C_T^{cloud},
\]

\[
C_T=clip(C_T^{raw}/C^{ref},0,1).
\]

\(C^{ref}\) 只使用配置上界推导，不使用基线结果或训练/验证轨迹。固定算法为：

\[
C^{ref}=\max\left(
C_{floor},
C_{deploy}^{max}+C_{cold}^{max}
+N_{slow}\left(
C_{retention}^{max}+C_{UL-energy}^{max}
+C_{comp-energy}^{max}+C_{CPU}^{max}
+C_{wired}^{max}+C_{cloud}^{max}
\right)
\right).
\]

各最大项使用与运行计费完全相同的函数和价格，只把实例数、功率、处理时间、链路 bit 和云用量替换成配置上限。其中计算能耗采用确定性安全上界
\(\sum_{k,i}\kappa_i n_{k,i}^{max}(F_{k,i}^{inst,max})^3\Delta t\)，即使节点总容量使该上界偏松也不在实验后调整。配置加载时生成并保存 `reference_cost_algorithm_version`、各分项上界和最终数值；检查点再次保存并严格校验。\(C_{floor}>0\) 只防止零价格场景除零，也由配置固定。论文同时报告未经归一化的真实货币成本。

服务损失和最终奖励：

\[
S_T=V_T+D_T-V_TD_T,
\]

\[
r_T=-[\alpha S_T+(1-\alpha)C_T],
\qquad \alpha=0.8.
\]

论文主实验和所有主模型选择固定使用 \(\alpha=0.8\)。\(\alpha=0.7\) 和 \(0.9\) 只作为预先声明的敏感性实验，分别独立训练和报告，不能参与主模型或主检查点选择。

可靠性、冷启动、功率、CPU、带宽、部署修改和投影率不作为独立奖励项。`NO_SAFE_FEASIBLE_DEPLOYMENT` 是有效环境结果；解码搜索限制、内部错误、非法评分、求解失败和快照错误是内部失败，不能伪装成策略奖励。

### 9.2 三类因果教师

教师只能接收当前 `ObservationSnapshot` 与只读配置：

\[
T_j:(ObservationSnapshot,Config)\rightarrow DeploymentPlan.
\]

禁止访问未来故障、未来真实负载、未来信道或完整环境对象；允许使用观察中的历史到达率、当前/下一服务 MEC 和因果预测值。

三类教师共用一个完全因果的单慢帧代理评估器。令慢帧时长 \(H=N_{slow}\Delta t\)，`ObservationSpec` 中固定的到达率 EWMA 为 \(\hat\lambda_s\)。VNF \(k\) 在下一慢帧需要处理的原始输入等效工作量代理为：

\[
\widehat B_k^{eq}=
Q_{UL}^{eq}
+\sum_{r\le k}\sum_i Q_{r,i}^{eq}
+\sum_{v:\,stage(v)\le k}B_v^{in\ transit,eq}
+H\sum_s\hat\lambda_s.
\]

这里 \(Q_{UL}^{eq}\) 只表示尚未上传的原始输入；阶段求和只包含“下一步尚未执行到 VNF \(k\)”的数据；在途求和采用同样的下一阶段条件。因此同一片数据在快照中只属于上行、某个阶段队列或某条在途记录，并且只在尚需 VNF \(k\) 服务时计入 \(\widehat B_k^{eq}\)。新到达代理对每个尚未执行的 VNF 各贡献一次未来工作量。物理 bit 与周期需求分别为
\(\widehat B_k^{physical}=\gamma_k\widehat B_k^{eq}\) 和
\(\widehat W_k=C_k\widehat B_k^{physical}\)。

对计划 \(a\) 的目标节点，定义慢帧代理容量
\(cap_{k,i}=n_{k,i}F_{k,i}^{inst,max}H\)。当前和下一服务 MEC 到节点 \(i\) 的正容量最短路径成本均值记为 \(\bar d_i\)，配置固定正尺度 \(d_{scale}\)。代理工作量按闭式、确定性份额分配：

\[
\widehat W_{k,i}=\widehat W_k
\frac{cap_{k,i}/(1+\bar d_i/d_{scale})}
{\sum_{j:n_{k,j}>0}cap_{k,j}/(1+\bar d_j/d_{scale})}.
\]

不可达节点的份额为零；若计划支持集合没有任何可达节点，则代理目标为正无穷并排在所有可评估计划之后。

节点 CPU 和内存代理利用率为：

\[
u_i^{CPU}=\frac{\sum_k\widehat W_{k,i}}{F_i^{max}H},
\qquad
u_i^{memory}=\frac{\sum_k\max(m_{k,i}^{locked},n_{k,i})M_k}
{M_i^{max}},
\]

\[
u_i(a)=\max(u_i^{CPU},u_i^{memory}).
\]

利用率不裁剪，预测过载可表现为大于 1。均衡指标明确定义为：

\[
J_{balance}(a)=
\left(\max_i u_i(a),
\frac{1}{|I|}\sum_i(u_i(a)-\bar u(a))^2\right).
\]

成本代理 \(J_{cost}(a)\) 使用与环境完全相同的价格字段，并固定为：

\[
\widehat n^{new}_{k,i}=\max(0,n_{k,i}-n^{existing}_{k,i}),
\qquad
\widehat n^{post}_{k,i}=\max(m^{locked}_{k,i},n_{k,i}),
\]

\[
\widehat\tau_{k,i}=\begin{cases}
0,&\widehat W_{k,i}=0,\\
\widehat W_{k,i}/(n_{k,i}F_{k,i}^{inst,max}),&\widehat W_{k,i}>0,
\end{cases}
\]

\[
\widehat E^{comp}_{k,i}=\begin{cases}
0,&\widehat W_{k,i}=0,\\
\kappa_i\widehat W_{k,i}^3/\widehat\tau_{k,i}^2,&\widehat W_{k,i}>0.
\end{cases}
\]

若正工作量被分给零实例、\(\widehat\tau_{k,i}>n_{k,i}H\)，或节点聚合代理超过核心时间/CPU 周期上限，则该计划的成本和均衡代理记为正无穷。

计划相关成本为：

\[
\begin{aligned}
J_{cost}(a)=
&\sum_{k,i}(\pi^{deploy}_{k,i}+\pi^{cold}_{k,i})
\widehat n^{new}_{k,i}\\
&+\sum_{k,i:n_{k,i}>0}
\pi^{retention}_{k,i}n_{k,i}L_{k,i}\\
&+H\sum_{k,i}\pi^{running}_{k,i}\widehat n^{post}_{k,i}\\
&+\sum_{k,i}\left(
\pi_i^{CPU}\widehat\tau_{k,i}
+\pi^{energy}\widehat E^{comp}_{k,i}
\right)\\
&+\widehat C^{wired}(a)+\widehat C^{cloud}(a)
+\widehat C^{UL}.
\end{aligned}
\]

其中 \(\widehat C^{wired}\) 按当前/下一服务 MEC 各占一半、目标节点正容量最短路径的逐链路 bit 价格计算；\(\widehat C^{cloud}\) 按分配到云的物理 bit 和 CPU 周期使用配置云价格计算。\(\widehat C^{UL}\) 使用预测上行量、当前 \(h_t\)、全上行带宽 \(B^{UL}\) 和时长 \(H\) 反解满足香农约束的最小平均功率：

\[
\widehat p^{UL}=\frac{N_0B^{UL}}{h_t}
\left(2^{\widehat Q^{UL}/(HB^{UL})}-1\right),
\qquad
\widehat C^{UL}=\pi^{energy}H\widehat p^{UL}.
\]

若 \(h_t\le0\) 且预测上行量为正，或 \(\widehat p^{UL}>P^{max}\)，代理值为正无穷。该无线项对同一状态中的部署计划是共同常数，但保留在完整成本报告中。

这里不运行未来仿真，也不读取未来实际轨迹。代理评估时点固定为当前慢帧边界；当前/下一服务 MEC 各占路径代理的一半。所有公式、价格版本、\(H\)、\(d_{scale}\) 和 EWMA 参数均写入教师数据规格。

规范排序键固定为：先按 `ActionSpec` 顺序展开全部目标实例数，再按相同顺序展开保留时隙数，最后附加计划 SHA-256；数值元组按升序比较。

对安全计划集合 \(F^{safe}(s)\)，三类无权重词典序教师为：

成本教师先按规则把所有已部署组合的保留时间固定为最短合法档，再比较：

\[
\min_{a\in F^{safe}(s)}
(J_{cost}(a),-\Delta R(a),J_{balance}(a),key(a)).
\]

可靠性教师先把所有已部署组合的保留时间固定为最长合法档，再采用强调持续保障的唯一顺序：

\[
\min_{a\in F^{safe}(s)}
\left(-\Delta R(a),
J_{cost}(a),J_{balance}(a),key(a)\right).
\]

均衡教师先按下述队列清空时间规则固定保留档，再比较：

\[
\min_{a\in F^{safe}(s)}
(\max_i u_i(a),Var_i[u_i(a)],J_{cost}(a),-\Delta R(a),key(a)).
\]

保留时间规则：成本教师取最短档；可靠性教师取最长档；均衡教师对每个已部署组合计算
\(\widehat t_{clear,k,i}=\widehat W_{k,i}/(n_{k,i}F_{k,i}^{inst,max})\)，再选择不小于该时间的最近合法档，超过最大档时取最大档。这些都是因果预训练启发，不声称未来最优。

标签写入前必须通过安全/内存审计、反向编码、同快照重新解码完全一致，以及生命周期只读原子提交预审。相同状态的重复计划按计划哈希去重，并记录多个 `teacher_types`。

目标实例为零时，无效保留评分统一编码为 0.5。`NO_SAFE_FEASIBLE_DEPLOYMENT` 和任何内部/搜索失败不生成标签。

### 9.3 教师数据隔离

先按 `source_trajectory_id` 划分教师训练集和教师验证集，再保证相同 `state_group_id` 不能跨集合。同一状态的全部教师计划必须在同一集合。教师验证集只验证预训练效果，不参与在线检查点选择。

数据集保存状态/计划哈希、原始评分、有效掩码、计划、教师类型、三个目标值、规格版本和预审结果。数据只来自训练轨迹。

### 9.4 预训练辅助任务

预训练损失：

\[
L_{pretrain}=L_{diffusion}+
0.1\frac{L_{rank}+L_{count}+L_{retention}}{3}.
\]

辅助头只能读取纯观察编码器，不得读取教师动作、扩散噪声或带噪动作。

节点排序使用无间隔成对逻辑损失：

\[
L_{rank}=\frac{1}{|E|}\sum_{(i,j)\in E}
softplus[-(g_{k,i}-g_{k,j})].
\]

实例数对所有允许组合做分类交叉熵，并屏蔽配置不允许类别。实例数评分始终有效，包括目标为零。

保留时间只在教师目标实例数大于零时计算分类交叉熵；若整批无有效标签则为零。`effective_action_mask` 只用于保留时间辅助损失，不用于实例数和节点排序。

辅助头推理时关闭，不生成环境动作。

### 9.5 联合扩散 PPO 概率

原始评分 \(u=(u^n,u^L)\) 来自联合扩散策略，环境动作是确定性映射：

\[
u_T\sim\pi_\theta(\cdot|s_T),
\qquad a_T=g(s_T,u_T).
\]

PPO 新旧策略概率、概率比和 KL 始终使用完整原始动作，不能按 `effective_action_mask` 删除维度。对联合扩散策略，坐标屏蔽不等价于边缘化。

掩码只用于环境语义、轨迹校验、统计和保留时间辅助损失。首版不加入坐标级 KL、额外熵奖励、投影惩罚或动作修改惩罚。

### 9.6 在线早期 BC

在线损失：

\[
L_{online}=L_{PPO}+\beta_{BC}(m)L_{diffusion\_BC},
\]

\[
\beta_{BC}(m)=0.05\max
\left(0,1-\frac{m}{\lceil0.2M\rceil}\right),
\quad m=0,\ldots,M-1.
\]

每个 PPO 梯度步骤额外抽取一个同批量大小教师批次，两项损失分别取均值。BC 覆盖完整原始动作，在线不再加入三个辅助损失。在前 20% 更新后 BC 严格归零。

### 9.7 三套轨迹清单

训练、固定验证和最终测试清单完全隔离，保存完整哈希并检查轨迹 ID 与基础种子集合互不重叠。每份清单保存故障、到达、信道、移动和策略噪声种子、仿真时长、排空期及结构规格哈希。

每次慢决策的策略噪声种子由下列元组确定性派生：

```text
(trajectory_id, diffusion_repeat_id,
 slow_decision_index, base_policy_noise_seed)
```

避免因不同检查点排空长度或决策次数不同而发生随机流错位。

### 9.8 PPO 核心复现配置

所有影响 PPO/GAE/扩散概率的参数都必须来自版本化配置，并完整保存到检查点，不能依赖源码默认值。首版调试配置固定为：

```text
environment_discount_gamma: 0.99
gae_lambda: 0.95
normalize_advantages: true
advantage_normalization_scope: complete_committed_rollout
advantage_minimum_std: 1.0e-8
value_loss: mean_squared_error
policy_learning_rate: 1.0e-4
value_learning_rate: 3.0e-4
ppo_update_epochs: 10
ppo_minibatch_size: 64
gradient_clip_norm: 5.0
denoising_steps: 20
fine_tuned_denoising_steps: 5
denoising_discount_gamma: 0.99
clip_ratio_base: 0.001
clip_ratio_growth_rate: 3.0
probability_min_std: 0.10
target_kl: 1.0
entropy_bonus_coefficient: 0.0
```

正式规模配置可以在训练前另行声明，但必须生成新的完整配置哈希，不能在运行中修改。

GAE 先在完整、已经事务提交的环境 rollout 上按 \(\gamma=0.99\)、\(\lambda=0.95\) 计算；启用优势归一化时，只在整条 rollout 上用总体均值和总体标准差归一化一次。之后才把同一个环境优势扩展到最后 5 个可训练去噪步骤，并按从早到晚的指数 \(4,3,2,1,0\) 乘以去噪折扣 \(0.99^e\)。

对每个可训练去噪转移 \(j\)，反向核使用对角高斯。完整原始动作的联合对数密度严格定义为各坐标对数密度之和：

\[
\ell_{T,j}=\sum_{d=1}^{D_a}
\log\mathcal N(v_{T,j+1,d};
\mu_{\theta,j,d},\sigma_{j,d}^2).
\]

不同场景规模分别训练并重新校准 `clip_ratio`，因此不能为减弱动作维度影响而把联合对数密度改成坐标均值。PPO 不把不同去噪步骤先求和成单一比率，而是按步骤计算
\(r_{T,j}=\exp(\ell_{T,j}^{new}-\ell_{T,j}^{old})\)，使用从 `clip_ratio_base` 到所选 `clip_ratio` 的指数裁剪调度，再对环境样本、去噪步骤和 minibatch 取均值。这里“完整原始动作概率”是指每个去噪核都包含全部动作坐标；不得用 `effective_action_mask` 删除坐标。

价值网络使用未裁剪均方误差单独优化，不与策略损失再乘一个价值系数。策略和价值优化器分别裁剪全局梯度范数至 5.0。每次 update 打乱环境样本但保持其完整去噪链不拆散；同一批 committed rollout 重复 10 个 epoch。近似 KL 达到 `target_kl` 时，在任何新策略/价值参数更新前停止当前及后续 minibatch。

### 9.9 多种子 clip_ratio 校准

从训练清单单独划出校准子集。候选例如 `[0.05, 0.10, 0.15]`，使用相同预训练检查点、预算、公共轨迹、归一化状态和求解器配置。

稳定候选必须同时满足：

- 内部失败数为零；
- 有效 PPO 更新数达到要求；
- 损失、概率、梯度和参数全部有限；
- \(0\le clip\_fraction\le1\)；
- \(KL_{max}\le KL_{limit}\)。

无合格候选返回 `NO_STABLE_CLIP_RATIO`。合格候选按固定精度词典序比较验证 SLA、服务缺口、原始成本、最大 KL 和 `clip_ratio`，完全相同时选更小值。

选择完成后，正式在线训练必须重新加载完全相同的预训练检查点，从更新 0 开始，不能继续使用校准模型。

### 9.10 事务式 PPO 轨迹

一个回合或尚未更新的 rollout 先进入暂存缓冲区。只有整段无内部失败时才提交到正式 PPO 缓冲区。

内部失败时：

- 丢弃自上次成功更新后尚未使用的暂存转移；
- 不回滚已经完成的历史 PPO 更新；
- 当前回合立即截断并重置环境；
- 保存失败状态、版本、扩散链和求解器诊断；
- 超过配置失败上限时终止训练任务。

`NO_SAFE_FEASIBLE_DEPLOYMENT` 是合法环境转移，可进入缓冲区。

### 9.11 固定验证和检查点

验证使用检查点自身归一化状态并冻结，启用 eval 模式和无梯度执行，不修改模型、优化器、训练随机状态或缓冲区。所有检查点从相同初始仿真状态运行相同轨迹与噪声。

验证跨轨迹与扩散重复采用微平均：

\[
V^{validation}=\frac{\sum_jN_j^{new\_violation}}
{\sum_jN_j^{risk}},
\]

\[
D^{validation}=\frac{\sum_{j,t,q}\xi_{j,q,t}^{equivalent}}
{\sum_{j,t,q}Q_{j,q,t}^{equivalent}}.
\]

资源成本包含正常业务期和排空期的真实货币成本。固定验证成本口径为“每条外部轨迹总成本的微平均”：

\[
C_{validation}^{raw}=
\frac{\sum_j C_j^{raw}}{N_{external\ trajectories}}.
\]

若每条外部轨迹具有多个扩散噪声重复，则先对同一外部轨迹的重复总成本取平均，再在外部轨迹间取平均，避免重复数量改变轨迹权重。该口径不除以快时隙或输入 bit；单位成本仅作为附加报告指标。

只在零内部失败集合中选择：

\[
K^{valid}=\{k:N_k^{internal\ failure}=0\}.
\]

若为空，返回 `NO_VALID_CHECKPOINT`。指标按训练前配置精度量化：

\[
Q_\epsilon(x)=\left\lfloor\frac{x}{\epsilon}+0.5\right\rfloor.
\]

在合格集合按下列元组升序选择：

\[
(Q_{\epsilon_V}(V),Q_{\epsilon_D}(D),
Q_{\epsilon_C}(C^{raw}),update\_index).
\]

完全相同时选择更早更新。

训练保存预训练、验证最佳和最终检查点。论文主模型预先固定为每个训练种子的 `validation_best_checkpoint`；预训练和最终检查点仅作消融或诊断。最终测试只运行预登记评估，不能反向替换主模型。

多训练种子各自选择验证最佳，在共同最终测试清单上测试。置信区间使用按训练种子或外部轨迹分层 Bootstrap，不能把同一外部轨迹上的多个扩散噪声重复当作完全独立样本。

### 9.12 训练顺序与运行设备

1. 生成互斥轨迹清单；
2. 采集训练观察并生成三教师标签；
3. 按轨迹分组划分教师数据；
4. 扩散 BC 预训练；
5. 在独立训练校准子集上选择稳定 `clip_ratio`；
6. 重新加载预训练检查点，从 0 开始正式 DPPO；
7. 前 20% 更新使用衰减 BC；
8. 固定验证选择零内部失败的最佳检查点；
9. 训练结束前锁定主模型；
10. 最终测试清单只执行预登记评估。

PyTorch 模型、扩散采样和梯度使用 GPU；CLARABEL 每时隙凸优化使用 CPU。

### 9.13 实时指标与扩大训练门槛

VS Code 终端每次在线更新只显示：

```text
update / episode
mean_reward
validation_V / validation_D / raw_cost
approx_kl / clip_fraction
service_deficit / SLA_new_violations
decoder_status / fast_solver_failure_count
CLARABEL mean / P95 time
checkpoint_status
```

详细功率、带宽、CPU、实例、可靠性和成本分解写入 CSV/JSON。

只有阶段 A–D 核心测试通过、最小端到端正常期与排空期通过、内部失败为零、CLARABEL 残差/耗时可接受、奖励可手算复核、三类检查点可严格加载、GPU/CPU 联合资源稳定后，才扩大训练规模。

## 10. 阶段 E 核心测试

- 奖励公式、分母和零数据边界；
- 无效保留评分不影响环境但仍进入完整 PPO 概率；
- 教师不读取未来信息，轨迹和状态组不跨数据集；
- 三教师去重、反向编码、重新解码和生命周期预审一致；
- 辅助头不能读取动作或扩散噪声；
- 只有保留时间辅助损失使用有效掩码；
- 在线 BC 在规定更新后严格归零；
- clip 校准满足 KL 硬门槛，失败返回 `NO_STABLE_CLIP_RATIO`；
- 正式训练从预训练检查点重新开始；
- 事务式 rollout 在内部失败时不进入 PPO 缓冲区；
- 固定验证不改变模型、归一化或随机状态；
- 无零失败检查点时返回 `NO_VALID_CHECKPOINT`；
- 排空期继续处理已违约批次并报告截断；
- 规格不一致检查点明确拒绝；
- 最小 GPU DPPO 与 CPU CLARABEL 联合流程跑通。

## 11. 工程状态与错误分层

有效环境结果：

- `NO_SAFE_FEASIBLE_DEPLOYMENT`
- 正常容量不足及相应服务缺口
- 实际故障、冷启动等待和 SLA 违约

时隙/轨迹内部故障：

- `DECODER_SEARCH_LIMIT`
- `DECODER_INTERNAL_FAILURE`
- `INVALID_POLICY_SCORE`
- `FAST_SOLVER_FAILURE`
- `STALE_SNAPSHOT`
- 非有限概率、损失、梯度或参数

这类故障在训练模式触发当前事务式 rollout 丢弃、回合截断和环境重置；在评估模式继续推进可审计状态，并单独记录失败率。

工作流终止状态：

- `CHECKPOINT_SPEC_MISMATCH`
- `NO_STABLE_CLIP_RATIO`
- `NO_VALID_CHECKPOINT`

工作流终止状态发生在加载、校准或检查点选择边界，不进入环境 step，也不走回合失败清理逻辑。调用方必须停止对应工作流并报告原因。

任何内部故障或工作流终止状态均不得转换成启发式部署、固定模板、投影动作或冷启动回退。

## 12. 论文实验口径

不同 MEC/VNF 规模分别创建并训练相应尺寸模型。规模实验衡量算法重新训练后的性能与计算开销，不宣称零样本泛化。

主论文后续实验包括：Always-Warm、On-Demand、Reliability-Greedy、Gaussian-PPO、扩散 BC 不微调和 DPPO。消融包括可靠性约束、可学习离散保留时间、中心云和慢周期。

“无安全解码”消融固定命名为 `Strict-Rejection`：对原始连续评分只按配置整数档位直接量化，不做全局可完成性屏蔽；量化结果只要违反健康、内存、故障域或可靠性约束，就拒绝该慢层动作并保持现有存活实例，不投影、不修复、不套用模板。这样比较的是安全解码的价值，而不是重新引入已删除的启发式投影器。

正式报告至少包含多训练种子、公共测试轨迹、均值、95% 置信区间、配对显著性检验，以及成本分解、SLA 违约率、理论可靠性、实际成功率、冷启动、内存、云使用率、服务缺口和求解器 P95 耗时。

基线和完整论文对比在主算法闭环稳定后实施，不阻塞 A→E 主线。

## 13. 实施完成定义

完整重构完成必须同时满足：

1. 配置、代码、测试和文档中不再存在车载计算候选；
2. 故障、生命周期、队列和状态更新的所有权唯一；
3. 快层只使用健康温实例，连续模型通过 DCP；
4. 跨时隙 EDF 能给出批次真实完成和 SLA；
5. DPPO 动作为 \(n_{k,i,T}\) 与 \(L_{k,i,T}\)，无独立副本数；
6. 安全解码没有通用投影器或启发式回退；
7. PPO 使用完整联合原始动作概率；
8. 教师、校准、验证和最终测试严格隔离；
9. 内部失败不会进入 PPO 事务缓冲区或成为策略奖励；
10. 每阶段核心测试和仍适用回归测试通过，并形成独立提交；
11. 最小闭环稳定后才扩大训练规模。
