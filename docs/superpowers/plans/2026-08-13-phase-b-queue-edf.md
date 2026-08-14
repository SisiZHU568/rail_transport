# Phase B Queue, Pipeline, EDF, and SLA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立跨快时隙的上行/阶段队列、不可变在途与完成事件、确定性 EDF 提交和批次级 SLA 审计，同时保持旧快层独立运行直至阶段 C 替换。

**Architecture:** `QueueStateManager` 是阶段 B 唯一状态所有者，对外只发布不可变 `QueueSnapshot`。手工 `FastAllocationPlan` 仅描述聚合服务量；管理器在临时副本中按 EDF 将服务量映射到片段，验证版本、流量守恒和阶段因果性后一次性提交。最终 VNF 产生下一时隙完成事件，跨节点转发产生不可变在途记录。

**Tech Stack:** Python 3.12、dataclasses、pytest；不新增第三方依赖。

---

## 文件结构

- Create: `src/queue_state.py` — 批次、片段、在途、完成事件与不可变快照。
- Create: `src/queue_manager.py` — 边界事件、EDF、原子服务提交和 SLA 状态所有者。
- Create: `src/drain_audit.py` — 排空终止、截断批次与成功率汇总。
- Create: `tests/test_queue_state.py` — 数据模型、物理/等效量和只读快照。
- Create: `tests/test_queue_manager.py` — 上行、VNF、有线、EDF、故障与原子拒绝。
- Create: `tests/test_sla_drain.py` — 截止边界、迟到完成、单次违约和排空统计。
- Modify: `src/orchestration_core.py` — 增加阶段 B 边界调用器，不改变旧调度器。
- Create: `tests/test_phase_b_orchestration.py` — 到达/完成先提交、故障后解绑的事件顺序。
- Modify: `README.md` — 说明阶段 B 完成边界和核心检查命令。

### Task 1: 定义不可变队列数据模型

**Files:** Create `src/queue_state.py`; create `tests/test_queue_state.py`.

- [ ] **Step 1:** 写失败测试，覆盖非法等效量、片段字段、在途双字段守恒、累计输出比例、物理量往返和快照不可修改。
- [ ] **Step 2:** 运行 `python -m pytest tests/test_queue_state.py -q`，确认因模块缺失而 RED。
- [ ] **Step 3:** 最小实现 `BatchRecord`、`QueueFragment`、`InTransitRecord`、`CompletionEvent`、`QueueSnapshot`、`StageFlowConfig`；片段只保存等效量。
- [ ] **Step 4:** 运行 GREEN，并提交 `feat: add immutable pipeline queue state`。

### Task 2: 实现边界事件与原子 EDF 服务提交

**Files:** Create `src/queue_manager.py`; create `tests/test_queue_manager.py`.

- [ ] **Step 1:** 写失败测试，覆盖上行、VNF、有线和最终完成的下一时隙因果性，以及 EDF、部分服务、多出口、目标绑定、故障节点和原子拒绝。
- [ ] **Step 2:** 运行 `python -m pytest tests/test_queue_manager.py -q`，确认 RED。
- [ ] **Step 3:** 实现 `QueueKey`、`AllocationOperation`、`FastAllocationPlan`、`QueueCommitContext`、`QueueCommitResult` 和 `QueueStateManager`。`begin_slot()` 只提交到期事件；`commit_allocation()` 在副本中按 EDF 扣减；ID 使用规范字段 SHA-256 派生。
- [ ] **Step 4:** 运行队列测试 GREEN，并提交 `feat: add atomic EDF queue transitions`。

### Task 3: 实现 SLA 与排空审计

**Files:** Create `src/drain_audit.py`; create `tests/test_sla_drain.py`; modify `src/queue_manager.py`.

- [ ] **Step 1:** 写失败测试，覆盖截止边界、单次违约、迟到完成、排空终止、截断批次和成功率分母。
- [ ] **Step 2:** 运行 `python -m pytest tests/test_sla_drain.py -q`，确认 RED。
- [ ] **Step 3:** 实现 `audit_deadlines()`、`SLAReport`、`CensoredBatch`、`DrainPolicy` 和 `build_drain_report()`；最大排空上限补齐到慢帧边界。
- [ ] **Step 4:** 运行 SLA 与队列测试 GREEN，并提交 `feat: add batch SLA and drain audit`。

### Task 4: 接入阶段 B 事件顺序并完成回归

**Files:** Modify `src/orchestration_core.py`; create `tests/test_phase_b_orchestration.py`; modify `README.md`.

- [ ] **Step 1:** 写失败测试，覆盖时隙边界先提交事件、再更新生命周期/故障、随后解绑已到达片段，且在途记录不变。
- [ ] **Step 2:** 运行 `python -m pytest tests/test_phase_b_orchestration.py -q`，确认 RED。
- [ ] **Step 3:** 新增 `PhaseBSlotCoordinator`，只负责调用顺序，不复制模块状态；阶段 C 前不改旧快层。
- [ ] **Step 4:** 运行阶段 A/B 核心回归、`py_compile` 和 `git diff --check`，提交 `feat: integrate phase B queue orchestration`。

完成后推送 `codex/dppo-main-algorithm`，不合并主分支，不创建兼容开关或 `_old` 文件。
